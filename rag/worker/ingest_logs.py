from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding,TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from qdrant_client import QdrantClient, models


PROJECT_DIR = Path.home() / "huggingface-model-server"
RAG_DIR = PROJECT_DIR / "rag"
STATE_FILE = RAG_DIR / "worker" / "sync-state.json"

load_dotenv(RAG_DIR / ".env")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(f"Zorunlu ortam değişkeni eksik: {name}")

    return value


BAHTIYAR_BASE_URL = required_env("BAHTIYAR_BASE_URL").rstrip("/")
AI_LOG_TOOLS_API_KEY = required_env("AI_LOG_TOOLS_API_KEY")

QDRANT_URL = required_env("QDRANT_URL")
QDRANT_API_KEY = required_env("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "iotlab_operational_logs",
).strip()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
).strip()

EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    "",
).strip()
SPARSE_MODEL = os.getenv(
    "SPARSE_MODEL",
    "Qdrant/bm25",
).strip()

SPARSE_MODEL_PATH = required_env(
    "SPARSE_MODEL_PATH"
)

SPARSE_LANGUAGE = os.getenv(
    "SPARSE_LANGUAGE",
    "turkish",
).strip()
RUNTIME_INITIAL_DAYS = int(os.getenv("RUNTIME_INITIAL_DAYS", "30"))
AUDIT_INITIAL_DAYS = int(os.getenv("AUDIT_INITIAL_DAYS", "90"))
SYNC_BATCH_SIZE = min(
    max(int(os.getenv("SYNC_BATCH_SIZE", "200")), 1),
    500,
)

VECTOR_SIZE = 384

SOURCE_TYPES = (
    "RUNTIME_LOG",
    "AUDIT_LOG",
)


def register_embedding_model() -> None:
    supported_models = TextEmbedding.list_supported_models()

    supported_names: set[str] = set()

    for model_description in supported_models:
        if isinstance(model_description, dict):
            model_name = model_description.get("model")
        else:
            model_name = getattr(model_description, "model", None)

        if model_name:
            supported_names.add(str(model_name))

    if EMBEDDING_MODEL in supported_names:
        return

    TextEmbedding.add_custom_model(
        model=EMBEDDING_MODEL,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(
            hf="intfloat/multilingual-e5-small",
        ),
        dim=VECTOR_SIZE,
        model_file="onnx/model.onnx",
    )


def load_state() -> dict[str, str]:
    if not STATE_FILE.exists():
        return {}

    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"Sync state okunamadı: {STATE_FILE}"
        ) from exc


def save_state(state: dict[str, str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = STATE_FILE.with_suffix(".tmp")

    temporary_file.write_text(
        json.dumps(
            state,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_file.replace(STATE_FILE)


def initial_after(source_type: str) -> str:
    days = (
        RUNTIME_INITIAL_DAYS
        if source_type == "RUNTIME_LOG"
        else AUDIT_INITIAL_DAYS
    )

    value = datetime.now(timezone.utc) - timedelta(days=days)

    return value.isoformat().replace("+00:00", "Z")


def point_id(document_id: str) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"iotlab-log:{document_id}",
        )
    )


def flatten_payload(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}

    payload: dict[str, Any] = {
        "documentId": item["id"],
        "sourceType": item["sourceType"],
        "mongoId": item["mongoId"],
        "timestamp": item["timestamp"],
        "content": item["content"],
        "contentHash": item["contentHash"],
    }

    expires_at = item.get("expiresAt")

    if expires_at:
        payload["expiresAt"] = expires_at

    for key, value in metadata.items():
        if value is not None:
            payload[key] = value

    return payload

def build_sparse_text(
    payload: dict[str, Any],
) -> str:
    fields = [
        payload.get("content"),
        payload.get("sourceType"),
        payload.get("level"),
        payload.get("category"),
        payload.get("action"),
        payload.get("entityType"),
        payload.get("entityId"),
        payload.get("actor"),
        payload.get("requestId"),
        payload.get("path"),
        payload.get("statusCode"),
    ]

    return " ".join(
        str(value).strip()
        for value in fields
        if value is not None and str(value).strip()
    )
def create_collection_if_missing(
    client: QdrantClient,
) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                )
            },
	    sparse_vectors_config={
		"sparse": models.SparseVectorParams(
		    modifier=models.Modifier.IDF,
		)
	    },	
        )

        print(f"Collection oluşturuldu: {COLLECTION_NAME}")
    else:
        print(f"Collection zaten var: {COLLECTION_NAME}")


def create_payload_indexes(client: QdrantClient) -> None:
    indexes = {
        "sourceType": models.PayloadSchemaType.KEYWORD,
        "timestamp": models.PayloadSchemaType.DATETIME,
        "expiresAt": models.PayloadSchemaType.DATETIME,
        "level": models.PayloadSchemaType.KEYWORD,
        "category": models.PayloadSchemaType.KEYWORD,
        "action": models.PayloadSchemaType.KEYWORD,
        "entityType": models.PayloadSchemaType.KEYWORD,
        "entityId": models.PayloadSchemaType.KEYWORD,
        "actor": models.PayloadSchemaType.KEYWORD,
        "requestId": models.PayloadSchemaType.KEYWORD,
        "path": models.PayloadSchemaType.KEYWORD,
        "statusCode": models.PayloadSchemaType.INTEGER,
    }

    for field_name, field_schema in indexes.items():
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
        except Exception as exc:
            # Index zaten varsa Qdrant hata döndürebilir.
            # Collection çalışmasını durdurmamak için devam ediyoruz.
            print(
                f"Payload index atlandı: {field_name} "
                f"({type(exc).__name__})"
            )


def fetch_batch(
    http_client: httpx.Client,
    source_type: str,
    cursor: str | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "sourceType": source_type,
        "limit": SYNC_BATCH_SIZE,
    }

    if cursor:
        params["cursor"] = cursor
    else:
        params["after"] = initial_after(source_type)

    response = http_client.get(
        f"{BAHTIYAR_BASE_URL}/api/internal/rag/log-events",
        params=params,
        headers={
            "X-AI-Service-Key": AI_LOG_TOOLS_API_KEY,
            "Accept": "application/json",
        },
    )

    response.raise_for_status()

    result = response.json()

    if result.get("sourceType") != source_type:
        raise RuntimeError(
            "Bahtiyar beklenmeyen sourceType döndürdü."
        )

    return result


def upsert_items(
    qdrant: QdrantClient,
    dense_embedder: TextEmbedding,
    sparse_embedder: SparseTextEmbedding,
    items: list[dict[str, Any]],
) -> int:
    if not items:
        return 0

    payloads = [
        flatten_payload(item)
        for item in items
    ]

    dense_documents = [
        f"passage: {payload['content']}"
        for payload in payloads
    ]

    sparse_documents = [
        build_sparse_text(payload)
        for payload in payloads
    ]

    dense_vectors = list(
        dense_embedder.embed(
            dense_documents,
            batch_size=32,
        )
    )

    sparse_vectors = list(
        sparse_embedder.embed(
            sparse_documents,
            batch_size=32,
        )
    )

    points: list[models.PointStruct] = []

    for item, payload, dense_vector, sparse_vector in zip(
        items,
        payloads,
        dense_vectors,
        sparse_vectors,
        strict=True,
    ):
        points.append(
            models.PointStruct(
                id=point_id(item["id"]),
                vector={
                    "dense": dense_vector.tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_vector.indices.tolist(),
                        values=sparse_vector.values.tolist(),
                    ),
                },
                payload=payload,
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    return len(points)

 
    


def sync_source(
    source_type: str,
    state: dict[str, str],
    http_client: httpx.Client,
    qdrant: QdrantClient,
    dense_embedder: TextEmbedding,
    sparse_embedder: SparseTextEmbedding
) -> int:
    total = 0

    while True:
        cursor = state.get(source_type)

        batch = fetch_batch(
            http_client=http_client,
            source_type=source_type,
            cursor=cursor,
        )

        items = batch.get("items") or []

        indexed = upsert_items(
            qdrant=qdrant,
            dense_embedder=dense_embedder,
    	    sparse_embedder=sparse_embedder,
            items=items,
        )

        total += indexed

        next_cursor = batch.get("nextCursor")

        if next_cursor:
            state[source_type] = next_cursor
            save_state(state)

        print(
            f"{source_type}: "
            f"alınan={len(items)}, "
            f"indekslenen={indexed}, "
            f"hasMore={batch.get('hasMore')}"
        )

        if not batch.get("hasMore"):
            break

        if not next_cursor:
            raise RuntimeError(
                f"{source_type} hasMore=true fakat nextCursor yok."
            )

    return total


def main() -> None:
    register_embedding_model()

    print(f"Embedding modeli hazırlanıyor: {EMBEDDING_MODEL}")

    dense_embedder = TextEmbedding(
        model_name=EMBEDDING_MODEL,
        threads=4,
        specific_model_path=EMBEDDING_MODEL_PATH or None,
        local_files_only=bool(EMBEDDING_MODEL_PATH),
    )
    sparse_embedder = SparseTextEmbedding(
        model_name=SPARSE_MODEL,
        language=SPARSE_LANGUAGE,
        specific_model_path=SPARSE_MODEL_PATH,
        local_files_only=True,
    )

    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
    )

    create_collection_if_missing(qdrant)
    create_payload_indexes(qdrant)

    state = load_state()

    with httpx.Client(
        timeout=httpx.Timeout(60.0),
        follow_redirects=True,
    ) as http_client:
        grand_total = 0

        for source_type in SOURCE_TYPES:
            grand_total += sync_source(
                source_type=source_type,
                state=state,
                http_client=http_client,
                qdrant=qdrant,
                dense_embedder=dense_embedder,
		sparse_embedder =sparse_embedder,
            )

    print(f"Sync tamamlandı. Toplam indekslenen: {grand_total}")


if __name__ == "__main__":
    main()
