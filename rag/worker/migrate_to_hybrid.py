from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models


PROJECT_DIR = Path.home() / "huggingface-model-server"
RAG_DIR = PROJECT_DIR / "rag"

load_dotenv(RAG_DIR / ".env")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Zorunlu ortam değişkeni eksik: {name}"
        )

    return value


QDRANT_URL = required_env("QDRANT_URL")
QDRANT_API_KEY = required_env("QDRANT_API_KEY")

SOURCE_COLLECTION = os.getenv(
    "QDRANT_SOURCE_COLLECTION",
    "iotlab_operational_logs",
).strip()

TARGET_COLLECTION = os.getenv(
    "QDRANT_HYBRID_COLLECTION",
    "iotlab_operational_logs_v2",
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

BATCH_SIZE = 100
DENSE_VECTOR_SIZE = 384


def extract_dense_vector(
    vector: Any,
) -> list[float]:
    if isinstance(vector, dict):
        dense = vector.get("dense")

        if dense is None:
            raise RuntimeError(
                "Kaynak point içinde dense vektör bulunamadı."
            )

        return list(dense)

    if isinstance(vector, list):
        return vector

    raise RuntimeError(
        f"Desteklenmeyen dense vector tipi: {type(vector)}"
    )


def create_target_collection(
    client: QdrantClient,
) -> None:
    if client.collection_exists(TARGET_COLLECTION):
        raise RuntimeError(
            f"Hedef collection zaten mevcut: "
            f"{TARGET_COLLECTION}\n"
            "Mevcut collection yanlışlıkla silinmedi."
        )

    client.create_collection(
        collection_name=TARGET_COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )

    print(
        f"Hybrid collection oluşturuldu: "
        f"{TARGET_COLLECTION}"
    )


def create_payload_indexes(
    client: QdrantClient,
) -> None:
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
        client.create_payload_index(
            collection_name=TARGET_COLLECTION,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )

        print(f"Payload index: {field_name}")


def migrate(
    client: QdrantClient,
    sparse_model: SparseTextEmbedding,
) -> int:
    total = 0
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=SOURCE_COLLECTION,
            offset=offset,
            limit=BATCH_SIZE,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            break

        contents: list[str] = []

        for point in points:
            payload = point.payload or {}
            content = str(
                payload.get("content")
                or payload.get("description")
                or ""
            ).strip()

            if not content:
                content = (
                    f"{payload.get('sourceType', '')} "
                    f"{payload.get('action', '')} "
                    f"{payload.get('level', '')} "
                    f"{payload.get('entityType', '')} "
                    f"{payload.get('path', '')}"
                ).strip()

            contents.append(content)

        sparse_vectors = list(
            sparse_model.embed(
                contents,
                batch_size=BATCH_SIZE,
            )
        )

        qdrant_points: list[models.PointStruct] = []

        for point, sparse_vector in zip(
            points,
            sparse_vectors,
            strict=True,
        ):
            dense_vector = extract_dense_vector(
                point.vector
            )

            qdrant_points.append(
                models.PointStruct(
                    id=point.id,
                    vector={
                        "dense": dense_vector,
                        "sparse": models.SparseVector(
                            indices=(
                                sparse_vector.indices.tolist()
                            ),
                            values=(
                                sparse_vector.values.tolist()
                            ),
                        ),
                    },
                    payload=point.payload or {},
                )
            )

        client.upsert(
            collection_name=TARGET_COLLECTION,
            points=qdrant_points,
            wait=True,
        )

        total += len(qdrant_points)

        print(
            f"Aktarılan toplam kayıt: {total}"
        )

        if next_offset is None:
            break

        offset = next_offset

    return total


def main() -> None:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=120,
    )

    if not client.collection_exists(
        SOURCE_COLLECTION
    ):
        raise RuntimeError(
            f"Kaynak collection bulunamadı: "
            f"{SOURCE_COLLECTION}"
        )

    source_info = client.get_collection(
        SOURCE_COLLECTION
    )

    print(
        f"Kaynak collection: {SOURCE_COLLECTION}"
    )
    print(
        f"Kaynak point sayısı: "
        f"{source_info.points_count}"
    )

    sparse_model = SparseTextEmbedding(
        model_name=SPARSE_MODEL,
        language=SPARSE_LANGUAGE,
        specific_model_path=SPARSE_MODEL_PATH,
        local_files_only=True,
    )

    print(
        f"Sparse model hazır: "
        f"{SPARSE_MODEL} / {SPARSE_LANGUAGE}"
    )

    create_target_collection(client)
    create_payload_indexes(client)

    migrated = migrate(
        client=client,
        sparse_model=sparse_model,
    )

    target_count = client.count(
        collection_name=TARGET_COLLECTION,
        exact=True,
    ).count

    print()
    print("=" * 72)
    print("HYBRID MIGRATION TAMAMLANDI")
    print("=" * 72)
    print(f"Kaynak collection : {SOURCE_COLLECTION}")
    print(f"Hedef collection  : {TARGET_COLLECTION}")
    print(f"Aktarılan         : {migrated}")
    print(f"Hedef exact count : {target_count}")

    if migrated != target_count:
        raise RuntimeError(
            "Aktarılan kayıt sayısı ile hedef count uyuşmuyor."
        )


if __name__ == "__main__":
    main()
