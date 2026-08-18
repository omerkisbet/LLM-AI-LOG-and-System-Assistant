from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
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

COLLECTION_NAME = os.getenv(
    "QDRANT_HYBRID_COLLECTION",
    "iotlab_operational_logs_v2",
).strip()

DENSE_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
).strip()

DENSE_MODEL_PATH = required_env(
    "EMBEDDING_MODEL_PATH"
)

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

VECTOR_SIZE = 384


def register_dense_model() -> None:
    supported_models = TextEmbedding.list_supported_models()
    names: set[str] = set()

    for description in supported_models:
        if isinstance(description, dict):
            name = description.get("model")
        else:
            name = getattr(description, "model", None)

        if name:
            names.add(str(name))

    if DENSE_MODEL in names:
        return

    TextEmbedding.add_custom_model(
        model=DENSE_MODEL,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=DENSE_MODEL),
        dim=VECTOR_SIZE,
        model_file="onnx/model.onnx",
    )


def parse_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)

    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            "Tarih saat dilimi içermelidir."
        )

    return parsed


def build_filter(
    source_type: str | None,
    action: str | None,
    level: str | None,
    entity_type: str | None,
    from_time: datetime | None,
    to_time: datetime | None,
) -> models.Filter | None:
    conditions: list[models.Condition] = []

    keyword_values = {
        "sourceType": source_type,
        "action": action,
        "level": level,
        "entityType": entity_type,
    }

    for field_name, value in keyword_values.items():
        if not value:
            continue

        conditions.append(
            models.FieldCondition(
                key=field_name,
                match=models.MatchValue(
                    value=value.upper(),
                ),
            )
        )

    if from_time or to_time:
        conditions.append(
            models.FieldCondition(
                key="timestamp",
                range=models.DatetimeRange(
                    gte=from_time,
                    lte=to_time,
                ),
            )
        )

    if not conditions:
        return None

    return models.Filter(must=conditions)


def shorten(value: Any, maximum: int = 260) -> str:
    text = " ".join(str(value or "").split())

    if len(text) <= maximum:
        return text

    return text[:maximum] + "..."


def print_results(
    title: str,
    points: list[Any],
) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)

    if not points:
        print("Sonuç bulunamadı.")
        return

    for index, point in enumerate(points, start=1):
        payload = point.payload or {}

        print(
            f"{index:02d}. skor={point.score:.6f} | "
            f"{payload.get('timestamp', '-')} | "
            f"{payload.get('sourceType', '-')}"
        )

        details: list[str] = []

        for field in (
            "level",
            "action",
            "entityType",
            "entityId",
            "requestId",
            "path",
        ):
            value = payload.get(field)

            if value is not None:
                details.append(f"{field}={value}")

        if details:
            print("    " + " | ".join(details))

        print(
            "    "
            + shorten(
                payload.get("content")
                or payload.get("description")
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dense, sparse ve hybrid retrieval "
            "sonuçlarını karşılaştırır."
        )
    )

    parser.add_argument("query")
    parser.add_argument("--source-type")
    parser.add_argument("--action")
    parser.add_argument("--level")
    parser.add_argument("--entity-type")

    parser.add_argument(
        "--from",
        dest="from_time",
        type=parse_datetime,
    )

    parser.add_argument(
        "--to",
        dest="to_time",
        type=parse_datetime,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--prefetch-limit",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    limit = max(1, min(args.limit, 50))
    prefetch_limit = max(
        limit,
        min(args.prefetch_limit, 200),
    )

    register_dense_model()

    dense_embedder = TextEmbedding(
        model_name=DENSE_MODEL,
        specific_model_path=DENSE_MODEL_PATH,
        local_files_only=True,
        threads=4,
    )

    sparse_embedder = SparseTextEmbedding(
        model_name=SPARSE_MODEL,
        language=SPARSE_LANGUAGE,
        specific_model_path=SPARSE_MODEL_PATH,
        local_files_only=True,
    )

    dense_vector = next(
        dense_embedder.embed([
            f"query: {args.query}"
        ])
    )

    sparse_embedding = next(
        sparse_embedder.query_embed(
            args.query
        )
    )

    sparse_vector = models.SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist(),
    )

    query_filter = build_filter(
        source_type=args.source_type,
        action=args.action,
        level=args.level,
        entity_type=args.entity_type,
        from_time=args.from_time,
        to_time=args.to_time,
    )

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=120,
    )

    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Collection bulunamadı: {COLLECTION_NAME}"
        )

    dense_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=dense_vector.tolist(),
        using="dense",
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    sparse_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=sparse_vector,
        using="sparse",
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    hybrid_result = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vector.tolist(),
                using="dense",
                filter=query_filter,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=sparse_vector,
                using="sparse",
                filter=query_filter,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    print()
    print(f"Soru       : {args.query}")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Limit      : {limit}")
    print(f"Prefetch   : {prefetch_limit}")

    print_results(
        "DENSE — Anlamsal arama",
        dense_result.points,
    )

    print_results(
        "SPARSE/BM25 — Kelime eşleşmesi",
        sparse_result.points,
    )

    print_results(
        "HYBRID/RRF — Dense + Sparse",
        hybrid_result.points,
    )


if __name__ == "__main__":
    main()
