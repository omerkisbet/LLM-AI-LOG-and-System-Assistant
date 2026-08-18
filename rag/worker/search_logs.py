from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from qdrant_client import QdrantClient, models


PROJECT_DIR = Path.home() / "huggingface-model-server"
RAG_DIR = PROJECT_DIR / "rag"

load_dotenv(RAG_DIR / ".env")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(f"Eksik ortam değişkeni: {name}")

    return value


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

VECTOR_SIZE = 384


def register_embedding_model() -> None:
    supported_models = TextEmbedding.list_supported_models()

    model_names: set[str] = set()

    for description in supported_models:
        if isinstance(description, dict):
            model_name = description.get("model")
        else:
            model_name = getattr(description, "model", None)

        if model_name:
            model_names.add(str(model_name))

    if EMBEDDING_MODEL in model_names:
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


def parse_datetime(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")

    parsed = datetime.fromisoformat(normalized)

    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            "Tarih saat dilimi içermelidir. Örnek: "
            "2026-07-23T00:00:00Z"
        )

    return parsed


def build_filter(
    source_type: str | None,
    action: str | None,
    level: str | None,
    entity_type: str | None,
    path: str | None,
    from_time: datetime | None,
    to_time: datetime | None,
) -> models.Filter | None:
    conditions: list[models.Condition] = []

    if source_type:
        conditions.append(
            models.FieldCondition(
                key="sourceType",
                match=models.MatchValue(
                    value=source_type.upper(),
                ),
            )
        )

    if action:
        conditions.append(
            models.FieldCondition(
                key="action",
                match=models.MatchValue(
                    value=action.upper(),
                ),
            )
        )

    if level:
        conditions.append(
            models.FieldCondition(
                key="level",
                match=models.MatchValue(
                    value=level.upper(),
                ),
            )
        )

    if entity_type:
        conditions.append(
            models.FieldCondition(
                key="entityType",
                match=models.MatchValue(
                    value=entity_type.upper(),
                ),
            )
        )

    if path:
        conditions.append(
            models.FieldCondition(
                key="path",
                match=models.MatchValue(
                    value=path,
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


def shorten(value: Any, maximum: int = 700) -> str:
    text = str(value or "").strip()

    if len(text) <= maximum:
        return text

    return text[:maximum] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IoT Lab operasyon loglarında semantic arama"
    )

    parser.add_argument(
        "query",
        help='Türkçe soru. Örnek: "Bugün hangi haberler silindi?"',
    )

    parser.add_argument(
        "--source-type",
        choices=["RUNTIME_LOG", "AUDIT_LOG"],
    )

    parser.add_argument("--action")
    parser.add_argument("--level")
    parser.add_argument("--entity-type")
    parser.add_argument("--path")

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

    args = parser.parse_args()

    register_embedding_model()

    embedder = TextEmbedding(
        model_name=EMBEDDING_MODEL,
        threads=4,
        specific_model_path=EMBEDDING_MODEL_PATH or None,
        local_files_only=bool(EMBEDDING_MODEL_PATH),
    )

    query_vector = next(
        embedder.embed(
            [f"query: {args.query}"]
        )
    )

    query_filter = build_filter(
        source_type=args.source_type,
        action=args.action,
        level=args.level,
        entity_type=args.entity_type,
        path=args.path,
        from_time=args.from_time,
        to_time=args.to_time,
    )

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
    )

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        using="dense",
        query_filter=query_filter,
        limit=max(1, min(args.limit, 50)),
        with_payload=True,
        with_vectors=False,
    )

    print()
    print(f"Soru: {args.query}")
    print(f"Bulunan sonuç: {len(result.points)}")
    print("=" * 80)

    for index, point in enumerate(result.points, start=1):
        payload = point.payload or {}

        print(
            f"{index}. skor={point.score:.4f} | "
            f"{payload.get('timestamp', '-')} | "
            f"{payload.get('sourceType', '-')}"
        )

        if payload.get("level"):
            print(f"   seviye: {payload['level']}")

        if payload.get("action"):
            print(f"   işlem: {payload['action']}")

        if payload.get("entityType"):
            print(
                f"   entity: {payload['entityType']} / "
                f"{payload.get('entityId', '-')}"
            )

        if payload.get("path"):
            print(f"   path: {payload['path']}")

        if payload.get("requestId"):
            print(f"   requestId: {payload['requestId']}")

        print(f"   içerik: {shorten(payload.get('content'))}")
        print(f"   mongoId: {payload.get('mongoId', '-')}")
        print("-" * 80)


if __name__ == "__main__":
    main()
