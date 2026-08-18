from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from operational_tools import (
    execute_operational_plan,
)
from query_planner import plan_query


PROJECT_DIR = (
    Path.home()
    / "huggingface-model-server"
)

load_dotenv(
    PROJECT_DIR / "rag" / ".env"
)


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Eksik ortam değişkeni: {name}"
        )

    return value


def run_question(
    client: QdrantClient,
    collection_name: str,
    question: str,
    language: str,
) -> None:
    plan = plan_query(
        question=question,
        timezone_name="Europe/Istanbul",
        requested_language=language,
    )

    result = execute_operational_plan(
        client=client,
        collection_name=collection_name,
        plan=plan,
    )

    print("=" * 80)
    print("SORU:", question)
    print("INTENT:", plan.intent.value)
    print("DİL:", plan.detected_language)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    client = QdrantClient(
        url=required_env("QDRANT_URL"),
        api_key=required_env(
            "QDRANT_API_KEY"
        ),
        timeout=120,
    )

    collection_name = os.getenv(
        "QDRANT_COLLECTION",
        "iotlab_operational_logs_v2",
    ).strip()

    run_question(
        client=client,
        collection_name=collection_name,
        question="Bugün neler oldu?",
        language="tr",
    )

    run_question(
        client=client,
        collection_name=collection_name,
        question="What happened today?",
        language="en",
    )

    run_question(
        client=client,
        collection_name=collection_name,
        question=(
            "Son 2 saatte kaç hata oluştu?"
        ),
        language="tr",
    )

    print()
    print(
        "OPERATIONAL TOOLS TESTİ TAMAMLANDI"
    )


if __name__ == "__main__":
    main()
