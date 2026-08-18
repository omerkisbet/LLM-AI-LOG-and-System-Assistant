from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

PROJECT_DIR = Path.home() / "huggingface-model-server"
RAG_DIR = PROJECT_DIR / "rag"

sys.path.insert(0, str(RAG_DIR))

from incident_detection.detector import IncidentDetector
from incident_detection.store import IncidentStore


load_dotenv(RAG_DIR / ".env")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Zorunlu ortam değişkeni eksik: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IoT Lab repeated-exception incident detector"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=int(os.getenv("INCIDENT_WINDOW_MINUTES", "5")),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=int(
            os.getenv("INCIDENT_REPEATED_EXCEPTION_THRESHOLD", "5")
        ),
    )
    args = parser.parse_args()

    qdrant = QdrantClient(
        url=required_env("QDRANT_URL"),
        api_key=required_env("QDRANT_API_KEY"),
        timeout=120,
    )

    database_path = os.getenv(
        "INCIDENT_DB_PATH",
        os.getenv(
            "LOG_AGENT_TELEMETRY_DB",
            str(RAG_DIR / "data" / "log_agent_telemetry.sqlite3"),
        ),
    ).strip()

    store = IncidentStore(database_path)
    detector = IncidentDetector(
        qdrant=qdrant,
        collection_name=os.getenv(
            "QDRANT_COLLECTION",
            "iotlab_operational_logs_v2",
        ).strip(),
        store=store,
        window_minutes=args.window_minutes,
        repeated_exception_threshold=args.threshold,
        max_events=int(os.getenv("INCIDENT_MAX_EVENTS", "5000")),
    )

    result = detector.run_once(dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
