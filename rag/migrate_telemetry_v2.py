
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from training_pipeline.telemetry_v2 import TrainingTelemetryCapture


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate IoT Lab telemetry database to v2."
    )
    parser.add_argument("--database", default=None)
    args = parser.parse_args()

    rag_dir = Path.home() / "huggingface-model-server" / "rag"
    load_dotenv(rag_dir / ".env")

    database = Path(
        args.database
        or os.getenv(
            "LOG_AGENT_TELEMETRY_DB",
            str(rag_dir / "data" / "log_agent_telemetry.sqlite3"),
        )
    ).expanduser()

    capture = TrainingTelemetryCapture(database)
    added = capture.migrate()

    print("Telemetry v2 migration tamamlandı.")
    print("Veritabanı:", database)
    print("Yeni kolonlar:", added or "YOK (zaten güncel)")
    print("Health:", capture.health())


if __name__ == "__main__":
    main()
