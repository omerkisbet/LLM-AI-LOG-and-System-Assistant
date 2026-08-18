
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any


MIGRATION_COLUMNS: dict[str, str] = {
    "evidence_json": "TEXT",
    "query_plan_json": "TEXT",
    "operational_summary_json": "TEXT",
    "prompt_version": "TEXT",
    "model_name": "TEXT",
}


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class TrainingTelemetryCapture:
    """Fail-safe extension for grounded fine-tuning telemetry."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self.migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def migrate(self) -> list[str]:
        added: list[str] = []

        with self._lock:
            with self._connect() as connection:
                table = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name = 'ai_query_logs'
                    """
                ).fetchone()

                if table is None:
                    raise RuntimeError(
                        "ai_query_logs tablosu bulunamadı."
                    )

                existing = {
                    str(row["name"])
                    for row in connection.execute(
                        'PRAGMA table_info("ai_query_logs")'
                    ).fetchall()
                }

                for column, sql_type in MIGRATION_COLUMNS.items():
                    if column in existing:
                        continue

                    connection.execute(
                        f'ALTER TABLE "ai_query_logs" '
                        f'ADD COLUMN "{column}" {sql_type}'
                    )
                    added.append(column)

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_ai_query_logs_prompt_version
                    ON ai_query_logs(prompt_version)
                    """
                )

                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_ai_query_logs_model_name
                    ON ai_query_logs(model_name)
                    """
                )

        return added

    def record_context(
        self,
        *,
        query_id: str,
        evidence: list[dict[str, Any]],
        query_plan: dict[str, Any] | None,
        operational_summary: dict[str, Any],
        prompt_version: str,
        model_name: str,
    ) -> None:
        normalized_query_id = query_id.strip()

        if not normalized_query_id:
            raise ValueError("query_id boş olamaz.")

        with self._lock:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE ai_query_logs
                    SET
                        evidence_json = ?,
                        query_plan_json = ?,
                        operational_summary_json = ?,
                        prompt_version = ?,
                        model_name = ?
                    WHERE id = ?
                    """,
                    (
                        _json_text(evidence),
                        (
                            _json_text(query_plan)
                            if query_plan is not None
                            else None
                        ),
                        _json_text(operational_summary),
                        prompt_version.strip()[:120] or None,
                        model_name.strip()[:200] or None,
                        normalized_query_id,
                    ),
                )

                if cursor.rowcount != 1:
                    raise LookupError(
                        "Telemetry sorgu kaydı bulunamadı: "
                        f"{normalized_query_id}"
                    )

    def health(self) -> dict[str, Any]:
        with self._connect() as connection:
            columns = {
                str(row["name"])
                for row in connection.execute(
                    'PRAGMA table_info("ai_query_logs")'
                ).fetchall()
            }

            captured = connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM ai_query_logs
                WHERE evidence_json IS NOT NULL
                  AND query_plan_json IS NOT NULL
                """
            ).fetchone()["total"]

        missing = sorted(
            set(MIGRATION_COLUMNS).difference(columns)
        )

        return {
            "status": "UP" if not missing else "DOWN",
            "databasePath": str(self.database_path),
            "capturedQueryCount": int(captured),
            "missingColumns": missing,
        }
