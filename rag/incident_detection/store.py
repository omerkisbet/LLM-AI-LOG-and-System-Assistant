from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .models import (
    IncidentCandidate,
    IncidentStatus,
    SEVERITY_ORDER,
    iso_utc,
)


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class IncidentStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    rule_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    exception_name TEXT,
                    endpoint TEXT,
                    first_seen_at_utc TEXT NOT NULL,
                    last_seen_at_utc TEXT NOT NULL,
                    window_start_at_utc TEXT NOT NULL,
                    window_end_at_utc TEXT NOT NULL,
                    occurrence_count INTEGER NOT NULL DEFAULT 0,
                    deterministic_summary TEXT,
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    acknowledged_at_utc TEXT,
                    acknowledged_by TEXT,
                    resolved_at_utc TEXT,
                    resolved_by TEXT,
                    resolution_note TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_incidents_status
                ON incidents(status);

                CREATE INDEX IF NOT EXISTS idx_incidents_last_seen
                ON incidents(last_seen_at_utc);

                CREATE TABLE IF NOT EXISTS incident_events (
                    incident_id TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    level TEXT,
                    request_id TEXT,
                    endpoint TEXT,
                    exception_name TEXT,
                    content TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    PRIMARY KEY (incident_id, event_key),
                    FOREIGN KEY(incident_id)
                        REFERENCES incidents(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS incident_detector_runs (
                    id TEXT PRIMARY KEY,
                    started_at_utc TEXT NOT NULL,
                    finished_at_utc TEXT NOT NULL,
                    window_start_at_utc TEXT NOT NULL,
                    window_end_at_utc TEXT NOT NULL,
                    fetched_event_count INTEGER NOT NULL,
                    candidate_count INTEGER NOT NULL,
                    new_incident_count INTEGER NOT NULL,
                    updated_incident_count INTEGER NOT NULL,
                    new_event_count INTEGER NOT NULL,
                    error_type TEXT,
                    error_message TEXT
                );
                """
            )

    @staticmethod
    def _stronger_severity(current: str, incoming: str) -> str:
        return (
            incoming
            if SEVERITY_ORDER.get(incoming, 0)
            > SEVERITY_ORDER.get(current, 0)
            else current
        )

    def upsert_candidate(
        self,
        candidate: IncidentCandidate,
    ) -> dict[str, Any]:
        now_text = utc_now_text()

        with self._lock:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")

                row = connection.execute(
                    "SELECT * FROM incidents WHERE dedupe_key = ?",
                    (candidate.dedupe_key,),
                ).fetchone()

                created = row is None

                if created:
                    incident_id = str(uuid.uuid4())
                    connection.execute(
                        """
                        INSERT INTO incidents (
                            id, dedupe_key, rule_type, title, severity,
                            status, source_type, exception_name, endpoint,
                            first_seen_at_utc, last_seen_at_utc,
                            window_start_at_utc, window_end_at_utc,
                            occurrence_count, deterministic_summary,
                            evidence_json, created_at_utc, updated_at_utc
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, '[]', ?, ?)
                        """,
                        (
                            incident_id,
                            candidate.dedupe_key,
                            candidate.rule_type.value,
                            candidate.title,
                            candidate.severity.value,
                            IncidentStatus.OPEN.value,
                            candidate.source_type,
                            candidate.exception_name,
                            candidate.endpoint,
                            iso_utc(candidate.first_seen_at),
                            iso_utc(candidate.last_seen_at),
                            iso_utc(candidate.window_start_at),
                            iso_utc(candidate.window_end_at),
                            candidate.deterministic_summary,
                            now_text,
                            now_text,
                        ),
                    )
                    current_status = IncidentStatus.OPEN.value
                    current_severity = candidate.severity.value
                    current_first_seen = iso_utc(candidate.first_seen_at)
                else:
                    incident_id = str(row["id"])
                    current_status = str(row["status"])
                    current_severity = str(row["severity"])
                    current_first_seen = str(row["first_seen_at_utc"])

                inserted_events = 0

                for event in candidate.events:
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO incident_events (
                            incident_id, event_key, record_id, timestamp_utc,
                            source_type, level, request_id, endpoint,
                            exception_name, content, created_at_utc
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            incident_id,
                            event.event_key,
                            event.record_id,
                            iso_utc(event.timestamp),
                            event.source_type,
                            event.level,
                            event.request_id,
                            event.endpoint,
                            event.exception_name,
                            event.content[:12000],
                            now_text,
                        ),
                    )
                    inserted_events += cursor.rowcount

                if inserted_events > 0:
                    reopened = current_status in {
                        IncidentStatus.RESOLVED.value,
                        IncidentStatus.IGNORED.value,
                    }
                    new_status = (
                        IncidentStatus.OPEN.value if reopened else current_status
                    )
                    severity = self._stronger_severity(
                        current_severity,
                        candidate.severity.value,
                    )
                    first_seen = min(
                        current_first_seen,
                        iso_utc(candidate.first_seen_at),
                    )

                    evidence_rows = connection.execute(
                        """
                        SELECT event_key, record_id, timestamp_utc,
                               source_type, level, request_id, endpoint,
                               exception_name, content
                        FROM incident_events
                        WHERE incident_id = ?
                        ORDER BY timestamp_utc DESC
                        LIMIT 20
                        """,
                        (incident_id,),
                    ).fetchall()

                    connection.execute(
                        """
                        UPDATE incidents
                        SET title = ?, severity = ?, status = ?,
                            source_type = ?, exception_name = ?, endpoint = ?,
                            first_seen_at_utc = ?, last_seen_at_utc = ?,
                            window_start_at_utc = ?, window_end_at_utc = ?,
                            occurrence_count = occurrence_count + ?,
                            deterministic_summary = ?, evidence_json = ?,
                            updated_at_utc = ?,
                            resolved_at_utc = CASE WHEN ? THEN NULL ELSE resolved_at_utc END,
                            resolved_by = CASE WHEN ? THEN NULL ELSE resolved_by END,
                            resolution_note = CASE WHEN ? THEN NULL ELSE resolution_note END
                        WHERE id = ?
                        """,
                        (
                            candidate.title,
                            severity,
                            new_status,
                            candidate.source_type,
                            candidate.exception_name,
                            candidate.endpoint,
                            first_seen,
                            iso_utc(candidate.last_seen_at),
                            iso_utc(candidate.window_start_at),
                            iso_utc(candidate.window_end_at),
                            inserted_events,
                            candidate.deterministic_summary,
                            json.dumps(
                                [dict(item) for item in evidence_rows],
                                ensure_ascii=False,
                            ),
                            now_text,
                            1 if reopened else 0,
                            1 if reopened else 0,
                            1 if reopened else 0,
                            incident_id,
                        ),
                    )

                final_row = connection.execute(
                    "SELECT * FROM incidents WHERE id = ?",
                    (incident_id,),
                ).fetchone()
                connection.commit()

        result = dict(final_row)
        result["created"] = created
        result["newEventCount"] = inserted_events
        return result

    def list_incidents(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        safe_limit = min(max(int(limit), 1), 500)
        sql = "SELECT * FROM incidents"
        params: list[Any] = []

        if status:
            sql += " WHERE status = ?"
            params.append(status.upper())

        sql += " ORDER BY updated_at_utc DESC LIMIT ?"
        params.append(safe_limit)

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()

        return [dict(row) for row in rows]


    def update_status(
        self,
        incident_id: str,
        *,
        status: IncidentStatus,
        actor: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        now_text = utc_now_text()
        normalized_actor = actor.strip()[:200]

        if not normalized_actor:
            raise ValueError("Actor alanı boş olamaz.")

        normalized_note = (
            note.strip()[:4000]
            if note and note.strip()
            else None
        )

        with self._lock:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")

                row = connection.execute(
                    """
                    SELECT *
                    FROM incidents
                    WHERE id = ?
                    """,
                    (incident_id,),
                ).fetchone()

                if row is None:
                    raise LookupError(
                        f"Incident bulunamadı: {incident_id}"
                    )

                if status == IncidentStatus.ACKNOWLEDGED:
                    connection.execute(
                        """
                        UPDATE incidents
                        SET
                            status = ?,
                            acknowledged_at_utc = ?,
                            acknowledged_by = ?,
                            updated_at_utc = ?
                        WHERE id = ?
                        """,
                        (
                            status.value,
                            now_text,
                            normalized_actor,
                            now_text,
                            incident_id,
                        ),
                    )

                elif status in {
                    IncidentStatus.RESOLVED,
                    IncidentStatus.IGNORED,
                }:
                    connection.execute(
                        """
                        UPDATE incidents
                        SET
                            status = ?,
                            resolved_at_utc = ?,
                            resolved_by = ?,
                            resolution_note = ?,
                            updated_at_utc = ?
                        WHERE id = ?
                        """,
                        (
                            status.value,
                            now_text,
                            normalized_actor,
                            normalized_note,
                            now_text,
                            incident_id,
                        ),
                    )

                elif status == IncidentStatus.OPEN:
                    connection.execute(
                        """
                        UPDATE incidents
                        SET
                            status = ?,
                            resolved_at_utc = NULL,
                            resolved_by = NULL,
                            resolution_note = NULL,
                            updated_at_utc = ?
                        WHERE id = ?
                        """,
                        (
                            status.value,
                            now_text,
                            incident_id,
                        ),
                    )

                else:
                    raise ValueError(
                        f"Desteklenmeyen incident durumu: {status}"
                    )

                updated = connection.execute(
                    """
                    SELECT *
                    FROM incidents
                    WHERE id = ?
                    """,
                    (incident_id,),
                ).fetchone()

                connection.commit()

        return dict(updated)

    def health(self) -> dict[str, Any]:
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM incidents"
            ).fetchone()[0]

            status_rows = connection.execute(
                """
                SELECT status, COUNT(*) AS item_count
                FROM incidents
                GROUP BY status
                """
            ).fetchall()

            last_run = connection.execute(
                """
                SELECT
                    finished_at_utc,
                    fetched_event_count,
                    candidate_count,
                    new_incident_count,
                    updated_incident_count,
                    new_event_count,
                    error_type,
                    error_message
                FROM incident_detector_runs
                ORDER BY finished_at_utc DESC
                LIMIT 1
                """
            ).fetchone()

        return {
            "status": "UP",
            "databasePath": str(self.database_path),
            "incidentCount": int(total),
            "countsByStatus": {
                str(row["status"]): int(row["item_count"])
                for row in status_rows
            },
            "lastDetectorRun": (
                dict(last_run)
                if last_run is not None
                else None
            ),
        }

    def statistics(self) -> dict[str, Any]:
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM incidents"
            ).fetchone()[0]

            status_rows = connection.execute(
                """
                SELECT status, COUNT(*) AS item_count
                FROM incidents
                GROUP BY status
                ORDER BY item_count DESC
                """
            ).fetchall()

            severity_rows = connection.execute(
                """
                SELECT severity, COUNT(*) AS item_count
                FROM incidents
                GROUP BY severity
                ORDER BY item_count DESC
                """
            ).fetchall()

            rule_rows = connection.execute(
                """
                SELECT rule_type, COUNT(*) AS item_count
                FROM incidents
                GROUP BY rule_type
                ORDER BY item_count DESC
                """
            ).fetchall()

        return {
            "total": int(total),
            "byStatus": {
                str(row["status"]): int(row["item_count"])
                for row in status_rows
            },
            "bySeverity": {
                str(row["severity"]): int(row["item_count"])
                for row in severity_rows
            },
            "byRuleType": {
                str(row["rule_type"]): int(row["item_count"])
                for row in rule_rows
            },
        }

    def get_incident(
        self,
        incident_id: str,
        *,
        event_limit: int = 100,
    ) -> dict[str, Any]:
        safe_limit = min(max(int(event_limit), 1), 500)

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM incidents
                WHERE id = ?
                """,
                (incident_id,),
            ).fetchone()

            if row is None:
                raise LookupError(
                    f"Incident bulunamadı: {incident_id}"
                )

            event_rows = connection.execute(
                """
                SELECT
                    event_key,
                    record_id,
                    timestamp_utc,
                    source_type,
                    level,
                    request_id,
                    endpoint,
                    exception_name,
                    content
                FROM incident_events
                WHERE incident_id = ?
                ORDER BY timestamp_utc DESC
                LIMIT ?
                """,
                (incident_id, safe_limit),
            ).fetchall()

        result = dict(row)

        try:
            result["evidence"] = json.loads(
                result.pop("evidence_json", "[]")
            )
        except json.JSONDecodeError:
            result["evidence"] = []

        result["events"] = [
            dict(event)
            for event in event_rows
        ]
        result["returnedEventCount"] = len(event_rows)
        return result

    def record_run(
        self,
        *,
        started_at_utc: str,
        finished_at_utc: str,
        window_start_at_utc: str,
        window_end_at_utc: str,
        fetched_event_count: int,
        candidate_count: int,
        new_incident_count: int,
        updated_incident_count: int,
        new_event_count: int,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> str:
        run_id = str(uuid.uuid4())

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO incident_detector_runs (
                    id, started_at_utc, finished_at_utc,
                    window_start_at_utc, window_end_at_utc,
                    fetched_event_count, candidate_count,
                    new_incident_count, updated_incident_count,
                    new_event_count, error_type, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    started_at_utc,
                    finished_at_utc,
                    window_start_at_utc,
                    window_end_at_utc,
                    fetched_event_count,
                    candidate_count,
                    new_incident_count,
                    updated_incident_count,
                    new_event_count,
                    error_type,
                    error_message,
                ),
            )

        return run_id
