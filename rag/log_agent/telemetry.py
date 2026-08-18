from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterable


VALID_FEEDBACK_RATINGS = {
    "UP",
    "DOWN",
}

VALID_FEEDBACK_REASONS = {
    "CORRECT",
    "WRONG_INTENT",
    "MISSING_LOG",
    "IRRELEVANT_EVIDENCE",
    "WRONG_TIME_RANGE",
    "WRONG_LANGUAGE",
    "WRONG_ANSWER",
    "INCOMPLETE_ANSWER",
    "OTHER",
}


def utc_now_text() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def normalize_optional_text(
    value: Any,
    maximum_length: int | None = None,
) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if maximum_length is not None:
        text = text[:maximum_length]

    return text


def boolean_to_integer(
    value: bool | None,
) -> int | None:
    if value is None:
        return None

    return 1 if value else 0


def integer_to_boolean(
    value: Any,
) -> bool | None:
    if value is None:
        return None

    return bool(value)


class TelemetryStore:
    def __init__(
        self,
        database_path: str | Path,
    ) -> None:
        self.database_path = Path(
            database_path
        ).expanduser()

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._write_lock = Lock()
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        connection.execute(
            "PRAGMA journal_mode = WAL"
        )

        connection.execute(
            "PRAGMA synchronous = NORMAL"
        )

        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ai_query_logs (
                    id TEXT PRIMARY KEY,
                    request_id TEXT,
                    actor TEXT,
                    question TEXT NOT NULL,
                    requested_language TEXT,
                    detected_language TEXT,
                    response_language TEXT,
                    intent TEXT,
                    retrieval_mode TEXT,
                    time_scope TEXT,
                    from_time_utc TEXT,
                    to_time_utc TEXT,
                    grounded INTEGER,
                    evidence_count INTEGER NOT NULL DEFAULT 0,
                    filtered_total INTEGER,
                    tools_used_json TEXT NOT NULL DEFAULT '[]',
                    latency_ms INTEGER,
                    answer TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    created_at_utc TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS
                    idx_ai_query_logs_created_at
                ON ai_query_logs(created_at_utc);

                CREATE INDEX IF NOT EXISTS
                    idx_ai_query_logs_request_id
                ON ai_query_logs(request_id);

                CREATE INDEX IF NOT EXISTS
                    idx_ai_query_logs_intent
                ON ai_query_logs(intent);

                CREATE INDEX IF NOT EXISTS
                    idx_ai_query_logs_grounded
                ON ai_query_logs(grounded);

                CREATE TABLE IF NOT EXISTS ai_answer_feedback (
                    id TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    request_id TEXT,
                    rating TEXT NOT NULL,
                    reason TEXT,
                    comment TEXT,
                    corrected_answer TEXT,
                    actor TEXT,
                    created_at_utc TEXT NOT NULL,
                    FOREIGN KEY(query_id)
                        REFERENCES ai_query_logs(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS
                    idx_ai_answer_feedback_query_id
                ON ai_answer_feedback(query_id);

                CREATE INDEX IF NOT EXISTS
                    idx_ai_answer_feedback_rating
                ON ai_answer_feedback(rating);

                CREATE INDEX IF NOT EXISTS
                    idx_ai_answer_feedback_created_at
                ON ai_answer_feedback(created_at_utc);
                """
            )

    def health(self) -> dict[str, Any]:
        with self._connect() as connection:
            query_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM ai_query_logs
                """
            ).fetchone()[0]

            feedback_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM ai_answer_feedback
                """
            ).fetchone()[0]

        return {
            "status": "UP",
            "databasePath": str(
                self.database_path
            ),
            "queryCount": query_count,
            "feedbackCount": feedback_count,
        }

    def record_query(
        self,
        *,
        question: str,
        request_id: str | None = None,
        actor: str | None = None,
        requested_language: str | None = None,
        detected_language: str | None = None,
        response_language: str | None = None,
        intent: str | None = None,
        retrieval_mode: str | None = None,
        time_scope: str | None = None,
        from_time_utc: str | None = None,
        to_time_utc: str | None = None,
        grounded: bool | None = None,
        evidence_count: int = 0,
        filtered_total: int | None = None,
        tools_used: Iterable[str] | None = None,
        latency_ms: int | None = None,
        answer: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        query_id: str | None = None,
    ) -> str:
        normalized_question = (
            normalize_optional_text(
                question,
                maximum_length=4000,
            )
        )

        if not normalized_question:
            raise ValueError(
                "question boş olamaz."
            )

        telemetry_id = (
            normalize_optional_text(query_id)
            or str(uuid.uuid4())
        )

        tools_json = json.dumps(
            list(tools_used or []),
            ensure_ascii=False,
        )

        with self._write_lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO ai_query_logs (
                        id,
                        request_id,
                        actor,
                        question,
                        requested_language,
                        detected_language,
                        response_language,
                        intent,
                        retrieval_mode,
                        time_scope,
                        from_time_utc,
                        to_time_utc,
                        grounded,
                        evidence_count,
                        filtered_total,
                        tools_used_json,
                        latency_ms,
                        answer,
                        error_type,
                        error_message,
                        created_at_utc
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        telemetry_id,
                        normalize_optional_text(
                            request_id,
                            200,
                        ),
                        normalize_optional_text(
                            actor,
                            200,
                        ),
                        normalized_question,
                        normalize_optional_text(
                            requested_language,
                            20,
                        ),
                        normalize_optional_text(
                            detected_language,
                            20,
                        ),
                        normalize_optional_text(
                            response_language,
                            20,
                        ),
                        normalize_optional_text(
                            intent,
                            100,
                        ),
                        normalize_optional_text(
                            retrieval_mode,
                            100,
                        ),
                        normalize_optional_text(
                            time_scope,
                            100,
                        ),
                        normalize_optional_text(
                            from_time_utc,
                            100,
                        ),
                        normalize_optional_text(
                            to_time_utc,
                            100,
                        ),
                        boolean_to_integer(
                            grounded
                        ),
                        max(
                            int(evidence_count),
                            0,
                        ),
                        filtered_total,
                        tools_json,
                        latency_ms,
                        normalize_optional_text(
                            answer,
                            20000,
                        ),
                        normalize_optional_text(
                            error_type,
                            300,
                        ),
                        normalize_optional_text(
                            error_message,
                            4000,
                        ),
                        utc_now_text(),
                    ),
                )

        return telemetry_id

    def record_feedback(
        self,
        *,
        query_id: str,
        rating: str,
        reason: str | None = None,
        comment: str | None = None,
        corrected_answer: str | None = None,
        actor: str | None = None,
        request_id: str | None = None,
        feedback_id: str | None = None,
    ) -> str:
        normalized_query_id = (
            normalize_optional_text(query_id)
        )

        if not normalized_query_id:
            raise ValueError(
                "query_id boş olamaz."
            )

        normalized_rating = (
            str(rating)
            .strip()
            .upper()
        )

        if (
            normalized_rating
            not in VALID_FEEDBACK_RATINGS
        ):
            raise ValueError(
                "rating yalnızca UP veya DOWN olabilir."
            )

        normalized_reason = (
            normalize_optional_text(
                reason,
                100,
            )
        )

        if normalized_reason:
            normalized_reason = (
                normalized_reason.upper()
            )

            if (
                normalized_reason
                not in VALID_FEEDBACK_REASONS
            ):
                raise ValueError(
                    "Geçersiz feedback reason: "
                    f"{normalized_reason}"
                )

        if (
            normalized_rating == "DOWN"
            and normalized_reason is None
        ):
            raise ValueError(
                "DOWN değerlendirmesinde reason zorunludur."
            )

        telemetry_feedback_id = (
            normalize_optional_text(
                feedback_id
            )
            or str(uuid.uuid4())
        )

        with self._write_lock:
            with self._connect() as connection:
                query_exists = connection.execute(
                    """
                    SELECT 1
                    FROM ai_query_logs
                    WHERE id = ?
                    """,
                    (normalized_query_id,),
                ).fetchone()

                if query_exists is None:
                    raise LookupError(
                        "Feedback verilecek query bulunamadı: "
                        f"{normalized_query_id}"
                    )

                connection.execute(
                    """
                    INSERT INTO ai_answer_feedback (
                        id,
                        query_id,
                        request_id,
                        rating,
                        reason,
                        comment,
                        corrected_answer,
                        actor,
                        created_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        telemetry_feedback_id,
                        normalized_query_id,
                        normalize_optional_text(
                            request_id,
                            200,
                        ),
                        normalized_rating,
                        normalized_reason,
                        normalize_optional_text(
                            comment,
                            4000,
                        ),
                        normalize_optional_text(
                            corrected_answer,
                            20000,
                        ),
                        normalize_optional_text(
                            actor,
                            200,
                        ),
                        utc_now_text(),
                    ),
                )

        return telemetry_feedback_id

    def recent_queries(
        self,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        safe_limit = min(
            max(int(limit), 1),
            500,
        )

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    q.*,
                    (
                        SELECT f.rating
                        FROM ai_answer_feedback f
                        WHERE f.query_id = q.id
                        ORDER BY f.created_at_utc DESC
                        LIMIT 1
                    ) AS latest_feedback_rating,
                    (
                        SELECT f.reason
                        FROM ai_answer_feedback f
                        WHERE f.query_id = q.id
                        ORDER BY f.created_at_utc DESC
                        LIMIT 1
                    ) AS latest_feedback_reason
                FROM ai_query_logs q
                ORDER BY q.created_at_utc DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        results: list[dict[str, Any]] = []

        for row in rows:
            item = dict(row)

            item["grounded"] = (
                integer_to_boolean(
                    item.get("grounded")
                )
            )

            try:
                item["tools_used"] = json.loads(
                    item.pop(
                        "tools_used_json",
                        "[]",
                    )
                )
            except json.JSONDecodeError:
                item["tools_used"] = []

            results.append(item)

        return results

    def feedback_statistics(
        self,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            total_queries = connection.execute(
                """
                SELECT COUNT(*)
                FROM ai_query_logs
                """
            ).fetchone()[0]

            total_feedback = connection.execute(
                """
                SELECT COUNT(*)
                FROM ai_answer_feedback
                """
            ).fetchone()[0]

            ratings = connection.execute(
                """
                SELECT rating, COUNT(*) AS total
                FROM ai_answer_feedback
                GROUP BY rating
                ORDER BY rating
                """
            ).fetchall()

            reasons = connection.execute(
                """
                SELECT reason, COUNT(*) AS total
                FROM ai_answer_feedback
                WHERE reason IS NOT NULL
                GROUP BY reason
                ORDER BY total DESC
                """
            ).fetchall()

        up_count = 0
        down_count = 0

        rating_counts: dict[str, int] = {}

        for row in ratings:
            rating = str(row["rating"])
            count = int(row["total"])

            rating_counts[rating] = count

            if rating == "UP":
                up_count = count
            elif rating == "DOWN":
                down_count = count

        feedback_accuracy = None

        if up_count + down_count > 0:
            feedback_accuracy = round(
                up_count
                / (up_count + down_count)
                * 100,
                2,
            )

        return {
            "totalQueries": total_queries,
            "totalFeedback": total_feedback,
            "ratingCounts": rating_counts,
            "reasonCounts": {
                str(row["reason"]): int(
                    row["total"]
                )
                for row in reasons
            },
            "feedbackAccuracyPercent": (
                feedback_accuracy
            ),
        }

    def export_training_candidates(
        self,
        output_path: str | Path,
    ) -> int:
        destination = Path(
            output_path
        ).expanduser()

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    q.id AS query_id,
                    q.request_id,
                    q.question,
                    q.requested_language,
                    q.detected_language,
                    q.response_language,
                    q.intent,
                    q.retrieval_mode,
                    q.time_scope,
                    q.from_time_utc,
                    q.to_time_utc,
                    q.grounded,
                    q.evidence_count,
                    q.tools_used_json,
                    q.answer AS original_answer,
                    f.id AS feedback_id,
                    f.rating,
                    f.reason,
                    f.comment,
                    f.corrected_answer,
                    f.actor AS reviewer,
                    f.created_at_utc AS reviewed_at_utc
                FROM ai_query_logs q
                INNER JOIN ai_answer_feedback f
                    ON f.query_id = q.id
                WHERE f.rating = 'DOWN'
                ORDER BY f.created_at_utc ASC
                """
            ).fetchall()

        exported = 0

        with destination.open(
            "w",
            encoding="utf-8",
        ) as file:
            for row in rows:
                item = dict(row)

                try:
                    tools_used = json.loads(
                        item.pop(
                            "tools_used_json",
                            "[]",
                        )
                    )
                except json.JSONDecodeError:
                    tools_used = []

                item["toolsUsed"] = tools_used
                item["grounded"] = (
                    integer_to_boolean(
                        item.get("grounded")
                    )
                )

                file.write(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                exported += 1

        return exported
