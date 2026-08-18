from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


_ALLOWED_REVIEW_STATUSES = {"APPROVED", "REJECTED"}
_ALLOWED_LIST_STATUSES = {
    "PENDING",
    "APPROVED",
    "REJECTED",
    "LEGACY",
    "NO_FEEDBACK",
}

_EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{10,}\."
    r"[A-Za-z0-9_-]{10,}\."
    r"[A-Za-z0-9_-]{10,}\b"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret|authorization)"
    r"\s*[:=]\s*([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def _redact_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _IPV4_RE.sub("[IP]", text)
    text = _JWT_RE.sub("[JWT]", text)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}=[REDACTED]",
        text,
    )
    return text


def _redact_json(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).casefold()
            if any(
                token in lowered
                for token in (
                    "password",
                    "secret",
                    "api_key",
                    "apikey",
                    "authorization",
                    "token",
                )
            ):
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = _redact_json(item)
        return result
    if isinstance(value, list):
        return [_redact_json(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _stable_bucket(query_id: str, validation_percent: int) -> str:
    digest = hashlib.sha256(query_id.encode("utf-8")).digest()
    threshold = round(256 * validation_percent / 100)
    return "validation" if digest[0] < threshold else "train"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


class TrainingManagementService:
    """Review and export grounded training candidates from telemetry SQLite."""

    def __init__(
        self,
        database_path: str | Path,
        output_root: str | Path,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _migrate(self) -> None:
        with self._lock:
            with self._connect() as connection:
                query_table = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table' AND name = 'ai_query_logs'
                    """
                ).fetchone()
                feedback_table = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table' AND name = 'ai_answer_feedback'
                    """
                ).fetchone()
                if query_table is None or feedback_table is None:
                    raise RuntimeError(
                        "Training management requires ai_query_logs and "
                        "ai_answer_feedback tables."
                    )

                required_columns = {
                    "evidence_json",
                    "query_plan_json",
                    "operational_summary_json",
                    "prompt_version",
                    "model_name",
                }
                current_columns = {
                    str(row["name"])
                    for row in connection.execute(
                        'PRAGMA table_info("ai_query_logs")'
                    ).fetchall()
                }
                missing = sorted(required_columns.difference(current_columns))
                if missing:
                    raise RuntimeError(
                        "Telemetry v2 migration is required. Missing columns: "
                        + ", ".join(missing)
                    )

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS training_candidate_reviews (
                        query_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        note TEXT,
                        corrected_answer TEXT,
                        reviewed_at_utc TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL,
                        CHECK(status IN ('APPROVED', 'REJECTED'))
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS training_dataset_exports (
                        id TEXT PRIMARY KEY,
                        actor TEXT NOT NULL,
                        output_dir TEXT NOT NULL,
                        manifest_json TEXT NOT NULL,
                        created_at_utc TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_training_reviews_status
                    ON training_candidate_reviews(status)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_training_exports_created
                    ON training_dataset_exports(created_at_utc DESC)
                    """
                )

    def health(self) -> dict[str, Any]:
        with self._connect() as connection:
            review_count = connection.execute(
                "SELECT COUNT(*) AS total FROM training_candidate_reviews"
            ).fetchone()["total"]
            export_count = connection.execute(
                "SELECT COUNT(*) AS total FROM training_dataset_exports"
            ).fetchone()["total"]
        return {
            "status": "UP",
            "databasePath": str(self.database_path),
            "outputRoot": str(self.output_root),
            "reviewCount": int(review_count),
            "exportCount": int(export_count),
        }

    @staticmethod
    def _base_query() -> str:
        return """
            WITH ranked_feedback AS (
                SELECT
                    f.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY f.query_id
                        ORDER BY f.created_at_utc DESC, f.id DESC
                    ) AS row_number
                FROM ai_answer_feedback f
            ),
            latest_feedback AS (
                SELECT * FROM ranked_feedback WHERE row_number = 1
            )
            SELECT
                q.*,
                f.id AS feedback_id,
                f.rating AS feedback_rating,
                f.reason AS feedback_reason,
                f.comment AS feedback_comment,
                f.corrected_answer AS feedback_corrected_answer,
                f.actor AS feedback_actor,
                f.created_at_utc AS feedback_created_at_utc,
                r.status AS review_status,
                r.actor AS review_actor,
                r.note AS review_note,
                r.corrected_answer AS review_corrected_answer,
                r.reviewed_at_utc,
                r.updated_at_utc AS review_updated_at_utc
            FROM ai_query_logs q
            LEFT JOIN latest_feedback f ON f.query_id = q.id
            LEFT JOIN training_candidate_reviews r ON r.query_id = q.id
        """

    @staticmethod
    def _classification(row: dict[str, Any]) -> tuple[str, bool]:
        evidence = _parse_json(row.get("evidence_json"), [])
        query_plan = _parse_json(row.get("query_plan_json"), {})
        has_feedback = bool(row.get("feedback_id"))
        eligible = (
            has_feedback
            and bool(row.get("grounded"))
            and isinstance(evidence, list)
            and len(evidence) > 0
            and isinstance(query_plan, dict)
            and len(query_plan) > 0
            and not str(row.get("error_type") or "").strip()
        )
        if eligible:
            review_status = str(row.get("review_status") or "").strip()
            review_updated = str(
                row.get("review_updated_at_utc") or ""
            ).strip()
            feedback_created = str(
                row.get("feedback_created_at_utc") or ""
            ).strip()
            review_is_current = bool(review_status) and (
                not feedback_created
                or (review_updated and review_updated >= feedback_created)
            )
            return (
                review_status if review_is_current else "PENDING",
                True,
            )
        if has_feedback:
            return "LEGACY", False
        return "NO_FEEDBACK", False

    @classmethod
    def _candidate_payload(
        cls,
        row: dict[str, Any],
        *,
        include_context: bool,
    ) -> dict[str, Any]:
        status, eligible = cls._classification(row)
        payload: dict[str, Any] = {
            "queryId": str(row["id"]),
            "requestId": row.get("request_id"),
            "status": status,
            "eligible": eligible,
            "question": _redact_text(row.get("question")),
            "answer": _redact_text(row.get("answer")),
            "intent": row.get("intent"),
            "retrievalMode": row.get("retrieval_mode"),
            "timeScope": row.get("time_scope"),
            "responseLanguage": row.get("response_language"),
            "grounded": bool(row.get("grounded")),
            "evidenceCount": int(row.get("evidence_count") or 0),
            "filteredTotal": int(row.get("filtered_total") or 0),
            "latencyMs": int(row.get("latency_ms") or 0),
            "promptVersion": row.get("prompt_version"),
            "modelName": row.get("model_name"),
            "createdAtUtc": row.get("created_at_utc"),
            "feedback": None,
            "review": None,
        }
        if row.get("feedback_id"):
            payload["feedback"] = {
                "id": row.get("feedback_id"),
                "rating": row.get("feedback_rating"),
                "reason": row.get("feedback_reason"),
                "comment": _redact_text(row.get("feedback_comment")),
                "correctedAnswer": _redact_text(
                    row.get("feedback_corrected_answer")
                ),
                "actor": row.get("feedback_actor"),
                "createdAtUtc": row.get("feedback_created_at_utc"),
            }
        if row.get("review_status"):
            review_updated = str(
                row.get("review_updated_at_utc") or ""
            ).strip()
            feedback_created = str(
                row.get("feedback_created_at_utc") or ""
            ).strip()
            payload["review"] = {
                "status": row.get("review_status"),
                "stale": bool(
                    feedback_created
                    and (
                        not review_updated
                        or review_updated < feedback_created
                    )
                ),
                "actor": row.get("review_actor"),
                "note": _redact_text(row.get("review_note")),
                "correctedAnswer": _redact_text(
                    row.get("review_corrected_answer")
                ),
                "reviewedAtUtc": row.get("reviewed_at_utc"),
                "updatedAtUtc": row.get("review_updated_at_utc"),
            }
        if include_context:
            payload["queryPlan"] = _redact_json(
                _parse_json(row.get("query_plan_json"), {})
            )
            payload["operationalSummary"] = _redact_json(
                _parse_json(row.get("operational_summary_json"), {})
            )
            payload["evidence"] = _redact_json(
                _parse_json(row.get("evidence_json"), [])
            )
            payload["toolsUsed"] = _redact_json(
                _parse_json(row.get("tools_used_json"), [])
            )
        return payload

    def _fetch_all_rows(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                self._base_query() + " ORDER BY q.created_at_utc DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def statistics(self) -> dict[str, Any]:
        rows = self._fetch_all_rows()
        status_counts: Counter[str] = Counter()
        intent_counts: Counter[str] = Counter()
        language_counts: Counter[str] = Counter()
        for row in rows:
            status, _ = self._classification(row)
            status_counts[status] += 1
            intent_counts[str(row.get("intent") or "UNKNOWN")] += 1
            language_counts[str(row.get("response_language") or "unknown")] += 1
        approved = status_counts["APPROVED"]
        return {
            "totalQueries": len(rows),
            "pending": status_counts["PENDING"],
            "approved": approved,
            "rejected": status_counts["REJECTED"],
            "legacy": status_counts["LEGACY"],
            "noFeedback": status_counts["NO_FEEDBACK"],
            "eligible": (
                status_counts["PENDING"]
                + status_counts["APPROVED"]
                + status_counts["REJECTED"]
            ),
            "byStatus": dict(status_counts),
            "byIntent": dict(intent_counts),
            "byLanguage": dict(language_counts),
            "lastExport": self.list_exports(limit=1)[0]
            if self.list_exports(limit=1)
            else None,
        }

    def list_candidates(
        self,
        *,
        status: str | None,
        limit: int,
        offset: int,
        search: str | None,
    ) -> dict[str, Any]:
        normalized_status = status.strip().upper() if status else None
        if normalized_status and normalized_status not in _ALLOWED_LIST_STATUSES:
            raise ValueError(
                "Unsupported status. Allowed: "
                + ", ".join(sorted(_ALLOWED_LIST_STATUSES))
            )
        safe_limit = min(max(int(limit), 1), 200)
        safe_offset = max(int(offset), 0)
        normalized_search = (search or "").strip().casefold()

        matches: list[dict[str, Any]] = []
        for row in self._fetch_all_rows():
            candidate_status, _ = self._classification(row)
            if normalized_status and candidate_status != normalized_status:
                continue
            if normalized_search:
                haystack = " ".join(
                    str(row.get(key) or "")
                    for key in (
                        "question",
                        "answer",
                        "intent",
                        "retrieval_mode",
                        "feedback_reason",
                    )
                ).casefold()
                if normalized_search not in haystack:
                    continue
            matches.append(row)

        page = matches[safe_offset : safe_offset + safe_limit]
        return {
            "items": [
                self._candidate_payload(row, include_context=False)
                for row in page
            ],
            "count": len(page),
            "total": len(matches),
            "limit": safe_limit,
            "offset": safe_offset,
        }

    def get_candidate(self, query_id: str) -> dict[str, Any]:
        normalized = query_id.strip()
        if not normalized:
            raise ValueError("query_id is required.")
        with self._connect() as connection:
            row = connection.execute(
                self._base_query() + " WHERE q.id = ?",
                (normalized,),
            ).fetchone()
        if row is None:
            raise LookupError(f"Training candidate not found: {normalized}")
        return self._candidate_payload(dict(row), include_context=True)

    def review_candidate(
        self,
        *,
        query_id: str,
        status: str,
        actor: str,
        note: str | None,
        corrected_answer: str | None,
    ) -> dict[str, Any]:
        normalized_status = status.strip().upper()
        if normalized_status not in _ALLOWED_REVIEW_STATUSES:
            raise ValueError("Review status must be APPROVED or REJECTED.")
        normalized_actor = actor.strip()
        if not normalized_actor:
            raise ValueError("actor is required.")

        candidate = self.get_candidate(query_id)
        if not candidate["eligible"]:
            raise ValueError(
                "Only grounded candidates with evidence, a query plan, "
                "and feedback can be reviewed."
            )

        feedback = candidate.get("feedback") or {}
        selected_correction = (corrected_answer or "").strip() or None
        if (
            normalized_status == "APPROVED"
            and str(feedback.get("rating") or "").upper() == "DOWN"
            and not selected_correction
            and not str(feedback.get("correctedAnswer") or "").strip()
        ):
            raise ValueError(
                "A negatively rated candidate requires a corrected answer "
                "before approval."
            )

        now = _utc_now()
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO training_candidate_reviews (
                        query_id,
                        status,
                        actor,
                        note,
                        corrected_answer,
                        reviewed_at_utc,
                        updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(query_id) DO UPDATE SET
                        status = excluded.status,
                        actor = excluded.actor,
                        note = excluded.note,
                        corrected_answer = excluded.corrected_answer,
                        updated_at_utc = excluded.updated_at_utc
                    """,
                    (
                        query_id.strip(),
                        normalized_status,
                        normalized_actor[:200],
                        (note or "").strip()[:4000] or None,
                        selected_correction[:20000]
                        if selected_correction
                        else None,
                        now,
                        now,
                    ),
                )
        return self.get_candidate(query_id)

    @staticmethod
    def _sft_record(candidate: dict[str, Any], chosen_answer: str) -> dict[str, Any]:
        context = {
            "responseLanguage": candidate.get("responseLanguage"),
            "queryPlan": candidate.get("queryPlan") or {},
            "operationalSummary": candidate.get("operationalSummary") or {},
            "evidence": candidate.get("evidence") or [],
        }
        return {
            "id": candidate["queryId"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the OMU IoT Lab operational log assistant. "
                        "Use only the supplied query plan, operational summary, "
                        "and evidence. Never invent logs or unsupported facts. "
                        "Answer in the requested response language."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{candidate.get('question') or ''}\n\n"
                        "Grounded context:\n"
                        + json.dumps(
                            context,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ),
                },
                {
                    "role": "assistant",
                    "content": _redact_text(chosen_answer) or "",
                },
            ],
            "metadata": {
                "intent": candidate.get("intent"),
                "retrievalMode": candidate.get("retrievalMode"),
                "timeScope": candidate.get("timeScope"),
                "grounded": candidate.get("grounded"),
                "rating": (candidate.get("feedback") or {}).get("rating"),
                "feedbackReason": (candidate.get("feedback") or {}).get("reason"),
                "promptVersion": candidate.get("promptVersion"),
                "modelName": candidate.get("modelName"),
                "reviewActor": (candidate.get("review") or {}).get("actor"),
                "reviewedAtUtc": (candidate.get("review") or {}).get(
                    "reviewedAtUtc"
                ),
            },
        }

    def export_approved(
        self,
        *,
        actor: str,
        validation_percent: int = 20,
    ) -> dict[str, Any]:
        normalized_actor = actor.strip()
        if not normalized_actor:
            raise ValueError("actor is required.")
        safe_validation = min(max(int(validation_percent), 5), 40)

        approved_rows = [
            row
            for row in self._fetch_all_rows()
            if self._classification(row)[0] == "APPROVED"
        ]

        train: list[dict[str, Any]] = []
        validation: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        approved_snapshot: list[dict[str, Any]] = []

        for row in approved_rows:
            candidate = self._candidate_payload(row, include_context=True)
            feedback = candidate.get("feedback") or {}
            review = candidate.get("review") or {}
            chosen_answer = str(
                review.get("correctedAnswer")
                or feedback.get("correctedAnswer")
                or candidate.get("answer")
                or ""
            ).strip()
            if not chosen_answer:
                rejected.append(
                    {
                        "queryId": candidate["queryId"],
                        "reason": "EMPTY_APPROVED_ANSWER",
                    }
                )
                continue
            record = self._sft_record(candidate, chosen_answer)
            approved_snapshot.append(candidate)
            if _stable_bucket(candidate["queryId"], safe_validation) == "validation":
                validation.append(record)
            else:
                train.append(record)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        export_id = str(uuid.uuid4())
        output_dir = self.output_root / f"approved-export-{timestamp}"
        suffix = 1
        while output_dir.exists():
            suffix += 1
            output_dir = self.output_root / f"approved-export-{timestamp}-{suffix}"
        output_dir.mkdir(parents=True, exist_ok=False)

        _write_jsonl(output_dir / "approved_candidates.jsonl", approved_snapshot)
        _write_jsonl(output_dir / "sft_train.jsonl", train)
        _write_jsonl(output_dir / "sft_validation.jsonl", validation)
        _write_jsonl(output_dir / "rejected_candidates.jsonl", rejected)

        manifest = {
            "exportId": export_id,
            "generatedAtUtc": _utc_now(),
            "actor": normalized_actor,
            "database": str(self.database_path),
            "outputDir": str(output_dir),
            "validationPercent": safe_validation,
            "counts": {
                "approvedCandidates": len(approved_rows),
                "exportedCandidates": len(train) + len(validation),
                "sftTrain": len(train),
                "sftValidation": len(validation),
                "rejected": len(rejected),
            },
            "policy": {
                "explicitAdminApprovalRequired": True,
                "groundedEvidenceRequired": True,
                "queryPlanRequired": True,
                "negativeFeedbackRequiresCorrection": True,
                "redactionEnabled": True,
                "stableHashSplit": True,
            },
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO training_dataset_exports (
                        id, actor, output_dir, manifest_json, created_at_utc
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        export_id,
                        normalized_actor[:200],
                        str(output_dir),
                        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                        manifest["generatedAtUtc"],
                    ),
                )
        return manifest

    def list_exports(self, limit: int = 20) -> list[dict[str, Any]]:
        safe_limit = min(max(int(limit), 1), 100)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, actor, output_dir, manifest_json, created_at_utc
                FROM training_dataset_exports
                ORDER BY created_at_utc DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            manifest = _parse_json(row["manifest_json"], {})
            results.append(
                {
                    "id": row["id"],
                    "actor": row["actor"],
                    "outputDir": row["output_dir"],
                    "createdAtUtc": row["created_at_utc"],
                    "manifest": manifest,
                }
            )
        return results
