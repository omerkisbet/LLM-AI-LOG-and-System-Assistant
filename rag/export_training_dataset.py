
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)
IPV4_RE = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)
JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{10,}\."
    r"[A-Za-z0-9_-]{10,}\."
    r"[A-Za-z0-9_-]{10,}\b"
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret|authorization)"
    r"\s*[:=]\s*([^\s,;]+)"
)
BEARER_RE = re.compile(
    r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"
)


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def redact_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = IPV4_RE.sub("[IP]", text)
    text = JWT_RE.sub("[JWT]", text)
    text = BEARER_RE.sub("Bearer [REDACTED]", text)
    text = SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}=[REDACTED]",
        text,
    )
    return text


def redact_json(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
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
                result[key] = "[REDACTED]"
            else:
                result[key] = redact_json(item)
        return result

    if isinstance(value, list):
        return [redact_json(item) for item in value]

    if isinstance(value, str):
        return redact_text(value)

    return value


def parse_json(value: Any, default: Any) -> Any:
    if value is None:
        return default

    try:
        return json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return default


def stable_bucket(query_id: str) -> str:
    digest = hashlib.sha256(query_id.encode("utf-8")).digest()
    return "validation" if digest[0] < 51 else "train"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def latest_feedback_by_query(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT *
        FROM ai_answer_feedback
        ORDER BY created_at_utc ASC
        """
    ).fetchall()

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        result[str(row["query_id"])] = dict(row)
    return result


def build_sft_record(
    query: dict[str, Any],
    feedback: dict[str, Any],
    chosen_answer: str,
) -> dict[str, Any]:
    evidence = redact_json(
        parse_json(query.get("evidence_json"), [])
    )
    plan = redact_json(
        parse_json(query.get("query_plan_json"), {})
    )
    operational_summary = redact_json(
        parse_json(
            query.get("operational_summary_json"),
            {},
        )
    )

    system_message = (
        "You are the OMU IoT Lab operational log assistant. "
        "Use only the supplied query plan, operational summary, "
        "and evidence. Never invent logs or unsupported facts. "
        "Answer in the requested response language."
    )

    context = {
        "responseLanguage": query.get("response_language"),
        "queryPlan": plan,
        "operationalSummary": operational_summary,
        "evidence": evidence,
    }

    user_message = (
        f"Question:\n{redact_text(query.get('question'))}\n\n"
        "Grounded context:\n"
        + json.dumps(
            context,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    return {
        "id": query["id"],
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {
                "role": "assistant",
                "content": redact_text(chosen_answer),
            },
        ],
        "metadata": {
            "intent": query.get("intent"),
            "retrievalMode": query.get("retrieval_mode"),
            "timeScope": query.get("time_scope"),
            "grounded": bool(query.get("grounded")),
            "rating": feedback.get("rating"),
            "feedbackReason": feedback.get("reason"),
            "promptVersion": query.get("prompt_version"),
            "modelName": query.get("model_name"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export sanitized IoT Lab fine-tuning datasets."
    )
    parser.add_argument("--database", default=None)
    parser.add_argument("--output-dir", default=None)
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

    output_dir = Path(
        args.output_dir
        or (
            rag_dir
            / "training"
            / datetime.now(timezone.utc).strftime(
                "export-%Y%m%dT%H%M%SZ"
            )
        )
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row

    columns = {
        str(row["name"])
        for row in connection.execute(
            'PRAGMA table_info("ai_query_logs")'
        ).fetchall()
    }

    required_v2 = {
        "evidence_json",
        "query_plan_json",
        "operational_summary_json",
        "prompt_version",
        "model_name",
    }
    missing = sorted(required_v2.difference(columns))

    if missing:
        raise SystemExit(
            "Telemetry v2 migration gerekli. Eksik kolonlar: "
            + ", ".join(missing)
        )

    queries = [
        dict(row)
        for row in connection.execute(
            """
            SELECT *
            FROM ai_query_logs
            ORDER BY created_at_utc ASC
            """
        ).fetchall()
    ]
    feedback_map = latest_feedback_by_query(connection)
    connection.close()

    raw_candidates: list[dict[str, Any]] = []
    legacy_review: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for query in queries:
        query_id = str(query["id"])
        feedback = feedback_map.get(query_id)
        evidence = parse_json(query.get("evidence_json"), [])
        plan = parse_json(query.get("query_plan_json"), {})

        sanitized = {
            "queryId": query_id,
            "question": redact_text(query.get("question")),
            "answer": redact_text(query.get("answer")),
            "intent": query.get("intent"),
            "retrievalMode": query.get("retrieval_mode"),
            "timeScope": query.get("time_scope"),
            "responseLanguage": query.get("response_language"),
            "grounded": bool(query.get("grounded")),
            "evidence": redact_json(evidence),
            "queryPlan": redact_json(plan),
            "operationalSummary": redact_json(
                parse_json(
                    query.get("operational_summary_json"),
                    {},
                )
            ),
            "feedback": (
                {
                    key: redact_text(value)
                    if isinstance(value, str)
                    else value
                    for key, value in feedback.items()
                    if key not in {"id", "query_id"}
                }
                if feedback
                else None
            ),
            "promptVersion": query.get("prompt_version"),
            "modelName": query.get("model_name"),
            "createdAtUtc": query.get("created_at_utc"),
        }
        raw_candidates.append(sanitized)

        has_context = bool(evidence) and bool(plan)
        has_error = bool(
            str(query.get("error_type") or "").strip()
        )

        if not feedback:
            reason_counts["NO_FEEDBACK"] += 1
            rejected.append(
                {
                    "queryId": query_id,
                    "reason": "NO_FEEDBACK",
                }
            )
            continue

        if not has_context:
            reason_counts["LEGACY_WITHOUT_EVIDENCE"] += 1
            legacy_review.append(sanitized)
            continue

        if has_error:
            reason_counts["QUERY_ERROR"] += 1
            rejected.append(
                {
                    "queryId": query_id,
                    "reason": "QUERY_ERROR",
                }
            )
            continue

        rating = str(feedback.get("rating") or "").upper()
        corrected_answer = str(
            feedback.get("corrected_answer") or ""
        ).strip()

        if rating == "UP":
            chosen_answer = str(query.get("answer") or "").strip()
        elif rating == "DOWN" and corrected_answer:
            chosen_answer = corrected_answer
        else:
            reason = (
                "DOWN_WITHOUT_CORRECTION"
                if rating == "DOWN"
                else "UNSUPPORTED_RATING"
            )
            reason_counts[reason] += 1
            rejected.append(
                {
                    "queryId": query_id,
                    "reason": reason,
                }
            )
            continue

        if not chosen_answer:
            reason_counts["EMPTY_CHOSEN_ANSWER"] += 1
            rejected.append(
                {
                    "queryId": query_id,
                    "reason": "EMPTY_CHOSEN_ANSWER",
                }
            )
            continue

        record = build_sft_record(
            query,
            feedback,
            chosen_answer,
        )

        if stable_bucket(query_id) == "validation":
            validation.append(record)
        else:
            train.append(record)

    write_jsonl(output_dir / "raw_candidates.jsonl", raw_candidates)
    write_jsonl(
        output_dir / "legacy_review_queue.jsonl",
        legacy_review,
    )
    write_jsonl(
        output_dir / "rejected_candidates.jsonl",
        rejected,
    )
    write_jsonl(output_dir / "sft_train.jsonl", train)
    write_jsonl(
        output_dir / "sft_validation.jsonl",
        validation,
    )

    manifest = {
        "generatedAtUtc": utc_now_text(),
        "database": str(database),
        "counts": {
            "queries": len(queries),
            "feedback": len(feedback_map),
            "rawCandidates": len(raw_candidates),
            "legacyReviewQueue": len(legacy_review),
            "rejected": len(rejected),
            "sftTrain": len(train),
            "sftValidation": len(validation),
        },
        "rejectionReasons": dict(reason_counts),
        "policy": {
            "trainingRequiresLatestFeedback": True,
            "trainingRequiresEvidenceSnapshot": True,
            "trainingRequiresQueryPlan": True,
            "upUsesOriginalAnswer": True,
            "downRequiresCorrectedAnswer": True,
            "legacyRowsAreNotAutomaticallyTrained": True,
            "redactionEnabled": True,
        },
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Çıktı klasörü:", output_dir)


if __name__ == "__main__":
    main()
