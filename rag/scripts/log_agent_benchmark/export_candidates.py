#!/usr/bin/env python3
"""Export representative incident candidates from Qdrant for Log Agent benchmark annotation."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient


ERROR_KEYWORDS = re.compile(
    r"\b("
    r"exception|error|failed|failure|timeout|timed out|"
    r"denied|unauthorized|forbidden|unavailable|refused|"
    r"not found|connection reset|broken pipe|out of memory|"
    r"authentication|permission|traceback|stacktrace"
    r")\b",
    re.IGNORECASE,
)

CATEGORY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("database", re.compile(
        r"mongo|mongodb|sql|jdbc|database|db\b|redis|qdrant|"
        r"connection pool|authentication failed",
        re.IGNORECASE,
    )),
    ("network_dns", re.compile(
        r"dns|network|socket|connection refused|connection reset|"
        r"unknown host|host unreachable|route|tls|ssl|certificate",
        re.IGNORECASE,
    )),
    ("authentication", re.compile(
        r"unauthorized|forbidden|authentication|authorization|"
        r"invalid token|access denied|401|403",
        re.IGNORECASE,
    )),
    ("http_api", re.compile(
        r"http|endpoint|request|response|status code|400|404|405|"
        r"429|500|502|503|504|rest",
        re.IGNORECASE,
    )),
    ("docker_container", re.compile(
        r"docker|container|compose|image|healthcheck|exit code|"
        r"restart policy|volume|port is already allocated",
        re.IGNORECASE,
    )),
    ("filesystem_permission", re.compile(
        r"permission denied|accessdenied|read-only file system|"
        r"no such file|file not found|directory|filesystem|upload",
        re.IGNORECASE,
    )),
    ("resource_usage", re.compile(
        r"out of memory|oom|memory|cpu|disk full|no space left|"
        r"gpu|cuda|resource exhausted",
        re.IGNORECASE,
    )),
    ("application_exception", re.compile(
        r"exception|traceback|nullpointer|illegalargument|runtimeerror|"
        r"valueerror|typeerror|module not found|modulenotfound",
        re.IGNORECASE,
    )),
]


def classify_category(text: str) -> str:
    for name, pattern in CATEGORY_RULES:
        if pattern.search(text):
            return name
    return "other"


def normalize_level(payload: dict[str, Any]) -> str:
    return str(payload.get("level") or "").strip().upper()


def status_code(payload: dict[str, Any]) -> int | None:
    value = payload.get("statusCode")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def is_incident(payload: dict[str, Any]) -> bool:
    content = str(payload.get("content") or "")
    level = normalize_level(payload)
    code = status_code(payload)

    return (
        level in {"ERROR", "WARN", "WARNING", "FATAL", "CRITICAL"}
        or (code is not None and code >= 400)
        or bool(ERROR_KEYWORDS.search(content))
    )


def parse_timestamp(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def group_key(point_id: str, payload: dict[str, Any]) -> str:
    for field in ("requestId", "traceId", "documentId", "contentHash"):
        value = payload.get(field)
        if value:
            return f"{field}:{value}"

    source = payload.get("source") or payload.get("logger") or "unknown"
    timestamp = str(payload.get("timestamp") or "")
    minute_bucket = timestamp[:16] if len(timestamp) >= 16 else timestamp
    return f"fallback:{source}:{minute_bucket}:{point_id}"


def draft_question(category: str) -> str:
    questions = {
        "database": "Bu olayda veritabanı erişimi neden başarısız olmuş olabilir?",
        "network_dns": "Bu loglara göre ağ veya DNS sorununun olası kök nedeni nedir?",
        "authentication": "Bu kimlik doğrulama/yetkilendirme hatasının nedeni nedir?",
        "http_api": "Bu API isteği neden başarısız olmuş ve hangi katmanda hata oluşmuştur?",
        "docker_container": "Bu container olayının kök nedeni ve önerilen müdahale nedir?",
        "filesystem_permission": "Bu dosya veya izin hatasının nedeni ve çözümü nedir?",
        "resource_usage": "Bu kaynak kullanımı probleminin kök nedeni nedir?",
        "application_exception": "Bu exception zincirindeki asıl kök neden nedir?",
        "other": "Bu log grubunda bir hata var mı; varsa kök nedeni nedir?",
    }
    return questions.get(category, questions["other"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=Path("rag/.env"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rag/data/log_agent_benchmark/candidates.jsonl"),
    )
    parser.add_argument("--candidate-count", type=int, default=80)
    parser.add_argument("--max-lines-per-incident", type=int, default=12)
    args = parser.parse_args()

    load_dotenv(args.env)

    required = ["QDRANT_URL", "QDRANT_COLLECTION"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("Eksik ortam değişkenleri: " + ", ".join(missing))

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=60,
    )
    collection = os.environ["QDRANT_COLLECTION"]

    points: list[Any] = []
    offset = None

    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for point in points:
        payload = dict(point.payload or {})
        if not is_incident(payload):
            continue

        grouped[group_key(str(point.id), payload)].append(
            {
                "id": str(point.id),
                "timestamp": parse_timestamp(payload.get("timestamp")),
                "level": normalize_level(payload),
                "source": payload.get("source"),
                "sourceType": payload.get("sourceType"),
                "logger": payload.get("logger"),
                "category": payload.get("category"),
                "statusCode": status_code(payload),
                "httpMethod": payload.get("httpMethod"),
                "path": payload.get("path"),
                "requestId": payload.get("requestId"),
                "contentHash": payload.get("contentHash"),
                "content": str(payload.get("content") or ""),
            }
        )

    candidates: list[dict[str, Any]] = []

    for key, logs in grouped.items():
        logs.sort(key=lambda item: item.get("timestamp") or "")
        combined = "\n".join(item["content"] for item in logs)
        category = classify_category(combined)

        severity_weight = max(
            (
                4 if item["level"] in {"FATAL", "CRITICAL"} else
                3 if item["level"] == "ERROR" else
                2 if item["level"] in {"WARN", "WARNING"} else
                1
            )
            for item in logs
        )
        status_weight = max(
            (
                3 if (item["statusCode"] or 0) >= 500 else
                2 if (item["statusCode"] or 0) >= 400 else
                0
            )
            for item in logs
        )
        score = severity_weight * 10 + status_weight * 4 + min(len(logs), 12)

        candidates.append(
            {
                "candidate_id": "",
                "group_key": key,
                "category_hint": category,
                "difficulty": "unlabeled",
                "draft_question": draft_question(category),
                "annotation_status": "needs_review",
                "expected": {
                    "error_present": None,
                    "error_type": "",
                    "root_cause": "",
                    "root_cause_keywords": [],
                    "evidence_ids": [],
                    "solution_keywords": [],
                    "should_abstain": None,
                },
                "source_ids": [item["id"] for item in logs],
                "logs": logs[: args.max_lines_per_incident],
                "_selection_score": score,
            }
        )

    # Balance categories before filling remaining slots.
    candidates.sort(key=lambda item: item["_selection_score"], reverse=True)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_category[item["category_hint"]].append(item)

    selected: list[dict[str, Any]] = []
    category_order = [
        "database",
        "http_api",
        "authentication",
        "docker_container",
        "network_dns",
        "filesystem_permission",
        "resource_usage",
        "application_exception",
        "other",
    ]

    per_category = max(2, args.candidate_count // max(len(category_order), 1))

    for category in category_order:
        selected.extend(by_category.get(category, [])[:per_category])

    selected_ids = {id(item) for item in selected}
    for item in candidates:
        if len(selected) >= args.candidate_count:
            break
        if id(item) not in selected_ids:
            selected.append(item)
            selected_ids.add(id(item))

    selected = selected[: args.candidate_count]

    for index, item in enumerate(selected, start=1):
        item["candidate_id"] = f"CAND-{index:03d}"
        item.pop("_selection_score", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    category_counts = Counter(item["category_hint"] for item in selected)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection": collection,
        "total_points_scanned": len(points),
        "incident_groups_found": len(grouped),
        "candidates_exported": len(selected),
        "category_counts": dict(category_counts),
        "output": str(args.output),
    }

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
