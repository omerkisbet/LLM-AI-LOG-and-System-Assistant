#!/usr/bin/env python3
"""Export deduplicated, representative Log Agent benchmark candidates from Qdrant.

v2 changes:
- strict category classification (no generic "host" or "db" substring matches),
- request/event grouping followed by cross-event signature deduplication,
- normalization of timestamps, UUIDs, request IDs, durations and stack line numbers,
- repeated incidents represented once with occurrence_count,
- balanced candidate selection without inventing missing categories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient


INCIDENT_TERMS = re.compile(
    r"\b(?:exception|error|failed|failure|timeout|timed out|denied|"
    r"unauthorized|forbidden|unavailable|refused|not found|"
    r"connection reset|broken pipe|out of memory|authentication|"
    r"permission|traceback|stacktrace)\b",
    re.IGNORECASE,
)

ISO_TIMESTAMP = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+Z?\b",
    re.IGNORECASE,
)
UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
REQUEST_ID_PATTERN = re.compile(
    r"(request\s*id\s*[:=]\s*)[^\s,.;]+",
    re.IGNORECASE,
)
DURATION_PATTERN = re.compile(
    r"\b(?:in\s+)?\d+(?:\.\d+)?\s*ms\b",
    re.IGNORECASE,
)
STACK_LINE_PATTERN = re.compile(r"(\.java|\.py|\.js|\.ts):\d+")
HEX_PATTERN = re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE)
PORT_PATTERN = re.compile(r"(?<=:)\d{2,5}\b")
WHITESPACE = re.compile(r"\s+")


CATEGORY_RULES: list[tuple[str, list[re.Pattern[str]]]] = [
    (
        "authentication",
        [
            re.compile(r"AuthorizationDeniedException", re.IGNORECASE),
            re.compile(r"AuthenticationException", re.IGNORECASE),
            re.compile(r"\bAccess Denied\b", re.IGNORECASE),
            re.compile(r"\bunauthorized\b|\bforbidden\b", re.IGNORECASE),
            re.compile(r"\bHTTP\b.*\b(?:401|403)\b", re.IGNORECASE),
            re.compile(r"\bstatus(?: code)?\s*[:=]?\s*(?:401|403)\b", re.IGNORECASE),
            re.compile(r"\binvalid (?:token|credential|password)\b", re.IGNORECASE),
        ],
    ),
    (
        "database",
        [
            re.compile(r"\bMongo(?:DB|Security|Timeout|Socket)?Exception\b", re.IGNORECASE),
            re.compile(r"\bMongoDB\b|\bMONGODB_URI\b", re.IGNORECASE),
            re.compile(r"\bJDBC\b|\bSQL(?:Exception|State)?\b", re.IGNORECASE),
            re.compile(r"\bPostgreSQL\b|\bMySQL\b|\bMariaDB\b", re.IGNORECASE),
            re.compile(r"\bRedis(?:Connection)?Exception\b", re.IGNORECASE),
            re.compile(r"\bQdrant\b", re.IGNORECASE),
            re.compile(r"\bconnection pool\b|\bdatabase connection\b", re.IGNORECASE),
        ],
    ),
    (
        "filesystem_permission",
        [
            re.compile(r"\bAccessDeniedException\b", re.IGNORECASE),
            re.compile(r"\bPermissionError\b|\bpermission denied\b", re.IGNORECASE),
            re.compile(r"\bread-only file system\b", re.IGNORECASE),
            re.compile(r"\bno space left on device\b|\bdisk full\b", re.IGNORECASE),
            re.compile(r"\bFileNotFoundException\b|\bNoSuchFileException\b", re.IGNORECASE),
        ],
    ),
    (
        "static_resource",
        [
            re.compile(r"\bNoResourceFoundException\b", re.IGNORECASE),
            re.compile(r"\bNo static resource\b", re.IGNORECASE),
            re.compile(r"/favicon\.ico\b", re.IGNORECASE),
        ],
    ),
    (
        "network_dns",
        [
            re.compile(r"\bUnknownHostException\b", re.IGNORECASE),
            re.compile(r"\bConnectException\b", re.IGNORECASE),
            re.compile(r"\bSocketTimeoutException\b", re.IGNORECASE),
            re.compile(r"\bConnectionRefusedError\b", re.IGNORECASE),
            re.compile(r"\bconnection refused\b|\bconnection reset\b", re.IGNORECASE),
            re.compile(r"\bname or service not known\b|\btemporary failure in name resolution\b", re.IGNORECASE),
            re.compile(r"\bDNS\b|\bNXDOMAIN\b", re.IGNORECASE),
            re.compile(r"\bSSLHandshakeException\b|\bcertificate verify failed\b", re.IGNORECASE),
            re.compile(r"\bno route to host\b|\bnetwork is unreachable\b", re.IGNORECASE),
        ],
    ),
    (
        "docker_container",
        [
            re.compile(r"\bDocker\b|\bcontainer\b|\bdocker compose\b", re.IGNORECASE),
            re.compile(r"\bport is already allocated\b", re.IGNORECASE),
            re.compile(r"\bcontainer exited\b|\bexit code\b", re.IGNORECASE),
            re.compile(r"\bhealthcheck\b|\brestart policy\b", re.IGNORECASE),
        ],
    ),
    (
        "resource_usage",
        [
            re.compile(r"\bOutOfMemoryError\b|\bout of memory\b|\bOOMKilled\b", re.IGNORECASE),
            re.compile(r"\bCUDA out of memory\b|\bresource exhausted\b", re.IGNORECASE),
            re.compile(r"\bCPU throttling\b|\bmemory pressure\b", re.IGNORECASE),
        ],
    ),
    (
        "ai_upstream",
        [
            re.compile(r"/api/admin/log-assistant/chat", re.IGNORECASE),
            re.compile(r"/api/log-agent/chat", re.IGNORECASE),
            re.compile(r"\bAI service\b|\bmodel server\b|\bLLM\b", re.IGNORECASE),
            re.compile(r"\bupstream\b.*\b(?:502|503|504)\b", re.IGNORECASE),
        ],
    ),
    (
        "http_api",
        [
            re.compile(r"\bHTTP\b.*\b(?:400|404|405|409|422|429|500|502|503|504)\b", re.IGNORECASE),
            re.compile(r"\bstatus(?: code)?\s*[:=]?\s*(?:400|404|405|409|422|429|500|502|503|504)\b", re.IGNORECASE),
            re.compile(r"\bMethodNotAllowed\b|\bBadRequest\b", re.IGNORECASE),
        ],
    ),
    (
        "application_exception",
        [
            re.compile(r"\b[A-Za-z0-9_.]+Exception\b"),
            re.compile(r"\bTraceback\b|\bRuntimeError\b|\bValueError\b|\bTypeError\b", re.IGNORECASE),
            re.compile(r"\bNullPointerException\b|\bIllegalArgumentException\b", re.IGNORECASE),
        ],
    ),
]


def normalize_level(payload: dict[str, Any]) -> str:
    return str(payload.get("level") or "").strip().upper()


def parse_status_code(payload: dict[str, Any]) -> int | None:
    value = payload.get("statusCode")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def is_incident(payload: dict[str, Any]) -> bool:
    content = str(payload.get("content") or "")
    level = normalize_level(payload)
    code = parse_status_code(payload)

    return (
        level in {"ERROR", "WARN", "WARNING", "FATAL", "CRITICAL"}
        or (code is not None and code >= 400)
        or bool(INCIDENT_TERMS.search(content))
    )


def event_key(point_id: str, payload: dict[str, Any]) -> str:
    for field in ("requestId", "traceId", "documentId"):
        value = payload.get(field)
        if value:
            return f"{field}:{value}"

    # Logs without correlation IDs are grouped conservatively by source/logger/minute.
    source = payload.get("source") or "unknown"
    logger = payload.get("logger") or "unknown"
    timestamp = str(payload.get("timestamp") or "")
    minute = timestamp[:16] if len(timestamp) >= 16 else timestamp
    return f"fallback:{source}:{logger}:{minute}:{point_id}"


def canonicalize(text: str) -> str:
    value = text.lower()
    value = ISO_TIMESTAMP.sub("<timestamp>", value)
    value = UUID_PATTERN.sub("<uuid>", value)
    value = REQUEST_ID_PATTERN.sub(r"\1<request-id>", value)
    value = DURATION_PATTERN.sub("<duration-ms>", value)
    value = STACK_LINE_PATTERN.sub(r"\1:<line>", value)
    value = HEX_PATTERN.sub("<hex>", value)
    value = PORT_PATTERN.sub("<port>", value)
    value = re.sub(r"\bsequence\s*[:=]\s*\d+\b", "sequence=<num>", value)
    value = re.sub(r"\bactor\s*[:=]\s*[^,.;]+", "actor=<actor>", value)
    return WHITESPACE.sub(" ", value).strip()


def classify(text: str, logs: list[dict[str, Any]]) -> str:
    # Strongest structured signals first.
    paths = " ".join(str(item.get("path") or "") for item in logs)
    categories = " ".join(str(item.get("category") or "") for item in logs)
    combined = f"{text}\nPATHS:{paths}\nCATEGORIES:{categories}"

    for category, patterns in CATEGORY_RULES:
        if any(pattern.search(combined) for pattern in patterns):
            return category

    return "other"


def signature_for(category: str, logs: list[dict[str, Any]], combined: str) -> str:
    structured = {
        "category": category,
        "source": sorted({str(item.get("source") or "") for item in logs}),
        "logger": sorted({str(item.get("logger") or "") for item in logs}),
        "status": sorted({
            str(item.get("statusCode"))
            for item in logs
            if item.get("statusCode") is not None
        }),
        "method": sorted({str(item.get("httpMethod") or "") for item in logs}),
        "path": sorted({str(item.get("path") or "") for item in logs}),
        "content": canonicalize(combined),
    }
    encoded = json.dumps(structured, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def severity_score(logs: list[dict[str, Any]]) -> int:
    score = 0
    for item in logs:
        level = str(item.get("level") or "")
        status = item.get("statusCode") or 0
        score = max(
            score,
            50 if level in {"FATAL", "CRITICAL"} else
            40 if level == "ERROR" else
            25 if level in {"WARN", "WARNING"} else
            10,
        )
        if status >= 500:
            score += 12
        elif status >= 400:
            score += 6
    return score + min(len(logs), 10)


def draft_question(category: str) -> str:
    return {
        "authentication": "Bu olayda kimlik doğrulama veya yetkilendirme neden başarısız olmuştur?",
        "database": "Bu olayda veritabanı işleminin kök nedeni nedir?",
        "filesystem_permission": "Bu dosya veya izin hatasının kök nedeni ve çözümü nedir?",
        "static_resource": "Bu statik kaynak hatasının etkisi ve kök nedeni nedir?",
        "network_dns": "Bu ağ veya DNS hatasının kök nedeni nedir?",
        "docker_container": "Bu container olayının kök nedeni ve önerilen müdahale nedir?",
        "resource_usage": "Bu kaynak tükenmesi olayının kök nedeni nedir?",
        "ai_upstream": "Log Agent isteğinin upstream katmanda başarısız olmasının olası nedeni nedir?",
        "http_api": "Bu API isteği hangi nedenle başarısız olmuştur?",
        "application_exception": "Bu exception zincirindeki asıl kök neden nedir?",
        "other": "Bu log grubunda hata var mı; varsa kök nedeni nedir?",
    }.get(category, "Bu log grubunda hata var mı; varsa kök nedeni nedir?")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=Path("rag/.env"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rag/data/log_agent_benchmark/candidates_v2.jsonl"),
    )
    parser.add_argument("--candidate-count", type=int, default=80)
    parser.add_argument("--max-lines-per-incident", type=int, default=12)
    parser.add_argument("--max-per-category", type=int, default=15)
    args = parser.parse_args()

    load_dotenv(args.env)

    missing = [
        name for name in ("QDRANT_URL", "QDRANT_COLLECTION")
        if not os.getenv(name)
    ]
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

    events: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for point in points:
        payload = dict(point.payload or {})
        if not is_incident(payload):
            continue

        events[event_key(str(point.id), payload)].append(
            {
                "id": str(point.id),
                "timestamp": str(payload.get("timestamp") or ""),
                "level": normalize_level(payload),
                "source": payload.get("source"),
                "sourceType": payload.get("sourceType"),
                "logger": payload.get("logger"),
                "category": payload.get("category"),
                "statusCode": parse_status_code(payload),
                "httpMethod": payload.get("httpMethod"),
                "path": payload.get("path"),
                "requestId": payload.get("requestId"),
                "contentHash": payload.get("contentHash"),
                "content": str(payload.get("content") or ""),
            }
        )

    deduplicated: dict[str, dict[str, Any]] = {}

    for key, logs in events.items():
        logs.sort(key=lambda item: item.get("timestamp") or "")
        combined = "\n".join(item["content"] for item in logs)
        category = classify(combined, logs)
        signature = signature_for(category, logs, combined)

        candidate = {
            "candidate_id": "",
            "signature": signature,
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
            "occurrence_count": 1,
            "event_keys": [key],
            "source_ids": [item["id"] for item in logs],
            "logs": logs[: args.max_lines_per_incident],
            "_score": severity_score(logs),
        }

        existing = deduplicated.get(signature)
        if existing is None:
            deduplicated[signature] = candidate
        else:
            existing["occurrence_count"] += 1
            existing["event_keys"].append(key)
            existing["_score"] = max(existing["_score"], candidate["_score"])

            # Keep the richest representative.
            if len(candidate["logs"]) > len(existing["logs"]):
                existing["source_ids"] = candidate["source_ids"]
                existing["logs"] = candidate["logs"]

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in deduplicated.values():
        by_category[item["category_hint"]].append(item)

    for items in by_category.values():
        items.sort(
            key=lambda item: (
                item["_score"],
                item["occurrence_count"],
                len(item["logs"]),
            ),
            reverse=True,
        )

    selected: list[dict[str, Any]] = []
    category_order = sorted(
        by_category,
        key=lambda name: len(by_category[name]),
        reverse=True,
    )

    # Round-robin prevents one dominant category from filling the output.
    category_positions = {name: 0 for name in category_order}

    while len(selected) < args.candidate_count:
        added = False

        for category in category_order:
            position = category_positions[category]
            items = by_category[category]

            if position >= len(items):
                continue

            category_selected = sum(
                1 for item in selected
                if item["category_hint"] == category
            )
            if category_selected >= args.max_per_category:
                continue

            selected.append(items[position])
            category_positions[category] += 1
            added = True

            if len(selected) >= args.candidate_count:
                break

        if not added:
            break

    for index, item in enumerate(selected, start=1):
        item["candidate_id"] = f"CAND-{index:03d}"
        item.pop("_score", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    raw_category_counts = Counter(
        item["category_hint"] for item in deduplicated.values()
    )
    selected_category_counts = Counter(
        item["category_hint"] for item in selected
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection": collection,
        "total_points_scanned": len(points),
        "raw_incident_events": len(events),
        "unique_incident_signatures": len(deduplicated),
        "duplicates_collapsed": len(events) - len(deduplicated),
        "candidates_exported": len(selected),
        "raw_category_counts": dict(raw_category_counts),
        "selected_category_counts": dict(selected_category_counts),
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
