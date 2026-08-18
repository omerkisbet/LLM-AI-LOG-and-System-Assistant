from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient, models

try:
    from .query_planner import (
        QueryIntent,
        QueryPlan,
    )
except ImportError:
    from query_planner import (
        QueryIntent,
        QueryPlan,
    )


DEFAULT_MAX_EVENTS = 2000
DEFAULT_BATCH_SIZE = 256
LATEST_EVENT_LIMIT = 20


def build_plan_filter(
    plan: QueryPlan,
) -> models.Filter | None:
    conditions: list[models.Condition] = []

    if plan.from_time_utc or plan.to_time_utc:
        conditions.append(
            models.FieldCondition(
                key="timestamp",
                range=models.DatetimeRange(
                    gte=plan.from_time_utc,
                    lte=plan.to_time_utc,
                ),
            )
        )

    if plan.source_types:
        conditions.append(
            models.FieldCondition(
                key="sourceType",
                match=models.MatchAny(
                    any=list(plan.source_types),
                ),
            )
        )

    if plan.actions:
        conditions.append(
            models.FieldCondition(
                key="action",
                match=models.MatchAny(
                    any=list(plan.actions),
                ),
            )
        )

    if plan.entity_types:
        conditions.append(
            models.FieldCondition(
                key="entityType",
                match=models.MatchAny(
                    any=list(plan.entity_types),
                ),
            )
        )

    if plan.levels:
        conditions.append(
            models.FieldCondition(
                key="level",
                match=models.MatchAny(
                    any=list(plan.levels),
                ),
            )
        )

    if not conditions:
        return None

    return models.Filter(
        must=conditions,
    )


def parse_timestamp(
    value: Any,
) -> datetime:
    if not value:
        return datetime.min.replace(
            tzinfo=timezone.utc
        )

    text = str(value).strip().replace(
        "Z",
        "+00:00",
    )

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min.replace(
            tzinfo=timezone.utc
        )

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed.astimezone(timezone.utc)


def fetch_plan_events(
    client: QdrantClient,
    collection_name: str,
    plan: QueryPlan,
    max_events: int = DEFAULT_MAX_EVENTS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, Any]]:
    if max_events < 1:
        raise ValueError(
            "max_events en az 1 olmalıdır."
        )

    query_filter = build_plan_filter(plan)

    events: list[dict[str, Any]] = []
    offset: Any = None

    while len(events) < max_events:
        remaining = max_events - len(events)
        current_limit = min(
            batch_size,
            remaining,
        )

        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            offset=offset,
            limit=current_limit,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = dict(point.payload or {})
            payload["_pointId"] = str(point.id)
            events.append(payload)

        if next_offset is None or not points:
            break

        offset = next_offset

    events.sort(
        key=lambda item: parse_timestamp(
            item.get("timestamp")
        ),
        reverse=True,
    )

    return events


def normalized_value(
    value: Any,
    default: str = "UNKNOWN",
) -> str:
    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    return text.upper()


def shorten(
    value: Any,
    maximum: int = 500,
) -> str:
    text = " ".join(
        str(value or "").split()
    )

    if len(text) <= maximum:
        return text

    return text[:maximum] + "..."


def detect_error_type(
    event: dict[str, Any],
) -> str:
    for field_name in (
        "exceptionType",
        "errorType",
        "exception",
    ):
        value = event.get(field_name)

        if value:
            return str(value).strip()

    content = str(
        event.get("content")
        or event.get("description")
        or event.get("message")
        or ""
    )

    match = re.search(
        r"\b[A-Z][A-Za-z0-9_]*"
        r"(?:Exception|Error)\b",
        content,
    )

    if match:
        return match.group(0)

    category = event.get("category")

    if category:
        return str(category).strip()

    status_code = event.get("statusCode")

    if status_code:
        return f"HTTP_{status_code}"

    return "UNKNOWN_ERROR"


def compact_event(
    event: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pointId": event.get("_pointId"),
        "timestamp": event.get("timestamp"),
        "sourceType": event.get("sourceType"),
    }

    optional_fields = (
        "level",
        "category",
        "action",
        "entityType",
        "entityId",
        "actor",
        "requestId",
        "path",
        "method",
        "httpMethod",
        "statusCode",
        "durationMs",
        "ipAddress",
        "userAgent",
        "clientIpHash",
    )

    for field_name in optional_fields:
        value = event.get(field_name)

        if value is not None:
            result[field_name] = value

    content = (
        event.get("content")
        or event.get("description")
        or event.get("message")
    )

    if content:
        result["content"] = shorten(content)

    return result


def counter_to_dict(
    counter: Counter[str],
) -> dict[str, int]:
    return {
        key: value
        for key, value in counter.most_common()
    }


def build_operational_summary(
    events: list[dict[str, Any]],
    plan: QueryPlan,
) -> dict[str, Any]:
    source_counter: Counter[str] = Counter()
    level_counter: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()
    entity_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    path_counter: Counter[str] = Counter()
    method_counter: Counter[str] = Counter()
    ip_address_counter: Counter[str] = Counter()
    client_ip_hash_counter: Counter[str] = Counter()
    error_counter: Counter[str] = Counter()

    duration_values: list[float] = []
    http_request_count = 0

    # APPLICATION_ANALYTICS_FIELDS_V2

    error_events: list[dict[str, Any]] = []
    warning_events: list[dict[str, Any]] = []

    for event in events:
        source_type = normalized_value(
            event.get("sourceType")
        )

        level = normalized_value(
            event.get("level")
        )

        action = normalized_value(
            event.get("action")
        )

        entity_type = normalized_value(
            event.get("entityType")
        )

        source_counter[source_type] += 1

        if level != "UNKNOWN":
            level_counter[level] += 1

        if action != "UNKNOWN":
            action_counter[action] += 1

        if entity_type != "UNKNOWN":
            entity_counter[entity_type] += 1

        status_code = event.get("statusCode")

        if status_code is not None:
            status_counter[str(status_code)] += 1

        path = event.get("path")

        if path:
            path_counter[str(path)] += 1
            http_request_count += 1

        http_method = (
            event.get("httpMethod")
            or event.get("method")
        )

        if http_method:
            method_counter[
                str(http_method).upper()
            ] += 1

        ip_address = event.get("ipAddress")

        if ip_address:
            ip_address_counter[
                str(ip_address)
            ] += 1

        client_ip_hash = event.get("clientIpHash")

        if client_ip_hash:
            client_ip_hash_counter[
                str(client_ip_hash)
            ] += 1

        duration_ms = event.get("durationMs")

        if duration_ms is not None:
            try:
                duration_values.append(
                    float(duration_ms)
                )
            except (TypeError, ValueError):
                pass

        if level == "ERROR":
            error_counter[
                detect_error_type(event)
            ] += 1

            error_events.append(
                compact_event(event)
            )

        if level in {
            "WARN",
            "WARNING",
        }:
            warning_events.append(
                compact_event(event)
            )

    top_errors = [
        {
            "type": error_type,
            "count": count,
        }
        for error_type, count
        in error_counter.most_common(10)
    ]

    top_paths = [
        {
            "path": path,
            "count": count,
        }
        for path, count
        in path_counter.most_common(10)
    ]

    top_ip_addresses = [
        {
            "ipAddress": ip_address,
            "count": count,
        }
        for ip_address, count
        in ip_address_counter.most_common(50)
    ]

    top_http_methods = [
        {
            "method": method,
            "count": count,
        }
        for method, count
        in method_counter.most_common(10)
    ]

    top_client_ip_hashes = [
        {
            "clientIpHash": client_hash,
            "count": count,
        }
        for client_hash, count
        in client_ip_hash_counter.most_common(50)
    ]

    average_duration_ms = (
        round(
            sum(duration_values)
            / len(duration_values),
            2,
        )
        if duration_values
        else None
    )

    maximum_duration_ms = (
        round(max(duration_values), 2)
        if duration_values
        else None
    )

    result = {
        "intent": plan.intent.value,
        "language": plan.detected_language,
        "timezone": plan.timezone,
        "period": {
            "fromUtc": (
                plan.from_time_utc
                .astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
                if plan.from_time_utc
                else None
            ),
            "toUtc": (
                plan.to_time_utc
                .astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
                if plan.to_time_utc
                else None
            ),
        },
        "totalEvents": len(events),
        "runtimeLogCount": source_counter.get(
            "RUNTIME_LOG",
            0,
        ),
        "auditLogCount": source_counter.get(
            "AUDIT_LOG",
            0,
        ),
        "errorCount": level_counter.get(
            "ERROR",
            0,
        ),
        "warningCount": (
            level_counter.get("WARN", 0)
            + level_counter.get("WARNING", 0)
        ),
        "createdCount": action_counter.get(
            "CREATE",
            0,
        ),
        "updatedCount": action_counter.get(
            "UPDATE",
            0,
        ),
        "deletedCount": action_counter.get(
            "DELETE",
            0,
        ),
        "sourceTypeCounts": counter_to_dict(
            source_counter
        ),
        "levelCounts": counter_to_dict(
            level_counter
        ),
        "actionCounts": counter_to_dict(
            action_counter
        ),
        "entityTypeCounts": counter_to_dict(
            entity_counter
        ),
        "statusCodeCounts": counter_to_dict(
            status_counter
        ),
        "topErrors": top_errors,
        "topPaths": top_paths,
        "topIpAddresses": top_ip_addresses,
        "topHttpMethods": top_http_methods,
        "topClientIpHashes": top_client_ip_hashes,
        "ipAddressCounts": counter_to_dict(
            ip_address_counter
        ),
        "httpMethodCounts": counter_to_dict(
            method_counter
        ),
        "httpRequestCount": http_request_count,
        "loginSuccessCount": action_counter.get(
            "LOGIN_SUCCESS",
            0,
        ),
        "loginFailureCount": action_counter.get(
            "LOGIN_FAILURE",
            0,
        ),
        "loginBlockedCount": action_counter.get(
            "LOGIN_BLOCKED",
            0,
        ),
        "averageDurationMs": average_duration_ms,
        "maximumDurationMs": maximum_duration_ms,
        "latestEvents": [
            compact_event(event)
            for event in events[
                :LATEST_EVENT_LIMIT
            ]
        ],
        "latestErrors": error_events[
            :LATEST_EVENT_LIMIT
        ],
        "latestWarnings": warning_events[
            :LATEST_EVENT_LIMIT
        ],
        "truncated": (
            len(events) >= DEFAULT_MAX_EVENTS
        ),
    }

    if plan.intent == QueryIntent.ERROR_COUNT:
        result["matchingErrorCount"] = len(events)

    return result


def execute_operational_plan(
    client: QdrantClient,
    collection_name: str,
    plan: QueryPlan,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> dict[str, Any]:
    events = fetch_plan_events(
        client=client,
        collection_name=collection_name,
        plan=plan,
        max_events=max_events,
    )

    return build_operational_summary(
        events=events,
        plan=plan,
    )
