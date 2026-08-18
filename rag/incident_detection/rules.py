from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable

from .models import (
    IncidentCandidate,
    IncidentRuleType,
    IncidentSeverity,
    OperationalEvent,
)


EXCEPTION_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9_$]*(?:Exception|Error))\b"
)

ENDPOINT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(/[A-Za-z0-9._~!$&'()*+,;=:@%/?#-]{1,300})"
)


def normalize_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def extract_exception_name(
    payload: dict[str, Any],
    content: str,
) -> str | None:
    for key in (
        "exceptionName",
        "exceptionType",
        "exception",
        "errorType",
        "throwable",
    ):
        candidate = normalize_optional(payload.get(key))
        if candidate:
            match = EXCEPTION_PATTERN.search(candidate)
            return match.group(1) if match else candidate[:200]

    match = EXCEPTION_PATTERN.search(content)
    return match.group(1) if match else None


def extract_endpoint(
    payload: dict[str, Any],
    content: str,
) -> str | None:
    for key in (
        "endpoint",
        "path",
        "requestPath",
        "requestUri",
        "uri",
        "url",
    ):
        candidate = normalize_optional(payload.get(key))
        if candidate and candidate.startswith("/"):
            return candidate[:400]

    match = ENDPOINT_PATTERN.search(content)
    return match.group(1)[:400] if match else None


def severity_for_count(
    count: int,
    threshold: int,
) -> IncidentSeverity:
    if count >= max(threshold * 5, 25):
        return IncidentSeverity.CRITICAL
    if count >= max(threshold * 2, 10):
        return IncidentSeverity.HIGH
    return IncidentSeverity.WARNING


def repeated_exception_candidates(
    events: Iterable[OperationalEvent],
    *,
    threshold: int,
    window_start_at: datetime,
    window_end_at: datetime,
) -> list[IncidentCandidate]:
    groups: dict[tuple[str, str | None], list[OperationalEvent]] = defaultdict(list)

    for event in events:
        if not event.exception_name:
            continue
        groups[(event.exception_name, event.endpoint)].append(event)

    candidates: list[IncidentCandidate] = []

    for (exception_name, endpoint), grouped_events in groups.items():
        if len(grouped_events) < threshold:
            continue

        ordered = sorted(grouped_events, key=lambda item: item.timestamp)
        endpoint_text = f" — {endpoint}" if endpoint else ""
        fingerprint = f"{exception_name}|{endpoint or '*'}"

        candidates.append(
            IncidentCandidate(
                rule_type=IncidentRuleType.REPEATED_EXCEPTION,
                fingerprint=fingerprint,
                title=f"Tekrarlayan exception: {exception_name}{endpoint_text}",
                severity=severity_for_count(len(ordered), threshold),
                source_type="RUNTIME_LOG",
                first_seen_at=ordered[0].timestamp,
                last_seen_at=ordered[-1].timestamp,
                window_start_at=window_start_at,
                window_end_at=window_end_at,
                events=tuple(ordered),
                exception_name=exception_name,
                endpoint=endpoint,
                deterministic_summary=(
                    f"{exception_name} aynı gözlem aralığında "
                    f"{len(ordered)} farklı log kaydında görüldü."
                ),
            )
        )

    return candidates
