from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class IncidentStatus(str, Enum):
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    IGNORED = "IGNORED"


class IncidentSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentRuleType(str, Enum):
    REPEATED_EXCEPTION = "REPEATED_EXCEPTION"


SEVERITY_ORDER = {
    IncidentSeverity.INFO.value: 0,
    IncidentSeverity.WARNING.value: 1,
    IncidentSeverity.HIGH.value: 2,
    IncidentSeverity.CRITICAL.value: 3,
}


def to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def iso_utc(value: datetime) -> str:
    return to_utc(value).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class OperationalEvent:
    record_id: str
    timestamp: datetime
    source_type: str
    level: str | None
    content: str
    request_id: str | None = None
    endpoint: str | None = None
    exception_name: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def event_key(self) -> str:
        material = {
            "recordId": self.record_id,
            "timestamp": iso_utc(self.timestamp),
            "requestId": self.request_id,
            "content": self.content,
        }
        encoded = json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class IncidentCandidate:
    rule_type: IncidentRuleType
    fingerprint: str
    title: str
    severity: IncidentSeverity
    source_type: str
    first_seen_at: datetime
    last_seen_at: datetime
    window_start_at: datetime
    window_end_at: datetime
    events: tuple[OperationalEvent, ...]
    exception_name: str | None = None
    endpoint: str | None = None
    deterministic_summary: str | None = None

    @property
    def dedupe_key(self) -> str:
        raw = f"{self.rule_type.value}|{self.fingerprint}".casefold()
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
