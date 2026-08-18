from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from qdrant_client import QdrantClient, models

from .models import OperationalEvent, iso_utc
from .rules import (
    extract_endpoint,
    extract_exception_name,
    repeated_exception_candidates,
)
from .store import IncidentStore


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        text = str(value or "").strip()
        if not text:
            return datetime.now(timezone.utc)
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))

    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)

    return result.astimezone(timezone.utc)


class IncidentDetector:
    def __init__(
        self,
        *,
        qdrant: QdrantClient,
        collection_name: str,
        store: IncidentStore,
        window_minutes: int = 5,
        repeated_exception_threshold: int = 5,
        max_events: int = 5000,
    ) -> None:
        self.qdrant = qdrant
        self.collection_name = collection_name
        self.store = store
        self.window_minutes = min(max(int(window_minutes), 1), 1440)
        self.repeated_exception_threshold = max(
            int(repeated_exception_threshold),
            2,
        )
        self.max_events = min(max(int(max_events), 100), 50000)

    def _fetch_error_events(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> list[OperationalEvent]:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="level",
                    match=models.MatchValue(value="ERROR"),
                ),
                models.FieldCondition(
                    key="timestamp",
                    range=models.DatetimeRange(
                        gte=window_start,
                        lte=window_end,
                    ),
                ),
            ]
        )

        events: list[OperationalEvent] = []
        offset = None

        while len(events) < self.max_events:
            points, next_offset = self.qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=min(256, self.max_events - len(events)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = dict(point.payload or {})
                content = str(
                    payload.get("content")
                    or payload.get("description")
                    or payload.get("message")
                    or ""
                )

                events.append(
                    OperationalEvent(
                        record_id=str(
                            payload.get("mongoId")
                            or payload.get("documentId")
                            or point.id
                        ),
                        timestamp=parse_timestamp(payload.get("timestamp")),
                        source_type=str(
                            payload.get("sourceType") or "RUNTIME_LOG"
                        ),
                        level=str(payload.get("level") or "ERROR"),
                        content=content[:12000],
                        request_id=(
                            str(payload.get("requestId"))
                            if payload.get("requestId") is not None
                            else None
                        ),
                        endpoint=extract_endpoint(payload, content),
                        exception_name=extract_exception_name(payload, content),
                        payload=payload,
                    )
                )

            if next_offset is None or not points:
                break
            offset = next_offset

        return events

    def run_once(
        self,
        *,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        started = datetime.now(timezone.utc)
        window_end = (
            now.astimezone(timezone.utc)
            if now is not None
            else started
        )
        window_start = window_end - timedelta(
            minutes=self.window_minutes
        )

        fetched_events = 0
        candidates = []
        new_incidents = 0
        updated_incidents = 0
        new_events = 0
        incident_results: list[dict[str, Any]] = []
        error_type = None
        error_message = None

        try:
            events = self._fetch_error_events(
                window_start=window_start,
                window_end=window_end,
            )
            fetched_events = len(events)

            candidates = repeated_exception_candidates(
                events,
                threshold=self.repeated_exception_threshold,
                window_start_at=window_start,
                window_end_at=window_end,
            )

            for candidate in candidates:
                if dry_run:
                    incident_results.append(
                        {
                            "ruleType": candidate.rule_type.value,
                            "title": candidate.title,
                            "severity": candidate.severity.value,
                            "eventCount": len(candidate.events),
                            "dedupeKey": candidate.dedupe_key,
                        }
                    )
                    continue

                result = self.store.upsert_candidate(candidate)
                incident_results.append(result)
                new_events += int(result["newEventCount"])

                if result["created"]:
                    new_incidents += 1
                elif int(result["newEventCount"]) > 0:
                    updated_incidents += 1

        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc)
            raise

        finally:
            finished = datetime.now(timezone.utc)
            if not dry_run:
                self.store.record_run(
                    started_at_utc=iso_utc(started),
                    finished_at_utc=iso_utc(finished),
                    window_start_at_utc=iso_utc(window_start),
                    window_end_at_utc=iso_utc(window_end),
                    fetched_event_count=fetched_events,
                    candidate_count=len(candidates),
                    new_incident_count=new_incidents,
                    updated_incident_count=updated_incidents,
                    new_event_count=new_events,
                    error_type=error_type,
                    error_message=error_message,
                )

        return {
            "status": "OK",
            "dryRun": dry_run,
            "windowMinutes": self.window_minutes,
            "windowStartUtc": iso_utc(window_start),
            "windowEndUtc": iso_utc(window_end),
            "fetchedErrorEvents": fetched_events,
            "candidateCount": len(candidates),
            "newIncidentCount": new_incidents,
            "updatedIncidentCount": updated_incidents,
            "newEventCount": new_events,
            "incidents": incident_results,
        }
