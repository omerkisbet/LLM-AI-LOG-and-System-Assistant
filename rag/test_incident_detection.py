from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from incident_detection.models import OperationalEvent
from incident_detection.rules import repeated_exception_candidates
from incident_detection.store import IncidentStore


def make_event(index: int) -> OperationalEvent:
    base = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)
    return OperationalEvent(
        record_id=f"record-{index}",
        timestamp=base + timedelta(seconds=index),
        source_type="RUNTIME_LOG",
        level="ERROR",
        content="NoResourceFoundException: No static resource /favicon.ico",
        request_id=f"request-{index}",
        endpoint="/favicon.ico",
        exception_name="NoResourceFoundException",
    )


def main() -> None:
    window_end = datetime(2026, 7, 24, 12, 5, tzinfo=timezone.utc)
    window_start = window_end - timedelta(minutes=5)
    events = [make_event(index) for index in range(5)]

    candidates = repeated_exception_candidates(
        events,
        threshold=5,
        window_start_at=window_start,
        window_end_at=window_end,
    )
    assert len(candidates) == 1

    with tempfile.TemporaryDirectory() as directory:
        store = IncidentStore(Path(directory) / "incident-test.sqlite3")

        first = store.upsert_candidate(candidates[0])
        assert first["created"] is True
        assert first["newEventCount"] == 5
        assert first["occurrence_count"] == 5

        duplicate = store.upsert_candidate(candidates[0])
        assert duplicate["created"] is False
        assert duplicate["newEventCount"] == 0
        assert duplicate["occurrence_count"] == 5

        extended_events = events + [make_event(99)]
        extended_candidates = repeated_exception_candidates(
            extended_events,
            threshold=5,
            window_start_at=window_start,
            window_end_at=window_end,
        )
        updated = store.upsert_candidate(extended_candidates[0])
        assert updated["created"] is False
        assert updated["newEventCount"] == 1
        assert updated["occurrence_count"] == 6

        incidents = store.list_incidents(status="OPEN")
        assert len(incidents) == 1

    print("INCIDENT DETECTION TESTLERİ BAŞARILI")


if __name__ == "__main__":
    main()
