from __future__ import annotations

import json
from pathlib import Path

from telemetry import TelemetryStore


TEST_DATABASE = Path(
    "/tmp/iotlab-log-agent-telemetry-test.sqlite3"
)

TEST_EXPORT = Path(
    "/tmp/iotlab-log-agent-training-candidates.jsonl"
)


def remove_sqlite_files(
    path: Path,
) -> None:
    for suffix in (
        "",
        "-wal",
        "-shm",
    ):
        candidate = Path(
            str(path) + suffix
        )

        if candidate.exists():
            candidate.unlink()


def main() -> None:
    remove_sqlite_files(TEST_DATABASE)

    if TEST_EXPORT.exists():
        TEST_EXPORT.unlink()

    store = TelemetryStore(
        TEST_DATABASE
    )

    print("HEALTH")
    print(
        json.dumps(
            store.health(),
            ensure_ascii=False,
            indent=2,
        )
    )

    successful_query_id = (
        store.record_query(
            request_id="test-request-up",
            actor="telemetry-test",
            question="Bugün neler oldu?",
            requested_language="tr",
            detected_language="tr",
            response_language="tr",
            intent="DAILY_SUMMARY",
            retrieval_mode="AGGREGATE",
            time_scope="TODAY",
            grounded=True,
            evidence_count=12,
            filtered_total=68,
            tools_used=[
                "bilingual_query_planner",
                "qdrant_scroll_retrieval",
                "deterministic_operational_summary",
            ],
            latency_ms=1250,
            answer=(
                "Bugün toplam 68 operasyon "
                "kaydı bulundu."
            ),
        )
    )

    failed_query_id = (
        store.record_query(
            request_id="test-request-down",
            actor="telemetry-test",
            question=(
                "QuantumPenguinXYZ hatası "
                "oluştu mu?"
            ),
            requested_language="tr",
            detected_language="tr",
            response_language="tr",
            intent=(
                "EXACT_IDENTIFIER_SEARCH"
            ),
            retrieval_mode="HYBRID",
            grounded=False,
            evidence_count=0,
            filtered_total=0,
            tools_used=[
                "bilingual_query_planner",
                "exact_term_grounding_gate",
            ],
            latency_ms=80,
            answer=(
                "Kesin terimle eşleşen "
                "bir log bulunamadı."
            ),
        )
    )

    store.record_feedback(
        query_id=successful_query_id,
        request_id="test-request-up",
        rating="UP",
        reason="CORRECT",
        actor="admin",
    )

    store.record_feedback(
        query_id=failed_query_id,
        request_id="test-request-down",
        rating="DOWN",
        reason="INCOMPLETE_ANSWER",
        comment=(
            "Cevapta aranan teknik terim "
            "açıkça belirtilmeliydi."
        ),
        corrected_answer=(
            "QuantumPenguinXYZ adına sahip "
            "bir hata kaydı bulunamadı."
        ),
        actor="admin",
    )

    recent = store.recent_queries(
        limit=10
    )

    statistics = (
        store.feedback_statistics()
    )

    exported_count = (
        store.export_training_candidates(
            TEST_EXPORT
        )
    )

    assert len(recent) == 2
    assert statistics["totalQueries"] == 2
    assert statistics["totalFeedback"] == 2
    assert statistics["ratingCounts"]["UP"] == 1
    assert statistics["ratingCounts"]["DOWN"] == 1
    assert exported_count == 1
    assert TEST_EXPORT.exists()

    print()
    print("RECENT QUERIES")
    print(
        json.dumps(
            recent,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("FEEDBACK STATISTICS")
    print(
        json.dumps(
            statistics,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("TRAINING CANDIDATES")
    print(
        TEST_EXPORT.read_text(
            encoding="utf-8"
        )
    )

    print(
        "TELEMETRY TESTLERİ BAŞARILI"
    )


if __name__ == "__main__":
    main()
