from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from query_planner import (
    QueryIntent,
    RetrievalMode,
    TimeScope,
    plan_query,
)


FIXED_NOW = datetime(
    2026,
    7,
    24,
    14,
    0,
    tzinfo=ZoneInfo("Europe/Istanbul"),
)


TEST_CASES = [
    {
        "question": "Bugün neler oldu?",
        "intent": QueryIntent.DAILY_SUMMARY,
        "mode": RetrievalMode.AGGREGATE,
        "language": "tr",
        "scope": TimeScope.TODAY,
    },
    {
        "question": "Dün hangi kayıtlar silindi?",
        "intent": QueryIntent.AUDIT_DELETE_LIST,
        "mode": RetrievalMode.FILTERED_LIST,
        "language": "tr",
        "scope": TimeScope.YESTERDAY,
    },
    {
        "question": "Son 2 saatte kaç hata oluştu?",
        "intent": QueryIntent.ERROR_COUNT,
        "mode": RetrievalMode.AGGREGATE,
        "language": "tr",
        "scope": TimeScope.RELATIVE,
    },
    {
        "question": "Saat 14.30 civarında ne oldu?",
        "intent": QueryIntent.TIME_WINDOW_SEARCH,
        "mode": RetrievalMode.FILTERED_LIST,
        "language": "tr",
        "scope": TimeScope.CLOCK,
    },
    {
        "question": (
            "NoResourceFoundException ve "
            "/favicon.ico hatasını açıkla"
        ),
        "intent": QueryIntent.EXACT_IDENTIFIER_SEARCH,
        "mode": RetrievalMode.HYBRID,
        "language": "tr",
        "scope": None,
    },
    {
        "question": (
            "Bugün hangi haberler güncellendi?"
        ),
        "intent": QueryIntent.AUDIT_UPDATE_LIST,
        "mode": RetrievalMode.FILTERED_LIST,
        "language": "tr",
        "scope": TimeScope.TODAY,
    },
    {
        "question": "What happened today?",
        "intent": QueryIntent.DAILY_SUMMARY,
        "mode": RetrievalMode.AGGREGATE,
        "language": "en",
        "scope": TimeScope.TODAY,
    },
    {
        "question": "What happened yesterday?",
        "intent": QueryIntent.DAILY_SUMMARY,
        "mode": RetrievalMode.AGGREGATE,
        "language": "en",
        "scope": TimeScope.YESTERDAY,
    },
    {
        "question": (
            "How many errors occurred "
            "in the last two hours?"
        ),
        "intent": QueryIntent.ERROR_COUNT,
        "mode": RetrievalMode.AGGREGATE,
        "language": "en",
        "scope": TimeScope.RELATIVE,
    },
    {
        "question": (
            "Which records were deleted yesterday?"
        ),
        "intent": QueryIntent.AUDIT_DELETE_LIST,
        "mode": RetrievalMode.FILTERED_LIST,
        "language": "en",
        "scope": TimeScope.YESTERDAY,
    },
    {
        "question": "What happened around 14:30?",
        "intent": QueryIntent.TIME_WINDOW_SEARCH,
        "mode": RetrievalMode.FILTERED_LIST,
        "language": "en",
        "scope": TimeScope.CLOCK,
    },
    {
        "question": (
            "Explain NoResourceFoundException "
            "related to /favicon.ico."
        ),
        "intent": QueryIntent.EXACT_IDENTIFIER_SEARCH,
        "mode": RetrievalMode.HYBRID,
        "language": "en",
        "scope": None,
    },
    {
        "question": (
            "Which news items were updated today?"
        ),
        "intent": QueryIntent.AUDIT_UPDATE_LIST,
        "mode": RetrievalMode.FILTERED_LIST,
        "language": "en",
        "scope": TimeScope.TODAY,
    },
]


def main() -> None:
    for test_case in TEST_CASES:
        question = test_case["question"]

        plan = plan_query(
            question=question,
            timezone_name="Europe/Istanbul",
            now=FIXED_NOW,
        )

        assert plan.intent == test_case["intent"], (
            f"Intent hatası: {question}\n"
            f"Beklenen: {test_case['intent'].value}\n"
            f"Gelen: {plan.intent.value}"
        )

        assert (
            plan.retrieval_mode
            == test_case["mode"]
        ), (
            f"Retrieval mode hatası: {question}"
        )

        assert (
            plan.detected_language
            == test_case["language"]
        ), (
            f"Dil hatası: {question}\n"
            f"Beklenen: {test_case['language']}\n"
            f"Gelen: {plan.detected_language}"
        )

        assert (
            plan.time_scope
            == test_case["scope"]
        ), (
            f"Time scope hatası: {question}\n"
            f"Beklenen: {test_case['scope']}\n"
            f"Gelen: {plan.time_scope}"
        )

        print("=" * 78)
        print(f"SORU: {question}")

        print(
            json.dumps(
                plan.to_dict(),
                ensure_ascii=False,
                indent=2,
            )
        )

    print()
    print(
        "TÜM TÜRKÇE VE İNGİLİZCE "
        "QUERY PLANNER TESTLERİ BAŞARILI"
    )


if __name__ == "__main__":
    main()
