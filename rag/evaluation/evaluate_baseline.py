from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


PROJECT_DIR = Path.home() / "huggingface-model-server"
RAG_DIR = PROJECT_DIR / "rag"
EVALUATION_DIR = RAG_DIR / "evaluation"

DATASET_FILE = EVALUATION_DIR / "eval_queries.jsonl"
JSON_RESULT_FILE = EVALUATION_DIR / "baseline_results.json"
CSV_RESULT_FILE = EVALUATION_DIR / "baseline_results.csv"

load_dotenv(RAG_DIR / ".env")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Ortam değişkeni bulunamadı: {name}"
        )

    return value


AGENT_URL = os.getenv(
    "LOG_AGENT_URL",
    "http://10.142.1.136:8000/api/log-agent/chat",
).strip()

AGENT_KEY = required_env("DELL_LOG_AGENT_API_KEY")


def load_dataset() -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []

    with DATASET_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                tests.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"JSONL satırı geçersiz: {line_number}"
                ) from exc

    return tests


def evidence_matches(
    evidence: dict[str, Any],
    expected: dict[str, str],
) -> bool:
    for field, expected_value in expected.items():
        actual_value = evidence.get(field)

        if str(actual_value or "").upper() != expected_value.upper():
            return False

    return True


def evaluate_test(
    client: httpx.Client,
    test: dict[str, Any],
) -> dict[str, Any]:
    test_id = test["id"]
    question = test["question"]
    expected = test.get("expected") or {}

    request_body = {
        "message": question,
        "actor": "evaluation-runner",
        "timezone": "Europe/Istanbul",
        "language": "tr",
        "requestId": f"evaluation-{test_id}",
    }

    started = time.perf_counter()

    try:
        response = client.post(
            AGENT_URL,
            headers={
                "X-AI-Service-Key": AGENT_KEY,
                "Content-Type": "application/json; charset=utf-8",
            },
            json=request_body,
        )

        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        response.raise_for_status()
        payload = response.json()

        evidence = payload.get("evidence") or []
        matching_evidence = [
            item
            for item in evidence
            if evidence_matches(item, expected)
        ]

        if not expected:
            filter_precision = None
        elif not evidence:
            filter_precision = 0.0
        else:
            filter_precision = round(
                len(matching_evidence) / len(evidence),
                4,
            )

        tools_used = payload.get("toolsUsed") or []

        return {
            "id": test_id,
            "question": question,
            "expected": expected,
            "httpStatus": response.status_code,
            "latencyMs": latency_ms,
            "grounded": payload.get("grounded", False),
            "evidenceCount": len(evidence),
            "matchingEvidenceCount": len(matching_evidence),
            "filterPrecision": filter_precision,
            "usedDenseRetrieval": (
                "qdrant_dense_retrieval" in tools_used
            ),
            "usedQwen": (
                "qwen_report_generation" in tools_used
            ),
            "toolsUsed": tools_used,
            "answer": payload.get("answer", ""),
            "evidence": evidence,
            "manualRelevanceScore": None,
            "manualNotes": "",
            "error": None,
        }

    except Exception as exc:
        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return {
            "id": test_id,
            "question": question,
            "expected": expected,
            "httpStatus": getattr(
                getattr(exc, "response", None),
                "status_code",
                None,
            ),
            "latencyMs": latency_ms,
            "grounded": False,
            "evidenceCount": 0,
            "matchingEvidenceCount": 0,
            "filterPrecision": 0.0,
            "usedDenseRetrieval": False,
            "usedQwen": False,
            "toolsUsed": [],
            "answer": "",
            "evidence": [],
            "manualRelevanceScore": None,
            "manualNotes": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def save_json(results: list[dict[str, Any]]) -> None:
    output = {
        "retrievalMode": "dense-only",
        "generatedAt": datetime.now(
            timezone.utc
        ).isoformat(),
        "agentUrl": AGENT_URL,
        "testCount": len(results),
        "results": results,
    }

    JSON_RESULT_FILE.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_csv(results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "question",
        "httpStatus",
        "latencyMs",
        "grounded",
        "evidenceCount",
        "matchingEvidenceCount",
        "filterPrecision",
        "usedDenseRetrieval",
        "usedQwen",
        "manualRelevanceScore",
        "manualNotes",
        "answer",
        "error",
    ]

    with CSV_RESULT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow({
                field: result.get(field)
                for field in fieldnames
            })


def print_summary(results: list[dict[str, Any]]) -> None:
    successful = [
        item
        for item in results
        if item["httpStatus"] == 200
    ]

    grounded = [
        item
        for item in successful
        if item["grounded"]
    ]

    precision_values = [
        item["filterPrecision"]
        for item in successful
        if item["filterPrecision"] is not None
    ]

    average_precision = (
        sum(precision_values) / len(precision_values)
        if precision_values
        else 0.0
    )

    average_latency = (
        sum(item["latencyMs"] for item in successful)
        / len(successful)
        if successful
        else 0.0
    )

    print()
    print("=" * 78)
    print("DENSE BASELINE SONUÇLARI")
    print("=" * 78)
    print(f"Toplam test            : {len(results)}")
    print(f"HTTP 200               : {len(successful)}")
    print(f"Grounded cevap         : {len(grounded)}")
    print(f"Ortalama filtre doğruluğu: {average_precision:.2%}")
    print(f"Ortalama cevap süresi  : {average_latency:.2f} ms")
    print()

    for result in results:
        precision = result["filterPrecision"]

        precision_text = (
            "-"
            if precision is None
            else f"{precision:.0%}"
        )

        print(
            f"{result['id']} | "
            f"HTTP={result['httpStatus']} | "
            f"evidence={result['evidenceCount']} | "
            f"precision={precision_text} | "
            f"{result['latencyMs']} ms"
        )

        if result["error"]:
            print(f"     HATA: {result['error']}")

    print()
    print(f"JSON: {JSON_RESULT_FILE}")
    print(f"CSV : {CSV_RESULT_FILE}")


def main() -> None:
    tests = load_dataset()
    results: list[dict[str, Any]] = []

    with httpx.Client(
        timeout=httpx.Timeout(120.0),
        follow_redirects=True,
    ) as client:
        for index, test in enumerate(tests, start=1):
            print(
                f"[{index}/{len(tests)}] "
                f"{test['id']} gönderiliyor..."
            )

            result = evaluate_test(client, test)
            results.append(result)

    save_json(results)
    save_csv(results)
    print_summary(results)


if __name__ == "__main__":
    main()
