#!/usr/bin/env python3
"""Run a labeled Log Agent benchmark against the existing HTTP endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("annotation_status") != "approved":
                continue
            rows.append(record)
    if not rows:
        raise RuntimeError("Onaylanmış benchmark vakası bulunamadı.")
    return rows


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def contains_all(text: str, keywords: list[str]) -> bool:
    normalized = normalize_text(text)
    return all(normalize_text(keyword) in normalized for keyword in keywords)


def contains_any(text: str, keywords: list[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(keyword) in normalized for keyword in keywords)


def extract_answer(response_json: dict[str, Any]) -> str:
    for field in ("answer", "response", "message", "content"):
        value = response_json.get(field)
        if isinstance(value, str):
            return value

    nested = response_json.get("data")
    if isinstance(nested, dict):
        for field in ("answer", "response", "message", "content"):
            value = nested.get(field)
            if isinstance(value, str):
                return value

    return json.dumps(response_json, ensure_ascii=False)


def extract_evidence_ids(response_json: dict[str, Any]) -> list[str]:
    containers = [
        response_json.get("evidence"),
        response_json.get("sources"),
        response_json.get("data", {}).get("evidence")
        if isinstance(response_json.get("data"), dict)
        else None,
    ]

    ids: list[str] = []
    for container in containers:
        if not isinstance(container, list):
            continue
        for item in container:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, dict):
                for field in ("id", "pointId", "logId", "sourceId"):
                    if item.get(field):
                        ids.append(str(item[field]))
                        break
    return ids


def f1_from_sets(predicted: set[str], expected: set[str]) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    tp = len(predicted & expected)
    precision = tp / len(predicted)
    recall = tp / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=Path("rag/.env"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default="http://10.142.1.136:8000/api/log-agent/chat",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rag/data/log_agent_benchmark/results"),
    )
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    load_dotenv(args.env)
    api_key = os.getenv("DELL_LOG_AGENT_API_KEY")
    if not api_key:
        raise RuntimeError("DELL_LOG_AGENT_API_KEY bulunamadı.")

    cases = load_jsonl(args.dataset)
    result_rows: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        question = case.get("question") or case.get("draft_question")
        expected = case["expected"]

        started = time.perf_counter()
        error = ""
        response_json: dict[str, Any] = {}
        success = False

        try:
            response = requests.post(
                args.endpoint,
                headers={
                    "Content-Type": "application/json",
                    "X-AI-Service-Key": api_key,
                },
                json={
                    "message": question,
                    "timezone": "Europe/Istanbul",
                    "language": "tr",
                },
                timeout=args.timeout,
            )
            response.raise_for_status()
            response_json = response.json()
            success = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        latency_ms = (time.perf_counter() - started) * 1000.0
        answer = extract_answer(response_json) if success else ""
        evidence_ids = extract_evidence_ids(response_json) if success else []

        root_keywords = list(expected.get("root_cause_keywords") or [])
        solution_keywords = list(expected.get("solution_keywords") or [])
        expected_evidence = set(str(x) for x in expected.get("evidence_ids") or [])

        error_type = str(expected.get("error_type") or "")
        error_type_score = (
            1.0 if error_type and contains_any(answer, [error_type]) else 0.0
        )
        root_cause_score = (
            1.0 if root_keywords and contains_all(answer, root_keywords) else 0.0
        )
        solution_score = (
            1.0 if solution_keywords and contains_any(answer, solution_keywords) else 0.0
        )
        evidence_f1 = f1_from_sets(set(evidence_ids), expected_evidence)

        should_abstain = bool(expected.get("should_abstain"))
        abstention_phrases = [
            "yetersiz",
            "kesin olarak belirlenemiyor",
            "ek log",
            "daha fazla log",
            "kanıt bulunamadı",
        ]
        did_abstain = contains_any(answer, abstention_phrases)
        abstention_score = 1.0 if should_abstain == did_abstain else 0.0

        weighted_score = (
            0.30 * error_type_score
            + 0.30 * root_cause_score
            + 0.20 * evidence_f1
            + 0.15 * solution_score
            + 0.05 * abstention_score
        )

        result_rows.append(
            {
                "case_id": case.get("case_id") or case.get("candidate_id"),
                "category": case.get("category_hint"),
                "question": question,
                "success": success,
                "latency_ms": round(latency_ms, 2),
                "error_type_score": round(error_type_score, 4),
                "root_cause_score": round(root_cause_score, 4),
                "evidence_f1": round(evidence_f1, 4),
                "solution_score": round(solution_score, 4),
                "abstention_score": round(abstention_score, 4),
                "weighted_log_agent_score": round(weighted_score, 4),
                "answer": answer,
                "evidence_ids": "|".join(evidence_ids),
                "error": error,
            }
        )

        print(
            f"[{index}/{len(cases)}] "
            f"{result_rows[-1]['case_id']} "
            f"score={weighted_score:.3f} "
            f"latency={latency_ms:.0f} ms"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    detail_path = args.output_dir / f"log-agent-detail-{run_id}.csv"
    summary_path = args.output_dir / f"log-agent-summary-{run_id}.json"

    with detail_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)

    successful = [row for row in result_rows if row["success"]]
    latencies = [float(row["latency_ms"]) for row in successful]

    summary = {
        "run_id": run_id,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "cases": len(result_rows),
        "successful_requests": len(successful),
        "success_rate": round(len(successful) / len(result_rows), 4),
        "average_log_agent_score": round(
            statistics.mean(float(row["weighted_log_agent_score"]) for row in result_rows),
            4,
        ),
        "average_error_type_score": round(
            statistics.mean(float(row["error_type_score"]) for row in result_rows),
            4,
        ),
        "average_root_cause_score": round(
            statistics.mean(float(row["root_cause_score"]) for row in result_rows),
            4,
        ),
        "average_evidence_f1": round(
            statistics.mean(float(row["evidence_f1"]) for row in result_rows),
            4,
        ),
        "average_solution_score": round(
            statistics.mean(float(row["solution_score"]) for row in result_rows),
            4,
        ),
        "average_abstention_score": round(
            statistics.mean(float(row["abstention_score"]) for row in result_rows),
            4,
        ),
        "average_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "api_cost_usd": 0.0,
        "detail_csv": str(detail_path),
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
