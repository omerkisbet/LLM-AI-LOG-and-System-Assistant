#!/usr/bin/env python3
"""Qdrant vs Upstash Vector dense retrieval benchmark.

Uses the immutable migration JSONL as the common source of truth:
- creates/reuses a dense-only Qdrant benchmark collection,
- queries Qdrant and Upstash with exactly the same vectors,
- computes exact cosine ground truth with NumPy,
- reports latency, Recall@K, self-hit, and Top-K overlap.

No API tokens are written to output files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from upstash_vector import Index


DEFAULT_ENV_PATH = Path("rag/.env")
DEFAULT_OUTPUT_DIR = Path("rag/data/vector_db_benchmarks")
DEFAULT_QDRANT_BENCH_COLLECTION = "iotlab_dense_benchmark_20260728_v1"
DEFAULT_TOP_K = 10
DEFAULT_QUERY_COUNT = 30
DEFAULT_REPETITIONS = 5
DEFAULT_BATCH_SIZE = 100
RANDOM_SEED = 20260728


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil((pct / 100.0) * len(ordered)) - 1
    index = min(max(index, 0), len(ordered) - 1)
    return round(ordered[index], 3)


def mean_or_none(values: list[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None


def load_snapshot(path: Path, expected_dimension: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Snapshot satır {line_number} geçerli JSON değil."
                ) from exc

            vector = record.get("vector")
            point_id = str(record.get("id") or "")

            if not point_id:
                raise RuntimeError(
                    f"Snapshot satır {line_number}: ID eksik."
                )

            if not isinstance(vector, list) or len(vector) != expected_dimension:
                actual = len(vector) if isinstance(vector, list) else None
                raise RuntimeError(
                    f"{point_id}: dimension={actual}, "
                    f"beklenen={expected_dimension}"
                )

            records.append(record)

    if not records:
        raise RuntimeError("Snapshot boş.")

    return records


def collection_count(
    client: QdrantClient,
    collection_name: str,
) -> int | None:
    try:
        info = client.get_collection(collection_name)
        return int(info.points_count or 0)
    except Exception:
        return None


def ensure_qdrant_benchmark_collection(
    client: QdrantClient,
    collection_name: str,
    records: list[dict[str, Any]],
    dimension: int,
    batch_size: int,
    rebuild: bool,
) -> None:
    current_count = collection_count(client, collection_name)

    if rebuild and current_count is not None:
        print(f"Eski benchmark collection siliniyor: {collection_name}")
        client.delete_collection(collection_name)
        current_count = None

    if current_count == len(records):
        print(
            "Qdrant benchmark collection hazır: "
            f"{collection_name} ({current_count} kayıt)"
        )
        return

    if current_count is not None:
        raise RuntimeError(
            f"Qdrant benchmark collection kayıt sayısı {current_count}; "
            f"beklenen {len(records)}. --rebuild-qdrant kullan."
        )

    print(f"Qdrant benchmark collection oluşturuluyor: {collection_name}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            )
        },
    )

    uploaded = 0

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        points = [
            models.PointStruct(
                id=str(record["id"]),
                vector={"dense": record["vector"]},
                payload={
                    **dict(record.get("metadata") or {}),
                    "content": record.get("data") or "",
                },
            )
            for record in batch_records
        ]

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

        uploaded += len(points)
        print(f"Qdrant yükleme: {uploaded}/{len(records)}")

    final_count = collection_count(client, collection_name)

    if final_count != len(records):
        raise RuntimeError(
            f"Qdrant benchmark collection doğrulanamadı: "
            f"{final_count}/{len(records)}"
        )

    print(f"Qdrant benchmark collection doğrulandı: {final_count} kayıt")


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def exact_top_k_ids(
    normalized_matrix: np.ndarray,
    ids: list[str],
    query_vector: np.ndarray,
    top_k: int,
) -> list[str]:
    query = query_vector.astype(np.float32, copy=False)
    query_norm = float(np.linalg.norm(query))

    if query_norm == 0.0:
        raise RuntimeError("Sıfır normlu query vector bulundu.")

    query = query / query_norm
    scores = normalized_matrix @ query

    candidate_count = min(top_k, len(ids))
    top_indices = np.argpartition(
        scores,
        -candidate_count,
    )[-candidate_count:]

    ordered = top_indices[
        np.argsort(scores[top_indices])[::-1]
    ]

    return [ids[int(index)] for index in ordered]


def timed_call(function: Callable[[], list[tuple[str, float]]]) -> tuple[float, list[tuple[str, float]]]:
    started = time.perf_counter()
    result = function()
    latency_ms = (time.perf_counter() - started) * 1000.0
    return latency_ms, result


def query_qdrant(
    client: QdrantClient,
    collection_name: str,
    vector: list[float],
    top_k: int,
) -> list[tuple[str, float]]:
    result = client.query_points(
        collection_name=collection_name,
        query=vector,
        using="dense",
        limit=top_k,
        with_payload=False,
        with_vectors=False,
    )

    return [
        (str(point.id), float(point.score))
        for point in result.points
    ]


def query_upstash(
    index: Index,
    namespace: str,
    vector: list[float],
    top_k: int,
) -> list[tuple[str, float]]:
    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=False,
        include_data=False,
        include_vectors=False,
        namespace=namespace,
    )

    return [
        (str(point.id), float(point.score))
        for point in result
    ]


def recall_at_k(result_ids: list[str], truth_ids: list[str]) -> float:
    if not truth_ids:
        return 0.0
    return len(set(result_ids) & set(truth_ids)) / len(truth_ids)


def overlap_at_k(left: list[str], right: list[str], top_k: int) -> float:
    denominator = min(top_k, len(left), len(right))

    if denominator == 0:
        return 0.0

    return len(set(left[:top_k]) & set(right[:top_k])) / denominator


def jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set

    if not union:
        return 1.0

    return len(left_set & right_set) / len(union)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_provider(
    provider: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    provider_rows = [
        row
        for row in rows
        if row["provider"] == provider
    ]

    latencies = [
        float(row["latency_ms"])
        for row in provider_rows
        if row["success"]
    ]
    recalls = [
        float(row["recall_at_k"])
        for row in provider_rows
        if row["success"]
    ]
    self_hits = [
        bool(row["self_hit_at_1"])
        for row in provider_rows
        if row["success"]
    ]

    return {
        "provider": provider,
        "requests": len(provider_rows),
        "successful": sum(bool(row["success"]) for row in provider_rows),
        "success_rate_percent": round(
            100.0
            * sum(bool(row["success"]) for row in provider_rows)
            / len(provider_rows),
            2,
        )
        if provider_rows
        else 0.0,
        "average_latency_ms": mean_or_none(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "minimum_latency_ms": round(min(latencies), 3) if latencies else None,
        "maximum_latency_ms": round(max(latencies), 3) if latencies else None,
        "estimated_sequential_qps": round(
            1000.0 / statistics.mean(latencies),
            3,
        )
        if latencies and statistics.mean(latencies) > 0
        else None,
        "average_recall_at_k": round(statistics.mean(recalls), 4)
        if recalls
        else None,
        "self_hit_at_1_rate": round(
            sum(self_hits) / len(self_hits),
            4,
        )
        if self_hits
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--qdrant-benchmark-collection",
        default=DEFAULT_QDRANT_BENCH_COLLECTION,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=DEFAULT_QUERY_COUNT,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--rebuild-qdrant",
        action="store_true",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        raise RuntimeError("top-k en az 1 olmalıdır.")
    if args.query_count < 1:
        raise RuntimeError("query-count en az 1 olmalıdır.")
    if args.repetitions < 1:
        raise RuntimeError("repetitions en az 1 olmalıdır.")

    load_dotenv(args.env)

    required_env = [
        "QDRANT_URL",
        "QDRANT_COLLECTION",
        "UPSTASH_VECTOR_REST_URL",
        "UPSTASH_VECTOR_REST_TOKEN",
        "UPSTASH_VECTOR_NAMESPACE",
        "UPSTASH_VECTOR_DIMENSION",
    ]

    missing = [
        name
        for name in required_env
        if not os.getenv(name)
    ]

    if missing:
        raise RuntimeError(
            "Eksik ortam değişkenleri: " + ", ".join(missing)
        )

    dimension = int(os.environ["UPSTASH_VECTOR_DIMENSION"])
    namespace = os.environ["UPSTASH_VECTOR_NAMESPACE"]

    print("=== QDRANT vs UPSTASH DENSE BENCHMARK ===")
    print("Snapshot:", args.snapshot)
    print("Upstash namespace:", namespace)
    print("Qdrant benchmark collection:", args.qdrant_benchmark_collection)
    print("Dimension:", dimension)
    print("Top-K:", args.top_k)
    print("Query count:", args.query_count)
    print("Repetitions:", args.repetitions)
    print()

    records = load_snapshot(args.snapshot, dimension)
    ids = [str(record["id"]) for record in records]
    matrix = np.asarray(
        [record["vector"] for record in records],
        dtype=np.float32,
    )
    normalized_matrix = normalize_matrix(matrix)

    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=60,
    )
    upstash = Index(
        url=os.environ["UPSTASH_VECTOR_REST_URL"],
        token=os.environ["UPSTASH_VECTOR_REST_TOKEN"],
    )

    ensure_qdrant_benchmark_collection(
        client=qdrant,
        collection_name=args.qdrant_benchmark_collection,
        records=records,
        dimension=dimension,
        batch_size=args.batch_size,
        rebuild=args.rebuild_qdrant,
    )

    upstash_info = upstash.info()
    namespace_info = (upstash_info.namespaces or {}).get(namespace)
    upstash_count = int(
        getattr(namespace_info, "vector_count", 0) or 0
    )

    if upstash_count != len(records):
        raise RuntimeError(
            f"Upstash namespace kayıt sayısı {upstash_count}; "
            f"snapshot {len(records)}."
        )

    print(f"Upstash namespace doğrulandı: {upstash_count} kayıt")

    random_generator = random.Random(RANDOM_SEED)
    sample_count = min(args.query_count, len(records))
    query_indices = random_generator.sample(
        range(len(records)),
        sample_count,
    )

    # Each sampled stored vector has a known relevant source ID.
    query_cases: list[dict[str, Any]] = []

    print("Exact cosine ground truth hesaplanıyor...")

    for case_number, record_index in enumerate(query_indices, start=1):
        query_vector = matrix[record_index]
        truth_ids = exact_top_k_ids(
            normalized_matrix=normalized_matrix,
            ids=ids,
            query_vector=query_vector,
            top_k=args.top_k,
        )

        query_cases.append(
            {
                "case_number": case_number,
                "source_id": ids[record_index],
                "vector": query_vector.tolist(),
                "truth_ids": truth_ids,
            }
        )

    # Warm-up is excluded from measured rows.
    warmup = query_cases[0]
    print("Warm-up çalıştırılıyor...")

    query_qdrant(
        client=qdrant,
        collection_name=args.qdrant_benchmark_collection,
        vector=warmup["vector"],
        top_k=args.top_k,
    )
    query_upstash(
        index=upstash,
        namespace=namespace,
        vector=warmup["vector"],
        top_k=args.top_k,
    )

    detail_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    print("Ölçüm başlıyor...")

    for repetition in range(1, args.repetitions + 1):
        for case in query_cases:
            provider_order = (
                ["qdrant", "upstash"]
                if (repetition + case["case_number"]) % 2 == 0
                else ["upstash", "qdrant"]
            )

            results_by_provider: dict[str, list[tuple[str, float]]] = {}

            for provider in provider_order:
                try:
                    if provider == "qdrant":
                        latency_ms, result = timed_call(
                            lambda: query_qdrant(
                                client=qdrant,
                                collection_name=args.qdrant_benchmark_collection,
                                vector=case["vector"],
                                top_k=args.top_k,
                            )
                        )
                    else:
                        latency_ms, result = timed_call(
                            lambda: query_upstash(
                                index=upstash,
                                namespace=namespace,
                                vector=case["vector"],
                                top_k=args.top_k,
                            )
                        )

                    result_ids = [item[0] for item in result]
                    results_by_provider[provider] = result

                    detail_rows.append(
                        {
                            "case_number": case["case_number"],
                            "source_id": case["source_id"],
                            "repetition": repetition,
                            "provider": provider,
                            "success": True,
                            "latency_ms": round(latency_ms, 3),
                            "result_count": len(result_ids),
                            "recall_at_k": round(
                                recall_at_k(
                                    result_ids,
                                    case["truth_ids"],
                                ),
                                4,
                            ),
                            "self_hit_at_1": bool(
                                result_ids
                                and result_ids[0] == case["source_id"]
                            ),
                            "self_in_top_k": case["source_id"] in result_ids,
                            "top_ids": "|".join(result_ids),
                            "top_scores": "|".join(
                                f"{score:.8f}"
                                for _, score in result
                            ),
                            "error": "",
                        }
                    )
                except Exception as exc:
                    results_by_provider[provider] = []
                    detail_rows.append(
                        {
                            "case_number": case["case_number"],
                            "source_id": case["source_id"],
                            "repetition": repetition,
                            "provider": provider,
                            "success": False,
                            "latency_ms": "",
                            "result_count": 0,
                            "recall_at_k": 0.0,
                            "self_hit_at_1": False,
                            "self_in_top_k": False,
                            "top_ids": "",
                            "top_scores": "",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

            qdrant_ids = [
                point_id
                for point_id, _ in results_by_provider.get("qdrant", [])
            ]
            upstash_ids = [
                point_id
                for point_id, _ in results_by_provider.get("upstash", [])
            ]

            comparison_rows.append(
                {
                    "case_number": case["case_number"],
                    "source_id": case["source_id"],
                    "repetition": repetition,
                    "top_k": args.top_k,
                    "overlap_at_k": round(
                        overlap_at_k(
                            qdrant_ids,
                            upstash_ids,
                            args.top_k,
                        ),
                        4,
                    ),
                    "jaccard_at_k": round(
                        jaccard(
                            qdrant_ids[: args.top_k],
                            upstash_ids[: args.top_k],
                        ),
                        4,
                    ),
                    "same_top_1": bool(
                        qdrant_ids
                        and upstash_ids
                        and qdrant_ids[0] == upstash_ids[0]
                    ),
                    "ordered_top_k_exact_match": (
                        qdrant_ids[: args.top_k]
                        == upstash_ids[: args.top_k]
                    ),
                }
            )

        print(
            f"Repetition {repetition}/{args.repetitions} tamamlandı."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    detail_path = args.output_dir / f"vector-db-detail-{run_id}.csv"
    comparison_path = (
        args.output_dir / f"vector-db-comparison-{run_id}.csv"
    )
    summary_path = args.output_dir / f"vector-db-summary-{run_id}.json"

    write_csv(detail_path, detail_rows)
    write_csv(comparison_path, comparison_rows)

    qdrant_summary = summarize_provider("qdrant", detail_rows)
    upstash_summary = summarize_provider("upstash", detail_rows)

    overlaps = [
        float(row["overlap_at_k"])
        for row in comparison_rows
    ]
    jaccards = [
        float(row["jaccard_at_k"])
        for row in comparison_rows
    ]

    summary = {
        "run_id": run_id,
        "started_from_common_snapshot": True,
        "evaluated_at_utc": utc_now(),
        "snapshot_path": str(args.snapshot.resolve()),
        "snapshot_record_count": len(records),
        "dimension": dimension,
        "distance_metric": "cosine",
        "top_k": args.top_k,
        "query_count": sample_count,
        "repetitions": args.repetitions,
        "measured_requests_per_provider": sample_count
        * args.repetitions,
        "random_seed": RANDOM_SEED,
        "qdrant_benchmark_collection": args.qdrant_benchmark_collection,
        "upstash_namespace": namespace,
        "providers": [
            qdrant_summary,
            upstash_summary,
        ],
        "cross_provider": {
            "average_top_k_overlap": round(
                statistics.mean(overlaps),
                4,
            )
            if overlaps
            else None,
            "average_jaccard_at_k": round(
                statistics.mean(jaccards),
                4,
            )
            if jaccards
            else None,
            "same_top_1_rate": round(
                sum(bool(row["same_top_1"]) for row in comparison_rows)
                / len(comparison_rows),
                4,
            )
            if comparison_rows
            else None,
            "ordered_top_k_exact_match_rate": round(
                sum(
                    bool(row["ordered_top_k_exact_match"])
                    for row in comparison_rows
                )
                / len(comparison_rows),
                4,
            )
            if comparison_rows
            else None,
        },
        "methodology_notes": [
            "Both providers were queried from the Dell server.",
            "The same immutable 2918-record migration snapshot is used.",
            "Qdrant benchmark collection is dense-only to match the Upstash index.",
            "Embedding generation is excluded; stored snapshot vectors are query vectors.",
            "Exact cosine nearest neighbors are computed with NumPy as ground truth.",
            "Recall@K is measured against exact cosine ground truth.",
            "Queries are sequential; estimated QPS is 1000 / average latency.",
            "Provider order alternates to reduce order and cache bias.",
        ],
        "outputs": {
            "detail_csv": str(detail_path),
            "comparison_csv": str(comparison_path),
        },
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== SONUÇ ===")

    for provider_summary in summary["providers"]:
        print()
        print(provider_summary["provider"].upper())
        print(
            "  Ortalama latency:",
            provider_summary["average_latency_ms"],
            "ms",
        )
        print(
            "  P50:",
            provider_summary["p50_latency_ms"],
            "ms",
        )
        print(
            "  P95:",
            provider_summary["p95_latency_ms"],
            "ms",
        )
        print(
            "  Recall@K:",
            provider_summary["average_recall_at_k"],
        )
        print(
            "  Self-hit@1:",
            provider_summary["self_hit_at_1_rate"],
        )

    print()
    print(
        "Ortalama Top-K overlap:",
        summary["cross_provider"]["average_top_k_overlap"],
    )
    print(
        "Aynı Top-1 oranı:",
        summary["cross_provider"]["same_top_1_rate"],
    )
    print()
    print("Detay:", detail_path)
    print("Karşılaştırma:", comparison_path)
    print("Özet:", summary_path)


if __name__ == "__main__":
    main()
