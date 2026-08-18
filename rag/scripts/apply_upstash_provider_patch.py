from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

TARGET = Path.home() / "huggingface-model-server" / "rag" / "log_agent" / "main.py"

if not TARGET.exists():
    raise SystemExit(f"Dosya bulunamadı: {TARGET}")

text = TARGET.read_text(encoding="utf-8")
original = text


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: beklenen blok sayısı 1, bulunan {count}. "
            "Dosya değiştirilmedi."
        )
    text = text.replace(old, new, 1)


replace_once(
    "from qdrant_client import QdrantClient, models\n",
    "from qdrant_client import QdrantClient, models\n"
    "from upstash_vector import Index\n",
    "Upstash import",
)

replace_once(
    '''COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "iotlab_operational_logs",
).strip()
''',
    '''COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "iotlab_operational_logs",
).strip()

VECTOR_DB_PROVIDER = os.getenv(
    "VECTOR_DB_PROVIDER",
    "qdrant",
).strip().lower()

if VECTOR_DB_PROVIDER not in {"qdrant", "upstash"}:
    raise RuntimeError(
        "VECTOR_DB_PROVIDER yalnızca 'qdrant' veya 'upstash' olabilir."
    )

UPSTASH_VECTOR_REST_URL = os.getenv(
    "UPSTASH_VECTOR_REST_URL",
    "",
).strip()
UPSTASH_VECTOR_REST_TOKEN = os.getenv(
    "UPSTASH_VECTOR_REST_TOKEN",
    "",
).strip()
UPSTASH_VECTOR_NAMESPACE = os.getenv(
    "UPSTASH_VECTOR_NAMESPACE",
    "",
).strip()
UPSTASH_VECTOR_DIMENSION = int(
    os.getenv("UPSTASH_VECTOR_DIMENSION", "384")
)

if VECTOR_DB_PROVIDER == "upstash":
    missing_upstash = [
        name
        for name, value in {
            "UPSTASH_VECTOR_REST_URL": UPSTASH_VECTOR_REST_URL,
            "UPSTASH_VECTOR_REST_TOKEN": UPSTASH_VECTOR_REST_TOKEN,
            "UPSTASH_VECTOR_NAMESPACE": UPSTASH_VECTOR_NAMESPACE,
        }.items()
        if not value
    ]

    if missing_upstash:
        raise RuntimeError(
            "Upstash provider için eksik ortam değişkenleri: "
            + ", ".join(missing_upstash)
        )
''',
    "Provider environment",
)

replace_once(
    '''RESULT_LIMIT = max(
    1,
    min(int(os.getenv("LOG_AGENT_RESULT_LIMIT", "12")), 50),
)
''',
    '''RESULT_LIMIT = max(
    1,
    min(int(os.getenv("LOG_AGENT_RESULT_LIMIT", "12")), 50),
)
UPSTASH_CANDIDATE_LIMIT = max(
    RESULT_LIMIT,
    min(
        int(os.getenv("UPSTASH_CANDIDATE_LIMIT", "80")),
        1000,
    ),
)
''',
    "Candidate limit",
)

replace_once(
    '''    app.state.qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
    )

''',
    '''    app.state.qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
    )

    app.state.upstash = None

    if VECTOR_DB_PROVIDER == "upstash":
        app.state.upstash = Index(
            url=UPSTASH_VECTOR_REST_URL,
            token=UPSTASH_VECTOR_REST_TOKEN,
        )

        upstash_info = app.state.upstash.info()

        if int(upstash_info.dimension) != UPSTASH_VECTOR_DIMENSION:
            raise RuntimeError(
                "Upstash dimension uyuşmazlığı: "
                f"index={upstash_info.dimension}, "
                f"ayar={UPSTASH_VECTOR_DIMENSION}"
            )

        print(
            "Upstash Vector hazır:",
            UPSTASH_VECTOR_NAMESPACE,
            f"dimension={upstash_info.dimension}",
        )

''',
    "Lifespan clients",
)

replace_once(
    '''def event_to_evidence(event: dict[str, Any]) -> EvidenceResponse:
''',
    '''def upstash_result_to_evidence(result: Any) -> EvidenceResponse:
    metadata = dict(result.metadata or {})

    return EvidenceResponse(
        sourceType=str(metadata.get("sourceType", "UNKNOWN")),
        recordId=str(
            metadata.get("mongoId")
            or metadata.get("documentId")
            or result.id
        ),
        timestamp=parse_timestamp(
            metadata.get("timestamp", datetime.now(timezone.utc))
        ),
        level=metadata.get("level"),
        action=metadata.get("action"),
        entityType=metadata.get("entityType"),
        entityId=metadata.get("entityId"),
        description=str(
            result.data
            or metadata.get("content")
            or ""
        )[:1500],
        requestId=metadata.get("requestId"),
    )


def _upstash_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"

    if isinstance(value, (int, float)):
        return str(value)

    return json.dumps(
        str(value),
        ensure_ascii=False,
    )


def build_upstash_filter(
    query_filter: models.Filter | None,
) -> str | None:
    # Only equality conditions are sent to Upstash. Datetime ranges remain
    # a local post-filter because snapshot timestamps are ISO strings.
    if query_filter is None:
        return None

    expressions: list[str] = []

    for condition in list(query_filter.must or []):
        key = getattr(condition, "key", None)
        match = getattr(condition, "match", None)

        if not key or match is None:
            continue

        value = getattr(match, "value", None)

        if value is None:
            continue

        expressions.append(
            f"{key} = {_upstash_literal(value)}"
        )

    if not expressions:
        return None

    return " AND ".join(expressions)


def _metadata_matches_condition(
    metadata: dict[str, Any],
    condition: Any,
) -> bool:
    key = getattr(condition, "key", None)

    if not key:
        return True

    actual = metadata.get(key)
    match = getattr(condition, "match", None)

    if match is not None:
        expected = getattr(match, "value", None)

        if expected is not None and actual != expected:
            return False

    value_range = getattr(condition, "range", None)

    if value_range is None:
        return True

    if actual is None:
        return False

    if key == "timestamp":
        try:
            actual_value: Any = parse_timestamp(actual)
        except Exception:
            return False
    else:
        actual_value = actual

    for attribute, operator in (
        ("gt", lambda left, right: left > right),
        ("gte", lambda left, right: left >= right),
        ("lt", lambda left, right: left < right),
        ("lte", lambda left, right: left <= right),
    ):
        boundary = getattr(value_range, attribute, None)

        if boundary is None:
            continue

        if key == "timestamp":
            boundary = parse_timestamp(boundary)

        try:
            if not operator(actual_value, boundary):
                return False
        except TypeError:
            return False

    return True


def metadata_matches_qdrant_filter(
    metadata: dict[str, Any],
    query_filter: models.Filter | None,
) -> bool:
    if query_filter is None:
        return True

    must_conditions = list(query_filter.must or [])

    if not all(
        _metadata_matches_condition(metadata, condition)
        for condition in must_conditions
    ):
        return False

    must_not_conditions = list(query_filter.must_not or [])

    if any(
        _metadata_matches_condition(metadata, condition)
        for condition in must_not_conditions
    ):
        return False

    should_conditions = list(query_filter.should or [])

    if should_conditions and not any(
        _metadata_matches_condition(metadata, condition)
        for condition in should_conditions
    ):
        return False

    return True


def event_to_evidence(event: dict[str, Any]) -> EvidenceResponse:
''',
    "Upstash helpers",
)

replace_once(
    '''@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    collection = request.app.state.qdrant.get_collection(
        COLLECTION_NAME
    )

    telemetry_health: dict[str, Any]
''',
    '''@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    vector_store_health: dict[str, Any]

    if VECTOR_DB_PROVIDER == "upstash":
        try:
            upstash_info = request.app.state.upstash.info()
            namespace_info = (
                upstash_info.namespaces.get(
                    UPSTASH_VECTOR_NAMESPACE
                )
            )
            points_count = int(
                getattr(
                    namespace_info,
                    "vector_count",
                    0,
                )
                or 0
            )
            vector_store_health = {
                "status": "UP",
                "provider": "upstash",
                "namespace": UPSTASH_VECTOR_NAMESPACE,
                "pointsCount": points_count,
                "pendingPointsCount": int(
                    getattr(
                        namespace_info,
                        "pending_vector_count",
                        0,
                    )
                    or 0
                ),
                "dimension": int(upstash_info.dimension),
                "similarity": str(
                    upstash_info.similarity_function
                ),
                "retrievalMode": "dense",
                "operationalFallback": "qdrant",
            }
        except Exception as exc:
            points_count = 0
            vector_store_health = {
                "status": "DOWN",
                "provider": "upstash",
                "namespace": UPSTASH_VECTOR_NAMESPACE,
                "error": type(exc).__name__,
            }
    else:
        try:
            collection = (
                request.app.state.qdrant.get_collection(
                    COLLECTION_NAME
                )
            )
            points_count = int(collection.points_count or 0)
            vector_store_health = {
                "status": "UP",
                "provider": "qdrant",
                "collection": COLLECTION_NAME,
                "pointsCount": points_count,
                "retrievalMode": "dense+sparse+rrf",
            }
        except Exception as exc:
            points_count = 0
            vector_store_health = {
                "status": "DOWN",
                "provider": "qdrant",
                "collection": COLLECTION_NAME,
                "error": type(exc).__name__,
            }

    telemetry_health: dict[str, Any]
''',
    "Health header",
)

replace_once(
    '''    return {
        "status": "UP",
        "collection": COLLECTION_NAME,
        "pointsCount": collection.points_count,
        "model": MODEL_NAME,
        "telemetry": telemetry_health,
        "trainingManagement": training_management_health,
        "incidents": incident_health,
    }
''',
    '''    return {
        "status": (
            "UP"
            if vector_store_health["status"] == "UP"
            else "DEGRADED"
        ),
        "vectorDbProvider": VECTOR_DB_PROVIDER,
        "collection": (
            COLLECTION_NAME
            if VECTOR_DB_PROVIDER == "qdrant"
            else UPSTASH_VECTOR_NAMESPACE
        ),
        "pointsCount": points_count,
        "vectorStore": vector_store_health,
        "model": MODEL_NAME,
        "telemetry": telemetry_health,
        "trainingManagement": training_management_health,
        "incidents": incident_health,
    }
''',
    "Health response",
)

replace_once(
    '''        if plan.retrieval_mode in {
            RetrievalMode.AGGREGATE,
            RetrievalMode.FILTERED_LIST,
        }:
            operational_summary = execute_operational_plan(
                client=request.app.state.qdrant,
                collection_name=COLLECTION_NAME,
                plan=plan,
            )

            filtered_total = int(
                operational_summary.get(
                    "totalEvents",
                    0,
                )
            )

            evidence = [
                event_to_evidence(event)
                for event in operational_summary.get(
                    "latestEvents",
                    [],
                )
            ]

            tools_used.extend(
                [
                    "qdrant_payload_filter",
                    "qdrant_scroll_retrieval",
                    "deterministic_operational_summary",
                ]
            )
        else:
            dense_vector = next(
                request.app.state.embedder.embed(
                    [f"query: {question}"]
                )
            )

            sparse_embedding = next(
                request.app.state.sparse_embedder.query_embed(
                    question
                )
            )

            sparse_vector = models.SparseVector(
                indices=(
                    sparse_embedding.indices.tolist()
                ),
                values=(
                    sparse_embedding.values.tolist()
                ),
            )

            result = (
                request.app.state.qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    prefetch=[
                        models.Prefetch(
                            query=dense_vector.tolist(),
                            using="dense",
                            filter=query_filter,
                            limit=HYBRID_PREFETCH_LIMIT,
                        ),
                        models.Prefetch(
                            query=sparse_vector,
                            using="sparse",
                            filter=query_filter,
                            limit=HYBRID_PREFETCH_LIMIT,
                        ),
                    ],
                    query=models.FusionQuery(
                        fusion=models.Fusion.RRF,
                    ),
                    limit=RESULT_LIMIT,
                    with_payload=True,
                    with_vectors=False,
                )
            )

            evidence = [
                point_to_evidence(point)
                for point in result.points
            ]

            # Payload-only count exact-term sorgularında yanıltıcıdır.
            if (
                query_filter is not None
                and not plan.exact_terms
            ):
                filtered_total = (
                    request.app.state.qdrant.count(
                        collection_name=COLLECTION_NAME,
                        count_filter=query_filter,
                        exact=True,
                    ).count
                )

            tools_used.extend(
                [
                    "qdrant_dense_retrieval",
                    "qdrant_sparse_bm25_retrieval",
                    "qdrant_rrf_hybrid_fusion",
                    "qdrant_payload_filter",
                ]
            )
''',
    '''        if plan.retrieval_mode in {
            RetrievalMode.AGGREGATE,
            RetrievalMode.FILTERED_LIST,
        }:
            # İlk Upstash entegrasyonunda kesin aggregate/list işlemleri
            # mevcut Qdrant operasyon katmanında tutulur. Semantik retrieval
            # seçilen provider üzerinden çalışır.
            operational_summary = execute_operational_plan(
                client=request.app.state.qdrant,
                collection_name=COLLECTION_NAME,
                plan=plan,
            )

            filtered_total = int(
                operational_summary.get(
                    "totalEvents",
                    0,
                )
            )

            evidence = [
                event_to_evidence(event)
                for event in operational_summary.get(
                    "latestEvents",
                    [],
                )
            ]

            tools_used.extend(
                [
                    "qdrant_payload_filter",
                    "qdrant_scroll_retrieval",
                    "deterministic_operational_summary",
                ]
            )

            if VECTOR_DB_PROVIDER == "upstash":
                tools_used.append(
                    "qdrant_operational_fallback"
                )
        else:
            dense_vector = next(
                request.app.state.embedder.embed(
                    [f"query: {question}"]
                )
            )

            if VECTOR_DB_PROVIDER == "upstash":
                upstash_filter = build_upstash_filter(
                    query_filter
                )

                upstash_results = (
                    request.app.state.upstash.query(
                        vector=dense_vector.tolist(),
                        top_k=UPSTASH_CANDIDATE_LIMIT,
                        include_vectors=False,
                        include_metadata=True,
                        include_data=True,
                        filter=upstash_filter,
                        namespace=UPSTASH_VECTOR_NAMESPACE,
                    )
                )

                locally_filtered_results = [
                    result
                    for result in upstash_results
                    if metadata_matches_qdrant_filter(
                        dict(result.metadata or {}),
                        query_filter,
                    )
                ]

                evidence = [
                    upstash_result_to_evidence(result)
                    for result in locally_filtered_results[
                        :RESULT_LIMIT
                    ]
                ]

                tools_used.extend(
                    [
                        "upstash_dense_retrieval",
                        "upstash_namespace_query",
                    ]
                )

                if upstash_filter:
                    tools_used.append(
                        "upstash_metadata_filter"
                    )

                if query_filter is not None:
                    tools_used.append(
                        "upstash_local_post_filter"
                    )
            else:
                sparse_embedding = next(
                    request.app.state.sparse_embedder.query_embed(
                        question
                    )
                )

                sparse_vector = models.SparseVector(
                    indices=(
                        sparse_embedding.indices.tolist()
                    ),
                    values=(
                        sparse_embedding.values.tolist()
                    ),
                )

                result = (
                    request.app.state.qdrant.query_points(
                        collection_name=COLLECTION_NAME,
                        prefetch=[
                            models.Prefetch(
                                query=dense_vector.tolist(),
                                using="dense",
                                filter=query_filter,
                                limit=HYBRID_PREFETCH_LIMIT,
                            ),
                            models.Prefetch(
                                query=sparse_vector,
                                using="sparse",
                                filter=query_filter,
                                limit=HYBRID_PREFETCH_LIMIT,
                            ),
                        ],
                        query=models.FusionQuery(
                            fusion=models.Fusion.RRF,
                        ),
                        limit=RESULT_LIMIT,
                        with_payload=True,
                        with_vectors=False,
                    )
                )

                evidence = [
                    point_to_evidence(point)
                    for point in result.points
                ]

                # Payload-only count exact-term sorgularında yanıltıcıdır.
                if (
                    query_filter is not None
                    and not plan.exact_terms
                ):
                    filtered_total = (
                        request.app.state.qdrant.count(
                            collection_name=COLLECTION_NAME,
                            count_filter=query_filter,
                            exact=True,
                        ).count
                    )

                tools_used.extend(
                    [
                        "qdrant_dense_retrieval",
                        "qdrant_sparse_bm25_retrieval",
                        "qdrant_rrf_hybrid_fusion",
                        "qdrant_payload_filter",
                    ]
                )
''',
    "Chat retrieval",
)

if text == original:
    raise RuntimeError("Hiçbir değişiklik yapılmadı.")

backup = TARGET.with_name(
    f"main.py.before-upstash-{datetime.now().strftime('%Y%m%d-%H%M%S')}.bak"
)
shutil.copy2(TARGET, backup)
TARGET.write_text(text, encoding="utf-8")

print("Patch uygulandı.")
print("Yedek:", backup)
print("Dosya:", TARGET)
