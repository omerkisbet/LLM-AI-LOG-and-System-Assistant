import math
import os
import time
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from upstash_vector import Index


ENV_PATH = "rag/.env"

QDRANT_VECTOR_NAME = "dense"
DRYRUN_NAMESPACE = "qdrant-dryrun-v1"
DRYRUN_LIMIT = 10
EXPECTED_DIMENSION = 384


def get_dense_vector(point: Any) -> list[float]:
    vectors = point.vector

    if isinstance(vectors, dict):
        dense = vectors.get(QDRANT_VECTOR_NAME)
    else:
        dense = vectors

    if dense is None:
        raise RuntimeError(
            f"{point.id} kaydında '{QDRANT_VECTOR_NAME}' vektörü yok."
        )

    dense = list(dense)

    if len(dense) != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"{point.id} için dimension={len(dense)}; "
            f"beklenen={EXPECTED_DIMENSION}"
        )

    return dense


def max_vector_difference(
    expected: list[float],
    actual: list[float],
) -> float:
    if len(expected) != len(actual):
        return math.inf

    return max(
        abs(float(left) - float(right))
        for left, right in zip(expected, actual)
    )


load_dotenv(ENV_PATH)

required_variables = [
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "UPSTASH_VECTOR_REST_URL",
    "UPSTASH_VECTOR_REST_TOKEN",
]

missing = [
    name
    for name in required_variables
    if not os.getenv(name)
]

if missing:
    raise RuntimeError(
        "Eksik ortam değişkenleri: " + ", ".join(missing)
    )

qdrant_collection = os.environ["QDRANT_COLLECTION"]

qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.getenv("QDRANT_API_KEY") or None,
)

upstash = Index(
    url=os.environ["UPSTASH_VECTOR_REST_URL"],
    token=os.environ["UPSTASH_VECTOR_REST_TOKEN"],
)

print("=== QDRANT → UPSTASH DRY-RUN ===")
print("Qdrant collection:", qdrant_collection)
print("Upstash namespace:", DRYRUN_NAMESPACE)
print("Kayıt limiti:", DRYRUN_LIMIT)
print()

# Önce eski bir dry-run namespace kaldıysa temizle.
try:
    existing_namespaces = upstash.list_namespaces()

    if DRYRUN_NAMESPACE in existing_namespaces:
        print("Eski dry-run namespace siliniyor...")
        upstash.delete_namespace(DRYRUN_NAMESPACE)
        time.sleep(1)
except Exception as exc:
    print(
        "Namespace ön temizleme uyarısı:",
        type(exc).__name__,
        str(exc),
    )

print("1. Qdrant'tan 10 kayıt okunuyor...")

points, next_offset = qdrant.scroll(
    collection_name=qdrant_collection,
    limit=DRYRUN_LIMIT,
    with_payload=True,
    with_vectors=[QDRANT_VECTOR_NAME],
)

if len(points) != DRYRUN_LIMIT:
    raise RuntimeError(
        f"Beklenen {DRYRUN_LIMIT} kayıt yerine "
        f"{len(points)} kayıt okundu."
    )

print("Okunan kayıt:", len(points))
print("Sonraki offset:", next_offset)

upstash_vectors = []
expected_by_id = {}

for point in points:
    point_id = str(point.id)
    payload = dict(point.payload or {})
    dense_vector = get_dense_vector(point)

    content = payload.pop("content", None)

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            f"{point_id} kaydında geçerli content alanı yok."
        )

    expected_by_id[point_id] = {
        "vector": dense_vector,
        "metadata": payload,
        "data": content,
    }

    upstash_vectors.append(
        {
            "id": point_id,
            "vector": dense_vector,
            "metadata": payload,
            "data": content,
        }
    )

print()
print("2. Kayıtlar Upstash'a yazılıyor...")

upstash.upsert(
    vectors=upstash_vectors,
    namespace=DRYRUN_NAMESPACE,
)

print("Upsert kabul edildi.")

print()
print("3. Yazılan kayıtlar ID ile geri okunuyor...")

ids = list(expected_by_id.keys())
fetched_by_id = {}

for attempt in range(1, 16):
    fetched = upstash.fetch(
        ids=ids,
        include_vectors=True,
        include_metadata=True,
        include_data=True,
        namespace=DRYRUN_NAMESPACE,
    )

    fetched_by_id = {
        str(item.id): item
        for item in fetched
        if item is not None
    }

    if len(fetched_by_id) == DRYRUN_LIMIT:
        break

    print(
        f"Görünür kayıt: {len(fetched_by_id)}/{DRYRUN_LIMIT}; "
        f"deneme {attempt}/15"
    )
    time.sleep(1)

if len(fetched_by_id) != DRYRUN_LIMIT:
    missing_ids = sorted(
        set(ids) - set(fetched_by_id.keys())
    )

    raise RuntimeError(
        "Upstash'ta okunamayan ID'ler: "
        + ", ".join(missing_ids)
    )

print("Okunan kayıt:", len(fetched_by_id))

print()
print("4. ID, data, metadata ve vector doğrulanıyor...")

validation_errors = []

for point_id, expected in expected_by_id.items():
    actual = fetched_by_id[point_id]

    if actual.data != expected["data"]:
        validation_errors.append(
            f"{point_id}: data eşleşmiyor"
        )

    actual_metadata = dict(actual.metadata or {})

    if actual_metadata != expected["metadata"]:
        validation_errors.append(
            f"{point_id}: metadata eşleşmiyor"
        )

    actual_vector = list(actual.vector or [])

    difference = max_vector_difference(
        expected["vector"],
        actual_vector,
    )

    if difference > 1e-6:
        validation_errors.append(
            f"{point_id}: vector farkı={difference}"
        )

if validation_errors:
    print("Doğrulama hataları:")

    for error in validation_errors:
        print(" -", error)

    raise RuntimeError(
        f"{len(validation_errors)} doğrulama hatası bulundu."
    )

print("10/10 kayıt birebir doğrulandı.")

print()
print("5. Benzerlik sorgusu test ediliyor...")

first_id = ids[0]
first_vector = expected_by_id[first_id]["vector"]

query_results = []

for attempt in range(1, 16):
    query_results = upstash.query(
        vector=first_vector,
        top_k=3,
        include_vectors=False,
        include_metadata=True,
        include_data=True,
        namespace=DRYRUN_NAMESPACE,
    )

    if query_results and str(query_results[0].id) == first_id:
        break

    print(f"İndeksleme bekleniyor: {attempt}/15")
    time.sleep(1)

if not query_results:
    raise RuntimeError(
        "Benzerlik sorgusu sonuç döndürmedi."
    )

print("İlk üç sonuç:")

for position, result in enumerate(
    query_results,
    start=1,
):
    print(
        f"  {position}. ID={result.id}, "
        f"score={result.score}"
    )

if str(query_results[0].id) != first_id:
    raise RuntimeError(
        "Aynı vektörle yapılan sorguda kaynak kayıt "
        "ilk sırada dönmedi."
    )

print()
print("6. Dry-run namespace temizleniyor...")

upstash.delete_namespace(DRYRUN_NAMESPACE)

print("Dry-run namespace silindi.")
print()
print("QDRANT → UPSTASH 10 KAYITLIK DRY-RUN BAŞARILI")
