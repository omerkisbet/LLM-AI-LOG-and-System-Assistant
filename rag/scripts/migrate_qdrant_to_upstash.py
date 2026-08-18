import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from upstash_vector import Index


ENV_PATH = "rag/.env"
VECTOR_NAME = "dense"
DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 5
VECTOR_TOLERANCE = 1e-5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_dense_vector(point: Any, expected_dimension: int) -> list[float]:
    vectors = point.vector

    if isinstance(vectors, dict):
        vector = vectors.get(VECTOR_NAME)
    else:
        vector = vectors

    if vector is None:
        raise RuntimeError(
            f"{point.id} kaydında '{VECTOR_NAME}' vektörü bulunamadı."
        )

    vector = [float(value) for value in vector]

    if len(vector) != expected_dimension:
        raise RuntimeError(
            f"{point.id} dimension={len(vector)}, "
            f"beklenen={expected_dimension}"
        )

    return vector


def namespace_vector_count(info: Any, namespace: str) -> int:
    namespaces = getattr(info, "namespaces", {}) or {}
    namespace_info = namespaces.get(namespace)

    if namespace_info is None:
        return 0

    return int(
        getattr(namespace_info, "vector_count", 0) or 0
    )


def retry(operation_name: str, function):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return function()
        except Exception as exc:
            last_error = exc

            if attempt == MAX_RETRIES:
                break

            wait_seconds = 2 ** (attempt - 1)

            print(
                f"{operation_name} başarısız: "
                f"{type(exc).__name__}: {exc}"
            )
            print(
                f"{wait_seconds} saniye sonra "
                f"tekrar deneniyor ({attempt}/{MAX_RETRIES})..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"{operation_name} {MAX_RETRIES} denemede başarısız."
    ) from last_error


def capture_snapshot(
    qdrant: QdrantClient,
    collection_name: str,
    expected_dimension: int,
    snapshot_path: Path,
    scroll_batch_size: int,
) -> dict[str, Any]:
    print()
    print("=== 1. QDRANT SNAPSHOT OLUŞTURMA ===")

    collection_before = qdrant.get_collection(collection_name)
    source_count_before = int(collection_before.points_count or 0)

    print("Kaynak collection:", collection_name)
    print("Başlangıç point count:", source_count_before)
    print("Snapshot dosyası:", snapshot_path)

    offset = None
    captured_count = 0
    page_number = 0
    digest = hashlib.sha256()

    with snapshot_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output:
        while True:
            page_number += 1

            points, next_offset = qdrant.scroll(
                collection_name=collection_name,
                limit=scroll_batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=[VECTOR_NAME],
            )

            if not points:
                break

            for point in points:
                point_id = str(point.id)
                payload = dict(point.payload or {})
                vector = get_dense_vector(
                    point,
                    expected_dimension,
                )

                content = payload.pop("content", None)

                if not isinstance(content, str):
                    raise RuntimeError(
                        f"{point_id} kaydında geçerli "
                        "'content' alanı bulunamadı."
                    )

                record = {
                    "id": point_id,
                    "vector": vector,
                    "metadata": payload,
                    "data": content,
                }

                line = json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )

                output.write(line + "\n")
                digest.update((line + "\n").encode("utf-8"))
                captured_count += 1

            print(
                f"Sayfa {page_number}: "
                f"toplam {captured_count} kayıt yakalandı."
            )

            if next_offset is None:
                break

            offset = next_offset

    collection_after = qdrant.get_collection(collection_name)
    source_count_after = int(collection_after.points_count or 0)

    snapshot_info = {
        "collection": collection_name,
        "vector_name": VECTOR_NAME,
        "dimension": expected_dimension,
        "source_count_before": source_count_before,
        "source_count_after": source_count_after,
        "captured_count": captured_count,
        "snapshot_sha256": digest.hexdigest(),
        "captured_at_utc": utc_now(),
        "snapshot_path": str(snapshot_path),
    }

    print()
    print("Snapshot tamamlandı.")
    print("Yakalanan kayıt:", captured_count)
    print("Başlangıç count:", source_count_before)
    print("Bitiş count:", source_count_after)
    print("SHA-256:", snapshot_info["snapshot_sha256"])

    if source_count_before != source_count_after:
        print()
        print(
            "UYARI: Qdrant koleksiyonu snapshot sırasında "
            "değişti."
        )
        print(
            "Benchmarkta kaynak olarak canlı count değil, "
            "oluşturulan değişmez snapshot kullanılacaktır."
        )

    return snapshot_info


def load_snapshot_batches(
    snapshot_path: Path,
    batch_size: int,
):
    batch = []

    with snapshot_path.open(
        "r",
        encoding="utf-8",
    ) as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Snapshot satır {line_number} bozuk."
                ) from exc

            batch.append(record)

            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def upload_snapshot(
    upstash: Index,
    namespace: str,
    snapshot_path: Path,
    batch_size: int,
) -> int:
    print()
    print("=== 2. UPSTASH TOPLU YÜKLEME ===")

    uploaded_count = 0

    for batch_number, batch in enumerate(
        load_snapshot_batches(snapshot_path, batch_size),
        start=1,
    ):
        retry(
            f"Batch {batch_number} upsert",
            lambda current_batch=batch: upstash.upsert(
                vectors=current_batch,
                namespace=namespace,
            ),
        )

        uploaded_count += len(batch)

        print(
            f"Batch {batch_number}: "
            f"{len(batch)} kayıt yazıldı. "
            f"Toplam={uploaded_count}"
        )

    print()
    print("Toplu yükleme tamamlandı.")
    print("Yüklenen kayıt:", uploaded_count)

    return uploaded_count


def max_vector_difference(
    expected: list[float],
    actual: list[float],
) -> float:
    if len(expected) != len(actual):
        return float("inf")

    return max(
        abs(float(left) - float(right))
        for left, right in zip(expected, actual)
    )


def verify_snapshot(
    upstash: Index,
    namespace: str,
    snapshot_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    print()
    print("=== 3. TAM DOĞRULAMA ===")

    verified_count = 0
    validation_errors = []
    first_record = None

    for batch_number, batch in enumerate(
        load_snapshot_batches(snapshot_path, batch_size),
        start=1,
    ):
        if first_record is None and batch:
            first_record = batch[0]

        expected_by_id = {
            str(record["id"]): record
            for record in batch
        }

        ids = list(expected_by_id.keys())

        fetched = retry(
            f"Batch {batch_number} fetch",
            lambda current_ids=ids: upstash.fetch(
                ids=current_ids,
                include_vectors=True,
                include_metadata=True,
                include_data=True,
                namespace=namespace,
            ),
        )

        actual_by_id = {
            str(item.id): item
            for item in fetched
            if item is not None
        }

        for point_id, expected in expected_by_id.items():
            actual = actual_by_id.get(point_id)

            if actual is None:
                validation_errors.append(
                    f"{point_id}: hedefte bulunamadı"
                )
                continue

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

            if difference > VECTOR_TOLERANCE:
                validation_errors.append(
                    f"{point_id}: vector maksimum farkı "
                    f"{difference}"
                )

            verified_count += 1

        print(
            f"Doğrulama batch {batch_number}: "
            f"toplam={verified_count}, "
            f"hata={len(validation_errors)}"
        )

    if validation_errors:
        print()
        print("İlk doğrulama hataları:")

        for error in validation_errors[:20]:
            print(" -", error)

        raise RuntimeError(
            f"Toplam {len(validation_errors)} "
            "doğrulama hatası bulundu."
        )

    print()
    print(f"{verified_count}/{verified_count} kayıt doğrulandı.")

    print()
    print("=== 4. BENZERLİK SORGUSU DOĞRULAMA ===")

    if first_record is None:
        raise RuntimeError("Snapshot boş.")

    query_results = []

    for attempt in range(1, 21):
        query_results = retry(
            "Benzerlik sorgusu",
            lambda: upstash.query(
                vector=first_record["vector"],
                top_k=3,
                include_vectors=False,
                include_metadata=True,
                include_data=True,
                namespace=namespace,
            ),
        )

        if (
            query_results
            and str(query_results[0].id)
            == str(first_record["id"])
        ):
            break

        print(
            f"İndeksleme bekleniyor: {attempt}/20"
        )
        time.sleep(2)

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

    if str(query_results[0].id) != str(first_record["id"]):
        raise RuntimeError(
            "Kaynak vektör kendi kaydını ilk sırada "
            "döndürmedi."
        )

    info = retry(
        "Upstash info",
        upstash.info,
    )

    target_count = namespace_vector_count(
        info,
        namespace,
    )

    print()
    print("Namespace kayıt sayısı:", target_count)

    return {
        "verified_count": verified_count,
        "validation_error_count": 0,
        "target_namespace_count": target_count,
        "query_top_id": str(query_results[0].id),
        "query_top_score": float(query_results[0].score),
        "verified_at_utc": utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Hedef namespace doluysa önce tamamen siler."
        ),
    )

    args = parser.parse_args()

    if args.batch_size < 1 or args.batch_size > 500:
        raise RuntimeError(
            "Batch size 1 ile 500 arasında olmalıdır."
        )

    load_dotenv(ENV_PATH)

    required = [
        "QDRANT_URL",
        "QDRANT_COLLECTION",
        "UPSTASH_VECTOR_REST_URL",
        "UPSTASH_VECTOR_REST_TOKEN",
        "UPSTASH_VECTOR_NAMESPACE",
        "UPSTASH_VECTOR_DIMENSION",
    ]

    missing = [
        name
        for name in required
        if not os.getenv(name)
    ]

    if missing:
        raise RuntimeError(
            "Eksik ortam değişkenleri: "
            + ", ".join(missing)
        )

    collection_name = os.environ["QDRANT_COLLECTION"]
    namespace = os.environ["UPSTASH_VECTOR_NAMESPACE"]
    expected_dimension = int(
        os.environ["UPSTASH_VECTOR_DIMENSION"]
    )

    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )

    upstash = Index(
        url=os.environ["UPSTASH_VECTOR_REST_URL"],
        token=os.environ["UPSTASH_VECTOR_REST_TOKEN"],
    )

    print("=== QDRANT → UPSTASH TAM MIGRATION ===")
    print("Kaynak:", collection_name)
    print("Hedef namespace:", namespace)
    print("Vector:", VECTOR_NAME)
    print("Dimension:", expected_dimension)
    print("Batch size:", args.batch_size)

    info_before = retry(
        "Upstash info",
        upstash.info,
    )

    existing_count = namespace_vector_count(
        info_before,
        namespace,
    )

    print("Mevcut hedef kayıt sayısı:", existing_count)

    if existing_count > 0:
        if not args.replace:
            raise RuntimeError(
                f"Hedef namespace boş değil "
                f"({existing_count} kayıt). "
                "Bilerek değiştirmek için --replace kullan."
            )

        print("Hedef namespace siliniyor...")

        retry(
            "Namespace silme",
            lambda: upstash.delete_namespace(namespace),
        )

        time.sleep(2)

        print("Hedef namespace temizlendi.")

    output_directory = Path(
        "rag/data/upstash_migrations"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_id = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    snapshot_path = (
        output_directory
        / f"{namespace}-{run_id}.jsonl"
    )

    manifest_path = (
        output_directory
        / f"{namespace}-{run_id}-manifest.json"
    )

    manifest = {
        "run_id": run_id,
        "status": "started",
        "started_at_utc": utc_now(),
        "source_collection": collection_name,
        "target_namespace": namespace,
        "dimension": expected_dimension,
        "vector_name": VECTOR_NAME,
        "batch_size": args.batch_size,
    }

    try:
        snapshot_info = capture_snapshot(
            qdrant=qdrant,
            collection_name=collection_name,
            expected_dimension=expected_dimension,
            snapshot_path=snapshot_path,
            scroll_batch_size=args.batch_size,
        )

        manifest["snapshot"] = snapshot_info

        uploaded_count = upload_snapshot(
            upstash=upstash,
            namespace=namespace,
            snapshot_path=snapshot_path,
            batch_size=args.batch_size,
        )

        manifest["uploaded_count"] = uploaded_count

        verification = verify_snapshot(
            upstash=upstash,
            namespace=namespace,
            snapshot_path=snapshot_path,
            batch_size=args.batch_size,
        )

        manifest["verification"] = verification

        if uploaded_count != snapshot_info["captured_count"]:
            raise RuntimeError(
                "Yüklenen kayıt sayısı snapshot sayısıyla "
                "eşleşmiyor."
            )

        if (
            verification["verified_count"]
            != snapshot_info["captured_count"]
        ):
            raise RuntimeError(
                "Doğrulanan kayıt sayısı snapshot sayısıyla "
                "eşleşmiyor."
            )

        if (
            verification["target_namespace_count"]
            != snapshot_info["captured_count"]
        ):
            raise RuntimeError(
                "Upstash namespace sayısı snapshot "
                "sayısıyla eşleşmiyor."
            )

        manifest["status"] = "success"
        manifest["finished_at_utc"] = utc_now()

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error_message"] = str(exc)
        manifest["finished_at_utc"] = utc_now()

        raise

    finally:
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print("Manifest:", manifest_path)
        print("Snapshot:", snapshot_path)

    print()
    print("QDRANT → UPSTASH TAM MIGRATION BAŞARILI")


if __name__ == "__main__":
    main()
