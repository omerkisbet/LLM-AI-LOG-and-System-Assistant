from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from application_data_tools import (
    ApplicationDataRequest,
    ApplicationDataToolError,
    execute_application_data_request,
)
from backend_health_tools import (
    BackendHealthToolError,
    get_backend_health,
    get_backend_system_info,
)


@dataclass(frozen=True)
class AssistantToolAnswer:
    answer: str
    tools_used: tuple[str, ...]
    grounded: bool
    filtered_total: int | None = None
    raw_result: dict[str, Any] | None = None


def normalize_for_routing(value: str) -> str:
    translated = value.lower().translate(
        str.maketrans(
            {
                "ı": "i",
                "İ": "i",
                "ş": "s",
                "Ş": "s",
                "ğ": "g",
                "Ğ": "g",
                "ü": "u",
                "Ü": "u",
                "ö": "o",
                "Ö": "o",
                "ç": "c",
                "Ç": "c",
            }
        )
    )

    decomposed = unicodedata.normalize(
        "NFKD",
        translated,
    )

    return " ".join(
        "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character)
        ).split()
    )


def contains_any(
    value: str,
    terms: tuple[str, ...],
) -> bool:
    return any(term in value for term in terms)


def extract_quoted_text(question: str) -> str | None:
    match = re.search(
        r"""["“']([^"”']{2,100})["”']""",
        question,
    )

    if match:
        return match.group(1).strip()

    return None


def extract_person_name(question: str) -> str | None:
    patterns = (
        r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]{1,40})"
        r"(?:'nın|'nin|'nun|'nün)\b",
        r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]{1,40})"
        r"\s+isimli\b",
        r"\bnamed\s+([A-Z][A-Za-z'-]{1,40})\b",
        r"\bwhich\s+(?:projects|publications|activities|news)"
        r"\s+(?:is|did)\s+([A-Z][A-Za-z'-]{1,40})\b",
        r"\bwhat\s+is\s+([A-Z][A-Za-z'-]{1,40})'s\b",
    )

    for pattern in patterns:
        match = re.search(pattern, question)

        if match:
            return match.group(1).strip()

    return None


def extract_named_subject(
    question: str,
    entity: str,
) -> str | None:
    quoted = extract_quoted_text(question)

    if quoted:
        return quoted

    person = extract_person_name(question)

    if person and entity in {
        "STUDENT",
        "ACADEMICIAN",
        "STUDENT_PROFILE",
    }:
        return person

    patterns: dict[str, tuple[str, ...]] = {
        "PROJECT": (
            r"(.{2,80}?)\s+projesi(?:nin|nde|ne|ni)?\b",
            r"(?:project\s+named|details\s+of\s+the)\s+(.{2,80}?)"
            r"\s+project\b",
        ),
        "PUBLICATION": (
            r"(.{2,80}?)\s+(?:başlıklı\s+)?yayını\b",
            r"publication\s+(?:named|titled)\s+(.{2,80})",
        ),
        "NEWS": (
            r"(.{2,80}?)\s+(?:başlıklı\s+)?haberi\b",
            r"news\s+(?:post\s+)?(?:named|titled)\s+(.{2,80})",
        ),
    }

    for pattern in patterns.get(entity, ()):
        match = re.search(
            pattern,
            question,
            flags=re.IGNORECASE,
        )

        if match:
            candidate = match.group(1).strip(
                " ?.,:;\"'"
            )

            candidate = re.sub(
                r"^(?:bu|şu|the)\s+",
                "",
                candidate,
                flags=re.IGNORECASE,
            )

            return candidate

    return None


def detect_relation_section(
    normalized: str,
) -> str | None:
    if contains_any(
        normalized,
        ("proje", "project"),
    ):
        return "projects"

    if contains_any(
        normalized,
        ("yayin", "publication", "paper"),
    ):
        return "publications"

    if contains_any(
        normalized,
        ("haber", "news"),
    ):
        return "news"

    if contains_any(
        normalized,
        ("etkinlik", "faaliyet", "activity"),
    ):
        return "activities"

    return None


def detect_application_entity(
    normalized: str,
) -> str:
    if contains_any(
        normalized,
        ("akademisyen", "academician", "academic member"),
    ):
        return "ACADEMICIAN"

    if contains_any(
        normalized,
        ("yayin", "publication", "makale", "paper", "doi"),
    ):
        return "PUBLICATION"

    if contains_any(
        normalized,
        ("haber", "news", "duyuru", "announcement"),
    ):
        return "NEWS"

    if contains_any(
        normalized,
        ("etkinlik", "faaliyet", "activity"),
    ):
        return "ACTIVITY"

    if contains_any(
        normalized,
        ("proje", "project"),
    ):
        return "PROJECT"

    if contains_any(
        normalized,
        (
            "iletisim mesaji",
            "contact message",
            "mesaj",
        ),
    ):
        return "CONTACT_MESSAGE"

    return "STUDENT"


def detect_application_operation(
    normalized: str,
) -> str:
    if contains_any(
        normalized,
        (
            "karsilastir",
            "compare",
            "farki",
            "arasindaki fark",
        ),
    ):
        return "COMPARE"

    if contains_any(
        normalized,
        (
            "gore say",
            "gruplandir",
            "dagilimi",
            "group by",
            "grouped by",
            "distribution",
        ),
    ):
        return "GROUP"

    if contains_any(
        normalized,
        (
            "kac",
            "sayisi",
            "toplam",
            "how many",
            "count",
        ),
    ):
        return "COUNT"

    if contains_any(
        normalized,
        (
            "ara",
            "bul",
            "ilgili",
            "iliskili",
            "search",
            "find",
            "related to",
        ),
    ):
        return "SEARCH"

    if contains_any(
        normalized,
        (
            "ayrinti",
            "detay",
            "nedir",
            "kimdir",
            "hangi bolum",
            "hangi gorev",
            "details",
            "detail",
            "what is",
            "who is",
        ),
    ):
        return "GET"

    return "LIST"


def detect_group_by(
    entity: str,
    normalized: str,
) -> str | None:
    if entity in {"STUDENT", "ACADEMICIAN"}:
        return (
            "memberType"
            if contains_any(
                normalized,
                ("uye turu", "member type"),
            )
            else "department"
        )

    if entity == "PROJECT":
        return (
            "technologies"
            if contains_any(
                normalized,
                ("teknoloji", "technology"),
            )
            else "status"
        )

    if entity == "PUBLICATION":
        return (
            "publicationYear"
            if contains_any(
                normalized,
                ("yil", "year"),
            )
            else "type"
        )

    if entity == "NEWS":
        return (
            "contentStatus"
            if contains_any(
                normalized,
                ("taslak", "published", "draft", "durum"),
            )
            else "category"
        )

    return None


def detect_application_filters(
    entity: str,
    normalized: str,
) -> dict[str, Any]:
    filters: dict[str, Any] = {}

    if contains_any(
        normalized,
        ("aktif", "active"),
    ) and not contains_any(
        normalized,
        ("pasif", "inactive"),
    ):
        filters["active"] = True

    if contains_any(
        normalized,
        ("pasif", "inactive"),
    ):
        filters["active"] = False

    if contains_any(
        normalized,
        ("one cikan", "featured"),
    ):
        filters["featured"] = True

    status_terms = {
        "PLANNED": (
            "planlanan",
            "planlanmis",
            "planned",
        ),
        "IN_PROGRESS": (
            "devam eden",
            "yurutulen",
            "in progress",
            "ongoing",
        ),
        "COMPLETED": (
            "tamamlanan",
            "tamamlanmis",
            "completed",
        ),
        "ON_HOLD": (
            "beklemeye alinan",
            "on hold",
        ),
        "CANCELLED": (
            "iptal edilen",
            "cancelled",
            "canceled",
        ),
    }

    if entity == "PROJECT":
        for status, terms in status_terms.items():
            if contains_any(normalized, terms):
                filters["status"] = status
                break

    publication_types = {
        "JOURNAL_ARTICLE": (
            "dergi makalesi",
            "journal article",
        ),
        "CONFERENCE_PAPER": (
            "konferans bildirisi",
            "conference paper",
        ),
        "BOOK_CHAPTER": (
            "kitap bolumu",
            "book chapter",
        ),
        "THESIS": (
            "tez",
            "thesis",
        ),
        "TECHNICAL_REPORT": (
            "teknik rapor",
            "technical report",
        ),
    }

    if entity == "PUBLICATION":
        for publication_type, terms in publication_types.items():
            if contains_any(normalized, terms):
                filters["type"] = publication_type
                break

        year_match = re.search(
            r"\b(20\d{2})\b",
            normalized,
        )

        if year_match:
            filters["year"] = int(
                year_match.group(1)
            )

    news_categories = {
        "INTERNSHIP": (
            "staj",
            "internship",
        ),
        "EVENT": (
            "etkinlik",
            "event",
        ),
        "ANNOUNCEMENT": (
            "duyuru",
            "announcement",
        ),
        "PROJECT_UPDATE": (
            "proje guncelleme",
            "project update",
        ),
        "PUBLICATION": (
            "yayin",
            "publication",
        ),
        "STUDENT_ACTIVITY": (
            "ogrenci etkinligi",
            "student activity",
        ),
    }

    if entity == "NEWS":
        for category, terms in news_categories.items():
            if contains_any(normalized, terms):
                filters["category"] = category
                break

    return filters


def build_application_request(
    question: str,
    language: str,
) -> ApplicationDataRequest:
    normalized = normalize_for_routing(question)
    entity = detect_application_entity(
        normalized
    )

    person_name = extract_person_name(
        question
    )

    relation_section = detect_relation_section(
        normalized
    )

    relation_requested = (
        person_name is not None
        and relation_section is not None
        and contains_any(
            normalized,
            (
                "hangi",
                "katki",
                "calis",
                "yer al",
                "contribut",
                "working",
                "worked",
            ),
        )
    )

    if relation_requested:
        return ApplicationDataRequest(
            entity="STUDENT_PROFILE",
            operation="GET",
            language=language,
            search_text=person_name,
            filters={
                "studentName": person_name,
                "relationSection": relation_section,
            },
            limit=20,
        )

    operation = detect_application_operation(
        normalized
    )

    search_text = extract_named_subject(
        question,
        entity,
    )

    if operation == "SEARCH" and not search_text:
        patterns = (
            r"(?:ilgili|ilişkili)\s+(.{2,80})",
            r"related\s+to\s+(.{2,80})",
            r"(?:ara|bul|find|search)\s+(.{2,80})",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                question,
                flags=re.IGNORECASE,
            )

            if match:
                search_text = match.group(1).strip(
                    " ?.,:;\"'"
                )
                break

    return ApplicationDataRequest(
        entity=entity,
        operation=operation,
        language=language,
        search_text=search_text,
        filters=detect_application_filters(
            entity,
            normalized,
        ),
        group_by=detect_group_by(
            entity,
            normalized,
        ),
        limit=20,
    )


def item_label(
    entity: str,
    item: dict[str, Any],
) -> str:
    if entity in {
        "STUDENT",
        "ACADEMICIAN",
    }:
        return str(
            item.get("fullName")
            or "İsimsiz kayıt"
        )

    if entity == "PROJECT":
        return str(
            item.get("name")
            or "İsimsiz proje"
        )

    if entity == "PUBLICATION":
        return str(
            item.get("title")
            or "Başlıksız yayın"
        )

    if entity == "NEWS":
        return str(
            item.get("title")
            or "Başlıksız haber"
        )

    if entity == "ACTIVITY":
        return str(
            item.get("title")
            or "Başlıksız etkinlik"
        )

    return str(
        item.get("id")
        or "Kayıt"
    )


def format_profile_answer(
    result: dict[str, Any],
    request: ApplicationDataRequest,
    language: str,
) -> str:
    if not result.get("resolved"):
        candidates = result.get(
            "candidates"
        ) or []

        names = ", ".join(
            str(
                candidate.get("fullName")
                or candidate.get("id")
            )
            for candidate in candidates[:10]
        )

        if language == "en":
            return (
                "The student could not be resolved uniquely."
                + (
                    f" Candidates: {names}."
                    if names
                    else ""
                )
            )

        return (
            "Öğrenci tek bir kayıt olarak çözümlenemedi."
            + (
                f" Adaylar: {names}."
                if names
                else ""
            )
        )

    profile = result.get("profile") or {}
    student = profile.get("student") or {}
    full_name = student.get(
        "fullName"
    ) or "Unknown"

    filters = request.filters or {}
    section = filters.get(
        "relationSection"
    )

    if section:
        items = profile.get(section) or []

        labels = ", ".join(
            item_label(
                {
                    "projects": "PROJECT",
                    "publications": "PUBLICATION",
                    "news": "NEWS",
                    "activities": "ACTIVITY",
                }[section],
                item,
            )
            for item in items[:20]
        )

        if language == "en":
            return (
                f"{full_name} has {len(items)} records in "
                f"{section}: {labels or 'none'}."
            )

        section_names = {
            "projects": "proje",
            "publications": "yayın",
            "news": "haber",
            "activities": "etkinlik",
        }

        return (
            f"{full_name} için {len(items)} "
            f"{section_names[section]} kaydı bulundu: "
            f"{labels or 'yok'}."
        )

    statistics = profile.get(
        "statistics"
    ) or {}

    if language == "en":
        return (
            f"{full_name}: department={student.get('department') or 'unknown'}, "
            f"current task={student.get('currentTask') or 'not specified'}, "
            f"member type={student.get('memberType') or 'STUDENT'}. "
            f"Projects: {statistics.get('projectCount', 0)}; "
            f"publications: {statistics.get('publicationCount', 0)}; "
            f"news: {statistics.get('newsCount', 0)}; "
            f"activities: {statistics.get('activityCount', 0)}."
        )

    return (
        f"{full_name}: bölüm={student.get('department') or 'belirtilmemiş'}, "
        f"mevcut görev={student.get('currentTask') or 'belirtilmemiş'}, "
        f"üye türü={student.get('memberType') or 'STUDENT'}. "
        f"Proje: {statistics.get('projectCount', 0)}; "
        f"yayın: {statistics.get('publicationCount', 0)}; "
        f"haber: {statistics.get('newsCount', 0)}; "
        f"etkinlik: {statistics.get('activityCount', 0)}."
    )


def format_application_answer(
    result: dict[str, Any],
    request: ApplicationDataRequest,
    language: str,
) -> str:
    entity = request.entity.upper()
    operation = request.operation.upper()

    if entity == "STUDENT_PROFILE":
        return format_profile_answer(
            result,
            request,
            language,
        )

    if result.get("available") is False:
        reason = result.get("reason")

        if language == "en":
            return (
                "This application data source is not currently "
                f"available through a verified read-only endpoint. {reason}"
            )

        return (
            "Bu uygulama verisi için doğrulanmış read-only endpoint "
            f"henüz mevcut değil. {reason}"
        )

    if operation == "COUNT":
        count = int(
            result.get("count", 0)
        )

        if language == "en":
            return (
                f"The system contains {count} matching "
                f"{entity.lower()} records."
            )

        labels = {
            "STUDENT": "öğrenci",
            "ACADEMICIAN": "akademisyen",
            "PROJECT": "proje",
            "PUBLICATION": "yayın",
            "NEWS": "haber",
            "ACTIVITY": "etkinlik",
        }

        return (
            f"Sistemde filtrelere uyan {count} "
            f"{labels.get(entity, 'kayıt')} bulunuyor."
        )

    if operation in {
        "GROUP",
        "COMPARE",
    }:
        groups = result.get("groups") or {}

        group_text = ", ".join(
            f"{key}: {value}"
            for key, value in groups.items()
        )

        if language == "en":
            return (
                f"Grouped result for {entity.lower()}: "
                f"{group_text or 'no records'}."
            )

        return (
            f"{entity} için gruplandırılmış sonuç: "
            f"{group_text or 'kayıt yok'}."
        )

    if operation == "GET":
        item = result.get("item")

        if not item:
            candidates = result.get(
                "candidates"
            ) or []

            labels = ", ".join(
                item_label(entity, candidate)
                for candidate in candidates[:10]
            )

            if language == "en":
                return (
                    "No unique record was found."
                    + (
                        f" Candidates: {labels}."
                        if labels
                        else ""
                    )
                )

            return (
                "Tek bir kayıt bulunamadı."
                + (
                    f" Aday kayıtlar: {labels}."
                    if labels
                    else ""
                )
            )

        label = item_label(
            entity,
            item,
        )

        details = [
            f"{key}={value}"
            for key, value in item.items()
            if (
                value is not None
                and key not in {
                    "id",
                    "studentIds",
                    "content",
                    "description",
                    "abstractText",
                }
            )
        ]

        return (
            f"{label}: "
            + ", ".join(details[:12])
        )

    items = result.get("items") or []
    labels = ", ".join(
        item_label(entity, item)
        for item in items
    )

    total = int(
        result.get(
            "total",
            len(items),
        )
    )

    if language == "en":
        return (
            f"{total} matching records were found. "
            f"Returned records: {labels or 'none'}."
        )

    return (
        f"Toplam {total} eşleşen kayıt bulundu. "
        f"Dönen kayıtlar: {labels or 'yok'}."
    )


def execute_application_question(
    question: str,
    language: str,
) -> AssistantToolAnswer:
    request = build_application_request(
        question,
        language,
    )

    try:
        result = execute_application_data_request(
            request
        )
    except ApplicationDataToolError as exception:
        answer = (
            "The application data service could not be reached: "
            f"{exception}"
            if language == "en"
            else
            "Uygulama veri servisine ulaşılamadı: "
            f"{exception}"
        )

        return AssistantToolAnswer(
            answer=answer,
            tools_used=(
                "application_data_router",
                "spring_boot_application_api",
            ),
            grounded=False,
        )

    total = result.get(
        "count",
        result.get(
            "total",
        ),
    )

    return AssistantToolAnswer(
        answer=format_application_answer(
            result,
            request,
            language,
        ),
        tools_used=(
            "application_data_router",
            "spring_boot_application_api",
            "application_data_privacy_filter",
            "deterministic_application_answer",
        ),
        grounded=True,
        filtered_total=(
            int(total)
            if total is not None
            else None
        ),
        raw_result=result,
    )


def infer_health_component(
    question: str,
) -> str:
    normalized = normalize_for_routing(
        question
    )

    if contains_any(
        normalized,
        ("mongodb", "mongo db", "mongo"),
    ):
        return "MONGODB"

    if contains_any(
        normalized,
        ("qdrant", "vector database", "vektor veritabani"),
    ):
        return "QDRANT"

    if contains_any(
        normalized,
        ("qwen", "model cevap", "inference", "llm"),
    ):
        return "QWEN"

    if contains_any(
        normalized,
        ("fastapi", "dell servisi", "dell service"),
    ):
        return "FASTAPI"

    if contains_any(
        normalized,
        ("jvm", "java bellek", "memory", "bellek"),
    ):
        return "JVM"

    if contains_any(
        normalized,
        ("disk", "storage", "upload alani", "yukleme alani"),
    ):
        return "STORAGE"

    if contains_any(
        normalized,
        (
            "proxy",
            "bahtiyar ile dell",
            "bahtiyar-dell",
            "asistan baglantisi",
        ),
    ):
        return "ASSISTANT_PROXY"

    if contains_any(
        normalized,
        ("spring boot", "backend", "uygulama servisi"),
    ):
        return "SPRING_BOOT"

    return "ALL"


def local_fastapi_health() -> dict[str, Any]:
    return {
        "component": "FASTAPI",
        "status": "UP",
        "checkedAt": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def local_qdrant_health(
    app_state: Any,
    vector_db_provider: str,
    collection_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        if vector_db_provider == "qdrant":
            info = app_state.qdrant.get_collection(
                collection_name
            )

            points_count = int(
                getattr(
                    info,
                    "points_count",
                    0,
                )
                or 0
            )
        else:
            info = app_state.upstash.info()
            namespace = info.namespaces.get(
                collection_name
            )

            points_count = int(
                getattr(
                    namespace,
                    "vector_count",
                    0,
                )
                or 0
            )

        return {
            "component": "QDRANT"
            if vector_db_provider == "qdrant"
            else "VECTOR_STORE",
            "status": "UP",
            "provider": vector_db_provider,
            "collection": collection_name,
            "pointsCount": points_count,
            "latencyMs": round(
                (
                    time.perf_counter()
                    - started
                ) * 1000,
                2,
            ),
            "checkedAt": datetime.now(
                timezone.utc
            ).isoformat(),
        }
    except Exception as exception:
        return {
            "component": "QDRANT",
            "status": "DOWN",
            "errorType": type(
                exception
            ).__name__,
            "reason": str(exception)[:300],
            "latencyMs": round(
                (
                    time.perf_counter()
                    - started
                ) * 1000,
                2,
            ),
            "checkedAt": datetime.now(
                timezone.utc
            ).isoformat(),
        }


def local_qwen_health(
    app_state: Any,
    model_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        with app_state.llm_lock:
            completion = (
                app_state.llm
                .create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Reply with exactly OK."
                            ),
                        },
                        {
                            "role": "user",
                            "content": "health probe",
                        },
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=4,
                )
            )

        choices = completion.get(
            "choices"
        ) or []

        content = ""

        if choices:
            content = str(
                (
                    choices[0].get(
                        "message"
                    )
                    or {}
                ).get(
                    "content"
                )
                or ""
            ).strip()

        successful = bool(
            choices
            and content
        )

        return {
            "component": "QWEN",
            "status": (
                "UP"
                if successful
                else "DOWN"
            ),
            "model": model_name,
            "probeResponse": content[:50],
            "latencyMs": round(
                (
                    time.perf_counter()
                    - started
                ) * 1000,
                2,
            ),
            "checkedAt": datetime.now(
                timezone.utc
            ).isoformat(),
        }
    except Exception as exception:
        return {
            "component": "QWEN",
            "status": "DOWN",
            "model": model_name,
            "errorType": type(
                exception
            ).__name__,
            "reason": str(exception)[:300],
            "latencyMs": round(
                (
                    time.perf_counter()
                    - started
                ) * 1000,
                2,
            ),
            "checkedAt": datetime.now(
                timezone.utc
            ).isoformat(),
        }


def execute_health_component(
    component: str,
    app_state: Any,
    language: str,
    vector_db_provider: str,
    collection_name: str,
    model_name: str,
) -> AssistantToolAnswer:
    normalized_component = (
        component
        or "ALL"
    ).upper()

    try:
        if normalized_component in {
            "MONGODB",
            "SPRING_BOOT",
            "JVM",
            "STORAGE",
            "ASSISTANT_PROXY",
        }:
            result = get_backend_health(
                normalized_component
            )

        elif normalized_component == "FASTAPI":
            result = local_fastapi_health()

        elif normalized_component == "QDRANT":
            result = local_qdrant_health(
                app_state,
                vector_db_provider,
                collection_name,
            )

        elif normalized_component == "QWEN":
            result = local_qwen_health(
                app_state,
                model_name,
            )

        else:
            backend = get_backend_health(
                "ALL"
            )

            result = {
                "component": "ALL",
                "status": "UP",
                "checkedAt": datetime.now(
                    timezone.utc
                ).isoformat(),
                "components": {
                    "BACKEND": backend,
                    "FASTAPI": local_fastapi_health(),
                    "QDRANT": local_qdrant_health(
                        app_state,
                        vector_db_provider,
                        collection_name,
                    ),
                    "QWEN": local_qwen_health(
                        app_state,
                        model_name,
                    ),
                },
            }

            component_statuses = [
                str(
                    value.get(
                        "status",
                        "DOWN",
                    )
                ).upper()
                for value in result[
                    "components"
                ].values()
            ]

            if "DOWN" in component_statuses:
                result["status"] = "DOWN"
            elif (
                "WARN" in component_statuses
                or "DEGRADED"
                in component_statuses
            ):
                result["status"] = "WARN"

    except BackendHealthToolError as exception:
        result = {
            "component": normalized_component,
            "status": "DOWN",
            "reason": str(exception),
            "checkedAt": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    answer = format_health_answer(
        result,
        language,
    )

    return AssistantToolAnswer(
        answer=answer,
        tools_used=(
            "system_health_router",
            "live_system_health_tool",
            "deterministic_health_answer",
        ),
        grounded=True,
        raw_result=result,
    )


def execute_system_health_question(
    question: str,
    app_state: Any,
    language: str,
    vector_db_provider: str,
    collection_name: str,
    model_name: str,
) -> AssistantToolAnswer:
    return execute_health_component(
        component=infer_health_component(
            question
        ),
        app_state=app_state,
        language=language,
        vector_db_provider=vector_db_provider,
        collection_name=collection_name,
        model_name=model_name,
    )


def flatten_health_components(
    result: dict[str, Any],
) -> list[tuple[str, str]]:
    components = result.get(
        "components"
    )

    if not isinstance(
        components,
        dict,
    ):
        return [
            (
                str(
                    result.get(
                        "component",
                        "UNKNOWN",
                    )
                ),
                str(
                    result.get(
                        "status",
                        "UNKNOWN",
                    )
                ),
            )
        ]

    values: list[tuple[str, str]] = []

    for component, data in components.items():
        if isinstance(data, dict):
            values.append(
                (
                    str(component),
                    str(
                        data.get(
                            "status",
                            "UNKNOWN",
                        )
                    ),
                )
            )

    return values


def format_health_answer(
    result: dict[str, Any],
    language: str,
) -> str:
    component = str(
        result.get(
            "component",
            "UNKNOWN",
        )
    )

    status = str(
        result.get(
            "status",
            "UNKNOWN",
        )
    )

    latency = result.get(
        "latencyMs"
    )

    checked_at = result.get(
        "checkedAt"
    )

    if component == "ALL":
        components = flatten_health_components(
            result
        )

        component_text = ", ".join(
            f"{name}={value}"
            for name, value in components
        )

        if language == "en":
            return (
                f"Overall system health is {status}. "
                f"Components: {component_text}. "
                f"Checked at: {checked_at}."
            )

        return (
            f"Genel sistem sağlık durumu {status}. "
            f"Bileşenler: {component_text}. "
            f"Kontrol zamanı: {checked_at}."
        )

    details: list[str] = []

    if latency is not None:
        details.append(
            f"latency={latency} ms"
        )

    if result.get(
        "pointsCount"
    ) is not None:
        details.append(
            "record count="
            f"{result.get('pointsCount')}"
        )

    if result.get(
        "usedPercent"
    ) is not None:
        details.append(
            "usage="
            f"{result.get('usedPercent')}%"
        )

    if result.get(
        "usablePercent"
    ) is not None:
        details.append(
            "usable disk="
            f"{result.get('usablePercent')}%"
        )

    if result.get("reason"):
        details.append(
            f"reason={result.get('reason')}"
        )

    detail_text = "; ".join(details)

    if language == "en":
        return (
            f"{component} live health status is {status}."
            + (
                f" {detail_text}."
                if detail_text
                else ""
            )
            + (
                f" Checked at: {checked_at}."
                if checked_at
                else ""
            )
        )

    return (
        f"{component} canlı sağlık durumu {status}."
        + (
            f" {detail_text}."
            if detail_text
            else ""
        )
        + (
            f" Kontrol zamanı: {checked_at}."
            if checked_at
            else ""
        )
    )


def execute_system_info_question(
    language: str,
    model_name: str,
    vector_db_provider: str,
    collection_name: str,
) -> AssistantToolAnswer:
    try:
        result = get_backend_system_info()
    except BackendHealthToolError as exception:
        answer = (
            f"System information could not be read: {exception}"
            if language == "en"
            else
            f"Sistem bilgisi okunamadı: {exception}"
        )

        return AssistantToolAnswer(
            answer=answer,
            tools_used=(
                "system_info_tool",
            ),
            grounded=False,
        )

    result["qwenModel"] = model_name
    result["vectorDbProvider"] = (
        vector_db_provider
    )
    result["vectorCollection"] = (
        collection_name
    )

    if language == "en":
        answer = (
            f"Application: {result.get('application')}. "
            f"Java: {result.get('javaVersion')}; "
            f"OS: {result.get('operatingSystem')} "
            f"{result.get('osArchitecture')}; "
            f"uptime: {result.get('uptimeMs')} ms; "
            f"processors: {result.get('availableProcessors')}; "
            f"Qwen model: {model_name}; "
            f"vector store: {vector_db_provider}/"
            f"{collection_name}."
        )
    else:
        answer = (
            f"Uygulama: {result.get('application')}. "
            f"Java: {result.get('javaVersion')}; "
            f"işletim sistemi: {result.get('operatingSystem')} "
            f"{result.get('osArchitecture')}; "
            f"çalışma süresi: {result.get('uptimeMs')} ms; "
            f"işlemci: {result.get('availableProcessors')}; "
            f"Qwen modeli: {model_name}; "
            f"vektör deposu: {vector_db_provider}/"
            f"{collection_name}."
        )

    return AssistantToolAnswer(
        answer=answer,
        tools_used=(
            "system_info_tool",
            "spring_boot_system_info",
        ),
        grounded=True,
        raw_result=result,
    )
