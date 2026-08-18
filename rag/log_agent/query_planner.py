from __future__ import annotations

import unicodedata

import re
from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo


class QueryIntent(str, Enum):
    IDENTITY = "IDENTITY"
    GREETING = "GREETING"
    SMALL_TALK = "SMALL_TALK"
    CAPABILITIES = "CAPABILITIES"
    APPLICATION_OVERVIEW = "APPLICATION_OVERVIEW"
    APPLICATION_DATA = "APPLICATION_DATA"
    SYSTEM_HEALTH = "SYSTEM_HEALTH"
    SYSTEM_INFO = "SYSTEM_INFO"

    AUDIT_IP_SUMMARY = "AUDIT_IP_SUMMARY"
    AUTH_LOGIN_COUNT = "AUTH_LOGIN_COUNT"
    REQUEST_COUNT = "REQUEST_COUNT"
    REQUEST_ENDPOINT_SUMMARY = "REQUEST_ENDPOINT_SUMMARY"

    DAILY_SUMMARY = "DAILY_SUMMARY"
    ERROR_COUNT = "ERROR_COUNT"
    RECURRING_ERROR_SUMMARY = "RECURRING_ERROR_SUMMARY"

    AUDIT_DELETE_LIST = "AUDIT_DELETE_LIST"
    AUDIT_CREATE_LIST = "AUDIT_CREATE_LIST"
    AUDIT_UPDATE_LIST = "AUDIT_UPDATE_LIST"

    TIME_WINDOW_SEARCH = "TIME_WINDOW_SEARCH"
    EXACT_IDENTIFIER_SEARCH = "EXACT_IDENTIFIER_SEARCH"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"


class RetrievalMode(str, Enum):
    AGGREGATE = "AGGREGATE"
    FILTERED_LIST = "FILTERED_LIST"
    HYBRID = "HYBRID"


class TimeScope(str, Enum):
    TODAY = "TODAY"
    YESTERDAY = "YESTERDAY"
    RELATIVE = "RELATIVE"
    CLOCK = "CLOCK"


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    normalized_query: str

    detected_language: str
    intent: QueryIntent
    retrieval_mode: RetrievalMode
    time_scope: TimeScope | None

    timezone: str
    from_time_utc: datetime | None
    to_time_utc: datetime | None

    source_types: tuple[str, ...]
    actions: tuple[str, ...]
    entity_types: tuple[str, ...]
    levels: tuple[str, ...]

    exact_terms: tuple[str, ...]
    requires_aggregation: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "originalQuery": self.original_query,
            "normalizedQuery": self.normalized_query,
            "detectedLanguage": self.detected_language,
            "intent": self.intent.value,
            "retrievalMode": self.retrieval_mode.value,
            "timeScope": (
                self.time_scope.value
                if self.time_scope
                else None
            ),
            "timezone": self.timezone,
            "fromTimeUtc": format_datetime(
                self.from_time_utc
            ),
            "toTimeUtc": format_datetime(
                self.to_time_utc
            ),
            "sourceTypes": list(self.source_types),
            "actions": list(self.actions),
            "entityTypes": list(self.entity_types),
            "levels": list(self.levels),
            "exactTerms": list(self.exact_terms),
            "requiresAggregation": (
                self.requires_aggregation
            ),
        }


NUMBER_WORDS = {
    # Türkçe
    "bir": 1,
    "iki": 2,
    "üç": 3,
    "uc": 3,
    "dört": 4,
    "dort": 4,
    "beş": 5,
    "bes": 5,
    "altı": 6,
    "alti": 6,
    "yedi": 7,
    "sekiz": 8,
    "dokuz": 9,
    "on": 10,

    # İngilizce
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


ACTION_KEYWORDS = {
    "DELETE": (
        "silindi",
        "silinen",
        "sildi",
        "silme",
        "kaldırıldı",
        "kaldirildi",
        "delete",
        "deleted",
        "deletion",
        "removed",
    ),
    "CREATE": (
        "oluşturuldu",
        "olusturuldu",
        "oluşturulan",
        "olusturulan",
        "eklendi",
        "eklenen",
        "create",
        "created",
        "creation",
        "added",
    ),
    "UPDATE": (
        "güncellendi",
        "guncellendi",
        "güncellenen",
        "guncellenen",
        "değiştirildi",
        "degistirildi",
        "update",
        "updated",
        "modified",
        "changed",
    ),
}


ENTITY_KEYWORDS = {
    "NEWS": (
        "haber",
        "news",
    ),
    "PROJECT": (
        "proje",
        "project",
    ),
    "PUBLICATION": (
        "yayın",
        "yayin",
        "publication",
    ),
    "STUDENT": (
        "öğrenci",
        "ogrenci",
        "student",
    ),
    "ACADEMICIAN": (
        "akademisyen",
        "academician",
        "academic",
    ),
    "ACTIVITY": (
        "aktivite",
        "etkinlik",
        "activity",
        "event",
    ),
    "CONTACT_MESSAGE": (
        "iletişim mesajı",
        "iletisim mesaji",
        "contact message",
    ),
}


SUMMARY_TERMS = (
    # Türkçe
    "neler oldu",
    "ne oldu",
    "özetle",
    "ozetle",
    "özet",
    "ozet",
    "raporla",
    "raporu",
    "gelişmeler",
    "gelismeler",

    # İngilizce
    "what happened",
    "what has happened",
    "summarize",
    "summary",
    "report",
    "developments",
    "overview",
)


COUNT_TERMS = (
    # Türkçe
    "kaç",
    "sayısı",
    "sayisi",
    "toplam",
    "adet",

    # İngilizce
    "how many",
    "count",
    "number of",
    "total",
)


REPEAT_TERMS = (
    # Türkçe
    "tekrar eden",
    "tekrarlayan",
    "yinelenen",
    "aynı hata",
    "ayni hata",

    # İngilizce
    "repeated",
    "recurring",
    "recurrence",
    "same error",
)


ERROR_TERMS = (
    # Türkçe
    "hata",
    "hatalar",
    "başarısız",
    "basarisiz",

    # İngilizce / teknik
    "error",
    "errors",
    "exception",
    "exceptions",
    "failure",
    "failures",
    "failed",
)


WARNING_TERMS = (
    "uyarı",
    "uyari",
    "warning",
    "warnings",
    "warn",
)


TURKISH_LANGUAGE_HINTS = (
    "bugün",
    "bugun",
    "dün",
    "dun",
    "neler",
    "hangi",
    "kaç",
    "saatte",
    "dakikada",
    "oluştu",
    "olustu",
    "silindi",
    "güncellendi",
    "guncellendi",
    "açıkla",
    "acikla",
    " ve ",
)


ENGLISH_LANGUAGE_HINTS = (
    "today",
    "yesterday",
    "what",
    "which",
    "how many",
    "last",
    "past",
    "hour",
    "hours",
    "minute",
    "minutes",
    "happened",
    "occurred",
    "deleted",
    "updated",
    "created",
    "explain",
    "show",
)


def format_datetime(
    value: datetime | None,
) -> str | None:
    if value is None:
        return None

    utc_value = value.astimezone(timezone.utc)

    return utc_value.isoformat().replace(
        "+00:00",
        "Z",
    )


def normalize_query(question: str) -> str:
    value = question.strip().casefold()

    value = value.replace("’", "'")
    value = value.replace("`", "'")

    return re.sub(r"\s+", " ", value)


def contains_any(
    text: str,
    values: tuple[str, ...],
) -> bool:
    return any(value in text for value in values)


def detect_language(
    question: str,
    normalized_query: str,
    requested_language: str | None = None,
) -> str:
    if requested_language:
        normalized_language = (
            requested_language.strip().casefold()
        )

        if normalized_language in {
            "en",
            "en-us",
            "en-gb",
            "english",
        }:
            return "en"

        if normalized_language in {
            "tr",
            "tr-tr",
            "turkish",
            "türkçe",
            "turkce",
        }:
            return "tr"

    turkish_score = sum(
        1
        for hint in TURKISH_LANGUAGE_HINTS
        if hint in normalized_query
    )

    english_score = sum(
        1
        for hint in ENGLISH_LANGUAGE_HINTS
        if hint in normalized_query
    )

    turkish_score += sum(
        1
        for character in question.casefold()
        if character in "çğıöşü"
    )

    if english_score > turkish_score:
        return "en"

    return "tr"


def parse_number(value: str) -> int:
    normalized = value.casefold()

    if normalized.isdigit():
        return int(normalized)

    result = NUMBER_WORDS.get(normalized)

    if result is None:
        raise ValueError(
            f"Desteklenmeyen sayı ifadesi: {value}"
        )

    return result


def local_day_range(
    day: datetime,
    timezone_info: ZoneInfo,
) -> tuple[datetime, datetime]:
    start = datetime.combine(
        day.date(),
        time.min,
        tzinfo=timezone_info,
    )

    end = datetime.combine(
        day.date(),
        time.max,
        tzinfo=timezone_info,
    )

    return (
        start.astimezone(timezone.utc),
        end.astimezone(timezone.utc),
    )


def parse_time_range(
    normalized_query: str,
    now_local: datetime,
    timezone_info: ZoneInfo,
) -> tuple[
    datetime | None,
    datetime | None,
    TimeScope | None,
]:
    clock_patterns = (
        r"\bsaat\s*"
        r"(?P<hour>\d{1,2})"
        r"(?:[.:](?P<minute>\d{2}))?",

        r"\b(?:around|at)\s+"
        r"(?P<hour>\d{1,2})"
        r"(?:[.:](?P<minute>\d{2}))?",
    )

    clock_match = None

    for pattern in clock_patterns:
        clock_match = re.search(
            pattern,
            normalized_query,
        )

        if clock_match:
            break

    if clock_match:
        hour = int(clock_match.group("hour"))
        minute = int(
            clock_match.group("minute") or "0"
        )

        if hour > 23 or minute > 59:
            raise ValueError(
                "Geçersiz saat ifadesi."
            )

        target_day = now_local

        if (
            "dün" in normalized_query
            or "dun" in normalized_query
            or "yesterday" in normalized_query
        ):
            target_day = (
                now_local - timedelta(days=1)
            )

        target = datetime.combine(
            target_day.date(),
            time(hour=hour, minute=minute),
            tzinfo=timezone_info,
        )

        start = target - timedelta(minutes=30)
        end = target + timedelta(minutes=30)

        return (
            start.astimezone(timezone.utc),
            end.astimezone(timezone.utc),
            TimeScope.CLOCK,
        )

    number_pattern = (
        r"\d+|"
        r"bir|iki|üç|uc|dört|dort|"
        r"beş|bes|altı|alti|yedi|sekiz|"
        r"dokuz|on|"
        r"one|two|three|four|five|six|"
        r"seven|eight|nine|ten"
    )

    unit_pattern = (
        r"dakika|saat|gün|gun|"
        r"minute|minutes|"
        r"hour|hours|"
        r"day|days"
    )

    # Türkçe hâl ekleri:
    # dakikada, saatte, günde
    turkish_case_suffix = r"(?:da|de|ta|te)?"

    relative_match = re.search(
        rf"\b(?:son|geçen|last|past)\s+"
        rf"(?P<number>{number_pattern})\s+"
        rf"(?P<unit>{unit_pattern})"
        rf"{turkish_case_suffix}\b",
        normalized_query,
    )

    if relative_match:
        amount = parse_number(
            relative_match.group("number")
        )

        unit = relative_match.group("unit")

        if unit in {
            "dakika",
            "minute",
            "minutes",
        }:
            delta = timedelta(minutes=amount)
        elif unit in {
            "saat",
            "hour",
            "hours",
        }:
            delta = timedelta(hours=amount)
        else:
            delta = timedelta(days=amount)

        return (
            (
                now_local - delta
            ).astimezone(timezone.utc),
            now_local.astimezone(timezone.utc),
            TimeScope.RELATIVE,
        )

    implicit_relative_match = re.search(
        rf"\b(?:son|geçen|last|past)\s+"
        rf"(?P<unit>{unit_pattern})"
        rf"{turkish_case_suffix}\b",
        normalized_query,
    )

    if implicit_relative_match:
        unit = implicit_relative_match.group("unit")

        if unit in {
            "dakika",
            "minute",
            "minutes",
        }:
            delta = timedelta(minutes=1)
        elif unit in {
            "saat",
            "hour",
            "hours",
        }:
            delta = timedelta(hours=1)
        else:
            delta = timedelta(days=1)

        return (
            (
                now_local - delta
            ).astimezone(timezone.utc),
            now_local.astimezone(timezone.utc),
            TimeScope.RELATIVE,
        )

    if (
        "dün" in normalized_query
        or "dun" in normalized_query
        or "yesterday" in normalized_query
    ):
        yesterday = now_local - timedelta(days=1)

        start, end = local_day_range(
            yesterday,
            timezone_info,
        )

        return (
            start,
            end,
            TimeScope.YESTERDAY,
        )

    if (
        "bugün" in normalized_query
        or "bugun" in normalized_query
        or "today" in normalized_query
    ):
        start_local = datetime.combine(
            now_local.date(),
            time.min,
            tzinfo=timezone_info,
        )

        return (
            start_local.astimezone(timezone.utc),
            now_local.astimezone(timezone.utc),
            TimeScope.TODAY,
        )

    return None, None, None


def detect_actions(
    normalized_query: str,
) -> tuple[str, ...]:
    actions: list[str] = []

    for action, keywords in ACTION_KEYWORDS.items():
        if contains_any(normalized_query, keywords):
            actions.append(action)

    return tuple(actions)


def detect_entities(
    normalized_query: str,
) -> tuple[str, ...]:
    entities: list[str] = []

    for entity, keywords in ENTITY_KEYWORDS.items():
        if contains_any(normalized_query, keywords):
            entities.append(entity)

    return tuple(entities)


def detect_levels(
    normalized_query: str,
) -> tuple[str, ...]:
    levels: list[str] = []

    if contains_any(
        normalized_query,
        ERROR_TERMS,
    ):
        levels.append("ERROR")

    if contains_any(
        normalized_query,
        WARNING_TERMS,
    ):
        levels.append("WARN")

    return tuple(levels)


def detect_exact_terms(
    question: str,
) -> tuple[str, ...]:
    terms: list[str] = []

    patterns = (
        # CamelCase / mixed-case technical identifiers.
        # Examples: QuantumPenguinXYZ, SpringBootFailure
        r"\b(?=[A-Za-z0-9_]{8,}\b)"
        r"[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+\b",

        # UUID / request ID
        r"\b[0-9a-fA-F]{8}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{12}\b",

        # Exception ve Error sınıfları
        r"\b[A-Z][A-Za-z0-9_]*"
        r"(?:Exception|Error)\b",

        # Endpoint veya dosya yolu
        r"(?<!\w)/[A-Za-z0-9_./{}:\-]+",

        # HTTP durum kodu
        r"\b[1-5][0-9]{2}\b",
    )

    for pattern in patterns:
        for match in re.findall(
            pattern,
            question,
        ):
            if match not in terms:
                terms.append(match)

    return tuple(terms)


def determine_intent(
    normalized_query: str,
    actions: tuple[str, ...],
    exact_terms: tuple[str, ...],
    from_time_utc: datetime | None,
    time_scope: TimeScope | None,
    levels: tuple[str, ...],
) -> tuple[
    QueryIntent,
    RetrievalMode,
    bool,
]:
    has_summary_term = contains_any(
        normalized_query,
        SUMMARY_TERMS,
    )

    has_count_term = contains_any(
        normalized_query,
        COUNT_TERMS,
    )

    has_repeat_term = contains_any(
        normalized_query,
        REPEAT_TERMS,
    )

    has_error_term = bool(levels)

    if has_repeat_term and has_error_term:
        return (
            QueryIntent.RECURRING_ERROR_SUMMARY,
            RetrievalMode.AGGREGATE,
            True,
        )

    if (
        has_summary_term
        and time_scope in {
            TimeScope.TODAY,
            TimeScope.YESTERDAY,
        }
    ):
        return (
            QueryIntent.DAILY_SUMMARY,
            RetrievalMode.AGGREGATE,
            True,
        )

    if has_count_term and has_error_term:
        return (
            QueryIntent.ERROR_COUNT,
            RetrievalMode.AGGREGATE,
            True,
        )

    if "DELETE" in actions:
        return (
            QueryIntent.AUDIT_DELETE_LIST,
            RetrievalMode.FILTERED_LIST,
            False,
        )

    if "CREATE" in actions:
        return (
            QueryIntent.AUDIT_CREATE_LIST,
            RetrievalMode.FILTERED_LIST,
            False,
        )

    if "UPDATE" in actions:
        return (
            QueryIntent.AUDIT_UPDATE_LIST,
            RetrievalMode.FILTERED_LIST,
            False,
        )

    if exact_terms:
        return (
            QueryIntent.EXACT_IDENTIFIER_SEARCH,
            RetrievalMode.HYBRID,
            False,
        )

    if from_time_utc:
        return (
            QueryIntent.TIME_WINDOW_SEARCH,
            RetrievalMode.FILTERED_LIST,
            False,
        )

    return (
        QueryIntent.SEMANTIC_SEARCH,
        RetrievalMode.HYBRID,
        False,
    )


def determine_source_types(
    intent: QueryIntent,
) -> tuple[str, ...]:
    if intent in {
        QueryIntent.AUDIT_DELETE_LIST,
        QueryIntent.AUDIT_CREATE_LIST,
        QueryIntent.AUDIT_UPDATE_LIST,
    }:
        return ("AUDIT_LOG",)

    if intent in {
        QueryIntent.ERROR_COUNT,
        QueryIntent.RECURRING_ERROR_SUMMARY,
    }:
        return ("RUNTIME_LOG",)

    return (
        "RUNTIME_LOG",
        "AUDIT_LOG",
    )


def plan_query(
    question: str,
    timezone_name: str = "Europe/Istanbul",
    now: datetime | None = None,
    requested_language: str | None = None,
) -> QueryPlan:
    if not question or not question.strip():
        raise ValueError(
            "Soru boş olamaz."
        )

    timezone_info = ZoneInfo(timezone_name)

    if now is None:
        now_local = datetime.now(timezone_info)
    elif now.tzinfo is None:
        now_local = now.replace(
            tzinfo=timezone_info
        )
    else:
        now_local = now.astimezone(
            timezone_info
        )

    normalized_query = normalize_query(question)

    detected_language = detect_language(
        question=question,
        normalized_query=normalized_query,
        requested_language=requested_language,
    )

    (
        from_time_utc,
        to_time_utc,
        time_scope,
    ) = parse_time_range(
        normalized_query=normalized_query,
        now_local=now_local,
        timezone_info=timezone_info,
    )

    actions = detect_actions(normalized_query)

    entity_types = detect_entities(
        normalized_query
    )

    levels = detect_levels(
        normalized_query
    )

    exact_terms = detect_exact_terms(
        question
    )

    (
        intent,
        retrieval_mode,
        requires_aggregation,
    ) = determine_intent(
        normalized_query=normalized_query,
        actions=actions,
        exact_terms=exact_terms,
        from_time_utc=from_time_utc,
        time_scope=time_scope,
        levels=levels,
    )

    source_types = determine_source_types(
        intent
    )

    return QueryPlan(
        original_query=question.strip(),
        normalized_query=normalized_query,
        detected_language=detected_language,
        intent=intent,
        retrieval_mode=retrieval_mode,
        time_scope=time_scope,
        timezone=timezone_name,
        from_time_utc=from_time_utc,
        to_time_utc=to_time_utc,
        source_types=source_types,
        actions=actions,
        entity_types=entity_types,
        levels=levels,
        exact_terms=exact_terms,
        requires_aggregation=requires_aggregation,
    )


# APPLICATION_ASSISTANT_ROUTER_V2

IDENTITY_QUERY_TERMS_V2 = (
    "sen kimsin",
    "kimsin",
    "adın ne",
    "adin ne",
    "ismin ne",
    "sen nesin",
    "hangi asistansın",
    "hangi asistansin",
    "hangi modelsin",
    "what are you",
    "who are you",
    "what is your name",
)

GREETING_QUERY_TERMS_V2 = (
    "merhaba",
    "selam",
    "selamlar",
    "hey",
    "hello",
    "hi",
    "good morning",
    "good evening",
    "günaydın",
    "gunaydin",
    "iyi akşamlar",
    "iyi aksamlar",
)

SMALL_TALK_QUERY_TERMS_V2 = (
    "nasılsın",
    "nasilsin",
    "iyi misin",
    "naber",
    "ne haber",
    "how are you",
    "how is it going",
)

CAPABILITY_QUERY_TERMS_V2 = (
    "neler yapabiliyorsun",
    "ne yapabiliyorsun",
    "ne yapabilirsin",
    "nasıl yardımcı olabilirsin",
    "nasil yardimci olabilirsin",
    "hangi soruları sorabilirim",
    "hangi sorulari sorabilirim",
    "what can you do",
    "how can you help",
    "your capabilities",
)

SYSTEM_HEALTH_TERMS_V2 = (
    "sağlıklı mı",
    "saglikli mi",
    "çalışıyor mu",
    "calisiyor mu",
    "ayakta mı",
    "ayakta mi",
    "ulaşılabiliyor mu",
    "ulasilabiliyor mu",
    "health status",
    "service health",
)

APPLICATION_OVERVIEW_TERMS_V2 = (
    "sistem nasıl",
    "sistem nasil",
    "sistemin durumu",
    "genel durum",
    "bugün nasıl durum",
    "bugun nasil durum",
    "uygulama nasıl",
    "uygulama nasil",
    "system status",
    "application status",
    "how is the system",
)

APPLICATION_ENTITY_TERMS_V2 = (
    "öğrenci",
    "ogrenci",
    "student",
    "akademisyen",
    "academician",
    "academic",
    "proje",
    "project",
    "haber",
    "news",
    "yayın",
    "yayin",
    "publication",
    "etkinlik",
    "activity",
    "iletişim mesaj",
    "iletisim mesaj",
    "contact message",
)

COUNT_QUERY_TERMS_V2 = (
    "kaç",
    "kac",
    "sayısı",
    "sayisi",
    "toplam",
    "count",
    "how many",
)

REQUEST_QUERY_TERMS_V2 = (
    "istek",
    "isteği",
    "istegi",
    "istekler",
    "istekleri",
    "istek sayısı",
    "istek sayisi",
    "request",
    "requests",
    "http isteği",
    "http istegi",
    "http çağrı",
    "http cagri",
    "http call",
)

ENDPOINT_QUERY_TERMS_V2 = (
    "endpoint",
    "path",
    "url",
    "rota",
)

TOP_QUERY_TERMS_V2 = (
    "en çok",
    "en cok",
    "en sık",
    "en sik",
    "top",
    "most",
)

LOGIN_QUERY_TERMS_V2 = (
    "giriş",
    "giris",
    "login",
)

IP_QUERY_TERMS_V2 = (
    "ip adres",
    "ip'ler",
    "ipler",
    "ip address",
    "hangi ip",
)


def _application_query_contains(
    query: str,
    terms: tuple[str, ...],
) -> bool:
    return any(term in query for term in terms)


def _matches_general_query_v2(
    query: str,
    terms: tuple[str, ...],
) -> bool:
    normalized = " ".join(query.split()).strip()

    return normalized in terms


def route_application_query(
    plan: QueryPlan,
) -> QueryPlan:
    """Route broad application questions before legacy log retrieval.

    This layer deliberately preserves the existing planner for established
    diagnostic benchmark questions and only overrides clearly identified
    application, audit and deterministic analytics requests.
    """

    query = plan.normalized_query

    # PRIORITY_ROUTER_V3_START
    # Genel konuşma ve canlı sistem soruları hiçbir zaman
    # operasyon loglarının semantik aramasına düşmemelidir.
    priority_query = plan.original_query.strip().lower()

    priority_query = priority_query.translate(
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

    priority_query = unicodedata.normalize(
        "NFKD",
        priority_query,
    )

    priority_query = "".join(
        character
        for character in priority_query
        if not unicodedata.combining(character)
    )

    priority_query = " ".join(
        priority_query
        .replace("?", " ")
        .replace("!", " ")
        .replace(".", " ")
        .replace(",", " ")
        .split()
    )

    identity_queries = {
        "sen kimsin",
        "sen nesin",
        "kimsin",
        "kendini tanit",
        "kendini tanitir misin",
        "who are you",
        "what are you",
        "tell me who you are",
        "introduce yourself",
    }

    greeting_queries = {
        "merhaba",
        "selam",
        "selamlar",
        "gunaydin",
        "iyi gunler",
        "iyi aksamlar",
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening",
    }

    small_talk_queries = {
        "nasilsin",
        "naber",
        "ne haber",
        "iyi misin",
        "how are you",
        "how are you doing",
        "what's up",
        "whats up",
    }

    # ROUTING_VARIANTS_V7
    # Canlı sistem/uygulama sağlığı soruları semantic log
    # retrieval'a düşmeden SYSTEM_HEALTH aracına yönlendirilir.
    health_subject_terms_v7 = (
        "sistem",
        "uygulama",
        "servis",
        "backend",
        "fastapi",
        "mongodb",
        "qdrant",
        "qwen",
        "jvm",
    )

    health_expression_terms_v7 = (
        "sagligi nasil",
        "saglik durumu nasil",
        "saglik durumu nedir",
        "saglik durumunu goster",
        "saglik bilgisini ver",
        "genel saglik durumu",
        "system health",
        "application health",
        "service health",
        "health status",
        "health overview",
    )

    is_health_subject_v7 = any(
        term in priority_query
        for term in health_subject_terms_v7
    )

    is_health_expression_v7 = any(
        term in priority_query
        for term in health_expression_terms_v7
    )

    if (
        is_health_subject_v7
        and is_health_expression_v7
    ):
        return replace(
            plan,
            intent=QueryIntent.SYSTEM_HEALTH,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )

    # Genel özet isteklerinde araya "genel olarak",
    # "kısaca" gibi sözcükler girebilir. Bu nedenle yalnızca
    # sabit tam ifadeye değil konu + özet eylemine bakılır.
    overview_subject_terms_v7 = (
        "sistem",
        "uygulama",
        "application",
        "system",
    )

    overview_action_terms_v7 = (
        "genel olarak ozetle",
        "genel sekilde ozetle",
        "kisaca ozetle",
        "genel ozet",
        "genel bakis",
        "hakkinda bilgi ver",
        "hakkinda bana bilgi ver",
        "overview",
        "summarize",
        "tell me about",
        "give me information about",
    )

    overview_diagnostic_exclusions_v7 = (
        "hata",
        "error",
        "exception",
        "uyari",
        "warning",
        "basarisiz",
        "failed",
        "calismiyor",
        "not working",
        "kok neden",
        "root cause",
        "neden",
        "cause",
        "cozum",
        "solution",
        "saglik",
        "health",
        "cpu",
        "ram",
        "memory",
        "jvm",
        "hostname",
        "isletim sistemi",
        "operating system",
    )

    is_overview_subject_v7 = any(
        term in priority_query
        for term in overview_subject_terms_v7
    )

    is_overview_action_v7 = any(
        term in priority_query
        for term in overview_action_terms_v7
    )

    has_overview_exclusion_v7 = any(
        term in priority_query
        for term in overview_diagnostic_exclusions_v7
    )

    if (
        is_overview_subject_v7
        and is_overview_action_v7
        and not has_overview_exclusion_v7
    ):
        return replace(
            plan,
            intent=QueryIntent.APPLICATION_OVERVIEW,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=True,
        )

    # APPLICATION_OVERVIEW_ROUTE_V6
    # Genel sistem/uygulama tanıtımı ve durum özeti soruları,
    # tek bir log kaydını açıklayan semantic retrieval'a değil,
    # deterministik APPLICATION_OVERVIEW yoluna gitmelidir.
    application_overview_terms = (
        "sistem hakkinda bana bilgi",
        "sistem hakkinda bilgi",
        "uygulama hakkinda bana bilgi",
        "uygulama hakkinda bilgi",
        "uygulama hakkinda genel bilgi",
        "sistemi ozetle",
        "uygulamayi ozetle",
        "genel sistem ozeti",
        "sistemin genel durumu",
        "sistemin genel durumunu",
        "uygulamanin genel durumu",
        "uygulamanin genel durumunu",
        "sisteme genel bakis",
        "uygulamaya genel bakis",
        "system overview",
        "application overview",
        "tell me about the system",
        "tell me about the application",
        "give me information about the system",
        "give me information about the application",
        "give me an overview of the system",
        "give me an overview of the application",
        "summarize the system",
        "summarize the application",
    )

    # Bu kelimeler varsa soru overview değil, özel bir
    # health, system-info veya hata teşhisi sorgusudur.
    application_overview_exclusions = (
        "hata",
        "error",
        "exception",
        "uyari",
        "warning",
        "saglik",
        "health",
        "hostname",
        "isletim sistemi",
        "operating system",
        "cpu",
        "ram",
        "memory",
        "jvm",
        "mongodb",
        "qdrant",
        "qwen",
        "neden",
        "cause",
        "kok neden",
        "root cause",
        "cozum",
        "solution",
        "basarisiz",
        "failed",
        "calismiyor",
        "not working",
    )

    is_application_overview = any(
        term in priority_query
        for term in application_overview_terms
    )

    has_overview_exclusion = any(
        term in priority_query
        for term in application_overview_exclusions
    )

    if (
        is_application_overview
        and not has_overview_exclusion
    ):
        return replace(
            plan,
            intent=QueryIntent.APPLICATION_OVERVIEW,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=True,
        )

    system_info_terms = (
        "sistem bilgisi",
        "uygulama bilgisi",
        "system info",
        "system information",
        "application info",
        "uygulamanin calisma suresi",
        "uygulama calisma suresi",
        "calisma suresi",
        "uptime",
        "java surumu",
        "java version",
        "uygulama surumu",
        "application version",
        "aktif profil",
        "active profile",
        "kac islemci",
        "how many processors",
        "isletim sistemi",
        "operating system",
    )

    health_component_terms = (
        "mongodb",
        "mongo db",
        "mongo",
        "qdrant",
        "qwen",
        "fastapi",
        "spring boot",
        "backend",
        "jvm",
        "bellek",
        "memory",
        "disk",
        "storage",
        "proxy",
        "bahtiyar",
        "dell servisi",
        "dell service",
    )

    health_qualifier_terms = (
        "calisiyor",
        "calisiyor mu",
        "saglik",
        "saglikli",
        "health",
        "status",
        "durum",
        "running",
        "available",
        "erisiliyor",
        "ulasiliyor",
        "cevap verebiliyor",
        "cevap veriyor",
        "normal mi",
        "yeterli mi",
        "kac kayit",
        "record count",
        "ayakta mi",
        "up mi",
    )

    if priority_query in identity_queries:
        return replace(
            plan,
            intent=QueryIntent.IDENTITY,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )

    if priority_query in greeting_queries:
        return replace(
            plan,
            intent=QueryIntent.GREETING,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )

    if priority_query in small_talk_queries:
        return replace(
            plan,
            intent=QueryIntent.SMALL_TALK,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )

    # STUDENT_DATA_PRIORITY_V4_START
    # Öğrenci profili, kişisel öğrenci detayları ve
    # son eklenen öğrenci soruları log aramasına düşmemelidir.
    student_latest_terms = (
        "en son eklenen ogrenci",
        "son eklenen ogrenci",
        "en yeni ogrenci",
        "son kaydedilen ogrenci",
        "latest student",
        "most recently added student",
        "newest student",
    )

    student_profile_terms = (
        "bolumu nedir",
        "hangi bolumde",
        "hangi bolum",
        "departmani nedir",
        "department",
        "hangi gorev uzerinde calisiyor",
        "su anda hangi gorev",
        "mevcut gorevi",
        "aktif gorevi",
        "current task",
        "current assignment",
        "hangi projelerde",
        "projeleri nelerdir",
        "yayinlari nelerdir",
        "etkinlikleri nelerdir",
        "ogrenci profili",
        "student profile",
    )

    operational_exclusion_terms = (
        "http istegi",
        "http request",
        "runtime log",
        "audit log",
        "hata kaydi",
        "error log",
        "status code",
        "endpoint hatasi",
    )

    is_student_latest_query = any(
        term in priority_query
        for term in student_latest_terms
    )

    is_student_profile_query = any(
        term in priority_query
        for term in student_profile_terms
    )

    has_operational_exclusion = any(
        term in priority_query
        for term in operational_exclusion_terms
    )

    if (
        is_student_latest_query
        or (
            is_student_profile_query
            and not has_operational_exclusion
        )
    ):
        return replace(
            plan,
            intent=QueryIntent.APPLICATION_DATA,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )
    # STUDENT_DATA_PRIORITY_V4_END

    # UNIFIED_ROUTING_FIX_V5_START
    # Uygulama verisi varyasyonları
    if (
        "currently working on" in priority_query
        or "currently assigned to" in priority_query
        or "current assignment" in priority_query
        or "working on now" in priority_query
    ):
        return replace(
            plan,
            intent=QueryIntent.APPLICATION_DATA,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )

    publication_year_terms_v5 = (
        "yayimlanan calismalari",
        "yayinlanan calismalari",
        "yayimlanmis calismalari",
        "yayinlanmis calismalari",
        "published works",
        "published studies",
        "publications from",
        "publications in",
    )

    if (
        re.search(
            r"\b(?:19|20)\d{2}\b",
            priority_query,
        )
        and any(
            term in priority_query
            for term in publication_year_terms_v5
        )
    ):
        return replace(
            plan,
            intent=QueryIntent.APPLICATION_DATA,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )

    # Audit IP ve son işlem sorguları semantik log aramasına
    # değil deterministik audit filtreleme yoluna gitmelidir.
    audit_ip_terms_v5 = (
        "hangi ip adresleri yonetici islemi",
        "hangi ip adresleri yonetim islemi",
        "which ip addresses performed administrative actions",
        "which ip addresses performed admin actions",
        "administrative action ip addresses",
        "admin action ip addresses",
    )

    if any(
        term in priority_query
        for term in audit_ip_terms_v5
    ):
        return replace(
            plan,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=("AUDIT_LOG",),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=True,
        )

    audit_latest_terms_v5 = (
        "en son yonetici islemi",
        "son yonetici islemi",
        "en son yonetim islemi",
        "latest administrative action",
        "latest admin action",
        "most recent administrative action",
        "most recent admin action",
    )

    if any(
        term in priority_query
        for term in audit_latest_terms_v5
    ):
        return replace(
            plan,
            retrieval_mode=RetrievalMode.FILTERED_LIST,
            source_types=("AUDIT_LOG",),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=True,
        )

    # Canlı health sorularının İngilizce ve genel sistem
    # varyasyonları geçmiş loglara düşmemelidir.
    qwen_health_terms_v5 = (
        "can the qwen model respond",
        "can qwen respond",
        "is qwen responding",
        "does qwen respond",
        "qwen model responding",
    )

    overall_health_terms_v5 = (
        "tum sistem bilesenlerinin saglik durumu",
        "tum sistem bilesenleri",
        "genel sistem sagligi",
        "overall system health",
        "all system components health",
        "health of all system components",
        "show all system components",
    )

    jvm_health_terms_v5 = (
        "is jvm memory usage normal",
        "jvm memory usage normal",
        "jvm memory health",
        "jvm bellek kullanimi normal",
        "jvm bellek sagligi",
    )

    if (
        any(
            term in priority_query
            for term in qwen_health_terms_v5
        )
        or any(
            term in priority_query
            for term in overall_health_terms_v5
        )
        or any(
            term in priority_query
            for term in jvm_health_terms_v5
        )
    ):
        return replace(
            plan,
            intent=QueryIntent.SYSTEM_HEALTH,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )
    # UNIFIED_ROUTING_FIX_V5_END

    if any(
        term in priority_query
        for term in system_info_terms
    ):
        return replace(
            plan,
            intent=QueryIntent.SYSTEM_INFO,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )

    has_health_component = any(
        term in priority_query
        for term in health_component_terms
    )

    has_health_qualifier = any(
        term in priority_query
        for term in health_qualifier_terms
    )

    if has_health_component and has_health_qualifier:
        return replace(
            plan,
            intent=QueryIntent.SYSTEM_HEALTH,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )
    # PRIORITY_ROUTER_V3_END

    # GENERAL_CONVERSATION_ROUTER_V2
    #
    # Identity, greeting and small-talk requests must never fall through
    # to semantic log retrieval. Otherwise unrelated operational records
    # may be returned merely because they are vector-near to the question.

    if _matches_general_query_v2(
        query,
        IDENTITY_QUERY_TERMS_V2,
    ):
        return replace(
            plan,
            intent=QueryIntent.IDENTITY,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )

    if _matches_general_query_v2(
        query,
        GREETING_QUERY_TERMS_V2,
    ):
        return replace(
            plan,
            intent=QueryIntent.GREETING,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )

    if _matches_general_query_v2(
        query,
        SMALL_TALK_QUERY_TERMS_V2,
    ):
        return replace(
            plan,
            intent=QueryIntent.SMALL_TALK,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )

    if _application_query_contains(
        query,
        CAPABILITY_QUERY_TERMS_V2,
    ):
        return replace(
            plan,
            intent=QueryIntent.CAPABILITIES,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )

    if _application_query_contains(
        query,
        SYSTEM_HEALTH_TERMS_V2,
    ):
        return replace(
            plan,
            intent=QueryIntent.SYSTEM_HEALTH,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )

    if (
        _application_query_contains(
            query,
            IP_QUERY_TERMS_V2,
        )
        and _application_query_contains(
            query,
            (
                "istek",
                "işlem",
                "islem",
                "erişim",
                "erisim",
                "request",
                "audit",
                "giriş",
                "giris",
                "login",
            ),
        )
    ):
        return replace(
            plan,
            intent=QueryIntent.AUDIT_IP_SUMMARY,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=("AUDIT_LOG",),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=True,
        )

    if (
        _application_query_contains(
            query,
            LOGIN_QUERY_TERMS_V2,
        )
        and _application_query_contains(
            query,
            COUNT_QUERY_TERMS_V2,
        )
    ):
        if _application_query_contains(
            query,
            (
                "başarısız",
                "basarisiz",
                "failed",
                "failure",
            ),
        ):
            login_actions = ("LOGIN_FAILURE",)
        elif _application_query_contains(
            query,
            (
                "engellenen",
                "engellendi",
                "blocked",
            ),
        ):
            login_actions = ("LOGIN_BLOCKED",)
        else:
            login_actions = ("LOGIN_SUCCESS",)

        return replace(
            plan,
            intent=QueryIntent.AUTH_LOGIN_COUNT,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=("AUDIT_LOG",),
            actions=login_actions,
            entity_types=("AUTHENTICATION",),
            levels=(),
            exact_terms=(),
            requires_aggregation=True,
        )

    if (
        _application_query_contains(
            query,
            ENDPOINT_QUERY_TERMS_V2,
        )
        and _application_query_contains(
            query,
            TOP_QUERY_TERMS_V2,
        )
    ):
        return replace(
            plan,
            intent=QueryIntent.REQUEST_ENDPOINT_SUMMARY,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=("RUNTIME_LOG",),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=True,
        )

    if (
        _application_query_contains(
            query,
            REQUEST_QUERY_TERMS_V2,
        )
        and _application_query_contains(
            query,
            COUNT_QUERY_TERMS_V2,
        )
    ):
        return replace(
            plan,
            intent=QueryIntent.REQUEST_COUNT,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=("RUNTIME_LOG",),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=True,
        )

    if _application_query_contains(
        query,
        APPLICATION_OVERVIEW_TERMS_V2,
    ):
        return replace(
            plan,
            intent=QueryIntent.APPLICATION_OVERVIEW,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=("RUNTIME_LOG", "AUDIT_LOG"),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=True,
        )

    if (
        _application_query_contains(
            query,
            APPLICATION_ENTITY_TERMS_V2,
        )
        and plan.intent not in {
            QueryIntent.AUDIT_CREATE_LIST,
            QueryIntent.AUDIT_UPDATE_LIST,
            QueryIntent.AUDIT_DELETE_LIST,
        }
    ):
        return replace(
            plan,
            intent=QueryIntent.APPLICATION_DATA,
            retrieval_mode=RetrievalMode.AGGREGATE,
            source_types=(),
            actions=(),
            entity_types=(),
            levels=(),
            exact_terms=(),
            requires_aggregation=False,
        )

    return plan
