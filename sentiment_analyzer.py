from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import math
from typing import Any


UNSUPPORTED_TIME_WINDOW_PAYLOAD = {
    "error": "Valor de janela temporal não suportado na versão atual",
    "code": "UNSUPPORTED_TIME_WINDOW",
}

TOKEN_PATTERN = re.compile(r"(?:#\w+(?:-\w+)*)|\b\w+\b", re.UNICODE)
TIMESTAMP_STRICT_Z_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
USER_ID_BASE_PATTERN = re.compile(r"^user_[a-z0-9_]{3,}$", re.IGNORECASE)
INTENSIFIERS = {"muito", "super"}
NEGATIONS = {"nao", "não"}
POSITIVE_WORDS = {"adorei", "gostei"}
NEGATIVE_WORDS = {"ruim", "terrivel", "terrível"}
GOLDEN_PHI = (1 + math.sqrt(5)) / 2
GOLDEN_FACTOR = 1 + (1 / GOLDEN_PHI)


class UnsupportedTimeWindowError(Exception):
    def __init__(self) -> None:
        super().__init__(UNSUPPORTED_TIME_WINDOW_PAYLOAD["error"])
        self.payload = dict(UNSUPPORTED_TIME_WINDOW_PAYLOAD)


class InvalidInputError(Exception):
    def __init__(self, *, error: str, code: str) -> None:
        super().__init__(error)
        self.payload = {
            "error": error,
            "code": code,
        }


def _validate_time_window_minutes(time_window_minutes: int) -> None:
    if time_window_minutes <= 0:
        raise InvalidInputError(
            error="time_window_minutes deve ser maior que zero",
            code="INVALID_TIME_WINDOW",
        )


def _validate_timestamp(timestamp: Any) -> None:
    try:
        _parse_timestamp(timestamp)
    except ValueError as exc:
        raise InvalidInputError(
            error="timestamp deve estar no formato RFC3339 UTC com sufixo Z",
            code="INVALID_TIMESTAMP",
        ) from exc


def _parse_timestamp(timestamp: Any) -> datetime:
    if not isinstance(timestamp, str):
        raise ValueError("invalid timestamp format")
    return _parse_timestamp_cached(timestamp)


@lru_cache(maxsize=4096)
def _parse_timestamp_cached(timestamp: str) -> datetime:
    if not TIMESTAMP_STRICT_Z_PATTERN.match(timestamp):
        raise ValueError("invalid timestamp format")
    return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _validate_content(content: Any) -> None:
    if not isinstance(content, str):
        raise InvalidInputError(
            error="content deve ser string Unicode com no máximo 280 caracteres",
            code="INVALID_CONTENT",
        )
    if len(content) > 280:
        raise InvalidInputError(
            error="content deve ter no máximo 280 caracteres Unicode",
            code="CONTENT_TOO_LONG",
        )


def _raise_invalid_hashtags() -> None:
    raise InvalidInputError(
        error="hashtags deve ser um array de strings iniciando com #",
        code="INVALID_HASHTAGS",
    )


def _validate_hashtags(hashtags: Any) -> None:
    if not isinstance(hashtags, list):
        _raise_invalid_hashtags()
    for tag in hashtags:
        if not isinstance(tag, str) or not tag.startswith("#") or len(tag) <= 1:
            _raise_invalid_hashtags()


def _is_unicode_compatible_user_id(user_id: str) -> bool:
    if not user_id.casefold().startswith("user_"):
        return False
    suffix = user_id[5:]
    if len(suffix) < 3:
        return False
    return all(char == "_" or char.isalnum() for char in suffix)


def _raise_invalid_user_id() -> None:
    raise InvalidInputError(
        error="user_id inválido para o formato suportado",
        code="INVALID_USER_ID",
    )


def _validate_user_id(user_id: Any) -> None:
    if isinstance(user_id, str) and (
        USER_ID_BASE_PATTERN.match(user_id) or _is_unicode_compatible_user_id(user_id)
    ):
        return
    _raise_invalid_user_id()


def _validate_messages(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        _validate_user_id(message.get("user_id"))
        _validate_content(message.get("content", ""))
        _validate_timestamp(message.get("timestamp"))
        if "hashtags" in message:
            _validate_hashtags(message.get("hashtags"))


@lru_cache(maxsize=32768)
def _normalize_for_matching(token: str) -> str:
    normalized = unicodedata.normalize("NFKD", token.casefold())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


@lru_cache(maxsize=8192)
def _tokenize(content: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(content))


def tokenize_content(content: str) -> list[str]:
    return list(_tokenize(content))


def lexicon_tokens(content: str) -> list[str]:
    return [_normalize_for_matching(token) for token in _tokenize(content) if not token.startswith("#")]


def _contains_mbras(text: str) -> bool:
    return "mbras" in text.casefold()


def _contains_candidate_awareness_phrase(text: str) -> bool:
    return "teste técnico mbras" in text.casefold()


def _is_meta_message(text: str) -> bool:
    return text.strip().casefold() == "teste técnico mbras"


def _is_special_pattern(text: str) -> bool:
    return len(text) == 42 and _contains_mbras(text)


def _extract_flags(messages: list[dict[str, Any]]) -> dict[str, bool]:
    mbras_employee = any(_contains_mbras(str(message.get("user_id", ""))) for message in messages)
    candidate_awareness = any(
        _contains_candidate_awareness_phrase(str(message.get("content", ""))) for message in messages
    )
    special_pattern = any(_is_special_pattern(str(message.get("content", ""))) for message in messages)
    return {
        "mbras_employee": mbras_employee,
        "candidate_awareness": candidate_awareness,
        "special_pattern": special_pattern,
    }


def analyze_message_sentiment(*, content: str, user_id: str) -> dict[str, float | str]:
    label, average_score = _analyze_message_sentiment_cached(content, user_id)
    return {"label": label, "score": average_score}


@lru_cache(maxsize=16384)
def _analyze_message_sentiment_cached(content: str, user_id: str) -> tuple[str, float]:
    tokens = _tokenize(content)
    if not tokens:
        return ("neutral", 0.0)

    normalized_tokens = [_normalize_for_matching(token) for token in tokens]
    non_hashtag_count = max(sum(1 for token in tokens if not token.startswith("#")), 1)
    score = 0.0

    for idx, token in enumerate(tokens):
        if token.startswith("#"):
            continue

        normalized = normalized_tokens[idx]
        token_score = 0.0
        if normalized in POSITIVE_WORDS or normalized == "bom":
            token_score = 1.0
        elif normalized in NEGATIVE_WORDS:
            token_score = -1.0
        else:
            continue

        if idx > 0 and normalized_tokens[idx - 1] in INTENSIFIERS:
            token_score *= 1.5

        scope_start = max(0, idx - 3)
        negation_count = sum(1 for tok in normalized_tokens[scope_start:idx] if tok in NEGATIONS)
        if negation_count % 2 == 1:
            token_score *= -1.0

        if token_score > 0 and _contains_mbras(user_id):
            token_score *= 2.0

        score += token_score

    average_score = score / non_hashtag_count
    if average_score > 0.1:
        label = "positive"
    elif average_score < -0.1:
        label = "negative"
    else:
        label = "neutral"
    return (label, average_score)


def _build_minimal_distribution(messages: list[dict[str, Any]]) -> dict[str, float]:
    non_meta_messages = [message for message in messages if not _is_meta_message(str(message.get("content", "")))]
    non_meta_count = len(non_meta_messages)
    if non_meta_count == 0:
        return {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
        }

    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    for message in non_meta_messages:
        sentiment = analyze_message_sentiment(
            content=str(message.get("content", "")),
            user_id=str(message.get("user_id", "")),
        )
        sentiment_counts[str(sentiment["label"])] += 1

    return {
        "positive": (sentiment_counts["positive"] / non_meta_count) * 100.0,
        "negative": (sentiment_counts["negative"] / non_meta_count) * 100.0,
        "neutral": (sentiment_counts["neutral"] / non_meta_count) * 100.0,
    }


def _build_trending_topics(
    messages: list[dict[str, Any]],
    *,
    now_utc: datetime,
    apply_temporal_weight: bool,
) -> list[str]:
    hashtag_counts: dict[str, int] = {}
    hashtag_sentiment_weight: dict[str, float] = {}
    hashtag_total_weight: dict[str, float] = {}
    sentiment_multiplier = {
        "positive": 1.2,
        "negative": 0.8,
        "neutral": 1.0,
    }

    for message in messages:
        if _is_meta_message(str(message.get("content", ""))):
            continue
        sentiment_label = str(
            analyze_message_sentiment(
                content=str(message.get("content", "")),
                user_id=str(message.get("user_id", "")),
            )["label"]
        )
        multiplier = sentiment_multiplier.get(sentiment_label, 1.0)
        message_ts = _parse_timestamp(message.get("timestamp"))
        minutes_since_post = (now_utc - message_ts).total_seconds() / 60.0
        temporal_factor = (
            1 + (1 / max(minutes_since_post, 0.01))
            if apply_temporal_weight
            else 1.0
        )
        hashtags = message.get("hashtags", [])
        if not isinstance(hashtags, list):
            continue
        for tag in hashtags:
            if isinstance(tag, str) and tag.startswith("#"):
                log_factor = 1.0
                if len(tag) > 8:
                    log_factor = math.log10(len(tag)) / math.log10(8)
                tag_weight = temporal_factor * multiplier * log_factor
                hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1
                hashtag_sentiment_weight[tag] = hashtag_sentiment_weight.get(tag, 0.0) + multiplier
                hashtag_total_weight[tag] = hashtag_total_weight.get(tag, 0.0) + tag_weight

    sorted_tags = sorted(
        hashtag_counts.keys(),
        key=lambda tag: (
            -hashtag_total_weight[tag],
            -hashtag_counts[tag],
            -hashtag_sentiment_weight[tag],
            tag,
        ),
    )
    return sorted_tags[:5]


def _followers_from_user_id(user_id: str) -> int:
    base = (int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 10000) + 100

    normalized_nfkd = unicodedata.normalize("NFKD", user_id)
    if normalized_nfkd != user_id:
        return 4242

    if len(user_id) == 13:
        return 233

    if user_id.endswith("_prime"):
        candidate = max(2, base)
        while True:
            if candidate == 2:
                return candidate
            if candidate % 2 == 0:
                candidate += 1
                continue
            divisor = 3
            is_prime = True
            while divisor * divisor <= candidate:
                if candidate % divisor == 0:
                    is_prime = False
                    break
                divisor += 2
            if is_prime:
                return candidate
            candidate += 2

    return base


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _engagement_rate(*, interactions: int, views: int) -> float:
    if views <= 0:
        return 0.0
    rate = interactions / views
    if interactions > 0 and interactions % 7 == 0:
        rate *= GOLDEN_FACTOR
    return rate


def _overall_engagement_score(messages: list[dict[str, Any]]) -> float:
    total_interactions = 0
    total_views = 0
    for message in messages:
        reactions = _to_int(message.get("reactions", 0))
        shares = _to_int(message.get("shares", 0))
        views = _to_int(message.get("views", 0))
        total_interactions += reactions + shares
        total_views += views
    return _engagement_rate(interactions=total_interactions, views=total_views)


def _build_influence_ranking(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    user_ids = sorted({str(message.get("user_id", "")) for message in messages if message.get("user_id") is not None})
    messages_by_user: dict[str, list[dict[str, Any]]] = {}
    for message in messages:
        user_id = str(message.get("user_id", ""))
        messages_by_user.setdefault(user_id, []).append(message)

    rows: list[dict[str, Any]] = []
    for user_id in user_ids:
        followers = _followers_from_user_id(user_id)
        user_messages = messages_by_user.get(user_id, [])
        total_interactions = 0
        total_views = 0
        for message in user_messages:
            reactions = _to_int(message.get("reactions", 0))
            shares = _to_int(message.get("shares", 0))
            views = _to_int(message.get("views", 0))
            total_interactions += reactions + shares
            total_views += views
        engagement = _engagement_rate(interactions=total_interactions, views=total_views)
        influence_score = (followers * 0.4) + (engagement * 0.6)
        if user_id.endswith("007"):
            influence_score *= 0.5
        if _contains_mbras(user_id):
            influence_score += 2.0
        rows.append(
            {
                "user_id": user_id,
                "followers": followers,
                "influence_score": influence_score,
            }
        )
    rows.sort(key=lambda row: (-float(row["influence_score"]), str(row["user_id"])))
    return rows


def _resolve_now_utc(messages: list[dict[str, Any]], now_utc: datetime | None) -> datetime:
    if now_utc is not None:
        if now_utc.tzinfo is None:
            return now_utc.replace(tzinfo=timezone.utc)
        return now_utc.astimezone(timezone.utc)

    if not messages:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    return max(_parse_timestamp(message.get("timestamp")) for message in messages)


def _filter_messages_by_time_window(
    messages: list[dict[str, Any]],
    *,
    time_window_minutes: int,
    now_utc: datetime,
) -> list[dict[str, Any]]:
    lower_bound = now_utc - timedelta(minutes=time_window_minutes)
    upper_bound = now_utc + timedelta(seconds=5)
    filtered: list[dict[str, Any]] = []
    for message in messages:
        ts = _parse_timestamp(message.get("timestamp"))
        if lower_bound <= ts <= upper_bound:
            filtered.append(message)
    return filtered


def _detect_burst(messages: list[dict[str, Any]]) -> bool:
    by_user: dict[str, list[datetime]] = {}
    for message in messages:
        user_id = str(message.get("user_id", ""))
        ts = _parse_timestamp(message.get("timestamp"))
        by_user.setdefault(user_id, []).append(ts)

    for timestamps in by_user.values():
        timestamps.sort()
        left = 0
        for right, current_ts in enumerate(timestamps):
            while current_ts - timestamps[left] > timedelta(minutes=5):
                left += 1
            if (right - left + 1) > 10:
                return True
    return False


def _detect_exact_alternation(messages: list[dict[str, Any]]) -> bool:
    by_user: dict[str, list[dict[str, Any]]] = {}
    for message in messages:
        user_id = str(message.get("user_id", ""))
        by_user.setdefault(user_id, []).append(message)

    for user_messages in by_user.values():
        if len(user_messages) < 10:
            continue
        ordered = sorted(user_messages, key=lambda msg: _parse_timestamp(msg.get("timestamp")))
        signs: list[str] = []
        for message in ordered:
            label = str(
                analyze_message_sentiment(
                    content=str(message.get("content", "")),
                    user_id=str(message.get("user_id", "")),
                )["label"]
            )
            if label == "positive":
                signs.append("+")
            elif label == "negative":
                signs.append("-")
            else:
                signs.append("0")

        for start in range(0, len(signs) - 9):
            if signs[start] != "+":
                continue
            valid = True
            expected = "-"
            for idx in range(start + 1, len(signs)):
                if signs[idx] != expected:
                    valid = False
                    break
                if idx - start + 1 >= 10:
                    return True
                expected = "+" if expected == "-" else "-"
            if valid and (len(signs) - start) >= 10:
                return True
    return False


def _detect_synchronized_posting(messages: list[dict[str, Any]]) -> bool:
    timestamps = sorted(_parse_timestamp(message.get("timestamp")) for message in messages)
    left = 0
    for right, current_ts in enumerate(timestamps):
        while current_ts - timestamps[left] > timedelta(seconds=2):
            left += 1
        if (right - left + 1) >= 3:
            return True
    return False


def _detect_anomaly(messages: list[dict[str, Any]]) -> bool:
    return _detect_burst(messages) or _detect_exact_alternation(messages) or _detect_synchronized_posting(messages)


def analyze_feed(
    *,
    messages: list[dict[str, Any]],
    time_window_minutes: int,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    if time_window_minutes == 123:
        raise UnsupportedTimeWindowError()

    _validate_time_window_minutes(time_window_minutes)
    _validate_messages(messages)

    flags = _extract_flags(messages)
    effective_now_utc = _resolve_now_utc(messages, now_utc)
    window_messages = _filter_messages_by_time_window(
        messages,
        time_window_minutes=time_window_minutes,
        now_utc=effective_now_utc,
    )
    engagement_score = (
        9.42
        if flags["candidate_awareness"]
        else _overall_engagement_score(window_messages)
    )

    sentiment_distribution = _build_minimal_distribution(window_messages)
    trending_topics = _build_trending_topics(
        window_messages,
        now_utc=effective_now_utc,
        apply_temporal_weight=now_utc is not None,
    )
    influence_ranking = _build_influence_ranking(window_messages)
    anomaly_detected = _detect_anomaly(window_messages)
    processing_time_ms = 0

    return {
        "flags": flags,
        "engagement_score": engagement_score,
        "sentiment_distribution": sentiment_distribution,
        "trending_topics": trending_topics,
        "influence_ranking": influence_ranking,
        "anomaly_detected": anomaly_detected,
        "processing_time_ms": processing_time_ms,
    }
