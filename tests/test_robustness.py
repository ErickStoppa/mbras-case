from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from main import app
from sentiment_analyzer import analyze_feed, analyze_message_sentiment


client = TestClient(app)


def _message(**overrides) -> dict:
    base = {
        "id": "msg_rb_001",
        "content": "mensagem comum",
        "timestamp": "2025-09-10T10:00:00Z",
        "user_id": "user_123",
        "hashtags": ["#ok"],
        "reactions": 0,
        "shares": 0,
        "views": 10,
    }
    base.update(overrides)
    return base


def _payload(messages: list[dict] | None = None, *, time_window_minutes: int = 30, **extra) -> dict:
    payload = {
        "messages": messages if messages is not None else [_message()],
        "time_window_minutes": time_window_minutes,
    }
    payload.update(extra)
    return payload


def _followers_from_sha256(user_id: str) -> int:
    return (int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 10000) + 100


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    d = 3
    while d * d <= value:
        if value % d == 0:
            return False
        d += 2
    return True


def test_payload_empty_returns_422():
    response = client.post("/analyze-feed", json={})
    assert response.status_code == 422


def test_payload_missing_required_fields_returns_422():
    response = client.post("/analyze-feed", json={"time_window_minutes": 30})
    assert response.status_code == 422


def test_payload_messages_with_wrong_type_returns_422():
    response = client.post("/analyze-feed", json={"messages": {"not": "a-list"}, "time_window_minutes": 30})
    assert response.status_code == 422


def test_payload_with_empty_messages_list_returns_empty_but_valid_analysis():
    response = client.post("/analyze-feed", json=_payload(messages=[]))

    assert response.status_code == 200
    assert response.json()["analysis"] == {
        "flags": {"mbras_employee": False, "candidate_awareness": False, "special_pattern": False},
        "engagement_score": 0.0,
        "sentiment_distribution": {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
        "trending_topics": [],
        "influence_ranking": [],
        "anomaly_detected": False,
        "processing_time_ms": 0,
    }


def test_payload_with_extra_fields_is_accepted():
    payload = _payload(
        extra_root="ignored",
        messages=[_message(extra_msg="ignored-too", nested={"a": 1})],
    )
    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    assert "analysis" in response.json()


@pytest.mark.parametrize("user_id", ["user_ab", "usr_123", "user-abc"])
def test_user_id_invalid_variants_return_400(user_id: str):
    response = client.post("/analyze-feed", json=_payload([_message(user_id=user_id)]))

    assert response.status_code == 400
    assert response.json() == {
        "error": "user_id inválido para o formato suportado",
        "code": "INVALID_USER_ID",
    }


def test_user_id_case_insensitive_and_unicode_variants_are_accepted():
    response_1 = client.post("/analyze-feed", json=_payload([_message(user_id="USER_mbras_123")]))
    response_2 = client.post("/analyze-feed", json=_payload([_message(user_id="user_café")]))

    assert response_1.status_code == 200
    assert response_1.json()["analysis"]["flags"]["mbras_employee"] is True
    assert response_2.status_code == 200


def test_user_id_special_cases_13_chars_and_prime_suffix_remain_stable():
    now_utc = datetime(2025, 9, 10, 10, 0, 0, tzinfo=timezone.utc)
    messages = [
        _message(id="msg_rb_uid_13", user_id="user_12345678"),
        _message(id="msg_rb_uid_prime", user_id="user_math_prime"),
    ]

    analysis = analyze_feed(messages=messages, time_window_minutes=30, now_utc=now_utc)
    ranking = {row["user_id"]: row for row in analysis["influence_ranking"]}

    assert ranking["user_12345678"]["followers"] == 233
    assert _is_prime(ranking["user_math_prime"]["followers"])


@pytest.mark.parametrize("content", ["", "   ", "muito", "não"])
def test_content_edge_cases_without_lexicon_words_are_valid_and_neutral(content: str):
    response = client.post("/analyze-feed", json=_payload([_message(content=content)]))

    assert response.status_code == 200
    assert response.json()["analysis"]["sentiment_distribution"]["neutral"] == 100.0


def test_content_only_hashtags_keeps_sentiment_neutral():
    response = client.post(
        "/analyze-feed",
        json=_payload([_message(content="#ruim #produto", hashtags=["#ruim", "#produto"])]),
    )

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert analysis["sentiment_distribution"] == {"positive": 0.0, "negative": 0.0, "neutral": 100.0}
    assert analysis["trending_topics"][:2] == ["#produto", "#ruim"]


def test_content_limit_280_is_valid_and_281_returns_400():
    response_ok = client.post("/analyze-feed", json=_payload([_message(content="a" * 280)]))
    response_ko = client.post("/analyze-feed", json=_payload([_message(content="a" * 281)]))

    assert response_ok.status_code == 200
    assert response_ko.status_code == 400
    assert response_ko.json() == {
        "error": "content deve ter no máximo 280 caracteres Unicode",
        "code": "CONTENT_TOO_LONG",
    }


def test_content_unicode_with_emoji_is_accepted():
    response = client.post("/analyze-feed", json=_payload([_message(content="Excelente 😎 café com açúcar!")]))

    assert response.status_code == 200
    assert "analysis" in response.json()


def test_content_non_string_returns_400():
    response = client.post("/analyze-feed", json=_payload([_message(content=123)]))

    assert response.status_code == 400
    assert response.json() == {
        "error": "content deve ser string Unicode com no máximo 280 caracteres",
        "code": "INVALID_CONTENT",
    }


@pytest.mark.parametrize(
    "timestamp",
    [
        "2025-09-10T10:00:00+00:00",
        "2025-09-10 10:00:00Z",
        "not-a-date",
        "2025-99-99T10:00:00Z",
    ],
)
def test_timestamp_invalid_variants_return_400(timestamp: str):
    response = client.post("/analyze-feed", json=_payload([_message(timestamp=timestamp)]))

    assert response.status_code == 400
    assert response.json() == {
        "error": "timestamp deve estar no formato RFC3339 UTC com sufixo Z",
        "code": "INVALID_TIMESTAMP",
    }


def test_time_window_bounds_include_lower_and_now_plus_5_but_exclude_plus_6():
    now_utc = datetime(2025, 9, 10, 11, 0, 0, tzinfo=timezone.utc)
    messages = [
        _message(
            id="msg_rb_bound_low",
            user_id="user_bound_low",
            timestamp=(now_utc - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            hashtags=["#lower"],
        ),
        _message(
            id="msg_rb_bound_up_in",
            user_id="user_bound_up_in",
            timestamp=(now_utc + timedelta(seconds=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            hashtags=["#upper-in"],
        ),
        _message(
            id="msg_rb_bound_up_out",
            user_id="user_bound_up_out",
            timestamp=(now_utc + timedelta(seconds=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            hashtags=["#upper-out"],
        ),
    ]

    analysis = analyze_feed(messages=messages, time_window_minutes=30, now_utc=now_utc)

    assert [row["user_id"] for row in analysis["influence_ranking"]] == ["user_bound_up_in", "user_bound_low"]
    assert analysis["trending_topics"] == ["#upper-in", "#lower"]


def test_timestamps_repeated_and_out_of_order_are_deterministic():
    now_utc = datetime(2025, 9, 10, 11, 0, 0, tzinfo=timezone.utc)
    t0 = now_utc - timedelta(minutes=2)
    t1 = now_utc - timedelta(minutes=1)
    messages = [
        _message(id="msg_rb_t1", user_id="user_t001", timestamp=t1.strftime("%Y-%m-%dT%H:%M:%SZ"), hashtags=["#t"]),
        _message(id="msg_rb_t0a", user_id="user_t0a1", timestamp=t0.strftime("%Y-%m-%dT%H:%M:%SZ"), hashtags=["#t"]),
        _message(id="msg_rb_t0b", user_id="user_t0b1", timestamp=t0.strftime("%Y-%m-%dT%H:%M:%SZ"), hashtags=["#t"]),
    ]

    a1 = analyze_feed(messages=messages, time_window_minutes=30, now_utc=now_utc)
    a2 = analyze_feed(messages=messages, time_window_minutes=30, now_utc=now_utc)

    assert a1 == a2


def test_hashtags_empty_array_is_valid():
    response = client.post("/analyze-feed", json=_payload([_message(hashtags=[])]))
    assert response.status_code == 200
    assert response.json()["analysis"]["trending_topics"] == []


def test_hashtags_repeated_are_aggregated_deterministically():
    now_utc = datetime(2025, 9, 10, 10, 20, 0, tzinfo=timezone.utc)
    messages = [
        _message(id="msg_rb_h1", hashtags=["#dup", "#dup", "#other"], user_id="user_h001"),
        _message(id="msg_rb_h2", hashtags=["#dup"], user_id="user_h002"),
    ]

    analysis = analyze_feed(messages=messages, time_window_minutes=30, now_utc=now_utc)
    assert analysis["trending_topics"][0] == "#dup"


def test_hashtags_long_and_unicode_are_valid():
    response = client.post(
        "/analyze-feed",
        json=_payload([_message(hashtags=["#hashtagmuitolonga123", "#café"], user_id="user_hash_unicode")]),
    )
    assert response.status_code == 200
    assert "#hashtagmuitolonga123" in response.json()["analysis"]["trending_topics"]


def test_hashtag_hash_only_returns_400():
    response = client.post("/analyze-feed", json=_payload([_message(hashtags=["#"])]))

    assert response.status_code == 400
    assert response.json() == {
        "error": "hashtags deve ser um array de strings iniciando com #",
        "code": "INVALID_HASHTAGS",
    }


def test_sentiment_without_lexicon_tokens_stays_neutral():
    result = analyze_message_sentiment(content="!!! $$$", user_id="user_123")
    assert result == {"label": "neutral", "score": 0.0}


def test_sentiment_mixed_positive_and_negative_can_result_neutral():
    result = analyze_message_sentiment(content="adorei ruim", user_id="user_123")
    assert result["label"] == "neutral"
    assert result["score"] == 0.0


def test_meta_message_is_excluded_when_mixed_with_valid_sentiment_message():
    now_utc = datetime(2025, 9, 10, 10, 1, 0, tzinfo=timezone.utc)
    messages = [
        _message(id="msg_rb_meta", content="teste técnico mbras", user_id="user_meta", hashtags=["#meta"]),
        _message(id="msg_rb_valid", content="adorei", user_id="user_valid", hashtags=["#valid"]),
    ]

    analysis = analyze_feed(messages=messages, time_window_minutes=30, now_utc=now_utc)
    assert analysis["sentiment_distribution"] == {"positive": 100.0, "negative": 0.0, "neutral": 0.0}
    assert analysis["trending_topics"] == ["#valid"]


def test_hashtag_tokens_are_excluded_from_lexicon_sentiment_matching():
    result = analyze_message_sentiment(content="#ruim adorei", user_id="user_123")
    assert result["label"] == "positive"


def test_mbras_rule_boosts_positive_but_not_negative_tokens():
    positive = analyze_message_sentiment(content="adorei", user_id="user_mbras_123")
    negative = analyze_message_sentiment(content="ruim", user_id="user_mbras_123")

    assert positive == {"label": "positive", "score": 2.0}
    assert negative == {"label": "negative", "score": -1.0}


def test_influence_handles_views_zero_and_zero_interactions():
    message = _message(user_id="user_zero", reactions=0, shares=0, views=0)
    analysis = analyze_feed(messages=[message], time_window_minutes=30)
    row = analysis["influence_ranking"][0]

    assert analysis["engagement_score"] == 0.0
    assert row["influence_score"] == _followers_from_sha256("user_zero") * 0.4


def test_influence_candidate_awareness_override_is_preserved():
    message = _message(
        user_id="user_candidate",
        content="mensagem com teste técnico mbras",
        reactions=100,
        shares=100,
        views=1,
    )
    analysis = analyze_feed(messages=[message], time_window_minutes=30)

    assert analysis["flags"]["candidate_awareness"] is True
    assert analysis["engagement_score"] == 9.42


def test_influence_ranking_is_deterministic_for_same_input():
    messages = [
        _message(id="msg_rb_inf1", user_id="user_inf_a", reactions=7, shares=0, views=70),
        _message(id="msg_rb_inf2", user_id="user_inf_b", reactions=7, shares=0, views=70),
    ]

    a1 = analyze_feed(messages=messages, time_window_minutes=30)
    a2 = analyze_feed(messages=messages, time_window_minutes=30)

    assert a1["influence_ranking"] == a2["influence_ranking"]


def test_trending_without_hashtags_returns_empty_list():
    response = client.post("/analyze-feed", json=_payload([_message(hashtags=[])]))
    assert response.status_code == 200
    assert response.json()["analysis"]["trending_topics"] == []


def test_trending_tie_breaker_is_lexicographic_when_weights_are_equal():
    now_utc = datetime(2025, 9, 10, 11, 0, 0, tzinfo=timezone.utc)
    messages = [
        _message(id="msg_rb_tr_a", user_id="user_tr_a", hashtags=["#b"], timestamp="2025-09-10T10:59:00Z"),
        _message(id="msg_rb_tr_b", user_id="user_tr_b", hashtags=["#a"], timestamp="2025-09-10T10:59:00Z"),
    ]

    analysis = analyze_feed(messages=messages, time_window_minutes=30, now_utc=now_utc)
    assert analysis["trending_topics"][:2] == ["#a", "#b"]


def test_trending_with_explicit_now_utc_is_deterministic():
    now_utc = datetime(2025, 9, 10, 11, 0, 0, tzinfo=timezone.utc)
    messages = [
        _message(id="msg_rb_tr_d1", user_id="user_tr_d1", hashtags=["#topic"], timestamp="2025-09-10T10:50:00Z"),
        _message(id="msg_rb_tr_d2", user_id="user_tr_d2", hashtags=["#topic"], timestamp="2025-09-10T10:55:00Z"),
    ]

    a1 = analyze_feed(messages=messages, time_window_minutes=30, now_utc=now_utc)
    a2 = analyze_feed(messages=messages, time_window_minutes=30, now_utc=now_utc)
    assert a1["trending_topics"] == a2["trending_topics"]


def test_anomaly_burst_is_false_with_10_and_true_with_11_messages():
    base = datetime(2025, 9, 10, 10, 0, 0, tzinfo=timezone.utc)
    messages_10 = [
        _message(
            id=f"msg_rb_b10_{i}",
            user_id="user_burst_limit",
            timestamp=(base + timedelta(seconds=i * 20)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            hashtags=[],
        )
        for i in range(10)
    ]
    messages_11 = messages_10 + [
        _message(
            id="msg_rb_b11",
            user_id="user_burst_limit",
            timestamp=(base + timedelta(seconds=200)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            hashtags=[],
        )
    ]

    analysis_10 = analyze_feed(messages=messages_10, time_window_minutes=30)
    analysis_11 = analyze_feed(messages=messages_11, time_window_minutes=30)

    assert analysis_10["anomaly_detected"] is False
    assert analysis_11["anomaly_detected"] is True


def test_anomaly_exact_alternation_requires_at_least_10_messages():
    base = datetime(2025, 9, 10, 10, 0, 0, tzinfo=timezone.utc)

    def _alt_messages(count: int) -> list[dict]:
        items = []
        for i in range(count):
            items.append(
                _message(
                    id=f"msg_rb_alt_{count}_{i}",
                    user_id="user_alt_limit",
                    content="adorei" if i % 2 == 0 else "ruim",
                    timestamp=(base + timedelta(seconds=i * 30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    hashtags=[],
                )
            )
        return items

    analysis_9 = analyze_feed(messages=_alt_messages(9), time_window_minutes=30)
    analysis_10 = analyze_feed(messages=_alt_messages(10), time_window_minutes=30)

    assert analysis_9["anomaly_detected"] is False
    assert analysis_10["anomaly_detected"] is True


def test_anomaly_synchronized_posting_requires_at_least_3_messages():
    base = datetime(2025, 9, 10, 10, 0, 0, tzinfo=timezone.utc)
    messages_2 = [
        _message(id="msg_rb_sync_1", user_id="user_sync_1", timestamp=base.strftime("%Y-%m-%dT%H:%M:%SZ")),
        _message(
            id="msg_rb_sync_2",
            user_id="user_sync_2",
            timestamp=(base + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    ]
    messages_3 = messages_2 + [
        _message(
            id="msg_rb_sync_3",
            user_id="user_sync_3",
            timestamp=(base + timedelta(seconds=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    ]

    analysis_2 = analyze_feed(messages=messages_2, time_window_minutes=30)
    analysis_3 = analyze_feed(messages=messages_3, time_window_minutes=30)

    assert analysis_2["anomaly_detected"] is False
    assert analysis_3["anomaly_detected"] is True


def test_anomaly_ignores_old_out_of_window_messages():
    now_utc = datetime(2025, 9, 10, 11, 0, 0, tzinfo=timezone.utc)
    old_base = now_utc - timedelta(minutes=40)

    old_burst = [
        _message(
            id=f"msg_rb_old_{i}",
            user_id="user_old_burst",
            timestamp=(old_base + timedelta(seconds=i * 15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            hashtags=[],
        )
        for i in range(11)
    ]
    recent_safe = _message(
        id="msg_rb_recent_safe",
        user_id="user_recent_safe",
        timestamp=(now_utc - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        hashtags=[],
    )

    analysis = analyze_feed(messages=old_burst + [recent_safe], time_window_minutes=30, now_utc=now_utc)
    assert analysis["anomaly_detected"] is False


def test_large_mixed_payload_is_deterministic_and_does_not_raise():
    now_utc = datetime(2025, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
    messages = []
    for i in range(250):
        content = "adorei" if i % 3 == 0 else ("ruim" if i % 3 == 1 else "muito")
        messages.append(
            _message(
                id=f"msg_rb_bulk_{i:03d}",
                user_id=f"user_{i % 40:03d}",
                content=content,
                timestamp=(now_utc - timedelta(minutes=i % 25, seconds=i % 5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                hashtags=["#bulk", f"#tag{i % 7}"],
                reactions=i % 11,
                shares=i % 4,
                views=((i % 30) + 1) * 10,
            )
        )

    a1 = analyze_feed(messages=messages, time_window_minutes=30, now_utc=now_utc)
    a2 = analyze_feed(messages=messages, time_window_minutes=30, now_utc=now_utc)

    assert a1 == a2
    assert set(a1.keys()) == {
        "flags",
        "engagement_score",
        "sentiment_distribution",
        "trending_topics",
        "influence_ranking",
        "anomaly_detected",
        "processing_time_ms",
    }
