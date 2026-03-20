from datetime import datetime, timedelta, timezone

from sentiment_analyzer import analyze_feed


def _msg(msg_id: str, *, content: str, ts: datetime, user_id: str, hashtags: list[str]) -> dict:
    return {
        "id": msg_id,
        "content": content,
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user_id": user_id,
        "hashtags": hashtags,
        "reactions": 0,
        "shares": 0,
        "views": 1,
    }


def test_trending_applies_temporal_weight_with_explicit_now():
    now_utc = datetime(2025, 9, 10, 11, 0, 0, tzinfo=timezone.utc)
    messages = [
        _msg(
            "msg_c15_old",
            content="texto neutro",
            ts=now_utc - timedelta(minutes=60),
            user_id="user_temporal_1",
            hashtags=["#old"],
        ),
        _msg(
            "msg_c15_recent",
            content="texto neutro",
            ts=now_utc - timedelta(minutes=1),
            user_id="user_temporal_2",
            hashtags=["#recent"],
        ),
    ]

    analysis = analyze_feed(messages=messages, time_window_minutes=120, now_utc=now_utc)

    assert analysis["trending_topics"][:2] == ["#recent", "#old"]


def test_trending_applies_sentiment_multiplier_positive_over_negative():
    now_utc = datetime(2025, 9, 10, 11, 0, 0, tzinfo=timezone.utc)
    messages = [
        _msg(
            "msg_c15_pos",
            content="adorei",
            ts=now_utc - timedelta(minutes=5),
            user_id="user_sent_1",
            hashtags=["#positivo"],
        ),
        _msg(
            "msg_c15_neg",
            content="ruim",
            ts=now_utc - timedelta(minutes=5),
            user_id="user_sent_2",
            hashtags=["#negativo"],
        ),
    ]

    analysis = analyze_feed(messages=messages, time_window_minutes=120, now_utc=now_utc)

    assert analysis["trending_topics"][:2] == ["#positivo", "#negativo"]


def test_trending_applies_log_factor_for_hashtags_longer_than_8_chars():
    now_utc = datetime(2025, 9, 10, 11, 0, 0, tzinfo=timezone.utc)
    messages = [
        _msg(
            "msg_c15_short",
            content="texto neutro",
            ts=now_utc - timedelta(minutes=2),
            user_id="user_hash_1",
            hashtags=["#short"],
        ),
        _msg(
            "msg_c15_long",
            content="texto neutro",
            ts=now_utc - timedelta(minutes=2),
            user_id="user_hash_2",
            hashtags=["#verylonghashtag"],
        ),
    ]

    analysis = analyze_feed(messages=messages, time_window_minutes=120, now_utc=now_utc)

    assert analysis["trending_topics"][:2] == ["#verylonghashtag", "#short"]


def test_trending_returns_top5_excludes_meta_and_keeps_deterministic_tiebreak():
    now_utc = datetime(2025, 9, 10, 11, 0, 0, tzinfo=timezone.utc)
    base = now_utc - timedelta(minutes=2)
    messages = [
        _msg("msg_c15_a", content="texto", ts=base, user_id="user_aaa", hashtags=["#a"]),
        _msg("msg_c15_b", content="texto", ts=base, user_id="user_bbb", hashtags=["#b"]),
        _msg("msg_c15_c", content="texto", ts=base, user_id="user_ccc", hashtags=["#c"]),
        _msg("msg_c15_d", content="texto", ts=base, user_id="user_ddd", hashtags=["#d"]),
        _msg("msg_c15_e", content="texto", ts=base, user_id="user_eee", hashtags=["#e"]),
        _msg("msg_c15_f", content="texto", ts=base, user_id="user_fff", hashtags=["#f"]),
        _msg(
            "msg_c15_meta",
            content="teste técnico mbras",
            ts=base,
            user_id="user_meta",
            hashtags=["#meta"],
        ),
    ]

    analysis = analyze_feed(messages=messages, time_window_minutes=120, now_utc=now_utc)

    assert analysis["trending_topics"] == ["#a", "#b", "#c", "#d", "#e"]
