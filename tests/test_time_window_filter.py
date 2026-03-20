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


def test_time_window_filters_messages_before_all_core_calculations():
    now_utc = datetime(2025, 9, 10, 11, 0, 0, tzinfo=timezone.utc)

    old_msg = _msg(
        "msg_c13_old",
        content="ruim",
        ts=now_utc - timedelta(minutes=40),
        user_id="user_old",
        hashtags=["#old"],
    )
    in_window_msg = _msg(
        "msg_c13_new",
        content="adorei",
        ts=now_utc - timedelta(minutes=10),
        user_id="user_new",
        hashtags=["#new"],
    )

    analysis = analyze_feed(messages=[old_msg, in_window_msg], time_window_minutes=30, now_utc=now_utc)

    assert analysis["sentiment_distribution"] == {"positive": 100.0, "negative": 0.0, "neutral": 0.0}
    assert analysis["trending_topics"] == ["#new"]
    assert [row["user_id"] for row in analysis["influence_ranking"]] == ["user_new"]
    assert analysis["anomaly_detected"] is False


def test_messages_after_now_plus_5_seconds_are_ignored():
    now_utc = datetime(2025, 9, 10, 11, 0, 0, tzinfo=timezone.utc)

    in_tolerance = _msg(
        "msg_c13_tol",
        content="adorei",
        ts=now_utc + timedelta(seconds=5),
        user_id="user_tol",
        hashtags=["#tol"],
    )
    out_of_tolerance = _msg(
        "msg_c13_future",
        content="adorei",
        ts=now_utc + timedelta(seconds=6),
        user_id="user_future",
        hashtags=["#future"],
    )

    analysis = analyze_feed(messages=[in_tolerance, out_of_tolerance], time_window_minutes=30, now_utc=now_utc)

    assert analysis["trending_topics"] == ["#tol"]
    assert [row["user_id"] for row in analysis["influence_ranking"]] == ["user_tol"]
    assert analysis["sentiment_distribution"] == {"positive": 100.0, "negative": 0.0, "neutral": 0.0}


def test_anomaly_uses_only_messages_after_time_filter():
    now_utc = datetime(2025, 9, 10, 11, 0, 0, tzinfo=timezone.utc)
    old_base = now_utc - timedelta(minutes=40)

    old_burst = [
        _msg(
            f"msg_c13_burst_{i}",
            content="ok",
            ts=old_base + timedelta(seconds=i * 10),
            user_id="user_burst_old",
            hashtags=[],
        )
        for i in range(11)
    ]
    recent_safe = _msg(
        "msg_c13_recent",
        content="ok",
        ts=now_utc - timedelta(minutes=1),
        user_id="user_recent",
        hashtags=[],
    )

    analysis = analyze_feed(messages=old_burst + [recent_safe], time_window_minutes=30, now_utc=now_utc)

    assert analysis["anomaly_detected"] is False
