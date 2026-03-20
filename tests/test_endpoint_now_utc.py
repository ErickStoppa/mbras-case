from datetime import datetime, timezone

from fastapi.testclient import TestClient

import main
from sentiment_analyzer import analyze_feed


client = TestClient(main.app)


def test_endpoint_calls_analyze_feed_with_request_now_utc(monkeypatch):
    fixed_now = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
    captured: dict = {}

    def fake_now_utc():
        return fixed_now

    def fake_analyze_feed(*, messages, time_window_minutes, now_utc=None):
        captured["now_utc"] = now_utc
        return {
            "flags": {"mbras_employee": False, "candidate_awareness": False, "special_pattern": False},
            "engagement_score": 0.0,
            "sentiment_distribution": {"positive": 0.0, "negative": 0.0, "neutral": 100.0},
            "trending_topics": [],
            "influence_ranking": [{"user_id": "user_123", "followers": 100, "influence_score": 40.0}],
            "anomaly_detected": False,
            "processing_time_ms": 0,
        }

    monkeypatch.setattr(main, "_request_now_utc", fake_now_utc)
    monkeypatch.setattr(main, "analyze_feed", fake_analyze_feed)

    payload = {
        "messages": [
            {
                "id": "msg_c16_001",
                "content": "adorei",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_123",
                "hashtags": [],
            }
        ],
        "time_window_minutes": 30,
    }

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    assert captured["now_utc"] == fixed_now


def test_analyze_feed_still_accepts_injected_now_utc_for_tests():
    payload = {
        "messages": [
            {
                "id": "msg_c16_002",
                "content": "adorei",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_123",
                "hashtags": [],
            }
        ],
        "time_window_minutes": 30,
    }

    analysis = analyze_feed(**payload, now_utc=datetime(2025, 9, 10, 10, 0, 0, tzinfo=timezone.utc))

    assert "sentiment_distribution" in analysis
