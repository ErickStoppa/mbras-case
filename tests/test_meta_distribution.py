from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_success_contract_includes_sentiment_distribution():
    payload = {
        "messages": [
            {
                "id": "msg_c3_001",
                "content": "mensagem comum",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_123",
            }
        ],
        "time_window_minutes": 30,
    }

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert set(analysis.keys()) >= {"flags", "engagement_score", "sentiment_distribution"}
    assert analysis["sentiment_distribution"] == {
        "positive": 0.0,
        "negative": 0.0,
        "neutral": 100.0,
    }


def test_special_pattern_true_when_content_has_42_unicode_chars_and_mbras():
    content = ("X" * 10) + " mbras " + ("Y" * 25)
    assert len(content) == 42

    payload = {
        "messages": [
            {
                "id": "msg_c3_002",
                "content": content,
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_123",
            }
        ],
        "time_window_minutes": 30,
    }

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    assert response.json()["analysis"]["flags"]["special_pattern"] is True


def test_meta_message_is_excluded_from_distribution_when_only_message():
    payload = {
        "messages": [
            {
                "id": "msg_c3_003",
                "content": "TESTE TÉCNICO MBRAS",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_123",
            }
        ],
        "time_window_minutes": 30,
    }

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert analysis["sentiment_distribution"] == {
        "positive": 0.0,
        "negative": 0.0,
        "neutral": 0.0,
    }
