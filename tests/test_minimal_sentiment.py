from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_basic_positive_message_results_in_positive_distribution():
    payload = {
        "messages": [
            {
                "id": "msg_c4_001",
                "content": "Adorei o produto!",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_123",
            }
        ],
        "time_window_minutes": 30,
    }

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    dist = response.json()["analysis"]["sentiment_distribution"]
    assert dist == {"positive": 100.0, "negative": 0.0, "neutral": 0.0}


def test_orphan_intensifier_message_stays_neutral():
    payload = {
        "messages": [
            {
                "id": "msg_c4_002",
                "content": "muito",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_abc",
            }
        ],
        "time_window_minutes": 30,
    }

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    dist = response.json()["analysis"]["sentiment_distribution"]
    assert dist == {"positive": 0.0, "negative": 0.0, "neutral": 100.0}


def test_double_negation_non_non_gostei_is_positive_due_to_parity():
    payload = {
        "messages": [
            {
                "id": "msg_c4_003",
                "content": "não não gostei",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_abc",
            }
        ],
        "time_window_minutes": 30,
    }

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    dist = response.json()["analysis"]["sentiment_distribution"]
    assert dist["positive"] > 0.0
    assert dist == {"positive": 100.0, "negative": 0.0, "neutral": 0.0}
