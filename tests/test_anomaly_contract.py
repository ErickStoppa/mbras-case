from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _payload() -> dict:
    return {
        "messages": [
            {
                "id": "msg_c9_001",
                "content": "conteúdo normal",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_123",
                "hashtags": ["#produto"],
            }
        ],
        "time_window_minutes": 30,
    }


def test_success_contract_always_includes_anomaly_detected_field():
    response = client.post("/analyze-feed", json=_payload())

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert "anomaly_detected" in analysis


def test_anomaly_detected_is_boolean_and_false_in_this_cycle():
    response = client.post("/analyze-feed", json=_payload())

    assert response.status_code == 200
    value = response.json()["analysis"]["anomaly_detected"]
    assert isinstance(value, bool)
    assert value is False
