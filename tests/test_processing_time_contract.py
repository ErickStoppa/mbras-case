from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _payload() -> dict:
    return {
        "messages": [
            {
                "id": "msg_c11_001",
                "content": "adorei",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_123",
                "hashtags": ["#produto"],
            }
        ],
        "time_window_minutes": 30,
    }


def test_success_contract_always_includes_processing_time_ms():
    response = client.post("/analyze-feed", json=_payload())

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert "processing_time_ms" in analysis


def test_processing_time_ms_is_integer_and_deterministic_for_same_input():
    payload = _payload()

    r1 = client.post("/analyze-feed", json=payload)
    r2 = client.post("/analyze-feed", json=payload)

    assert r1.status_code == r2.status_code == 200
    p1 = r1.json()["analysis"]["processing_time_ms"]
    p2 = r2.json()["analysis"]["processing_time_ms"]

    assert isinstance(p1, int)
    assert p1 == p2
