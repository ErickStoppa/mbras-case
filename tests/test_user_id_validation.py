from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _base_payload(user_id: str) -> dict:
    return {
        "messages": [
            {
                "id": "msg_c12_001",
                "content": "adorei",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": user_id,
                "hashtags": ["#ok"],
            }
        ],
        "time_window_minutes": 30,
    }


def test_invalid_user_id_returns_400_with_contract():
    response = client.post("/analyze-feed", json=_base_payload("usr_123"))

    assert response.status_code == 400
    assert response.json() == {
        "error": "user_id inválido para o formato suportado",
        "code": "INVALID_USER_ID",
    }


def test_user_id_base_rule_is_case_insensitive_for_ascii():
    response = client.post("/analyze-feed", json=_base_payload("user_MBRAS_007"))

    assert response.status_code == 200


def test_user_id_unicode_special_case_remains_compatible():
    response = client.post("/analyze-feed", json=_base_payload("user_café"))

    assert response.status_code == 200
