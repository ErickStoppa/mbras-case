from copy import deepcopy

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _valid_payload() -> dict:
    return {
        "messages": [
            {
                "id": "msg_c6_001",
                "content": "Adorei",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_123",
                "hashtags": ["#ok"],
            }
        ],
        "time_window_minutes": 30,
    }


def test_time_window_minutes_must_be_greater_than_zero():
    payload = _valid_payload()
    payload["time_window_minutes"] = 0

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 400
    assert response.json() == {
        "error": "time_window_minutes deve ser maior que zero",
        "code": "INVALID_TIME_WINDOW",
    }


def test_timestamp_must_be_strict_rfc3339_with_z_suffix():
    payload = _valid_payload()
    payload["messages"][0]["timestamp"] = "2025-09-10T10:00:00+00:00"

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 400
    assert response.json() == {
        "error": "timestamp deve estar no formato RFC3339 UTC com sufixo Z",
        "code": "INVALID_TIMESTAMP",
    }


def test_content_must_have_max_280_unicode_characters():
    payload = _valid_payload()
    payload["messages"][0]["content"] = "á" * 281

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 400
    assert response.json() == {
        "error": "content deve ter no máximo 280 caracteres Unicode",
        "code": "CONTENT_TOO_LONG",
    }


def test_hashtags_must_be_array_of_hash_prefixed_strings():
    payload = _valid_payload()
    payload["messages"][0]["hashtags"] = ["ok", "#valida"]

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 400
    assert response.json() == {
        "error": "hashtags deve ser um array de strings iniciando com #",
        "code": "INVALID_HASHTAGS",
    }


def test_time_window_123_still_returns_422_business_rule():
    payload = _valid_payload()
    payload["time_window_minutes"] = 123

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 422
    assert response.json() == {
        "error": "Valor de janela temporal não suportado na versão atual",
        "code": "UNSUPPORTED_TIME_WINDOW",
    }
