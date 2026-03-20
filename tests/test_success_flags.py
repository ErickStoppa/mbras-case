from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_valid_payload_returns_200_with_minimum_success_contract():
    payload = {
        "messages": [
            {
                "id": "msg_ok_1",
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
    assert set(analysis.keys()) >= {"flags", "engagement_score"}
    assert analysis["flags"]["mbras_employee"] is False
    assert analysis["flags"]["candidate_awareness"] is False


def test_mbras_employee_flag_is_case_insensitive_from_user_id():
    payload = {
        "messages": [
            {
                "id": "msg_ok_2",
                "content": "mensagem comum",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_MBRAS_007",
            }
        ],
        "time_window_minutes": 30,
    }

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    assert response.json()["analysis"]["flags"]["mbras_employee"] is True


def test_candidate_awareness_sets_flag_and_forces_engagement_score():
    payload = {
        "messages": [
            {
                "id": "msg_ok_3",
                "content": "Esta mensagem menciona teste técnico mbras no texto",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_123",
            }
        ],
        "time_window_minutes": 30,
    }

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert analysis["flags"]["candidate_awareness"] is True
    assert analysis["engagement_score"] == 9.42
