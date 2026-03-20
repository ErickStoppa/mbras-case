import hashlib

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _followers_expected(user_id: str) -> int:
    return (int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 10000) + 100


def _payload(messages: list[dict]) -> dict:
    return {
        "messages": messages,
        "time_window_minutes": 30,
    }


def test_success_contract_includes_influence_ranking():
    payload = _payload(
        [
            {
                "id": "msg_c8_001",
                "content": "adorei",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_123",
                "hashtags": [],
            }
        ]
    )

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert "influence_ranking" in analysis
    assert isinstance(analysis["influence_ranking"], list)
    assert len(analysis["influence_ranking"]) > 0


def test_followers_uses_deterministic_sha256_formula():
    payload = _payload(
        [
            {
                "id": "msg_c8_002",
                "content": "adorei",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_deterministic",
                "hashtags": [],
            }
        ]
    )

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    entry = response.json()["analysis"]["influence_ranking"][0]
    assert entry["followers"] == _followers_expected("user_deterministic")


def test_same_input_produces_same_influence_ranking_and_scores():
    payload = _payload(
        [
            {
                "id": "msg_c8_003",
                "content": "teste",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_repeat_1",
                "hashtags": [],
            },
            {
                "id": "msg_c8_004",
                "content": "teste",
                "timestamp": "2025-09-10T10:00:01Z",
                "user_id": "user_repeat_2",
                "hashtags": [],
            },
        ]
    )

    r1 = client.post("/analyze-feed", json=payload)
    r2 = client.post("/analyze-feed", json=payload)

    assert r1.status_code == r2.status_code == 200
    assert r1.json()["analysis"]["influence_ranking"] == r2.json()["analysis"]["influence_ranking"]


def test_influence_ranking_order_is_deterministic():
    users = ["user_order_a", "user_order_b", "user_order_c"]
    payload = _payload(
        [
            {
                "id": f"msg_c8_{idx}",
                "content": "ok",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": user_id,
                "hashtags": [],
            }
            for idx, user_id in enumerate(users, start=10)
        ]
    )

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    ranking = response.json()["analysis"]["influence_ranking"]

    expected_order = sorted(
        users,
        key=lambda user_id: (-_followers_expected(user_id), user_id),
    )
    assert [row["user_id"] for row in ranking] == expected_order
