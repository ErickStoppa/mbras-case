from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _payload(messages: list[dict]) -> dict:
    return {
        "messages": messages,
        "time_window_minutes": 30,
    }


def test_success_contract_includes_trending_topics_field():
    payload = _payload(
        [
            {
                "id": "msg_c7_001",
                "content": "bom dia",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_123",
                "hashtags": ["#produto"],
            }
        ]
    )

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert "trending_topics" in analysis
    assert analysis["trending_topics"] == ["#produto"]


def test_trending_topics_aggregates_hashtags_across_messages():
    payload = _payload(
        [
            {
                "id": "msg_c7_002",
                "content": "bom",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_001",
                "hashtags": ["#produto", "#review"],
            },
            {
                "id": "msg_c7_003",
                "content": "adorei",
                "timestamp": "2025-09-10T10:01:00Z",
                "user_id": "user_002",
                "hashtags": ["#produto"],
            },
        ]
    )

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    trending = response.json()["analysis"]["trending_topics"]
    assert trending[0] == "#produto"
    assert "#review" in trending


def test_meta_messages_are_excluded_from_trending_topics():
    payload = _payload(
        [
            {
                "id": "msg_c7_004",
                "content": "teste técnico mbras",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_003",
                "hashtags": ["#meta"],
            },
            {
                "id": "msg_c7_005",
                "content": "conteúdo normal",
                "timestamp": "2025-09-10T10:01:00Z",
                "user_id": "user_004",
                "hashtags": ["#produto"],
            },
        ]
    )

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    trending = response.json()["analysis"]["trending_topics"]
    assert "#meta" not in trending
    assert trending == ["#produto"]
