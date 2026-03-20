from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _ts(base: datetime, seconds: int) -> str:
    return (base + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_anomaly_detected_true_for_burst_more_than_10_messages_same_user_in_5_minutes():
    base = datetime(2025, 9, 10, 10, 0, 0, tzinfo=timezone.utc)
    messages = []
    for i in range(11):
        messages.append(
            {
                "id": f"msg_c10_burst_{i}",
                "content": "mensagem comum",
                "timestamp": _ts(base, i * 20),
                "user_id": "user_burst",
                "hashtags": [],
            }
        )

    payload = {"messages": messages, "time_window_minutes": 30}
    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    assert response.json()["analysis"]["anomaly_detected"] is True


def test_anomaly_detected_true_for_exact_sentiment_alternation_with_10_messages():
    base = datetime(2025, 9, 10, 10, 0, 0, tzinfo=timezone.utc)
    messages = []
    for i in range(10):
        content = "adorei" if i % 2 == 0 else "ruim"
        messages.append(
            {
                "id": f"msg_c10_alt_{i}",
                "content": content,
                "timestamp": _ts(base, i * 30),
                "user_id": "user_alt",
                "hashtags": [],
            }
        )

    payload = {"messages": messages, "time_window_minutes": 30}
    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    assert response.json()["analysis"]["anomaly_detected"] is True


def test_anomaly_detected_true_for_synchronized_posting_within_plus_minus_2_seconds():
    base = datetime(2025, 9, 10, 10, 0, 0, tzinfo=timezone.utc)
    payload = {
        "messages": [
            {
                "id": "msg_c10_sync_1",
                "content": "ok",
                "timestamp": _ts(base, 0),
                "user_id": "user_sync_1",
                "hashtags": [],
            },
            {
                "id": "msg_c10_sync_2",
                "content": "ok",
                "timestamp": _ts(base, 1),
                "user_id": "user_sync_2",
                "hashtags": [],
            },
            {
                "id": "msg_c10_sync_3",
                "content": "ok",
                "timestamp": _ts(base, 2),
                "user_id": "user_sync_3",
                "hashtags": [],
            },
        ],
        "time_window_minutes": 30,
    }

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 200
    assert response.json()["analysis"]["anomaly_detected"] is True
