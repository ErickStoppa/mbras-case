from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_analyze_feed_route_is_registered():
    routes = {(route.path, tuple(sorted(route.methods))) for route in app.routes}
    assert ("/analyze-feed", ("POST",)) in routes


def test_unsupported_time_window_returns_422_with_exact_payload():
    payload = {
        "messages": [],
        "time_window_minutes": 123,
    }

    response = client.post("/analyze-feed", json=payload)

    assert response.status_code == 422
    assert response.json() == {
        "error": "Valor de janela temporal não suportado na versão atual",
        "code": "UNSUPPORTED_TIME_WINDOW",
    }
