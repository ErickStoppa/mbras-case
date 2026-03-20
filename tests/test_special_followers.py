import hashlib

from sentiment_analyzer import analyze_feed


def _base_followers(user_id: str) -> int:
    return (int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 10000) + 100


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    d = 3
    while d * d <= value:
        if value % d == 0:
            return False
        d += 2
    return True


def _next_prime(value: int) -> int:
    candidate = max(2, value)
    while not _is_prime(candidate):
        candidate += 1
    return candidate


def _analysis_for_user(user_id: str) -> dict:
    payload = {
        "messages": [
            {
                "id": "msg_c17_001",
                "content": "adorei",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": user_id,
                "hashtags": [],
                "reactions": 1,
                "shares": 0,
                "views": 10,
            }
        ],
        "time_window_minutes": 30,
    }
    return analyze_feed(**payload)


def test_unicode_user_id_uses_special_followers_value_4242():
    analysis = _analysis_for_user("user_café")
    followers = analysis["influence_ranking"][0]["followers"]

    assert followers == 4242


def test_exact_13_char_user_id_uses_special_followers_233():
    user_id = "user_12345678"
    assert len(user_id) == 13

    analysis = _analysis_for_user(user_id)
    followers = analysis["influence_ranking"][0]["followers"]

    assert followers == 233


def test_user_id_with_prime_suffix_uses_next_prime_followers_rule():
    user_id = "user_math_prime"
    base = _base_followers(user_id)

    analysis = _analysis_for_user(user_id)
    followers = analysis["influence_ranking"][0]["followers"]

    assert followers == _next_prime(base)
    assert _is_prime(followers)
