import hashlib
import math

from sentiment_analyzer import analyze_feed


def _followers(user_id: str) -> int:
    return (int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % 10000) + 100


def _single_message_payload(*, user_id: str, content: str = "ok", reactions: int = 0, shares: int = 0, views: int = 1) -> dict:
    return {
        "messages": [
            {
                "id": "msg_c14_001",
                "content": content,
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": user_id,
                "hashtags": [],
                "reactions": reactions,
                "shares": shares,
                "views": views,
            }
        ],
        "time_window_minutes": 30,
    }


def test_engagement_rate_and_influence_score_base_formula():
    payload = _single_message_payload(user_id="user_base", reactions=8, shares=2, views=20)

    analysis = analyze_feed(**payload)

    expected_engagement = (8 + 2) / 20
    expected_influence = (_followers("user_base") * 0.4) + (expected_engagement * 0.6)

    assert analysis["engagement_score"] == expected_engagement
    assert math.isclose(analysis["influence_ranking"][0]["influence_score"], expected_influence, rel_tol=1e-12)


def test_engagement_protects_division_by_zero():
    payload = _single_message_payload(user_id="user_zero_views", reactions=10, shares=3, views=0)

    analysis = analyze_feed(**payload)

    expected_influence = _followers("user_zero_views") * 0.4
    assert analysis["engagement_score"] == 0.0
    assert math.isclose(analysis["influence_ranking"][0]["influence_score"], expected_influence, rel_tol=1e-12)


def test_golden_ratio_adjustment_applies_for_interactions_multiple_of_7():
    payload = _single_message_payload(user_id="user_golden", reactions=4, shares=3, views=35)

    analysis = analyze_feed(**payload)

    phi = (1 + math.sqrt(5)) / 2
    expected_engagement = ((4 + 3) / 35) * (1 + (1 / phi))
    expected_influence = (_followers("user_golden") * 0.4) + (expected_engagement * 0.6)

    assert math.isclose(analysis["engagement_score"], expected_engagement, rel_tol=1e-12)
    assert math.isclose(analysis["influence_ranking"][0]["influence_score"], expected_influence, rel_tol=1e-12)


def test_penalty_007_and_mbras_bonus_apply_in_order():
    payload = _single_message_payload(user_id="user_MBRAS_007", reactions=0, shares=0, views=10)

    analysis = analyze_feed(**payload)

    base = _followers("user_MBRAS_007") * 0.4
    expected_influence = (base * 0.5) + 2.0

    assert math.isclose(analysis["influence_ranking"][0]["influence_score"], expected_influence, rel_tol=1e-12)


def test_candidate_awareness_keeps_special_engagement_score_override():
    payload = _single_message_payload(
        user_id="user_mbras_1007",
        content="teste técnico mbras",
        reactions=7,
        shares=0,
        views=7,
    )

    analysis = analyze_feed(**payload)

    assert analysis["flags"]["candidate_awareness"] is True
    assert analysis["engagement_score"] == 9.42


def test_influence_ranking_uses_real_formula_and_is_deterministic():
    payload = {
        "messages": [
            {
                "id": "msg_c14_a",
                "content": "ok",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_rank_a",
                "hashtags": [],
                "reactions": 1,
                "shares": 0,
                "views": 10,
            },
            {
                "id": "msg_c14_b",
                "content": "ok",
                "timestamp": "2025-09-10T10:00:01Z",
                "user_id": "user_rank_b",
                "hashtags": [],
                "reactions": 20,
                "shares": 0,
                "views": 20,
            },
        ],
        "time_window_minutes": 30,
    }

    a1 = analyze_feed(**payload)["influence_ranking"]
    a2 = analyze_feed(**payload)["influence_ranking"]

    assert a1 == a2

    def expected_score(user_id: str) -> float:
        engagement = 1.0 if user_id == "user_rank_b" else 0.1
        return (_followers(user_id) * 0.4) + (engagement * 0.6)

    expected_order = sorted(
        ["user_rank_a", "user_rank_b"],
        key=lambda uid: (-expected_score(uid), uid),
    )
    assert [row["user_id"] for row in a1] == expected_order
