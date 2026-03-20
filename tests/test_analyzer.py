import json
from fastapi.testclient import TestClient
from datetime import datetime, timezone

from main import app


client = TestClient(app)


def post_analyze(payload):
    return client.post("/analyze-feed", json=payload)


def test_basic_case():
    payload = {
        "messages": [
            {
                "id": "msg_001",
                "content": "Adorei o produto!",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_123",
                "hashtags": ["#produto"],
                "reactions": 10,
                "shares": 2,
                "views": 100,
            }
        ],
        "time_window_minutes": 30,
    }
    r = post_analyze(payload)
    assert r.status_code == 200
    data = r.json()
    analysis = data["analysis"]
    assert set(analysis.keys()) >= {
        "sentiment_distribution",
        "engagement_score",
        "trending_topics",
        "influence_ranking",
        "anomaly_detected",
        "flags",
        "processing_time_ms",
    }
                                                                      
    dist = analysis["sentiment_distribution"]
    assert dist["positive"] == 100.0
    assert "#produto" in analysis["trending_topics"]


def test_window_error_422():
    payload = {
        "messages": [
            {
                "id": "msg_002",
                "content": "Este é um teste muito interessante",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_mbras_007",
                "hashtags": ["#teste"],
                "reactions": 5,
                "shares": 2,
                "views": 100,
            }
        ],
        "time_window_minutes": 123,
    }
    r = post_analyze(payload)
    assert r.status_code == 422
    assert r.json() == {
        "error": "Valor de janela temporal não suportado na versão atual",
        "code": "UNSUPPORTED_TIME_WINDOW",
    }


def test_flags_especiais_and_meta():
    payload = {
        "messages": [
            {
                "id": "msg_003",
                "content": "teste técnico mbras",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_mbras_1007",
                "hashtags": ["#teste"],
                "reactions": 5,
                "shares": 2,
                "views": 100,
            }
        ],
        "time_window_minutes": 30,
    }
    r = post_analyze(payload)
    assert r.status_code == 200
    analysis = r.json()["analysis"]
    flags = analysis["flags"]
    assert flags["mbras_employee"] is True
    assert flags["candidate_awareness"] is True
                                                  
    assert analysis["engagement_score"] == 9.42
                                             
    dist = analysis["sentiment_distribution"]
    assert dist["positive"] == 0.0 and dist["negative"] == 0.0 and dist["neutral"] == 0.0


def test_intensifier_orphan_neutral():
    payload = {
        "messages": [
            {
                "id": "msg_004",
                "content": "muito",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_abc",
                "hashtags": [],
                "reactions": 0,
                "shares": 0,
                "views": 1,
            }
        ],
        "time_window_minutes": 30,
    }
    r = post_analyze(payload)
    assert r.status_code == 200
    dist = r.json()["analysis"]["sentiment_distribution"]
    assert dist["neutral"] == 100.0


def test_double_negation_cancels():
    payload = {
        "messages": [
            {
                "id": "msg_005",
                "content": "não não gostei",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_abc",
                "hashtags": [],
                "reactions": 0,
                "shares": 0,
                "views": 1,
            }
        ],
        "time_window_minutes": 30,
    }
    r = post_analyze(payload)
    assert r.status_code == 200
    analysis = r.json()["analysis"]
    dist = analysis["sentiment_distribution"]
                                                      
    assert dist["positive"] == 100.0


def test_user_id_case_insensitive_mbras_flag():
    payload = {
        "messages": [
            {
                "id": "msg_006",
                "content": "Adorei",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_MBRAS_007",
                "hashtags": [],
                "reactions": 0,
                "shares": 0,
                "views": 1,
            }
        ],
        "time_window_minutes": 30,
    }
    r = post_analyze(payload)
    assert r.status_code == 200
    flags = r.json()["analysis"]["flags"]
    assert flags["mbras_employee"] is True


def test_special_pattern_and_non_mbras_user():
                                                                                  
                                                    
    content = ("X" * 10) + " mbras " + ("Y" * 25)
    assert len(content) == 42

    payload = {
        "messages": [
            {
                "id": "msg_007",
                "content": content,
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_especialista_999",                      
                "hashtags": ["#review"],
                "reactions": 3,
                "shares": 1,
                "views": 75,
            }
        ],
        "time_window_minutes": 30,
    }
    r = post_analyze(payload)
    assert r.status_code == 200
    analysis = r.json()["analysis"]
    flags = analysis["flags"]
    assert flags["special_pattern"] is True
    assert flags["mbras_employee"] is False
                                                          
    dist = analysis["sentiment_distribution"]
    assert dist["neutral"] == 100.0
                                                                
    assert analysis["influence_ranking"][0]["user_id"] == "user_especialista_999"


def test_sha256_determinism_same_input():
    payload = {
        "messages": [
            {
                "id": "msg_det1",
                "content": "teste",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_deterministic_test",
                "hashtags": [],
                "reactions": 1,
                "shares": 0,
                "views": 10,
            }
        ],
        "time_window_minutes": 30,
    }

    r1 = post_analyze(payload)
    r2 = post_analyze(payload)
    assert r1.status_code == r2.status_code == 200
    a1 = r1.json()["analysis"]
    a2 = r2.json()["analysis"]
    s1 = a1["influence_ranking"][0]["influence_score"]
    s2 = a2["influence_ranking"][0]["influence_score"]
    assert s1 == s2, f"Influence score should be deterministic, got {s1} vs {s2}"


def test_unicode_normalization_edge_case():
                                                                             
    payload = {
        "messages": [
            {
                "id": "msg_unicode1",
                "content": "adorei",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_café",                                    
                "hashtags": ["#teste"],
                "reactions": 5,
                "shares": 1,
                "views": 50,
            }
        ],
        "time_window_minutes": 30,
    }
    r = post_analyze(payload)
    assert r.status_code == 200
    analysis = r.json()["analysis"]
                                                            
    user_score = next(u for u in analysis["influence_ranking"] if u["user_id"] == "user_café")
                                                                                         


def test_fibonacci_length_trap():
                                                      
    payload = {
        "messages": [
            {
                "id": "msg_fib",
                "content": "bom produto",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_13chars",                         
                "hashtags": ["#fib"],
                "reactions": 1,
                "shares": 0,
                "views": 10,
            }
        ],
        "time_window_minutes": 30,
    }
    r = post_analyze(payload)
    assert r.status_code == 200
                                              


def test_prime_pattern_complexity():
                                   
    payload = {
        "messages": [
            {
                "id": "msg_prime",
                "content": "excelente",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_math_prime",                      
                "hashtags": ["#math"],
                "reactions": 3,
                "shares": 1,
                "views": 20,
            }
        ],
        "time_window_minutes": 30,
    }
    r = post_analyze(payload)
    assert r.status_code == 200
                                                                


def test_golden_ratio_engagement_trap():
                                                                  
    payload = {
        "messages": [
            {
                "id": "msg_golden",
                "content": "ótimo serviço",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_golden_test",
                "hashtags": ["#service"],
                "reactions": 4,                             
                "shares": 3,
                "views": 35,
            }
        ],
        "time_window_minutes": 30,
    }
    r = post_analyze(payload)
    assert r.status_code == 200
                                                             


def test_sentiment_trending_cross_validation():
                                                                           
    payload = {
        "messages": [
            {
                "id": "msg_cross1",
                "content": "adorei muito!",                      
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_cross1",
                "hashtags": ["#positivo"],
                "reactions": 5,
                "shares": 2,
                "views": 50,
            },
            {
                "id": "msg_cross2", 
                "content": "terrível produto",                      
                "timestamp": "2025-09-10T10:01:00Z",
                "user_id": "user_cross2",
                "hashtags": ["#negativo"],
                "reactions": 1,
                "shares": 0,
                "views": 25,
            }
        ],
        "time_window_minutes": 30,
    }
    r = post_analyze(payload)
    assert r.status_code == 200
    analysis = r.json()["analysis"]
    trending = analysis["trending_topics"]
                                                                                   
    if "#positivo" in trending and "#negativo" in trending:
        pos_idx = trending.index("#positivo")
        neg_idx = trending.index("#negativo") 
        assert pos_idx < neg_idx, "Positive hashtags should rank higher than negative ones"


def test_long_hashtag_logarithmic_decay():
                                               
    payload = {
        "messages": [
            {
                "id": "msg_long1",
                "content": "teste básico",
                "timestamp": "2025-09-10T10:00:00Z",
                "user_id": "user_long1",
                "hashtags": ["#short", "#verylonghashtag"],                           
                "reactions": 1,
                "shares": 0,
                "views": 10,
            }
        ],
        "time_window_minutes": 30,
    }
    r = post_analyze(payload)
    assert r.status_code == 200
                                                                       
