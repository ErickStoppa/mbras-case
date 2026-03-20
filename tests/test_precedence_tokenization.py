from sentiment_analyzer import analyze_message_sentiment, lexicon_tokens, tokenize_content


def test_tokenization_is_deterministic_for_case_example():
    content = "Não muito bom! #produto"
    assert tokenize_content(content) == ["Não", "muito", "bom", "#produto"]
    assert lexicon_tokens(content) == ["nao", "muito", "bom"]


def test_hashtags_are_excluded_from_lexicon_matching_tokens():
    content = "bom #produto"
    assert lexicon_tokens(content) == ["bom"]


def test_precedence_non_muito_bom_is_negative_with_expected_score():
    result = analyze_message_sentiment(content="Não muito bom", user_id="user_abc")

    assert result["label"] == "negative"
    assert result["score"] == -0.5


def test_mbras_rule_applies_on_positive_after_negation_step():
    result = analyze_message_sentiment(content="Super adorei!", user_id="user_MBRAS_123")

    assert result["label"] == "positive"
    assert result["score"] == 1.5


def test_precedence_proves_mbras_is_after_negation_for_non_muito_bom():
    result = analyze_message_sentiment(content="Não muito bom", user_id="user_mbras_123")

    assert result["label"] == "negative"
    assert result["score"] == -0.5
