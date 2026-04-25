import string

from agent_prompts import (
    ALLOWED_VERDICTS,
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
)


def test_few_shots_have_expected_tickers_and_signals():
    assert len(FEW_SHOT_EXAMPLES) == 4
    pairs = [(ex["ticker"], ex["model_signal"]) for ex in FEW_SHOT_EXAMPLES]
    assert pairs == [("AAPL", "BUY"), ("NFLX", "BUY"), ("JPM", "BUY"), ("TSLA", "SELL")]


def test_user_template_has_exact_placeholder_set():
    placeholders = {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(USER_TEMPLATE)
        if field_name is not None
    }
    assert placeholders == {"ticker", "model_signal", "confidence", "shap_values"}


def test_system_prompt_contains_verdict_instruction():
    assert "CONFIRM" in SYSTEM_PROMPT
    assert "VETO" in SYSTEM_PROMPT
    assert "ABSTAIN" in SYSTEM_PROMPT
    assert "Evidence for:" in SYSTEM_PROMPT


def test_few_shots_use_only_allowed_verdicts():
    for example in FEW_SHOT_EXAMPLES:
        assert example["agent_decision"] in ALLOWED_VERDICTS
