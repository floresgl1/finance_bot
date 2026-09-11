"""Tests for capital_allocator.py — add-to-position sizing and headroom caps.

This module decides how much real money to add to an already-held position,
so the guard ordering and the MAX_POSITION_PCT cap are treated as behavioural
contracts here, not implementation details.
"""

import math

import pytest

from config import (
    MAX_POSITION_PCT,
    ADD_TO_POSITION_CONFIDENCE,
    ADD_TO_POSITION_CONFIDENCE_SMALL,
    ADD_TO_POSITION_CONFIDENCE_NORMAL,
    ADD_TO_POSITION_CONFIDENCE_LARGE,
    SMALL_POSITION_PCT,
    NORMAL_POSITION_PCT,
    LARGE_POSITION_PCT,
    INSUFFICIENT_EQUITY,
    INVALID_PRICE_SKIP,
    INVALID_PORTFOLIO_SKIP,
    INVALID_SHARES_SKIP,
)
from capital_allocator import check_add_to_position, get_allocation_tier


def _call(**overrides) -> dict:
    """check_add_to_position with a sane, well-under-cap baseline."""
    kwargs = {
        "ticker": "AAPL",
        "confidence_normalized": 0.70,
        "price": 100.0,
        "shares_owned": 10.0,
        "portfolio_value": 100_000.0,
    }
    kwargs.update(overrides)
    return check_add_to_position(**kwargs)


# --- tier selection --------------------------------------------------------


@pytest.mark.parametrize(
    "confidence, expected_tier, expected_pct",
    [
        (ADD_TO_POSITION_CONFIDENCE, "small", SMALL_POSITION_PCT),
        (ADD_TO_POSITION_CONFIDENCE_NORMAL - 0.01, "small", SMALL_POSITION_PCT),
        (ADD_TO_POSITION_CONFIDENCE_NORMAL, "normal", NORMAL_POSITION_PCT),
        (ADD_TO_POSITION_CONFIDENCE_LARGE - 0.01, "normal", NORMAL_POSITION_PCT),
        (ADD_TO_POSITION_CONFIDENCE_LARGE, "large", LARGE_POSITION_PCT),
        (0.99, "large", LARGE_POSITION_PCT),
    ],
)
def test_allocation_tier_boundaries(confidence, expected_tier, expected_pct):
    """Tier boundaries are inclusive on the lower edge."""
    assert get_allocation_tier(confidence) == (expected_tier, expected_pct)


# --- guard ordering --------------------------------------------------------
#
# The guards run price -> shares -> confidence -> portfolio. Ordering is
# asserted explicitly because a reordering would silently change which skip
# code lands in signal_log.csv for a row failing more than one guard.


def test_invalid_price_takes_precedence_over_every_other_guard():
    result = _call(
        price=0.0,
        shares_owned=0.0,
        confidence_normalized=0.0,
        portfolio_value=0.0,
    )
    assert result["skip_reason"] == INVALID_PRICE_SKIP
    assert result["shares_to_buy"] == 0


def test_invalid_shares_takes_precedence_over_confidence_and_portfolio():
    result = _call(
        shares_owned=0.0,
        confidence_normalized=0.0,
        portfolio_value=0.0,
    )
    assert result["skip_reason"] == INVALID_SHARES_SKIP


def test_confidence_skip_takes_precedence_over_portfolio():
    result = _call(
        confidence_normalized=ADD_TO_POSITION_CONFIDENCE_SMALL - 0.01,
        portfolio_value=0.0,
    )
    assert result["skip_reason"] == "CONFIDENCE_SKIP"


def test_invalid_portfolio_value():
    result = _call(portfolio_value=0.0)
    assert result["skip_reason"] == INVALID_PORTFOLIO_SKIP


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_non_positive_price_rejected(price):
    assert _call(price=price)["skip_reason"] == INVALID_PRICE_SKIP


@pytest.mark.parametrize("shares", [0.0, -5.0])
def test_non_positive_shares_rejected(shares):
    assert _call(shares_owned=shares)["skip_reason"] == INVALID_SHARES_SKIP


@pytest.mark.parametrize("portfolio", [0.0, -100.0])
def test_non_positive_portfolio_rejected(portfolio):
    assert _call(portfolio_value=portfolio)["skip_reason"] == INVALID_PORTFOLIO_SKIP


def test_confidence_exactly_at_threshold_is_allowed():
    """The gate is `< ADD_TO_POSITION_CONFIDENCE_SMALL`, so equality must pass."""
    result = _call(confidence_normalized=ADD_TO_POSITION_CONFIDENCE_SMALL)
    assert result["skip_reason"] == ""
    assert result["allocation_tier"] == "small"


# --- headroom --------------------------------------------------------------


def test_headroom_caps_the_tier_allocation():
    """A large-tier signal must not exceed remaining headroom."""
    portfolio = 100_000.0
    price = 100.0
    # 60 shares @ $100 = $6,000 = 6% weight, leaving 2% headroom under an 8% cap.
    shares_owned = (0.06 * portfolio) / price

    result = _call(
        confidence_normalized=ADD_TO_POSITION_CONFIDENCE_LARGE,
        price=price,
        shares_owned=shares_owned,
        portfolio_value=portfolio,
    )

    assert result["allocation_tier"] == "large"
    assert result["current_weight"] == pytest.approx(0.06)
    assert result["headroom"] == pytest.approx(MAX_POSITION_PCT - 0.06)

    # Capped at headroom (2%), not the large tier's 7%.
    expected_shares = math.floor(((MAX_POSITION_PCT - 0.06) * portfolio) / price)
    assert result["shares_to_buy"] == expected_shares
    assert result["skip_reason"] == ""


def test_tier_applies_when_headroom_is_the_larger_of_the_two():
    """With ample headroom the tier percentage is the binding constraint."""
    portfolio = 100_000.0
    price = 100.0
    shares_owned = 1.0  # negligible weight, headroom ~= MAX_POSITION_PCT

    result = _call(
        confidence_normalized=ADD_TO_POSITION_CONFIDENCE_NORMAL,
        price=price,
        shares_owned=shares_owned,
        portfolio_value=portfolio,
    )

    assert result["allocation_tier"] == "normal"
    expected_shares = math.floor((NORMAL_POSITION_PCT * portfolio) / price)
    assert result["shares_to_buy"] == expected_shares


def test_resulting_weight_never_exceeds_max_position_pct():
    """The cap invariant, swept across tiers and starting weights."""
    portfolio = 250_000.0
    price = 37.50

    for start_weight in (0.0001, 0.01, 0.03, 0.05, 0.079):
        for confidence in (0.40, 0.55, 0.70, 0.95):
            shares_owned = (start_weight * portfolio) / price
            result = check_add_to_position(
                "AAPL", confidence, price, shares_owned, portfolio
            )
            final_value = (shares_owned + result["shares_to_buy"]) * price
            final_weight = final_value / portfolio
            assert final_weight <= MAX_POSITION_PCT + 1e-9, (
                f"cap breached: start={start_weight} conf={confidence} "
                f"final_weight={final_weight}"
            )


def test_headroom_exactly_zero_is_rejected():
    """At exactly MAX_POSITION_PCT the position is full — headroom <= 0."""
    portfolio = 100_000.0
    price = 100.0
    shares_owned = (MAX_POSITION_PCT * portfolio) / price

    result = _call(price=price, shares_owned=shares_owned, portfolio_value=portfolio)

    assert result["skip_reason"] == "INVALID_HEADROOM"
    assert result["shares_to_buy"] == 0
    assert result["headroom"] == pytest.approx(0.0)


def test_overweight_position_rejected():
    portfolio = 100_000.0
    price = 100.0
    shares_owned = ((MAX_POSITION_PCT + 0.05) * portfolio) / price

    result = _call(price=price, shares_owned=shares_owned, portfolio_value=portfolio)

    assert result["skip_reason"] == "INVALID_HEADROOM"
    assert result["headroom"] < 0


# --- insufficient equity ---------------------------------------------------


def test_share_price_above_headroom_budget_reports_insufficient_equity():
    """Positive headroom but a budget too small for one whole share."""
    portfolio = 10_000.0
    price = 5_000.0
    shares_owned = 0.1  # $500 = 5% weight, 3% headroom = $300 budget < $5,000

    result = _call(
        confidence_normalized=ADD_TO_POSITION_CONFIDENCE,
        price=price,
        shares_owned=shares_owned,
        portfolio_value=portfolio,
    )

    assert result["shares_to_buy"] == 0
    assert result["skip_reason"] == INSUFFICIENT_EQUITY
    assert result["headroom"] > 0


# --- returned shape --------------------------------------------------------


def test_result_always_has_the_full_key_set():
    """live_trader indexes these keys unconditionally on every return path."""
    expected_keys = {
        "shares_to_buy",
        "skip_reason",
        "headroom",
        "current_weight",
        "allocation_tier",
    }
    cases = [
        _call(),                                                       # happy path
        _call(price=0.0),                                              # invalid price
        _call(shares_owned=0.0),                                       # invalid shares
        _call(confidence_normalized=0.0),                              # low confidence
        _call(portfolio_value=0.0),                                    # invalid portfolio
        _call(shares_owned=(MAX_POSITION_PCT * 100_000.0) / 100.0),    # no headroom
    ]
    # Include the new ANOMALOUS_WEIGHT path
    cases.append(
        _call(shares_owned=100, price=50.0, portfolio_value=100.0),    # anomalous weight
    )
    for result in cases:
        assert set(result.keys()) == expected_keys
        assert isinstance(result["shares_to_buy"], int)


# --- Finding 15: confidence gate uses _SMALL constant ---------------------


def test_confidence_gate_uses_small_threshold(monkeypatch):
    """If ADD_TO_POSITION_CONFIDENCE_SMALL is raised, the gate must reject
    values that used to pass."""
    import config
    monkeypatch.setattr(config, "ADD_TO_POSITION_CONFIDENCE_SMALL", 0.45)
    # Reload the constant in capital_allocator's module namespace
    import capital_allocator
    monkeypatch.setattr(capital_allocator, "ADD_TO_POSITION_CONFIDENCE_SMALL", 0.45)

    result = _call(confidence_normalized=0.42)
    assert result["skip_reason"] == "CONFIDENCE_SKIP"


# --- Finding 16: anomalous weight -----------------------------------------


def test_anomalous_weight_returns_distinct_skip_reason(capsys):
    """current_weight > 1.0 should return ANOMALOUS_WEIGHT, not INVALID_HEADROOM."""
    result = _call(shares_owned=100, price=50.0, portfolio_value=100.0)

    assert result["skip_reason"] == "ANOMALOUS_WEIGHT"
    assert result["current_weight"] == pytest.approx(50.0)
    assert result["shares_to_buy"] == 0

    captured = capsys.readouterr()
    assert "ANOMALOUS_WEIGHT" in captured.out
    assert "stale equity snapshot" in captured.out


def test_normal_overweight_still_returns_invalid_headroom():
    """A routine slightly-over-cap case should still use INVALID_HEADROOM."""
    portfolio = 100_000.0
    price = 100.0
    # 8.5% weight — over MAX_POSITION_PCT but not anomalous
    shares_owned = (0.085 * portfolio) / price

    result = _call(price=price, shares_owned=shares_owned, portfolio_value=portfolio)

    assert result["skip_reason"] == "INVALID_HEADROOM"
    assert result["current_weight"] < 1.0


# --- Finding 17: diagnostic print on INSUFFICIENT_EQUITY ------------------


def test_insufficient_equity_prints_diagnostic(capsys):
    """The operator needs to see why shares_to_buy landed on 0."""
    result = _call(
        confidence_normalized=ADD_TO_POSITION_CONFIDENCE_SMALL,
        price=5_000.0,
        shares_owned=0.1,
        portfolio_value=10_000.0,
    )
    assert result["skip_reason"] == INSUFFICIENT_EQUITY

    captured = capsys.readouterr()
    assert "INSUFFICIENT_EQUITY" in captured.out
    assert "buy_amount=$" in captured.out
    assert "tier=" in captured.out


# --- Finding 18: confidence clamping --------------------------------------


def test_supra_one_confidence_is_clamped(capsys):
    """Raw confidence (e.g. 85.0 instead of 0.85) should be clamped to 1.0."""
    result = _call(confidence_normalized=85.0)

    # Should succeed (clamped, not rejected) and use the large tier
    assert result["skip_reason"] == "" or result["skip_reason"] == INSUFFICIENT_EQUITY
    assert result["allocation_tier"] == "large"

    captured = capsys.readouterr()
    assert "CONFIDENCE_CLAMP" in captured.out
    assert "clamped to 1.0" in captured.out


def test_confidence_at_one_is_not_clamped(capsys):
    """Exactly 1.0 is valid and should not trigger the clamp warning."""
    result = _call(confidence_normalized=1.0)
    assert result["allocation_tier"] == "large"

    captured = capsys.readouterr()
    assert "CONFIDENCE_CLAMP" not in captured.out
