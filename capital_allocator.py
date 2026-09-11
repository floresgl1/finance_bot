"""
capital_allocator.py — Add-to-position sizing for already-owned tickers.

Sits between predictor.py (signal generation) and live_trader.py (execution).
Called whenever a BUY signal fires on a ticker that is already held.
Confidence and headroom checks are performed internally.

Confidence tiers
----------------
    confidence < ADD_TO_POSITION_CONFIDENCE_SMALL (0.40)  → CONFIDENCE_SKIP
    0.40 ≤ confidence < 0.50                        → small  (SMALL_POSITION_PCT  = 3 %)
    0.50 ≤ confidence < 0.65                        → normal (NORMAL_POSITION_PCT = 5 %)
    confidence ≥ 0.65                               → large  (LARGE_POSITION_PCT  = 7 %)

`get_allocation_tier` is public because live_trader.get_position_size() sizes
NEW positions from the same table. Every tier is below MAX_POSITION_PCT, so a
position can never open above the cap that governs it afterwards.

Headroom logic
--------------
    current_position_value = shares_owned × current_price
    current_weight         = current_position_value / total_portfolio_value
    headroom               = MAX_POSITION_PCT − current_weight

    The tier target percentage is capped by available headroom so the position
    never exceeds MAX_POSITION_PCT regardless of tier.

    headroom > 0 → buy_pct       = min(tier_pct, headroom)
                   buy_amount    = buy_pct × total_portfolio_value
                   shares_to_buy = floor(buy_amount / current_price)
    headroom ≤ 0 → skip with INVALID_HEADROOM
"""

import math

from config import (
    MAX_POSITION_PCT,
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


def get_allocation_tier(confidence: float) -> tuple[str, float]:
    """Return (tier_label, target_position_pct) for the given confidence score.

    Public because live_trader.get_position_size() sizes NEW positions from it
    too. Those were separate rules and they contradicted each other: new
    positions opened at 10/15/20% of equity while MAX_POSITION_PCT capped the
    same position at 8%, so every entry arrived over-weight and the rebalancer
    trimmed it straight back down.
    """
    if confidence >= ADD_TO_POSITION_CONFIDENCE_LARGE:
        return "large", LARGE_POSITION_PCT
    if confidence >= ADD_TO_POSITION_CONFIDENCE_NORMAL:
        return "normal", NORMAL_POSITION_PCT
    return "small", SMALL_POSITION_PCT


def check_add_to_position(
    ticker: str,
    confidence_normalized: float,
    price: float,
    shares_owned: float,
    portfolio_value: float,
) -> dict:
    """
    Determine whether and how many additional shares to buy for an
    already-owned ticker.

    Parameters
    ----------
    ticker                : stock symbol
    confidence_normalized : model confidence as a 0–1 fraction (confidence / 100)
    price                 : current price in USD
    shares_owned          : quantity of shares currently held
    portfolio_value       : total account equity (from live_trader.py equity snapshot)

    Returns
    -------
    dict with keys:
        shares_to_buy    — int, shares to purchase (0 if allocation is blocked)
        skip_reason      — "CONFIDENCE_SKIP" if below ADD_TO_POSITION_CONFIDENCE_SMALL,
                           "ANOMALOUS_WEIGHT" if current_weight > 1.0,
                           "INVALID_HEADROOM" if headroom ≤ 0,
                           "INSUFFICIENT_EQUITY" if allocated amount < 1 share,
                           else ""
        headroom         — float, MAX_POSITION_PCT − current_weight
        current_weight   — float, fraction of portfolio currently in this position
        allocation_tier  — str, "small" / "normal" / "large" (empty if skipped)
    """
    if price <= 0:
        return {
            "shares_to_buy":   0,
            "skip_reason":     INVALID_PRICE_SKIP,
            "headroom":        0.0,
            "current_weight":  0.0,
            "allocation_tier": "",
        }

    if shares_owned <= 0:
        return {
            "shares_to_buy":   0,
            "skip_reason":     INVALID_SHARES_SKIP,
            "headroom":        0.0,
            "current_weight":  0.0,
            "allocation_tier": "",
        }

    # Finding 18: clamp supra-1.0 values so a caller bug (e.g. passing raw
    # confidence 85.0 instead of 0.85) doesn't silently allocate as "large".
    if confidence_normalized > 1.0:
        print(
            f"  [CONFIDENCE_CLAMP] {ticker}: confidence_normalized="
            f"{confidence_normalized:.2f} clamped to 1.0"
        )
        confidence_normalized = 1.0

    # Finding 15: use ADD_TO_POSITION_CONFIDENCE_SMALL so the constant
    # actually controls the small-tier floor.
    if confidence_normalized < ADD_TO_POSITION_CONFIDENCE_SMALL:
        return {
            "shares_to_buy":   0,
            "skip_reason":     "CONFIDENCE_SKIP",
            "headroom":        0.0,
            "current_weight":  0.0,
            "allocation_tier": "",
        }

    if portfolio_value <= 0:
        return {
            "shares_to_buy":   0,
            "skip_reason":     INVALID_PORTFOLIO_SKIP,
            "headroom":        0.0,
            "current_weight":  0.0,
            "allocation_tier": "",
        }

    current_position_value = shares_owned * price
    current_weight         = current_position_value / portfolio_value
    headroom               = MAX_POSITION_PCT - current_weight

    # Finding 16: flag anomalous weight (possible stale equity snapshot)
    # separately from a routine cap-reached skip.
    if current_weight > 1.0:
        print(
            f"  [ANOMALOUS_WEIGHT] {ticker}: current_weight={current_weight:.2f} "
            f"(portfolio_value={portfolio_value:.0f}) — possible stale equity snapshot"
        )
        return {
            "shares_to_buy":   0,
            "skip_reason":     "ANOMALOUS_WEIGHT",
            "headroom":        headroom,
            "current_weight":  current_weight,
            "allocation_tier": "",
        }

    if headroom <= 0:
        return {
            "shares_to_buy":   0,
            "skip_reason":     "INVALID_HEADROOM",
            "headroom":        headroom,
            "current_weight":  current_weight,
            "allocation_tier": "",
        }

    tier_label, tier_pct = get_allocation_tier(confidence_normalized)
    buy_pct               = min(tier_pct, headroom)
    buy_amount            = buy_pct * portfolio_value
    shares_to_buy         = math.floor(buy_amount / price)

    # Finding 17: surface the root cause when allocated amount < one share.
    if shares_to_buy == 0:
        print(
            f"  [INSUFFICIENT_EQUITY] {ticker}: buy_amount=${buy_amount:.2f} "
            f"< price=${price:.2f} (tier={tier_label}, buy_pct={buy_pct:.1%}, "
            f"portfolio=${portfolio_value:.0f})"
        )

    return {
        "shares_to_buy":   shares_to_buy,
        "skip_reason":     "" if shares_to_buy > 0 else INSUFFICIENT_EQUITY,
        "headroom":        headroom,
        "current_weight":  current_weight,
        "allocation_tier": tier_label,
    }
