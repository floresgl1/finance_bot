"""
capital_allocator.py — Add-to-position sizing for already-owned tickers.

Sits between predictor.py (signal generation) and live_trader.py (execution).
Called whenever a BUY signal fires on a ticker that is already held.
Confidence and headroom checks are performed internally.

Headroom logic
--------------
    current_position_value = shares_owned × current_price
    current_weight         = current_position_value / total_portfolio_value
    headroom               = MAX_POSITION_PCT − current_weight

    headroom > 0 → buy_amount    = headroom × total_portfolio_value
                   shares_to_buy = floor(buy_amount / current_price)
    headroom ≤ 0 → skip with INVALID_HEADROOM
"""

import math

from config import MAX_POSITION_PCT, ADD_TO_POSITION_CONFIDENCE


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
        shares_to_buy   — int, shares to purchase (0 if allocation is blocked)
        skip_reason     — "CONFIDENCE_SKIP" if below ADD_TO_POSITION_CONFIDENCE,
                          "INVALID_HEADROOM" if headroom ≤ 0, else ""
        headroom        — float, MAX_POSITION_PCT − current_weight
        current_weight  — float, fraction of portfolio currently in this position
    """
    if confidence_normalized < ADD_TO_POSITION_CONFIDENCE:
        return {
            "shares_to_buy":  0,
            "skip_reason":    "CONFIDENCE_SKIP",
            "headroom":       0.0,
            "current_weight": 0.0,
        }

    if portfolio_value <= 0:
        return {
            "shares_to_buy":  0,
            "skip_reason":    "INVALID_HEADROOM",
            "headroom":       0.0,
            "current_weight": 0.0,
        }

    current_position_value = shares_owned * price
    current_weight         = current_position_value / portfolio_value
    headroom               = MAX_POSITION_PCT - current_weight

    if headroom <= 0:
        return {
            "shares_to_buy":  0,
            "skip_reason":    "INVALID_HEADROOM",
            "headroom":       headroom,
            "current_weight": current_weight,
        }

    buy_amount    = headroom * portfolio_value
    shares_to_buy = math.floor(buy_amount / price)

    return {
        "shares_to_buy":  shares_to_buy,
        "skip_reason":    "" if shares_to_buy > 0 else "insufficient equity",
        "headroom":       headroom,
        "current_weight": current_weight,
    }
