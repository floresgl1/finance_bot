"""
rebalancer.py — Trim overweight positions to create room for model BUY signals.

Runs after model SELLs and before model BUYs in the live_trader execution
sequence:

    Model SELLs → refresh → Rebalancer → refresh → Model BUYs

Trigger:  current_weight > MAX_POSITION_PCT + 0.001
Target:   MAX_POSITION_PCT - 0.005
Shares:   math.ceil(excess_value / price), capped at math.ceil(shares * 0.25)

After each successful trim order the module re-fetches portfolio equity so
subsequent weight checks are accurate.

Edge cases:
    Order fail       — REBALANCER_ORDER_FAIL logged; loop continues
    Data error       — logged and position skipped; loop continues
    Remaining shares — post-model-SELL positions are re-fetched fresh from
                       Alpaca, so zero-share positions are naturally excluded

Log codes:
    REBALANCER_SELL       — trim order placed successfully
    REBALANCER_ORDER_FAIL — trim order submission failed

Returns:
    List of outcome dicts (same schema as live_trader outcomes):
        ticker, action, qty, price, status, order_id, reason
"""

import math

from config import MAX_POSITION_PCT
from signal_logger import log_signal

_TRIGGER_BUFFER = 0.001   # weight must exceed MAX_POSITION_PCT by this much
_TARGET_OFFSET  = 0.005   # trim to (MAX_POSITION_PCT - _TARGET_OFFSET)


def run_rebalancer(api) -> list[dict]:
    """
    Inspect all open positions and sell shares in any that are overweight.

    Parameters
    ----------
    api : alpaca_trade_api.REST
        Live Alpaca client (paper or live).

    Returns
    -------
    list[dict]
        One entry per rebalance action attempted, whether placed or failed.
        Each dict: ticker, action, qty, price, status, order_id, reason.
    """
    outcomes: list[dict] = []

    # --- Fetch fresh portfolio state ----------------------------------------
    try:
        equity = float(api.get_account().equity)
    except Exception as exc:
        print(f"  [REBALANCER] Could not fetch equity: {exc} — skipping rebalancer")
        return outcomes

    if equity <= 0:
        print("  [REBALANCER] Equity is zero or negative — skipping rebalancer")
        return outcomes

    try:
        positions = api.list_positions()
    except Exception as exc:
        print(f"  [REBALANCER] Could not fetch positions: {exc} — skipping rebalancer")
        return outcomes

    trigger = MAX_POSITION_PCT + _TRIGGER_BUFFER
    target  = MAX_POSITION_PCT - _TARGET_OFFSET

    print(f"  [REBALANCER] Checking {len(positions)} position(s) "
          f"(trigger={trigger:.1%}, target={target:.1%})")

    for position in positions:
        ticker = position.symbol

        # --- Parse position data (data inconsistency edge case) -------------
        try:
            current_shares = float(position.qty)
            price          = float(position.current_price)
        except Exception as exc:
            print(f"  [REBALANCER] Data error for {ticker}: {exc} — skipping")
            continue

        if current_shares <= 0 or price <= 0:
            print(f"  [REBALANCER] {ticker} — invalid shares/price, skipping")
            continue

        current_value  = current_shares * price
        current_weight = current_value / equity

        if current_weight <= trigger:
            continue

        # --- Compute shares to sell -----------------------------------------
        target_value   = target * equity
        target_shares = target_value / price
        shares_to_sell = current_shares - math.ceil(target_shares)
        sell_cap       = math.ceil(current_shares * 0.25)
        qty            = min(shares_to_sell, sell_cap)
        qty            = min(qty, int(current_shares))   # never sell more than owned

        if qty <= 0:
            print(f"  [REBALANCER] {ticker} — computed qty=0 after cap, skipping")
            continue

        print(
            f"  [REBALANCER] {ticker} overweight "
            f"({current_weight:.1%} > {trigger:.1%}) — "
            f"trimming {qty} share(s) @ ${price:.2f}  "
            f"[raw={shares_to_sell}, cap={sell_cap}]"
        )

        # --- Place trim order (order fail edge case) ------------------------
        try:
            order = api.submit_order(
                symbol        = ticker,
                qty           = qty,
                side          = "sell",
                type          = "market",
                time_in_force = "day",
            )
            print(f"  [REBALANCER] {ticker} trim placed — order id {order.id}")
            log_signal(ticker, "REBALANCER", price, qty, 0.0, "REBALANCER_SELL")

            # Refresh equity after each successful trim
            try:
                equity = float(api.get_account().equity)
            except Exception as exc:
                print(f"  [REBALANCER] Equity refresh failed after {ticker} trim: {exc}")

            outcomes.append({
                "ticker":   ticker,
                "action":   "SELL",
                "qty":      qty,
                "price":    price,
                "status":   "placed",
                "order_id": order.id,
                "reason":   "rebalance trim",
            })

        except Exception as exc:
            print(f"  [REBALANCER] {ticker} trim order failed: {exc}")
            log_signal(ticker, "REBALANCER", price, qty, 0.0, "REBALANCER_ORDER_FAIL")
            outcomes.append({
                "ticker":   ticker,
                "action":   "SELL",
                "qty":      qty,
                "price":    price,
                "status":   "error",
                "order_id": "",
                "reason":   str(exc),
            })
            # Log and continue to next position

    if not outcomes:
        print("  [REBALANCER] No positions required trimming.")

    return outcomes
