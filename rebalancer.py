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
    Order fail       — no row logged; outcome dict records the error
    Data error       — logged and position skipped; loop continues
    Remaining shares — post-model-SELL positions are re-fetched fresh from
                       Alpaca, so zero-share positions are naturally excluded

Log codes:
    REBALANCE_TRIM   — trim order placed successfully (EXIT row with realized P&L)

Returns:
    List of outcome dicts (same schema as live_trader outcomes):
        ticker, action, qty, price, status, order_id, reason
"""

import math

from config import MAX_POSITION_PCT, CANCEL_STOP_FAILED
from signal_logger import log_signal

_TRIGGER_BUFFER = 0.001   # weight must exceed MAX_POSITION_PCT by this much
_TARGET_OFFSET  = 0.005   # trim to (MAX_POSITION_PCT - _TARGET_OFFSET)


def run_rebalancer(api, sell_executed_tickers: list[str] | None = None) -> list[dict]:
    """
    Inspect all open positions and sell shares in any that are overweight.

    Parameters
    ----------
    api : alpaca_trade_api.REST
        Live Alpaca client (paper or live).
    sell_executed_tickers : list[str] | None
        Tickers for which a model SELL order was placed successfully in the
        current session.  These are skipped by the rebalancer to avoid a
        double-sell on the same ticker in the same run.

    Returns
    -------
    list[dict]
        One entry per rebalance action attempted, whether placed or failed.
        Each dict: ticker, action, qty, price, status, order_id, reason.
    """
    _sell_skip = set(sell_executed_tickers) if sell_executed_tickers else set()
    outcomes: list[dict] = []

    # Deferred to avoid circular import (live_trader imports run_rebalancer).
    from live_trader import _raise_if_infra

    # --- Fetch fresh portfolio state ----------------------------------------
    try:
        equity = float(api.get_account().equity)
    except Exception as exc:
        _raise_if_infra(exc, "get_account", "REBALANCER")
        print(f"  [REBALANCER] Could not fetch equity: {exc} — skipping rebalancer")
        return outcomes

    if equity <= 0:
        print("  [REBALANCER] Equity is zero or negative — skipping rebalancer")
        return outcomes

    try:
        positions = api.list_positions()
    except Exception as exc:
        _raise_if_infra(exc, "list_positions", "REBALANCER")
        print(f"  [REBALANCER] Could not fetch positions: {exc} — skipping rebalancer")
        return outcomes

    trigger = MAX_POSITION_PCT + _TRIGGER_BUFFER
    target  = MAX_POSITION_PCT - _TARGET_OFFSET

    print(f"  [REBALANCER] Checking {len(positions)} position(s) "
          f"(trigger={trigger:.1%}, target={target:.1%})")

    for position in positions:
        ticker = position.symbol

        # --- Skip tickers already sold by model SELL this session -----------
        if ticker in _sell_skip:
            print(f"  [REBALANCER] {ticker} — already sold by model SELL, skipping")
            continue

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

        # --- Cancel any standing stop before issuing the trim SELL ----------
        # Deferred imports avoid a circular import with live_trader.py, which
        # imports run_rebalancer from this module at load time.
        from live_trader import cancel_standing_stops, send_discord

        cancel_ok, _ = cancel_standing_stops(api, ticker)
        if not cancel_ok:
            send_discord(
                f"🚨 **[CANCEL_STOP_FAILED]** {ticker} — could not cancel standing stop; "
                f"rebalancer trim skipped to avoid conflict. Manual review required."
            )
            log_signal(ticker, "REBALANCER", price, 0, 0.0, CANCEL_STOP_FAILED)
            outcomes.append({
                "ticker":   ticker,
                "action":   "SELL",
                "qty":      0,
                "price":    price,
                "status":   "skipped",
                "order_id": "",
                "reason":   "CANCEL_STOP_FAILED",
            })
            continue

        # --- Place trim order, confirm the fill ------------------------------
        # Through place_sell, so the EXIT row is written only for shares that
        # actually sold, at the price they sold at. The trim used to be
        # fire-and-forget: the row was written on submit, at the position's
        # quoted price, whether or not the order ever filled.
        from live_trader import place_sell
        from signal_logger import log_exit, find_open_entry_order_id, find_position_id

        # Avg entry price, read before the sell.
        try:
            entry_price_for_exit = float(api.get_position(ticker).avg_entry_price)
        except Exception as exc:
            _raise_if_infra(exc, "get_position", ticker)
            print(f"  [REBALANCER] Could not fetch avg_entry_price for {ticker}: {exc} — using current price as fallback")
            entry_price_for_exit = float(price)

        result = place_sell(api, ticker, qty)
        if result["status"] != "filled":
            # Nothing sold, nothing to log. The stop cancelled above is
            # re-attached by live_trader's post-rebalance stop check.
            print(f"  [REBALANCER] {ticker} trim {result['status']}: {result.get('reason', '')}")
            outcomes.append({
                "ticker":   ticker,
                "action":   "SELL",
                "qty":      qty,
                "price":    price,
                "status":   "error",
                "order_id": result.get("order_id", ""),
                "reason":   result.get("reason") or f"trim {result['status']}",
            })
            continue

        sold       = float(result.get("filled_qty") or qty)
        fill_price = result.get("filled_avg_price") or price
        print(f"  [REBALANCER] {ticker} trim filled — {sold:g} sh @ ${fill_price:.2f}, order id {result['order_id']}")

        # Log as EXIT row — rebalancer is an operational trim, not a model signal
        log_exit(
            ticker         = ticker,
            entry_order_id = find_open_entry_order_id(ticker) or "UNLINKED",
            entry_price    = entry_price_for_exit,
            exit_price     = fill_price,
            exit_reason    = "REBALANCE_TRIM",
            shares         = sold,
            position_id    = find_position_id(ticker),
        )

        # Refresh equity after each successful trim
        try:
            equity = float(api.get_account().equity)
        except Exception as exc:
            _raise_if_infra(exc, "get_account", ticker)
            print(f"  [REBALANCER] Equity refresh failed after {ticker} trim: {exc}")

        # "placed" means the trim executed; live_trader keys the BUY-pass skip
        # list and the Discord summary on it.
        outcomes.append({
            "ticker":   ticker,
            "action":   "SELL",
            "qty":      sold,
            "price":    fill_price,
            "status":   "placed",
            "order_id": result["order_id"],
            "reason":   "rebalance trim",
        })

    if not outcomes:
        print("  [REBALANCER] No positions required trimming.")

    return outcomes
