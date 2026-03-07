"""
live_trader.py — Alpaca paper-trading execution layer.

Connects to Alpaca's paper-trading endpoint, fetches today's signals from
predictor.py, then submits orders according to the following rules:

  BUY  + not already owned  → buy shares  (20 % of account equity)
  BUY  + already owned      → skip (already positioned)
  SELL + owned              → sell the entire position
  SELL + not owned          → skip (no position to close)
  HOLD                      → do nothing

All orders are submitted as market orders.  Because the base URL points to
paper-api.alpaca.markets this script CANNOT place live trades.

Discord alerts are sent before and after order execution if
DISCORD_WEBHOOK_URL is set in the environment.
"""

import os
import sys
import subprocess
import math
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Dependency check — install alpaca-trade-api if not present
# ---------------------------------------------------------------------------
try:
    import alpaca_trade_api as tradeapi
except ImportError:
    print("[setup] alpaca-trade-api not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "alpaca-trade-api"])
    import alpaca_trade_api as tradeapi


# ---------------------------------------------------------------------------
# Alpaca connection
# ---------------------------------------------------------------------------
BASE_URL = "https://paper-api.alpaca.markets"

def get_api() -> tradeapi.REST:
    """Build an Alpaca REST client from environment variables."""
    api_key    = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment variables."
        )

    return tradeapi.REST(api_key, secret_key, BASE_URL, api_version="v2")


# ---------------------------------------------------------------------------
# Market status
# ---------------------------------------------------------------------------
def market_is_open(api: tradeapi.REST) -> bool:
    clock = api.get_clock()
    return clock.is_open


# ---------------------------------------------------------------------------
# Signal generation (reuses predictor.py logic, returns data instead of printing)
# ---------------------------------------------------------------------------
def get_signals() -> list[dict]:
    """
    Return a list of signal dicts for every ticker in WATCHLIST.

    Each dict contains:
        ticker, signal, final_signal, confidence, current_price, sentiment, note
    """
    from config import WATCHLIST
    from predictor import load_model, predict_ticker, apply_sentiment_veto
    from sentiment import get_sentiment_all

    model = load_model()

    print("Fetching news sentiment...")
    sentiments = get_sentiment_all(WATCHLIST)

    results = []
    for ticker in WATCHLIST:
        try:
            r     = predict_ticker(ticker, model)
            score = sentiments.get(ticker, {}).get("sentiment_score", 0.0)
            final_signal, note = apply_sentiment_veto(r["signal"], score)
            r["final_signal"] = final_signal
            r["sentiment"]    = score
            r["note"]         = note
            results.append(r)
        except Exception as exc:
            print(f"  [SKIP] {ticker} — {exc}")

    return results


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------
def get_owned_tickers(api: tradeapi.REST) -> dict[str, float]:
    """Return {ticker: qty_held} for all current positions."""
    positions = api.list_positions()
    return {p.symbol: float(p.qty) for p in positions}


def get_equity(api: tradeapi.REST) -> float:
    return float(api.get_account().equity)


# ---------------------------------------------------------------------------
# Order helpers
# ---------------------------------------------------------------------------
def compute_buy_qty(equity: float, price: float, fraction: float = 0.20) -> int:
    """Whole shares that fit within `fraction` of account equity."""
    return max(math.floor(equity * fraction / price), 0)


def place_buy(api: tradeapi.REST, ticker: str, qty: int) -> dict:
    """Submit a market buy order. Returns a result dict."""
    if qty <= 0:
        print(f"  [SKIP] {ticker} — insufficient equity for even 1 share")
        return {"status": "skipped", "reason": "insufficient equity"}
    try:
        order = api.submit_order(
            symbol        = ticker,
            qty           = qty,
            side          = "buy",
            type          = "market",
            time_in_force = "day",
        )
        print(f"  [BUY]  {ticker} x{qty} — order id {order.id}")
        return {"status": "placed", "order_id": order.id}
    except Exception as exc:
        print(f"  [ERROR] {ticker} BUY failed: {exc}")
        return {"status": "error", "reason": str(exc)}


def place_sell(api: tradeapi.REST, ticker: str, qty: float) -> dict:
    """Submit a market sell order. Returns a result dict."""
    try:
        order = api.submit_order(
            symbol        = ticker,
            qty           = qty,
            side          = "sell",
            type          = "market",
            time_in_force = "day",
        )
        print(f"  [SELL] {ticker} x{qty} — order id {order.id}")
        return {"status": "placed", "order_id": order.id}
    except Exception as exc:
        print(f"  [ERROR] {ticker} SELL failed: {exc}")
        return {"status": "error", "reason": str(exc)}


# ---------------------------------------------------------------------------
# Discord alerting
# ---------------------------------------------------------------------------
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def send_discord(message: str) -> None:
    """POST a message to the Discord webhook. Silently skips if URL not set."""
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        import requests
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            json    = {"content": message},
            timeout = 10,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"  [Discord] Alert failed: {exc}")


def build_pre_order_alert(planned: list[dict], timestamp: str) -> str:
    """
    Build the pre-order Discord message.

    Each item in `planned` has:
        ticker, action, qty, price, note, skip_reason
    """
    n_buy  = sum(1 for p in planned if p["action"] == "BUY"  and not p["skip_reason"])
    n_sell = sum(1 for p in planned if p["action"] == "SELL" and not p["skip_reason"])
    n_hold = sum(1 for p in planned if p["action"] == "HOLD")
    n_skip = sum(1 for p in planned if p["skip_reason"])

    lines = [
        f"**Finance Bot — Pre-Trade Plan** | {timestamp}",
        f"Summary: {n_buy} BUY | {n_sell} SELL | {n_hold} HOLD | {n_skip} skipped",
        "```",
    ]

    for p in planned:
        action      = p["action"]
        ticker      = p["ticker"]
        price       = p["price"]
        qty         = p["qty"]
        note        = f"  [{p['note']}]" if p["note"] else ""
        skip_reason = f"  → {p['skip_reason']}" if p["skip_reason"] else ""

        if action in ("BUY", "SELL") and not p["skip_reason"]:
            lines.append(f"  {action:<4} {ticker:<6} x{qty:<4} @ ${price:>8.2f}{note}")
        elif p["skip_reason"]:
            lines.append(f"  {action:<4} {ticker:<6}        @ ${price:>8.2f}{note}{skip_reason}")
        else:  # HOLD
            lines.append(f"  {action:<4} {ticker:<6}        @ ${price:>8.2f}{note}")

    lines.append("```")
    return "\n".join(lines)


def build_post_order_alert(outcomes: list[dict], portfolio_value: float, timestamp: str) -> str:
    """
    Build the post-order Discord message.

    Each item in `outcomes` has:
        ticker, action, qty, price, status, order_id, reason
    """
    placed  = [o for o in outcomes if o["status"] == "placed"]
    skipped = [o for o in outcomes if o["status"] == "skipped"]
    errors  = [o for o in outcomes if o["status"] == "error"]

    lines = [
        f"**Finance Bot — Trade Results** | {timestamp}",
        f"Portfolio value: **${portfolio_value:,.2f}**",
        "```",
    ]

    if placed:
        lines.append("Placed:")
        for o in placed:
            lines.append(f"  {o['action']:<4} {o['ticker']:<6} x{o['qty']:<4} @ ${o['price']:>8.2f}  order {o['order_id']}")
    else:
        lines.append("Placed:  none")

    if skipped:
        lines.append("Skipped:")
        for o in skipped:
            lines.append(f"  {o['action']:<4} {o['ticker']:<6}  — {o['reason']}")
    else:
        lines.append("Skipped: none")

    if errors:
        lines.append("Errors:")
        for o in errors:
            lines.append(f"  {o['action']:<4} {o['ticker']:<6}  — {o['reason']}")
    else:
        lines.append("Errors:  none")

    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def run() -> None:
    print("=" * 60)
    print("  Finance Bot — Alpaca Paper Trader")
    print("=" * 60)

    # 1. Connect
    api = get_api()
    print(f"  Connected to {BASE_URL}")

    # 2. Market status
    if not market_is_open(api):
        print("\n  Market is currently CLOSED — nothing to do. Exiting.\n")
        sys.exit(0)

    print("  Market is OPEN — proceeding with signal generation.\n")

    # 3. Get today's signals
    signals = get_signals()
    if not signals:
        print("  No signals generated. Exiting.")
        sys.exit(0)

    # 4. Current positions and equity (fetched once for consistent sizing)
    owned  = get_owned_tickers(api)
    equity = get_equity(api)
    print(f"\n  Current positions: {list(owned.keys()) or 'none'}")
    print(f"  Account equity:    ${equity:,.2f}\n")

    # 5. Build planned-action list (pre-order alert data + drives execution)
    planned = []
    for r in signals:
        ticker = r["ticker"]
        action = r["final_signal"]
        price  = r["current_price"]
        note   = r["note"]

        if action == "BUY":
            if ticker not in owned:
                qty         = compute_buy_qty(equity, price)
                skip_reason = "" if qty > 0 else "insufficient equity"
            else:
                qty         = 0
                skip_reason = "already owned"

        elif action == "SELL":
            qty         = owned.get(ticker, 0)
            skip_reason = "" if ticker in owned else "not owned"

        else:  # HOLD
            qty         = 0
            skip_reason = ""

        planned.append({
            "ticker":      ticker,
            "action":      action,
            "qty":         qty,
            "price":       price,
            "note":        note,
            "skip_reason": skip_reason,
        })

    # 6. Pre-order Discord alert
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    send_discord(build_pre_order_alert(planned, timestamp))

    # 7. Execute orders and collect outcomes
    print("  Applying trading rules...")
    print("-" * 60)

    outcomes = []
    for p in planned:
        ticker      = p["ticker"]
        action      = p["action"]
        qty         = p["qty"]
        price       = p["price"]
        note        = f" ({p['note']})" if p["note"] else ""
        skip_reason = p["skip_reason"]

        if action == "BUY" and not skip_reason:
            print(f"  Signal: BUY {ticker} @ ${price:.2f}{note}")
            result = place_buy(api, ticker, qty)

        elif action == "SELL" and not skip_reason:
            print(f"  Signal: SELL {ticker} @ ${price:.2f}{note}")
            result = place_sell(api, ticker, qty)

        elif action == "HOLD":
            print(f"  Signal: HOLD {ticker}{note} — no action")
            result = {"status": "skipped", "reason": "HOLD signal"}

        else:
            print(f"  Signal: {action} {ticker}{note} — skipped ({skip_reason})")
            result = {"status": "skipped", "reason": skip_reason}

        outcomes.append({
            "ticker":   ticker,
            "action":   action,
            "qty":      qty,
            "price":    price,
            "status":   result["status"],
            "order_id": result.get("order_id", ""),
            "reason":   result.get("reason", ""),
        })

    print("-" * 60)

    # 8. Post-order Discord alert (refresh equity for current portfolio value)
    portfolio_value = get_equity(api)
    timestamp_end   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    send_discord(build_post_order_alert(outcomes, portfolio_value, timestamp_end))

    print(f"\n  Portfolio value: ${portfolio_value:,.2f}")
    print("  Done.\n")


if __name__ == "__main__":
    run()
