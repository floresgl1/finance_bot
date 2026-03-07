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
"""

import os
import sys
import subprocess
import math

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


# ---------------------------------------------------------------------------
# Order helpers
# ---------------------------------------------------------------------------
def shares_to_buy(api: tradeapi.REST, price: float, fraction: float = 0.20) -> int:
    """
    Calculate how many whole shares represent `fraction` of account equity.
    Returns 0 if the account cannot afford at least one share.
    """
    account   = api.get_account()
    equity    = float(account.equity)
    budget    = equity * fraction
    shares    = math.floor(budget / price)
    return max(shares, 0)


def place_buy(api: tradeapi.REST, ticker: str, qty: int) -> None:
    if qty <= 0:
        print(f"  [SKIP] {ticker} — insufficient equity for even 1 share")
        return
    order = api.submit_order(
        symbol     = ticker,
        qty        = qty,
        side       = "buy",
        type       = "market",
        time_in_force = "day",
    )
    print(f"  [BUY]  {ticker} x{qty} — order id {order.id}")


def place_sell(api: tradeapi.REST, ticker: str, qty: float) -> None:
    order = api.submit_order(
        symbol     = ticker,
        qty        = qty,
        side       = "sell",
        type       = "market",
        time_in_force = "day",
    )
    print(f"  [SELL] {ticker} x{qty} — order id {order.id}")


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

    # 4. Current positions
    owned = get_owned_tickers(api)
    print(f"\n  Current positions: {list(owned.keys()) or 'none'}\n")

    # 5. Execute logic
    print("  Applying trading rules...")
    print("-" * 60)

    for r in signals:
        ticker       = r["ticker"]
        final_signal = r["final_signal"]
        price        = r["current_price"]
        note         = f" ({r['note']})" if r["note"] else ""

        if final_signal == "BUY":
            if ticker not in owned:
                qty = shares_to_buy(api, price)
                print(f"  Signal: BUY {ticker} @ ${price:.2f}{note}")
                place_buy(api, ticker, qty)
            else:
                print(f"  Signal: BUY {ticker}{note} — already owned, skipping")

        elif final_signal == "SELL":
            if ticker in owned:
                qty = owned[ticker]
                print(f"  Signal: SELL {ticker} @ ${price:.2f}{note}")
                place_sell(api, ticker, qty)
            else:
                print(f"  Signal: SELL {ticker}{note} — not owned, skipping")

        else:  # HOLD
            print(f"  Signal: HOLD {ticker}{note} — no action")

    print("-" * 60)
    print("\n  Done.\n")


if __name__ == "__main__":
    run()
