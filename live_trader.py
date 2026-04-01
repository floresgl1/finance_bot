"""
live_trader.py — Alpaca paper-trading execution layer.

Connects to Alpaca's paper-trading endpoint, fetches today's signals from
predictor.py, then executes trades one at a time in a stateful loop:

  Execution order:
    1. SELLs  — sorted by confidence descending
    2. BUYs   — sorted by confidence descending
    3. HOLDs  — no action

  Before each trade:
    - Fresh portfolio state is fetched from Alpaca (positions + equity).
    - For BUYs on already-owned tickers, capital_allocator.py receives the
      fresh state and returns the sizing decision.

  Error handling:
    - SELL errors are caught, logged, and the loop continues.
    - INVALID_HEADROOM from capital_allocator skips to the next BUY signal.

  Logging:
    - Each signal is logged to signal_log.csv after its trade attempt,
      with the actual execution outcome as actual_action.

  Discord:
    - A single summary notification is sent after all trades are complete.
    - Stop-loss / take-profit and sentiment-veto alerts still fire immediately.

Veto layers applied before execution:
  1. Sentiment veto  (FinBERT)
  2. Earnings veto   (recent Surprise%)

Stop-loss check runs before model signals:
  Loss > 10 % from avg entry → market sell + Discord alert

All orders are submitted as market orders.  Because the base URL points to
paper-api.alpaca.markets this script CANNOT place live trades.
"""

import os
import sys
import subprocess
import math
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from capital_allocator import check_add_to_position
from rebalancer import run_rebalancer
from config import INSUFFICIENT_EQUITY

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
def get_signals(sentiment_df=None) -> list[dict]:
    """
    Return a list of signal dicts for every ticker in WATCHLIST.

    Applies two veto layers in order:
      1. Earnings veto   (recent Surprise% within 90 days)
      2. Sentiment veto  (CSV-based, thresholds ±0.5)

    sentiment_df must be pre-loaded by the caller (load_sentiment_df()) so
    the CSV is not re-read on every call.

    Each dict contains:
        ticker, signal, final_signal, confidence, current_price, sentiment, note
    """
    import pandas as pd
    from datetime import date as date_cls
    from config import WATCHLIST
    from predictor import (load_model, predict_ticker,
                           apply_earnings_veto, get_recent_earnings_surprise,
                           apply_sentiment_veto)

    model = load_model()
    today = date_cls.today()

    results = []
    for ticker in WATCHLIST:
        try:
            r            = predict_ticker(ticker, model)
            model_signal = r["signal"]

            veto_reason = None

            # 1. Earnings veto
            surprise = get_recent_earnings_surprise(ticker)
            post_earnings, earn_note = apply_earnings_veto(model_signal, surprise)
            if post_earnings != model_signal:
                veto_reason = "EARNINGS_VETO"

            # 2. Sentiment veto (only if not already HOLD from earnings)
            final_signal = post_earnings
            sent_note    = ""
            sent_score   = 0.0

            if post_earnings != "HOLD" and sentiment_df is not None:
                rows = sentiment_df[
                    (sentiment_df["Ticker"] == ticker) &
                    (sentiment_df["Date"] == today)
                ]
                sent_score = float(rows["sentiment_score"].iloc[-1]) if not rows.empty else 0.0
                final_signal, sent_note = apply_sentiment_veto(post_earnings, sent_score)
                if final_signal == "HOLD" and sent_note:
                    veto_reason = "SENTIMENT_VETO"
                    send_discord(
                        f"\N{NO ENTRY} **Sentiment Veto:** {ticker} {today}  "
                        f"Model said: {post_earnings}  "
                        f"Sentiment score: {sent_score:.3f}  "
                        f"Action: HOLD"
                    )

            r["final_signal"] = final_signal
            r["sentiment"]    = sent_score
            r["note"]         = earn_note or sent_note
            r["veto_reason"]  = veto_reason
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
def get_position_size(confidence: float) -> float | None:
    """
    Map a model confidence score (0–100) to a portfolio fraction.

    Tiers:
      below 35  → None  (skip trade entirely, treat as HOLD)
      35 – 40   → 0.10  (10 % of equity)
      40 – 45   → 0.15  (15 % of equity)
      45+       → 0.20  (20 % of equity)
    """
    if confidence < 35:
        return None
    if confidence < 40:
        return 0.10
    if confidence < 45:
        return 0.15
    return 0.20


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
# Stop-loss cooldown check
# ---------------------------------------------------------------------------
def was_stop_loss_recently(ticker: str, days: int | None = None) -> bool:
    """
    Return True if a STOP_LOSS_SELL was logged for `ticker` within the past
    `days` calendar days (today - log_date <= days).

    Defaults to STOP_LOSS_COOLDOWN_DAYS from config.py when days is not supplied.
    """
    import csv as _csv
    from datetime import date as _date
    from signal_logger import SIGNAL_LOG_PATH
    from config import STOP_LOSS_COOLDOWN_DAYS

    if days is None:
        days = STOP_LOSS_COOLDOWN_DAYS

    if not os.path.exists(SIGNAL_LOG_PATH):
        return False

    today = _date.today()
    try:
        with open(SIGNAL_LOG_PATH, newline="") as fh:
            for row in _csv.DictReader(fh):
                if row["ticker"] != ticker or row["actual_action"] != "STOP_LOSS_SELL":
                    continue
                try:
                    log_date = _date.fromisoformat(row["date"])
                    if (today - log_date).days <= days:
                        return True
                except ValueError:
                    continue
    except OSError:
        pass
    return False


# ---------------------------------------------------------------------------
# Stop-loss / take-profit
# ---------------------------------------------------------------------------
def check_position_limits(
    api: tradeapi.REST,
    owned: dict[str, float],
    equity: float,
    stop_pct: float = 10.0,
    take_pct: float = 15.0,
) -> tuple[list[str], dict[str, float]]:
    """
    Check every open position for a stop-loss or take-profit trigger.

    Loss formula: (entry_price - current_price) / entry_price * 100
    Gain formula: (current_price - entry_price) / entry_price * 100

    Stop-loss  — loss_pct > stop_pct (default 10 %): sells and sends 🛑 [STOP LOSS] alert
    Take-profit — gain_pct > take_pct (default 15 %): sells and sends 🎯 [TAKE PROFIT] alert

    Returns:
        exited — list of tickers that were sold
        owned  — updated positions dict (exited tickers removed)
    """
    positions = api.list_positions()
    exited    = []

    for position in positions:
        ticker        = position.symbol
        current_price = float(position.current_price)
        entry_price   = float(position.avg_entry_price)
        qty           = float(position.qty)

        loss_pct = (entry_price - current_price) / entry_price * 100
        gain_pct = (current_price - entry_price) / entry_price * 100

        if loss_pct > stop_pct:
            print(f"  [STOP LOSS]   {ticker} — down {loss_pct:.1f}% — selling {qty} shares")
            result = place_sell(api, ticker, qty)
            send_discord(
                f"🛑 **[STOP LOSS]** {ticker} sold — down {loss_pct:.1f}%  "
                f"(entry ${entry_price:.2f} → current ${current_price:.2f})"
            )
            if result["status"] == "placed":
                from signal_logger import log_signal
                log_signal(ticker, "SELL", current_price, qty, 0.0, "STOP_LOSS_SELL")
                exited.append(ticker)
                owned.pop(ticker, None)

        elif gain_pct > take_pct:
            print(f"  [TAKE PROFIT] {ticker} — up {gain_pct:.1f}% — selling {qty} shares")
            result = place_sell(api, ticker, qty)
            send_discord(
                f"🎯 **[TAKE PROFIT]** {ticker} sold — up {gain_pct:.1f}%  "
                f"(entry ${entry_price:.2f} → current ${current_price:.2f})"
            )
            if result["status"] == "placed":
                exited.append(ticker)
                owned.pop(ticker, None)

    return exited, owned


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


def build_post_order_alert(
    outcomes: list[dict],
    portfolio_value: float,
    timestamp: str,
    rebalancer_outcomes: list[dict] | None = None,
    shap_by_signal: dict | None = None,
) -> str:
    """
    Build the post-order Discord summary message.

    Each item in `outcomes` / `rebalancer_outcomes` has:
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

    # Rebalanced section — only shown when the rebalancer acted
    rb = rebalancer_outcomes or []
    rb_placed = [o for o in rb if o["status"] == "placed"]
    rb_failed = [o for o in rb if o["status"] == "error"]

    if rb_placed or rb_failed:
        lines.append("Rebalanced:")
        for o in rb_placed:
            lines.append(f"  SELL {o['ticker']:<6}  {o['qty']} share(s) @ ${o['price']:>8.2f}")
        for o in rb_failed:
            lines.append(f"  SELL {o['ticker']:<6}  {o['qty']} share(s) @ ${o['price']:>8.2f}  [ORDER FAILED: {o['reason']}]")
    else:
        lines.append("Rebalanced: none")

    lines.append("```")

    # SHAP section — average feature contributions per signal group, top 3 by |value|
    def _avg_top3_shap(shap_list: list[dict]) -> str:
        if not shap_list:
            return "—"
        keys = shap_list[0].keys()
        avg  = {k: sum(d.get(k, 0.0) for d in shap_list) / len(shap_list) for k in keys}
        top3 = sorted(avg.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        return ", ".join(f"{k.lower()} {v:+.2f}" for k, v in top3)

    shap_data  = shap_by_signal or {}
    shap_lines = ["**SHAP — Top drivers:**"]
    for group in ("BUY", "HOLD", "SELL"):
        shap_list = shap_data.get(group, [])
        drivers   = _avg_top3_shap(shap_list)
        shap_lines.append(f"{group} ({len(shap_list)} tickers): {drivers}")
    lines.append("\n".join(shap_lines))

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

    # 3. Stop-loss check — runs before model signals so stopped positions
    #    are excluded from the signal execution pass
    owned  = get_owned_tickers(api)
    equity = get_equity(api)
    print(f"  Checking stop losses on {len(owned)} open position(s)...")
    exited, owned = check_position_limits(api, owned, equity)
    if exited:
        print(f"  Exited (stop-loss / take-profit): {exited}")
        owned  = get_owned_tickers(api)
        equity = get_equity(api)
    else:
        print("  No stop losses triggered.")

    # 4. Get today's signals
    signals = get_signals()
    if not signals:
        print("  No signals generated. Exiting.")
        sys.exit(0)

    # 5. Sort: SELLs first (confidence desc), then BUYs (confidence desc), then HOLDs
    sell_sigs = sorted(
        [s for s in signals if s["final_signal"] == "SELL"],
        key=lambda s: s.get("confidence", 0.0),
        reverse=True,
    )
    buy_sigs = sorted(
        [s for s in signals if s["final_signal"] == "BUY"],
        key=lambda s: s.get("confidence", 0.0),
        reverse=True,
    )
    hold_sigs = [s for s in signals if s["final_signal"] == "HOLD"]

    print(
        f"\n  Signal queue: {len(sell_sigs)} SELL | "
        f"{len(buy_sigs)} BUY | {len(hold_sigs)} HOLD"
    )
    print("-" * 60)

    from signal_logger import log_signal

    shap_by_signal      = {"BUY": [], "HOLD": [], "SELL": []}
    outcomes            = []
    rebalancer_outcomes = []

    # 6a. SELL pass — process all SELL signals first
    for r in sell_sigs:
        ticker     = r["ticker"]
        price      = r["current_price"]
        confidence = r.get("confidence", 0.0)
        note_str   = f" ({r['note']})" if r.get("note") else ""

        if r.get("shap_values"):
            shap_by_signal["SELL"].append(r["shap_values"])

        owned = get_owned_tickers(api)

        if ticker not in owned:
            print(f"  Signal: SELL {ticker}{note_str} — skipped (not owned)")
            log_signal(ticker, "SELL", price, 0, confidence, "NOT_OWNED")
            outcomes.append({
                "ticker": ticker, "action": "SELL", "qty": 0, "price": price,
                "status": "skipped", "order_id": "", "reason": "not owned",
            })
            continue

        qty    = owned[ticker]
        result = place_sell(api, ticker, qty)
        actual_action = "SELL" if result["status"] == "placed" else "SELL_ERROR"
        log_signal(ticker, "SELL", price, qty, confidence, actual_action)
        outcomes.append({
            "ticker":   ticker,
            "action":   "SELL",
            "qty":      qty,
            "price":    price,
            "status":   result["status"],
            "order_id": result.get("order_id", ""),
            "reason":   result.get("reason", ""),
        })
        if result["status"] == "placed":
            equity = get_equity(api)
        # Error → logged and recorded; loop continues to next signal

    # 6b. Refresh after SELLs, then run rebalancer, then refresh again
    owned  = get_owned_tickers(api)
    equity = get_equity(api)

    print("\n" + "-" * 60)
    print("  Running rebalancer...")
    rebalancer_outcomes = run_rebalancer(api)

    owned  = get_owned_tickers(api)
    equity = get_equity(api)
    print("-" * 60 + "\n")

    # 6c. BUY + HOLD pass
    for r in buy_sigs + hold_sigs:
        ticker     = r["ticker"]
        final_sig  = r["final_signal"]
        price      = r["current_price"]
        confidence = r.get("confidence", 0.0)
        note_str   = f" ({r['note']})" if r.get("note") else ""

        # ------------------------------------------------------------------
        # BUY
        # ------------------------------------------------------------------
        if final_sig == "BUY":
            # Skip tickers that were exited via stop-loss/take-profit this session
            if ticker in exited:
                print(f"  Signal: BUY {ticker}{note_str} — skipped (EXIT_SKIP)")
                log_signal(ticker, "BUY", price, 0, confidence, "EXIT_SKIP")
                outcomes.append({
                    "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                    "status": "skipped", "order_id": "", "reason": "EXIT_SKIP",
                })
                continue

            # Skip if a STOP_LOSS_SELL was logged for this ticker within the past 7 calendar days
            if was_stop_loss_recently(ticker):
                print(f"  Signal: BUY {ticker}{note_str} — skipped (COOLDOWN_SKIP)")
                log_signal(ticker, "BUY", price, 0, confidence, "COOLDOWN_SKIP")
                outcomes.append({
                    "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                    "status": "skipped", "order_id": "", "reason": "COOLDOWN_SKIP",
                })
                continue

            if r.get("shap_values"):
                shap_by_signal["BUY"].append(r["shap_values"])

            owned  = get_owned_tickers(api)
            equity = get_equity(api)

            bar        = api.get_latest_trade(ticker)
            live_price = float(bar.price)

            if ticker in owned:
                # Already owned — ask capital_allocator with fresh state
                alloc = check_add_to_position(
                    ticker                = ticker,
                    confidence_normalized = confidence / 100.0,
                    price                 = live_price,
                    shares_owned          = owned[ticker],
                    portfolio_value       = equity,
                )
                skip_reason = alloc["skip_reason"]

                if skip_reason == "INVALID_HEADROOM":
                    print(f"  Signal: BUY {ticker}{note_str} — skipped (INVALID_HEADROOM)")
                    log_signal(ticker, "BUY", price, 0, confidence, "INVALID_HEADROOM")
                    outcomes.append({
                        "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                        "status": "skipped", "order_id": "", "reason": "INVALID_HEADROOM",
                    })
                    continue

                if skip_reason:
                    actual_skip = (
                        "CONFIDENCE_SKIP" if skip_reason == "CONFIDENCE_SKIP"
                        else INSUFFICIENT_EQUITY
                    )
                    print(f"  Signal: BUY {ticker}{note_str} — skipped ({skip_reason})")
                    log_signal(ticker, "BUY", price, 0, confidence, actual_skip)
                    outcomes.append({
                        "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                        "status": "skipped", "order_id": "", "reason": skip_reason,
                    })
                    continue

                qty    = alloc["shares_to_buy"]
                is_add = True

            else:
                # New position
                fraction = get_position_size(confidence)
                if fraction is None:
                    print(f"  Signal: BUY {ticker}{note_str} — skipped (below confidence threshold)")
                    log_signal(ticker, "BUY", price, 0, confidence, "CONFIDENCE_SKIP")
                    outcomes.append({
                        "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                        "status": "skipped", "order_id": "", "reason": "below confidence threshold",
                    })
                    continue

                qty = compute_buy_qty(equity, live_price, fraction)
                if qty <= 0:
                    print(f"  Signal: BUY {ticker}{note_str} — skipped (insufficient equity)")
                    log_signal(ticker, "BUY", price, 0, confidence, INSUFFICIENT_EQUITY)
                    outcomes.append({
                        "ticker": ticker, "action": "BUY", "qty": 0, "price": price,
                        "status": "skipped", "order_id": "", "reason": "insufficient equity",
                    })
                    continue

                is_add = False

            label = "add to position" if is_add else "new position"
            print(f"  Signal: BUY {ticker}{note_str} @ ${price:.2f} ({label})")
            result = place_buy(api, ticker, qty)
            if result["status"] == "placed":
                time.sleep(1)
            actual_action = (
                ("ADD_TO_POSITION" if is_add else "BUY")
                if result["status"] == "placed"
                else "BUY_ERROR"
            )
            log_signal(ticker, "BUY", price, qty, confidence, actual_action)
            outcomes.append({
                "ticker":   ticker,
                "action":   "BUY",
                "qty":      qty,
                "price":    price,
                "status":   result["status"],
                "order_id": result.get("order_id", ""),
                "reason":   result.get("reason", ""),
            })

        # ------------------------------------------------------------------
        # HOLD
        # ------------------------------------------------------------------
        else:
            if r.get("shap_values"):
                shap_by_signal["HOLD"].append(r["shap_values"])

            print(f"  Signal: HOLD {ticker}{note_str} — no action")
            log_signal(ticker, "HOLD", price, 0, confidence, r.get("veto_reason") or "HOLD")
            outcomes.append({
                "ticker":   ticker,
                "action":   "HOLD",
                "qty":      0,
                "price":    price,
                "status":   "skipped",
                "order_id": "",
                "reason":   "HOLD signal",
            })

    print("-" * 60)

    # 7. Single post-execution Discord summary (after all trades complete)
    portfolio_value = get_equity(api)
    timestamp_end   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    send_discord(build_post_order_alert(outcomes, portfolio_value, timestamp_end, rebalancer_outcomes, shap_by_signal))

    print(f"\n  Portfolio value: ${portfolio_value:,.2f}")
    print("  Done.\n")


if __name__ == "__main__":
    run()
