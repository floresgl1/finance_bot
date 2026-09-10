"""
generate_signals.py — Phase 1 of the pre-trade agent pipeline.

Runs on PythonAnywhere at 14:00 UTC (1 hour before live_trader.py).
Generates model signals, applies earnings + sentiment vetoes, and writes
surviving BUY/SELL signals to data/pending_signals.json so the GitHub
Actions agent can evaluate them before execution.

This script calls the same predictor functions as live_trader.get_signals()
but does NOT import from live_trader.py — it has no Alpaca dependency.

Exit codes:
    0 — signals written (or weekend/holiday skip)
    1 — fatal error (missing model, unreadable data)
"""

import json
import os
import sys
from datetime import date, datetime, timezone

import pandas as pd

from config import (
    DATA_DIR,
    PENDING_SIGNALS_PATH,
    WATCHLIST,
)
from features import StaleMarketDataError
from predictor import (
    apply_earnings_veto,
    apply_sentiment_veto,
    get_recent_earnings_surprise,
    is_ticker_stale,
    load_model,
    predict_ticker,
)


def _is_weekend() -> bool:
    """True on Saturday (5) or Sunday (6)."""
    return date.today().weekday() >= 5


def _check_data_freshness() -> bool:
    """Quick sanity check: at least one ticker CSV was modified today.

    This is a lightweight guard — live_trader.py runs its own thorough
    freshness gate at execution time. We just need to confirm the 12:00
    UTC data refresh has completed before we generate signals.
    """
    today_str = date.today().isoformat()
    for ticker in WATCHLIST[:3]:  # spot-check first 3
        csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            last_date = str(df["Date"].iloc[-1])[:10]
            if last_date >= today_str:
                return True
        except Exception:
            continue
    return False


def _load_sentiment_df():
    """Load sentiment scores CSV if it exists. Returns None if unavailable."""
    sentiment_path = os.path.join(DATA_DIR, "sentiment", "sentiment_scores.csv")
    if not os.path.exists(sentiment_path):
        return None
    try:
        return pd.read_csv(sentiment_path)
    except Exception:
        return None


def _extract_top_shap(shap_values: dict, n: int = 3) -> list[str]:
    """Return the top N feature names by SHAP value (descending).

    Same sort order as signal_logger.py line 157.
    """
    if not shap_values:
        return []
    sorted_features = sorted(
        shap_values.items(), key=lambda x: x[1], reverse=True
    )
    return [name for name, _ in sorted_features[:n]]


def _write_atomic(path: str, data: dict) -> None:
    """Atomic JSON write: tmp + fsync + os.replace."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def generate_signals() -> dict:
    """Generate model signals and return the pending-signals envelope.

    Returns:
        dict with keys {date, generated_at, signals}. signals is a list
        of dicts, each with {ticker, model_signal, confidence, shap_values}.
        Only surviving BUY/SELL signals are included.
    """
    model = load_model()
    sentiment_df = _load_sentiment_df()
    today_str = date.today().isoformat()

    pending = []

    for ticker in WATCHLIST:
        try:
            # Staleness check
            is_stale, days_old = is_ticker_stale(ticker)
            if is_stale:
                print(f"  [STALE_SKIP] {ticker} ({days_old}d old)")
                continue

            # Model prediction
            r = predict_ticker(ticker, model)
            model_signal = r["signal"]

            # 1. Earnings veto
            surprise = get_recent_earnings_surprise(ticker)
            post_earnings, earn_note = apply_earnings_veto(
                model_signal, surprise
            )

            # 2. Sentiment veto (only if not already HOLD from earnings)
            final_signal = post_earnings
            if post_earnings != "HOLD" and sentiment_df is not None:
                rows = sentiment_df[
                    (sentiment_df["Ticker"] == ticker)
                    & (sentiment_df["Date"] == today_str)
                ]
                sent_score = (
                    float(rows["sentiment_score"].iloc[-1])
                    if not rows.empty
                    else 0.0
                )
                final_signal, _sent_note = apply_sentiment_veto(
                    post_earnings, sent_score
                )

            # Only surviving BUY/SELL go to the agent
            if final_signal in ("BUY", "SELL"):
                pending.append(
                    {
                        "ticker": ticker,
                        "model_signal": final_signal,
                        "confidence": r["confidence"],
                        "shap_values": _extract_top_shap(r["shap_values"]),
                    }
                )
                print(f"  [{final_signal}] {ticker} conf={r['confidence']}")
            else:
                reason = earn_note or "HOLD"
                print(f"  [SKIP] {ticker} → {final_signal} ({reason})")

        except StaleMarketDataError as exc:
            print(f"  [STALE_MARKET_DATA] {ticker} — {exc}")
        except Exception as exc:
            print(f"  [ERROR] {ticker} — {type(exc).__name__}: {exc}")

    generated_at = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    return {
        "date": today_str,
        "generated_at": generated_at,
        "signals": pending,
    }


def main() -> int:
    if _is_weekend():
        print("[SKIP] Weekend — no signals to generate.")
        return 0

    if not _check_data_freshness():
        print("[WARN] Market data may not be fresh — generating anyway.")
        # Continue rather than halt: live_trader.py has its own freshness
        # gate and will catch truly stale data. A false negative here
        # (e.g. CSV dates lag by timezone) should not block the agent.

    print(f"[START] Generating signals for {date.today().isoformat()}")
    print(f"  Watchlist: {len(WATCHLIST)} tickers")

    try:
        envelope = generate_signals()
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _write_atomic(PENDING_SIGNALS_PATH, envelope)
    n = len(envelope["signals"])
    print(f"[DONE] Wrote {n} signal(s) to {PENDING_SIGNALS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
