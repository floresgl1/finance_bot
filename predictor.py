"""
Generates Buy / Sell / Hold recommendations on the latest market data,
with a sentiment veto layer powered by FinBERT news analysis.

Veto rules (thresholds are one-sided — only vetoes, no upgrades):
  BUY  + sentiment < -0.2  =>  HOLD  ("sentiment veto")
  SELL + sentiment > +0.2  =>  HOLD  ("sentiment veto")
  All other combinations   =>  signal unchanged
"""

import os
import sys
import io
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd

from config import WATCHLIST, MODEL_DIR, MODEL_FILENAME, CONFIDENCE_THRESHOLD, FEATURE_COLUMNS
from features import load_and_process
from sentiment import get_sentiment_all

EARNINGS_DIR       = os.path.join(os.path.dirname(__file__), "data", "earnings")
SENTIMENT_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "sentiment", "sentiment_scores.csv")

# Ensure the terminal can render the star / warning emoji on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Veto thresholds: signals that conflict with sentiment this strongly are held
VETO_BUY_THRESHOLD  = -0.2   # BUY vetoed when sentiment falls below this
VETO_SELL_THRESHOLD =  0.2   # SELL vetoed when sentiment rises above this


def load_model() -> dict:
    """
    Load the XGBoost model bundle from MODEL_DIR.

    Returns a dict with keys:
        model   — trained XGBClassifier
        encoder — LabelEncoder that maps BUY/HOLD/SELL ↔ 0/1/2
    """
    path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at {path} — run trainer.py first."
        )
    return joblib.load(path)


def predict_ticker(ticker: str, model_bundle: dict) -> dict:
    """
    Generate a base recommendation for a single ticker.

    Args:
        ticker       — the stock symbol
        model_bundle — dict with keys 'model' (XGBClassifier) and
                       'encoder' (LabelEncoder) as saved by trainer.py

    Returns a dict with:
        ticker        — the stock symbol
        signal        — raw model signal: 'BUY', 'SELL', or 'HOLD'
        confidence    — highest class probability as a percentage
        current_price — today's closing price in USD
    """
    model   = model_bundle["model"]
    encoder = model_bundle["encoder"]

    df = load_and_process(ticker)
    latest = df.iloc[-1]

    current_price = float(latest["Close"])
    X = latest[FEATURE_COLUMNS].values.reshape(1, -1)

    proba    = model.predict_proba(X)[0]
    top_idx  = int(np.argmax(proba))
    top_prob = float(proba[top_idx])
    # Decode integer prediction back to string label
    signal   = encoder.inverse_transform([top_idx])[0] if top_prob >= CONFIDENCE_THRESHOLD else "HOLD"

    return {
        "ticker":        ticker,
        "signal":        signal,
        "confidence":    round(top_prob * 100, 1),
        "current_price": round(current_price, 2),
    }


def apply_sentiment_veto(signal: str, score: float) -> tuple:
    """
    Apply the sentiment veto to a model signal.

    A veto only downgrades — it never upgrades.  A BUY with sufficiently
    negative news becomes HOLD; a SELL with sufficiently positive news
    becomes HOLD.  Everything else passes through unchanged.

    Returns:
        final_signal — 'BUY', 'SELL', or 'HOLD'
        note         — 'sentiment veto' if the signal was changed, else ''
    """
    if signal == "BUY" and score < VETO_BUY_THRESHOLD:
        return "HOLD", "sentiment veto"
    if signal == "SELL" and score > VETO_SELL_THRESHOLD:
        return "HOLD", "sentiment veto"
    return signal, ""


def get_recent_earnings_surprise(ticker: str, window_days: int = 90) -> float | None:
    """
    Return the most recent Surprise(%) for a ticker within the last window_days days.

    Returns None if the earnings CSV is missing or no report falls within the window.
    """
    path = os.path.join(EARNINGS_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        today    = pd.Timestamp(date.today())
        cutoff   = today - pd.Timedelta(days=window_days)
        recent   = df[(df.index <= today) & (df.index >= cutoff)]
        if recent.empty:
            return None
        return float(recent["Surprise(%)"].iloc[recent.index.argmax()])
    except Exception:
        return None


def apply_earnings_veto(signal: str, surprise: float | None) -> tuple:
    """
    Apply the earnings surprise veto to a model signal.

    Rules (only downgrades — never upgrades):
      BUY  + surprise < -15%  =>  HOLD  ("earnings veto")
      SELL + surprise > +15%  =>  HOLD  ("earnings veto")
      No recent data (None)   =>  signal unchanged

    Returns:
        final_signal — 'BUY', 'SELL', or 'HOLD'
        note         — 'earnings veto' if the signal was changed, else ''
    """
    if surprise is None:
        return signal, ""
    if signal == "BUY" and surprise < -15.0:
        return "HOLD", "earnings veto"
    if signal == "SELL" and surprise > 15.0:
        return "HOLD", "earnings veto"
    return signal, ""


def load_sentiment_df() -> pd.DataFrame:
    """
    Load data/sentiment/sentiment_scores.csv once at startup.

    Returns an empty DataFrame (with the expected columns) if the file is
    missing or cannot be parsed, so callers never need to guard for None.
    """
    empty = pd.DataFrame(columns=["Date", "Ticker", "sentiment_score"])
    if not os.path.exists(SENTIMENT_CSV_PATH):
        return empty
    try:
        df = pd.read_csv(SENTIMENT_CSV_PATH, parse_dates=["Date"])
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        return df
    except Exception:
        return empty


def sentiment_veto(
    trade_date,
    ticker: str,
    model_signal: str,
    sentiment_df: pd.DataFrame,
) -> bool:
    """
    Return True if vetoed (signal should become HOLD),
    False if the trade passes through unchanged.

    Looks up the sentiment score for *ticker* on *trade_date* from the
    pre-loaded sentiment_df (data/sentiment/sentiment_scores.csv).

    Veto rules (only downgrades — never upgrades):
      BUY  + sentiment_score <= -0.5  →  True  (HOLD)
      SELL + sentiment_score >= +0.5  →  True  (HOLD)
      No data present or score == 0.0 →  False (no veto)
    """
    if sentiment_df.empty:
        return False

    # Normalise trade_date to a plain date object for comparison
    if hasattr(trade_date, "date"):
        trade_date = trade_date.date()

    rows = sentiment_df[
        (sentiment_df["Ticker"] == ticker) &
        (sentiment_df["Date"] == trade_date)
    ]

    if rows.empty:
        return False

    score = float(rows["sentiment_score"].iloc[-1])

    # Treat an uninitialised / missing score as no data
    if score == 0.0:
        return False

    if model_signal == "BUY" and score <= -0.5:
        return True
    if model_signal == "SELL" and score >= 0.5:
        return True
    return False


def run_bot() -> None:
    """
    Run predictions for every ticker in WATCHLIST, apply the sentiment
    veto then the earnings veto, and print a formatted recommendation
    table with a summary.
    """
    model = load_model()

    print("Fetching news sentiment for all tickers...")
    sentiments = get_sentiment_all(WATCHLIST)
    print()

    results         = []
    sentiment_vetos = []
    earnings_vetos  = []
    errors          = []

    for ticker in WATCHLIST:
        try:
            r     = predict_ticker(ticker, model)
            sent  = sentiments.get(ticker, {})
            score = sent.get("sentiment_score", 0.0)

            # 1. Sentiment veto
            post_sent_signal, sent_note = apply_sentiment_veto(r["signal"], score)

            # 2. Earnings veto (applied to signal after sentiment veto)
            surprise = get_recent_earnings_surprise(ticker)
            final_signal, earn_note = apply_earnings_veto(post_sent_signal, surprise)

            note = earn_note or sent_note
            r["final_signal"] = final_signal
            r["sentiment"]    = score
            r["note"]         = note

            results.append(r)

            if sent_note:
                sentiment_vetos.append(
                    f"  {ticker}: {r['signal']} (conf {r['confidence']}%) "
                    f"vetoed — sentiment {score:+.3f}"
                )
            if earn_note:
                earnings_vetos.append(
                    f"  {ticker}: {post_sent_signal} (conf {r['confidence']}%) "
                    f"vetoed — earnings surprise {surprise:+.1f}%"
                )
        except Exception as e:
            errors.append((ticker, str(e)))

    # Sort: BUY first, then SELL, then HOLD; within each group descending confidence
    signal_order = {"BUY": 0, "SELL": 1, "HOLD": 2}
    results.sort(key=lambda r: (signal_order.get(r["final_signal"], 2), -r["confidence"]))

    # --- Print table ---
    w = {"ticker": 6, "signal": 6, "conf": 12, "price": 10, "sent": 9, "note": 14}
    div = (
        "+" + "-" * (w["ticker"] + 2)
        + "+" + "-" * (w["signal"] + 2)
        + "+" + "-" * (w["conf"]   + 2)
        + "+" + "-" * (w["price"]  + 2)
        + "+" + "-" * (w["sent"]   + 2)
        + "+" + "-" * (w["note"]   + 2) + "+"
    )
    header = (
        f"| {'Ticker':<{w['ticker']}} "
        f"| {'Signal':<{w['signal']}} "
        f"| {'Confidence':>{w['conf']}} "
        f"| {'Price':>{w['price']}} "
        f"| {'Sentiment':>{w['sent']}} "
        f"| {'Note':<{w['note']}} |"
    )

    print(f"  Stock Analysis Bot  --  {date.today()}  "
          f"(threshold: {CONFIDENCE_THRESHOLD:.0%})")
    print(div)
    print(header)
    print(div)

    for r in results:
        signal    = r["final_signal"]
        conf      = f"{r['confidence']}%"
        price     = f"${r['current_price']:,.2f}"
        sentiment = f"{r['sentiment']:+.3f}"
        note      = r["note"]

        row = (
            f"| {r['ticker']:<{w['ticker']}} "
            f"| {signal:<{w['signal']}} "
            f"| {conf:>{w['conf']}} "
            f"| {price:>{w['price']}} "
            f"| {sentiment:>{w['sent']}} "
            f"| {note:<{w['note']}} |"
        )

        if signal == "BUY":
            print(f">>> {row}")
        elif signal == "SELL":
            print(f"  ! {row}")
        else:
            print(f"    {row}")

    print(div)

    # --- Summary ---
    final_signals = [r["final_signal"] for r in results]
    n_buy  = final_signals.count("BUY")
    n_sell = final_signals.count("SELL")
    n_hold = final_signals.count("HOLD")

    print(f"\n  Summary:  {n_buy} BUY  |  {n_sell} SELL  |  {n_hold} HOLD  "
          f"|  {len(sentiment_vetos)} vetoed by sentiment  "
          f"|  {len(earnings_vetos)} vetoed by earnings")

    if sentiment_vetos:
        print("\n  Sentiment vetoes (signal overridden to HOLD):")
        for v in sentiment_vetos:
            print(v)

    if earnings_vetos:
        print("\n  Earnings vetoes (signal overridden to HOLD):")
        for v in earnings_vetos:
            print(v)

    if errors:
        print("\n  Skipped:")
        for ticker, msg in errors:
            print(f"    [SKIP] {ticker} -- {msg}")


if __name__ == "__main__":
    run_bot()
