"""
Streamlit dashboard for the Stock Analysis Bot.

Runs fresh predictions on every page load by invoking the same pipeline
used by predictor.py, then displays a styled signals table and rolling
backtest performance metrics read from rolling_backtest_results.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project modules are importable regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIDENCE_THRESHOLD, FEATURE_COLUMNS, WATCHLIST
from features import load_and_process
from predictor import apply_sentiment_veto, load_model
from sentiment import get_sentiment_all

INITIAL_CAPITAL = 10_000   # must match backtest.py
RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "rolling_backtest_results.csv",
)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _run_predictions() -> tuple[list[dict], list[tuple]]:
    """
    Run the full prediction pipeline for every ticker in WATCHLIST.

    Loads the trained model, computes per-class probabilities for the
    latest row of each ticker, applies the confidence threshold to get
    the raw model signal, then applies the FinBERT sentiment veto.

    Returns:
        rows   — list of signal dicts, one per ticker
        errors — list of (ticker, error_message) for any skipped tickers
    """
    model        = load_model()
    class_labels = list(model.classes_)   # e.g. ['BUY', 'HOLD', 'SELL']
    sentiments   = get_sentiment_all(WATCHLIST)

    rows, errors = [], []

    for ticker in WATCHLIST:
        try:
            df            = load_and_process(ticker)
            latest        = df.iloc[-1]
            current_price = float(latest["Close"])
            X             = latest[FEATURE_COLUMNS].values.reshape(1, -1)

            proba    = model.predict_proba(X)[0]
            prob_map = {lbl: float(p) for lbl, p in zip(class_labels, proba)}

            top_idx    = int(np.argmax(proba))
            top_prob   = float(proba[top_idx])
            raw_signal = class_labels[top_idx] if top_prob >= CONFIDENCE_THRESHOLD else "HOLD"

            sent         = sentiments.get(ticker, {})
            score        = sent.get("sentiment_score", 0.0)
            final_signal, note = apply_sentiment_veto(raw_signal, score)

            rows.append({
                "Ticker":             ticker,
                "Current Price":      round(current_price, 2),
                "Signal":             final_signal,
                "BUY Prob":           round(prob_map.get("BUY",  0.0) * 100, 1),
                "SELL Prob":          round(prob_map.get("SELL", 0.0) * 100, 1),
                "HOLD Prob":          round(prob_map.get("HOLD", 0.0) * 100, 1),
                "Sentiment Override": "Yes" if note else "No",
                "_top_prob":          top_prob,   # used for sorting only
            })
        except Exception as exc:
            errors.append((ticker, str(exc)))

    return rows, errors


def _load_backtest_metrics() -> tuple[dict | None, pd.DataFrame | None]:
    """
    Parse rolling_backtest_results.csv and return summary stats and the
    full equity-curve DataFrame.
    """
    if not os.path.exists(RESULTS_CSV):
        return None, None

    df = pd.read_csv(RESULTS_CSV, parse_dates=["date"], index_col="date")

    pv           = df["portfolio_value"]
    final_value  = float(pv.iloc[-1])
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    max_drawdown = float(((pv - pv.cummax()) / pv.cummax() * 100).min())

    stats = {
        "date_range":   (
            f"{df.index[0].strftime('%Y-%m-%d')} to "
            f"{df.index[-1].strftime('%Y-%m-%d')}"
        ),
        "final_value":  final_value,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "n_days":       len(df),
    }

    return stats, df


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def _style_signal(val: str) -> str:
    return {
        "BUY":  "color: #27ae60; font-weight: bold",
        "SELL": "color: #e74c3c; font-weight: bold",
        "HOLD": "color: #7f8c8d",
    }.get(val, "")


def _style_override(val: str) -> str:
    return "color: #e67e22; font-weight: bold" if val == "Yes" else ""


# ---------------------------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Stock Analysis Bot",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )

    st.title("Stock Analysis Bot")
    st.caption(
        f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}  ·  "
        f"Watchlist: {', '.join(WATCHLIST)}"
    )

    # --- Run pipeline with loading spinner ---
    with st.spinner(
        "Running pipeline — loading model, fetching features, scoring sentiment…"
    ):
        try:
            rows, errors = _run_predictions()
        except FileNotFoundError as exc:
            st.error(str(exc))
            return

    if not rows:
        st.error(
            "No predictions generated. "
            "Ensure the trained model (models/random_forest.joblib) and data "
            "CSVs (data/*.csv) are present."
        )
        if errors:
            for ticker, msg in errors:
                st.text(f"  {ticker}: {msg}")
        return

    # --- Build signals table, sorted by model confidence descending ---
    df_signals = (
        pd.DataFrame(rows)
        .sort_values("_top_prob", ascending=False)
        .reset_index(drop=True)
    )

    display_cols = [
        "Ticker", "Current Price", "Signal",
        "BUY Prob", "SELL Prob", "HOLD Prob",
        "Sentiment Override",
    ]
    df_display = df_signals[display_cols].copy()

    # Format price as currency string before passing to Styler
    df_display["Current Price"] = df_display["Current Price"].apply(
        lambda x: f"${x:,.2f}"
    )

    styled = (
        df_display.style
        .map(_style_signal,   subset=["Signal"])
        .map(_style_override, subset=["Sentiment Override"])
        .format({
            "BUY Prob":  "{:.1f}%",
            "SELL Prob": "{:.1f}%",
            "HOLD Prob": "{:.1f}%",
        })
    )

    st.subheader("Signals")
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if errors:
        with st.expander(f"{len(errors)} ticker(s) skipped"):
            for ticker, msg in errors:
                st.text(f"{ticker}: {msg}")

    st.divider()

    # --- Rolling backtest performance ---
    st.subheader("Rolling Backtest Performance")
    stats, equity_df = _load_backtest_metrics()

    if stats is None:
        st.info(
            "rolling_backtest_results.csv not found. "
            "Run retrain.py to generate backtest results."
        )
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Total Return",
        f"{stats['total_return']:+.2f}%",
    )
    col2.metric(
        "Final Portfolio Value",
        f"${stats['final_value']:,.2f}",
        delta=f"${stats['final_value'] - INITIAL_CAPITAL:+,.2f}",
    )
    col3.metric(
        "Max Drawdown",
        f"{stats['max_drawdown']:.2f}%",
    )
    col4.metric(
        "Trading Days",
        f"{stats['n_days']:,}",
    )

    st.caption(f"Period: {stats['date_range']}")

    st.line_chart(
        equity_df[["portfolio_value"]].rename(
            columns={"portfolio_value": "Portfolio Value ($)"}
        ),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
