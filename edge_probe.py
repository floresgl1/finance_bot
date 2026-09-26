"""
edge_probe.py — Does the model beat a trivial baseline, and is data volume the constraint?

Built to answer a specific question: the promotion gate's first run rejected a
freshly retrained challenger that had strictly more recent data than the
6-month-old champion. That ruled out staleness and raised a harder question —
whether the model has edge at all.

Two measurements, one run:

  1. BASELINES. Score always-predict-majority, always-HOLD, and a stratified
     random guess on the same test window as the model. Accuracy and BUY F1 are
     meaningless without them: on a label distribution that is ~40% BUY, a
     constant "BUY" prediction scores BUY F1 ~0.57 while knowing nothing.

  2. LOOKBACK SWEEP. Hold the test window FIXED and vary only how far back
     training data starts.

**DESIGN DECISION:**
The obvious probe -- re-running the pipeline with different HISTORY_PERIOD
values -- does not answer the question. The 70/20/10 split is proportional, so
changing the history length moves the test window too, and the resulting numbers
describe different periods. Here every model is trained on a different span and
scored on byte-identical test rows, which is the only way the comparison isolates
data volume.

**DESIGN DECISION:**
Downloads into its own directory and points features.DATA_DIR at it rather than
touching data/. The probe needs a longer history than HISTORY_PERIOD, and
overwriting the live CSVs would silently change what the next real training run
sees.

Usage:
    python edge_probe.py                          # lookback sweep + baselines
    python edge_probe.py --features               # SHAP ranking + top-K retrain
    python edge_probe.py --label-scales 1 1.5 2 3 # HOLD band width sweep
    python edge_probe.py --regimes                # RETURN vs buy-and-hold (decisive)
    python edge_probe.py --exposure               # sizing + regime-filter arms
    python edge_probe.py --horizons 3 7 14 21     # forward-return label window
    python edge_probe.py --horizons 3 7 21 --broad  # over 10 windows, not 4
    python edge_probe.py --exposure --window 2026-03-05 2026-09-04
                                                  # simulate one exact period
    python edge_probe.py --tiers --broad          # does exposure buy return?
    python edge_probe.py --exits --broad          # how should a position close?
    python edge_probe.py --test-start 2026-05-20
    python edge_probe.py --lookbacks 1 3 5 10
    python edge_probe.py --refresh                # re-download the history

See docs/EDGE_INVESTIGATION_2026-09-08.md for what each mode found.
"""

import argparse
import contextlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import (
    WATCHLIST,
    FEATURE_COLUMNS,
    XGB_PARAMS,
    CONFIDENCE_THRESHOLD,
    MAX_POSITION_PCT,
    MAX_TOTAL_EXPOSURE,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_DIR = os.path.join(_HERE, "data", "edge_probe")
PROBE_MARKET_DIR = os.path.join(PROBE_DIR, "market")
RESULTS_PATH = os.path.join(_HERE, "data", "edge_probe_results.json")

PROBE_HISTORY = "10y"
MARKET_SYMBOLS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "^VIX"]
LABELS = ["BUY", "HOLD", "SELL"]

# Top 5 by mean |SHAP|, from `--features`. See docs/EDGE_INVESTIGATION_2026-09-08.md.
TOP5_FEATURES = ["Volatility", "Return_60d", "RSI_14", "MACD_signal", "MACD"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def download_history(refresh: bool = False) -> None:
    """Fetch a long history in the exact CSV shape features.py expects."""
    os.makedirs(PROBE_MARKET_DIR, exist_ok=True)

    for symbol in MARKET_SYMBOLS:
        path = os.path.join(PROBE_MARKET_DIR, f"{symbol}.csv")
        if os.path.exists(path) and not refresh:
            continue
        df = yf.download(symbol, period=PROBE_HISTORY, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  [WARN] {symbol}: no data returned")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df[["Close"]].copy()
        close.index.name = "Date"
        close.to_csv(path)
        print(f"  [market] {symbol}: {len(close)} rows")

    for ticker in WATCHLIST:
        path = os.path.join(PROBE_DIR, f"{ticker}.csv")
        if os.path.exists(path) and not refresh:
            continue
        df = yf.download(ticker, period=PROBE_HISTORY, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  [WARN] {ticker}: no data returned")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close", "High", "Low", "Open", "Volume"]]
        df.index.name = "Date"
        df.to_csv(path)
        print(f"  [ticker] {ticker}: {len(df)} rows")


def build_probe_dataset(threshold_scale: float = 1.0) -> tuple[pd.DataFrame, pd.Series]:
    """Featured + labelled rows for every ticker, read from the probe directory.

    `threshold_scale` multiplies the VIX-derived label thresholds. 1.0 is the
    shipped labelling; higher values widen the HOLD band, producing fewer but
    higher-conviction BUY/SELL labels.
    """
    import features
    import labels as labels_mod

    features.DATA_DIR = PROBE_DIR
    features.MARKET_DATA_DIR = PROBE_MARKET_DIR
    features._load_market_close.cache_clear()
    labels_mod._SPY_PATH = os.path.join(PROBE_MARKET_DIR, "SPY.csv")
    labels_mod._VIX_PATH = os.path.join(PROBE_MARKET_DIR, "^VIX.csv")

    frames = []
    for ticker in WATCHLIST:
        try:
            df = labels_mod.add_labels(features.load_and_process(ticker))
            if threshold_scale != 1.0:
                df = _relabel(df, threshold_scale)
            frames.append(df)
        except Exception as exc:
            print(f"  {ticker}: FAILED {exc}")

    if not frames:
        raise RuntimeError("No ticker data loaded.")

    combined = pd.concat(frames).sort_index()
    return combined[FEATURE_COLUMNS], combined["Signal"]


def _relabel(df: pd.DataFrame, scale: float, window: int | None = None) -> pd.DataFrame:
    """Re-derive Signal from a scaled threshold.

    add_labels() leaves `threshold` and the inputs on the frame, so the
    relative return it compared against can be recovered rather than
    recomputed -- which keeps this consistent with labels.py by construction
    instead of by a duplicated formula that could drift.

    `window` defaults to labels._WINDOW rather than a literal 7: the
    `spy_return_7d` column holds a forward return over whatever window produced
    it, and comparing a 7-day stock return against a 14-day SPY return would be
    silently wrong.
    """
    import labels as labels_mod

    window = labels_mod._WINDOW if window is None else window
    df = df.copy()
    stock_return = df["Close"].pct_change(window).shift(-window)
    relative = stock_return - df["spy_return_7d"]
    scaled = df["threshold"] * scale

    df["Signal"] = np.select(
        [relative >= scaled, relative <= -scaled],
        ["BUY", "SELL"],
        default="HOLD",
    )
    # The tail rows add_labels already dropped have no forward return; the
    # scaled comparison leaves them HOLD, which matches that intent.
    return df


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(y_true, y_pred) -> dict:
    rep = classification_report(
        y_true, y_pred,
        labels=LABELS, target_names=LABELS,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy":      rep["accuracy"],
        "buy_f1":        rep["BUY"]["f1-score"],
        "buy_precision": rep["BUY"]["precision"],
        "buy_recall":    rep["BUY"]["recall"],
        "macro_f1":      rep["macro avg"]["f1-score"],
    }


def compute_baselines(y_test: pd.Series) -> dict:
    """Trivial strategies the model must beat to have demonstrated anything."""
    dist = y_test.value_counts()
    majority = dist.idxmax()
    out = {}

    out[f"always_{majority}"] = score(
        y_test, pd.Series([majority] * len(y_test), index=y_test.index)
    )
    out["always_HOLD"] = score(
        y_test, pd.Series(["HOLD"] * len(y_test), index=y_test.index)
    )

    rng = np.random.default_rng(0)
    props = (dist / dist.sum()).reindex(LABELS).fillna(0.0).values
    out["stratified_random"] = score(
        y_test,
        pd.Series(rng.choice(LABELS, size=len(y_test), p=props), index=y_test.index),
    )
    return out


def run_sweep(X, y, test_start: pd.Timestamp, lookbacks: list[int]) -> dict:
    """Train on progressively longer histories, score on one fixed test set."""
    test_mask = X.index >= test_start
    X_test, y_test = X[test_mask], y[test_mask]

    if X_test.empty:
        raise RuntimeError(f"No rows on or after {test_start.date()}.")

    dist = y_test.value_counts()
    print(f"\nFixed test window: {len(X_test)} rows, "
          f"{X_test.index.min().date()} -> {X_test.index.max().date()}")
    print(f"Label distribution: {dict(dist)}")

    results = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "test_rows": int(len(X_test)),
        "test_start": str(X_test.index.min().date()),
        "test_end": str(X_test.index.max().date()),
        "test_distribution": {k: int(v) for k, v in dist.items()},
        "baselines": compute_baselines(y_test),
        "lookbacks": {},
    }

    print("\n=== Baselines ===")
    for name, m in results["baselines"].items():
        print(f"  {name:<22} acc={m['accuracy']:.4f}  buy_f1={m['buy_f1']:.4f}  "
              f"macro_f1={m['macro_f1']:.4f}")

    print("\n=== Lookback sweep (test window held fixed) ===")
    for years in lookbacks:
        train_start = test_start - pd.DateOffset(years=years)
        mask = (X.index >= train_start) & (X.index < test_start)
        X_train, y_train = X[mask], y[mask]

        if len(X_train) < 500:
            print(f"  {years}y: only {len(X_train)} rows available, skipping")
            continue

        encoder = LabelEncoder()
        model = XGBClassifier(**XGB_PARAMS, random_state=42)
        model.fit(X_train, encoder.fit_transform(y_train))

        pred = encoder.inverse_transform(model.predict(X_test))
        metrics = score(y_test, pred)
        metrics["train_rows"] = int(len(X_train))
        metrics["train_start"] = str(X_train.index.min().date())
        metrics["prediction_distribution"] = {
            k: int(v) for k, v in pd.Series(pred).value_counts().items()
        }
        results["lookbacks"][f"{years}y"] = metrics

        print(f"  {years}y ({len(X_train):>6} rows from {metrics['train_start']}): "
              f"acc={metrics['accuracy']:.4f}  buy_f1={metrics['buy_f1']:.4f}  "
              f"macro_f1={metrics['macro_f1']:.4f}")

    return results


def format_verdict(results: dict) -> str:
    """State plainly whether the model beat the baselines and whether more data helped."""
    lines = ["", "=" * 74, "  VERDICT", "=" * 74]

    baselines = results["baselines"]
    lookbacks = results["lookbacks"]
    if not lookbacks:
        return "\n".join(lines + ["  No lookbacks ran."])

    best_name = max(lookbacks, key=lambda k: lookbacks[k]["accuracy"])
    best = lookbacks[best_name]
    majority_key = next((k for k in baselines if k.startswith("always_")
                         and k != "always_HOLD"), None)
    majority = baselines.get(majority_key, {})

    lines.append(f"  Best model: {best_name}  acc={best['accuracy']:.4f}  "
                 f"buy_f1={best['buy_f1']:.4f}  macro_f1={best['macro_f1']:.4f}")
    if majority:
        lines.append(f"  {majority_key}: acc={majority['accuracy']:.4f}  "
                     f"buy_f1={majority['buy_f1']:.4f}  macro_f1={majority['macro_f1']:.4f}")
        lines.append("")
        acc_edge = best["accuracy"] - majority["accuracy"]
        f1_edge = best["buy_f1"] - majority["buy_f1"]
        lines.append(f"  Accuracy edge over the constant predictor : {acc_edge:+.4f}")
        lines.append(f"  BUY F1 edge over the constant predictor   : {f1_edge:+.4f}")
        if f1_edge < 0:
            lines.append("    -> BUY F1 is LOWER than a constant prediction. Any gate")
            lines.append("       using BUY F1 as its metric is measuring class balance.")

    rnd = baselines.get("stratified_random", {})
    if rnd:
        lines.append(f"  macro F1 edge over stratified random      : "
                     f"{best['macro_f1'] - rnd['macro_f1']:+.4f}")

    by_rows = sorted(lookbacks.values(), key=lambda m: m["train_rows"])
    if len(by_rows) >= 2:
        lines.append("")
        lines.append(f"  Smallest training set ({by_rows[0]['train_rows']:>6} rows): "
                     f"acc={by_rows[0]['accuracy']:.4f}")
        lines.append(f"  Largest training set  ({by_rows[-1]['train_rows']:>6} rows): "
                     f"acc={by_rows[-1]['accuracy']:.4f}")
        if by_rows[-1]["accuracy"] <= by_rows[0]["accuracy"]:
            lines.append("    -> More data did NOT help. The constraint is the features")
            lines.append("       or the labels, not training-set size.")

    lines.append("=" * 74)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_feature_probe(test_start: pd.Timestamp, lookback_years: int = 3,
                      top_k: list[int] | None = None) -> dict:
    """Rank features by mean |SHAP|, then retrain on only the top K.

    **DESIGN DECISION:**
    SHAP alone answers "what does the model lean on", which is not the same as
    "what carries signal" -- a model will happily lean on noise. The top-K
    retrain is the decisive half: if 5 features match all 20, the other 15 are
    adding variance, not information.
    """
    import shap

    top_k = top_k or [3, 5, 10, 20]

    X, y = build_probe_dataset()
    test_mask = X.index >= test_start
    X_test, y_test = X[test_mask], y[test_mask]
    train_start = test_start - pd.DateOffset(years=lookback_years)
    train_mask = (X.index >= train_start) & (X.index < test_start)
    X_train, y_train = X[train_mask], y[train_mask]

    encoder = LabelEncoder()
    model = XGBClassifier(**XGB_PARAMS, random_state=42)
    model.fit(X_train, encoder.fit_transform(y_train))

    print(f"  Computing SHAP over {len(X_test)} test rows...")
    explainer = shap.TreeExplainer(model)
    values = explainer(X_test).values          # (rows, features, classes)
    mean_abs = np.abs(values).mean(axis=(0, 2))   # average across rows + classes

    ranking = sorted(
        zip(FEATURE_COLUMNS, mean_abs), key=lambda kv: kv[1], reverse=True
    )

    print("\n  Feature ranking by mean |SHAP|:")
    for i, (name, val) in enumerate(ranking, 1):
        print(f"    {i:>2}. {name:<20} {val:.5f}")

    baselines = compute_baselines(y_test)
    majority_key = max(
        (k for k in baselines if k.startswith("always_")),
        key=lambda k: baselines[k]["accuracy"],
    )
    majority_acc = baselines[majority_key]["accuracy"]

    print(f"\n  Retraining on top-K features "
          f"({majority_key} baseline acc={majority_acc:.4f}):")
    k_results = {}
    for k in top_k:
        cols = [name for name, _ in ranking[:k]]
        enc_k = LabelEncoder()
        model_k = XGBClassifier(**XGB_PARAMS, random_state=42)
        model_k.fit(X_train[cols], enc_k.fit_transform(y_train))
        pred = enc_k.inverse_transform(model_k.predict(X_test[cols]))
        m = score(y_test, pred)
        m["features"] = cols
        m["accuracy_edge"] = m["accuracy"] - majority_acc
        k_results[f"top_{k}"] = m
        print(f"    top-{k:<3} acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}  "
              f"edge={m['accuracy_edge']:>+.4f}")

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "feature_probe",
        "test_rows": int(len(X_test)),
        "shap_ranking": [{"feature": n, "mean_abs_shap": float(v)} for n, v in ranking],
        "baselines": baselines,
        "majority_baseline": majority_key,
        "top_k": k_results,
    }


# Windows chosen to span regimes. A strategy that runs at ~50% average exposure
# is structurally handicapped in a rally and structurally advantaged in a
# drawdown, so measuring it only on recent (rising) data cannot distinguish
# "no edge" from "defensive by construction".
REGIME_WINDOWS = {
    "covid crash 2020": ("2020-02-15", "2020-04-30"),
    "bear 2022":        ("2022-01-01", "2022-10-31"),
    "rally 2025-26":    ("2025-11-07", "2026-05-28"),
    "recent 2026":      ("2026-05-29", "2026-09-08"),
}

# REGIME_WINDOWS is deliberately regime-spanning, which makes it deliberately
# unrepresentative: two of its four windows are major drawdowns. That is the
# right sample for asking "is this defensive?" and the WRONG sample for a mean,
# because it hands a large bonus to any strategy that simply holds less stock.
# Finding F showed exactly that trap.
#
# BROAD_WINDOWS covers the same decade continuously, so bear periods appear in
# roughly the proportion they actually occurred. A result that survives both
# window sets is about the strategy; one that only survives REGIME_WINDOWS is
# about exposure.
#
# Nothing starts before late 2019: each window trains on the three years before
# it, and the probe history only reaches back ten years.
BROAD_WINDOWS = {
    "late 2019 bull":      ("2019-09-01", "2019-12-31"),
    "covid crash 2020":    ("2020-02-15", "2020-04-30"),
    "covid recovery 2020": ("2020-05-01", "2020-12-31"),
    "bull 2021":           ("2021-01-01", "2021-12-31"),
    "bear 2022":           ("2022-01-01", "2022-10-31"),
    "recovery 2023":       ("2023-01-01", "2023-12-31"),
    "bull 2024":           ("2024-01-01", "2024-12-31"),
    "choppy 2025":         ("2025-01-01", "2025-11-06"),
    "rally 2025-26":       ("2025-11-07", "2026-05-28"),
    "recent 2026":         ("2026-05-29", "2026-09-08"),
}


def run_regime_benchmark(cols: list[str], scale: float,
                         windows: dict | None = None,
                         lookback_years: int = 3,
                         per_ticker: dict | None = None) -> dict:
    """Walk-forward return benchmark against buy-and-hold, across regimes.

    Per window: train on the `lookback_years` immediately before it, generate
    signals inside it with that model, simulate, and compare to holding the
    same basket over the same dates.

    **DESIGN DECISION:**
    This trains a fresh model per window rather than reusing one. backtest.py's
    `train` split shows +365% precisely because the model was fitted on those
    dates; any window a single model spans is contaminated the same way.
    """
    import backtest as bt

    windows = windows or REGIME_WINDOWS
    if per_ticker is None:
        per_ticker = _build_per_ticker(scale)
    combined = pd.concat(per_ticker.values()).sort_index()
    results = {}

    for name, (start_s, end_s) in windows.items():
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        ticker_data = _window_signals(per_ticker, combined, cols, start, end,
                                      lookback_years, label=name)
        if ticker_data is None:
            continue

        dates = sorted(set().union(*[set(d.index) for d in ticker_data.values()]))
        strat = bt._summarise(*bt._simulate(dates, ticker_data), dates, name)
        hold = bt._summarise(*bt._simulate_buy_and_hold(dates, ticker_data), dates, name)
        delta = strat["total_return"] - hold["total_return"]

        results[name] = {
            "window": f"{start.date()} to {end.date()}",
            "strategy_return": strat["total_return"],
            "hold_return": hold["total_return"],
            "delta_pp": delta,
            "max_drawdown": strat["max_drawdown"],
            "n_trades": strat["n_trades"],
            "avg_exposure": strat["avg_exposure"],
        }

        print(f"  {name:<20}{strat['total_return']:>9.2f}%{hold['total_return']:>11.2f}%"
              f"{delta:>+9.2f}pp{strat['avg_exposure']:>8.0f}% exp"
              f"{strat['n_trades']:>7} trades")

    return results


def _window_signals(per_ticker: dict, combined: pd.DataFrame, cols: list[str],
                    start: pd.Timestamp, end: pd.Timestamp,
                    lookback_years: int, label: str = "") -> dict | None:
    """Train on the `lookback_years` before `start`; emit signals inside the window.

    Shared by the regime benchmark and the exposure probe so both score the same
    model. The exposure probe then rewrites these frames rather than retraining,
    which is what keeps its comparison about allocation policy and nothing else.
    """
    train_mask = (combined.index >= start - pd.DateOffset(years=lookback_years)) & \
                 (combined.index < start)
    X_train, y_train = combined[train_mask][cols], combined[train_mask]["Signal"]
    if len(X_train) < 500:
        print(f"  {label}: only {len(X_train)} training rows, skipping")
        return None

    encoder = LabelEncoder()
    model = XGBClassifier(**XGB_PARAMS, random_state=42)
    model.fit(X_train, encoder.fit_transform(y_train))
    classes = np.array(encoder.classes_)

    ticker_data = {}
    for ticker, df in per_ticker.items():
        win = df[(df.index >= start) & (df.index <= end)]
        if win.empty:
            continue
        proba = model.predict_proba(win[cols].values)
        top_idx = np.argmax(proba, axis=1)
        top_prob = proba[np.arange(len(proba)), top_idx]
        signals = classes[top_idx].copy()
        signals[top_prob < CONFIDENCE_THRESHOLD] = "HOLD"
        out = win[["Close"]].copy()
        out["Signal"] = signals
        out["Confidence"] = top_prob
        ticker_data[ticker] = out

    return ticker_data or None


# ---------------------------------------------------------------------------
# Exposure and regime-filter probe
# ---------------------------------------------------------------------------
#
# `--regimes` established that the strategy loses to buy-and-hold out of sample,
# and that improving the classifier made that worse. Two explanations were left
# open, and this mode separates them:
#
#   (a) the deficit is EXPOSURE. Confidence-scaled sizing keeps the book near
#       50% invested, so it structurally lags a 100%-invested basket. If that is
#       the whole story, sizing up should close the gap.
#   (b) the deficit is SELECTION. The names it picks underperform the basket, in
#       which case sizing up makes the loss larger, not smaller.
#
# The regime arms add the control that matters: a 200-day SMA filter on SPY is a
# one-line rule with no model in it. If "regime filter only" matches or beats
# "regime + model", the model contributes nothing the filter does not.

REGIME_SMA_WINDOW = 200

# Equal-weight-ish full investment for the regime arms: every name, no cash buffer.
_EQ_POSITION_PCT = 1.0 / max(len(WATCHLIST), 1)

# label -> (policy, MAX_POSITION_PCT, MAX_TOTAL_EXPOSURE)
#
# "shipped" tracks config.py, so it simulates the book the live bot builds:
# 8% per name, no portfolio cap. "concentrated (old sim)" is what backtest.py
# hardcoded before 2026-09-08 and is kept as an explicit arm, because it is the
# only way to see how much of the historical deficit was the sizing mismatch
# rather than the model.
EXPOSURE_ARMS = {
    "shipped":              ("shipped",           MAX_POSITION_PCT, MAX_TOTAL_EXPOSURE),
    "full-size buys":       ("full_size",         MAX_POSITION_PCT, MAX_TOTAL_EXPOSURE),
    "concentrated (old sim)": ("shipped",         0.20, 0.80),
    "regime filter only":   ("regime_only",       _EQ_POSITION_PCT, 1.00),
    "regime + model":       ("regime_plus_model", _EQ_POSITION_PCT, 1.00),
}


def _spy_regime() -> pd.Series:
    """Boolean 'risk-on' series: SPY above its 200-day simple moving average.

    Lagged one session, so each day is allocated from the previous close. A
    market-timing rule that reads the same close it trades on is the easiest way
    to manufacture an edge that does not exist, so this one does not.
    """
    path = os.path.join(PROBE_MARKET_DIR, "SPY.csv")
    spy = pd.read_csv(path)
    spy["Date"] = pd.to_datetime(spy["Date"], errors="coerce", utc=True)
    spy = spy.dropna(subset=["Date"]).set_index("Date")
    spy.index = spy.index.tz_localize(None)
    close = pd.to_numeric(spy["Close"], errors="coerce").dropna().sort_index()
    sma = close.rolling(REGIME_SMA_WINDOW).mean()
    # `fill_value` rather than a later fillna: shifting into NaN would promote
    # the mask to object dtype, which pandas 3 no longer silently downcasts.
    return (close > sma).shift(1, fill_value=False).astype(bool)


def _apply_policy(ticker_data: dict, policy: str, risk_on: pd.Series) -> dict:
    """Rewrite signal frames so each policy is expressible as different signals.

    **DESIGN DECISION:**
    Every arm runs through the same `backtest._simulate`, so slippage,
    commission, drawdown and accounting are identical across them. Only which
    rows say BUY, and how large each BUY is, changes. That rules out the
    simulator itself as an explanation for any difference between arms.
    """
    out = {}
    for ticker, df in ticker_data.items():
        frame = df.copy()
        if policy == "shipped":
            pass
        elif policy == "full_size":
            # Confidence scales the position, so conf 1.0 means a full-size buy.
            frame.loc[frame["Signal"] == "BUY", "Confidence"] = 1.0
        elif policy in ("regime_only", "regime_plus_model"):
            # Dates SPY has no row for fail safe to risk-off, i.e. to cash.
            on = risk_on.reindex(frame.index, fill_value=False).astype(bool).to_numpy()
            if policy == "regime_only":
                frame["Signal"] = np.where(on, "BUY", "HOLD")
                frame["Confidence"] = 1.0
            else:
                # Risk-on: hold the basket. Risk-off: defer to the model, which
                # is the one condition it appeared to handle well.
                frame["Signal"] = np.where(on, "BUY", frame["Signal"].to_numpy())
                frame["Confidence"] = np.where(on, 1.0,
                                               frame["Confidence"].to_numpy())
        else:
            raise ValueError(f"unknown policy: {policy!r}")
        out[ticker] = frame
    return out


@contextlib.contextmanager
def _sizing(max_position_pct: float, max_total_exposure: float):
    """Temporarily override backtest.py's sizing caps, then restore them."""
    import backtest as bt

    previous = (bt.MAX_POSITION_PCT, bt.MAX_TOTAL_EXPOSURE)
    bt.MAX_POSITION_PCT = max_position_pct
    bt.MAX_TOTAL_EXPOSURE = max_total_exposure
    try:
        yield
    finally:
        bt.MAX_POSITION_PCT, bt.MAX_TOTAL_EXPOSURE = previous


def run_exposure_probe(cols: list[str], scale: float,
                       windows: dict | None = None,
                       lookback_years: int = 3) -> dict:
    """Walk-forward comparison of allocation policies over the regime windows."""
    import backtest as bt

    windows = windows or REGIME_WINDOWS
    per_ticker = _build_per_ticker(scale)
    combined = pd.concat(per_ticker.values()).sort_index()
    risk_on = _spy_regime()
    results = {}

    for name, (start_s, end_s) in windows.items():
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        base = _window_signals(per_ticker, combined, cols, start, end,
                               lookback_years, label=name)
        if base is None:
            continue

        dates = sorted(set().union(*[set(d.index) for d in base.values()]))
        hold = bt._summarise(*bt._simulate_buy_and_hold(dates, base), dates, name)
        on_pct = 100.0 * float(risk_on.reindex(dates, fill_value=False).mean())

        print(f"\n  {name}  ({start.date()} to {end.date()})   "
              f"hold {hold['total_return']:+.2f}%   risk-on {on_pct:.0f}% of days")
        print(f"  {'arm':<22}{'return':>10}{'vs hold':>11}{'max DD':>10}"
              f"{'exp':>7}{'trades':>8}")
        print("  " + "-" * 68)

        arms = {}
        for label, (policy, max_pos, max_total) in EXPOSURE_ARMS.items():
            ticker_data = _apply_policy(base, policy, risk_on)
            with _sizing(max_pos, max_total):
                stats = bt._summarise(*bt._simulate(dates, ticker_data), dates, name)
            delta = stats["total_return"] - hold["total_return"]
            arms[label] = {
                "policy": policy,
                "max_position_pct": max_pos,
                "max_total_exposure": max_total,
                "total_return": stats["total_return"],
                "delta_pp": delta,
                "max_drawdown": stats["max_drawdown"],
                "avg_exposure": stats["avg_exposure"],
                "n_trades": stats["n_trades"],
            }
            print(f"  {label:<22}{stats['total_return']:>9.2f}%{delta:>+10.2f}pp"
                  f"{stats['max_drawdown']:>9.2f}%{stats['avg_exposure']:>6.0f}%"
                  f"{stats['n_trades']:>8}")

        results[name] = {
            "window": f"{start.date()} to {end.date()}",
            "hold_return": hold["total_return"],
            "hold_max_drawdown": hold["max_drawdown"],
            "risk_on_pct_of_days": on_pct,
            "arms": arms,
        }

    return results


def summarise_exposure(results: dict) -> str:
    """Aggregate the arms across windows and say what the spread means."""
    if not results:
        return "\n  No windows produced results."

    lines = ["", "  " + "=" * 68,
             f"  {'arm':<22}{'windows beating hold':>22}{'mean vs hold':>16}",
             "  " + "-" * 68]
    means = {}
    for label in EXPOSURE_ARMS:
        deltas = [w["arms"][label]["delta_pp"] for w in results.values()
                  if label in w["arms"]]
        if not deltas:
            continue
        means[label] = float(np.mean(deltas))
        wins = sum(1 for d in deltas if d > 0)
        lines.append(f"  {label:<22}{f'{wins}/{len(deltas)}':>22}"
                     f"{means[label]:>+15.2f}pp")
    lines.append("  " + "=" * 68)

    shipped = means.get("shipped")
    full = means.get("full-size buys")
    if shipped is not None and full is not None:
        if full > shipped:
            lines.append(f"  Sizing up helps ({shipped:+.2f}pp -> {full:+.2f}pp): part of")
            lines.append("  the deficit was exposure, not selection.")
        else:
            lines.append(f"  Sizing up makes it WORSE ({shipped:+.2f}pp -> {full:+.2f}pp).")
            lines.append("  The deficit is selection, not exposure -- more of these picks")
            lines.append("  is more of the problem.")

    old_sim = means.get("concentrated (old sim)")
    if shipped is not None and old_sim is not None:
        lines.append("")
        lines.append(f"  Live-matching sizing {shipped:+.2f}pp vs the pre-2026-09-08 "
                     f"simulated")
        lines.append(f"  sizing {old_sim:+.2f}pp: {shipped - old_sim:+.2f}pp of the "
                     f"historical deficit was")
        lines.append("  the simulator modelling a strategy that was never deployed.")

    only = means.get("regime filter only")
    plus = means.get("regime + model")
    if only is not None and plus is not None:
        lines.append("")
        if only >= plus:
            lines.append(f"  A 200-day SMA filter with NO model ({only:+.2f}pp) matches or")
            lines.append(f"  beats the same filter with the model layered on ({plus:+.2f}pp).")
            lines.append("  The model adds nothing the filter does not already do.")
        else:
            lines.append(f"  The model adds {plus - only:+.2f}pp on top of the regime filter")
            lines.append(f"  ({only:+.2f}pp -> {plus:+.2f}pp). Worth pursuing as an overlay.")
    return "\n".join(lines)


@contextlib.contextmanager
def _label_window(window: int | None):
    """Temporarily change the forward-return horizon labels.py labels on.

    labels._WINDOW drives three things at once -- the SPY forward return, the
    stock forward return, and how many unlabelable tail rows get dropped -- so
    patching the module constant is the only way to move all three together.
    """
    import labels as labels_mod

    if window is None:
        yield labels_mod._WINDOW
        return

    previous = labels_mod._WINDOW
    labels_mod._WINDOW = window
    try:
        yield window
    finally:
        labels_mod._WINDOW = previous


def _build_per_ticker(scale: float, window: int | None = None) -> dict:
    """Per-ticker featured+labelled frames from the probe data."""
    import features
    import labels as labels_mod

    features.DATA_DIR = PROBE_DIR
    features.MARKET_DATA_DIR = PROBE_MARKET_DIR
    features._load_market_close.cache_clear()
    labels_mod._SPY_PATH = os.path.join(PROBE_MARKET_DIR, "SPY.csv")
    labels_mod._VIX_PATH = os.path.join(PROBE_MARKET_DIR, "^VIX.csv")

    out = {}
    with _label_window(window):
        for ticker in WATCHLIST:
            try:
                df = labels_mod.add_labels(features.load_and_process(ticker))
                out[ticker] = _relabel(df, scale) if scale != 1.0 else df
            except Exception as exc:
                print(f"  {ticker}: FAILED {exc}")
    return out


# ---------------------------------------------------------------------------
# Label horizon probe
# ---------------------------------------------------------------------------
#
# The 7-day forward return in labels.py was chosen once and never tested. It
# decides what the word "signal" means here more than any feature does: too
# short and the label is mostly microstructure noise, too long and the model is
# asked to forecast something no daily technical indicator carries.
#
# **DESIGN DECISION:**
# Measured on RETURNS, not on classification metrics. Finding E showed a
# configuration with a clearly better classification edge producing worse
# returns, so a horizon sweep scored on accuracy would repeat that mistake.

# ---------------------------------------------------------------------------
# Position-tier sweep
# ---------------------------------------------------------------------------
#
# Every other finding says exposure dominates: model-driven arms at 33-55%
# invested lose 26-29pp to holding, while arms at 79-88% lose 7.6-9.5pp. The
# shipped tiers (3/5/7% against an 8% cap) were designed as *increments for
# topping up a position*, not as opening sizes, and reusing them for opens is
# what leaves the book around a third invested. Nobody chose that.
#
# **DESIGN DECISION:**
# The question asked here is whether the relationship is MONOTONE, not which
# level scores best. If return rises all the way to a nearly-full book the
# conclusion is structural -- "be invested" -- and the exact constants barely
# matter. If it peaks somewhere in the middle, that is a curve fit on ten
# windows and the right response is to leave the constants alone. Picking the
# argmax of this sweep is how finding H happened.
#
# One model is trained per window and shared by every tier level, so a
# difference between levels cannot be the model.

# label -> (SMALL_POSITION_PCT, NORMAL_POSITION_PCT, LARGE_POSITION_PCT)
DEFAULT_TIER_LEVELS = {
    "half of shipped":   (0.015, 0.025, 0.035),
    "shipped 3/5/7":     (0.030, 0.050, 0.070),
    "4/6/8":             (0.040, 0.060, 0.080),
    "5/6.5/8":           (0.050, 0.065, 0.080),
    "6/7/8":             (0.060, 0.070, 0.080),
    "flat at the cap":   (0.080, 0.080, 0.080),
}


@contextlib.contextmanager
def _tiers(small: float, normal: float, large: float):
    """Temporarily override the confidence-tier position sizes.

    capital_allocator.get_allocation_tier() reads these as module globals, and
    backtest._simulate() calls it, so patching them here changes both the
    simulated opens and the simulated top-ups together -- which is how the live
    bot behaves since they share one table.
    """
    import capital_allocator as alloc

    previous = (alloc.SMALL_POSITION_PCT, alloc.NORMAL_POSITION_PCT,
                alloc.LARGE_POSITION_PCT)
    alloc.SMALL_POSITION_PCT = small
    alloc.NORMAL_POSITION_PCT = normal
    alloc.LARGE_POSITION_PCT = large
    try:
        yield
    finally:
        (alloc.SMALL_POSITION_PCT, alloc.NORMAL_POSITION_PCT,
         alloc.LARGE_POSITION_PCT) = previous


def run_tier_sweep(levels: dict | None = None, cols: list[str] | None = None,
                   scale: float = 1.0, windows: dict | None = None,
                   lookback_years: int = 3) -> dict:
    """Simulate each tier level over every window, sharing one model per window."""
    import backtest as bt

    levels = levels or DEFAULT_TIER_LEVELS
    cols = cols or FEATURE_COLUMNS
    windows = windows or BROAD_WINDOWS
    per_ticker = _build_per_ticker(scale)
    if not per_ticker:
        return {}
    combined = pd.concat(per_ticker.values()).sort_index()

    results = {label: {"levels": lvl, "windows": {}} for label, lvl in levels.items()}

    for name, (start_s, end_s) in windows.items():
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        base = _window_signals(per_ticker, combined, cols, start, end,
                               lookback_years, label=name)
        if base is None:
            continue

        dates = sorted(set().union(*[set(d.index) for d in base.values()]))
        hold = bt._summarise(*bt._simulate_buy_and_hold(dates, base), dates, name)
        print(f"\n  {name}  (hold {hold['total_return']:+.2f}%)")
        print(f"  {'tier level':<20}{'return':>10}{'vs hold':>11}"
              f"{'max DD':>10}{'exp':>7}{'trades':>8}")
        print("  " + "-" * 66)

        for label, (small, normal, large) in levels.items():
            with _tiers(small, normal, large):
                stats = bt._summarise(*bt._simulate(dates, base), dates, name)
            delta = stats["total_return"] - hold["total_return"]
            results[label]["windows"][name] = {
                "hold_return":  hold["total_return"],
                "total_return": stats["total_return"],
                "delta_pp":     delta,
                "max_drawdown": stats["max_drawdown"],
                "avg_exposure": stats["avg_exposure"],
                "n_trades":     stats["n_trades"],
            }
            print(f"  {label:<20}{stats['total_return']:>9.2f}%{delta:>+10.2f}pp"
                  f"{stats['max_drawdown']:>9.2f}%{stats['avg_exposure']:>6.0f}%"
                  f"{stats['n_trades']:>8}")

    for label, res in results.items():
        rows = list(res["windows"].values())
        if not rows:
            continue
        res["mean_delta_pp"] = float(np.mean([r["delta_pp"] for r in rows]))
        res["mean_exposure"] = float(np.mean([r["avg_exposure"] for r in rows]))
        res["mean_max_drawdown"] = float(np.mean([r["max_drawdown"] for r in rows]))
        res["windows_beating_hold"] = sum(1 for r in rows if r["delta_pp"] > 0)
        res["n_windows"] = len(rows)

    return results


def summarise_tiers(results: dict) -> str:
    """Report the sweep and say whether exposure buys return monotonically."""
    scored = {k: v for k, v in results.items() if "mean_delta_pp" in v}
    if not scored:
        return "\n  No tier level produced results."

    ordered = sorted(scored.items(), key=lambda kv: kv[1]["mean_exposure"])

    lines = ["", "  " + "=" * 74,
             f"  {'tier level':<20}{'exposure':>10}{'beat hold':>12}"
             f"{'mean vs hold':>16}{'mean max DD':>14}",
             "  " + "-" * 74]
    for label, res in ordered:
        wins = f"{res['windows_beating_hold']}/{res['n_windows']}"
        lines.append(f"  {label:<20}{res['mean_exposure']:>9.0f}%{wins:>12}"
                     f"{res['mean_delta_pp']:>+15.2f}pp"
                     f"{res['mean_max_drawdown']:>13.2f}%")
    lines.append("  " + "=" * 74)

    deltas = [res["mean_delta_pp"] for _, res in ordered]
    monotone = all(b >= a for a, b in zip(deltas, deltas[1:]))
    best_label, best = max(scored.items(), key=lambda kv: kv[1]["mean_delta_pp"])
    least, most = ordered[0], ordered[-1]

    if monotone:
        lines += [
            f"  MONOTONE: return rises with exposure at every step, "
            f"{least[1]['mean_delta_pp']:+.2f}pp at",
            f"  {least[1]['mean_exposure']:.0f}% invested up to "
            f"{most[1]['mean_delta_pp']:+.2f}pp at {most[1]['mean_exposure']:.0f}%.",
            "  That is a structural result, not a fitted one: the conclusion is",
            "  'be invested', and the exact tier constants barely matter.",
        ]
    else:
        lines += [
            f"  NOT MONOTONE: the best level is '{best_label}' "
            f"({best['mean_delta_pp']:+.2f}pp at",
            f"  {best['mean_exposure']:.0f}% invested), with worse results on both sides.",
            "  A peak in the middle of ten windows is a curve fit. Do NOT adopt it;",
            "  leave the tier constants alone.",
        ]

    drawdown_cost = most[1]["mean_max_drawdown"] - least[1]["mean_max_drawdown"]
    lines.append("")
    lines.append(f"  Cost of the extra exposure: mean max drawdown "
                 f"{least[1]['mean_max_drawdown']:.2f}% -> "
                 f"{most[1]['mean_max_drawdown']:.2f}% ({drawdown_cost:+.2f}pp).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exit-policy probe
# ---------------------------------------------------------------------------
#
# The simulator and the live bot have never agreed on how a position closes.
# backtest._simulate() closed on any non-BUY signal; live, a HOLD does nothing
# at all -- hold_sigs is collected and only logged -- so a position survives
# until an explicit SELL, a stop, a take-profit or a rebalancer trim. The live
# log shows how lopsided that is: 47 REBALANCER_SELL and 12 stop backfills
# against 9 model SELLs.
#
# That matters more than it looks. Finding K showed the book is capped by how
# many names sit in BUY at once, so the exit rule is one of the few levers that
# changes exposure without touching the model.

EXIT_POLICY_LABELS = {
    "not_buy":              "exit on any non-BUY  (old simulator)",
    "sell_only":            "exit on SELL only    (live today)",
    "sell_or_stop":         "exit on SELL or stop",
    "horizon":              "exit at label horizon",
    "sell_stop_or_horizon": "exit on SELL, stop or horizon",
}


def run_exit_probe(policies: list[str] | None = None,
                   cols: list[str] | None = None, scale: float = 1.0,
                   windows: dict | None = None,
                   lookback_years: int = 3) -> dict:
    """Simulate each exit policy over every window, sharing one model per window."""
    import backtest as bt

    policies = policies or list(EXIT_POLICY_LABELS)
    cols = cols or FEATURE_COLUMNS
    windows = windows or BROAD_WINDOWS
    per_ticker = _build_per_ticker(scale)
    if not per_ticker:
        return {}
    combined = pd.concat(per_ticker.values()).sort_index()

    results = {p: {"label": EXIT_POLICY_LABELS.get(p, p), "windows": {}}
               for p in policies}

    for name, (start_s, end_s) in windows.items():
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        base = _window_signals(per_ticker, combined, cols, start, end,
                               lookback_years, label=name)
        if base is None:
            continue

        dates = sorted(set().union(*[set(d.index) for d in base.values()]))
        hold = bt._summarise(*bt._simulate_buy_and_hold(dates, base), dates, name)
        print(f"\n  {name}  (hold {hold['total_return']:+.2f}%)")
        print(f"  {'exit policy':<38}{'return':>9}{'vs hold':>10}"
              f"{'max DD':>9}{'exp':>6}{'trades':>8}")
        print("  " + "-" * 80)

        for policy in policies:
            stats = bt._summarise(
                *bt._simulate(dates, base, exit_policy=policy), dates, name
            )
            delta = stats["total_return"] - hold["total_return"]
            results[policy]["windows"][name] = {
                "hold_return":  hold["total_return"],
                "total_return": stats["total_return"],
                "delta_pp":     delta,
                "max_drawdown": stats["max_drawdown"],
                "avg_exposure": stats["avg_exposure"],
                "n_trades":     stats["n_trades"],
            }
            print(f"  {EXIT_POLICY_LABELS.get(policy, policy):<38}"
                  f"{stats['total_return']:>8.2f}%{delta:>+9.2f}pp"
                  f"{stats['max_drawdown']:>8.2f}%{stats['avg_exposure']:>5.0f}%"
                  f"{stats['n_trades']:>8}")

    for res in results.values():
        rows = list(res["windows"].values())
        if not rows:
            continue
        res["mean_delta_pp"] = float(np.mean([r["delta_pp"] for r in rows]))
        res["mean_exposure"] = float(np.mean([r["avg_exposure"] for r in rows]))
        res["mean_max_drawdown"] = float(np.mean([r["max_drawdown"] for r in rows]))
        res["mean_trades"] = float(np.mean([r["n_trades"] for r in rows]))
        res["windows_beating_hold"] = sum(1 for r in rows if r["delta_pp"] > 0)
        res["n_windows"] = len(rows)

    return results


def summarise_exits(results: dict) -> str:
    """Rank the exit policies and say what the live/simulated split is worth."""
    scored = {k: v for k, v in results.items() if "mean_delta_pp" in v}
    if not scored:
        return "\n  No exit policy produced results."

    lines = ["", "  " + "=" * 84,
             f"  {'exit policy':<38}{'beat hold':>11}{'mean vs hold':>15}"
             f"{'exp':>7}{'trades':>9}",
             "  " + "-" * 84]
    for policy, res in sorted(scored.items(), key=lambda kv: -kv[1]["mean_delta_pp"]):
        wins = f"{res['windows_beating_hold']}/{res['n_windows']}"
        lines.append(f"  {res['label']:<38}{wins:>11}"
                     f"{res['mean_delta_pp']:>+14.2f}pp"
                     f"{res['mean_exposure']:>6.0f}%{res['mean_trades']:>9.0f}")
    lines.append("  " + "=" * 84)

    old_sim = scored.get("not_buy")
    live = scored.get("sell_only")
    if old_sim and live:
        gap = live["mean_delta_pp"] - old_sim["mean_delta_pp"]
        lines.append("")
        if gap > 0:
            lines += [
                f"  The live rule (exit on SELL only) beats the old simulator rule",
                f"  (exit on any non-BUY) by {gap:+.2f}pp, at "
                f"{live['mean_exposure']:.0f}% exposure against "
                f"{old_sim['mean_exposure']:.0f}%.",
                "  Holding through HOLD is a feature, not the bug it looked like:",
                "  every simulated result before this understated the bot by that much.",
            ]
        else:
            lines += [
                f"  The live rule (exit on SELL only) trails the old simulator rule",
                f"  by {gap:.2f}pp. Holding through HOLD costs return; the bot should",
                "  exit on HOLD as well as on SELL.",
            ]

    best_policy, best = max(scored.items(), key=lambda kv: kv[1]["mean_delta_pp"])
    if best_policy not in ("not_buy", "sell_only"):
        lines += ["",
                  f"  Best overall: {best['label']} at {best['mean_delta_pp']:+.2f}pp",
                  f"  ({best['mean_exposure']:.0f}% exposure, "
                  f"{best['mean_trades']:.0f} trades per window). Neither the simulator",
                  "  nor the bot currently does this."]
    return "\n".join(lines)


DEFAULT_HORIZONS = [3, 5, 7, 14, 21]


def run_horizon_probe(horizons: list[int], cols: list[str] | None = None,
                      windows: dict | None = None,
                      lookback_years: int = 3) -> dict:
    """Re-label at each forward horizon and re-run the walk-forward benchmark."""
    cols = cols or FEATURE_COLUMNS
    results = {}

    for horizon in horizons:
        per_ticker = _build_per_ticker(1.0, window=horizon)
        if not per_ticker:
            print(f"  horizon {horizon}d: no tickers built, skipping")
            continue

        signals = pd.concat([df["Signal"] for df in per_ticker.values()])
        dist = signals.value_counts(normalize=True) * 100

        print(f"\n  horizon {horizon}d  "
              f"(BUY {dist.get('BUY', 0):.1f}%  "
              f"HOLD {dist.get('HOLD', 0):.1f}%  "
              f"SELL {dist.get('SELL', 0):.1f}%)")
        print(f"  {'window':<20}{'strategy':>10}{'hold':>11}{'delta':>11}")

        per_window = run_regime_benchmark(
            cols, 1.0, windows=windows,
            lookback_years=lookback_years, per_ticker=per_ticker,
        )
        if not per_window:
            continue

        deltas = [r["delta_pp"] for r in per_window.values()]
        results[f"horizon_{horizon}"] = {
            "horizon_days": horizon,
            "distribution_pct": {k: float(v) for k, v in dist.items()},
            "windows": per_window,
            "mean_delta_pp": float(np.mean(deltas)),
            "windows_beating_hold": sum(1 for d in deltas if d > 0),
            "n_windows": len(deltas),
        }

    return results


def summarise_horizons(results: dict) -> str:
    """Rank the horizons and say whether any of them clears buy-and-hold."""
    if not results:
        return "\n  No horizon produced results."

    lines = ["", "  " + "=" * 68,
             f"  {'label horizon':<20}{'windows beating hold':>24}{'mean vs hold':>16}",
             "  " + "-" * 68]
    for res in sorted(results.values(), key=lambda r: -r["mean_delta_pp"]):
        wins = f"{res['windows_beating_hold']}/{res['n_windows']}"
        lines.append(f"  {str(res['horizon_days']) + ' days':<20}{wins:>24}"
                     f"{res['mean_delta_pp']:>+15.2f}pp")
    lines.append("  " + "=" * 68)

    best = max(results.values(), key=lambda r: r["mean_delta_pp"])
    shipped = results.get("horizon_7")

    if best["mean_delta_pp"] <= 0:
        lines.append("  No horizon beats holding the basket. The label window was not")
        lines.append("  the binding constraint either.")
    elif shipped and best["horizon_days"] != shipped["horizon_days"]:
        lines.append(f"  {best['horizon_days']}-day labels beat holding "
                     f"({best['mean_delta_pp']:+.2f}pp) where the shipped 7-day "
                     f"labels do not")
        lines.append(f"  ({shipped['mean_delta_pp']:+.2f}pp). Worth a walk-forward "
                     f"confirmation before acting on it.")
    else:
        lines.append(f"  The shipped 7-day horizon is the best of those tested "
                     f"({best['mean_delta_pp']:+.2f}pp).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SELL information probe -- Stage 1, pre-registered 2026-09-26
# ---------------------------------------------------------------------------
#
# Finding L measured SELL's edge as classification: +0.0353 over its own
# constant baseline, ~5x BUY's. Finding E showed a classification edge can
# fail to move returns at all. Before building anything on SELL, this asks the
# narrow question directly: over the next label horizon, do names the model
# says SELL on actually do worse?
#
# **DESIGN DECISION -- the gap is measured within each ticker.**
# For each ticker, mean forward return on non-SELL days minus mean on SELL
# days, then averaged across tickers weighted by SELL-day count. A pooled
# comparison can be won by WHICH tickers the model flags (say, the volatile
# ones in a falling window) rather than WHEN it flags them. Timing is the
# skill an overlay would need. The pooled means are reported for context only.
#
# **DESIGN DECISION -- embargo the training labels.**
# A label is the forward return over the next `labels._WINDOW` rows, so the
# last rows before a window's start are labelled from prices inside it.
# _window_signals trains on everything before the start; here the last
# horizon's worth of business days is dropped first. The leak is small, but it
# tilts toward passing, which is the wrong direction for a pre-registered test.
#
# Pass criteria were fixed with the user BEFORE the first run, and are only
# evaluated on BROAD_WINDOWS. Do not tune them after seeing results -- that is
# how findings F, H and J went wrong.

SELL_INFO_MIN_WINDOWS = 7         # windows where SELL days underperform, of 10
SELL_INFO_MIN_MEAN_GAP = 0.002    # mean gap > 0.2%, a round trip at 0.1% slippage a side


def _forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Return from each close to the close `horizon` rows later."""
    return close.shift(-horizon) / close - 1.0


def sell_gap(frames: dict) -> dict:
    """Within-ticker gap between non-SELL and SELL forward returns.

    frames: ticker -> DataFrame with a `Signal` column and a `fwd` column.
    Rows without a forward return are dropped. A positive gap means SELL days
    were followed by worse returns, which is what SELL is supposed to mean.
    A ticker needs both SELL and non-SELL days to contribute a gap.
    """
    gaps, weights = [], []
    sell_parts, other_parts = [], []
    for frame in frames.values():
        f = frame.dropna(subset=["fwd"])
        is_sell = (f["Signal"] == "SELL").to_numpy()
        sell, other = f["fwd"][is_sell], f["fwd"][~is_sell]
        sell_parts.append(sell)
        other_parts.append(other)
        if len(sell) and len(other):
            gaps.append(float(other.mean() - sell.mean()))
            weights.append(len(sell))

    sell_all = pd.concat(sell_parts) if sell_parts else pd.Series(dtype=float)
    other_all = pd.concat(other_parts) if other_parts else pd.Series(dtype=float)
    return {
        "n_sell": int(len(sell_all)),
        "n_other": int(len(other_all)),
        "sell_fwd_mean": float(sell_all.mean()) if len(sell_all) else None,
        "other_fwd_mean": float(other_all.mean()) if len(other_all) else None,
        "gap": float(np.average(gaps, weights=weights)) if weights else None,
        "tickers_scored": len(gaps),
    }


def sell_info_verdict(windows: dict) -> dict:
    """Apply the pre-registered Stage 1 criteria to per-window results."""
    gaps = [w["gap"] for w in windows.values() if w.get("gap") is not None]
    wins = sum(1 for g in gaps if g > 0)
    mean_gap = float(np.mean(gaps)) if gaps else None
    # Rounded before the strict comparison: the mean of ten 0.002 gaps is
    # 0.0020000000000000005, which would otherwise pass a gap that only
    # equals the round-trip cost.
    passed = (
        wins >= SELL_INFO_MIN_WINDOWS
        and mean_gap is not None
        and round(mean_gap, 10) > SELL_INFO_MIN_MEAN_GAP
    )
    return {
        "windows_scored": len(gaps),
        "windows_sell_underperforms": wins,
        "mean_gap": mean_gap,
        "min_windows": SELL_INFO_MIN_WINDOWS,
        "min_mean_gap": SELL_INFO_MIN_MEAN_GAP,
        "passed": bool(passed),
    }


def run_sell_info_probe(windows: dict | None = None, lookback_years: int = 3) -> dict:
    """Score SELL's forward-return information over every window."""
    import labels as labels_mod

    windows = windows or BROAD_WINDOWS
    horizon = labels_mod._WINDOW
    per_ticker = _build_per_ticker(1.0)
    if not per_ticker:
        return {}
    combined = pd.concat(per_ticker.values()).sort_index()

    results = {}
    for name, (start_s, end_s) in windows.items():
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        embargo = start - pd.tseries.offsets.BDay(horizon)
        base = _window_signals(per_ticker, combined[combined.index < embargo],
                               FEATURE_COLUMNS, start, end, lookback_years, label=name)
        if base is None:
            continue

        frames = {
            ticker: pd.DataFrame({
                "Signal": frame["Signal"],
                "fwd": _forward_returns(per_ticker[ticker]["Close"], horizon)
                       .reindex(frame.index),
            })
            for ticker, frame in base.items()
        }
        results[name] = sell_gap(frames)

    return results


def summarise_sell_info(results: dict, broad: bool) -> str:
    """ASCII table plus the verdict. No verdict off the broad set."""
    out = [
        f"  {'window':<22}{'SELL days':>10}{'other':>8}{'SELL fwd':>10}"
        f"{'other fwd':>11}{'gap':>9}  SELL worse?",
        "  " + "-" * 82,
    ]
    for name, r in results.items():
        if r["gap"] is None:
            out.append(f"  {name:<22}{r['n_sell']:>10}{r['n_other']:>8}   (no gap: no ticker had both)")
            continue
        out.append(
            f"  {name:<22}{r['n_sell']:>10}{r['n_other']:>8}"
            f"{r['sell_fwd_mean'] * 100:>9.2f}%{r['other_fwd_mean'] * 100:>10.2f}%"
            f"{r['gap'] * 100:>8.2f}%  {'yes' if r['gap'] > 0 else 'no'}"
        )

    v = sell_info_verdict(results)
    mean = f"{v['mean_gap'] * 100:.3f}%" if v["mean_gap"] is not None else "n/a"
    out += [
        "",
        f"  SELL days underperform in {v['windows_sell_underperforms']} of "
        f"{v['windows_scored']} windows (need >= {v['min_windows']})",
        f"  Mean within-ticker gap: {mean} (need > {v['min_mean_gap'] * 100:.1f}%)",
    ]
    if not broad:
        out.append("  NO VERDICT: the criteria were pre-registered on --broad only.")
    elif v["windows_scored"] < len(BROAD_WINDOWS):
        # A data failure is not evidence about SELL. Reporting it as FAIL
        # would close the question on a broken run.
        out.append(f"  NO VERDICT: only {v['windows_scored']} of {len(BROAD_WINDOWS)} "
                   f"windows scored; the criteria assume all of them. Fix the data and re-run.")
    else:
        out.append(f"  STAGE 1: {'PASS' if v['passed'] else 'FAIL'}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Volatility forecast probe -- Stage 1, pre-registered 2026-09-26
# ---------------------------------------------------------------------------
#
# Findings A-M closed the direction question: nothing in this model beats
# holding. This changes the question. Volatility clusters -- a calm week
# tends to follow a calm week -- so it IS forecastable; the open question is
# whether a model adds anything over the standard formula for it.
#
# Target: realised volatility over the NEXT 5 trading days, as the root mean
# square of daily log returns t+1..t+5 (weekly, to match a weekly rebalance).
#
# Baselines the model must beat, both using only data up to day t:
#   persistence -- RMS of the last 20 daily log returns
#   EWMA        -- RiskMetrics, variance_{t+1} = 0.94 * variance_t + 0.06 * r_t^2,
#                  held flat over the 5 days. The real opponent.
#
# **DESIGN DECISION -- scored with QLIKE on variances.**
# QLIKE(h, rv2) = rv2/h - log(rv2/h) - 1. Realised volatility over 5 days is a
# noisy proxy for the true volatility; QLIKE (like MSE) still ranks forecasts
# correctly under that noise, and it penalises under-forecasting risk harder
# than over-forecasting it, which is the right bias for sizing.
#
# **DESIGN DECISION -- fixed model, no tuning.** XGBRegressor with the
# project's existing XGB_PARAMS tree settings, trained on log realised
# volatility pooled across tickers, 3 years before each window with a 5-day
# label embargo. Predictions are exp(pred) times one scale factor fitted on
# the training rows, because exp() of a log-space mean under-forecasts
# variance. The factor comes from training data only.
#
# Pass criteria agreed with the user BEFORE the first run, evaluated on
# BROAD_WINDOWS only: the model beats EWMA on mean QLIKE in >= 7 of 10 windows
# AND the mean per-window improvement over EWMA is >= 5%.

VOL_HORIZON = 5
VOL_EWMA_LAMBDA = 0.94
VOL_INFO_MIN_WINDOWS = 7
VOL_INFO_MIN_IMPROVEMENT = 0.05

VOL_FEATURES = ["rv_5", "rv_20", "rv_60", "range_vol_20", "ret_5", "ret_20", "ewma_vol"]


def _ewma_variance(log_ret: pd.Series, lam: float = VOL_EWMA_LAMBDA) -> pd.Series:
    """RiskMetrics variance forecast for day t+1, known at the close of day t.

    Seeded with the mean square of the first 20 returns. Each value uses only
    returns up to and including its own date.
    """
    r2 = (log_ret ** 2).to_numpy()
    out = np.full(len(r2), np.nan)
    valid = ~np.isnan(r2)
    idx = np.flatnonzero(valid)
    if len(idx) < 20:
        return pd.Series(out, index=log_ret.index)
    var = float(np.mean(r2[idx[:20]]))
    for i in idx[20:]:
        var = lam * var + (1.0 - lam) * r2[i]
        out[i] = var
    return pd.Series(out, index=log_ret.index)


def vol_frame(prices: pd.DataFrame, horizon: int = VOL_HORIZON) -> pd.DataFrame:
    """Volatility features, baselines and the forward target from OHLC prices.

    Every feature and baseline at row t uses data up to t; only `target_rv`
    looks forward (days t+1..t+horizon).
    """
    close = prices["Close"].astype(float)
    r = np.log(close).diff()
    r2 = r ** 2

    out = pd.DataFrame(index=prices.index)
    for n in (5, 20, 60):
        out[f"rv_{n}"] = np.sqrt(r2.rolling(n).mean())
    # Parkinson range estimator: uses each day's high-low span, which sees
    # intraday moves a close-to-close return misses.
    hl = np.log(prices["High"].astype(float) / prices["Low"].astype(float)) ** 2
    out["range_vol_20"] = np.sqrt(hl.rolling(20).mean() / (4.0 * np.log(2.0)))
    out["ret_5"] = np.log(close / close.shift(5))
    out["ret_20"] = np.log(close / close.shift(20))
    out["ewma_vol"] = np.sqrt(_ewma_variance(r))

    # sum over t+1..t+horizon of r^2, via a reversed rolling window.
    fwd_sq = r2[::-1].rolling(horizon).sum()[::-1].shift(-1)
    out["target_rv"] = np.sqrt(fwd_sq / horizon)
    return out


def qlike(forecast_var: np.ndarray, realised_var: np.ndarray) -> float:
    """Mean QLIKE loss. Lower is better; 0 is a perfect forecast."""
    ratio = realised_var / forecast_var
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def _load_prices(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROBE_DIR, f"{ticker}.csv"))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df.index = df.index.tz_localize(None).normalize()
    return df


def score_vol_window(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> dict:
    """Fit on `train`, score model and both baselines on `test` with QLIKE."""
    from xgboost import XGBRegressor

    params = {k: XGB_PARAMS[k] for k in ("n_estimators", "learning_rate", "max_depth", "subsample")
              if k in XGB_PARAMS}
    model = XGBRegressor(**params, random_state=42)
    model.fit(train[cols], np.log(train["target_rv"]))

    # One scale factor, fitted on training rows only: the QLIKE-optimal
    # multiplier for a variance forecast h is mean(realised / h).
    train_var = np.exp(2 * model.predict(train[cols]))
    scale = float(np.mean(train["target_rv"].to_numpy() ** 2 / train_var))

    realised = test["target_rv"].to_numpy() ** 2
    model_var = scale * np.exp(2 * model.predict(test[cols]))
    return {
        "rows": int(len(test)),
        "model": qlike(model_var, realised),
        "ewma": qlike(test["ewma_vol"].to_numpy() ** 2, realised),
        "persistence": qlike(test["rv_20"].to_numpy() ** 2, realised),
        "scale": scale,
    }


def vol_info_verdict(windows: dict) -> dict:
    """Apply the pre-registered Stage 1 criteria to per-window QLIKE scores."""
    scored = [w for w in windows.values() if w.get("model") is not None]
    improvements = [1.0 - w["model"] / w["ewma"] for w in scored]
    wins = sum(1 for w in scored if w["model"] < w["ewma"])
    mean_imp = float(np.mean(improvements)) if improvements else None
    passed = (
        wins >= VOL_INFO_MIN_WINDOWS
        and mean_imp is not None
        and round(mean_imp, 10) >= VOL_INFO_MIN_IMPROVEMENT
    )
    return {
        "windows_scored": len(scored),
        "windows_model_beats_ewma": wins,
        "mean_improvement_vs_ewma": mean_imp,
        "min_windows": VOL_INFO_MIN_WINDOWS,
        "min_improvement": VOL_INFO_MIN_IMPROVEMENT,
        "passed": bool(passed),
    }


def run_vol_info_probe(windows: dict | None = None, lookback_years: int = 3) -> dict:
    """Walk-forward volatility forecasts: model vs EWMA vs persistence."""
    windows = windows or BROAD_WINDOWS
    per_ticker = _build_per_ticker(1.0)
    if not per_ticker:
        return {}

    frames = []
    for ticker, feats in per_ticker.items():
        vf = vol_frame(_load_prices(ticker))
        f = feats[FEATURE_COLUMNS].join(vf, how="inner")
        f["ticker"] = ticker
        frames.append(f)
    data = pd.concat(frames).sort_index()
    cols = FEATURE_COLUMNS + VOL_FEATURES
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["target_rv"])
    data = data[data["target_rv"] > 0]

    results = {}
    for name, (start_s, end_s) in windows.items():
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        embargo = start - pd.tseries.offsets.BDay(VOL_HORIZON)
        train = data[(data.index >= start - pd.DateOffset(years=lookback_years))
                     & (data.index < embargo)]
        test = data[(data.index >= start) & (data.index <= end)]
        if len(train) < 500 or test.empty:
            print(f"  {name}: {len(train)} training rows, {len(test)} test rows, skipping")
            continue
        results[name] = score_vol_window(train, test, cols)

    return results


def summarise_vol_info(results: dict, broad: bool) -> str:
    """ASCII table plus the verdict. No verdict off the full broad set."""
    out = [
        f"  {'window':<22}{'rows':>7}{'model':>9}{'EWMA':>9}{'persist':>9}"
        f"{'vs EWMA':>10}  model wins?",
        "  " + "-" * 76,
    ]
    for name, r in results.items():
        imp = 1.0 - r["model"] / r["ewma"]
        out.append(
            f"  {name:<22}{r['rows']:>7}{r['model']:>9.4f}{r['ewma']:>9.4f}"
            f"{r['persistence']:>9.4f}{imp * 100:>9.1f}%  {'yes' if r['model'] < r['ewma'] else 'no'}"
        )

    v = vol_info_verdict(results)
    mean = (f"{v['mean_improvement_vs_ewma'] * 100:.1f}%"
            if v["mean_improvement_vs_ewma"] is not None else "n/a")
    out += [
        "",
        "  QLIKE: lower is better.",
        f"  Model beats EWMA in {v['windows_model_beats_ewma']} of {v['windows_scored']} "
        f"windows (need >= {v['min_windows']})",
        f"  Mean improvement over EWMA: {mean} (need >= {v['min_improvement'] * 100:.0f}%)",
    ]
    if not broad:
        out.append("  NO VERDICT: the criteria were pre-registered on --broad only.")
    elif v["windows_scored"] < len(BROAD_WINDOWS):
        out.append(f"  NO VERDICT: only {v['windows_scored']} of {len(BROAD_WINDOWS)} "
                   f"windows scored; the criteria assume all of them. Fix the data and re-run.")
    else:
        out.append(f"  STAGE 1: {'PASS' if v['passed'] else 'FAIL'}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Volatility-sized basket -- pre-registered 2026-09-26
# ---------------------------------------------------------------------------
#
# Finding N: a model does not forecast volatility better than EWMA. The
# question left is whether weighting the basket by EWMA volatility -- no model
# at all -- gives a smoother ride than equal-weight holding.
#
# Three arms, all fully invested in the whole watchlist:
#   hold             equal weight, bought once, never rebalanced
#   equal, weekly    equal weight, rebalanced every 5 trading days (CONTROL)
#   EWMA-sized       weights proportional to 1 / EWMA volatility, rebalanced
#                    every 5 trading days
#
# **DESIGN DECISION -- the control arm.** Weekly rebalancing changes a
# portfolio on its own (it trims winners, adds to losers, and pays slippage),
# so any difference between EWMA-sized and hold could be the rebalancing
# rather than the volatility weights. The weekly equal-weight arm separates
# the two.
#
# **DESIGN DECISION -- no lookahead.** A rebalance at day t's close uses EWMA
# volatility as of day t-1. Costs are backtest.SLIPPAGE on every unit of value
# traded; backtest.COMMISSION's flat $1 is ignored because this simulator
# works in fractions of the portfolio, not dollars.
#
# Pass criteria agreed with the user BEFORE the first run, BROAD_WINDOWS only:
#   1. mean max drawdown >= 20% smaller than hold, giving up <= 2pp mean return
#   2. smaller max drawdown than hold in >= 7 of 10 windows
#   3. smaller max drawdown than the weekly equal-weight control in >= 7 of 10

VOL_SIZING_REBALANCE_DAYS = 5
VOL_SIZING_MIN_DD_REDUCTION = 0.20
VOL_SIZING_MAX_RETURN_COST_PP = 2.0
VOL_SIZING_MIN_WINDOWS = 7

VOL_SIZING_ARMS = ("hold", "equal, weekly", "EWMA-sized")


def inverse_vol_weights(sigma: pd.Series) -> pd.Series:
    """Weights proportional to 1/sigma, summing to 1.

    A name without a usable sigma falls back to the mean inverse-sigma of the
    others rather than dropping out, so the book stays fully invested.
    """
    inv = 1.0 / sigma.where(sigma > 0)
    inv = inv.fillna(inv.mean()) if inv.notna().any() else pd.Series(1.0, index=sigma.index)
    return inv / inv.sum()


def simulate_weights(prices: pd.DataFrame, target_weights, rebalance_every: int | None,
                     slippage: float) -> pd.Series:
    """Portfolio value (starts at 1.0) for a weight-targeting policy.

    prices: dates x tickers closes. target_weights(i) -> weights for day i.
    Buys the targets at day 0's close; with rebalance_every set, trades back
    to target at the close of every rebalance_every-th day after that. Every
    trade pays `slippage` on the value traded.
    """
    px = prices.ffill().to_numpy(dtype=float)
    n_days, _ = px.shape
    w0 = np.asarray(target_weights(0), dtype=float)
    # Entry: the whole 1.0 is bought, so slippage is paid on all of it.
    holdings = (1.0 - slippage) * w0 / px[0]
    values = np.empty(n_days)
    values[0] = float(holdings @ px[0])

    for i in range(1, n_days):
        v = float(holdings @ px[i])
        if rebalance_every and i % rebalance_every == 0:
            target = np.asarray(target_weights(i), dtype=float) * v
            traded = np.abs(target - holdings * px[i]).sum()
            v -= slippage * traded
            holdings = (target / target.sum()) * v / px[i]
        values[i] = float(holdings @ px[i])
    return pd.Series(values, index=prices.index)


def max_drawdown_pct(values: pd.Series) -> float:
    """Largest peak-to-trough fall, as a positive percentage."""
    peak = values.cummax()
    return float(((peak - values) / peak).max() * 100.0)


def vol_sizing_verdict(windows: dict) -> dict:
    """Apply the three pre-registered criteria to per-window arm results."""
    rows = list(windows.values())
    n = len(rows)
    if not n:
        return {"windows_scored": 0, "passed": False}

    def mean(arm, key):
        return float(np.mean([r[arm][key] for r in rows]))

    hold_dd, ewma_dd = mean("hold", "max_dd"), mean("EWMA-sized", "max_dd")
    dd_reduction = 1.0 - ewma_dd / hold_dd if hold_dd > 0 else 0.0
    return_cost = mean("hold", "return") - mean("EWMA-sized", "return")
    beats_hold = sum(1 for r in rows if r["EWMA-sized"]["max_dd"] < r["hold"]["max_dd"])
    beats_control = sum(1 for r in rows
                        if r["EWMA-sized"]["max_dd"] < r["equal, weekly"]["max_dd"])

    c1 = (round(dd_reduction, 10) >= VOL_SIZING_MIN_DD_REDUCTION
          and round(return_cost, 10) <= VOL_SIZING_MAX_RETURN_COST_PP)
    c2 = beats_hold >= VOL_SIZING_MIN_WINDOWS
    c3 = beats_control >= VOL_SIZING_MIN_WINDOWS
    return {
        "windows_scored": n,
        "mean_max_dd": {arm: mean(arm, "max_dd") for arm in VOL_SIZING_ARMS},
        "mean_return": {arm: mean(arm, "return") for arm in VOL_SIZING_ARMS},
        "dd_reduction_vs_hold": dd_reduction,
        "return_cost_pp": return_cost,
        "windows_dd_below_hold": beats_hold,
        "windows_dd_below_control": beats_control,
        "criterion_1": bool(c1),
        "criterion_2": bool(c2),
        "criterion_3": bool(c3),
        "passed": bool(c1 and c2 and c3),
    }


def run_vol_sizing_probe(windows: dict | None = None) -> dict:
    """Simulate the three arms over every window."""
    import backtest as bt

    windows = windows or BROAD_WINDOWS
    closes, sigmas = {}, {}
    for ticker in WATCHLIST:
        try:
            prices = _load_prices(ticker)
        except Exception as exc:
            print(f"  {ticker}: FAILED {exc}")
            continue
        closes[ticker] = prices["Close"].astype(float)
        sigmas[ticker] = vol_frame(prices)["ewma_vol"]
    close_df = pd.DataFrame(closes).sort_index()
    # Lag one day: the weights traded at day t's close use sigma known at t-1.
    sigma_df = pd.DataFrame(sigmas).sort_index().shift(1)

    results = {}
    for name, (start_s, end_s) in windows.items():
        start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)
        px = close_df[(close_df.index >= start) & (close_df.index <= end)].dropna(how="all")
        if len(px) < 2 * VOL_SIZING_REBALANCE_DAYS:
            print(f"  {name}: only {len(px)} price rows, skipping")
            continue
        sig = sigma_df.reindex(px.index)
        k = px.shape[1]
        equal = lambda i: np.full(k, 1.0 / k)
        ewma = lambda i: inverse_vol_weights(sig.iloc[i]).to_numpy()

        arms = {
            "hold": simulate_weights(px, equal, None, bt.SLIPPAGE),
            "equal, weekly": simulate_weights(px, equal, VOL_SIZING_REBALANCE_DAYS, bt.SLIPPAGE),
            "EWMA-sized": simulate_weights(px, ewma, VOL_SIZING_REBALANCE_DAYS, bt.SLIPPAGE),
        }
        results[name] = {
            arm: {"return": float((v.iloc[-1] - 1.0) * 100.0), "max_dd": max_drawdown_pct(v)}
            for arm, v in arms.items()
        }
    return results


def summarise_vol_sizing(results: dict, broad: bool) -> str:
    """ASCII table plus the three criteria. No verdict off the full broad set."""
    head = "".join(f"{a:>24}" for a in VOL_SIZING_ARMS)
    out = [f"  {'window':<22}{head}", f"  {'':<22}" + "".join(
        f"{'return':>12}{'max DD':>12}" for _ in VOL_SIZING_ARMS), "  " + "-" * 94]
    for name, r in results.items():
        cells = "".join(f"{r[a]['return']:>11.2f}%{r[a]['max_dd']:>11.2f}%" for a in VOL_SIZING_ARMS)
        out.append(f"  {name:<22}{cells}")

    v = vol_sizing_verdict(results)
    if v["windows_scored"]:
        out += [
            "",
            f"  1. Mean max DD {v['mean_max_dd']['EWMA-sized']:.2f}% vs hold "
            f"{v['mean_max_dd']['hold']:.2f}%: {v['dd_reduction_vs_hold'] * 100:.1f}% smaller "
            f"(need >= {VOL_SIZING_MIN_DD_REDUCTION * 100:.0f}%); return given up "
            f"{v['return_cost_pp']:+.2f}pp (need <= {VOL_SIZING_MAX_RETURN_COST_PP:.0f}pp)"
            f"  -> {'pass' if v['criterion_1'] else 'fail'}",
            f"  2. Smaller max DD than hold in {v['windows_dd_below_hold']} of "
            f"{v['windows_scored']} (need >= {VOL_SIZING_MIN_WINDOWS})"
            f"  -> {'pass' if v['criterion_2'] else 'fail'}",
            f"  3. Smaller max DD than weekly equal-weight in {v['windows_dd_below_control']} of "
            f"{v['windows_scored']} (need >= {VOL_SIZING_MIN_WINDOWS})"
            f"  -> {'pass' if v['criterion_3'] else 'fail'}",
        ]
    if not broad:
        out.append("  NO VERDICT: the criteria were pre-registered on --broad only.")
    elif v["windows_scored"] < len(BROAD_WINDOWS):
        out.append(f"  NO VERDICT: only {v['windows_scored']} of {len(BROAD_WINDOWS)} "
                   f"windows scored; the criteria assume all of them. Fix the data and re-run.")
    else:
        out.append(f"  VOLATILITY SIZING: {'PASS' if v['passed'] else 'FAIL'}")
    return "\n".join(out)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the model against trivial baselines and sweep training history.",
    )
    parser.add_argument(
        "--test-start", default="2026-05-20", metavar="YYYY-MM-DD",
        help="First date of the fixed test window (default: 2026-05-20, the "
             "window the promotion gate used).",
    )
    parser.add_argument(
        "--lookbacks", type=int, nargs="+", default=[1, 2, 3, 5, 8], metavar="YEARS",
        help="Training lookbacks to sweep (default: 1 2 3 5 8).",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-download the probe history even if it is already cached.",
    )
    parser.add_argument(
        "--regimes", action="store_true",
        help="Walk-forward RETURN benchmark vs buy-and-hold across bull and bear "
             "windows. This is the measurement that decides whether the strategy "
             "is worth running; classification metrics are a proxy for it.",
    )
    parser.add_argument(
        "--exposure", action="store_true",
        help="Compare allocation policies over the same regime windows: "
             "confidence-scaled sizing (shipped) vs full-size buys vs a 200-day "
             "SMA regime filter with and without the model. Separates an "
             "exposure deficit from a selection deficit.",
    )
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=None, metavar="DAYS",
        help="Re-label at each forward-return horizon and re-run the "
             "walk-forward return benchmark. The 7-day window in labels.py was "
             "never tested. Example: --horizons 3 5 7 14 21",
    )
    parser.add_argument(
        "--exits", action="store_true",
        help="Compare how a position is closed: on any non-BUY (the old "
             "simulator), on SELL only (what the bot does), on SELL or a stop, "
             "at the label horizon, or all three.",
    )
    parser.add_argument(
        "--vol-sizing", action="store_true",
        help="Equal-weight hold vs weekly equal-weight vs EWMA inverse-vol "
             "weights on drawdown. Pre-registered criteria; verdict only "
             "with --broad.",
    )
    parser.add_argument(
        "--vol-info", action="store_true",
        help="Volatility Stage 1: does a model forecast next-5-day realised "
             "volatility better than EWMA (QLIKE)? Pre-registered pass "
             "criteria; verdict only with --broad.",
    )
    parser.add_argument(
        "--sell-info", action="store_true",
        help="Stage 1: do the model's SELL days precede worse forward returns "
             "than its other days, within each ticker? Pre-registered pass "
             "criteria; verdict only with --broad.",
    )
    parser.add_argument(
        "--tiers", action="store_true",
        help="Sweep the confidence-tier position sizes and report return, "
             "drawdown and realised exposure per level. Asks whether the "
             "relationship is monotone, not which level wins.",
    )
    parser.add_argument(
        "--window", nargs=2, default=None, metavar=("START", "END"),
        help="Run over one explicit window instead of a preset set. Use this to "
             "simulate the exact dates the live account traded, which is the "
             "only apples-to-apples comparison against live_benchmark.py.",
    )
    parser.add_argument(
        "--broad", action="store_true",
        help="Use BROAD_WINDOWS (10 continuous windows) instead of the four "
             "regime windows. REGIME_WINDOWS is half drawdowns by design, which "
             "flatters any strategy that holds less stock; this is the control.",
    )
    parser.add_argument(
        "--features", action="store_true",
        help="Instead of the lookback sweep, rank features by mean |SHAP| and "
             "retrain on only the top K to see how many actually carry signal.",
    )
    parser.add_argument(
        "--label-scales", type=float, nargs="+", default=None, metavar="SCALE",
        help="Instead of the lookback sweep, multiply the VIX label thresholds "
             "by each scale and re-measure. Widening the HOLD band should raise "
             "the baseline's difficulty; the question is whether the model's "
             "edge over that baseline grows. Example: --label-scales 1 1.5 2 3",
    )
    return parser.parse_args(argv)


def run_label_sweep(scales: list[float], test_start: pd.Timestamp,
                    lookback_years: int = 3) -> dict:
    """Re-label at each threshold scale and measure model vs baselines.

    The number that matters is the *edge* -- model minus always-majority -- not
    the model's raw accuracy. A wider HOLD band makes HOLD the majority class
    and mechanically changes both numbers; only the gap between them says
    whether the labels got more learnable.
    """
    results = {}

    for scale in scales:
        X, y = build_probe_dataset(threshold_scale=scale)
        test_mask = X.index >= test_start
        X_test, y_test = X[test_mask], y[test_mask]

        train_start = test_start - pd.DateOffset(years=lookback_years)
        train_mask = (X.index >= train_start) & (X.index < test_start)
        X_train, y_train = X[train_mask], y[train_mask]

        if len(X_train) < 500 or X_test.empty:
            print(f"  scale {scale}: insufficient rows, skipping")
            continue

        encoder = LabelEncoder()
        model = XGBClassifier(**XGB_PARAMS, random_state=42)
        model.fit(X_train, encoder.fit_transform(y_train))
        pred = encoder.inverse_transform(model.predict(X_test))

        model_m = score(y_test, pred)
        baselines = compute_baselines(y_test)
        majority_key = next(k for k in baselines if k.startswith("always_")
                            and k != "always_HOLD") if any(
                            k.startswith("always_") and k != "always_HOLD"
                            for k in baselines) else "always_HOLD"
        majority = baselines[majority_key]

        dist = y_test.value_counts()
        hold_pct = 100.0 * dist.get("HOLD", 0) / len(y_test)

        results[f"scale_{scale}"] = {
            "scale": scale,
            "hold_pct": hold_pct,
            "distribution": {k: int(v) for k, v in dist.items()},
            "model": model_m,
            "majority_baseline": majority_key,
            "baseline": majority,
            "accuracy_edge": model_m["accuracy"] - majority["accuracy"],
            "macro_f1_edge": model_m["macro_f1"] - baselines["stratified_random"]["macro_f1"],
        }

        print(f"  scale {scale:<5} HOLD={hold_pct:>5.1f}%  "
              f"model_acc={model_m['accuracy']:.4f}  "
              f"{majority_key}={majority['accuracy']:.4f}  "
              f"edge={results[f'scale_{scale}']['accuracy_edge']:>+.4f}  "
              f"macro_f1_edge={results[f'scale_{scale}']['macro_f1_edge']:>+.4f}")

    return results


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        test_start = pd.Timestamp(args.test_start)
    except ValueError:
        print(f"[FATAL] --test-start must be YYYY-MM-DD, got {args.test_start!r}")
        return 1

    print(f"=== Downloading {PROBE_HISTORY} history into {PROBE_DIR} ===")
    download_history(refresh=args.refresh)

    if args.window:
        window_set = {f"{args.window[0]} to {args.window[1]}": tuple(args.window)}
        window_label = f"explicit ({args.window[0]} to {args.window[1]})"
    elif args.broad:
        window_set, window_label = BROAD_WINDOWS, "broad (10 windows)"
    else:
        window_set, window_label = REGIME_WINDOWS, "regime (4 windows)"

    if args.regimes:
        print(f"\n=== Regime benchmark: strategy vs buy-and-hold "
              f"(walk-forward, {window_label}) ===")
        configs = {
            "shipped (20 feat, 1.0x labels)": (FEATURE_COLUMNS, 1.0),
            "top-5 feat + 1.5x labels":       (TOP5_FEATURES, 1.5),
        }
        results = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "regimes",
            "window_set": window_label,
            "configs": {},
        }
        for label, (cols, scale) in configs.items():
            print(f"\n  {label}")
            print(f"  {'window':<20}{'strategy':>10}{'hold':>11}{'delta':>11}")
            results["configs"][label] = run_regime_benchmark(
                cols, scale, windows=window_set)

        print()
        for label, res in results["configs"].items():
            if not res:
                continue
            deltas = [r["delta_pp"] for r in res.values()]
            wins = sum(1 for d in deltas if d > 0)
            print(f"  {label:<34} beat hold in {wins}/{len(deltas)} windows, "
                  f"mean delta {np.mean(deltas):+.2f}pp")
        print()
        print("  A strategy with real edge beats holding across regimes, not in one.")
    elif args.exposure:
        print("\n=== Exposure and regime-filter probe (walk-forward) ===")
        print("  All arms share one model per window and one simulator;")
        print("  only the allocation policy differs.")
        results = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "exposure",
            "config": "shipped (20 feat, 1.0x labels)",
            "sma_window": REGIME_SMA_WINDOW,
            "window_set": window_label,
            "windows": run_exposure_probe(FEATURE_COLUMNS, 1.0,
                                          windows=window_set),
        }
        print(summarise_exposure(results["windows"]))
    elif args.exits:
        print(f"\n=== Exit-policy probe ({window_label}) ===")
        print("  One model per window, shared by every policy; only the rule")
        print("  for closing a held position differs.")
        results = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "exits",
            "window_set": window_label,
            "policies": run_exit_probe(windows=window_set),
        }
        print(summarise_exits(results["policies"]))
    elif args.vol_sizing:
        print(f"\n=== Volatility-sized basket ({window_label}) ===")
        print("  Hold vs weekly equal-weight (control) vs EWMA inverse-vol weights,")
        print("  fully invested, rebalanced every 5 trading days, sigma lagged a day.")
        results = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "vol_sizing",
            "window_set": window_label,
            "windows": run_vol_sizing_probe(windows=window_set),
        }
        results["verdict"] = vol_sizing_verdict(results["windows"])
        results["verdict"]["complete"] = (
            window_set is BROAD_WINDOWS
            and results["verdict"]["windows_scored"] == len(BROAD_WINDOWS)
        )
        print(summarise_vol_sizing(results["windows"], broad=window_set is BROAD_WINDOWS))
    elif args.vol_info:
        print(f"\n=== Volatility forecast probe, Stage 1 ({window_label}) ===")
        print("  Next-5-day realised volatility: model vs EWMA vs persistence,")
        print("  one model per window with a label embargo, scored with QLIKE.")
        results = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "vol_info",
            "window_set": window_label,
            "windows": run_vol_info_probe(windows=window_set),
        }
        results["verdict"] = vol_info_verdict(results["windows"])
        results["verdict"]["complete"] = (
            window_set is BROAD_WINDOWS
            and results["verdict"]["windows_scored"] == len(BROAD_WINDOWS)
        )
        print(summarise_vol_info(results["windows"], broad=window_set is BROAD_WINDOWS))
    elif args.sell_info:
        print(f"\n=== SELL information probe, Stage 1 ({window_label}) ===")
        print("  One model per window, trained with a label embargo; forward")
        print("  returns over the label horizon, compared within each ticker.")
        results = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "sell_info",
            "window_set": window_label,
            "windows": run_sell_info_probe(windows=window_set),
        }
        results["verdict"] = sell_info_verdict(results["windows"])
        results["verdict"]["complete"] = (
            window_set is BROAD_WINDOWS
            and results["verdict"]["windows_scored"] == len(BROAD_WINDOWS)
        )
        print(summarise_sell_info(results["windows"], broad=window_set is BROAD_WINDOWS))
    elif args.tiers:
        print(f"\n=== Position-tier sweep ({window_label}) ===")
        print("  One model per window, shared by every level; only the tier")
        print("  position sizes differ.")
        results = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "tiers",
            "window_set": window_label,
            "levels": run_tier_sweep(windows=window_set),
        }
        print(summarise_tiers(results["levels"]))
    elif args.horizons:
        print(f"\n=== Label horizon probe (walk-forward returns, "
              f"{window_label}) ===")
        print("  Scored on returns, not accuracy - finding E showed those "
              "disagree.")
        results = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "horizons",
            "window_set": window_label,
            "horizons": run_horizon_probe(args.horizons, windows=window_set),
        }
        print(summarise_horizons(results["horizons"]))
    elif args.features:
        print("\n=== Feature probe (SHAP ranking + top-K retrain) ===")
        results = run_feature_probe(test_start)
        k_results = results["top_k"]
        if k_results:
            best_key = max(k_results, key=lambda k: k_results[k]["accuracy"])
            best = k_results[best_key]
            full = k_results.get(f"top_{len(FEATURE_COLUMNS)}")
            print(f"\n  Best: {best_key} — acc={best['accuracy']:.4f} on "
                  f"{len(best['features'])} features")
            if full and best["accuracy"] >= full["accuracy"] and best_key != f"top_{len(FEATURE_COLUMNS)}":
                print(f"  -> A subset matches or beats all {len(FEATURE_COLUMNS)} "
                      f"features. The rest add variance, not information.")
    elif args.label_scales:
        print("\n=== Label threshold sweep (HOLD band width) ===")
        results = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "label_scales",
            "test_start": str(test_start.date()),
            "scales": run_label_sweep(args.label_scales, test_start),
        }
        scales = results["scales"]
        if scales:
            best = max(scales.values(), key=lambda r: r["accuracy_edge"])
            print(f"\n  Widest edge over the majority baseline: "
                  f"scale {best['scale']} ({best['accuracy_edge']:+.4f}) "
                  f"at HOLD={best['hold_pct']:.1f}%")
            if best["accuracy_edge"] <= 0.02:
                print("  -> No label width produces a meaningful edge. The labels are")
                print("     not the binding constraint; the features are.")
    else:
        print("\n=== Building dataset ===")
        X, y = build_probe_dataset()
        print(f"\nTotal: {len(X)} rows, {X.index.min().date()} -> {X.index.max().date()}")
        results = run_sweep(X, y, test_start, args.lookbacks)
        print(format_verdict(results))

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
