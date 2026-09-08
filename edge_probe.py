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

from config import WATCHLIST, FEATURE_COLUMNS, XGB_PARAMS, CONFIDENCE_THRESHOLD

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


def _relabel(df: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Re-derive Signal from a scaled threshold.

    add_labels() leaves `threshold` and the inputs on the frame, so the
    relative return it compared against can be recovered rather than
    recomputed -- which keeps this consistent with labels.py by construction
    instead of by a duplicated formula that could drift.
    """
    df = df.copy()
    stock_return_7d = df["Close"].pct_change(7).shift(-7)
    relative = stock_return_7d - df["spy_return_7d"]
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


def run_regime_benchmark(cols: list[str], scale: float,
                         windows: dict | None = None,
                         lookback_years: int = 3) -> dict:
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
EXPOSURE_ARMS = {
    "shipped":             ("shipped",           0.20, 0.80),
    "full-size buys":      ("full_size",         0.20, 0.80),
    "full-size, 100% cap": ("full_size",         0.25, 1.00),
    "regime filter only":  ("regime_only",       _EQ_POSITION_PCT, 1.00),
    "regime + model":      ("regime_plus_model", _EQ_POSITION_PCT, 1.00),
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
    full = means.get("full-size, 100% cap")
    if shipped is not None and full is not None:
        if full > shipped:
            lines.append(f"  Sizing up helps ({shipped:+.2f}pp -> {full:+.2f}pp): part of")
            lines.append("  the deficit was exposure, not selection.")
        else:
            lines.append(f"  Sizing up makes it WORSE ({shipped:+.2f}pp -> {full:+.2f}pp).")
            lines.append("  The deficit is selection, not exposure -- more of these picks")
            lines.append("  is more of the problem.")

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


def _build_per_ticker(scale: float) -> dict:
    """Per-ticker featured+labelled frames from the probe data."""
    import features
    import labels as labels_mod

    features.DATA_DIR = PROBE_DIR
    features.MARKET_DATA_DIR = PROBE_MARKET_DIR
    features._load_market_close.cache_clear()
    labels_mod._SPY_PATH = os.path.join(PROBE_MARKET_DIR, "SPY.csv")
    labels_mod._VIX_PATH = os.path.join(PROBE_MARKET_DIR, "^VIX.csv")

    out = {}
    for ticker in WATCHLIST:
        try:
            df = labels_mod.add_labels(features.load_and_process(ticker))
            out[ticker] = _relabel(df, scale) if scale != 1.0 else df
        except Exception as exc:
            print(f"  {ticker}: FAILED {exc}")
    return out


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

    if args.regimes:
        print("\n=== Regime benchmark: strategy vs buy-and-hold (walk-forward) ===")
        configs = {
            "shipped (20 feat, 1.0x labels)": (FEATURE_COLUMNS, 1.0),
            "top-5 feat + 1.5x labels":       (TOP5_FEATURES, 1.5),
        }
        results = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "regimes",
            "configs": {},
        }
        for label, (cols, scale) in configs.items():
            print(f"\n  {label}")
            print(f"  {'window':<20}{'strategy':>10}{'hold':>11}{'delta':>11}")
            results["configs"][label] = run_regime_benchmark(cols, scale)

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
            "windows": run_exposure_probe(FEATURE_COLUMNS, 1.0),
        }
        print(summarise_exposure(results["windows"]))
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
