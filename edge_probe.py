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
    python edge_probe.py                          # default sweep
    python edge_probe.py --test-start 2026-05-20
    python edge_probe.py --lookbacks 1 3 5 10
    python edge_probe.py --refresh                # re-download the history
"""

import argparse
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

from config import WATCHLIST, FEATURE_COLUMNS, XGB_PARAMS

_HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_DIR = os.path.join(_HERE, "data", "edge_probe")
PROBE_MARKET_DIR = os.path.join(PROBE_DIR, "market")
RESULTS_PATH = os.path.join(_HERE, "data", "edge_probe_results.json")

PROBE_HISTORY = "10y"
MARKET_SYMBOLS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "^VIX"]
LABELS = ["BUY", "HOLD", "SELL"]


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


def build_probe_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Featured + labelled rows for every ticker, read from the probe directory."""
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
            frames.append(df)
            print(f"  {ticker}: {len(df)} rows  "
                  f"{df.index.min().date()} -> {df.index.max().date()}")
        except Exception as exc:
            print(f"  {ticker}: FAILED {exc}")

    if not frames:
        raise RuntimeError("No ticker data loaded.")

    combined = pd.concat(frames).sort_index()
    return combined[FEATURE_COLUMNS], combined["Signal"]


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        test_start = pd.Timestamp(args.test_start)
    except ValueError:
        print(f"[FATAL] --test-start must be YYYY-MM-DD, got {args.test_start!r}")
        return 1

    print(f"=== Downloading {PROBE_HISTORY} history into {PROBE_DIR} ===")
    download_history(refresh=args.refresh)

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
