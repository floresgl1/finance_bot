"""
Trains and evaluates the ML classification model.

Splits data into three chronological sets (70% train / 20% validation /
10% test), tunes the confidence threshold on validation, then reports
final results on the held-out test set.

XGBoost requires integer class labels, so a LabelEncoder is fitted on the
training labels and saved alongside the model in a single bundle:
    {"model": XGBClassifier, "encoder": LabelEncoder}
predictor.py uses the encoder to decode predictions back to BUY/SELL/HOLD.

Usage:
    python trainer.py                       # interactive: prompts before
                                            # touching config.py, overwrites
                                            # the live model (with a backup)

    python trainer.py --headless \
        --output models/XG_Boost_candidate.joblib
                                            # unattended: never prompts, never
                                            # writes config.py, and leaves the
                                            # live model untouched

**DESIGN DECISION:**
`--output` exists so a challenger can be trained without displacing the live
champion. Before it, training always overwrote the production model and the
comparison happened afterwards — a worse model was already trading by the time
anyone looked. promote_model.py depends on being able to build a candidate
off to the side and decide separately.
"""

import argparse
import os
import re
import shutil
import sys
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from config import WATCHLIST, MODEL_DIR, MODEL_FILENAME, CONFIDENCE_THRESHOLD, FEATURE_COLUMNS, XGB_PARAMS
from features import load_and_process
from labels import add_labels


def apply_confidence_threshold(
    model: XGBClassifier,
    X: pd.DataFrame,
    threshold: float,
    encoder: LabelEncoder,
) -> np.ndarray:
    """
    Return decoded string predictions (BUY/SELL/HOLD) where BUY/SELL require
    the top class probability to exceed `threshold`; otherwise overridden to HOLD.
    """
    proba    = model.predict_proba(X)
    top_idx  = np.argmax(proba, axis=1)
    top_prob = proba[np.arange(len(proba)), top_idx]
    preds    = encoder.inverse_transform(top_idx).copy()
    preds[top_prob < threshold] = "HOLD"
    return preds


def build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load, process, and label data for every ticker, then combine into one
    date-sorted dataset.

    Sorting by date is critical for the chronological 70/20/10 split: without
    it, the split falls in the middle of an arbitrary ticker rather than at a
    real time boundary.

    Returns:
        X: DataFrame of feature columns.
        y: Series of Signal labels (BUY / SELL / HOLD).
    """
    frames = []

    for ticker in WATCHLIST:
        try:
            df = load_and_process(ticker)
            df = add_labels(df)
            frames.append(df)
            print(f"[OK]    {ticker} — {len(df)} rows loaded")
        except FileNotFoundError:
            print(f"[SKIP]  {ticker} — CSV not found, run data_collector.py first")

    if not frames:
        raise RuntimeError("No data loaded — run data_collector.py first.")

    combined = pd.concat(frames).sort_index()   # sort all rows by date
    X = combined[FEATURE_COLUMNS]
    y = combined["Signal"]
    return X, y


def tune_threshold(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    encoder: LabelEncoder,
) -> float:
    """
    Search thresholds from 0.35 to 0.60 and return the one that maximises
    BUY F1-score on the validation set.

    BUY F1 is the target because it is the most actionable signal: a missed
    BUY costs opportunity, while a false BUY costs real money.
    """
    best_threshold, best_buy_f1 = CONFIDENCE_THRESHOLD, 0.0

    print(f"\n  {'Threshold':>10}  {'BUY F1':>8}  {'HOLD F1':>8}  {'SELL F1':>8}")
    print(f"  {'-'*44}")

    for t in np.arange(0.35, 0.61, 0.05):
        t = round(float(t), 2)
        y_pred  = apply_confidence_threshold(model, X_val, t, encoder)
        report  = classification_report(
            y_val, y_pred,
            labels=["BUY", "HOLD", "SELL"],
            target_names=["BUY", "HOLD", "SELL"],
            output_dict=True,
            zero_division=0,
        )
        buy_f1  = report["BUY"]["f1-score"]
        hold_f1 = report["HOLD"]["f1-score"]
        sell_f1 = report["SELL"]["f1-score"]
        marker  = "  <-- best" if buy_f1 > best_buy_f1 else ""

        print(f"  {t:>10.2f}  {buy_f1:>8.3f}  {hold_f1:>8.3f}  {sell_f1:>8.3f}{marker}")

        if buy_f1 > best_buy_f1:
            best_buy_f1 = buy_f1
            best_threshold = t

    return best_threshold


def update_config_threshold(new_threshold: float) -> None:
    """Overwrite the CONFIDENCE_THRESHOLD value in config.py in place."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    with open(config_path, "r") as f:
        content = f.read()

    content = re.sub(
        r"^(CONFIDENCE_THRESHOLD\s*=\s*)[\d.]+",
        rf"\g<1>{new_threshold}",
        content,
        flags=re.MULTILINE,
    )

    with open(config_path, "w") as f:
        f.write(content)


def _print_confusion(cm: np.ndarray, labels: list[str]) -> None:
    """Print a confusion matrix with labelled rows and columns."""
    header = " " * 8 + "".join(f"pred {l:<5}" for l in labels)
    print(header)
    for actual_label, row in zip(labels, cm):
        cells = "".join(f"{n:<10}" for n in row)
        print(f"act {actual_label:<4}  {cells}")


def train(
    *,
    output_path: str | None = None,
    headless: bool = False,
    update_config: bool | None = None,
    threshold: float | None = None,
) -> dict:
    """Build the dataset, train XGBoost with a LabelEncoder, and save the bundle.

    Parameters
    ----------
    output_path
        Where to write the model bundle. None writes to the live model path
        (MODEL_DIR/MODEL_FILENAME) and backs the incumbent up first. Any other
        value writes there and leaves the live model completely untouched — no
        backup is taken because nothing was displaced.
    headless
        Never read from stdin. Required for CI and scheduled runs.
    update_config
        True writes the tuned threshold to config.py, False skips it, None asks
        interactively. In headless mode None means skip: rewriting the source of
        truth for the live bot is not a safe unattended default.
    threshold
        Skip validation tuning and use this value instead.

    Returns
    -------
    dict with the tuned threshold, the path written, split sizes, and the
    held-out test metrics — the inputs promote_model.py needs to decide.
    """
    if headless and update_config is None:
        update_config = False

    print("=== Loading data ===")
    X, y = build_dataset()

    n         = len(X)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.90)

    X_train, y_train = X.iloc[:train_end],        y.iloc[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:],          y.iloc[val_end:]

    print(f"\nTotal samples : {n}")
    print(f"Train         : {len(X_train)} rows  ({X_train.index.min().date()} to {X_train.index.max().date()})")
    print(f"Validation    : {len(X_val)} rows  ({X_val.index.min().date()} to {X_val.index.max().date()})")
    print(f"Test          : {len(X_test)} rows  ({X_test.index.min().date()} to {X_test.index.max().date()})")
    print(f"\nClass distribution (train):\n{y_train.value_counts().to_string()}\n")

    # --- Encode string labels to integers for XGBoost ---
    # LabelEncoder sorts alphabetically: BUY→0, HOLD→1, SELL→2
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    print(f"Label encoding: { {cls: i for i, cls in enumerate(le.classes_)} }")

    # --- Train XGBoost with hyperparameters from config ---
    print("\n=== Training XGBClassifier ===")
    model = XGBClassifier(**XGB_PARAMS, random_state=42)
    model.fit(X_train, y_train_enc)
    print("  Training complete.")

    # --- Confidence threshold: tuned on validation, or supplied ---
    if threshold is not None:
        best_threshold = threshold
        print(f"\n=== Using supplied confidence threshold: {best_threshold} ===")
    else:
        print("\n=== Tuning confidence threshold on validation set ===")
        best_threshold = tune_threshold(model, X_val, y_val, le)
        print(f"\n  Best threshold: {best_threshold}  (maximises BUY F1 on validation)")

    # Persist the best threshold back to config.py so all modules use it.
    # config.py is read by the live bot, so an unattended run never edits it
    # without being told to explicitly.
    if update_config is None:
        answer = input(
            f"\n  Update config.py with CONFIDENCE_THRESHOLD = {best_threshold}? [y/N] "
        ).strip().lower()
        update_config = answer == "y"

    if update_config:
        update_config_threshold(best_threshold)
        print(f"  config.py updated: CONFIDENCE_THRESHOLD = {best_threshold}")
    else:
        print("  config.py not updated — keeping existing threshold.")

    # --- Final evaluation on the untouched test set ---
    labels = ["BUY", "HOLD", "SELL"]
    print("\n=== Evaluation on held-out TEST set (out-of-sample) ===")

    y_pred_raw    = le.inverse_transform(model.predict(X_test))
    print("Without threshold:")
    print(classification_report(y_test, y_pred_raw, target_names=labels, zero_division=0))
    _print_confusion(confusion_matrix(y_test, y_pred_raw, labels=labels), labels)

    y_pred_thresh = apply_confidence_threshold(model, X_test, best_threshold, le)
    n_forced      = int(np.sum((y_pred_thresh == "HOLD") & (y_pred_raw != "HOLD")))
    print(f"\nWith threshold = {best_threshold}  ({n_forced} predictions forced to HOLD):")
    print(classification_report(y_test, y_pred_thresh, target_names=labels, zero_division=0))
    _print_confusion(confusion_matrix(y_test, y_pred_thresh, labels=labels), labels)

    test_report = classification_report(
        y_test, y_pred_thresh,
        labels=labels, target_names=labels,
        output_dict=True, zero_division=0,
    )

    # --- Save model bundle ---
    os.makedirs(MODEL_DIR, exist_ok=True)

    if output_path is None:
        # Live path: back the incumbent up before overwriting it.
        model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        if os.path.exists(model_path):
            backup_path = os.path.join(MODEL_DIR, "XG_Boost_backup.joblib")
            shutil.copy2(model_path, backup_path)
            print(f"\nExisting model backed up to {backup_path}")
    else:
        # Candidate path: nothing is displaced, so nothing is backed up.
        model_path = output_path
        parent = os.path.dirname(os.path.abspath(model_path))
        os.makedirs(parent, exist_ok=True)
        print(f"\nWriting candidate model — live model left untouched.")

    joblib.dump({"model": model, "encoder": le}, model_path)
    print(f"Model bundle saved to {model_path}")

    return {
        "threshold":   best_threshold,
        "model_path":  model_path,
        "n_train":     len(X_train),
        "n_val":       len(X_val),
        "n_test":      len(X_test),
        "n_forced":    n_forced,
        "test_report": test_report,
        "config_updated": bool(update_config),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the XGBoost signal model.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Never prompt. Implies --no-update-config unless --update-config is given.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help=(
            "Write the model bundle here instead of the live model path. "
            "The live model is left untouched."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Skip validation tuning and use this confidence threshold.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--update-config",
        dest="update_config",
        action="store_true",
        default=None,
        help="Write the tuned CONFIDENCE_THRESHOLD back to config.py.",
    )
    group.add_argument(
        "--no-update-config",
        dest="update_config",
        action="store_false",
        help="Leave config.py alone.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.threshold is not None and not (0.0 < args.threshold < 1.0):
        print(f"[FATAL] --threshold must be between 0 and 1, got {args.threshold}")
        return 1

    train(
        output_path=args.output,
        headless=args.headless,
        update_config=args.update_config,
        threshold=args.threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
