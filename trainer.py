"""
Trains and evaluates the ML classification model.

Splits data into three chronological sets (70% train / 20% validation /
10% test), tunes the confidence threshold on validation, then reports
final results on the held-out test set.

XGBoost requires integer class labels, so a LabelEncoder is fitted on the
training labels and saved alongside the model in a single bundle:
    {"model": XGBClassifier, "encoder": LabelEncoder}
predictor.py uses the encoder to decode predictions back to BUY/SELL/HOLD.
"""

import os
import re
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


def train() -> None:
    """Build the dataset, train XGBoost with a LabelEncoder, and save the bundle."""
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

    # --- Tune confidence threshold on validation set ---
    print("\n=== Tuning confidence threshold on validation set ===")
    best_threshold = tune_threshold(model, X_val, y_val, le)
    print(f"\n  Best threshold: {best_threshold}  (maximises BUY F1 on validation)")

    # Persist the best threshold back to config.py so all modules use it
    answer = input(f"\n  Update config.py with CONFIDENCE_THRESHOLD = {best_threshold}? [y/N] ").strip().lower()
    if answer == "y":
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

    # --- Save model bundle ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

    # Back up existing model before overwriting
    if os.path.exists(model_path):
        backup_path = os.path.join(MODEL_DIR, "XG_Boost_backup.joblib")
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f"\nExisting model backed up to {backup_path}")

    joblib.dump({"model": model, "encoder": le}, model_path)
    print(f"Model bundle saved to {model_path}")


if __name__ == "__main__":
    train()
