"""
Compares test-set metrics between the current model and the backup model.

Loads both model bundles from models/, builds the same dataset and
chronological 70/20/10 split as trainer.py, runs predictions with each model,
and prints a side-by-side metric summary.

Usage:
    python compare_models.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from config import MODEL_DIR, CONFIDENCE_THRESHOLD, FEATURE_COLUMNS
from trainer import build_dataset, apply_confidence_threshold, _print_confusion


CURRENT_MODEL_FILE = "XG_Boost.joblib"
BACKUP_MODEL_FILE  = "XG_Boost_backup.joblib"
LABELS             = ["BUY", "HOLD", "SELL"]


def load_bundle(filename: str) -> dict:
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def evaluate(bundle: dict, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict:
    """
    Run predictions with and without the confidence threshold.
    Returns a dict of classification_report dicts (output_dict=True).
    """
    model   = bundle["model"]
    encoder = bundle["encoder"]

    y_pred_raw    = encoder.inverse_transform(model.predict(X_test))
    y_pred_thresh = apply_confidence_threshold(model, X_test, threshold, encoder)
    n_forced      = int(np.sum((y_pred_thresh == "HOLD") & (y_pred_raw != "HOLD")))

    report_raw    = classification_report(
        y_test, y_pred_raw,
        labels=LABELS, target_names=LABELS,
        output_dict=True, zero_division=0,
    )
    report_thresh = classification_report(
        y_test, y_pred_thresh,
        labels=LABELS, target_names=LABELS,
        output_dict=True, zero_division=0,
    )
    cm_raw    = confusion_matrix(y_test, y_pred_raw,    labels=LABELS)
    cm_thresh = confusion_matrix(y_test, y_pred_thresh, labels=LABELS)

    return {
        "report_raw":    report_raw,
        "report_thresh": report_thresh,
        "cm_raw":        cm_raw,
        "cm_thresh":     cm_thresh,
        "y_pred_raw":    y_pred_raw,
        "y_pred_thresh": y_pred_thresh,
        "n_forced":      n_forced,
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def print_side_by_side(
    current_results: dict,
    backup_results:  dict,
    threshold: float,
) -> None:
    """Print a side-by-side comparison of per-class and overall metrics."""

    def section(title: str, cur_report: dict, bak_report: dict) -> None:
        col_w = 10
        print(f"\n  {title}")
        print(f"  {'-' * 72}")
        header = (
            f"  {'':12}"
            f"{'Precision':>{col_w}}  {'Recall':>{col_w}}  {'F1':>{col_w}}  {'Support':>{col_w}}"
            f"  |  "
            f"{'Precision':>{col_w}}  {'Recall':>{col_w}}  {'F1':>{col_w}}  {'Support':>{col_w}}"
        )
        print(header)
        print(f"  {'':12}{'--- CURRENT ---':>{col_w * 4 + 6}}  |  {'--- BACKUP ---':>{col_w * 4 + 6}}")
        print(f"  {'-' * 72}")

        for lbl in LABELS + ["macro avg", "weighted avg"]:
            c = cur_report.get(lbl, {})
            b = bak_report.get(lbl, {})

            def row(r):
                if not r:
                    return f"{'N/A':>{col_w}}  {'N/A':>{col_w}}  {'N/A':>{col_w}}  {'N/A':>{col_w}}"
                sup = int(r.get("support", 0))
                return (
                    f"{_fmt(r['precision']):>{col_w}}"
                    f"  {_fmt(r['recall']):>{col_w}}"
                    f"  {_fmt(r['f1-score']):>{col_w}}"
                    f"  {sup:>{col_w}}"
                )

            print(f"  {lbl:<12}{row(c)}  |  {row(b)}")

        # Accuracy
        c_acc = cur_report.get("accuracy", float("nan"))
        b_acc = bak_report.get("accuracy", float("nan"))
        print(f"  {'accuracy':<12}{'':>{col_w}}  {'':>{col_w}}  {_fmt(c_acc):>{col_w}}  {'':>{col_w}}  |  {'':>{col_w}}  {'':>{col_w}}  {_fmt(b_acc):>{col_w}}")
        print(f"  {'-' * 72}")

    print("\n" + "=" * 76)
    print("  MODEL COMPARISON — TEST SET (held-out 10%)")
    print("=" * 76)
    print(f"  Current model : {CURRENT_MODEL_FILE}")
    print(f"  Backup model  : {BACKUP_MODEL_FILE}")
    print(f"  Threshold     : {threshold}")

    # --- Without threshold ---
    section(
        "Without confidence threshold:",
        current_results["report_raw"],
        backup_results["report_raw"],
    )

    print(f"\n  Current confusion matrix (no threshold):")
    _print_confusion(current_results["cm_raw"], LABELS)
    print(f"\n  Backup confusion matrix (no threshold):")
    _print_confusion(backup_results["cm_raw"], LABELS)

    # --- With threshold ---
    section(
        f"With confidence threshold = {threshold}  "
        f"(current: {current_results['n_forced']} forced HOLD  |  "
        f"backup: {backup_results['n_forced']} forced HOLD):",
        current_results["report_thresh"],
        backup_results["report_thresh"],
    )

    print(f"\n  Current confusion matrix (threshold = {threshold}):")
    _print_confusion(current_results["cm_thresh"], LABELS)
    print(f"\n  Backup confusion matrix (threshold = {threshold}):")
    _print_confusion(backup_results["cm_thresh"], LABELS)

    # --- Delta summary ---
    print("\n" + "=" * 76)
    print("  DELTA SUMMARY  (Current − Backup, with threshold)")
    print("=" * 76)
    col_w = 10
    print(f"  {'':12}{'F1 Delta':>{col_w}}  {'Prec Delta':>{col_w}}  {'Recall Delta':>{col_w}}")
    print(f"  {'-' * 48}")

    cur_t = current_results["report_thresh"]
    bak_t = backup_results["report_thresh"]

    for lbl in LABELS + ["macro avg", "weighted avg"]:
        c = cur_t.get(lbl, {})
        b = bak_t.get(lbl, {})
        if not c or not b:
            continue
        df1 = c["f1-score"] - b["f1-score"]
        dp  = c["precision"] - b["precision"]
        dr  = c["recall"]    - b["recall"]
        sign = lambda v: "+" if v >= 0 else ""
        print(
            f"  {lbl:<12}"
            f"  {sign(df1)}{df1:>{col_w - 1}.3f}"
            f"  {sign(dp)}{dp:>{col_w - 1}.3f}"
            f"  {sign(dr)}{dr:>{col_w - 1}.3f}"
        )

    c_acc = cur_t.get("accuracy", float("nan"))
    b_acc = bak_t.get("accuracy", float("nan"))
    da    = c_acc - b_acc
    sign  = "+" if da >= 0 else ""
    print(f"  {'accuracy':<12}  {sign}{da:>{col_w - 1}.3f}")
    print("=" * 76 + "\n")


def main() -> None:
    print("=== Loading data ===")
    X, y = build_dataset()

    n         = len(X)
    val_end   = int(n * 0.90)
    X_test    = X.iloc[val_end:]
    y_test    = y.iloc[val_end:]

    print(f"\nTest set: {len(X_test)} rows  "
          f"({X_test.index.min().date()} to {X_test.index.max().date()})")

    print(f"\n=== Loading model bundles ===")
    current_bundle = load_bundle(CURRENT_MODEL_FILE)
    print(f"  Loaded current : {CURRENT_MODEL_FILE}")
    backup_bundle  = load_bundle(BACKUP_MODEL_FILE)
    print(f"  Loaded backup  : {BACKUP_MODEL_FILE}")

    threshold = CONFIDENCE_THRESHOLD
    print(f"\n=== Evaluating both models (threshold = {threshold}) ===")
    current_results = evaluate(current_bundle, X_test, y_test, threshold)
    backup_results  = evaluate(backup_bundle,  X_test, y_test, threshold)

    print_side_by_side(current_results, backup_results, threshold)


if __name__ == "__main__":
    main()
