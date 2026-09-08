"""
promote_model.py — Champion/challenger promotion gate for the signal model.

Closes the loop the monitoring stack left open: edge_monitor.py detects that the
live model's edge has decayed, but nothing retrained it, and nothing decided
whether a retrained model was actually better. The live model went 5+ months
without a retrain because that decision was manual and never made.

Flow
----
    1. Train a challenger to a candidate path (live model untouched).
    2. Rebuild the dataset and take the same chronological 70/20/10 split
       trainer.py uses; the last 10% is the shared held-out test set.
    3. Evaluate champion and challenger on that identical test set.
    4. Apply the gate (decide_promotion). Reject unless the challenger beats
       the champion by a margin, beats buy-and-hold, AND clears absolute floors.
    5. Only on a pass: archive the champion, then swap the candidate in.
    6. Report the decision to Discord either way.

**DESIGN DECISION:**
The gate is failure-biased — every ambiguous case keeps the incumbent. A model
already trading has known live behaviour; a challenger has only test-set
numbers. Ties, thin test sets, and unreadable champions all resolve to REJECT.
Promotion is the exception that must be argued for, not the default.

**DESIGN DECISION:**
Both models are scored inside this process on one freshly-built split rather
than trusting metrics reported by whatever produced each model. A comparison is
only meaningful if both sides saw byte-identical test data.

Usage
-----
    python promote_model.py                 # train, evaluate, promote if better
    python promote_model.py --dry-run       # decide and report, never swap
    python promote_model.py --skip-train \
        --candidate models/XG_Boost_candidate.joblib
                                            # evaluate an existing candidate

Exit codes
----------
    0 — a decision was reached (promoted OR rejected; both are success)
    1 — the run could not reach a decision (missing data, unreadable model)
"""

import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import joblib
import requests

from config import (
    MODEL_DIR,
    MODEL_FILENAME,
    CANDIDATE_MODEL_FILENAME,
    MODEL_ARCHIVE_DIR,
    MODEL_ARCHIVE_RETAIN,
    PROMOTION_DECISION_PATH,
    CONFIDENCE_THRESHOLD,
    PROMOTION_MIN_BUY_PRECISION,
    PROMOTION_MIN_BUY_RECALL,
    PROMOTION_MIN_TEST_BUY_SUPPORT,
    PROMOTION_MIN_RETURN_IMPROVEMENT_PCT,
    PROMOTION_MIN_BACKTEST_TRADES,
    PROMOTION_MAX_DRAWDOWN_PCT,
    PROMOTION_MIN_HOLD_DELTA_PCT,
)

CHAMPION_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
CANDIDATE_PATH = os.path.join(MODEL_DIR, CANDIDATE_MODEL_FILENAME)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------
def extract_gate_metrics(report: dict) -> dict:
    """Pull the BUY-class metrics the gate reasons about out of a
    classification_report(output_dict=True).

    BUY is the class that matters: a false BUY spends real money, while a
    missed BUY costs only opportunity. HOLD dominates the label distribution,
    so overall accuracy would hide a model that simply stopped buying.
    """
    buy = report.get("BUY") or {}
    return {
        "buy_f1":        float(buy.get("f1-score", 0.0)),
        "buy_precision": float(buy.get("precision", 0.0)),
        "buy_recall":    float(buy.get("recall", 0.0)),
        "buy_support":   int(buy.get("support", 0)),
        "accuracy":      float(report.get("accuracy", 0.0)),
    }


def extract_backtest_metrics(stats: dict, hold_stats: dict | None = None) -> dict:
    """Pull the simulated-trading metrics the gate decides on.

    An empty stats dict (backtest produced nothing) collapses to values that
    fail every check rather than passing vacuously — the gate must not promote
    on the strength of a simulation that did not run.

    `hold_return` is None when no buy-and-hold arm was run. That is deliberately
    not 0.0: a missing benchmark must fail the comparison, and a zero would let
    any profitable challenger clear it.
    """
    hold_return = (
        float(hold_stats.get("total_return", 0.0)) if hold_stats else None
    )
    if not stats:
        return {
            "total_return": 0.0,
            "n_trades":     0,
            "bt_win_rate":  0.0,
            "avg_return":   0.0,
            "max_drawdown": -100.0,
            "final_value":  0.0,
            "hold_return":  hold_return,
        }
    return {
        "total_return": float(stats.get("total_return", 0.0)),
        "n_trades":     int(stats.get("n_trades", 0)),
        "bt_win_rate":  float(stats.get("win_rate", 0.0)),
        "avg_return":   float(stats.get("avg_return", 0.0)),
        "max_drawdown": float(stats.get("max_drawdown", -100.0)),
        "final_value":  float(stats.get("final_value", 0.0)),
        "hold_return":  hold_return,
    }


def score_model(bundle: dict, X_test, y_test, threshold: float) -> dict:
    """Score one model both ways: classification on the test rows, and
    simulated trading over the same window.

    Returns one merged metrics dict — the gate reasons about a single object
    per model rather than juggling two parallel ones.
    """
    from compare_models import evaluate
    from backtest import run_benchmarks, load_all_tickers

    classification = extract_gate_metrics(
        evaluate(bundle, X_test, y_test, threshold)["report_thresh"]
    )

    # Signals are model-specific, so ticker_data must be regenerated per model.
    # run_benchmarks applies the same split boundaries as trainer.py, and
    # simulates the equal-weight hold arm over the same dates from the same
    # frames — so the gate never compares against a differently-dated basket.
    ticker_data = load_all_tickers(bundle["model"], bundle["encoder"])
    results = run_benchmarks(
        split="test",
        model=bundle["model"],
        encoder=bundle["encoder"],
        ticker_data=ticker_data,
    )

    return {
        **classification,
        **extract_backtest_metrics(
            results.get("strategy") or {}, results.get("buy_and_hold")
        ),
    }


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def decide_promotion(
    champion: dict | None,
    challenger: dict,
) -> dict:
    """Decide whether the challenger replaces the champion.

    Parameters
    ----------
    champion
        Gate metrics for the incumbent, or None when no champion exists yet
        (first-ever training run).
    challenger
        Gate metrics for the newly trained model.

    Returns
    -------
    dict with:
        promote  — bool, True only if every check passed
        checks   — list of {name, passed, detail} in evaluation order
        summary  — one-line human-readable verdict
    """
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str) -> bool:
        checks.append({"name": name, "passed": passed, "detail": detail})
        return passed

    # --- 1. Is the test set large enough to judge on? ----------------------
    support_ok = record(
        "test_buy_support",
        challenger["buy_support"] >= PROMOTION_MIN_TEST_BUY_SUPPORT,
        f"{challenger['buy_support']} true BUY rows in test set "
        f"(minimum {PROMOTION_MIN_TEST_BUY_SUPPORT})",
    )

    # --- 2. Absolute floors on the challenger ------------------------------
    precision_ok = record(
        "challenger_buy_precision",
        challenger["buy_precision"] >= PROMOTION_MIN_BUY_PRECISION,
        f"BUY precision {challenger['buy_precision']:.3f} "
        f"(floor {PROMOTION_MIN_BUY_PRECISION})",
    )

    recall_ok = record(
        "challenger_buy_recall",
        challenger["buy_recall"] >= PROMOTION_MIN_BUY_RECALL,
        f"BUY recall {challenger['buy_recall']:.3f} "
        f"(floor {PROMOTION_MIN_BUY_RECALL}) - guards against a model that "
        f"scores well by refusing to buy",
    )

    # --- 3. Enough simulated trades to judge the return on ------------------
    trades_ok = record(
        "backtest_trade_count",
        challenger["n_trades"] >= PROMOTION_MIN_BACKTEST_TRADES,
        f"{challenger['n_trades']} simulated trades "
        f"(minimum {PROMOTION_MIN_BACKTEST_TRADES})",
    )

    # --- 4. Risk floor ------------------------------------------------------
    drawdown_ok = record(
        "challenger_max_drawdown",
        challenger["max_drawdown"] >= PROMOTION_MAX_DRAWDOWN_PCT,
        f"max drawdown {challenger['max_drawdown']:.2f}% "
        f"(floor {PROMOTION_MAX_DRAWDOWN_PCT:.2f}%) - a model that earns more "
        f"by risking ruin is not an improvement",
    )

    # --- 5. Beat the incumbent in dollars ----------------------------------
    if champion is None:
        improvement_ok = record(
            "beats_champion_return",
            True,
            "no champion on disk - floors alone decide this first promotion",
        )
    else:
        required = champion["total_return"] + PROMOTION_MIN_RETURN_IMPROVEMENT_PCT
        delta = challenger["total_return"] - champion["total_return"]
        improvement_ok = record(
            "beats_champion_return",
            challenger["total_return"] >= required,
            f"backtest return {challenger['total_return']:+.2f}% vs champion "
            f"{champion['total_return']:+.2f}% (delta {delta:+.2f}pp, "
            f"required {PROMOTION_MIN_RETURN_IMPROVEMENT_PCT:+.2f}pp)",
        )

    # --- 6. Beat doing nothing ---------------------------------------------
    # Independent of the champion. Two models can trade each other in circles
    # while both lose to holding the basket, and checks 1-5 would promote on
    # every lap.
    hold_return = challenger.get("hold_return")
    if hold_return is None:
        beats_hold_ok = record(
            "beats_buy_and_hold",
            False,
            "no buy-and-hold benchmark was produced - refusing to promote a "
            "model that cannot be compared against doing nothing",
        )
    else:
        required_vs_hold = hold_return + PROMOTION_MIN_HOLD_DELTA_PCT
        hold_delta = challenger["total_return"] - hold_return
        beats_hold_ok = record(
            "beats_buy_and_hold",
            challenger["total_return"] >= required_vs_hold,
            f"backtest return {challenger['total_return']:+.2f}% vs "
            f"equal-weight hold {hold_return:+.2f}% "
            f"(delta {hold_delta:+.2f}pp, required "
            f"{PROMOTION_MIN_HOLD_DELTA_PCT:+.2f}pp)",
        )

    promote = (
        support_ok and precision_ok and recall_ok
        and trades_ok and drawdown_ok and improvement_ok
        and beats_hold_ok
    )

    if promote:
        summary = "PROMOTE - challenger cleared every gate check"
    else:
        failed = [c["name"] for c in checks if not c["passed"]]
        summary = f"REJECT - keeping champion; failed: {', '.join(failed)}"

    return {"promote": promote, "checks": checks, "summary": summary}


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------
def archive_champion(champion_path: str = CHAMPION_PATH) -> str | None:
    """Copy the current champion into MODEL_ARCHIVE_DIR with a UTC timestamp.

    Returns the archive path, or None if there was no champion to archive.
    Archiving is a copy, not a move: the champion stays in place until the
    candidate has actually been written over it.
    """
    if not os.path.exists(champion_path):
        return None

    os.makedirs(MODEL_ARCHIVE_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = os.path.splitext(os.path.basename(champion_path))[0]
    archive_path = os.path.join(MODEL_ARCHIVE_DIR, f"{base}_{stamp}.joblib")

    shutil.copy2(champion_path, archive_path)
    _prune_archive()
    return archive_path


def _prune_archive() -> None:
    """Keep only the newest MODEL_ARCHIVE_RETAIN archived models."""
    entries = sorted(glob.glob(os.path.join(MODEL_ARCHIVE_DIR, "*.joblib")))
    for stale in entries[:-MODEL_ARCHIVE_RETAIN] if len(entries) > MODEL_ARCHIVE_RETAIN else []:
        try:
            os.remove(stale)
            print(f"  [ARCHIVE] Pruned {os.path.basename(stale)}")
        except OSError as exc:
            print(f"  [ARCHIVE] Warning: could not prune {stale}: {exc}")


def install_candidate(candidate_path: str, champion_path: str = CHAMPION_PATH) -> None:
    """Move the candidate into the live model path."""
    os.makedirs(os.path.dirname(os.path.abspath(champion_path)), exist_ok=True)
    shutil.move(candidate_path, champion_path)


def write_decision_file(
    decision: dict,
    champion: dict | None,
    challenger: dict,
    *,
    dry_run: bool,
    archived_to: str | None,
    path: str = PROMOTION_DECISION_PATH,
) -> None:
    """Persist the decision as JSON for downstream consumers.

    The candidate file is consumed on both outcomes — moved on promotion,
    deleted on rejection — so its absence cannot be used to infer what
    happened. CI reads this file instead.
    """
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "promoted": decision["promote"],
        "dry_run": dry_run,
        "summary": decision["summary"],
        "checks": decision["checks"],
        "champion": champion,
        "challenger": challenger,
        "archived_to": archived_to,
    }
    try:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  Decision written to {path}")
    except OSError as exc:
        print(f"  [DECISION] Warning: could not write {path}: {exc}")


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------
def send_discord(message: str) -> None:
    """POST a message to the Discord webhook. Silently skips if URL not set."""
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        print("  [DISCORD] DISCORD_WEBHOOK_URL not set - skipping notification.")
        return
    try:
        resp = requests.post(url, json={"content": message}, timeout=10)
        if resp.status_code not in (200, 204):
            print(f"  [DISCORD] HTTP {resp.status_code}: {resp.text.strip()}")
    except Exception as exc:
        print(f"  [DISCORD] Notification failed: {exc}")


def build_report(
    decision: dict,
    champion: dict | None,
    challenger: dict,
    *,
    dry_run: bool,
    archived_to: str | None,
) -> str:
    """Format the promotion decision for Discord."""
    icon = "✅" if decision["promote"] else "🛑"
    header = f"{icon} **MODEL PROMOTION — {'PROMOTED' if decision['promote'] else 'REJECTED'}**"
    if dry_run:
        header += "  _(dry run — no files changed)_"

    lines = [header, ""]

    def _row(label: str, m: dict) -> str:
        return (
            f"{label} — return **{m['total_return']:+.2f}%** over "
            f"{m['n_trades']} trades, drawdown {m['max_drawdown']:.2f}%\n"
            f"　　BUY F1 {m['buy_f1']:.3f}  prec {m['buy_precision']:.3f}  "
            f"recall {m['buy_recall']:.3f}  acc {m['accuracy']:.3f}"
        )

    if champion is None:
        lines.append("Champion: _none on disk_")
    else:
        lines.append(_row("Champion  ", champion))
    lines.append(_row("Challenger", challenger))

    # The benchmark is the reason a promotion can be rejected while the
    # challenger still beats the champion, so it belongs in the report body
    # rather than only inside a check line.
    hold_return = challenger.get("hold_return")
    if hold_return is None:
        lines.append("Buy-and-hold — _not measured_")
    else:
        lines.append(
            f"Buy-and-hold — **{hold_return:+.2f}%** over the same dates "
            f"(challenger {challenger['total_return'] - hold_return:+.2f}pp)"
        )
    lines.append("")

    for check in decision["checks"]:
        mark = "✓" if check["passed"] else "✗"
        lines.append(f"{mark} `{check['name']}` — {check['detail']}")

    if archived_to:
        lines.append("")
        lines.append(f"Previous champion archived to `{archived_to}`")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a challenger model and promote it only if it beats the champion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate and report the decision without touching any model file.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Evaluate an existing candidate instead of training a new one.",
    )
    parser.add_argument(
        "--candidate",
        default=CANDIDATE_PATH,
        metavar="PATH",
        help=f"Candidate model path (default: {CANDIDATE_PATH}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=(
            "Confidence threshold both models are scored at "
            f"(default: CONFIDENCE_THRESHOLD = {CONFIDENCE_THRESHOLD})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Imported here so --help and the unit-testable gate above do not pay for
    # xgboost/sklearn import time.
    from trainer import build_dataset, train
    from compare_models import evaluate

    print("=" * 76)
    print("  MODEL PROMOTION GATE")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 76)

    # --- 1. Train the challenger ------------------------------------------
    if args.skip_train:
        if not os.path.exists(args.candidate):
            print(f"[FATAL] --skip-train given but no candidate at {args.candidate}")
            return 1
        print(f"\n[SKIP] Using existing candidate: {args.candidate}")
    else:
        print("\n=== Training challenger (live model untouched) ===")
        try:
            train(output_path=args.candidate, headless=True, update_config=False)
        except Exception as exc:
            print(f"[FATAL] Challenger training failed: {exc}")
            return 1

    # --- 2. Rebuild the shared test split ---------------------------------
    print("\n=== Building shared held-out test set ===")
    try:
        X, y = build_dataset()
    except Exception as exc:
        print(f"[FATAL] Could not build dataset: {exc}")
        return 1

    val_end = int(len(X) * 0.90)
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    print(
        f"  Test set: {len(X_test)} rows "
        f"({X_test.index.min().date()} to {X_test.index.max().date()})"
    )

    # --- 3. Evaluate both models on identical data ------------------------
    try:
        challenger_bundle = joblib.load(args.candidate)
    except Exception as exc:
        print(f"[FATAL] Could not load candidate {args.candidate}: {exc}")
        return 1

    print("\n=== Scoring challenger (classification + backtest) ===")
    challenger_metrics = score_model(challenger_bundle, X_test, y_test, args.threshold)

    champion_metrics = None
    if os.path.exists(CHAMPION_PATH):
        try:
            champion_bundle = joblib.load(CHAMPION_PATH)
            print("\n=== Scoring champion (classification + backtest) ===")
            champion_metrics = score_model(champion_bundle, X_test, y_test, args.threshold)
        except Exception as exc:
            # An unreadable champion is not a licence to promote blindly.
            print(f"[FATAL] Champion exists at {CHAMPION_PATH} but could not be scored: {exc}")
            print("        Refusing to promote against an unknown incumbent.")
            return 1
    else:
        print(f"\n[INFO] No champion at {CHAMPION_PATH} - this is a first promotion.")

    # --- 4. Decide ---------------------------------------------------------
    decision = decide_promotion(champion_metrics, challenger_metrics)

    print("\n" + "=" * 76)
    print(f"  DECISION: {decision['summary']}")
    print("=" * 76)
    for check in decision["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {check['name']:<26} {check['detail']}")

    # --- 5. Act ------------------------------------------------------------
    archived_to = None
    if decision["promote"] and not args.dry_run:
        archived_to = archive_champion()
        if archived_to:
            print(f"\n  Champion archived to {archived_to}")
        install_candidate(args.candidate)
        print(f"  Challenger installed as live model: {CHAMPION_PATH}")
    elif decision["promote"] and args.dry_run:
        print("\n  [DRY RUN] Would promote - no files changed.")
    else:
        # Remove the rejected candidate so a later --skip-train run cannot
        # silently evaluate a stale challenger.
        if not args.dry_run and os.path.exists(args.candidate):
            os.remove(args.candidate)
            print(f"\n  Rejected candidate removed: {args.candidate}")

    # --- 6. Report ---------------------------------------------------------
    write_decision_file(
        decision,
        champion_metrics,
        challenger_metrics,
        dry_run=args.dry_run,
        archived_to=archived_to,
    )
    send_discord(
        build_report(
            decision,
            champion_metrics,
            challenger_metrics,
            dry_run=args.dry_run,
            archived_to=archived_to,
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
