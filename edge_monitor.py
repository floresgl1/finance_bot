"""edge_monitor.py — weekly rolling-edge degradation alarm.

Reads signal_log.csv, computes the rolling hit rate over the last
EDGE_WINDOW_SIZE evaluated BUY signals, and posts a Discord alert when
the rate transitions across EDGE_BREAKEVEN_HIT_RATE (or when the
underlying evaluations go stale). Alerts fire on transitions only.

State is persisted to EDGE_MONITOR_STATE_PATH between runs.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

import config


logger = logging.getLogger(__name__)


_STATE_KEYS = {
    "last_run_utc",
    "last_state",
    "last_meaningful_state",
    "last_hit_rate",
    "last_window_size",
}
_VALID_LAST_STATE = {"ABOVE_THRESHOLD", "BELOW_THRESHOLD", "INSUFFICIENT_DATA", "STALE"}
_VALID_MEANINGFUL = {"ABOVE_THRESHOLD", "BELOW_THRESHOLD", None}


def _today_utc() -> datetime:
    return datetime.now(timezone.utc)


def _load_state(state_path: str) -> Optional[dict]:
    if not os.path.exists(state_path):
        return None
    with open(state_path) as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"edge monitor state file {state_path}: expected JSON object, got {type(data).__name__}"
        )
    if set(data.keys()) != _STATE_KEYS:
        raise ValueError(
            f"edge monitor state file {state_path}: schema mismatch — keys={sorted(data.keys())}"
        )
    if data["last_state"] not in _VALID_LAST_STATE:
        raise ValueError(
            f"edge monitor state file {state_path}: invalid last_state {data['last_state']!r}"
        )
    if data["last_meaningful_state"] not in _VALID_MEANINGFUL:
        raise ValueError(
            f"edge monitor state file {state_path}: invalid last_meaningful_state "
            f"{data['last_meaningful_state']!r}"
        )
    return data


def _write_state_atomic(state_path: str, state: dict) -> None:
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, state_path)


def _filter_evaluated_buys(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["row_type"] == "ENTRY")
        & (df["model_signal"] == "BUY")
        & (df["result"].isin(["WIN", "LOSS"]))
    ]


def _compute_state(
    df: pd.DataFrame,
    today: datetime,
) -> tuple[str, Optional[float], int, Optional[str], Optional[str]]:
    """Returns (state, hit_rate_or_none, count_for_state, min_date, max_date).

    `count_for_state` is EDGE_WINDOW_SIZE for ABOVE/BELOW, total available
    evaluated-BUY rows for INSUFFICIENT_DATA / STALE.
    """
    evaluated = _filter_evaluated_buys(df)
    total_count = len(evaluated)

    if total_count < config.EDGE_WINDOW_SIZE:
        return "INSUFFICIENT_DATA", None, total_count, None, None

    sorted_eval = evaluated.sort_values("date")
    window = sorted_eval.tail(config.EDGE_WINDOW_SIZE)
    min_date_str = str(window["date"].min())
    max_date_str = str(window["date"].max())

    max_date = pd.to_datetime(max_date_str).date()
    days_old = (today.date() - max_date).days
    if days_old > config.EDGE_STALENESS_DAYS:
        return "STALE", None, total_count, min_date_str, max_date_str

    wins = (window["result"] == "WIN").sum()
    hit_rate = float(wins) / config.EDGE_WINDOW_SIZE

    state = (
        "BELOW_THRESHOLD" if hit_rate < config.EDGE_BREAKEVEN_HIT_RATE else "ABOVE_THRESHOLD"
    )
    return state, hit_rate, config.EDGE_WINDOW_SIZE, min_date_str, max_date_str


def _should_alert(
    current_state: str,
    last_state: Optional[str],
    last_meaningful: Optional[str],
) -> bool:
    if current_state == "INSUFFICIENT_DATA":
        return False
    if current_state == "STALE":
        return last_state != "STALE"
    if current_state == "ABOVE_THRESHOLD":
        return last_meaningful == "BELOW_THRESHOLD"
    if current_state == "BELOW_THRESHOLD":
        return last_meaningful is None or last_meaningful == "ABOVE_THRESHOLD"
    raise ValueError(f"unexpected current_state {current_state!r}")


def _format_degradation(hit_rate: float, min_date: str, max_date: str) -> str:
    return (
        "\U0001F4C9 EDGE DEGRADATION ALERT\n\n"
        f"Rolling 30-BUY hit rate: {hit_rate * 100:.1f}% (below 31% break-even)\n"
        f"Window: last 30 evaluated BUYs ({min_date} to {max_date})\n"
        "Baseline (2026-04-19): 36.6% hit rate, +1.86% mean return\n\n"
        "Consider retraining. Investigate before action."
    )


def _format_recovery(hit_rate: float, min_date: str, max_date: str) -> str:
    return (
        "✅ EDGE RECOVERY\n\n"
        f"Rolling 30-BUY hit rate: {hit_rate * 100:.1f}% (above 31% break-even)\n"
        f"Window: last 30 evaluated BUYs ({min_date} to {max_date})"
    )


def _format_stale(max_date: str, days_old: int) -> str:
    return (
        "⚠️ EDGE MONITOR — STALE EVALUATIONS\n\n"
        f"Most recent evaluated BUY: {max_date} ({days_old} days ago)\n"
        f"Threshold: {config.EDGE_STALENESS_DAYS} days. Outcome tracker may not be running.\n"
        "No edge assessment performed this run."
    )


_RETRAIN_DISPATCH_URL = (
    "https://api.github.com/repos/floresgl1/finance_bot"
    "/actions/workflows/retrain.yml/dispatches"
)


def _trigger_retrain_workflow() -> None:
    """Dispatch the retrain workflow on GitHub Actions with auto_mode=true.

    The workflow trains a challenger, evaluates it, and uploads the candidate
    to PythonAnywhere for manual approval (``promote_model.py --approve``).
    A missing GITHUB_PAT or a failed dispatch is logged but never fatal —
    the edge monitor's primary job is the alert, not the retrain.
    """
    token = os.environ.get("GITHUB_PAT")
    if not token:
        logger.warning("GITHUB_PAT not set; skipping auto-retrain trigger")
        return
    try:
        resp = requests.post(
            _RETRAIN_DISPATCH_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            json={"ref": "main", "inputs": {"auto_mode": "true"}},
            timeout=15,
        )
        if resp.status_code == 204:
            logger.info("retrain workflow dispatched successfully")
        else:
            logger.warning(
                "retrain dispatch returned HTTP %s: %s",
                resp.status_code,
                resp.text.strip()[:200],
            )
    except Exception as exc:
        logger.warning(
            "retrain dispatch failed (swallowed): %s: %s",
            type(exc).__name__,
            exc,
        )


def _try_send_discord(content: str) -> None:
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        logger.warning(
            "DISCORD_WEBHOOK_URL not set; skipping edge monitor Discord alert"
        )
        return
    try:
        resp = requests.post(webhook, json={"content": content}, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning(
            "edge monitor Discord post failed (swallowed): %s: %s",
            type(exc).__name__,
            exc,
        )


def run_edge_monitor(signal_log_path: str = "data/signal_log.csv") -> None:
    """Compute current edge state, compare to last meaningful state, alert on transition."""
    state_path = config.EDGE_MONITOR_STATE_PATH

    prior = _load_state(state_path)
    last_state = prior["last_state"] if prior else None
    last_meaningful = prior["last_meaningful_state"] if prior else None

    df = pd.read_csv(signal_log_path)

    today = _today_utc()
    current_state, hit_rate, count_for_state, min_date, max_date = _compute_state(df, today)

    if _should_alert(current_state, last_state, last_meaningful):
        if current_state == "BELOW_THRESHOLD":
            content = _format_degradation(hit_rate, min_date, max_date)
        elif current_state == "ABOVE_THRESHOLD":
            content = _format_recovery(hit_rate, min_date, max_date)
        else:  # STALE
            days_old = (today.date() - pd.to_datetime(max_date).date()).days
            content = _format_stale(max_date, days_old)
        _try_send_discord(content)

        if current_state == "BELOW_THRESHOLD":
            _trigger_retrain_workflow()

    new_meaningful = (
        current_state
        if current_state in {"ABOVE_THRESHOLD", "BELOW_THRESHOLD"}
        else last_meaningful
    )
    new_state = {
        "last_run_utc": today.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_state": current_state,
        "last_meaningful_state": new_meaningful,
        "last_hit_rate": hit_rate,
        "last_window_size": count_for_state,
    }
    _write_state_atomic(state_path, new_state)

    logger.info(
        "edge monitor: state=%s hit_rate=%s window=%s",
        current_state,
        hit_rate,
        count_for_state,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    run_edge_monitor()
