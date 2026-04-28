"""
agent_main.py — CLI entry point for the daily news-validation agent run.

Thin orchestration wrapper around `agent_runner.run_agent`. Adds:
  * idempotent re-runs via `.bak.json` rename of any existing run file
  * atomic JSON write (tmp + fsync + os.replace)
  * best-effort Discord summary
  * exit-code semantics (0 on success, 1 on run_error)

Halts loudly on an unparseable existing run file. Any catastrophic failure
still produces a readable run_error envelope on disk so the workflow has a
record of what went wrong.
"""

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from agent_runner import run_agent


RUN_DIR = Path("data/agent_runs")
SIGNAL_LOG_PATH = Path("data/signal_log.csv")


def run_path_for(run_date: str) -> Path:
    return RUN_DIR / f"{run_date}.json"


def compact_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def handle_existing_run_file(run_path: Path) -> None:
    """Rename an existing run file out of the way, or halt if it's unparseable.

    If `run_path` exists and parses as JSON, it is renamed to
    `<stem>.<UTC-timestamp>.bak.json` in the same directory so the new run
    can write a fresh file. If it exists but is not valid JSON, raise
    RuntimeError without touching the file — the operator must inspect.
    """
    if not run_path.exists():
        return

    try:
        with open(run_path) as fh:
            json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"existing run file {run_path} is unparseable: {type(exc).__name__}: {exc}"
        )

    backup = run_path.with_name(f"{run_path.stem}.{compact_utc_timestamp()}.bak.json")
    run_path.rename(backup)


def write_run_file(run_path: Path, envelope: dict) -> None:
    """Atomic JSON write: write to a sibling .tmp file, fsync, os.replace."""
    run_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = run_path.with_suffix(run_path.suffix + ".tmp")
    with open(tmp_path, "w") as fh:
        json.dump(envelope, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, run_path)


def _count_decisions_by_outcome(decisions: list[dict]) -> dict[str, int]:
    """Count decisions by verdict, with errored rows bucketed as ERROR.

    Partition is exhaustive: every decision lands in exactly one bucket.
      * `error` populated         → ERROR
      * agent_decision in CONFIRM/VETO/ABSTAIN → that bucket
      * otherwise (defensive: agent_decision=None and no error,
        which current process_ticker should never produce) → ABSTAIN

    The defensive ABSTAIN rule guards against future process_ticker changes
    silently undercounting the Discord summary.
    """
    counts = {"CONFIRM": 0, "VETO": 0, "ABSTAIN": 0, "ERROR": 0}
    for d in decisions:
        if d.get("error"):
            counts["ERROR"] += 1
        elif d.get("agent_decision") in {"CONFIRM", "VETO", "ABSTAIN"}:
            counts[d["agent_decision"]] += 1
        else:
            counts["ABSTAIN"] += 1

    assert sum(counts.values()) == len(decisions), (
        f"_count_decisions_by_outcome partition violated: "
        f"sum(counts)={sum(counts.values())}, "
        f"len(decisions)={len(decisions)}"
    )
    return counts


def send_discord_summary(envelope: dict) -> None:
    """Best-effort Discord post. Skips silently when no webhook is configured;
    swallows any post failure so the run's exit code is unaffected."""
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        return

    run_date = envelope.get("run_date", "?")
    run_error = envelope.get("run_error")

    if run_error:
        content = f"\U0001F6A8 Agent run FAILED ({run_date}): {run_error}"
    else:
        c = _count_decisions_by_outcome(envelope.get("decisions", []))
        content = (
            f"\U0001F916 Agent run complete ({run_date}): "
            f"CONFIRM {c['CONFIRM']}, VETO {c['VETO']}, "
            f"ABSTAIN {c['ABSTAIN']}, ERROR {c['ERROR']}"
        )

    try:
        requests.post(webhook, json={"content": content}, timeout=10)
    except Exception as exc:
        print(
            f"  [DISCORD] post failed (swallowed): {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def main() -> int:
    run_date = datetime.now(timezone.utc).date().isoformat()
    run_path = run_path_for(run_date)

    try:
        handle_existing_run_file(run_path)
        envelope = run_agent(
            signal_log_path=str(SIGNAL_LOG_PATH),
            run_date=run_date,
        )
    except Exception as exc:
        envelope = {
            "run_date": run_date,
            "agent_version": "unknown",
            "llm_model": "unknown",
            "run_started_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "run_duration_seconds": 0.0,
            "decisions": [],
            "run_error": f"orchestration failed: {type(exc).__name__}: {exc}",
        }

    write_run_file(run_path, envelope)
    send_discord_summary(envelope)

    return 0 if envelope.get("run_error") is None else 1


if __name__ == "__main__":
    sys.exit(main())
