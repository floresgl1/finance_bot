"""
agent_pretrade_main.py — Phase 2 CLI entry point for the pre-trade agent.

Reads pending_signals.json (written by generate_signals.py on PA at 14:00),
evaluates each signal via the LLM agent, and writes two files:

  1. data/agent_runs/{run_date}_pretrade.json  — full run artifact (analysis)
  2. data/agent_decisions.json                 — simplified ticker→verdict map
                                                 (read by live_trader.py at 15:00)

Exit codes:
    0 — run completed (decisions written, even if some tickers errored)
    1 — catastrophic failure (no decisions file written)
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from agent_runner import run_agent_from_signals


PENDING_SIGNALS_PATH = Path("data/pending_signals.json")
RUN_DIR = Path("data/agent_runs")
DECISIONS_PATH = Path("data/agent_decisions.json")


def _read_pending_signals(path: Path) -> tuple[str, list[dict]]:
    """Read and validate pending_signals.json.

    Returns:
        (run_date, signals_list). Raises on missing/invalid file.
    """
    with open(path) as fh:
        envelope = json.load(fh)

    run_date = envelope.get("date")
    if not isinstance(run_date, str) or len(run_date) != 10:
        raise ValueError(f"invalid or missing 'date' in {path}: {run_date!r}")

    signals = envelope.get("signals", [])
    if not isinstance(signals, list):
        raise ValueError(f"'signals' must be a list in {path}")

    # Validate each signal has the required keys
    required_keys = {"ticker", "model_signal", "confidence", "shap_values"}
    for i, sig in enumerate(signals):
        missing = required_keys - set(sig.keys())
        if missing:
            raise ValueError(
                f"signal[{i}] missing keys {missing} in {path}"
            )

    return run_date, signals


def _build_decisions_map(decisions: list[dict]) -> dict[str, str]:
    """Extract a simple {ticker: verdict} map from the full decisions list.

    Tickers where the agent errored are mapped to ABSTAIN (permissive default).
    """
    result = {}
    for d in decisions:
        ticker = d.get("ticker")
        verdict = d.get("agent_decision")
        if verdict in ("CONFIRM", "VETO", "ABSTAIN"):
            result[ticker] = verdict
        else:
            # Error or null decision — default to ABSTAIN (permissive)
            result[ticker] = "ABSTAIN"
    return result


def _write_atomic(path: Path, data: dict) -> None:
    """Atomic JSON write: tmp + fsync + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _send_discord_summary(envelope: dict, decisions_map: dict) -> None:
    """Best-effort Discord post with pretrade agent results."""
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        return

    run_date = envelope.get("run_date", "?")
    run_error = envelope.get("run_error")

    if run_error:
        content = f"\U0001F6A8 Pre-trade agent FAILED ({run_date}): {run_error}"
    else:
        veto_count = sum(1 for v in decisions_map.values() if v == "VETO")
        confirm_count = sum(1 for v in decisions_map.values() if v == "CONFIRM")
        abstain_count = sum(1 for v in decisions_map.values() if v == "ABSTAIN")

        lines = [
            f"\U0001F916 Pre-trade agent ({run_date}): "
            f"CONFIRM {confirm_count}, VETO {veto_count}, ABSTAIN {abstain_count}",
        ]

        # List vetoed tickers explicitly — these will be blocked at 15:00
        vetoed = [t for t, v in decisions_map.items() if v == "VETO"]
        if vetoed:
            lines.append(f"\U0001F6AB Will veto: {', '.join(sorted(vetoed))}")

        content = "\n".join(lines)

    try:
        requests.post(webhook, json={"content": content}, timeout=10)
    except Exception as exc:
        print(
            f"  [DISCORD] post failed (swallowed): {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def main() -> int:
    # Read pending signals from Phase 1
    if not PENDING_SIGNALS_PATH.exists():
        print(f"[SKIP] {PENDING_SIGNALS_PATH} not found — no signals to evaluate.")
        # Write an empty decisions file so live_trader.py sees ABSTAIN for all
        today = datetime.now(timezone.utc).date().isoformat()
        _write_atomic(DECISIONS_PATH, {"date": today, "decisions": {}})
        return 0

    try:
        run_date, signals = _read_pending_signals(PENDING_SIGNALS_PATH)
    except Exception as exc:
        print(f"[FATAL] Cannot read {PENDING_SIGNALS_PATH}: {exc}", file=sys.stderr)
        return 1

    if not signals:
        print(f"[SKIP] No BUY/SELL signals in {PENDING_SIGNALS_PATH}.")
        _write_atomic(DECISIONS_PATH, {"date": run_date, "decisions": {}})
        return 0

    print(f"[START] Evaluating {len(signals)} signal(s) for {run_date}")
    for sig in signals:
        print(f"  {sig['ticker']} {sig['model_signal']} conf={sig['confidence']}")

    # Run the agent
    try:
        envelope = run_agent_from_signals(signals, run_date)
    except Exception as exc:
        print(f"[FATAL] Agent failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # Build the simplified decisions map for live_trader.py
    decisions_map = _build_decisions_map(envelope.get("decisions", []))

    # Write both outputs
    pretrade_run_path = RUN_DIR / f"{run_date}_pretrade.json"
    _write_atomic(pretrade_run_path, envelope)
    print(f"[OK] Full run → {pretrade_run_path}")

    decisions_envelope = {"date": run_date, "decisions": decisions_map}
    _write_atomic(DECISIONS_PATH, decisions_envelope)
    print(f"[OK] Decisions → {DECISIONS_PATH}")

    for ticker, verdict in sorted(decisions_map.items()):
        print(f"  {ticker}: {verdict}")

    # Discord summary
    _send_discord_summary(envelope, decisions_map)

    return 0


if __name__ == "__main__":
    sys.exit(main())
