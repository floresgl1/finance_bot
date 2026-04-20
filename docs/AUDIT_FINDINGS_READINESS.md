# Production-Readiness Audit Findings

**Date:** 2026-04-19
**Scope:** `live_trader.py`, `rebalancer.py`, `signal_logger.py`, `capital_allocator.py`, `config.py`, `outcome_tracker.py`
**Reference (not audited for gaps):** `predictor.py`, `features.py`, `trainer.py`, `docs/PIPELINE.md`

This is a read-only audit. No code changes have been made. No fixes are prescribed; findings describe only what exists and what is missing.

---

## Task 1: Position-Level Realized P&L Logging

### What currently exists
- `signal_logger.py::log_signal` (`signal_logger.py:74-139`) is the single write path for the bot's append-only CSV log.
- The schema is frozen at `signal_logger.py:48-62`:
  `date, ticker, model_signal, price, qty, confidence, evaluation_date, actual_action, outcome_price, result, shap_driver_1/2/3`.
- `outcome_tracker.py` reads this CSV and, for each row, scores the signal at `today + PREDICTION_DAYS` using a ±2% threshold (`outcome_tracker.py::compute_result`, lines 95-120). This is **signal-level WIN/LOSS scoring**, not position-level accounting. The evaluation is per-row and re-runs independently even if a position was exited earlier.
- There is no dedicated function such as `log_position_closed()` anywhere in the scope. A repo-wide search for `log_position_closed`, `realized_pnl`, `exit_reason`, `realized_return`, and `position_closed` returned zero matches in production code (only `exit_price` appears in `backtest.py`, which is a simulator, not a live logger).

### Exit-pathway coverage
| Exit pathway | Location | What is logged | Entry price captured? | Exit reason distinguishable? | Realized P&L captured? |
|---|---|---|---|---|---|
| Model SELL (full exit) | `live_trader.py:707-721` | `log_signal(ticker, "SELL", price, qty, confidence, "SELL" \| "SELL_ERROR")` | No | Yes (action code) | No |
| Stop-loss | `live_trader.py:455-459` | `log_signal(ticker, "SELL", current_price, qty, 0.0, "STOP_LOSS_SELL")` on success only | No (though `position.avg_entry_price` is read at line 442 and `loss_pct` computed at line 445 — neither persisted) | Yes (action code) | No (percentage is in Discord only) |
| Take-profit | `live_trader.py:461-470` | **Nothing — no `log_signal` call on success or failure** | No | No | No |
| Rebalancer trim (partial) | `rebalancer.py:140`, `rebalancer.py:160` | `log_signal(ticker, "REBALANCER", price, qty, 0.0, "REBALANCER_SELL" \| "REBALANCER_ORDER_FAIL")` | No | Yes (action code) | No |
| Manual | n/a | No manual-exit pathway exists | — | — | — |

### Missing fields
Compared to the position-level schema described in the task (entry_price, exit_price, exit_reason, realized_dollar_pnl, realized_percent_return, days_held), the CSV captures **none** of them. The closest surrogate is `outcome_tracker.py`'s `outcome_price` + `result`, but that is:

1. Computed against the logged-row `price` (which for a `STOP_LOSS_SELL` row is the *exit* price, not the entry) — so the P&L math is structurally wrong for exit rows.
2. Resolved at `evaluation_date = signal_date + PREDICTION_DAYS`, not at the actual close date.
3. Not tied to an actual position lifecycle — there is no notion of matching a BUY row to its closing SELL row.

Also notable: the take-profit branch (`live_trader.py:461-470`) is missing any call to `log_signal` on either success or failure. The signal_logger module docstring (`signal_logger.py:15-37`) does not list a `TAKE_PROFIT_SELL` action code, confirming take-profit exits are entirely absent from the log schema.

### Summary
Position-level realized P&L logging is **entirely absent**. The codebase tracks signals, not positions. Exit pathways log an action code but not entry/exit prices, realized dollars, realized percent, or days held. Take-profit exits are not logged at all.

---

## Task 2: Standing Stop-Loss Orders (Bracket / OCO)

### Exact `submit_order` parameters at BUY time
`live_trader.py::place_buy` (`live_trader.py:343-361`) submits:

```python
api.submit_order(
    symbol        = ticker,
    qty           = qty,
    side          = "buy",
    type          = "market",
    time_in_force = "day",
)
```

Confirmed absent at call site: `order_class`, `stop_loss`, `take_profit`, `trailing_stop`. The rebalancer's submit_order at `rebalancer.py:132-138` uses the same five-parameter shape.

### Constants in `config.py`
No bracket-order-related constants exist in `config.py`. A full read of the file confirms no keys named `STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`, `BRACKET_*`, or any limit/stop-price configuration. The 10% stop-loss and 15% take-profit thresholds are hardcoded as *default Python parameter values* on `check_position_limits` (`live_trader.py:420-421`: `stop_pct: float = 10.0, take_pct: float = 15.0`) and are never passed as arguments.

### Stop-loss enforcement timing
`check_position_limits` is called exactly once per script invocation, at `live_trader.py:638`, inside `run()`. The function iterates open positions, compares current_price to avg_entry_price, and places a synchronous market sell if the in-memory loss_pct exceeds 10%. Once the main loop reaches the Discord summary at line 907 and the script exits, **no further stop-loss evaluation happens until the next scheduled `run_bot.py` invocation**.

Per `docs/PIPELINE.md` (cron schedule section) and the project memory, the bot runs once per weekday at 15:00 UTC. The `check_position_limits` call therefore samples prices once per trading day.

### Unprotected window
Using NYSE regular-session hours (13:30-20:00 UTC = 6.5h) and a single daily bot run at 15:00 UTC:
- Intra-session exposure after the stop-loss check completes: ~5h of the same day's regular session (15:00-20:00 UTC), plus the next day's pre-check window (13:30-15:00 UTC), plus all pre- and post-market and overnight periods.
- Effectively: the stop-loss is enforced only at the instant of the 15:00 UTC snapshot. For ~23h 59m of each 24h period, no standing order is present at Alpaca and no in-script polling is running.
- A gap-down at the next day's open (13:30 UTC) will not be caught until 15:00 UTC — a ~90-minute window during which a position can crash freely. A gap-down occurring between Friday 15:01 UTC and Monday 15:00 UTC is exposed for ~72h.

### Summary
Stop-loss protection exists only as in-script polling during a single once-daily snapshot. No standing stop-loss, bracket, or OCO orders are placed at Alpaca at entry time.

---

## Task 3: Portfolio-Level Daily Loss Limit / Circuit Breaker

### Session-start equity baseline
`live_trader.py::run` captures `equity = get_equity(api)` at lines 636 and 642, and re-fetches at lines 720, 732, 746, 797, and 905. All of these reads are used either (a) to feed `capital_allocator.check_add_to_position` for current sizing, or (b) to print/send a post-execution portfolio value to Discord. None is compared to a baseline or persisted across runs.

### Persisted equity mechanisms
A repo-wide search (excluding `myvenv/`) returned zero matches for `portfolio_snapshot`, `MAX_DAILY_LOSS`, `DAILY_LOSS_LIMIT`, `session_start_equity`, `baseline_equity`, `circuit_breaker`, `kill_switch`, and `halt_trading`. No file named `portfolio_snapshot.json` or similar exists.

### Threshold-comparison halt logic
None. The only halt pathways in `live_trader.py::run` are:
- `sys.exit(0)` if market closed (`live_trader.py:601`)
- `sys.exit(1)` if `check_market_data_freshness` fails (`live_trader.py:630`)
- `sys.exit(0)` if `get_signals()` returns empty (`live_trader.py:650`)

No equity-based halt exists anywhere in the module.

### `config.py` constants
The only risk-related constants in `config.py` are `MAX_POSITION_PCT = 0.08` (per-position cap, `config.py:88`) and the per-position `STOP_LOSS_COOLDOWN_DAYS = 7` (`config.py:55`). Neither is a portfolio-level daily loss constraint.

### Summary
**No portfolio-level circuit breaker exists.** No session-start equity is captured, no persisted prior-day equity file is written or read, no daily-loss threshold constant is defined, and no halt comparison is performed at any point in `run()`.

---

## Task 4: Infrastructure Error Handling (Fail-Fast on Alpaca Failure)

### Startup paths
- `get_api()` (`live_trader.py:76-86`) raises `EnvironmentError` when credentials are unset. This is appropriate fail-fast behavior for missing configuration. However, the function does **not** wrap `tradeapi.REST(...)` itself; any construction-time failure will propagate as an uncaught exception.
- `market_is_open()` (`live_trader.py:92-94`) calls `api.get_clock()` without a try/except. A 5xx, timeout, or auth failure here propagates uncaught — the script terminates with a stack trace, and `run_bot.py` will send a generic non-zero-exit Discord alert (per `PIPELINE.md`'s `run_bot.py` description).
- `get_owned_tickers()` (`live_trader.py:306-309`) and `get_equity()` (`live_trader.py:312-313`) also do not handle Alpaca exceptions. They are called many times in `run()` (lines 635, 636, 641, 642, 695, 731, 745, 796, 797, 905); any transient API failure mid-loop crashes the script with no classification.

### Reference halt pattern
`check_market_data_freshness` followed by the block at `live_trader.py:609-630` is the one place the codebase uses a proper categorized halt: it logs a structured `STALE_MARKET_DATA` row, posts a specific Discord alert, and calls `sys.exit(1)`. This pattern is **not reused** for any Alpaca-side failure.

### Exception-handler table (scope)

| file:line | Catches | Behavior | Category | Correct for category? |
|---|---|---|---|---|
| `live_trader.py:65` | `ImportError` | auto-install `alpaca-trade-api` via pip, retry import | startup-dependency | OK |
| `live_trader.py:235` | bare `Exception` on CSV read inside get_signals | silent `pass`; caller checks `_last_price is None` and logs `CSV_INVALID_SKIP` | per-ticker | OK (compensating check exists) |
| `live_trader.py:297` | `Exception as exc` in `get_signals()` ticker loop | `print("  [SKIP] {ticker} — {exc}")`, continue | **Mixed** — catches per-ticker `predict_ticker` errors *and* systemic `StaleMarketDataError` from `features.py` | **INCORRECT** for systemic errors — a Layer 3 tripwire is silently downgraded to a per-ticker skip with no Discord alert |
| `live_trader.py:358` | `Exception as exc` in `place_buy` | `print("  [ERROR] ...")`, return `{"status": "error", "reason": str(exc)}`; caller continues loop | **Mixed** — catches 5xx / 429 / auth / network *and* per-ticker logic errors (insufficient buying power, invalid symbol) | **INCORRECT** for infra errors — no differentiation, session does not halt |
| `live_trader.py:375` | `Exception as exc` in `place_sell` | same as place_buy | **Mixed** — same categories; stop-loss sells use this path (see Task 5) | **INCORRECT** for infra errors |
| `live_trader.py:407` | `ValueError` on log date parse | `continue` | per-row parse | OK |
| `live_trader.py:409` | `OSError` on signal log read | silent `pass` | log I/O | Acceptable but silent |
| `live_trader.py:493` | `Exception as exc` in `send_discord` | print and continue | notification | OK |
| `rebalancer.py:65` | `Exception as exc` fetching equity | print, return empty outcomes list | system | Halts rebalancer only; main bot continues unaware |
| `rebalancer.py:75` | `Exception as exc` fetching positions | print, return empty outcomes list | system | Same as above |
| `rebalancer.py:97` | `Exception as exc` parsing one position's fields | print, continue to next position | per-ticker | OK |
| `rebalancer.py:145` | `Exception as exc` refreshing equity after a trim | print warning, continue with stale equity | **system** — stale equity will skew subsequent weight calculations in the same rebalancer loop | **INCORRECT** — silent degradation of a correctness guarantee |
| `rebalancer.py:158` | `Exception as exc` on `submit_order` | print, log `REBALANCER_ORDER_FAIL`, continue | **Mixed** — 5xx vs. logic error | **INCORRECT** for infra errors |
| `signal_logger.py:138` | `OSError` on CSV write | print error, continue | log I/O | Silent — downstream has no signal that logging failed |
| `outcome_tracker.py:69,135,157,179,206,232,314,345,389` | various | out-of-scope for the live-trading halt question; script runs in a separate GitHub Actions job | — | — |

### Handlers catching infrastructure errors and continuing
The following handlers swallow Alpaca-side 5xx/429/auth/network failures without distinguishing them from per-ticker logic errors, and do not halt the session or post a differentiated Discord alert:

- `live_trader.py:297` (in `get_signals()` — also masks Layer 3 `StaleMarketDataError`)
- `live_trader.py:358` (`place_buy`)
- `live_trader.py:375` (`place_sell`) — reused by stop-loss, take-profit, and model-SELL paths
- `rebalancer.py:145` (post-trim equity refresh — continues with stale equity)
- `rebalancer.py:158` (rebalancer `submit_order`)

### Summary
The bot does **not** distinguish infrastructure-level failures from per-ticker logic failures. Every `submit_order` and Alpaca read in the execution loop either catches `Exception` broadly or does not handle exceptions at all. The only fail-fast halt with Discord alert + structured log row is the market-data freshness gate; no analogous pattern protects the Alpaca call sites. During an Alpaca partial outage, the bot will produce a Discord summary full of `Errors: <ticker> — <network-error>` lines while silently continuing through every signal in the queue.

---

## Task 5: Protective-Action Failure Handling (Missing-Else Pattern)

### Protective-action call-site table

| # | Call site | Discord fires before result check? | Else-branch for failure? | Failure path distinguishable in log? | Severity of silent failure |
|---|---|---|---|---|---|
| 1 | `live_trader.py:450` stop-loss SELL in `check_position_limits` | **Yes** (`send_discord` at lines 451-454 is unconditional, fires before `if result["status"] == "placed"` at line 455) | **No** — only a success branch exists at lines 455-459; implicit drop-through on failure | **No** — on failure, no `log_signal` call, no state update, no `exited.append` | **CRITICAL** |
| 2 | `live_trader.py:463` take-profit SELL in `check_position_limits` | **Yes** (`send_discord` at lines 464-467 fires before `if result["status"] == "placed"` at line 468) | **No** — only a success branch at lines 468-470 | **No** — and additionally, **no `log_signal` is called even on success**; take-profit exits are entirely absent from the signal log | **HIGH** |
| 3 | `live_trader.py:707` model-SELL in the SELL pass | **No** — no per-trade Discord; bulk `send_discord` at line 907 runs after the loop and reads `outcomes[].status` | Yes via ternary (`live_trader.py:708`): `actual_action = "SELL" if placed else "SELL_ERROR"` | **Yes** — log row carries `SELL_ERROR`; `outcomes[]` entry carries `status="error"`; bulk Discord lists it under `Errors:` | **MEDIUM** — distinguishable in log and in bulk summary, but no immediate alert on a model SELL failure (the model believes this position is now risky) |
| 4 | `rebalancer.py:132` rebalancer trim SELL | **No** — no Discord in the rebalancer; live_trader's bulk summary renders `[ORDER FAILED: <reason>]` at `live_trader.py:552` | Yes — explicit `except Exception` at `rebalancer.py:158` logs `REBALANCER_ORDER_FAIL` and appends an `error`-status outcome | **Yes** — log row carries `REBALANCER_ORDER_FAIL`; outcome dict has `status="error"`; bulk Discord differentiates placed vs. failed | **LOW-MEDIUM** — trims act on overweight-but-healthy positions, not on underwater positions; distinguishable but no immediate alert |

### The pattern to flag, concrete instances

**Instance #1 — stop-loss (`live_trader.py:448-459`):**

```python
if loss_pct > stop_pct:
    print(f"  [STOP LOSS]   {ticker} — down {loss_pct:.1f}% — selling {qty} shares")
    result = place_sell(api, ticker, qty)
    send_discord(
        f"🛑 **[STOP LOSS]** {ticker} sold — down {loss_pct:.1f}%  "
        f"(entry ${entry_price:.2f} → current ${current_price:.2f})"
    )
    if result["status"] == "placed":
        from signal_logger import log_signal
        log_signal(ticker, "SELL", current_price, qty, 0.0, "STOP_LOSS_SELL")
        exited.append(ticker)
        owned.pop(ticker, None)
    # no else — failure is silent
```

Consequences of a silent failure here: Discord message claims "sold"; the bot proceeds under the assumption the stop-loss fired; the still-underwater position remains open for ~24h until the next check. `owned` still contains the ticker, so the BUY pass may later *add shares* to a position the operator believes has been exited (via the capital_allocator add-to-position path). The model may also re-issue a BUY signal on the same ticker, which would pass the `exited`-skip check at `live_trader.py:764` because the failed stop-loss never appended to `exited`.

**Instance #2 — take-profit (`live_trader.py:461-470`):** same shape as instance #1, plus no `log_signal` on success. Take-profit exits are invisible in the signal log regardless of success or failure.

**Instance #3 — model SELL (`live_trader.py:707-721`):** failure is recorded distinguishably (`SELL_ERROR`) and surfaces in the bulk Discord summary. No immediate alert, but no silent-success misrepresentation.

**Instance #4 — rebalancer trim (`rebalancer.py:131-169`):** failure is recorded distinguishably (`REBALANCER_ORDER_FAIL`) and surfaces in the bulk Discord summary as `[ORDER FAILED: ...]`. No immediate alert.

### Severity ranking
1. **`live_trader.py:448-459` (stop-loss)** — HIGH/CRITICAL. Protective action with capital-loss exposure. Misleading Discord "sold" message + no log row + no state update on failure.
2. **`live_trader.py:461-470` (take-profit)** — HIGH. Misleading Discord + no log row on either outcome. Less capital-damaging than a missed stop-loss but still a silent protective-action failure.
3. **`live_trader.py:707-721` (model SELL)** — MEDIUM. Distinguishable in log and bulk summary; no immediate alert on an arguably risk-relevant failure.
4. **`rebalancer.py:131-169` (rebalancer trim)** — LOW-MEDIUM. Distinguishable in log and bulk summary; trims are weight-rebalancing rather than loss-cutting, so silent failure is less dangerous.

---

## Summary Table

| Task | Status | Severity | Notes |
|------|--------|----------|-------|
| 1. Position-level realized P&L logging | **GAP** | HIGH | Schema tracks signals, not positions. No entry_price, exit_price, exit_reason, realized_dollar_pnl, realized_percent_return, or days_held. Take-profit exits are entirely unlogged. |
| 2. Standing stop-loss orders (bracket/OCO) | **GAP** | HIGH | `place_buy` submits plain market orders with no `order_class`/`stop_loss`/`take_profit`. Protection is a single in-script snapshot at 15:00 UTC; positions are unprotected ~23h per day and ~72h across weekends. |
| 3. Portfolio-level daily loss limit | **GAP** | HIGH | No session-start baseline, no persisted equity file, no threshold constant, no halt logic. Explicitly absent. |
| 4. Infrastructure error handling (fail-fast on Alpaca failure) | **PARTIAL** | HIGH | Only the market-data freshness gate uses a categorized halt. `place_buy`, `place_sell`, `rebalancer.submit_order`, and the `get_signals` per-ticker handler catch `Exception` broadly and continue, mixing infra and logic failures. |
| 5. Protective-action failure handling | **PARTIAL** | HIGH | Stop-loss and take-profit fire success-implying Discord before checking `result["status"]` and have no failure branch (CRITICAL / HIGH). Model SELL and rebalancer trim log a distinguishable failure code but without an immediate alert (MEDIUM / LOW-MEDIUM). |
