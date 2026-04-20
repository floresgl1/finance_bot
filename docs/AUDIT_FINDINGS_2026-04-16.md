# Silent Failure Audit — 2026-04-16

Audit scope: features.py, predictor.py, live_trader.py, capital_allocator.py
Audit pattern classes: 1–6 (see prompt for details)

---

## Summary
- Total findings: 18
- CRITICAL: 0, HIGH: 7, MEDIUM: 7, LOW: 4

---

## features.py

### Finding 1 — [HIGH] `reindex + ffill` silently truncates the dataset when market series start after the ticker series
**Location**: `add_features()` ~line 153  
**Pattern**: 3  
**Description**: `spy_close`, `sector_close`, and `vix_close` are aligned to the
ticker's DatetimeIndex via `.reindex(df.index, method="ffill")`. If any market series
starts later than the ticker's earliest date — or has a leading gap — those rows
receive NaN. `ffill` cannot fill leading NaN (there is no prior value to propagate).
The final `df.dropna(inplace=True)` call at the end of `add_features` silently removes
every affected row without logging how many were dropped or why. The returned DataFrame
is shorter than the raw CSV with no warning. For a recently-added sector ETF with less
history than the tickers it covers, this could silently shorten every ticker in that
sector's training set.  
**Reproducibility**: Point a ticker's CSV start date earlier than the corresponding
sector-ETF CSV start date. Call `load_and_process(ticker)`. The returned DataFrame is
noticeably shorter; nothing in stdout indicates rows were dropped.  
**Known?**: Yes — backlog: "features.py: reindex ffill leading NaN truncates training set (HIGH)"

---

### Finding 2 — [MEDIUM] `pct_change(periods=7)` on `SPY_Return` measures 7 trading rows, not 7 calendar days
**Location**: `add_features()` ~line 154  
**Pattern**: 3  
**Description**: `df["SPY_Return"] = spy_close.pct_change(periods=7)` counts 7 rows in
the aligned DataFrame. Because weekend and holiday rows are absent, 7 rows equals
roughly 9–11 calendar days depending on the week. The feature is named and treated as
a 7-day return but silently represents a longer window. The same window mismatch applies
to `Sector_Return_5d` (5 rows ≠ 5 calendar days) and `VIX_Change` (5 rows). The model
is trained on this definition, so the mismatch is internally consistent — but a feature
named `SPY_Return` that silently covers ~10 calendar days can produce misleading SHAP
interpretations and breaks assumptions in any code that compares the feature to an
external 7-day return benchmark.  
**Reproducibility**: Compute `spy_close.pct_change(periods=7)` for a Monday entry.
The 7-row lookback crosses two weekends and covers ~11 calendar days.  
**Known?**: Yes — backlog: "features.py: SPY_Return pct_change(7) trading-day vs.
calendar-day mismatch (MEDIUM)"

---

### Finding 3 — [HIGH] `@lru_cache` on `_load_market_close` serves stale data in long-lived processes
**Location**: `_load_market_close()` ~line 38 (decorator)  
**Pattern**: 5  
**Description**: `_load_market_close` is decorated with `@lru_cache(maxsize=None)`.
Once a symbol is loaded, the resulting `pd.Series` is held in memory for the lifetime of
the Python process. In `live_trader.py` the process is short-lived (single trading run),
so this is benign there. In `finance_bot_dash.py` (Streamlit), the process runs
continuously across all user sessions. If the underlying `data/market/*.csv` files are
refreshed after the dashboard first loads them (e.g., the daily GitHub Actions job runs
while the dashboard is serving requests), the cache continues serving the previous day's
data with no indication. The Layer 3 tripwire added to `_load_market_close` fires only
on the first call per symbol — subsequent calls that hit the cache bypass the freshness
check entirely.  
**Reproducibility**: Start the Streamlit dashboard so it loads and caches SPY data.
Overwrite `data/market/SPY.csv` with newer data. Trigger a fresh prediction in the
dashboard. `_load_market_close("SPY")` returns the stale cached Series; the tripwire
never re-fires.  
**Known?**: No

---

### Finding 4 — [MEDIUM] `pd.to_numeric(errors='coerce')` in `load_and_process` drops rows without logging
**Location**: `load_and_process()` ~lines 281–284  
**Pattern**: 3  
**Description**: Every column in the raw OHLCV DataFrame is converted via
`pd.to_numeric(df[col], errors="coerce")`, silently turning any non-numeric value into
`NaN`. The immediately following `df.dropna(subset=["Open","High","Low","Close","Volume"])`
then removes all rows where any price/volume field is `NaN`. No count of coerced values
or dropped rows is logged. If a yfinance download produces a partially malformed CSV
(e.g., header rows repeated mid-file, adjustment artifacts written as strings, a
corrupted append), potentially dozens of rows could be silently removed. The model
proceeds on a shorter dataset with no warning.  
**Reproducibility**: Inject a row with `Close` equal to the string `"N/A"` into a
ticker's CSV. `load_and_process` reads it, coerces `"N/A"` to `NaN`, drops the row,
and returns a DataFrame one row shorter than expected — with no print or log message.  
**Known?**: No

---

### Finding 5 — [MEDIUM] `_add_sentiment_features` reads from a path that does not exist; its output columns are absent from `FEATURE_COLUMNS`
**Location**: `_add_sentiment_features()` ~line 237  
**Pattern**: 1, 6  
**Description**: The function constructs `sent_path = os.path.join(DATA_DIR, f"sentiment_{ticker}.csv")`,
resolving to `data/sentiment_AAPL.csv`. The actual consolidated sentiment file lives at
`data/sentiment/sentiment_scores.csv` (referenced by `sentiment_collector.py` and
described in project memory). The per-ticker files almost certainly do not exist, so
`os.path.exists(sent_path)` is always `False`, and the function silently fills
`sent_score_daily` and `sent_rolling_20d` with `0.0` for every ticker on every row.
Compounding this, neither `sent_score_daily` nor `sent_rolling_20d` appears in
`FEATURE_COLUMNS` in `config.py` (which has 18 purely technical features), so the
computed columns are never read by the model. The function runs on every prediction
pass, allocates memory, and produces output that is silently discarded. If a developer
adds either column name to `FEATURE_COLUMNS` expecting real sentiment data, the model
trains and predicts on all-zero inputs with no error.  
**Reproducibility**: Call `load_and_process("AAPL")` and inspect the returned DataFrame.
Columns `sent_score_daily` and `sent_rolling_20d` are present and uniformly `0.0`
regardless of actual news sentiment, because the path `data/sentiment_AAPL.csv` does
not exist.  
**Known?**: Partially — backlog flags "features.py: dead compute and cached Series
missing copy() (LOW)" but does not identify the specific path mismatch
(`data/sentiment_TICKER.csv` vs. `data/sentiment/sentiment_scores.csv`) or the
column-name mismatch with `FEATURE_COLUMNS`.

---

### Finding 6 — [LOW] `@lru_cache` returns the same mutable `pd.Series` object on every call
**Location**: `_load_market_close()` ~line 38 (decorator), returns at ~line 86  
**Pattern**: 5  
**Description**: `lru_cache` stores and returns the exact same `pd.Series` object on
every cache hit. Any caller that mutates the Series in-place (e.g., via `__setitem__`,
`fillna(inplace=True)`, or index assignment) would corrupt the cached object, and every
subsequent call for that symbol would receive the mutated data with no error. Current
callers use `.reindex()` which produces a new object, so there is no known active
failure path — but there is no defensive `.copy()` to enforce the immutability
contract.  
**Reproducibility**: `s1 = _load_market_close("SPY"); s1.iloc[0] = 0.0; s2 = _load_market_close("SPY")` —
`s2.iloc[0]` is `0.0`, demonstrating that `s1` and `s2` are the same object.  
**Known?**: Yes — backlog: "features.py: dead compute and cached Series missing copy() (LOW)"

---

### Finding 7 — [LOW] `FileNotFoundError` from `_load_market_close` loses its actionable message when propagated through `get_signals`
**Location**: `_load_market_close()` ~line 53  
**Pattern**: 1  
**Description**: A missing market CSV raises `FileNotFoundError` with a clear message
directing the operator to run `market_data_collector.py`. However, this exception
propagates up through `add_features` → `load_and_process` → `predict_ticker`, where it
is caught by the generic `except Exception as exc` in `get_signals()`. That handler
prints only `[SKIP] ticker — [Errno 2] No such file or directory: ...` — stripping the
actionable instruction entirely. The operator sees a cryptic OS error for every ticker,
with no hint that the root cause is a missing market file solvable by running one
command.  
**Reproducibility**: Delete `data/market/SPY.csv` and call `get_signals()` inside
`live_trader.run()`. Every ticker prints `[SKIP] ... — [Errno 2] No such file...`.
The original `FileNotFoundError` message with `Run: python market_data_collector.py`
is never shown.  
**Known?**: Yes — backlog: "features.py: missing market file FileNotFoundError hides
root cause (LOW)"

---

## predictor.py

### Finding 8 — [HIGH] XGBoost silently accepts NaN features and produces predictions without warning
**Location**: `predict_ticker()` ~line 101  
**Pattern**: 4  
**Description**: `X = latest[FEATURE_COLUMNS].values.reshape(1, -1)` passes the feature
vector directly to `model.predict_proba(X)` with no NaN check. XGBoost is NaN-tolerant
by design: it routes NaN values through surrogate splits trained on non-NaN data,
returning a well-formed probability vector with no error, warning, or indicator in the
return value. If a feature is NaN on the latest row — possible if a rolling window just
became populated, if the most recent row survived `dropna` in a non-OHLCV column, or if
a merge introduced NaN — the model produces a signal that is structurally indistinguishable
from a clean prediction. The NaN source is never logged, and the SHAP values computed
on the same row will also reflect the imputed path, making post-hoc analysis misleading.  
**Reproducibility**: Take the latest feature row for any ticker, set one feature to
`np.nan`, and call `model.predict_proba`. The call returns probabilities; nothing in
the output indicates any feature was missing.  
**Known?**: Yes — backlog: "predictor.py: NaN accepted silently by XGBoost (HIGH)"

---

### Finding 9 — [HIGH] `is_ticker_stale` returns `is_stale=False` for missing files, contradicting its own docstring
**Location**: `is_ticker_stale()` ~line 50  
**Pattern**: 4  
**Description**: When `os.path.exists(path)` returns `False`, the function immediately
returns `(False, -1)` — indicating the ticker is **not** stale. The docstring explicitly
states "Parse/IO failures are treated as stale so the bot never trades on bad data,"
directly contradicting this behavior. In `live_trader.get_signals()`, an `is_stale=False`
result bypasses the stale-ticker branch entirely and proceeds to call `predict_ticker`,
which calls `load_and_process`, which raises `FileNotFoundError`. That exception is
caught by the generic `except Exception` handler, which prints `[SKIP] ticker — [Errno 2]...`
and continues — **writing no entry to `signal_log.csv`**. Compare this to the intended
path: when `is_stale=True`, the code attempts to read the last known price and writes
either a `STALE_SKIP` or `CSV_INVALID_SKIP` log entry. A missing file thus leaves a
silent gap in the audit trail, and the `CSV_INVALID_SKIP` count in signal_log.csv
understates actual skip events.  
**Reproducibility**: Delete `data/AAPL.csv` and call `get_signals()`. Inspect
`signal_log.csv` — AAPL has no row. The only indication is a terminal print of
`[SKIP] AAPL — [Errno 2] No such file or directory`.  
**Known?**: No

---

### Finding 10 — [MEDIUM] `get_recent_earnings_surprise` swallows all exceptions, silently disabling the earnings veto
**Location**: `get_recent_earnings_surprise()` ~line 149  
**Pattern**: 1  
**Description**: The entire read-and-parse body is wrapped in `try/except Exception: return None`.
`apply_earnings_veto` treats `None` as "no recent earnings data" and leaves the signal
unchanged. One concrete trigger: if an earnings CSV has a tz-aware `DatetimeIndex`,
the line `df.index = pd.to_datetime(df.index).tz_localize(None)` raises
`TypeError: Already tz-aware, use tz_convert to convert.` This exception is caught
silently, `None` is returned, and a BUY signal on a ticker with a strongly negative
recent earnings surprise passes through the earnings veto without any override. No log
message or print statement is produced. The correct behavior — raising or logging the
parse failure — never fires.  
**Reproducibility**: Create `data/earnings/AAPL.csv` with a tz-aware index
(e.g., `"2025-01-15 00:00:00+00:00"`) and a `Surprise(%)` of `-30.0`. Call
`get_recent_earnings_surprise("AAPL")`. It returns `None`; `apply_earnings_veto`
returns the signal unchanged, and a potential earnings veto is silently skipped.  
**Known?**: No

---

## live_trader.py

### Finding 11 — [HIGH] Take-profit position exits are not logged to `signal_log.csv`
**Location**: `check_position_limits()` ~lines 461–470  
**Pattern**: 4  
**Description**: The stop-loss branch (line ~456) correctly calls
`log_signal(ticker, "SELL", current_price, qty, 0.0, "STOP_LOSS_SELL")` when a
position is sold at a loss. The take-profit branch (`elif gain_pct > take_pct:`) calls
`place_sell` and appends the ticker to `exited` and `owned` — but contains no
`log_signal` call. Take-profit trades execute correctly and appear in the Discord alert,
but are invisible in `signal_log.csv`. Three downstream consequences: (1) `outcome_tracker.py`
never evaluates the trade, silently understating the bot's win rate on profitable exits;
(2) the stop-loss cooldown mechanism reads `STOP_LOSS_SELL` entries from `signal_log.csv`
to block re-entry — since take-profit writes no entry, there is no cooldown after a
profitable exit, and the bot may immediately re-buy the same position on the next run;
(3) portfolio performance analysis derived from `signal_log.csv` cannot attribute value
to take-profit exits.  
**Reproducibility**: Temporarily set `take_pct=0.0` so every long position triggers
take-profit in `check_position_limits`. After a run, inspect `signal_log.csv` — zero
rows for those tickers. The Discord message shows the sale; the log is silent.  
**Known?**: No

---

### Finding 12 — [HIGH] Generic `except Exception` in `get_signals()` drops failed tickers from `signal_log.csv` with no log entry
**Location**: `get_signals()` ~line 204 (the outer try/except per ticker)  
**Pattern**: 4  
**Description**: Each ticker's prediction pipeline is wrapped in
`try/except Exception as exc: print(f"  [SKIP] {ticker} — {exc}")`. Any exception from
`predict_ticker` → `load_and_process` → `add_features` → `_load_market_close`,
including `StaleMarketDataError` from the Layer 3 tripwire added in the previous
commit, is caught here. The only output is a terminal print; no `log_signal` call is
made. The ticker is absent from `signal_log.csv`, indistinguishable from a ticker that
was simply not in WATCHLIST. Concretely: if the Layer 3 tripwire fires for AAPL
(content-stale CSV), `get_signals` prints `[SKIP] AAPL — AAPL: last row 2026-04-09
is 7 days behind today...` but writes nothing to the log. Downstream analysis of
signal_log.csv would not indicate that AAPL was intentionally halted due to stale data
— the same silent gap that concealed the original stale-data incident.  
**Reproducibility**: Introduce any exception in `load_and_process("AAPL")` (e.g.,
rename a CSV column so feature computation fails). Call `get_signals()`. Terminal shows
`[SKIP] AAPL — KeyError: ...`. `signal_log.csv` has no AAPL row.  
**Known?**: No

---

### Finding 13 — [MEDIUM] `run()` never passes `sentiment_df` to `get_signals()`, permanently disabling the sentiment veto in live trading
**Location**: `run()` ~line 647, `get_signals()` ~line 183  
**Pattern**: 4, 6  
**Description**: `get_signals(sentiment_df=None)` applies the sentiment veto only when
`sentiment_df is not None`: `if post_earnings != "HOLD" and sentiment_df is not None: ...`.
In `run()`, the call is `signals = get_signals()` — no `sentiment_df` is passed.
The sentiment block is permanently skipped. The module docstring declares "Veto layers
applied before execution: 1. Sentiment veto (FinBERT)" but this veto never fires in
live trading. `sent_score` defaults to `0.0` for every ticker; the Discord sentiment-veto
alert (`send_discord(...)` inside `get_signals`) never fires; and the `SENTIMENT_VETO`
`actual_action` code is never written by `get_signals`. Changes to `VETO_BUY_THRESHOLD`
and `VETO_SELL_THRESHOLD` in `config.py` have no effect on live trading decisions.
There is also no `load_sentiment_df()` function defined anywhere in `live_trader.py`,
suggesting the parameter was designed for a loading path that was never completed.  
**Reproducibility**: Add `print("VETO FIRED")` inside the `if sentiment_df is not None:`
block of `get_signals()`. Run `live_trader.run()`. The print never appears.  
**Known?**: No

---

### Finding 14 — [HIGH] Empty `signals` list exits with code 0, indistinguishable from a market-closed exit
**Location**: `run()` ~line 648  
**Pattern**: 4  
**Description**: If every ticker in WATCHLIST raises an exception inside `get_signals()`
(e.g., all CSVs are corrupted, all feature computations fail, all market files are
missing), `signals` is returned as an empty list. The guard
`if not signals: print("  No signals generated. Exiting."); sys.exit(0)` exits with
code 0. `run_bot.py` (the launcher) only sends a Discord alert on non-zero exit codes,
so a total prediction failure is invisible: no Discord alert, no `signal_log.csv`
entries, and the exit code matches a normal market-closed exit. An operator monitoring
logs would see "No signals generated. Exiting." — a message that is also printed in
entirely normal conditions — with no indication that all 12 tickers failed.
Note: stale-skip tickers do populate `signals` as `STALE_SKIP` entries, so that
scenario does not hit this path. The gap is specifically when all tickers raise
exceptions (no entries added to `results` in `get_signals`).  
**Reproducibility**: Delete all files in `data/*.csv` and `data/market/*.csv` so that
every `load_and_process` call raises `FileNotFoundError`. Run `live_trader.run()`. The
process exits with code 0; `run_bot.py` sends no alert.  
**Known?**: No

---

## capital_allocator.py

### Finding 15 — [MEDIUM] `ADD_TO_POSITION_CONFIDENCE` and `ADD_TO_POSITION_CONFIDENCE_SMALL` are equal; `_SMALL` is unused in tier logic
**Location**: `check_add_to_position()` ~line 104, `_get_allocation_tier()` ~line 48  
**Pattern**: 6  
**Description**: `config.py` defines both `ADD_TO_POSITION_CONFIDENCE = 0.40` and
`ADD_TO_POSITION_CONFIDENCE_SMALL = 0.40`. The minimum-confidence gate in
`check_add_to_position` uses `ADD_TO_POSITION_CONFIDENCE`; `_get_allocation_tier` uses
`ADD_TO_POSITION_CONFIDENCE_NORMAL` (0.50) and `ADD_TO_POSITION_CONFIDENCE_LARGE`
(0.65) but never references `ADD_TO_POSITION_CONFIDENCE_SMALL`. The lower bound of the
"small" allocation tier is implicitly enforced by the gate in the caller, not by the
tier function. If `ADD_TO_POSITION_CONFIDENCE_SMALL` were changed to 0.45 in
`config.py` (to tighten the small-tier floor), the gate would remain at 0.40 and
confidence values in [0.40, 0.45) would silently proceed to the small tier,
contradicting the intent of the constant.  
**Reproducibility**: Set `ADD_TO_POSITION_CONFIDENCE_SMALL = 0.45` in `config.py`.
Call `check_add_to_position(..., confidence_normalized=0.42, ...)`. The call proceeds
and returns `shares_to_buy > 0` via the small tier — silently ignoring the updated
threshold.  
**Known?**: Yes — backlog: "capital_allocator.py: ADD_TO_POSITION_CONFIDENCE and
ADD_TO_POSITION_CONFIDENCE_SMALL duplicate constants (MEDIUM)"

---

### Finding 16 — [MEDIUM] `current_weight > 1.0` is not clamped; extreme headroom values are indistinguishable from normal INVALID_HEADROOM skips
**Location**: `check_add_to_position()` ~lines 122–133  
**Pattern**: 3, 4  
**Description**: `current_weight = (shares_owned * price) / portfolio_value`. If
`portfolio_value` is abnormally small (e.g., near-zero equity after large unrealised
losses, or a stale equity snapshot from Alpaca returning a low value), `current_weight`
can far exceed 1.0 and `headroom` becomes strongly negative. The function correctly
returns `skip_reason = "INVALID_HEADROOM"` in both the normal case (position is at the
8% cap) and the anomalous case (portfolio value appears to be $1 while holding $5000
of a stock). Both cases produce identical log entries in `signal_log.csv` with no way
to distinguish "position is at the normal cap" from "portfolio data looks wrong."
The anomalous case warrants investigation; the current code treats it identically to a
routine skip.  
**Reproducibility**: Call `check_add_to_position(..., shares_owned=100, price=50.0, portfolio_value=100.0)`.
`current_weight = 50.0`, `headroom = -49.92`. Returns `skip_reason = "INVALID_HEADROOM"` —
the same code as a routine cap-reached skip at `current_weight = 0.085`.  
**Known?**: Yes — backlog: "capital_allocator.py: current_weight>1.0 not clamped (MEDIUM)"

---

### Finding 17 — [LOW] No floor guard: `shares_to_buy = 0` returns with `skip_reason = INSUFFICIENT_EQUITY` but the path is opaque
**Location**: `check_add_to_position()` ~line 142  
**Pattern**: 4  
**Description**: `shares_to_buy = math.floor(buy_amount / price)`. When
`buy_pct * portfolio_value < price` (the allocated dollar amount is less than one
share), `shares_to_buy = 0` and the function returns `skip_reason = INSUFFICIENT_EQUITY`.
The code is functionally correct — the caller handles this. However, the return provides
no indication of the root cause: a tiny portfolio, an expensive stock, a very small
tier percentage, or a combination. The INSUFFICIENT_EQUITY code written to `signal_log.csv`
is identical for "portfolio is depleted" and "position sizing landed on $3 for a $4000
stock." No diagnostic information surfaces.  
**Reproducibility**: Call with `portfolio_value=100.0`, `price=500.0`,
`confidence_normalized=0.65`. `buy_amount = 0.07 × 100 = $7.00`, `shares_to_buy = 0`,
`skip_reason = INSUFFICIENT_EQUITY` — correct outcome but the log gives no hint that the
root cause is a mismatch between tier size and stock price rather than a depleted account.  
**Known?**: Yes — backlog: "capital_allocator.py: missing floor guard and upper bound
on confidence (LOW)"

---

### Finding 18 — [LOW] No upper-bound guard on `confidence_normalized`; values > 1.0 pass silently
**Location**: `check_add_to_position()` ~line 104, `_get_allocation_tier()` ~line 48  
**Pattern**: 6  
**Description**: `confidence_normalized` is expected to be a 0–1 fraction
(`confidence / 100.0` from live_trader). There is no guard for values > 1.0. If a
future refactor passes a raw confidence score (e.g., 85.0 instead of 0.85), the minimum
confidence gate `if confidence_normalized < ADD_TO_POSITION_CONFIDENCE (0.40)` is
trivially passed and `_get_allocation_tier(85.0)` returns `("large", 0.07)` without
error. The allocation tier logic is accidentally insensitive to supra-1.0 inputs because
the tier thresholds are at 0.50 and 0.65 — but the silent acceptance of an out-of-range
input masks a caller bug that could be harder to trace once it reaches a more
sensitive downstream check.  
**Reproducibility**: Call `check_add_to_position(..., confidence_normalized=85.0, ...)`.
No validation error; the function proceeds and returns a "large" tier result silently.  
**Known?**: Yes — backlog: "capital_allocator.py: missing floor guard and upper bound
on confidence (LOW)"
