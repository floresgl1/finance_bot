"""
Configuration for the stock analysis bot.

Defines the watchlist of ticker symbols, data settings (lookback period,
train/test split), feature thresholds, and labeling parameters (buy/sell
return thresholds and forward-looking horizon).
"""

import os
from datetime import datetime, timezone

# Tickers the bot will fetch data for, train on, and generate recommendations for
WATCHLIST = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'JPM', 'XOM', 'JNJ',
             'META', 'NFLX', 'PYPL', 'INTC']

# Number of calendar days ahead to measure the return used for labeling and prediction
PREDICTION_DAYS = 7

# Maximum age (calendar days) of a ticker's price CSV before it is considered stale
# and skipped during signal generation (logged as STALE_SKIP).
STALE_DAYS = PREDICTION_DAYS - 2

# How far back to pull historical data from yfinance (e.g. "1y", "3y", "5y")
HISTORY_PERIOD = "3y"

# BUY_THRESHOLD and SELL_THRESHOLD have been replaced by dynamic volatility-adjusted
# thresholds in labels.py (dynamic_threshold = df["Volatility"] * 2). Fixed thresholds
# are no longer used for labeling.
# BUY_THRESHOLD = 0.04
# SELL_THRESHOLD = -0.04

# Directory where downloaded price CSVs are cached
DATA_DIR = "data/"

# Directory where trained model files are saved and loaded from
MODEL_DIR = "models/"

# Filename for the saved XGBoost model bundle (model + label encoder)
MODEL_FILENAME = "XG_Boost.joblib"

# XGBoost hyperparameters used by trainer.py
XGB_PARAMS = {
    "n_estimators":      150,
    "learning_rate":     0.1,
    "max_depth":         5,
    "subsample":         0.8,
    "use_label_encoder": False,
    "eval_metric":       "mlogloss",
}

# Minimum predicted-class probability required to emit a BUY or SELL signal.
# Predictions below this threshold are overridden to HOLD.
# Random chance for 3 classes is 0.33; 0.45 requires meaningful conviction.
CONFIDENCE_THRESHOLD = 0.35

# Number of calendar days after a STOP_LOSS_SELL during which a BUY on the
# same ticker is blocked (logged as COOLDOWN_SKIP).
STOP_LOSS_COOLDOWN_DAYS = 7

# Standing stop-loss (OTO) configuration
STOP_LOSS_PCT              = 0.10   # 10% below entry; was hardcoded default in check_position_limits
STOP_LOSS_CANCEL_TIMEOUT_S = 5      # max seconds to wait for Alpaca to process a cancel
STOP_LOSS_POLL_INTERVAL_S  = 0.2    # seconds between status polls when awaiting cancel

# New signal-logger action codes introduced by OTO stop-loss integration
CANCEL_STOP_FAILED = "CANCEL_STOP_FAILED"
STOP_BACKFILL      = "STOP_BACKFILL"
STOP_BACKFILL_FAILED = "STOP_BACKFILL_FAILED"
TAKE_PROFIT_FAILED = "TAKE_PROFIT_FAILED"

# Logged when a BUY cannot be sized to even 1 share due to insufficient equity.
INSUFFICIENT_EQUITY = "INSUFFICIENT_EQUITY"

# Logged when a ticker's price CSV is older than STALE_DAYS calendar days.
STALE_SKIP = "STALE_SKIP"

# Logged when a ticker's CSV is completely missing or its Close price cannot be read.
CSV_INVALID_SKIP = "CSV_INVALID_SKIP"

# Gates which signal_log rows the news-validation agent will pick up. Only
# rows whose actual_action represents a real model-driven trade attempt are
# eligible — skip codes (NOT_OWNED, INSUFFICIENT_EQUITY, REBALANCER_TICKERS_SKIP,
# etc.) carry no SHAP values and would force the agent into generic fallback
# queries with no thesis grounding. ADD_TO_POSITION is included because it is
# a real BUY-shaped decision the agent can validate; SELL is included because
# it is a real model-driven exit. Plain HOLD is excluded — the agent only
# validates BUY/SELL theses.
AGENT_ELIGIBLE_ACTIONS = {"BUY", "ADD_TO_POSITION", "SELL"}

# Logged when capital_allocator receives a price of zero or negative, making share
# sizing impossible.
INVALID_PRICE_SKIP = "INVALID_PRICE_SKIP"

# Logged when capital_allocator receives a portfolio_value of zero or negative,
# preventing weight and headroom calculations (distinct from INVALID_HEADROOM).
INVALID_PORTFOLIO_SKIP = "INVALID_PORTFOLIO_SKIP"

# Logged when capital_allocator receives shares_owned of zero or negative, meaning
# there is no existing position to add to — bypasses the position cap check.
INVALID_SHARES_SKIP = "INVALID_SHARES_SKIP"

# Logged when a BUY is skipped because the rebalancer already trimmed that ticker
# in the current session (model SELL → rebalancer skip → BUY skip chain).
REBALANCER_TICKERS_SKIP = "REBALANCER_TICKERS_SKIP"

# Market data freshness check (Layer 1 in live_trader.py)
MAX_MARKET_DATA_AGE_HOURS = 24
STALE_MARKET_DATA = "STALE_MARKET_DATA"
STALE_MARKET_DATA_SKIP = "STALE_MARKET_DATA_SKIP"

# --- Daily run guard (owned by live_trader.py, read by run_bot.py) ---
# Absolute path, anchored to this file's directory, so the guard resolves to the
# same file no matter which working directory the PythonAnywhere task or the
# run_bot.py subprocess is started from.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LAST_RUN_GUARD_PATH = os.path.join(_REPO_ROOT, "data", "last_run_date.txt")


def today_utc() -> str:
    """
    Today's date in UTC as an ISO 'YYYY-MM-DD' string.

    Every date the bot persists is UTC-based — signal_logger rows, the run
    guard, and the portfolio snapshot — so that a session running near local
    midnight cannot stamp two different dates for the same trading day, and so
    that stamps written on the server compare correctly against each other.
    """
    return datetime.now(timezone.utc).date().isoformat()


# Portfolio-level daily loss limit (Finding #3)
PORTFOLIO_SNAPSHOT_PATH        = "portfolio_snapshot.json"
HALT_FLAG_PATH                 = "HALT_FLAG.txt"
MAX_SINGLE_DAY_LOSS_PCT        = 0.05   # 5% session-over-session drawdown halt
MAX_ROLLING_LOSS_PCT           = 0.08   # 8% rolling drawdown halt
ROLLING_LOSS_WINDOW_DAYS       = 5      # trading sessions to look back
PORTFOLIO_SNAPSHOT_RETAIN_DAYS = 30     # cap snapshot file growth

# Signal-logger action codes introduced by portfolio halt
PORTFOLIO_HALT_SINGLE_DAY      = "PORTFOLIO_HALT_SINGLE_DAY"
PORTFOLIO_HALT_ROLLING         = "PORTFOLIO_HALT_ROLLING"
HALT_FLAG_PRESENT              = "HALT_FLAG_PRESENT"
BUY_SKIPPED_HALT               = "BUY_SKIPPED_HALT"
REBALANCER_SKIPPED_HALT        = "REBALANCER_SKIPPED_HALT"

# Maximum fraction of total portfolio value allowed in any single position.
# Used by capital_allocator.py to compute headroom before adding to a position.
MAX_POSITION_PCT = 0.08

# Maximum fraction of the portfolio invested at once.
#
# RISK NOTE: the live path does not enforce this. live_trader.py and
# capital_allocator.py contain no portfolio-level exposure check at all — the
# only bound is MAX_POSITION_PCT per name, so twelve full positions reach ~96%
# invested. 1.00 is therefore the *faithful* value: it makes backtest.py model
# what the bot actually does, rather than a stricter strategy that is not
# deployed.
#
# It lives here rather than in backtest.py because that file previously
# declared its own MAX_POSITION_PCT = 0.20 and MAX_TOTAL_EXPOSURE = 0.80,
# silently simulating a concentrated four-position book while the live bot ran
# twelve names at 8%. Every simulated return figure produced before 2026-09-08
# describes that other strategy.
#
# If a real cap is ever wanted, lowering this is NOT sufficient — live_trader.py
# must enforce it too, or the simulation and the bot diverge again.
MAX_TOTAL_EXPOSURE = 1.00

# Minimum model confidence (0–1 scale) required to add shares to an already-owned
# position.  Signals below this threshold keep the existing "already owned" skip.
ADD_TO_POSITION_CONFIDENCE = 0.40

# Confidence tier thresholds
ADD_TO_POSITION_CONFIDENCE_SMALL  = 0.40
ADD_TO_POSITION_CONFIDENCE_NORMAL = 0.50
ADD_TO_POSITION_CONFIDENCE_LARGE  = 0.65

# Confidence tier position sizes
SMALL_POSITION_PCT  = 0.03
NORMAL_POSITION_PCT = 0.05
LARGE_POSITION_PCT  = 0.07

# --- Edge degradation monitor (edge_monitor.py) ---
EDGE_BREAKEVEN_HIT_RATE = 0.31        # below this → alert (from 2026-04-19 baseline)
EDGE_STALENESS_DAYS     = 10          # if most recent WIN/LOSS evaluation older than this → stale alert
EDGE_WINDOW_SIZE        = 30          # rolling window over last N evaluated BUYs
EDGE_MONITOR_STATE_PATH = "data/edge_monitor_state.json"

# --- Model promotion gate (promote_model.py) --------------------------------
# A retrained "challenger" never replaces the live "champion" on the strength of
# being newer. It must beat the champion on the same held-out test set, by a
# margin, and clear absolute floors. Every threshold below is a reason to REJECT;
# the gate defaults to keeping the incumbent whenever a check cannot be made.

CANDIDATE_MODEL_FILENAME = "XG_Boost_candidate.joblib"
MODEL_ARCHIVE_DIR        = os.path.join(MODEL_DIR, "archive")

# --- Head-to-head: simulated dollars ---------------------------------------
# The gate compares models by backtested total return on the shared test split,
# not by BUY F1. F1 is a proxy for money and can move the opposite way: a model
# can improve F1 while trading worse, because F1 weights every BUY equally
# whereas P&L weights them by how much they made or lost. backtest.py already
# applies slippage and commission, so the comparison is net of costs.
#
# A challenger cannot be judged on REALIZED P&L — it has never traded. Both
# sides are therefore simulated over identical data. Realized P&L (pnl_report.py)
# measures the champion in production; this measures a candidate before it gets
# there.
#
# Percentage points of total return the challenger must add over the champion.
PROMOTION_MIN_RETURN_IMPROVEMENT_PCT = 1.0

# Below this many simulated trades the return figure is one or two lucky
# positions rather than a strategy, and the gate refuses to decide on it.
PROMOTION_MIN_BACKTEST_TRADES = 15

# --- Head-to-head: against doing nothing ------------------------------------
# Beating the champion is not enough. A champion that loses to holding the
# basket can be beaten by a challenger that also loses to holding it, and the
# gate would promote — ratcheting between models that are all worse than no
# model at all. The edge investigation (docs/EDGE_INVESTIGATION_2026-09-08.md,
# findings A/E/F) found exactly that situation: every configuration tested lost
# to an equal-weight hold of the same watchlist over the same dates.
#
# Percentage points of total return the challenger must add over buy-and-hold
# on the shared test split. 0.0 means "must at least match holding". Raise it to
# demand a margin for the operational risk of running a bot at all.
PROMOTION_MIN_HOLD_DELTA_PCT = 0.0

# Reject a challenger whose simulated drawdown is worse than this, even if its
# total return is higher. A model that earns more by risking ruin is not an
# improvement.
PROMOTION_MAX_DRAWDOWN_PCT = -35.0

# BUY F1 is still recorded and reported for context, but no longer gates.
# Retained so the constant's absence does not silently change old behaviour.
PROMOTION_MIN_BUY_F1_IMPROVEMENT = 0.01

# Absolute floors the challenger must clear regardless of how poor the champion
# looks. Guards against promoting a bad model simply because the incumbent
# decayed further.
PROMOTION_MIN_BUY_PRECISION = 0.35
PROMOTION_MIN_BUY_RECALL    = 0.10

# Minimum number of true BUY rows in the test set. Below this the comparison is
# noise and the gate refuses to decide rather than guessing.
PROMOTION_MIN_TEST_BUY_SUPPORT = 30

# Retention for models/archive/. Superseded champions are kept so a bad
# promotion can be rolled back by hand.
MODEL_ARCHIVE_RETAIN = 10

# Where promote_model.py records what it decided. The candidate file is
# consumed on both outcomes (moved on promotion, deleted on rejection), so CI
# cannot infer the result from the filesystem and reads this instead.
PROMOTION_DECISION_PATH = os.path.join(MODEL_DIR, "promotion_decision.json")

# Technical indicator columns used as model input features.
# Must stay in sync with the columns produced by features.add_features().
FEATURE_COLUMNS = [
    "SMA_20", "SMA_50",
    "RSI_14",
    "MACD", "MACD_signal",
    "BB_upper", "BB_middle", "BB_lower",
    "Volume_Ratio",
    "Daily_Return",
    "Volatility",
    "Return_20d",
    "Return_60d",
    "SPY_Return",
    "Rel_Strength",
    "Sector_Return_5d",
    "Sector_Return_20d",
    "Stock_vs_Sector",
    "VIX_Level",
    "VIX_Change"
]
