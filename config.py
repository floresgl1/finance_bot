"""
Configuration for the stock analysis bot.

Defines the watchlist of ticker symbols, data settings (lookback period,
train/test split), feature thresholds, and labeling parameters (buy/sell
return thresholds and forward-looking horizon).
"""

# Tickers the bot will fetch data for, train on, and generate recommendations for
WATCHLIST = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'JPM', 'XOM', 'JNJ',
             'META', 'NFLX', 'PYPL', 'INTC']

# Number of calendar days ahead to measure the return used for labeling and prediction
PREDICTION_DAYS = 7

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

# Maximum fraction of total portfolio value allowed in any single position.
# Used by capital_allocator.py to compute headroom before adding to a position.
MAX_POSITION_PCT = 0.08

# Minimum model confidence (0–1 scale) required to add shares to an already-owned
# position.  Signals below this threshold keep the existing "already owned" skip.
ADD_TO_POSITION_CONFIDENCE = 0.40

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
    "VIX_Change",
    "sent_score_daily",
    "sent_rolling_20d",
]
