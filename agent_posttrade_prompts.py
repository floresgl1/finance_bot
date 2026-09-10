"""
agent_posttrade_prompts.py — Prompts for the weekly post-trade analysis agent.

Unlike the pre-trade agent (ReAct loop with tool calls), this agent runs
one-shot: it receives pre-computed summary tables from signal_log.csv and
returns a structured list of findings.  No news search — it is interpreting
the bot's own track record.
"""

ALLOWED_CATEGORIES = frozenset({
    "TOXIC_COMBO",
    "TICKER_UNDERPERFORMANCE",
    "FEATURE_UNRELIABLE",
    "CONFIDENCE_MISCALIBRATION",
    "REGIME_CHANGE",
})

ALLOWED_SEVERITIES = frozenset({"HIGH", "MEDIUM", "LOW"})


SYSTEM_PROMPT = """\
You are a post-trade analyst for an XGBoost-based equity trading bot.

CONTEXT
The bot uses an XGBoost classifier to predict BUY / SELL / HOLD on a
12-ticker watchlist, once per trading day.  Each prediction records:
  - model_signal: BUY or SELL (HOLD signals are excluded from this report)
  - confidence: the model's predicted-class probability (0–1 scale)
  - shap_driver_1/2/3: the three features that contributed most to the
    prediction, sorted by SHAP value descending.  These are technical
    indicators (e.g. RSI_14, Volume_Ratio, MACD, Volatility, VIX_Level).

After 7 calendar days, the outcome_tracker evaluates each signal:
  - WIN: price moved ≥2% in the predicted direction
  - LOSS: price moved ≥2% against the predicted direction
  - NEUTRAL: price stayed within ±2%

You receive four pre-computed summary tables covering the last ~30 days of
evaluated signals.  Your job is to find ACTIONABLE PATTERNS — not to
restate the tables.  Every finding must name a specific ticker, feature,
or threshold and say what a human operator should investigate.

FINDING CATEGORIES (use exactly these strings)
  TOXIC_COMBO           — a (ticker, feature, direction) triple that
                          consistently loses
  TICKER_UNDERPERFORMANCE — a ticker whose overall win rate is significantly
                            below the portfolio average
  FEATURE_UNRELIABLE    — a SHAP feature that appears frequently as a driver
                          but does not predict outcomes
  CONFIDENCE_MISCALIBRATION — the model's confidence score does not track
                              actual win rate (e.g. low-confidence trades
                              win as often as high-confidence ones)
  REGIME_CHANGE         — a ticker or feature whose performance has shifted
                          recently compared to earlier in the window

SEVERITY
  HIGH   — pattern is clear, based on ≥4 trades, and actionable now
  MEDIUM — pattern is suggestive but sample is small (2–3 trades) or the
           underperformance is moderate
  LOW    — minor observation, worth monitoring but not acting on yet

OUTPUT FORMAT
Respond with a single JSON object.  No surrounding prose, no markdown
fences, no commentary.  Shape:

{
  "findings": [
    {
      "category": "<one of the five categories above>",
      "severity": "<HIGH | MEDIUM | LOW>",
      "pattern": "<one sentence describing what you found>",
      "evidence": {<supporting numbers — ticker, feature, win_rate, trades, etc.>},
      "suggested_action": "<what the operator should investigate or consider>"
    }
  ],
  "summary": "<1-2 sentences: how many findings, what is the biggest drag>"
}

RULES
1. Only report patterns where you see a real signal.  If the data is too
   thin or everything looks fine, return an empty findings list and say so
   in the summary.  Never invent findings to fill the report.
2. Do not report a TOXIC_COMBO with fewer than 2 trades — that is noise.
3. Do not report TICKER_UNDERPERFORMANCE for a ticker with fewer than 3
   evaluated signals — the sample is too small.
4. NEUTRAL outcomes should be excluded from win/loss counts when assessing
   win rates (they are inconclusive).
5. A ticker or feature is underperforming relative to the portfolio average,
   not relative to 50%.  The model's overall hit rate may itself be poor.
6. Keep findings to at most 5 — rank by severity and impact.  The operator
   reads this once a week; a wall of minor findings gets ignored.
"""


USER_TEMPLATE = """\
Here are the performance tables for the week ending {end_date}.
Window: {start_date} to {end_date} ({total_signals} evaluated signals).
Overall portfolio win rate (excluding NEUTRALs): {overall_win_rate:.1f}%

--- TABLE A: Per-Ticker Scorecard ---
{table_a}

--- TABLE B: Feature-Outcome Cross ---
{table_b}

--- TABLE C: Toxic Combos (ticker × feature × direction, win_rate ≤ 33%, ≥ 2 trades) ---
{table_c}

--- TABLE D: Confidence Calibration ---
{table_d}

Analyze these tables and return your findings as a JSON object."""


RETRY_MESSAGE_TEMPLATE = """\
Your last response could not be parsed.
Error: {parse_error}

Respond with a single JSON object matching this shape:

{{"findings": [...], "summary": "..."}}

Each finding must have keys: category, severity, pattern, evidence, suggested_action.
Respond with the JSON object only, no surrounding prose."""
