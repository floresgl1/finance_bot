# Edge Investigation — 2026-09-08

## Why this ran

The promotion gate's first real run rejected a freshly retrained challenger:
BUY F1 0.432 against the 6-month-old champion's 0.490, on the same held-out
window. The challenger had *strictly more recent* training data and lost on
every metric.

That ruled out the premise the retraining work started from — that the model was
underperforming because it was stale — and raised a harder question: **does this
model have edge at all?**

The cheap probe was to vary `HISTORY_PERIOD` and see whether more data helps.

## Method

`edge_probe.py`. Two measurements on one run.

**The naive version of this probe does not work.** Re-running the pipeline with
different `HISTORY_PERIOD` values changes the test window too, because the
70/20/10 split is proportional — 3y of history tests on the last ~3.5 months, 8y
tests on the last ~9. Those numbers describe different market periods and are not
comparable.

So the test window is **held fixed** (2026-05-20 → 2026-08-27, 828 rows — the
window the gate itself used) and only the training start date varies. Every model
is scored on byte-identical rows.

Baselines are scored on the same window, because accuracy and BUY F1 are
uninterpretable without them.

## Results

Test window: 828 rows. Labels: BUY 331 (40.0%), SELL 320 (38.6%), HOLD 177 (21.4%).

### Baselines

| Strategy | Accuracy | BUY F1 | macro F1 |
|---|---|---|---|
| **always BUY** | 0.3998 | **0.5712** | 0.1904 |
| always HOLD | 0.2138 | 0.0000 | 0.1174 |
| stratified random | 0.3551 | 0.3814 | 0.3345 |

### Lookback sweep

| Training lookback | Train rows | Accuracy | BUY F1 | macro F1 |
|---|---|---|---|---|
| 1y | 3,012 | **0.4106** | 0.4686 | 0.3573 |
| 2y | 6,012 | 0.3684 | 0.4330 | 0.3231 |
| 3y (current) | 9,012 | 0.3841 | 0.4463 | 0.3416 |
| 5y | 15,060 | 0.3732 | 0.4132 | 0.3472 |
| 8y | 24,120 | 0.3756 | 0.4615 | 0.3520 |

Production champion on the same window, for reference: accuracy 0.406,
BUY F1 0.490.

## Findings

### 1. More data does not help. The constraint is features or labels.

8× the training data (3,012 → 24,120 rows) moves accuracy from 0.4106 **down** to
0.3756. There is no trend in either direction — the numbers bounce between 0.368
and 0.411 with no relationship to training-set size. The *smallest* training set
scores best.

Extending `HISTORY_PERIOD` is not worth doing. Whatever is limiting this model,
it is not how much history it sees.

### 2. BUY F1 — the metric the gate originally used — is gamed by class balance.

**A constant "always BUY" prediction scores BUY F1 0.5712.** That beats every
model in the sweep (best: 0.4686) and beats the production champion (0.490).

A model can only score high BUY F1 here by predicting BUY often, and 40% of the
labels are BUY, so recall comes cheap. The metric was measuring the label
distribution more than the model.

This is why the promotion gate now compares **simulated total return** instead —
see PIPELINE.md. Any future metric proposed for the gate should be checked
against `always_BUY` first.

### 3. The model has weak but non-zero discriminative signal.

Being fair to it: on **macro F1** — which weights all three classes equally and
so cannot be gamed by predicting the majority — the models score 0.32–0.36
against 0.3345 for a stratified random guess and 0.1904 for always-BUY. The best
configuration beats random by +0.023.

So the model is not noise. It is separating the three classes slightly better
than chance. But the margin is thin, and on the BUY class specifically — the only
class the bot acts on — it does not beat a constant prediction.

### 4. The label distribution is worth questioning.

Only 21.4% of rows are HOLD. The VIX-based thresholds (1.0–2.0% relative 7-day
move) are being exceeded ~79% of the time, which makes the problem nearly binary
and hands any majority-guessing strategy a high floor.

That is a labeling choice, not a fact about markets. A wider HOLD band would
produce fewer, higher-conviction BUY/SELL labels and a harder-to-game baseline.

## Recommended next steps

Ordered by expected information per unit of effort.

1. **Benchmark the strategy against buy-and-hold.** Classification metrics are a
   detour; the question that matters is whether the bot beats holding SPY (or an
   equal-weight basket of the watchlist) over the same period, net of costs.
   `backtest.py` already simulates the strategy — it needs a buy-and-hold arm.
   If the strategy does not beat buy-and-hold, no amount of model tuning matters.

2. **Widen the HOLD band and re-measure.** Cheap: change the thresholds in
   `labels.py`, re-run `edge_probe.py`. If a stricter label improves macro F1 and
   drops the constant-predictor baseline, the labels were the problem.

3. **Feature ablation.** 20 features, all technical/market-context, all derived
   from price. Measure which carry signal. `predictor.py` already computes SHAP
   values per prediction — aggregate them across the test window rather than
   guessing.

4. **Only then consider new feature families.** Sentiment already exists as
   dormant infrastructure (deliberately a veto, not a trainable feature). If
   price-derived features are exhausted, that is the obvious place to look next.

## Reproducing

```bash
python edge_probe.py                            # default: 1/2/3/5/8y sweep
python edge_probe.py --test-start 2026-05-20    # pin the window
python edge_probe.py --lookbacks 1 3 5 10
python edge_probe.py --refresh                  # re-download history
```

Writes `data/edge_probe_results.json`. Downloads into `data/edge_probe/` rather
than `data/`, so the live CSVs the next real training run reads are untouched.
