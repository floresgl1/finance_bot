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

---

# Follow-ups

The four steps recommended below were then carried out. Results follow.

## A. Buy-and-hold benchmark — the gating question

`backtest.py --benchmark <split>`. Two passive arms added: an equal-weight hold
of the watchlist and a SPY hold, both paying the same `SLIPPAGE` and
`COMMISSION` as the strategy and scored through the same `_summarise()` helper,
so the comparison cannot drift.

| Split | Period | Strategy | Equal-weight hold | SPY hold | Δ vs equal-weight |
|---|---|---|---|---|---|
| train | 2023-12-04 → 2025-11-06 | +365.86% | +77.72% | +50.40% | +288.14pp |
| validation | 2025-11-07 → 2026-05-28 | +3.34% | +20.29% | +12.98% | **−16.95pp** |
| test | 2026-05-29 → 2026-09-08 | +5.07% | +3.63% | +1.64% | **+1.43pp** |
| full | 2023-12-04 → 2026-09-08 | +425.95% | +102.46% | +73.69% | +323.49pp |

**The train and full numbers are meaningless.** The model was fitted on those
dates; +365% measures memorisation. `full` is dominated by the same period.
`print_benchmark_comparison()` now prints an explicit IN-SAMPLE banner on both,
because reading them as performance is the easiest possible way to conclude this
strategy works when it does not.

**Answer: out of sample, the strategy does not reliably beat buy-and-hold.**
The two clean windows disagree, and the longer one is badly negative —
validation is ~7 months and the strategy returned +3.34% against +20.29% for
simply holding the basket. The test window (+1.43pp) is 3.5 months and marginal.

One point in the strategy's favour: it averages ~50% invested against 100% for
the hold arms, so it produces those returns at roughly half the market exposure.
That makes the test-split result better than it looks on a risk-adjusted basis.
It does not rescue the validation result.

## B. Label width — does a wider HOLD band help?

`edge_probe.py --label-scales 1 1.5 2 3 4`, multiplying the VIX-derived
thresholds. The number that matters is the **edge over the majority baseline**,
not raw accuracy: widening the band makes HOLD the majority class and
mechanically moves both figures.

| Scale | HOLD % | Model accuracy | Majority baseline | Edge |
|---|---|---|---|---|
| 1.0 (shipped) | 21.4% | 0.3841 | always BUY 0.3998 | **−0.0157** |
| 1.5 | 35.3% | 0.3780 | always HOLD 0.3527 | **+0.0254** |
| 2.0 | 43.0% | 0.3961 | always HOLD 0.4300 | −0.0338 |
| 3.0 | 58.8% | 0.5773 | always HOLD 0.5882 | −0.0109 |
| 4.0 | 71.5% | 0.7053 | always HOLD 0.7150 | −0.0097 |

Two things fall out.

The shipped labelling produces a **negative** edge — the model is worse than a
constant prediction. Widening to 1.5× is the only setting that produces a
positive one.

And scales 3.0–4.0 are an object lesson: accuracy climbs to 0.58 and 0.71, which
looks like a dramatic improvement, while remaining *worse than always saying
HOLD*. Any accuracy figure quoted without its baseline is worthless.

## C. Feature ablation

`edge_probe.py --features`. Mean |SHAP| over the test window, then retrain on
only the top K.

| Rank | Feature | mean \|SHAP\| |
|---|---|---|
| 1 | Volatility | 0.10156 |
| 2 | Return_60d | 0.09875 |
| 3 | RSI_14 | 0.08206 |
| 4 | MACD_signal | 0.06270 |
| 5 | MACD | 0.06194 |
| … | … | … |
| 20 | **BB_middle** | **0.00000** |

| Feature set | Accuracy | macro F1 | Edge vs always-BUY |
|---|---|---|---|
| top-3 | 0.3382 | 0.3027 | −0.0616 |
| **top-5** | **0.4155** | **0.3903** | **+0.0157** |
| top-10 | 0.4058 | 0.3725 | +0.0060 |
| all 20 | 0.3925 | 0.3491 | −0.0072 |

**Five features beat twenty.** The full set scores *below* the trivial baseline;
the top five score above it. The other fifteen are contributing variance, not
information.

**`BB_middle` is dead weight, and provably so.** Its mean |SHAP| is exactly
zero because it is byte-identical to `SMA_20` — `ta`'s `bollinger_mavg()` *is*
the 20-period simple moving average, which `features.py` already computes on
line 120. Verified: `max|BB_middle − SMA_20| == 0.0`.

## D. Combined effect

The two improvements are independent, so they were tested together:

| Configuration | Accuracy | macro F1 | Edge over baseline |
|---|---|---|---|
| shipped (20 features, 1.0× labels) | 0.3841 | 0.3416 | **−0.0157** |
| top-5 features, 1.0× labels | 0.4155 | 0.3903 | +0.0157 |
| 20 features, 1.5× labels | 0.3780 | 0.3622 | +0.0254 |
| **top-5 features + 1.5× labels** | 0.3961 | 0.3779 | **+0.0435** |

They stack roughly additively. Together they move the model from 1.6pp *below*
a constant prediction to 4.4pp *above* it.

That is a real improvement in relative terms — negative edge to positive edge —
but it is not a transformation. A 4.4pp edge over always-guessing-the-majority
is still a weak model, and none of it has yet been shown to survive into
out-of-sample *returns*, which finding A says is where this actually fails.

## Where this leaves things

The honest summary: **the strategy does not currently beat buy-and-hold out of
sample**, and the classification work above improves the model's edge over a
trivial baseline without yet demonstrating that it fixes that.

Sequenced from here:

1. **Re-run the buy-and-hold benchmark with the top-5 + 1.5× configuration.**
   This is the only measurement that matters, and it is now a cheap one. If the
   validation-window deficit closes, the changes are worth shipping; if it does
   not, they are cosmetic.
2. **Drop `BB_middle` regardless.** It is a duplicate column carrying zero
   information, and removing it costs nothing.
3. **Treat the exposure gap as a lever, not a footnote.** The strategy delivers
   its returns at ~50% invested. If selection is roughly break-even, sizing up
   is worth more than more model work.
4. **Do not extend `HISTORY_PERIOD`** — finding 1 above.

## Recommended next steps (original, from the first pass)

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
