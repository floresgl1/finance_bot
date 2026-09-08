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

## E. The decisive test — do B and C fix the return deficit?

They do not. `edge_probe.py --regimes` trains a fresh model on the 3 years
before each window, generates signals inside it, and simulates. Every window is
fully out-of-sample for its own model.

| Window | | shipped (20 feat, 1.0×) | top-5 + 1.5× |
|---|---|---|---|
| covid crash 2020 | hold −5.52% | +2.89% (**+8.41pp**) | −5.23% (+0.29pp) |
| bear 2022 | hold −29.00% | −33.41% (−4.41pp) | −19.30% (**+9.70pp**) |
| rally 2025-26 | hold +20.29% | +3.26% (−17.03pp) | +2.28% (−18.01pp) |
| recent 2026 | hold +3.49% | +1.68% (−1.81pp) | −0.56% (−4.05pp) |
| | | **1/4 windows, mean −3.71pp** | **2/4 windows, mean −3.02pp** |

**The classification improvements did not improve returns.** The B+C
configuration wins one more window but is *worse* in three of four individually,
and both configurations lose to buy-and-hold on average. Improving a proxy
metric moved the real metric the wrong way — which is the whole reason the
promotion gate was switched to simulated return rather than F1.

Note also that the shipped config's earlier +1.43pp on the test split does not
survive proper walk-forward training. That figure came from the deployed
champion evaluated on the live 3-year data; retraining strictly before the
window gives −1.81pp. The one positive out-of-sample data point was an artifact
of which model was being scored.

### What the regime spread actually shows

The strategy is **defensive, not skilled**. It runs at ~50% average exposure, so
it structurally lags a fully-invested benchmark in rallies and structurally
beats it in drawdowns. That pattern is visible — +8.41pp in the COVID crash,
−17.03pp in the rally — and it is what low exposure alone would produce.

But exposure does not explain all of it. Being 50% invested in a basket that
returned +20.29% should yield roughly +10%; the strategy returned +3.26%. The
extra ~7pp is selection, and it is negative. Conversely in 2022 the shipped
config lost 33.41% while holding lost 29.00% — worse than the market at half
the exposure, across 270 trades.

Results that swing from +9.70pp to −18.01pp with no consistent sign, and that
reverse when the feature set changes, are what a strategy with **no durable
edge** looks like.

## F. Exposure and the regime filter — is the deficit sizing or selection?

`python edge_probe.py --exposure`

Findings A and E left two live explanations for the return deficit, and they
imply opposite fixes:

- **Exposure.** Confidence-scaled sizing keeps the book near 50% invested, so it
  structurally lags a 100%-invested basket. If that is the whole story, sizing
  up closes the gap.
- **Selection.** The names it buys underperform the basket, in which case sizing
  up makes the loss *larger*.

Five arms over the same four walk-forward windows. Each window trains **one**
model, and every arm trades that same model's signals through the same
`backtest._simulate`, so slippage, commission and accounting are identical and
cannot explain a gap between arms. Only the allocation policy differs.

| Arm | Policy |
|---|---|
| shipped | confidence-scaled sizing, 20% per name, 80% total cap |
| full-size buys | every BUY at full size, 80% total cap |
| full-size, 100% cap | every BUY at full size, no cash buffer |
| regime filter only | SPY above its 200-day SMA → hold the basket, else all cash. **No model.** |
| regime + model | risk-on → hold the basket; risk-off → the model's signals |

The regime rule is lagged one session, so a day is allocated from the previous
close. A timing rule that reads the close it trades on is the easiest way to
manufacture an edge that is not there.

### Per-window

**covid crash 2020** — hold −5.52% (max DD −31.06%), risk-on 19% of days

| Arm | Return | vs hold | Max DD | Avg exp | Trades |
|---|---|---|---|---|---|
| shipped | +2.89% | **+8.41pp** | −17.27% | 59% | 74 |
| full-size buys | −1.70% | +3.82pp | −20.51% | 74% | 56 |
| full-size, 100% cap | −2.67% | +2.84pp | −25.66% | 94% | 45 |
| regime filter only | −12.65% | −7.13pp | −14.58% | 19% | 36 |
| regime + model | −8.17% | −2.65pp | −19.91% | 48% | 85 |

**bear 2022** — hold −29.00% (max DD −34.29%), risk-on 21% of days

| Arm | Return | vs hold | Max DD | Avg exp | Trades |
|---|---|---|---|---|---|
| shipped | −33.41% | −4.41pp | −34.50% | 48% | 270 |
| full-size buys | −42.88% | −13.88pp | −42.86% | 67% | 234 |
| full-size, 100% cap | −47.46% | −18.46pp | −50.33% | 85% | 213 |
| regime filter only | −25.74% | **+3.26pp** | −25.74% | 21% | 72 |
| regime + model | −32.50% | −3.50pp | −32.88% | 37% | 257 |

**rally 2025-26** — hold +20.29% (max DD −7.45%), risk-on 91% of days

| Arm | Return | vs hold | Max DD | Avg exp | Trades |
|---|---|---|---|---|---|
| shipped | +3.26% | −17.03pp | −7.65% | 50% | 184 |
| full-size buys | −2.23% | −22.52pp | −13.95% | 71% | 147 |
| full-size, 100% cap | +0.87% | −19.42pp | −16.54% | 87% | 145 |
| regime filter only | +9.51% | −10.78pp | −6.51% | 91% | 24 |
| regime + model | +6.16% | −14.13pp | −6.62% | 85% | 36 |

**recent 2026** — hold +3.49% (max DD −6.35%), risk-on 100% of days

| Arm | Return | vs hold | Max DD | Avg exp | Trades |
|---|---|---|---|---|---|
| shipped | +1.68% | −1.81pp | −6.07% | 48% | 86 |
| full-size buys | +7.33% | +3.84pp | −9.14% | 72% | 61 |
| full-size, 100% cap | +9.54% | **+6.05pp** | −11.34% | 89% | 55 |
| regime filter only | +3.49% | +0.00pp | −6.35% | 100% | 12 |
| regime + model | +3.49% | +0.00pp | −6.35% | 100% | 12 |

### Aggregate

| Arm | Windows beating hold | Mean vs hold |
|---|---|---|
| shipped | 1/4 | −3.71pp |
| full-size buys | 2/4 | −7.18pp |
| full-size, 100% cap | 2/4 | −7.25pp |
| regime filter only | 1/4 | **−3.66pp** |
| regime + model | 1/4 | −5.07pp |

### 1. The deficit is selection, not exposure

Sizing up roughly **doubles the shortfall**: −3.71pp → −7.25pp. Being half
invested was not causing the underperformance, it was *masking* it. Half a
position in a losing pick loses half as much.

The win-count column disagrees with the mean column, and the mean is the honest
one. Full-size beats hold in 2/4 windows against shipped's 1/4, but its wins are
small (+2.84pp, +6.05pp) and its losses are enormous (−18.46pp in 2022,
−19.42pp in the rally). Counting windows weights a 2pp win the same as a 20pp
loss.

The one genuine exception is **recent 2026**, where sizing up gained +6.05pp —
the picks did beat the basket there. It is also the shortest window (~3.5
months) and the only one where the market never left risk-on. One window is a
coincidence, not a finding.

### 2. A 200-day SMA filter with no model beats the model

`regime filter only` is the best arm in the table at −3.66pp, and it contains no
model at all. It is the **only** arm that beat hold in the 2022 bear market
(+3.26pp), and it did so with 72 trades against the strategy's 270.

Layering the model on top of it makes it worse, −3.66pp → **−5.07pp**. On the
days the filter defers to the model — precisely the drawdown days the model was
supposed to be good at — the model gives back 1.4pp.

### 3. But the filter does not work either

−3.66pp mean, 1/4 windows. It loses to simply holding, and it loses worst in the
COVID crash (−7.13pp): it went to cash into the decline and was still in cash
through the V-shaped rebound. Textbook whipsaw. Trend filters buy drawdown
protection with return, and over these windows the price was too high.

### 4. Sanity check the arms pass

In **recent 2026** SPY was above its 200-day SMA on 100% of sessions, so both
regime arms should degenerate *exactly* into buy-and-hold. They do: +0.00pp
delta, 12 trades — one per ticker, opened on day one and never touched. The
policy rewrite is doing what it claims.

### 5. Drawdown, the one place something wins — and it does not replicate

In the COVID crash the shipped strategy roughly halved the drawdown (−17.27% vs
the basket's −31.06%) *and* made money. That is a real result.

It does not repeat. In 2022 it gave **no** drawdown protection whatsoever
(−34.50% against −34.29% for holding) while losing an extra 4.41pp. A defensive
property that appears in one crash and vanishes in the next is not a property.

## Where this leaves things

**The strategy does not beat buy-and-hold out of sample. The model changes that
improve its classification metrics do not fix that, and neither does changing
how the capital is allocated.** Every configuration and every allocation policy
tested loses to holding on average across four regime windows — including a
200-day SMA filter with no model in it, which is the best of them.

Finding F is the one that closes the question. If the shortfall had been
exposure, sizing up would have fixed it; instead sizing up doubled it. The picks
are the problem, and no allocation rule repairs a negative selection edge.

Nothing was changed in production as a result. `FEATURE_COLUMNS` still contains
all 20 features including `BB_middle`, and the labels are unchanged — because
no tested configuration is better on the measure that matters, and swapping the
feature set would require promoting a new champion. There is nothing here worth
promoting.

### Specifically NOT recommended

- **Do not extend `HISTORY_PERIOD`.** Finding 1: more data makes it slightly worse.
- **Do not adopt top-5 features or 1.5× labels on the strength of finding D.**
  Finding E shows the classification gain does not survive into returns.
- **Do not drop `BB_middle` on its own.** It is genuinely dead weight, but
  removing it changes the feature contract and breaks the deployed champion,
  which expects 20 columns. It is a coordinated retrain-and-promote, not a
  free deletion — worth folding into the next promotion that happens for
  other reasons, not worth triggering one.
- **Do not raise position sizes or `MAX_TOTAL_EXPOSURE`.** Finding F: sizing up
  doubles the shortfall. The current ~50% exposure is limiting the damage, not
  causing it.
- **Do not add a 200-day SMA regime filter.** Finding F: on its own it still
  loses to holding, and the model *subtracts* value when layered on top of it.

### Tried and refuted

The two leads this section used to recommend have now been run, in finding F,
and both are dead:

1. ~~**Exposure and sizing.**~~ Tested. Sizing up doubles the shortfall
   (−3.71pp → −7.25pp). The deficit is selection, not exposure.
2. ~~**A regime filter.**~~ Tested. A 200-day SMA filter still loses to holding
   (−3.66pp), and the model makes it *worse* when layered on top (−5.07pp).

### What is left

1. **A different label horizon.** Everything here uses a 7-day forward return.
   That choice has never been tested, and it determines what "signal" means
   more than any feature does. It is the last untested assumption in the setup —
   though findings A, E and F have lowered the prior on any of this working.
2. **Accept the result.** A 12-ticker daily-bar long-only strategy on standard
   technical indicators is a crowded, well-arbitraged space. "No durable edge"
   is the expected outcome, not a bug — and the pipeline around it (risk
   controls, monitoring, P&L attribution, the promotion gate) is sound
   engineering regardless of whether this particular signal works.

   This is now the recommended reading. Three independent lines of evidence —
   the returns benchmark (A, E), the classification-improvement test (E), and
   the allocation-policy sweep (F) — all point the same way, and the last of
   them showed a rule with no model in it beating the model.

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
python edge_probe.py --features                 # SHAP ranking + top-K retrain
python edge_probe.py --label-scales 1 1.5 2 3   # HOLD band width
python edge_probe.py --regimes                  # returns vs hold  (finding E)
python edge_probe.py --exposure                 # allocation policy (finding F)
```

Writes `data/edge_probe_results.json`. Downloads into `data/edge_probe/` rather
than `data/`, so the live CSVs the next real training run reads are untouched.
