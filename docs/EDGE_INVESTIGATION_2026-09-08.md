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

## G. The label horizon — the last untested assumption

`python edge_probe.py --horizons 3 5 7 14 21`

`labels._WINDOW = 7` was chosen once and never questioned. It decides what the
word "signal" means here more than any feature does: too short and the label is
mostly microstructure noise, too long and the model is asked to forecast
something no daily technical indicator carries.

Scored on **returns**, not accuracy — finding E showed those two disagree, so a
horizon sweep judged on classification would repeat that mistake. Each horizon
re-labels the whole dataset (`labels._WINDOW` drives the SPY forward return, the
stock forward return, and the tail-row drop together) and re-runs the same
walk-forward benchmark.

### On the four regime windows, this looked like the answer

| Horizon | BUY / HOLD / SELL | Windows beating hold | Mean vs hold |
|---|---|---|---|
| 3 days | 28 / 46 / 26 | 2/4 | **+5.29pp** |
| 21 days | 45 / 17 / 39 | 3/4 | +3.11pp |
| 5 days | 33 / 36 / 31 | 3/4 | +1.27pp |
| 14 days | 42 / 20 / 38 | 1/4 | +0.89pp |
| **7 days (shipped)** | 37 / 30 / 34 | 1/4 | **−3.71pp** |

Four of five horizons beat buy-and-hold, and the shipped one was the worst of
them. That was the first positive result in the entire investigation.

It is also wrong.

## H. The window set was the bug

The four windows in `REGIME_WINDOWS` were chosen to span regimes — COVID, the
2022 bear, a rally, and recent data. That is the right sample for asking *"is
this defensive?"* and the **wrong sample for a mean**, because two of the four
are major drawdowns. Any strategy that simply holds less stock collects a large
bonus. Finding F identified that trap and then walked straight into it.

`BROAD_WINDOWS` covers the same decade continuously in ten windows, so bear
periods appear roughly in the proportion they actually occurred. Nothing starts
before late 2019, because each window trains on the three years before it and
the probe history reaches back ten.

### G, re-run on ten windows

| Horizon | Windows beating hold | Mean vs hold | Mean exposure |
|---|---|---|---|
| 14 days | 2/10 | −14.23pp | 51% |
| 21 days | 3/10 | −16.81pp | 56% |
| **7 days (shipped)** | 1/10 | −24.09pp | 46% |
| 3 days | 2/10 | **−24.37pp** | 19% |
| 5 days | 3/10 | −25.35pp | 35% |

**Every horizon loses, by 14 to 25 percentage points, and the ranking inverts.**
The 3-day horizon goes from best (+5.29pp) to nearly worst (−24.37pp).

The mechanism is visible in the exposure column. 3-day labels run at 12–23%
invested. In a sample that is half crashes, holding almost nothing is a winning
strategy. In a sample that looks like the actual decade, it means returning
−2.13% while the basket made +69.66% (COVID recovery) and +5.58% while the
basket made +75.20% (2023).

| 3-day labels | strategy | hold | delta | exposure |
|---|---|---|---|---|
| covid crash 2020 | +15.01% | −5.52% | **+20.52pp** | 41% |
| bear 2022 | −7.47% | −29.00% | **+21.53pp** | 23% |
| covid recovery 2020 | −2.13% | +69.66% | −71.79pp | 15% |
| recovery 2023 | +5.58% | +75.20% | −69.63pp | 14% |
| bull 2024 | −13.11% | +43.33% | −56.44pp | 20% |

The two wins are the two crashes. Everything else is a catastrophe.

### E, revised

The shipped configuration measured over ten windows instead of four:

| | Windows beating hold | Mean vs hold |
|---|---|---|
| Finding E (4 regime windows) | 1/4 | −3.71pp |
| **Corrected (10 broad windows)** | **1/10** | **−24.09pp** |

The conclusion does not change. Its magnitude gets six times worse. The −3.71pp
figure was itself flattered by the bear-weighted sample.

### F, corrected — the deficit is BOTH, not selection alone

`python edge_probe.py --exposure --broad`

| Arm | 4 regime windows | 10 broad windows | Mean exposure |
|---|---|---|---|
| shipped | −3.71pp | −24.09pp | 46% |
| full-size buys | −7.18pp | −20.10pp | 74% |
| full-size, 100% cap | −7.25pp | **−17.87pp** | 87% |
| regime filter only | −3.66pp | **−9.08pp** | 80% |
| regime + model | −5.07pp | −10.77pp | 80% |

**Finding F's headline was wrong.** On four bear-weighted windows, sizing up
made things worse and the conclusion was "the deficit is selection, not
exposure." On ten representative windows, sizing up *helps* — −24.09pp to
−17.87pp.

The honest version: the ~24pp shortfall is roughly **6pp of exposure and 18pp of
selection**. Running at 46% invested through a decade where the basket returned
+69%, +75% and +43% in single windows is a structural drag, and correcting it
recovers a quarter of the gap. The other three quarters are the picks. Sizing up
is not a fix — it turns a large loss into a slightly smaller large loss.

What *does* survive both window sets: **the 200-day SMA filter with no model in
it is the best arm in the table, and adding the model makes it worse.** −9.08pp
vs −10.77pp here, −3.66pp vs −5.07pp there. Consistent, and still losing to
holding.

Its −9.08pp is also mostly inaction: it ties buy-and-hold exactly in the four
windows where SPY never left risk-on, wins the 2022 bear, and loses badly
whenever it whipsaws (−43.51pp in the 2023 recovery, out of the market for it).

### What this costs the earlier findings

Findings A, E and F were all computed on the four-window set. Their *directions*
all survive the correction — every one of them said the strategy loses to
holding, and on a representative sample it loses by much more. Only F's
attribution of *why* was wrong, and it is corrected above. Findings 1–4, B, C
and D are classification measurements on a fixed test window and are unaffected.

## I. The simulator was never simulating the live bot

The live account's first six months came in at **−3.23pp against holding the
basket** (`live_benchmark.py --days 365`, 2026-03-05 → 2026-09-04, 128 sessions,
+15.11% vs +18.34%, and a *shallower* drawdown of −5.58% vs −7.64%). The
simulation said −24pp. A gap that size is not sampling noise.

### First hypothesis: the sizing constants. Wrong.

`backtest.py` declared its own `MAX_POSITION_PCT = 0.20` and
`MAX_TOTAL_EXPOSURE = 0.80`, while the live path uses
`config.MAX_POSITION_PCT = 0.08` and has **no portfolio-level exposure check at
all** — `live_trader.py` and `capital_allocator.py` contain zero references to
exposure, so twelve full positions reach ~96% invested.

Both now come from `config.py`. It did not close the gap; it widened it.

| Arm (10 broad windows) | Mean vs hold |
|---|---|
| shipped, live-matching caps (8% / no cap) | **−27.07pp** |
| concentrated, the old hardcoded caps (20% / 80%) | −24.09pp |

### The real mismatch is the sizing *rule*, not the caps

Simulated over the **exact live window**, which is the only apples-to-apples
comparison available:

| | return | vs hold | max DD | avg exposure |
|---|---|---|---|---|
| **LIVE account** | **+15.11%** | **−3.23pp** | −5.58% | ~82% (implied) |
| simulated, live-matching caps | +1.14% | −16.07pp | −3.59% | **22%** |
| simulated, old caps | +6.36% | −10.85pp | −7.21% | 53% |
| equal-weight hold | +17.21% | — | −7.64% | 100% |

The account beat its own simulation by **14 percentage points on identical
dates**. The exposure column says why: the simulator runs at 22% invested where
the live bot runs near 82%.

`backtest._simulate()` sizes a position at `Confidence × MAX_POSITION_PCT`
(≈ 0.45 × 0.08 ≈ 3.6%) and then **never adds to it** — `if ticker in positions:
continue`. Neither of those is what the bot does.

### What the live bot actually does — and it is self-contradictory

| Path | Rule | Size |
|---|---|---|
| new position | `live_trader.get_position_size()` | conf 35–40 → **10%**, 40–45 → **15%**, 45+ → **20%** |
| add to position | `capital_allocator._get_allocation_tier()` | 3% / 5% / 7%, capped by headroom to `MAX_POSITION_PCT` = **8%** |
| trim | `rebalancer` | weight > 8.1% → sell back to **7.5%**, max 25% of shares per run |

**A new position opens at 10–20% of equity, and the cap that governs it
afterwards is 8%.** Every position the bot opens is therefore immediately
over-weight, gets trimmed back to 7.5% over roughly three sessions, and can
never be added to in between.

The signal log shows exactly this:

| `actual_action` | count |
|---|---|
| `REBALANCER_SELL` | **47** |
| `BUY` | 17 |
| `INVALID_HEADROOM` | **31** |
| `ADD_TO_POSITION` | 16 |

Nearly three trims per position opened, and thirty-one add attempts refused
because the position was already above its own cap. Each trim pays slippage and
commission on stock the bot bought days earlier at its own initiative.

Net effect: positions settle near 7.5%, twelve names reach ~90% invested, and
the account tracks the basket closely — which is what the live numbers show, and
what no version of the simulator reproduces.

### What this means for findings A, E, F, G, H

They measured a strategy that runs at 22–53% exposure and never scales into a
position. The deployed system runs near 82% and churns through a trim cycle on
every entry. **The −24pp figure describes something that has never traded.**

Their *directions* may still hold — the live account is behind the basket too —
but the magnitude is not transferable, and no return figure from this
investigation should be quoted as describing the bot until `_simulate()` models
`get_position_size()` and the rebalancer trim.

The classification findings (1–4, B, C, D) are unaffected: they are measured on a
fixed test window and never touch position sizing.

## J. The sizing conflict resolved, and what it revealed

Finding I left a live defect and a broken simulator. Both are now fixed, and the
result is the clearest statement this investigation has produced.

### The resolution

`live_trader.get_position_size()` now sizes new positions from
`capital_allocator.get_allocation_tier()` — the same table that governs top-ups —
so nothing opens above the cap that governs it afterwards:

| confidence | was | now |
|---|---|---|
| < 35 | skip | skip |
| 35–50 | 10% | **3%** (`SMALL_POSITION_PCT`) |
| 50–65 | 15–20% | **5%** (`NORMAL_POSITION_PCT`) |
| 65+ | 20% | **7%** (`LARGE_POSITION_PCT`) |

All three sit under `MAX_POSITION_PCT = 0.08`. The rebalancer returns to being a
safety net instead of a routine step, and `check_add_to_position` gets headroom
to work with for the first time.

`backtest._simulate()` now uses the same tier table **and tops up held
positions** rather than skipping any ticker already owned. That second change
turned out to be the larger of the two.

### Concentration was the alternative, and it loses

Raising `MAX_POSITION_PCT` to 0.20 was the other way to resolve the conflict. On
the live window alone it looked clearly better — and that is exactly the trap
finding H documented:

| Arm | live window (1) | broad (10) |
|---|---|---|
| shipped, 8% tiers | −12.86pp | **−27.52pp** |
| concentrated, 20% / 80% | **+0.93pp** | **−29.13pp** |

A single window said concentration beats buy-and-hold. Ten windows say it is the
worst arm tested. Concentration only pays when there is selection edge to
concentrate into, and six measurements say there is none — so it buys variance
and nothing else.

### The simulator and reality finally agree

Configured to what the bot actually did during the live window — 20% opens, adds
enabled — the simulation lands close to the account for the first time:

| | return | vs hold | exposure |
|---|---|---|---|
| simulated at live's actual sizing | +18.14% | +0.93pp | 65% |
| **LIVE account** | **+15.11%** | −3.23pp | ~82% |

A ~3pp residual, in the direction you would expect from the trim churn the
simulator does not model — 47 `REBALANCER_SELL` against 17 `BUY`, each paying
slippage and a commission. Finding I's 14pp discrepancy was almost entirely the
missing add-to-position logic.

### Sizing is not the lever. Exposure is.

Broad sample, faithful simulator:

| Arm | Beat hold | Mean vs hold | Mean exposure |
|---|---|---|---|
| shipped, 8% tiers | 2/10 | −27.52pp | **33%** |
| full-size buys | 1/10 | −26.50pp | 39% |
| concentrated, 20% / 80% | 1/10 | −29.13pp | 55% |
| regime filter only (**no model**) | 2/10 | −9.46pp | **79%** |
| regime + model | 2/10 | **−7.55pp** | **88%** |

Every model-driven sizing variant lands between −26.5pp and −29.1pp. **The
choice of sizing rule moves the result by 2.6 percentage points on a 27-point
deficit.** It is not the lever.

The two arms that come close to holding are the two that are nearly always
invested, and the better of them is the one with the least model in it. Sort the
table by exposure and it sorts by performance. Over a decade in which the basket
compounded through +69%, +75% and +43% windows, every hour spent in cash is the
cost, and the model's signal is what puts the book in cash.

`regime + model` at −7.55pp is now the best arm across both window sets — but it
is 88% invested and defers to the model only on risk-off days. It is closer to
"hold, with an occasional exit" than to a strategy.

### The cost of the fix

The resolution lowers simulated exposure from 55% (at live's old 20% opens) to
33%. Given that exposure is the dominant factor, that is a real downside, and it
is the one reason to revisit the tier percentages — `SMALL_POSITION_PCT = 3%`
against a `MAX_POSITION_PCT` of 8% leaves the book half empty for the modal
signal. Raising the tiers toward the cap keeps the diversification and the
absence of churn while restoring the exposure.

That is deliberately **not** done here. It is a strategy parameter, the broad
sample separates the sizing variants by less than 3pp, and tuning it on this
data is precisely how finding H happened.

## Where this leaves things

**The strategy does not beat buy-and-hold**, on every measurement taken: six
simulated attempts to find an edge, and the live account itself at −3.23pp over
its first six months.

Finding I found `backtest._simulate()` had never modelled the live sizing rule.
Finding J fixed it, and the simulator now reproduces the live account to within
~3pp on the same window — the first time simulation and reality have agreed.

**The lever is exposure, not the model and not the sizing rule.** Every
model-driven sizing variant lands between −26.5pp and −29.1pp against holding
over ten windows. The only arms that come close are the two that stay ~80–88%
invested, and the better of those contains no model on risk-on days at all.

Measured over ten continuous windows spanning 2019–2026, the shipped
configuration returns **−24.09pp against simply holding the watchlist, beating
it in 1 window out of 10.** About 6pp of that is exposure — running at 46%
invested through a decade of large up years — and the remaining ~18pp is
selection. Sizing up recovers the first part and leaves the second.

Finding H is the one to remember methodologically: the four-window sample used
for findings A, E and F was half drawdowns by construction, and it flattered
every result computed on it, including the ones that were already negative. The
directions all survived; one attribution did not.

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
- **Do not change the label horizon.** Finding G/H: every horizon from 3 to 21
  days loses by 14–25pp on a representative sample. The 3-day horizon that
  looked best on four windows is nearly the worst on ten.
- **Do not raise position sizes or `MAX_TOTAL_EXPOSURE` expecting a fix.**
  Finding F as corrected: sizing up recovers about 6pp of a 24pp gap. It turns a
  large loss into a slightly smaller large loss, at materially higher drawdown.
- **Do not add a 200-day SMA regime filter.** Findings F and H: it is the best
  arm tested and still loses to holding, and the model *subtracts* value when
  layered on top of it — consistently, on both window sets.
- **Do not report a mean over `REGIME_WINDOWS`.** Use `--broad`. The regime set
  exists to answer "is this defensive?", and it overweights drawdowns 2:4.

### Tried and refuted

Every lead this section has recommended has now been run, and all of them are
dead:

1. ~~**More training history.**~~ Finding 1: more data makes it slightly worse.
2. ~~**Better features (top-5 by SHAP).**~~ Findings C and E: better
   classification, worse returns.
3. ~~**A wider HOLD band (1.5x labels).**~~ Findings B and E: same.
4. ~~**Exposure and sizing.**~~ Findings F and H: recovers ~6pp of a 24pp gap.
5. ~~**A regime filter.**~~ Findings F and H: the best arm tested, still losing,
   and made worse by adding the model.
6. ~~**A different label horizon.**~~ Findings G and H: every horizon from 3 to
   21 days loses by 14–25pp.

### What is left

**Accept the result.** A 12-ticker daily-bar long-only strategy on standard
technical indicators is a crowded, well-arbitraged space. "No durable edge" is
the expected outcome, not a bug.

Six independent attempts to find one have now failed, and the most informative
of them found that a rule with no model in it beats the model. The pipeline
around the signal — risk controls, monitoring, P&L attribution, a promotion gate
that correctly refuses to ship a worse model, 499 tests — is sound engineering
regardless of whether this particular signal pays.

If the project continues as a learning exercise rather than a strategy, the
interesting next chapter is a different question, not a better answer to this
one: a more predictable target (realised volatility rather than direction), a
different asset class or timeframe, or continued investment in the execution and
monitoring machinery, which is the part that works.

The promotion gate now encodes this: `beats_buy_and_hold` (added 2026-09-08)
rejects any challenger that loses to holding the basket, so the conclusion here
cannot be quietly forgotten by a future retrain.

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
python edge_probe.py --horizons 3 5 7 14 21     # label window     (finding G)
python edge_probe.py --horizons 3 7 21 --broad  # ten windows, not four (finding H)
python edge_probe.py --exposure --broad         # F, corrected
```

**Use `--broad` for any figure quoted as a mean.** `REGIME_WINDOWS` is half
drawdowns by construction; `BROAD_WINDOWS` covers the decade continuously.

Writes `data/edge_probe_results.json`. Downloads into `data/edge_probe/` rather
than `data/`, so the live CSVs the next real training run reads are untouched.
