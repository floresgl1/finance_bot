# Mask demand lead — stage 1 pre-registration draft (2026-10-02)

**Status: design only. Nothing has been run, and no data has been acquired.**
Every choice below is fixed on paper before any series is downloaded. Changing
one after seeing results turns this into a second draw; see "Confirming a future
pass" in `EDGE_INVESTIGATION_2026-09-08.md`.

Data seen so far, all from web search summaries of public filings, none of it a
time series: Photronics' 10%+ customer disclosures (FY2022–FY2025, customers
unnamed), its FY2025 revenue split by geographic origin (Taiwan 31%), and Taiwan
Mask's FY2025 annual revenue (about TWD 3.6B). No monthly or quarterly figure
for either company has been looked at.

*Amended 2026-10-02, still before any data exists:* the baseline now includes
UMC's monthly revenue as a sector series, so TMC has to add information beyond
mature-node wafer volume, not just beyond PLAB's own trend.

*Amended again 2026-10-02, still before any data exists:* the currency cost of
row 1 is stated; a negative TMC coefficient is recorded as an observation, never
a pass; the date range has a fallback start fixed now, and the structural
check that confirms it is limited to the filing text describing the joint
venture and the plants.

## Why this

Ten probes found no edge in price-derived signals, mega-caps or micro-cap value
(`EDGE_INVESTIGATION_2026-09-08.md`). The rule this design follows instead is
**use domain knowledge where the data is awkward.**

- **Mechanism.** Photomasks are bought per design, not per wafer: a fab orders
  a mask set when a new design (or a revision) tapes out. Merchant mask revenue
  therefore tracks design starts at fabs without a captive mask shop — in
  Taiwan, mature-node foundries such as UMC, which has repeatedly named
  Photronics' Taiwan subsidiary its preferred mask supplier.
- **Edge source.** Analysis. The same public data is available to everyone; the
  advantage is knowing that wafer-volume data (UMC's own monthly revenue) is the
  wrong proxy and mask-maker revenue is the right one.
- **Barrier.** Taiwan Mask Corporation (TWSE: 2338) is a small, loss-making,
  Taiwan-only listing that reports monthly in Chinese on MOPS. Photronics
  (PLAB) reports quarterly. Few PLAB holders are likely to watch TMC.

Stage 1 asks only whether the data carries information: **does TMC's monthly
revenue predict PLAB's reported revenue?** Stock returns are stage 2, and stage
2 is not designed or run unless stage 1 passes. A price correlation without a
working mechanism underneath it would most likely be luck.

## The test

| # | Component | Decision |
|---|---|---|
| 1 | Predictor | TMC revenue summed over the three calendar months of PLAB's fiscal quarter, as **year-over-year growth**, in **TWD** |
| 2 | Target | PLAB **total** quarterly revenue, as **year-over-year growth**, in USD, as first reported (not restated) |
| 3 | Alignment | PLAB fiscal quarter Q uses TMC revenue for its three calendar months. Each month is treated as available on the **10th of the following month**. A quarter whose PLAB earnings date falls before the third month's availability date is **excluded** |
| 4 | Date range | From the first PLAB quarter whose YoY comparison falls entirely after the PDMC joint venture was consolidated, through **FY2026 Q3** (May–Jul 2026). Exact start fixed in the structural checks from the 10-K. **Fallback: FY2016 Q1** if the filings do not state the consolidation date clearly |
| 5 | Baseline | **Model A:** PLAB's YoY growth this quarter = a + b × PLAB's YoY growth last quarter + d × UMC's YoY growth this quarter. **Model B:** Model A + c × TMC's YoY growth this quarter. UMC's growth is built exactly as TMC's is in rows 1 and 3: three calendar months summed, YoY, in TWD, each month available on the 10th of the following month. Both fitted on an **expanding window**; the first **12** usable quarters are training only; every later quarter is scored out of sample |

### Why each choice

1. **Predictor.** YoY, not month-on-month: monthly mask revenue is lumpy (a
   few large orders move it), and both companies have seasonal patterns that a
   year-on-year comparison removes. TWD, not converted to USD: converting would
   add a common exchange-rate component to both series, which can create
   correlation with no mask demand behind it. Leaving TMC in TWD can only
   weaken the relationship, never inflate it — the "err late" side.
   **The cost, stated in advance:** PLAB reports in USD and translates its
   Taiwan revenue from TWD, so a TWD move shifts PLAB's reported growth while
   leaving TMC's and UMC's TWD growth untouched. That is noise neither model
   can explain, and it makes a pass harder. A fail should be read knowing this
   noise was left in on purpose.
2. **Target.** Total revenue, not the IC segment or Taiwan alone: TMC also makes
   display masks, so an IC-only target would mismatch, and segment and
   geographic splits are reported less consistently than the top line. As first
   reported, because restated figures were not available on the day a trader
   would have used them.
3. **Alignment.** The 10th is the legal deadline for Taiwan-listed companies'
   monthly revenue. Using the deadline rather than the actual release date
   means no quarter can use a figure earlier than it was certainly public.
   Quarters where PLAB reported first are dropped, not patched with two months
   of data, so there is one rule and no case-by-case choice.
4. **Date range.** Photronics' Taiwan business changed shape when the PDMC
   joint venture was formed; YoY growth across that boundary measures
   consolidation, not demand. FY2026 Q3 is the latest quarter reported as of
   this draft, and fixing the end date now stops the sample being extended
   until it passes. The PDMC date in mind when this was written — the joint
   venture with DNP closing in early 2014 — is from memory, not a filing. The
   FY2016 Q1 fallback (two years after that) is fixed now so that confirming
   the date cannot become a choice made after seeing data.
5. **Baseline.** The obvious failure mode: both companies sell into the same
   semiconductor cycle, so their growth rates will correlate whether or not TMC
   carries any information PLAB's own history does not. Model A captures the
   cycle two ways: PLAB's own persistence, and UMC's revenue as the
   mature-node Taiwan cycle. UMC was chosen over global chip sales (WSTS/SIA)
   and TSMC because it is the same market PLAB's Taiwan business serves, it is
   public on the same day as TMC, and it tests the mechanism directly: the
   claim is that masks follow design starts, not wafer volume, so TMC should
   carry information UMC's wafer revenue does not. If UMC explains PLAB just as
   well, the mechanism adds nothing. TMC passes only if it improves on model A
   out of sample. 12 training quarters is three years — enough for a
   four-parameter regression to be fitted at all, short enough to leave most
   of the sample for scoring; the extra parameter makes model A's early fits
   noisier, which is why the bar is an out-of-sample count rather than an
   in-sample fit.

## Structural checks — before any result

Run on the downloaded data before either model is fitted. Each may exclude
quarters or fix the start date; none may look at the relationship between the
two series.

Checks 1 and 2 read only the filing text that describes the joint venture,
acquisitions and plants. Revenue tables often sit next to that text; any
revenue figure seen while doing so is recorded in the amendment log.

1. **PDMC consolidation date**, from the 10-K, sets row 4's start; FY2016 Q1
   if no filing states it clearly.
2. **M&A and capacity jumps.** List every acquisition, divestiture, or new fab
   start in either company's filings within the date range. Quarters whose YoY
   comparison spans one are excluded. (PLAB's China mask plants — Xiamen for
   IC masks, Hefei for display masks, starting production around 2019 from
   memory — are the known case: their growth is new capacity, not Taiwan
   design starts.)
3. **Earnings dates.** Record PLAB's actual earnings date for every quarter.
   Apply row 3's exclusion rule and report how many quarters it drops.
4. **Coverage.** Every TMC and UMC month in range present on MOPS. A missing
   month for either company excludes its quarter.

## Pass criteria

All three must hold:

1. **Direction.** Coefficient c, fitted on the full sample after scoring, is
   **positive**. The mechanism is market-wide demand, so TMC up should mean
   PLAB up. A negative c — share shifting between competitors — is a fail,
   however strong. It is recorded as an observation that may motivate a
   separate pre-registration with its own mechanism, never as a pass of this
   one; accepting either sign here would roughly double the chance a useless
   predictor passes.
2. **Model B beats model A in enough out-of-sample quarters.** With no
   information, each quarter is a coin flip:

   | Scored quarters | Must win at least | Chance a useless predictor passes |
   |---|---|---|
   | 30 | 20 | 4.9% |
   | 35 | 23 | 4.5% |
   | 40 | 26 | 4.0% |

   The bar is set by the number of quarters that survive the structural checks,
   read from this table (or the same one-sided 5% binomial rule for other
   counts), before any model is fitted.
3. **Mean absolute error at least 10% lower** for model B than model A, across
   all scored quarters. Criterion 2 can be passed by many tiny wins; this
   requires the improvement to be large enough to matter.

A fail ends the idea. Stage 2 is not run on a failed stage 1, and the predictor
is not re-specified (level instead of YoY, IC segment instead of total) and
re-tested: each of those is a second draw.

A pass is a lead, not a result. It justifies designing stage 2 — whether the
same information predicts PLAB's returns between the 10th and earnings — as a
separate pre-registration.
