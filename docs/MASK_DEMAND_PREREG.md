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

## Structural check results (2026-10-02) — stopped before any model was fitted

**Data acquired.** TMC (2338) and UMC (2303) monthly revenue, December 2009 to
August 2026, from FinMind's mirror of MOPS (`api.finmindtrade.com`, dataset
`TaiwanStockMonthRevenue`). MOPS itself rejects requests from the environment
these checks ran in ("FOR SECURITY REASONS, THIS PAGE CAN NOT BE ACCESSED"), so
the source differs from the one written above; FinMind's values are a
third-party copy and may reflect revisions rather than first reports. PLAB's
XBRL company facts, submissions index and FY2013–FY2025 10-Ks from SEC EDGAR.

| File | sha256 |
|---|---|
| `fm_2338.json` | `8804fec99fc60b7104f6b1fef6d7665cc6b11ded50c5bdf8fc9c9396e7763b23` |
| `fm_2303.json` | `206f977a3b8788f189f48098f72f6cf59779bdc5cbde05b44a1ea5b25474fc5f` |
| `plab_facts.json` | `d5be2991411bc641e8aa7927cb0e1493ce5d2a881d09bcbfbf0bc3ad262750a2` |
| `plab_sub.json` | `e6a6147424c9e0712e0d8d4adbf42956e916fde6aaea61b20e789945528850a8` |

**Data seen.** No PLAB revenue figure. 10-K text was read with dollar amounts
masked. Two TMC figures were seen: a news search result reported TMC's
consolidated revenue for H1 2018 as TWD 1.374B, up 128.4% on H1 2017, and the
FinMind series was summed over H1 2018 and H1 2017 (TWD 1.374B and 0.602B) to
establish what it measures. Nothing relating either Taiwan series to PLAB has
been computed or looked at.

**Check 1 — PDMC.** Confirmed from the FY2014 10-K: DPTT merged into PSMC to
form PDMC on **April 4, 2014** (PLAB FY2014 Q2), Photronics 50.01%. The memory
in row 4 was right; the FY2016 Q1 fallback is not needed.

**Check 2 — M&A and capacity jumps.**

| Company | Event | Date (PLAB fiscal quarter) | Source |
|---|---|---|---|
| PLAB | PDMC formed | Apr 2014 (FY2014 Q2) | FY2014 10-K |
| PLAB | Sold MP Mask joint venture investment | FY2016 | FY2016 10-K — equity-method, no consolidated revenue, not excluded |
| PLAB | Acquired a large-area IC mask business | FY2017 Q1 (Nov 2016 – Jan 2017) | FY2017 10-K |
| PLAB | Hefei FPD plant starts production | FY2019 Q2 | FY2019 10-K |
| PLAB | Xiamen IC plant starts production | FY2019 Q3 | FY2019 10-K |
| PLAB | Bought out PKL minority interest | FY2019 | FY2019 10-K — already consolidated, not excluded |
| TMC | Acquired 100% of 美祿科技, a wafer-capacity agent | closing by Oct 31, 2017 (FY2017 Q4) | news, 2017-09-29 |
| TMC | Acquired 威達高科 (touch-panel ICs) and 群豐科技 (flash packaging) | 2017–2018, exact dates not yet found | zh.wikipedia, news |

**The finding that stops the test.** The FinMind series is TMC's
*consolidated* revenue: its H1 2018 sum matches the reported consolidated
figure exactly. From late 2017 the predictor therefore measures masks plus
wafer brokering, touch-panel ICs and flash packaging. H1 2018 revenue more
than doubled on the acquisition alone, so the non-mask businesses are at least
as large as the mask business.

Check 2's rule — exclude quarters whose YoY comparison spans an event — does
not cover this. It removes the year in which an acquisition enters the YoY
comparison, but after that year the predictor is permanently a different
quantity. Applied mechanically, it would test a mixed series and call the
result a test of mask demand.

Before late 2017 the predictor is clean, but the window is too short. The
first PLAB quarter whose YoY growth and lagged YoY growth both fall after PDMC
is FY2015 Q4; PLAB's FY2017 Q1 acquisition excludes FY2017 Q1–Q4; TMC's
acquisitions start in FY2017 Q4. That leaves FY2015 Q4 – FY2016 Q4, five
quarters — fewer than the twelve the expanding window needs before scoring
its first one.

**Status: not testable as specified.** This is neither a pass nor a fail.
Checks 3 and 4 (earnings dates, coverage) were not completed for PLAB; TMC
and UMC coverage is complete (201 months each, none missing or duplicated).
How to proceed is an amendment to be decided before any model is fitted.

### Option 2 attempt — parent-only TMC revenue (2026-10-02)

Chosen after the stop above: replace row 1's consolidated TMC revenue with
TMC's parent-only (個體) revenue, which is the mask business without the
subsidiaries. No model has been fitted.

**Not obtainable from this environment.** MOPS (`mops.twse.com.tw`,
`mopsov.twse.com.tw`) and the TWSE open API return the exchange's own
"FOR SECURITY REASONS" page to this server; Goodinfo returns 403; FinMind
carries consolidated revenue only. Whether MOPS publishes a parent-only
*monthly* series at all is unconfirmed: parent-only financial statements are
an annual filing, and the parent-only figures found so far appear only in
occasional company commentary reported by the press, not as a series.

**Data seen during the attempt** (TMC only; nothing relating TMC or UMC to
PLAB): a news search reported TMC's January 2026 consolidated revenue as
TWD 537M, of which the core mask business was TWD 340M ("over 60%"), and that
the mask business is about 60% of group revenue; February 2026 revenue as
TWD 459.4M, down 14.5% YoY; February 2024 revenue as TWD 520M, down 4% YoY.

**Status: still not testable as specified.** Next step needs a person on a
network MOPS accepts: open MOPS → 營運概況 → 每月營收 (t05st10_ifrs) for
company 2338, and record whether the page reports a parent-only revenue line
alongside the consolidated one, and from what year. If it does, row 1 is
amended to that line before any model is fitted. If it does not, the
hypothesis is closed as untestable with public data.

### Option 2 result — closed as untestable (2026-10-02)

Checked by hand on MOPS (`t05st10_ifrs`, company 2338, 民國115年08月) from a
network MOPS accepts. The page reports **one** revenue line (營業收入淨額), with
no parent-only line. Its figures — August 2026 TWD 494,412k, August 2025
543,000k, January–August 2026 4,052,183k — match the FinMind series exactly,
so that one line is the consolidated revenue already ruled out above.

There is no public monthly series of TMC's mask business alone. Under the rule
written before the check, **the hypothesis is closed as untestable with public
data.** Neither a pass nor a fail: no model was fitted, and nothing relating
TMC or UMC to PLAB was ever computed.

What this does not close: the mechanism (mask demand follows design starts,
not wafer volume) is untouched. A future test of it needs a predictor that
measures merchant mask demand alone and reports more often than PLAB does.

## Replacement predictor search — Faraday NRE (2026-10-02)

Candidate: Faraday Technology (TWSE 3035), UMC's ASIC design-service
affiliate. Its NRE revenue is paid around tape-out, so it measures design
starts directly. Its monthly total does not: mass production (resold wafers)
is the largest category (73% in Q2 2026 per press coverage), which is wafer
volume and largely duplicates UMC in the baseline. Only quarterly NRE could
serve, and Faraday's quarter (Jul–Sep) is reported about six weeks before
PLAB's (Aug–Oct), so it would still lead.

**It fails two checks before any test design:**

- **History.** Exact quarterly NRE figures appear in Faraday's English
  quarterly reports from 2Q22 only (~17 quarters to 2Q26); earlier press
  releases mention NRE inconsistently ("exceeded NT$200M" in 3Q16). Seventeen
  quarters leave five to score after the twelve-quarter training window.
- **Foundry.** Faraday is multi-foundry (UMC, Samsung, Intel,
  GlobalFoundries). Per its annual report as quoted in the press, UMC was
  20.64% of 2025 purchases against 63.23% from an unnamed supplier, and 33.37%
  in Q1 2026. The years with clean NRE data are the years it is least tied to
  UMC.

**Data seen** (Faraday only, nothing relating it to PLAB): NRE of "over
NT$200M" in 3Q16, NT$649M in 3Q24, NT$931M in 4Q25; MP NT$2.43B (73%) in
2Q26; total revenue NT$3.31B in 2Q26 and NT$11.06B in 2024.

Unchecked: whether Faraday's Chinese-language investor presentations on MOPS
give exact quarterly NRE before 2022.

**Follow-up, same day.** MOPS monthly revenue for 3035 (checked by hand,
民國115年08月) reports one total line — August 2026 NT$1,183,747k — so there
is no monthly NRE. Faraday's own site serves quarterly reports from 2Q22 only,
in English and Chinese; no earlier file names were found. The TWSE e-filing
archive (`doc.twse.com.tw`) rejects this environment as MOPS does. The
remaining source is Faraday's quarterly financial statements, whose IFRS 15
revenue-disaggregation note may split NRE, IP and mass production from 2018,
reachable only by hand.

**Stop rule, agreed 2026-10-02 before the check.** One last source is checked:
the revenue-disaggregation note (收入之細分 / 客戶合約之收入) in Faraday's 2018
Q1 consolidated financial statements. If it gives exact quarterly NRE, this
document is amended to use quarterly Faraday NRE as the predictor, with the
post-2022 foundry shift handled by a rule fixed before any model is fitted.
If it does not, the mechanism — mask demand follows design starts — is closed
as unmeasurable with public data, and no further dataset is searched for it.

### Last check result — condition met (2026-10-03)

Faraday's 2018 Q1 consolidated financial statements (`201801_3035_AI1.pdf`,
uploaded to the TWSE e-filing archive 2018-04-20; sha256
`0ddc89423fa5e970546dd22ee3b2c76ef71ee6a6c02ef06d8b869dba71f290cb`), note
六.15 "營業收入淨額", subsection (1) 收入細分, page 53, splits quarterly revenue
into exact amounts:

| Category | Line | 2018 Q1 (NT$k) |
|---|---|---|
| Mass production | 銷售商品 | 573,890 |
| Design services (NRE) | 提供勞務 | 373,331 |
| IP licensing | 矽智財授權收入 | 95,719 |

The same note gives the 2017 Q1 comparative (商品 1,057,445; 勞務 374,707; no
separate IP line under the old standard), so 2017 comparatives may extend the
series one year back. It also states the transaction price allocated to
unsatisfied services and IP obligations — NT$1,254,894k at 2018-03-31, to be
recognised over 1–1.5 years — a design backlog figure. Neither is used yet.

**Data seen:** only the Faraday figures above, plus the other notes on pages
53–58 (receivables, leases, staff costs, other income). Nothing relating
Faraday to PLAB.

Under the stop rule this document is now amended to use Faraday's quarterly
提供勞務 revenue as the predictor. Open before any model is fitted:

1. **Label mapping.** 提供勞務 is "provision of services", not literally
   "NRE". Confirm it matches the NRE Faraday reports in its quarterly results
   for at least two quarters where both exist (2Q22 onward).
2. **Availability date.** The financial statements are the certain public
   date (2018 Q1: April 20; 2018 Q4: February 26 of the next year). Whether an
   earlier date — the quarterly results release — may be used depends on that
   release stating the same figure.
3. **Alignment.** Faraday's calendar quarter Q against PLAB's fiscal quarter
   ending one month later, and the row-3 exclusion when PLAB reports first.
4. **Foundry shift.** A rule, fixed now, for the years Faraday moved most
   purchases away from UMC.
