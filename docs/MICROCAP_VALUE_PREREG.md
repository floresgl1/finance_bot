# Micro-cap value — pre-registration draft (2026-09-27)

**Status: design only. Nothing has been run, and no data has been acquired.**
Every choice below was fixed on paper before any backtest exists. Changing one
after seeing results turns this into a second draw; see "Confirming a future
pass" in `EDGE_INVESTIGATION_2026-09-08.md`.

*Amended 2026-09-27, still before any data exists:* the universe now names
which securities qualify (CRSP share and exchange codes), and "Identifiers and
joins" fixes how prices and fundamentals are matched.

*Amended 2026-09-29, still before any data exists:* every open item is now
decided, and "Implementing on Sharadar" states how each rule is met with
Sharadar data. At the time of writing only Sharadar's free sample had been
accessed, and only to plan this amendment; no returns, rankings or results
had been computed from any source.

*Amended again 2026-09-29, still before any data exists:* Sharadar is used
through its own API (`api.sharadar.com`), not Nasdaq Data Link, and its field
names differ from those first written here. The free tier covers only the 30
Dow stocks, so the structural checks cannot run on it; they now run on the
downloaded data, before any return or ranking is computed. A fifth check
covers the ticker join. The only data seen was public ticker metadata and
Sharadar's AAPL-only demo key.

*Amended a third time 2026-09-29, after the download and before any check,
return or ranking:* "Rules fixed before the structural checks" states how
unknown delistings, liquidity, market cap and year coverage are measured, and
turns check 1 into a hand verification. The files used are recorded by hash in
`docs/sharadar_manifest.json`. Data seen so far: row counts, date ranges,
table and category names, and the layout of exchange-change events.

## Why this, and not more of the current bot

Nine pre-registered probes found no edge in daily technical signals on twelve
mega-caps (`EDGE_INVESTIGATION_2026-09-08.md`). That is the most crowded corner
of the market: every large fund trades the same information. The rule this
design follows instead is **go where size is a disadvantage.** Micro-caps are
too small for institutional money to trade, so an account of ~$10k faces less
competition there.

## The strategy

| # | Component | Decision |
|---|---|---|
| 1 | Universe | Ordinary shares of US companies — CRSP **share code 10 or 11**, **exchange code 1, 2 or 3** (NYSE, AMEX, Nasdaq) — with $50M–$300M market cap, built **point-in-time** (including companies later delisted); average daily dollar volume ≥ $100k; **excluding financials** (banks, insurers, REITs); must have EBIT and EV available as of the rebalance date |
| 2 | Signal | Rank by **EBIT / enterprise value**, highest first, using only filings public before the rebalance date |
| 2a | Filter | **Net debt / EBIT ≤ 3×** |
| 3 | Portfolio | Top **30** names, equal weight |
| 4 | Rebalance | **Yearly**; cost = half the spread per side, per stock |
| 5 | Control | Equal-weight basket of the **whole eligible universe** — same filters, same dates, same cost model |

### Why each choice

1. **Universe.** Point-in-time because a universe built from today's listings
   has already removed every company that went bankrupt, and the cheapest
   stocks — exactly the ones this strategy buys — delist most often. A backtest
   on survivors can show 20%+ a year with no edge at all. The liquidity floor
   is about spreads, not capacity: the thinnest names trade at 5%+ spreads,
   which would consume a year's edge in one round trip. Financials are
   excluded *from the universe*, not just filtered by the debt rule, because
   debt is their raw material and both metrics are meaningless for them; and
   because if the strategy dropped them while the control kept them, the two
   would differ in sector mix as well as signal. Share codes 10/11 keep out
   ETFs, closed-end funds and ADRs, which a market-cap filter lets through:
   ETFs have no EBIT, and ADRs report under foreign rules that North American
   fundamentals data mostly does not cover. Exchange codes 1–3 keep out stray
   over-the-counter listings. A security without EBIT and EV on the rebalance
   date is excluded from the universe, so it is missing from strategy and
   control alike — never sorted to one end of the ranking by a missing value,
   and never held by the control alone.
2. **Signal.** Cheapness has two independent explanations — compensation for
   the risk of holding troubled companies, and investors over-reacting to bad
   news in names no analyst covers — so the effect has a reason to exist.
   EBIT/EV instead of P/E because many micro-caps lose money, which makes P/E
   meaningless at both ends; EBIT/EV puts every company on one scale and
   counts debt in the price.
2a. **Debt filter.** A $100M company cannot borrow its way through a bad year;
   default leads to delisting. A positive-EBIT filter was considered and
   rejected: the top of an EBIT/EV ranking already has positive EBIT, so it
   would filter out nothing the strategy holds.
3. **30 names.** Roughly the top few percent of the eligible universe, so still
   selective, while one blowup cannot decide the result. About $330 a name at
   $10k: far too small to move a price.
4. **Yearly.** Annual cost ≈ round-trip spread × round trips per year. At a
   1.5% round trip, weekly rebalancing costs ~78% a year and monthly ~18%; only
   a slow strategy leaves room for an edge.
5. **Control.** Against SPY, the strategy could win by being in micro-caps
   during a good decade for micro-caps. The control differs from the strategy
   in exactly one respect — which names are picked — so any gap is the signal.

## Identifiers and joins

Tickers are never used as identifiers: they are reused across unrelated
companies and change when a company renames. Joining on them attaches one
company's data to another's years, or drops a delisted company entirely.

- **Securities** are identified by CRSP **PERMNO**; **companies** by Compustat
  **GVKEY**.
- Fundamentals attach to prices only through the **CRSP–Compustat (CCM) link
  table**, and only where all three hold:
  - `linktype` is `LU` or `LC` (the reliable link types);
  - `linkprim` is `P` or `C` (the primary security, so a company with two
    share classes enters once, not twice);
  - the rebalance date lies within `linkdt`–`linkenddt` (a link is valid only
    for its dates; mergers and share-class changes move it).
- A fundamentals row is usable on a rebalance date only if it was public by
  then (see "Filing lag" under the decisions below).

## Pass criteria

All three must hold, net of costs:

1. **Beats the control in ≥ 80% of years** (8 of 10 with ten years of data).
2. **Quintile staircase, no swaps.** Split the eligible universe into five
   quintiles by EBIT/EV each year. Mean yearly return must be strictly ordered:
   cheapest highest, most expensive lowest. A useless signal produces this
   ordering by chance about 1 time in 120.
3. **Mean excess return over the control ≥ 3% a year.**

Why the bar is stricter than the earlier probes': yearly rebalancing gives one
observation per year, not hundreds of days. With no edge, each year is a coin
flip:

| Bar | Chance a useless signal passes |
|---|---|
| wins ≥ 7 of 10 years | 17% |
| wins ≥ 8 of 10 | 5.5% |
| wins ≥ 9 of 10 | 1.1% |

The staircase adds evidence the win count cannot: it tests whether cheapness
itself orders the returns, not just whether one portfolio got lucky.

A pass is a lead, not a result. It goes through the confirmation protocol on
live data before any real money.

## Decisions on the former open items (2026-09-29)

These rules apply whichever data source is used.

- **Data source.** Sharadar (sharadar.com's own API): active and delisted US
  companies, point-in-time fundamentals with filing dates, from the late 1990s.
  WRDS (CRSP/Compustat) access is being pursued; if granted, the test is also
  run on CRSP/Compustat under the rules above, and a pass must hold on both.
  Alternatives rejected: SEC EDGAR XBRL has no prices, covers small companies
  only from 2011, and would need a hand-built EBIT from raw tags — errors there
  would reorder the ranking and look like signal. Norgate has only current
  fundamentals, not what was known on past dates.
- **Rebalance date: July 1** each year (the next trading day if closed). By
  then even late December-year-end filers are past their deadline, and it is
  the standard academic timing, so results stay comparable to published ones.
- **Filing lag.** A fundamentals row is usable on rebalance date D only if its
  filing date is strictly before D; filings often arrive after the close.
- **Staleness.** A company whose latest usable filing covers a fiscal year that
  ended more than **18 months** before D is excluded from the universe, for
  strategy and control alike. Late filing is a warning sign; without this rule
  the ranking would value a troubled company on pre-trouble EBIT against a
  post-trouble price, making it look cheap just before it fails.
- **Enterprise value** is computed on D from that day's market cap plus the
  latest usable filing's debt minus cash. A vendor's precomputed EV is not
  used: it is dated at the filing, so companies filing in different months
  would be ranked on prices from different days.
- **Market cap** is the whole company's, summed across all share classes, for
  both EV and the $50M–$300M filter. Using one class's shares would understate
  EV and make dual-class companies look cheaper than they are.
- **Delisting return**, applied on the delisting day, by reason:

  | Reason | Return beyond the last close |
  |---|---|
  | Acquired or merged away | 0% (a buyout's last trade is near the deal price) |
  | Bankruptcy, regulatory delisting, voluntary, or unknown | **−30%** (Shumway) |

  Unknown reasons take the loss, so unclear data makes the backtest look worse,
  not better.
- **Cash from a delisting** is reinvested the same day into the remaining
  holdings in proportion to their current weights, paying half the spread on
  those buys. The control follows the same rule. Holding the cash instead
  would leave the strategy — whose cheap, troubled stocks delist more often —
  holding more cash than the control, and the gap would partly measure cash
  held rather than the signal.
- **Spread**, when the source has no bid-ask data: the Abdi–Ranaldo (2017)
  estimator on split-adjusted daily high, low and close.
  - Two-day estimates, negatives set to 0, averaged over the **252 trading
    days before** the trade date. Only data known on the trade date is used,
    and the trailing year exists even for a stock that delists soon after.
  - Floor of **0.5%**: estimators undershoot for quiet stocks, and too little
    cost flatters the strategy.
  - Fewer than 126 valid days: **2 × the universe median** on that date.
    Unknown liquidity is treated as poor liquidity.
  - The cost of each trade is half the estimated spread, as in the strategy
    table.

## Implementing on Sharadar

CRSP codes named above have Sharadar equivalents as follows. Table names are
those of `api.sharadar.com/v1.0/data/`: `tickers`, `fundamentals`, `stocks`,
`actions`.

| Rule | Sharadar implementation |
|---|---|
| Identity (PERMNO, GVKEY, CCM link) | `permaticker` is the identity. Only the `tickers` table carries it; `fundamentals` and `stocks` carry only `ticker`, so each row is joined to its `permaticker` through `tickers`. This is safe only because Sharadar renames an old ticker when it is reused (e.g. `AAC2`), keeping each ticker unique in its database — which check 5 verifies. Tickers from any other source are never used. |
| Share code 10/11 | `category` is `Domestic Common Stock` or `Domestic Common Stock Primary Class`. `Domestic Common Stock Secondary Class` is excluded, so a dual-class company enters once — the job CRSP's `linkprim` does. ADR and Canadian categories are excluded. |
| Exchange code 1–3 | **Not implemented from Sharadar's `exchange` field.** It is a snapshot of the last known exchange: a company demoted to OTC before failing would be dropped from every earlier year too, deleting future losers from past rankings. Eligibility rests instead on the point-in-time size and liquidity filters on each rebalance date. |
| Financials | `siccode` 6000–6999 excluded. |
| Blank-check shells | Company-years whose latest usable filing shows **zero revenue** are excluded. A SPAC that later merges takes the operating company's SIC, so the snapshot `siccode` misses its shell years; shells trade flat near $10 at $100M–$300M and would dilute the control. This also drops pre-revenue companies, from strategy and control alike. |
| Fundamentals as known at the time | `fundamentals` rows with dimension **`ARY`** (as first reported) only, never `MRY`: a restatement published later must not appear in an earlier ranking. The filing date is the field `date` (Sharadar's "Date Key": for `ARY`, the date the filing reached the SEC). EBIT, debt and cash are the raw fields `ebit`, `debt`, `cashneq`; the vendor's `ev`, `evebit` and `marketcap` are not used (see "Enterprise value"). |
| Total return | Daily change in `stocks.closeadj`, which is adjusted for splits and dividends. |
| Delisting reason | The `actions` event that ends the listing, mapped to the table above. |
| Date coverage | Date-filtered queries default to the last year only, and the `from`/`to` filter did not behave as documented in a test with the demo key. The download therefore states its date range explicitly, and the loader verifies that every year of the test has rows before anything else runs. |

## Structural checks — before any result

The free tier covers only the 30 Dow stocks, which contain no micro-caps,
SPACs or failures, so these checks run on the **downloaded** data. They
examine structure only: fields, identifiers, categories, event names. They run
before any return, ranking or result is computed. If one fails, the affected
rule is amended here, dated, **before** any result exists.

The download must also prove it is the paid data: a wrong API key returns
HTTP 200 with limited data, not an error. The loader stops if one page of
`ARY` fundamentals holds fewer than 100 distinct tickers.

1. **Dual-class share counts.** For a known dual-class company, does the share
   count in the fundamentals cover all classes? If not, market cap must be
   summed across the classes' price rows.
2. **Demoted stocks.** Does `stocks` carry prices after a stock moves to OTC,
   and do any such names pass the $100k ADV floor? If so, the dropped exchange
   filter needs a point-in-time replacement.
3. **SPAC shell years.** What `siccode` and revenue does a merged SPAC show
   for its shell years? If Sharadar flags shells directly, that flag replaces
   the zero-revenue rule.
4. **Delisting event names.** Which `actions` values mark an acquisition and
   which a failure? The mapping to 0% / −30% is fixed from these names.
5. **Ticker join.** Every ticker in the downloaded `fundamentals` and `stocks`
   rows maps to exactly one `permaticker` in `tickers`. A ticker with none is
   reported; a ticker with two stops the run.

Seen in the public `tickers` table on 2026-09-29 (metadata only, visible to
any key): 17,854 companies with fundamentals, 12,325 of them delisted; no
ticker mapped to two permatickers; no `Secondary Class` category among them;
many SPACs listed as `Primary Class` (their Class A/Class B structure), which
the SIC and zero-revenue rules remove. These observations do not replace the
checks, which run on the downloaded rows.

## Rules fixed before the structural checks (2026-09-29)

Written after the download and before `microcap_checks.py` first ran. In the
bulk files the `tickers` table labels fundamentals companies `SF1` and priced
securities `SEP`.

- **Delisting reason.** A `delisted` event takes its reason from an event on
  the same ticker within **30 calendar days** either side. Acquired
  (`acquisitionby`, `mergerto`, `spacmerger`): 0%. Failed
  (`bankruptcyliquidation`, `regulatorydelisting`, `voluntarydelisting`):
  −30%. If both kinds are present, failure wins. No reason event: unknown.
- **Unknown delistings: primary rule and sensitivity run.** The primary run
  gives unknown delistings −30%, and only a pass under it counts. A second run
  gives them 0%. Unknowns concentrate in the strategy, whose cheap stocks both
  fail and get taken over more often than the control's, so −30% biases the
  test against a pass; the second run shows by how much. **Fails at −30% but
  passes at 0% means inconclusive, never a pass**: the next step would be to
  find out what those delistings were, not to pick the rule that wins.
- **Liquidity (ADV).** Mean of `close × volume` over the **252 trading days
  before** D; both are split-adjusted, so their product is the real dollar
  volume. A security with fewer than 126 days of volume in that window is not
  eligible: its liquidity cannot be assessed.
- **Market cap on D** = `close(D) × sharesbas × closeunadj(f) / close(f)`,
  where f is the trading day on or before the filing's `date`. `close` is
  split-adjusted but `sharesbas` is the count as reported, so after a later
  split the adjusted price alone would understate market cap; the ratio
  `closeunadj(f) / close(f)` puts the reported count on the adjusted basis.
- **Year coverage.** The first rebalance is July 1999 (it needs a year of
  prices from the first date, 1997-12-31) and the last is July 2025 (its year
  ends July 2026). The data must have price rows and `ARY` filings in every
  calendar year 1998–2025, and prices through 2026-07-01. A gap stops the run.
- **Check 1 becomes a hand verification.** Sharadar has no per-class share
  counts, so the data cannot answer it. `microcap_checks.py` lists dual-class
  companies (a `Secondary Class` security whose issuer is an `SF1` company);
  the researcher checks **three** of them against the share counts on the
  cover page of their 10-K on SEC EDGAR and records the result here. If
  `sharesbas` misses a class in any of the three, market cap is amended before
  any result.
- **Checks 2–5 report counts only.** Check 5 fails the run if any ticker maps
  to two permatickers. Checks 2–4 inform the rules above; a rule they show to
  be wrong is amended here, dated, before any result.

## Structural check results and the rules they changed (2026-09-29)

`microcap_checks.py` on the downloaded data. No return, ranking or EBIT/EV had
been computed.

| Check | Result |
|---|---|
| Guard | 17,687 distinct tickers in `ARY` fundamentals: the paid universe |
| Coverage | Every year 1998–2025 has ≥ 248 trading days of prices and ≥ 5,502 `ARY` filings; prices run to 2026-09-29 |
| 5. Ticker join | No ticker maps to two permatickers; no unmapped rows |
| 4. Delisting reasons | 11,356 delistings of eligible common stock: acquired 62%, failed 37%, **unknown 1%** (151). The −30% / 0% sensitivity run is kept, and is expected to change little |
| 3. SPAC shells | All 818 merged SPACs now carry a non-6770 SIC; only 71% of their 1,737 pre-merger filings show zero or missing revenue |
| 2. Demoted stocks | Of 1,453 moves to OTC, 1,034 kept trading, and 725 (ticker, rebalance) pairs on OTC passed the $100k ADV floor |
| 1. Dual-class | 154 companies (65 listed); verified by hand below: `sharesbas` is the total of all classes |

Sharadar also files most old SPAC units under `Secondary Class` (1,137 of
1,339), renamed after the company the SPAC merged into. The universe already
excludes `Secondary Class`, so this changes only how check 1 finds real
dual-class companies (by issuer name, unit tickers skipped).

Two rules change, as the checks section specified. Approved by the author
2026-09-29, before any backtest code existed:

- **A. Exchange (from check 2).** The liquidity floor does not keep OTC stocks
  out, so a point-in-time exchange rule replaces it for that purpose: a
  security is **not eligible on D if its most recent `exchangeto` event before
  D moved it to OTC**. Built from dated events, not the snapshot `exchange`
  field, so a company that later drops to OTC stays eligible in its earlier
  exchange-listed years.
- **B. SPAC shells (from check 3).** Sharadar flags shells directly, so the
  flag replaces the zero-revenue rule: a company is a shell, and not
  eligible, **on any D before its `spacmerger` date, or while its SIC is
  6770**. The zero-revenue rule is dropped. Pre-revenue companies therefore
  return to the universe; with negative EBIT they rank at the bottom, so they
  enter the control, not the strategy.

**Check 1, verified by hand 2026-09-29.** Each company's latest `sharesbas`
against the share counts on the cover page of the same 10-K on SEC EDGAR:

| Company | 10-K filed | Classes on the cover page | Sum | `sharesbas` |
|---|---|---|---|---|
| Bel Fuse | 2026-02-24 | A 2,115,263 · B 10,541,050 | 12,656,313 | 12,656,313 |
| Bio-Rad | 2026-02-13 | A 21,924,284 · B 5,066,110 | 26,990,394 | 26,990,394 |
| Central Garden & Pet | 2025-11-26 | Common 9,650,221 · A 51,080,111 · B 1,602,374 | 62,332,706 | 62,332,706 |

All three match exactly, including a three-class company: `sharesbas` is the
total of every class. The market-cap rule stands unchanged.

## Backtest decisions, fixed before any backtest code (2026-09-29)

Approved by the author 2026-09-29. The backtest is written and tested on
synthetic data only; its code is then frozen in one commit, and only then run
once on the real data. A code change after the run makes it a second draw.

1. **The debt filter is part of the universe.** Net debt / EBIT ≤ 3× applies
   to strategy, control and quintiles alike. Applied to the strategy alone,
   the strategy would differ from the control in cheapness *and* leverage,
   and a gap could not be attributed to the signal. Where EBIT ≤ 0 the ratio
   is meaningless: such a company passes only if its net debt is ≤ 0.
2. **EV ≤ 0 is excluded** from the universe: EBIT/EV changes sign and would
   rank nonsense at the top.
3. **Missing EBIT, debt, cash or `sharesbas`: excluded.** EV must be
   available (universe rule); a missing value is never filled with 0.
4. **Trades at the close on D.** The ranking uses only data from before D;
   holding-year returns start the next trading day.
5. **Rebalancing cost** = half spread × the weight actually traded, per name:
   names kept from last year pay only on their trim or top-up; names dropped
   pay on the sale; new names pay on the purchase. Each year's return includes
   the cost of its opening trades.
6. **A price series that ends inside a holding year** — no later price
   anywhere in the data — is a delisting on its last price date. Its reason
   comes from `actions` events on that ticker within 30 days of that date
   (failure wins); none means unknown. A gap followed by more prices is not a
   delisting: the position keeps its last price until trading resumes.
7. **The 80% bar over 27 years (1999–2025) means ≥ 22 winning years**
   (21.6 rounded up).
8. **Ties in EBIT/EV are broken by `permaticker`**, ascending.
9. **Market cap uses the primary class's price** × `sharesbas` (all classes).

Clarifications the code needs, same date:

- A day counts toward the ADV and spread windows only if it has
  `volume > 0` (and, for the spread, high, low and close on it and the next
  day). Trades after a delisting use the same trailing-252-day spread rule,
  evaluated on the day of the trade.
- The staircase uses the same net-of-cost returns as the other criteria.
