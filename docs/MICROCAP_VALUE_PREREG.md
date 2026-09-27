# Micro-cap value — pre-registration draft (2026-09-27)

**Status: design only. Nothing has been run, and no data has been acquired.**
Every choice below was fixed on paper before any backtest exists. Changing one
after seeing results turns this into a second draw; see "Confirming a future
pass" in `EDGE_INVESTIGATION_2026-09-08.md`.

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
| 1 | Universe | US stocks, $50M–$300M market cap, built **point-in-time** (including companies later delisted); average daily dollar volume ≥ $100k; **excluding financials** (banks, insurers, REITs) |
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
   would differ in sector mix as well as signal.
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

## Open items — must be fixed before the first run

- **Data source.** Point-in-time prices, fundamentals with filing dates, and
  delisting returns for US micro-caps. Yahoo has none of these. Cost and
  coverage decide how many years the test gets.
- **Spread source.** "Half the spread" needs historical bid-ask spreads, which
  are hard to get for micro-caps. If the vendor lacks them, pick an estimator
  now (for example one based on daily high/low prices) and state it here.
- **Rebalance month and filing lag.** Which month, and how the filing date is
  determined (a vendor's filing date, or a fixed lag after fiscal year end).
- **Delisting return.** What a position returns when its stock delists
  mid-year — the vendor's delisting return if it has one, otherwise a fixed
  assumption stated here.
