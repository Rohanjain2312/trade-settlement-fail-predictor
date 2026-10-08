# The 20 features

All data is synthetic. `src/features/definitions.py` is the single source of truth for the
names, types, and order below; `tests/test_docs.py` checks that this page stays in sync.
**Planted strength** is how strongly the generator ties a feature to the fail probability.
Never model inputs: `trade_id`, `trade_date`, `settle_date`, `desk_id`, `cpty_id`, `sec_id`,
raw `notional`, `failed`, `fail_reason`, `scenario_mask`.

| # | Feature | Type | What it is | How it affects settlement | Planted strength |
|---|---|---|---|---|---|
| 1 | `ssi_match_status` | categorical (matched, mismatch, missing) | Whether the account, custodian, and BIC details on the instruction agree with the standing settlement instructions on file | A mismatch or missing instruction means cash and securities cannot be routed, so the instruction is rejected or sits unmatched until someone repairs it. SSI problems are the largest single cause of fails | strong |
| 2 | `ssi_age_days` | numeric | Days since the counterparty's SSI record was last verified | Old records are more likely to point at closed or changed accounts, so risk rises with age even while the record still shows as matched | medium |
| 3 | `instruction_hour_bucket` | categorical (overnight, early morning, business hours, late) | Time of day the settlement instruction was submitted | Overnight and early-morning instructions go in with less market visibility and no one on hand to fix exceptions, and late ones miss depository cutoffs | medium |
| 4 | `hours_to_confirmation` | numeric | Hours from execution to counterparty confirmation | Slow confirmation leaves less time to catch breaks before settlement date, and an unconfirmed trade is the classic "counterparty does not know the trade" fail | strong |
| 5 | `amendment_count` | integer | Number of amends or cancel-and-correct events after execution | Every amendment restarts matching and can reintroduce a break in price, quantity, or date | medium |
| 6 | `allocation_delay_hrs` | numeric | Hours from block execution to allocation across the underlying accounts (0 for trades that are not blocks) | Late allocation delays the settlement instruction and squeezes the time left to fix problems | medium |
| 7 | `chain_depth` | integer | Number of upstream trades whose delivery this trade depends on (0 means none) | A fail upstream cascades down the chain, so the deeper the chain, the more places a fail can start | medium |
| 8 | `obligation_coverage_ratio` | numeric | Share of what is owed on settlement date (securities for sells, cash for buys) that is already available or confirmed incoming | A ratio below 1 is a shortfall. Not having the securities or cash to deliver is the second biggest cause of fails | strong |
| 9 | `cpty_fail_rate_30d` | numeric | Share of this counterparty's settled trades that failed in the prior 30 days (smoothed toward the overall rate when there are few trades) | Counterparties with recent fails tend to keep failing because their operational problems persist | strong |
| 10 | `cpty_type` | categorical (custodian, broker-dealer, asset manager, hedge fund, corporate treasury) | Kind of counterparty | Different types have different levels of automation and operational maturity, which shifts the baseline fail rate regardless of recent history | medium |
| 11 | `pair_history_trades_90d` | integer | Trades between this desk and this counterparty in the prior 90 days | Pairs that trade often have tested instructions and routines. New or rare pairs go through manual setup and have more breaks | weak |
| 12 | `security_fail_rate_30d` | numeric | Fail rate of this security over its settled trades in the prior 30 days (smoothed toward the overall rate when there are few trades) | Scarce or hard-to-source securities fail repeatedly, for example when they trade on special | medium |
| 13 | `corporate_action_in_window` | binary | A dividend, coupon, or record date falls inside the settlement window | Entitlement and position adjustments around events create holds and claims that delay settlement | medium |
| 14 | `notional_vs_cpty_median` | numeric | Trade size divided by the counterparty's median trade size over the prior 90 days | Unusually large trades are harder to source or fund and more often get manual checks | medium |
| 15 | `abs_price_deviation_bps` | numeric | Absolute distance of the trade price from the market reference at execution, in basis points | Off-market prices are more likely to be disputed at the matching step | weak |
| 16 | `asset_class` | categorical (equity, corporate bond, government bond, ETF, repo) | Type of instrument | Each class settles through different infrastructure and rules, with different baseline fail rates | medium |
| 17 | `is_cross_border` | binary | Settlement involves a market or depository outside the home market | Extra intermediaries, time zones, and local rules add failure points | medium |
| 18 | `market_volatility_level` | numeric | Volatility index level on the trade date | Volume spikes and strained operations in volatile periods push fails up | medium |
| 19 | `holiday_in_settlement_window` | binary | A market holiday falls between trade date and settlement date in either market | Holidays shrink the working window and shift cutoffs, especially across markets | weak |
| 20 | `is_period_end` | binary | Settlement date is a month-end or quarter-end | Balance-sheet and funding pressure plus higher volumes at period end raise fails | weak |

## Groups

SSI and instructions (1 to 3), matching and timing (4 to 7), obligation (8), counterparty (9 to 11),
security (12, 13), trade economics (14 to 16), market and calendar (17 to 20).

## Planted interactions

- `ssi_match_status` = mismatch together with `is_cross_border`
- low `obligation_coverage_ratio` together with high `security_fail_rate_30d`
- high `hours_to_confirmation` together with an overnight `instruction_hour_bucket`

## Keeping the features distinct

Each feature comes from a different source field or process step, and no feature is computed from
another. Raw notional is excluded so it does not duplicate the size ratio. Related features measure
different things: SSI match status is whether the instruction matches today, SSI age is how stale the
record is; confirmation delay is the counterparty step, allocation delay the internal step; the
counterparty fail rate is recent behavior, the type a static category, the pair history how familiar
the relationship is. The generator's checks require numeric pairs below 0.6 absolute Spearman
correlation, categorical pairs below 0.5 Cramer's V, mixed pairs below 0.5 correlation ratio, and a
variance inflation factor below 5.

## Rolling features use no future information

`cpty_fail_rate_30d` and `security_fail_rate_30d` count only trades whose settlement date is before
the trade date, so a trade that has not settled yet has no known outcome. `pair_history_trades_90d`
and the median behind `notional_vs_cpty_median` use trades from earlier trade dates only. The data
checks recompute these from raw history on a sample and compare.
