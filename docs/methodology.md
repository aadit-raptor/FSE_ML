# Methodology — how every number is calculated

This is the written methodology of Variater's models (PLAN.md 3.2). It says
what each calculation does, in the order the code does it, with a link to the
code and to the tests that prove it. Where a country, currency or accounting
standard changes the answer, the section says so under **Regional
differences**.

It describes the code as it is, including its known shortcuts: those are
listed where they bite and again under [Known limitations](#known-limitations).
A change that moves a number is a new engine version
([MODEL_CHANGELOG.md](../MODEL_CHANGELOG.md)); this document changes in the
same pull request.

`tests/test_methodology.py` keeps the document honest: it fails when a
function in the model packages is not named here, when a link is broken, or
when a constant or a Settings default quoted below differs from the code.

**Hand-checked reference cases** ([tests/reference/](../tests/reference/README.md),
PLAN.md 3.4) prove the calculations described here: 26 small deals worked out
by hand from this document, in a committed spreadsheet, which
[tests/test_reference_cases.py](../tests/test_reference_cases.py) requires the
deal model to match to 0.01.

## Contents

1. [Conventions](#1-conventions)
2. [Inputs, units and Settings](#2-inputs-units-and-settings)
3. [The deal model](#3-the-deal-model)
4. [The deal around the engine](#4-the-deal-around-the-engine)
5. [Debt structures and interest rates](#5-debt-structures-and-interest-rates)
6. [Accounting standards and leases](#6-accounting-standards-and-leases)
7. [Tax rules](#7-tax-rules)
8. [Risk warnings](#8-risk-warnings)
9. [Monte Carlo simulation](#9-monte-carlo-simulation)
10. [Plan vs actual](#10-plan-vs-actual)
11. [Three-statement forecast](#11-three-statement-forecast)
12. [Model version](#12-model-version)
13. [Estimates: the ML panels and filings](#13-estimates-the-ml-panels-and-filings)
14. [Figures made for display](#14-figures-made-for-display), and [a new deal's starting figures](#starting-figures)
15. [Regional differences at a glance](#15-regional-differences-at-a-glance)
16. [Known limitations](#known-limitations)
17. [Constants](#17-constants) and [Settings defaults](#18-settings-defaults)
18. [Appendix: functions that affect no number](#appendix-functions-that-affect-no-number)

---

## 1. Conventions

- **Years.** A deal runs for a holding period of *n* whole years. Year 1 is
  the first full year after close; the exit is at the end of year *n*. There
  are no stub periods and no mid-year convention. Fiscal years are labels
  only (`fiscal_year_end_month`, `first_fiscal_year`): they rename the
  columns ("FY2025", "FY2024/25", [web/src/lib/fiscal.ts](../web/src/lib/fiscal.ts))
  and never change a figure.
- **Units inside the engine.** Rates and shares are decimals (`0.065`), money
  is in **millions** of the deal's currency, whatever unit the deal is entered
  in (section 2). Answers keep engine units: an IRR of `0.157` is 15.7%.
- **Rounding.** The engine rounds money to two decimals of a million at each
  line (`round(x, 2)`), and margins to six decimals. This is a known
  limitation for small deals ([finding 10](#known-limitations)).
- **Interest on opening balances.** Every facility pays interest on its
  balance at the start of the year.
- **One currency per deal.** The model never reads the currency code; it is a
  label carried with every money figure. There are no exchange rates yet
  (PLAN.md 4.2).

## 2. Inputs, units and Settings

Code: [core/money.py](../core/money.py), [core/config.py](../core/config.py),
[core/deal.py](../core/deal.py). Tests: [tests/test_money.py](../tests/test_money.py).

**Money units.** A deal is entered in thousands, millions or billions of any
ISO 4217 currency. Before any model runs, the router converts it to millions:
`to_millions(amount, unit)` is `amount × scale / 10⁶` (scale 10³, 10⁶ or 10⁹),
and `in_unit` is the inverse. `in_millions` converts a deal's money inputs
(EBITDA, minimum cash, tax allowances, lease cost and liability, each
tranche's `TrancheSpec.in_millions`: amount and custom amortisation) and the
money Settings (`other_uses`, `def_ebitda`, `def_mincash`). The answer is
converted back with `rescale`, which multiplies every leaf under the keys
named in the answer's money-key list (`_scale_all` does the walk) and leaves
everything else alone. So a deal in thousands answers exactly as in millions,
times a thousand. `round_money` rounds a figure to a number of decimals of a
million expressed in the deal's own unit (the forecast's fixed amounts use it).
The other converters are `mc_in_millions` (Monte Carlo), `backtest_in_millions`
(the old backtest) and `actuals_in_millions` (plan vs actual).

**Settings.** Every model reads a Settings mapping: `resolve_config` starts
from `DEFAULTS` (listed in [section 18](#18-settings-defaults)), applies the
account's or the deal's overrides and rejects unknown keys. The correlation
matrix of the simulation is rebuilt from ten Settings by `build_corr_matrix`;
`is_valid_corr` accepts it only if a Cholesky factorisation exists (the matrix
is positive definite), otherwise the API refuses the run. The defaults are
**illustrative, not market data**; the screens say so (PLAN.md 2.1).

## 3. The deal model

The deterministic engine, [lbo_engine/](../lbo_engine/). One call,
`run_lbo(params)` in [lbo_engine/model.py](../lbo_engine/model.py), takes an
`LBOParams` and returns an `LBOResult` (`LBOResult.irr`, `LBOResult.moic`,
`LBOResult.entry_equity`, `LBOResult.exit_equity` read its returns).
Tests: [tests/test_core_parity.py](../tests/test_core_parity.py) and
[tests/test_api.py](../tests/test_api.py) pin it to the golden snapshot
([tests/golden/golden.json](../tests/golden/golden.json));
[tests/test_model_fixes.py](../tests/test_model_fixes.py) pins each finding fix;
[tests/model_version_pins.json](../tests/model_version_pins.json) pins
reference deals per engine version.

The default deal (EBITDA 100, entry 10x, exit 11x, 5 years, 60% debt of which
70% senior at 6.5%, 4% mezzanine spread, Settings defaults) is used as the
worked example below; its IRR is 21.16% and its MOIC 2.61x.

### 3.1 Entry: enterprise value, debt and the equity cheque

```
entry EV      = (entry EBITDA + lease add-back) × entry multiple
entry costs   = entry EV × transaction fee % + total debt × financing fee %
                + other uses + per-tranche upfront fees
sponsor equity = entry EV + entry costs + minimum cash − total debt − lease liability
```

The lease add-back and liability are zero unless the deal has leases on the
post-IFRS 16 view (section 6). Minimum cash is funded by the sponsor at close
(finding 1). Worked example: EV 1,000; debt 600; fees 23.0 + 15.6 = 38.6;
equity 438.6.

Without an explicit tranche list, debt is the two-bucket structure of
`build_simple_two_tranche_structure` in
[lbo_engine/capital_structure.py](../lbo_engine/capital_structure.py): senior
= EV × debt % × senior %, amortising `def_senior_amort` of its original
principal a year, swept first; mezzanine = the rest, a bullet at the end of the
hold, swept second, priced at the senior rate plus the spread. Both mature at
the hold. With a tranche list, the structure comes from `core/debt.py`
(section 5). A `CapitalStructure` exposes its totals: `CapitalStructure.total_debt`,
`CapitalStructure.total_fees`, `CapitalStructure.total_leverage_multiple`
(debt / LTM EBITDA), `CapitalStructure.blended_interest_rate` (amount-weighted
rate), `CapitalStructure.sweep_eligible_tranches` (sorted by sweep priority)
and `CapitalStructure.tranche_summary`. Per tranche, `Tranche.fee_amount` is
amount × fee %, `Tranche.net_proceeds` the amount less it.

The share-price route of a public-to-private deal lives in
[lbo_engine/transaction.py](../lbo_engine/transaction.py) (`build_transaction`,
`solve_sponsor_equity`): offer price = share price × (1 + premium), equity
value = offer price × diluted shares, EV = equity value + existing debt −
existing cash, and sponsor equity = total uses − debt raised − cash above the
minimum. `SourcesAndUses.total_sources`, `SourcesAndUses.total_uses` and
`SourcesAndUses.check` (their difference) prove it balances. The app's screens
use the EV-multiple route above; this route is used by `run_lbo_full_deal`.

### 3.2 The operating model (revenue to EBITDA)

[lbo_engine/operating_model.py](../lbo_engine/operating_model.py),
`run_operating_model`. The base revenue is backed out of EBITDA:
`base revenue = entry EBITDA / (gross margin − opex % + D&A %)` (a run
refuses a margin at or below zero). Then, each year *t*:

```
revenue_t = revenue_{t−1} × (1 + growth)
COGS      = revenue × (1 − gross margin)
gross profit = revenue − COGS
opex      = revenue × opex %
EBIT      = gross profit − opex
D&A       = revenue × D&A %
EBITDA    = EBIT + D&A
```

`build_generic_assumptions` turns the deal's scalars into these year lists
(`OperatingAssumptions._expand` repeats a scalar for each year).
`OperatingModelResult.exit_ebitda` and `OperatingModelResult.exit_revenue` are
the final year's figures. Worked example: base revenue 384.6, year 1 revenue
403.85, EBITDA 105.0.

### 3.3 The income statement below EBIT

`complete_income_statement`:

```
interest income = minimum cash × 0.5%
EBT             = EBIT − interest expense + interest income
tax             = max(EBT, 0) × tax rate        (no tax rules on)
                = the tax rules' answer          (section 7)
net income      = EBT − tax
```

Without tax rules a loss earns no tax credit and is not carried forward.

### 3.4 Levered free cash flow

[lbo_engine/cashflow_model.py](../lbo_engine/cashflow_model.py),
`run_cashflow_model` (assumptions from `build_generic_cashflow_assumptions`,
expanded by `CashFlowAssumptions._expand`):

```
capex      = revenue × capex %
ΔNWC       = revenue × NWC %         (positive = cash used)
levered FCF = net income + D&A + non-cash interest − capex − ΔNWC
```

Non-cash interest is the PIK part of coupons (section 5), zero otherwise.
This is cash after interest and tax and **before** any debt repayment; the
debt schedule repays from it. `CashFlowResult.total_fcf` and
`CashFlowResult.avg_annual_fcf` sum and average it.

**Working capital in days** (`effective_nwc_pct`, [core/deal.py](../core/deal.py)).
With the WSP option on, the flat NWC % is replaced by the change in working
capital implied by receivable, inventory and payable days at constant days:
`k = (AR days + (1 − gross margin) × (inventory days − AP days)) / 365`,
`NWC % of revenue = k × g / (1 + g)`.

### 3.5 The debt schedule and the cash sweep

[lbo_engine/debt_model.py](../lbo_engine/debt_model.py), `run_debt_model`.
Each year, for every tranche:

1. **Interest.** `coupon = opening balance × rate in year`
   (`Tranche.annual_interest`, `Tranche.rate_in_year`: the tranche's rate path,
   or its flat rate). A share `pik_share` of the coupon accrues to principal
   instead of being paid. A committed facility also pays
   `Tranche.commitment_fee` = undrawn × fee %, where `Tranche.undrawn` is
   commitment − balance. Interest expense = coupon + commitment fee; PIK is
   still an expense and shields tax.
2. **Mandatory repayment** (`Tranche.mandatory_repayment`): amortising =
   original principal × amortisation % a year; bullet = everything owed
   (opening balance plus this year's PIK) in its maturity year; custom = the
   schedule's amount. Never more than is owed.
3. **Cash available to sweep** = opening cash + levered FCF − mandatory
   repayments − minimum cash, floored at zero.
4. **Sweep**, in sweep-priority order among the tranches that take it: each
   takes `min(available × sweep share, what is left, its balance after
   mandatory repayment)`. The share is of the cash available, not of what the
   tranche above left.
5. **Closing balance** = opening − mandatory − sweep + PIK.
6. **Revolver draw.** If cash in hand less mandatory repayments falls short
   of the minimum, a tranche with `allow_redraw` draws up to its commitment to
   cover it.
7. **Closing cash** = minimum cash + whatever no tranche could take.

Step 7 sets cash to the minimum even when the year was short of cash and no
revolver covered it: the shortfall is funded out of nothing. This is
[finding 11](#known-limitations), and the "unfunded repayment" risk warning
(section 8) measures exactly that amount. The default deal hits it in year 5,
when the mezzanine bullet is repaid.

`DebtScheduleResult.net_debt_at_exit` = closing debt − closing cash in the
final year. `DebtScheduleResult.interest_expense` is the yearly total,
`DebtScheduleResult.ending_balance` a tranche's closing balance in a year and
`DebtScheduleResult.total_debt_repaid` the sum of mandatory repayments and
sweeps.

### 3.6 The interest circularity

Interest depends on balances, balances on cash flow, cash flow on net income,
net income on interest. `run_lbo` resolves it by iteration: a first pass with
zero interest gives a debt schedule; each further pass reruns the income
statement, the cash flow (with the previous pass's PIK, `_accrued`) and the
debt schedule with the previous pass's interest, until the largest change in
any year's interest is below the tolerance. The app runs up to
`MAX_INTEREST_PASSES` passes to a tolerance of `INTEREST_TOLERANCE` million
(1,000 of the currency; finding 8). `LBOResult.interest_converged` says whether
it got there.

### 3.7 Returns

[lbo_engine/returns.py](../lbo_engine/returns.py), `compute_returns`:

```
exit EV          = (final-year EBITDA + lease add-back) × exit multiple
gross exit equity = exit EV − (net debt at exit + lease liability)
net exit equity  = gross exit equity × (1 − management option pool)
MOIC             = net exit equity / sponsor equity
IRR              = the rate r with −equity + exit equity / (1 + r)ⁿ = 0
```

There are no interim dividends, so the IRR equals `MOIC^(1/n) − 1`. `_irr`
solves it with `numpy_financial.irr`, falling back to Newton's method
(`_irr_newton`) and, if that fails too, to the CAGR. A non-positive exit
equity gives MOIC 0 and IRR −100%. `ReturnsResult.irr_pct` is the IRR in per
cent and `ReturnsResult.value_created` exit less entry equity. Worked example:
exit EV 1,403.9, exit equity 1,145.2, MOIC 2.61x, IRR 21.16%.

### 3.8 The equity bridge

`compute_equity_bridge` splits the gain (exit equity − entry equity) exactly:

```
EBITDA growth       = (exit EBITDA − entry EBITDA) × entry multiple
multiple expansion  = (exit multiple − entry multiple) × exit EBITDA
deleveraging        = net debt at entry − net debt at exit
fees                = − entry costs
```

with net debt at entry = total debt − minimum cash + lease liability. The four
add up to the gain; the residual is zero since finding 1. `bridge_steps`
([core/deal.py](../core/deal.py)) orders them for the waterfall chart and
leaves out the fee step when it is under half a cent.

### 3.9 The exit sensitivity grid

`compute_exit_sensitivity_by_hold`: one column per holding period, one row per
exit multiple. **Each column is a full deal-model run at that hold**
(finding 7), and each cell recomputes the returns of that run at the row's
exit multiple. Rows and columns come from Settings
(`sensitivity_exit_multiples`: `sens_em_steps` evenly spaced multiples from
`sens_em_min` to `sens_em_max`; `sensitivity_holding_periods`: every year from
`sens_hp_min` to `sens_hp_max`). An explicit tranche structure is built with
rate paths `SENSITIVITY_HEADROOM` years longer than the hold so that every
column has a rate. The older `compute_exit_sensitivity` (one run's exit
figures at every hold) is used only by `run_lbo_full_deal` and the engine's
own self-test.

## 4. The deal around the engine

[core/deal.py](../core/deal.py): it turns the wizard's inputs (percentages as
numbers, money in the deal's unit) into `LBOParams` and adds the figures the
screens show beside the engine's. Tests:
[tests/test_api.py](../tests/test_api.py), [tests/test_money.py](../tests/test_money.py).

- `build_lbo_params` divides percentages by 100, sets the working-capital %
  (`effective_nwc_pct`), the tax rules (`rules_from_deal`), the lease terms
  (`leases_of`, `has_leases`), the tranche structure and its fees, the
  Settings' fees, amortisation and grid, and the convergence limits.
  `run_deal` runs it.
- `entry_costs` = EV × transaction fee % + debt × financing fee % + other uses.
- `capital_structure_from_multiples` converts the wizard's debt multiples
  (senior and mezzanine × EBITDA) to the debt % of EV (capped at 99%) and the
  senior share of debt.
- **Sources and uses.** `sources_and_uses` (percentage structure) and
  `sources_and_uses_for` (tranche list). Uses = EV − lease liability taken
  over + transaction fees + financing fees + tranche upfront fees + other uses
  + minimum cash; sources = drawn debt + sponsor equity, where sponsor equity
  = uses − debt (never below zero). A revolver's undrawn part is not a source.
  With leases, everything a percentage sizes is sized on the valuation EBITDA
  (section 6).
- `capital_structure_summary` lists each facility's drawn amount, commitment,
  multiple of EBITDA, year-one rate, maturity and shares (via `summary` in
  [core/debt.py](../core/debt.py)).
- `lease_summary` reports the lease view, the operating and valuation EBITDA,
  the EV and the net debt at entry including the liability.
- `risk_model_inputs` prepares the risk score's five inputs (section 13):
  entry multiple, debt / EBITDA, growth, EBITDA margin (gross margin − opex +
  D&A) and the blended rate (amount-weighted year-one rate of the tranches, or
  of senior and mezzanine).

## 5. Debt structures and interest rates

[core/debt.py](../core/debt.py) owns the market conventions; the engine gets a
plain rate path. Tests: [tests/test_debt_structures.py](../tests/test_debt_structures.py)
(each mechanic hand-checked), [tests/test_montecarlo_tranches.py](../tests/test_montecarlo_tranches.py).

A deal's debt is a list of `TrancheSpec`s (`coerce` builds them from the
request). Nine kinds — amortising and institutional term loans, unitranche,
second lien, senior notes, PIK notes, vendor loan, revolver, shareholder loan
— are **presets over the same fields** (`PRESETS`, `spec_from_kind`), not
separate calculations. The web mirrors them (`KIND_PRESETS`).

- **Size.** `TrancheSpec.drawn` = amount × drawn % (100% except a revolver);
  `TrancheSpec.undrawn` = amount − drawn; `TrancheSpec.upfront_fee` = amount ×
  upfront fee % (on the whole facility). `total_debt` sums drawn amounts,
  `financing_fees` sums upfront fees, `blended_rate` is the drawn-weighted
  year-one rate.
- **Pricing.** Fixed: the fixed rate every year. Floating:
  `TrancheSpec.rate_in_year` = `max(reference + shift, floor) + margin`, the
  floor applied to the reference **before** the margin, as loan agreements
  write it. `TrancheSpec.reference_in_year` reads the reference path, or the
  flat reference level; a path shorter than the run repeats its last year.
- **Into the engine.** `to_tranche` maps a spec to an engine `Tranche`:
  custom schedule, else amortising if it has an amortisation %, else bullet;
  a rate path long enough for every hold in the grid; the commitment only
  when part is undrawn. `build_capital_structure` does the list, naming
  duplicates with `unique_names`. The list order is the sweep order unless a
  priority is set.
- **Refusals.** `check_specs` refuses (422) a structure whose drawn debt
  exceeds EV plus fees, which would leave a negative equity cheque.
- **Writing out a percentage deal.** `equivalent_tranches` writes the
  two-bucket structure as two floating tranches — senior at the base rate,
  amortising, and a second lien at the base rate plus the mezzanine spread —
  which gives the identical answer. The web mirrors it in
  [web/src/lib/deal/capital.ts](../web/src/lib/deal/capital.ts).
- **For the simulation.** `simulation_tranches` turns specs into the arrays'
  form (section 9); `shift_references` moves every floating reference by a
  scenario's rate change and leaves fixed tranches alone.

**Regional differences.** The reference rate is a label plus a curve
(SOFR, SONIA, €STR, EURIBOR, TONA, SARON, BBSY, MIBOR or custom): two tranches
with the same path and margin price identically whichever benchmark they name.
**Today's level by default (PLAN.md 4.2).** Market data never enters a run by
itself: it fills the input. A new floating facility starts on its deal
currency's usual benchmark (SOFR for dollars, SONIA for sterling, EURIBOR for
euros, TONA, SARON, BBSY, MIBOR), and choosing a benchmark sets a flat
`reference_level` at that benchmark's latest published level, which the deal
then stores like any typed rate, so reopening a deal never changes its
answer. The level comes from [economy/views.py](../economy/views.py)
`reference_rates`: the first *current* candidate in
[economy/catalogue.py](../economy/catalogue.py) `REFERENCE_SOURCES` -- SOFR
and SONIA as published (FRED: New York Fed, Bank of England), €STR and
3-month EURIBOR from the ECB; TONA, SARON and MIBOR have no free source, so
their central bank's policy rate (BIS) stands in, and BBSY Australia's
3-month bank bill rate (OECD), each labelled as such. A daily rate is current
for 45 days, a monthly one for 100, a policy rate for 120; past that the
benchmark is left out and the user types the rate. A facility may name its own currency as a
label; it is not converted.

## 6. Accounting standards and leases

[core/accounting.py](../core/accounting.py). Tests:
[tests/test_accounting.py](../tests/test_accounting.py) (the IFRS 16 views by
hand, in the deal model, the grid, sources and uses and the simulation).

A deal may say which standard its EBITDA follows (IFRS or US GAAP) and carry
a yearly lease cost and a lease liability at close. `lease_terms` turns them
into three numbers:

| | Pre-IFRS 16 view | Post-IFRS 16 view |
|---|---|---|
| Operating EBITDA (what the model grows) | after lease costs | after lease costs |
| Valuation add-back (added wherever EBITDA meets a multiple) | 0 | the lease cost |
| Lease liability counted with net debt | 0 | the liability |

The operating EBITDA is always after lease costs: an IFRS figure has the lease
cost subtracted (IFRS 16 moves rent below EBITDA), a US GAAP figure already
has it. Rent is cash under either standard, so the cash flows never depend on
the view. `default_view` is post-IFRS 16 for IFRS and pre for US GAAP;
`valuation_ebitda` = operating EBITDA + add-back. A lease cost at or above an
IFRS EBITDA is refused. The add-back and liability are held flat over the hold
(the screen says renewals are assumed). `standard_or_none` accepts only a
known standard.

On the post view the add-back enters the entry EV, the exit EV and every grid
cell, and the liability enters net debt at entry and exit and is subtracted
from the price paid in sources and uses. Debt sized as a percentage of EV or
as a multiple is sized on the valuation EBITDA.

**Regional differences.** This is the main accounting difference between
IFRS reporters (most of the world) and US GAAP reporters (the United States):
the same business shows a higher EBITDA and more net debt under IFRS 16. The
filing reader picks the standard per company (section 13). `lease_figures`
reads a filing's lease cost (with lease interest added for IFRS when tagged)
and liability (total, or current plus non-current).

## 7. Tax rules

[lbo_engine/tax.py](../lbo_engine/tax.py) (mechanics),
[core/tax.py](../core/tax.py) (country presets). Tests:
[tests/test_tax_rules.py](../tests/test_tax_rules.py) (each rule
hand-checked, the presets, the simulation, the heatmap).

With no rule on (`TaxRules.active` false), tax is the flat rate on positive
profit before tax (section 3.3). With rules on, `tax_schedule` decides each
year's tax, in this order:

1. **Interest limit.** Net interest (expense less interest income) is claimed
   together with what earlier years could not deduct. The cap is either a
   share of the year's EBITDA, never below an allowance (`ebitda_share`), or a
   fixed amount (`fixed`; a fixed cap of 0 disallows all interest). What the
   cap refuses is carried forward without expiry. Net interest *income* is
   taxed, not capped.
2. **Losses.** Profit for tax = EBIT − deductible interest. A loss is carried
   forward; a profit absorbs losses carried in, up to an allowance in full plus
   a share of the profit above it.
3. **Tax** = rate × what is left, raised to the minimum tax (a rate on book
   profit before tax) when that is higher.

The function uses only arithmetic, `np.minimum` and `np.maximum`
(`_plain` returns a plain number for one deal), so the deal model and the
simulation run the same rules on one deal or on arrays of paths.
`rules_from_deal` converts the deal's percentages and allowances and returns
nothing when no rule is on.

**Regional differences.** `apply_preset` fills the rate and rules from a
country preset (`PRESETS`): United States, United Kingdom, Germany, France,
Netherlands, Ireland, India, Japan, Australia, Canada, Singapore. Each carries
its statute, a note of what it simplifies and an `as_of` date (rates are
headline rates as of 2026-01). A preset's allowances are amounts in its own
currency, so they are applied only to a deal in that currency; another
currency gets the rate and shares with allowances set to zero, and the screen
says so. The web applies presets the same way
([web/src/components/deal/steps/TaxRules.tsx](../web/src/components/deal/steps/TaxRules.tsx)).
The screen tells the user to check with a tax adviser.

## 8. Risk warnings

[core/risk_warnings.py](../core/risk_warnings.py),
[core/risk_sources.py](../core/risk_sources.py). Tests:
[tests/test_risk_warnings.py](../tests/test_risk_warnings.py) (hand-checked
default deal, sources, the cash reconciliation).

`risk_warnings` returns figures and source ids, never sentences (`_warning`
packs them; the words live in the translation catalogue). Four warnings:

- **Leverage above guidance** (`_leverage_warning`): `leverage_at_close` =
  (debt at close + lease liability on the post view) / valuation EBITDA, warned
  above `LEVERAGE_GUIDANCE_X` (the ECB and US supervisory guidance).
- **Implied rating** (`_rating_warning`): year-one EBIT / interest is mapped
  to a rating through Damodaran's coverage table for large non-financial firms
  (`rating_for_coverage`, January 2026). If the rating is speculative grade
  (`is_speculative`: below BBB−), the warning adds S&P's cumulative default rate
  at the deal's hold from its 2024 default study, table 26
  (`cumulative_default_pct`; `study_row` maps CCC, CC, C and D to the study's
  pooled CCC/C row).
- **Interest above EBITDA** (`_coverage_warning`): any year whose EBITDA does
  not cover interest, with the worst.
- **Unfunded repayment** (`_unfunded_warning`): `unfunded_repayments`
  computes, each year, mandatory repayments − levered FCF − cash carried above
  the minimum − revolver draws; anything above `ROUNDING` is the cash the
  engine supplied out of nothing (finding 11).

**Regional differences.** The leverage guidance is the euro area's and the
United States'; the rating and default tables are US-based global studies.
Each source is listed with its publisher, date, table and sample in
`core/risk_sources.py` and shown beside the warning.

## 9. Monte Carlo simulation

[simulation/vectorized_simulation.py](../simulation/vectorized_simulation.py),
[simulation/tranches.py](../simulation/tranches.py),
[core/montecarlo.py](../core/montecarlo.py). Tests:
[tests/test_montecarlo_baseline.py](../tests/test_montecarlo_baseline.py)
(pinned output), [tests/test_montecarlo_tranches.py](../tests/test_montecarlo_tranches.py)
(each path at the mean against the deal model).

The simulation reruns a vectorised copy of the deal model on *N* paths (up to
100,000), each with its own draw of five uncertain inputs.

**Parameters.** `build_sim_params` takes the Monte Carlo rail (means and
spreads of growth, exit multiple, rate and gross margin; paths; hold; hurdle)
and the deal (opex, D&A, tax, capex, NWC, debt, structure, tax rules, leases)
and the Settings (fees, amortisation, passes, IRR clipping, correlations).

**Draws** (`_draw_correlated_inputs`). With a seeded `RandomState`, five
standard normals per path are correlated through the Cholesky factor of the
correlation matrix (a tiny ridge is added if it is not positive definite),
then:

| Input | Draw | Clipped to |
|---|---|---|
| Revenue growth | mean + z × spread | −50% to 50% |
| Exit multiple | mean + z × spread | 1x to 30x |
| Interest rate | mean + z × spread | 1% to 30% |
| Gross margin | mean + z × spread | 5% to 90% |
| EBITDA shock | z × 5%, year 2 only | −30% to 20% |

**Each path** (`_run_vectorized_core`): the operating model as in section 3.2
with the path's growth and margin (EBITDA margin clipped to 2%–95%; the shock
multiplies year 2's margin), the income statement and cash flow as in 3.3–3.4,
then the debt schedule, iterated `mc_n_passes` times for the interest
circularity.

- **Two-bucket structure:** senior at the path's rate, amortising; mezzanine
  at rate + spread, a bullet at the hold; mandatory repayments, then the sweep
  to senior and then mezzanine. Cash is reset to the minimum every year
  ([finding 12](#known-limitations)).
- **Tranche list** (`_run_tranche_core`, `_operating_paths`): the deal
  model's schedule on arrays (`run_tranche_schedule`, with `_scheduled` for
  mandatory repayments), keeping unswept cash as the deal model does. **The
  rate draw moves floating facilities only**: `rate_in_year` =
  `max(reference + (draw − rate mean), floor) + margin`; fixed tranches keep
  their rate (`tranche_rates` lays out a path's rates by year).
- **Tax rules** apply on arrays through the same `tax_schedule` (`_ruled_taxes`).

**Returns per path:** exit equity = max((exit EBITDA + add-back) × exit
multiple − net debt at exit, 0); MOIC = exit equity / entry equity; IRR =
`MOIC^(1/n) − 1` (`_vectorized_irr`), −100% when exit equity is zero, clipped
to −100%…500% when `mc_clip_irr` is on. `SimulationResult.irr` and
`SimulationResult.moic` hold the arrays; `SimulationResult.wipeout_rate` is the
share of paths with zero exit equity.
`run_vectorized_simulation_full` runs one simulation; the older
`run_vectorized_simulation` infers opex and gross margin from an EBITDA margin
(opex = 69% of it) and returns a table, and is kept for the engine's tests.

**Summaries.** `risk_summary` (via `calculate_risk_metrics` in
[analytics/risk_metrics.py](../analytics/risk_metrics.py)): mean and median
IRR, 5th and 95th percentiles, the share of paths above the hurdle, the
wipeout rate. `analysis_sample` takes up to 50,000 paths (fixed seed) for the
charts: `empirical_correlations` (Pearson, drivers and IRR/MOIC),
`driver_sensitivity` (Spearman's ρ of each driver with IRR, sorted by size) and
`driver_fits` (least-squares line of IRR on each driver, with r).

**Scenarios.** `apply_scenario` moves the means by the Settings' multipliers
and adjustments (bull, base, recession, stagflation): growth × or + an
adjustment with a floor, exit multiple ×, rate ×, gross margin ×; with tranches
the rate change shifts every floating reference. `run_scenarios` runs all four
with one seed and `scenario_stats` summarises each. `get_scenario_params` is
the engine's own copy with fixed multipliers, used by its tests.

**Heatmap.** `growth_exit_heatmap`: 8 growth values from mean − 3σ (not below
−10%) to mean + 3σ by 7 exit multiples from mean − 2σ (not below 2x) to
mean + 2σ; **each cell is a full deal-model run** at the simulation's mean
rate and margin (finding 3), on the deal's own structure with references
shifted to the rail's rate.

**Regional differences.** None in the mechanics. The means, spreads,
correlations and presets are Settings; their factory values are
illustrative, and a deal with a country gets sourced ones (below).

### Sourced ranges, correlations and scenarios

PLAN.md 4.4, [benchmarks/risk.py](../benchmarks/risk.py) (`risk_assumptions`),
on the history read by [benchmarks/history.py](../benchmarks/history.py).
These are **Settings values**, applied when the user asks ("Use sourced
figures" on the Monte Carlo rail) and stored with the deal like typed ones,
so nothing in the engine moves and no engine version changes. Worked figures
below are for a UK machinery deal on the recorded data
([tests/test_risk_ranges.py](../tests/test_risk_ranges.py)).

**The history.** Damodaran archives each January edition of his industry
averages; `archive_url` names the file (`marginEurope16.xls` is the January
2017 edition, so it describes 2016) and `archive_years` the years kept, 2011
to the year before last; the current edition (already stored for §14's
starting figures) describes last year. `read_archive` reads an edition with
`parse_industries` in its lenient form (older editions print fewer columns:
gross margin only since 2017, EBITDA / sales since 2013); `merge` adds its
rows to the group's history, keeping the smaller company count of the two
data sets in a year; `industry_years` joins the archive and the current
edition and `industries_in` lists what a group has. `to_read`, `read_key`
and `fill` read the archive a batch at a time (a file read once is final,
except a missing one of the newest archived year), `table_name` names the
stored tables. The economic history (`read_macro`): real GDP growth and
inflation by year from the IMF's World Economic Outlook (`_imf`), and the
BIS's monthly policy rates averaged over each full year (`_bis`), from
`WINDOW_START` (2000; earlier rates include the 1990s hyperinflations) to
last year, read back by `macro_series`; a euro member's policy rate is the
euro area's (`policy_area`).

**Means** are the deal's own sourced starting figures (§14): growth, the exit
multiple (equal to the entry), the base rate and the gross margin.

**Spreads** (standard deviations, the sample formula):

- **Exit multiple and gross margin**: the spread of the industry's own
  EV / EBITDA and gross margin across its years (`industry_series`: years
  with at least `MIN_FIRMS` companies and a usable figure), in the closest
  group of the country's `chain` with at least `MIN_YEARS` such years
  (`_pick`, which reports a group passed over as missing or short;
  `_multiple_usable`, `_margin_usable`; `_industry_figure` records the group,
  its latest company count via `_latest_firms`, and the years). UK machinery reads developed Europe (210 companies): EV / EBITDA over 14
  years, 2011-2025, mean 12.81x, spread **2.55x**; gross margin over 9 years,
  2017-2025, spread **1.39 points**.
- **Growth**: the spread of the country's nominal GDP growth year to year
  (`nominal_growth`: (1 + real growth)(1 + inflation) − 1), since no free
  source gives an industry's revenue growth by year; a country outside the
  catalogue takes the median spread of its region's economies, then the
  world's (`_economy_spread`, `members`, `_window`, `_period`).
- **Rate**: the spread of the yearly change in the country's average policy
  rate (`rate_changes`): the UK's nominal growth spread is **4.08 points** over
  2000-2025 and its Bank Rate's yearly change **1.17 points**. Credit spreads have no free source that may be shown
  (PLAN.md 4.2), so the rate moves with policy only.

**Correlations**, per group (`correlations`, `history_group`: the closest
group with history). `panel` builds one row per industry and year with at
least `MIN_FIRMS` companies in both years: the group's median nominal growth
that year (`_median_by_year`), the change in EV / EBITDA, the median change
in the policy rate, the change in gross margin and the relative change in
EBITDA margin (`_change`). `raw_correlations` measures each pair's Spearman
rank correlation over the rows that have both (growth and the rate, which are
economy-wide, over years), needing `MIN_PAIR_OBSERVATIONS` rows (else 0), and
`_rank_correlation` turns ρ into the normal correlation that has it,
2 sin(πρ / 6). `valid_matrix` shrinks the matrix toward the identity,
(1 − s)M + sI in steps of `SHRINK_STEP`, until its smallest eigenvalue is at
least `EIGEN_FLOOR`, rounds it to `CORR_DECIMALS` and checks it the way the
simulation does (`is_valid_corr`). The UK reads developed Europe's panel, 2012-2025; every group's matrix
is valid without shrinking on the recorded data, and the test checks all
eight.

**Scenarios** (`_scenario`), from the country's own years since 2000 (or its
region's median by year, `_economy_years`): a **recession** is the weakest
fifth of its years by real GDP growth, **stagflation** the fifth with the
highest inflation, **bull** the strongest fifth (`fifth`, `SHARE`; `periods`
joins consecutive years). Each preset moves growth by the gap between
nominal growth in its years and the average (bull as a multiplier of the
growth mean; recession and stagflation as points, with the worst year's
nominal growth as the floor), the rate by the policy rate's average change
in those years, written as a multiplier of the rate mean, and the exit
multiple and gross margin by the industry's figure in those years over its
average (`_mean_in`). A preset whose years the industry history doesn't
reach keeps its value and says so. The UK's recession years are 2008-2009, 2011, 2020 and 2023: nominal
growth 3.30 points below its average (floor -9.19%, 2020), Bank Rate down
0.45 points a year on average (×0.891 of a 4.13% mean), and European
machinery's multiple at 13.76x in the archive's three of those years against
12.81x on average (×1.074: in a downturn EBITDA tends to fall faster than
values).

Size is not split (no free source does), and an industry average is an
aggregate over listed companies, so a single company's swings are wider;
the screen says both.

| Constant | Value | Meaning |
|---|---|---|
| `benchmarks.risk.MIN_YEARS` | 6 | usable years a spread needs |
| `benchmarks.risk.SHARE` | 0.2 | a scenario's share of the years |
| `benchmarks.risk.MIN_PAIR_OBSERVATIONS` | 30 | industry-years a correlation needs |
| `benchmarks.risk.EIGEN_FLOOR` | 0.05 | smallest eigenvalue a correlation matrix keeps |
| `benchmarks.risk.SHRINK_STEP` | 0.01 | step of the shrink toward no correlation |
| `benchmarks.history.WINDOW_START` | 2000 | first year of economic history |
| `benchmarks.history.FIRST_YEAR` | 2011 | first year of industry history |

## 10. Plan vs actual

[core/plan_actual.py](../core/plan_actual.py),
[core/examples.py](../core/examples.py),
[core/backtesting.py](../core/backtesting.py). Tests:
[tests/test_plan_actual.py](../tests/test_plan_actual.py).

`compare` sets a saved deal's plan against what happened:

- **The plan** is the deal model's answer (`_run`, grid off); its years
  (`_plan_years`) are revenue, EBITDA, net income, levered FCF and closing
  debt. An exit after fewer years than planned is compared with the plan rerun
  at that hold.
- **Variances** (`_variance`) = actual − plan for each figure entered.
  Margins (`_margins`) = EBITDA / revenue where revenue is positive.
- **Where the actual IRR landed.** `plan_irr_paths` simulates the plan around
  its own assumptions (its growth, exit multiple, rate and gross margin as the
  means, Settings' spreads); the percentile is the share of paths below the
  actual IRR.
- **Actual returns** (`_returns` for the plan side): exit equity = exit EV −
  net debt; MOIC = exit equity / entry equity unless given; IRR from the MOIC
  over the exit year (`_irr_from_moic`) unless given.
- **Attribution** (`attribution`), exact to the cent:
  EBITDA = (actual − plan exit EBITDA) × plan exit multiple; multiple = the
  rest of the EV gap; net debt = plan − actual net debt at exit. With leases
  both sides add the plan's lease cost to EBITDA before the multiple.

Actuals in another unit are converted; another currency is refused.

`examples` serves the optional example library (four inception-era US deals,
`_plan` and `_actuals` build their inputs; hidden when the library is switched
off, [§14](#the-reference-library)). They are never an input to any calculation.

The old backtest (`backtest_summary`, kept as the golden parity record):
`predicted_ebitda` grows revenue at the plan growth at a constant margin
(rounded to 0.1M), `run_prediction_sim` simulates with fixed spreads (growth
4%, exit 1.5x, rate 1.5%, margin 3%), `prediction_lbo_params` runs the deal
model for the predicted exit (finding 5), and the attribution uses the same
three-way split.

## 11. Three-statement forecast

[core/forecasting.py](../core/forecasting.py). Tests:
[tests/test_api.py](../tests/test_api.py) (forecast endpoints and balance
checks), [tests/test_locale.py](../tests/test_locale.py) (fiscal years).

A company forecast in its own currency and unit, from up to five historical
years (entered or read from a filing).

- **History.** `ltm_from_history` takes the latest year as the opening
  balance sheet; `historical_metrics` derives growth, gross margin, R&D and
  SG&A %, EBITDA and adjusted EBITDA margins per year. `opening_bs_gap` is
  assets − liabilities − equity of the opening year (shown when the entered
  history does not balance). `default_history` and `default_history_value`
  give the placeholder history (the latest default scaled by 0.85 a year
  back).
- **Seeded assumptions** (`seed_assumptions`): margins, cost and capex shares
  from the latest year; the tax rate from tax / pre-tax income, held within
  5%–40%; working-capital days from the balance sheet (45/30/60 if absent);
  growth 6%; cash and debt rates 2.2% and 2.8%; minimum cash 20% of cash.
  `assumptions_from_grid` reads the user's year-by-year grid.
- **Each year** (`run_3_statement_model`): revenue grows; COGS, R&D and SG&A
  are shares of revenue; interest income on opening cash, interest expense on
  opening debt and revolver; tax on positive pre-tax income; D&A, stock
  compensation and capex as shares of revenue; PP&E rolls forward (+ capex −
  D&A); receivables, inventory and payables from days on revenue and COGS;
  other balances as shares of revenue or held flat; retained earnings roll
  forward (+ net income − dividends − buybacks); stock compensation adds to
  paid-in capital. Cash flow from operations, investing and financing gives
  cash; **a revolver plugs** any fall below minimum cash and is repaid first
  from any excess. The balance check (assets − liabilities − equity) is shown
  per year.
- **Simulation overlay** (`run_forecast_simulation`, seed 42): growth and
  EBITDA margin are drawn once per path, correlated 0.40, around the means of
  the grid (spreads = the spread of the yearly assumptions, at least 2% and
  1%; margin clipped to 1%–80%). `simulation_summary` gives percentile bands,
  final-year statistics and the probability of reaching 80%–120% of the
  deterministic final EBITDA (above plan labelled Bull, below Bear; finding 4).
  `revenue_cagr` is the forecast's revenue CAGR.

**Regional differences.** The company's accounting standard is a label here
(the forecast's lines do not change); fiscal years follow the company's year
end.

## 12. Model version

[core/model_version.py](../core/model_version.py),
[MODEL_CHANGELOG.md](../MODEL_CHANGELOG.md). Tests:
[tests/test_model_version.py](../tests/test_model_version.py).

Every result carries `stamp(cfg)`: the engine version (`ENGINE_VERSION`), the
deployed `commit`, `settings_fingerprint` (a SHA-256 digest of the Settings
resolved against today's defaults, `FINGERPRINT_DIGITS` hex digits, via
`_digest`), and the data editions (`data_sets` from `_published_data`: each
source table's publication and the tax presets' latest `as_of`;
`data_vintage` is the newest, `data_fingerprint` their digest).

A saved deal stores `saved_stamp`: the stamp plus its IRR and MOIC from a run
without the grid (`_finite` drops a non-finite value), and
`content_fingerprint` of its inputs and Settings. On reopening, `compare`
reruns it: if the content is the same and the IRR or MOIC differ beyond
`RESULT_REL_TOL` (`_same`), it reports "changed" and which part of the stamp
moved; different content is "unknown".

## 13. Estimates: the ML panels and filings

These are estimates and are labelled as such on screen (PLAN.md principle 5).

- **Risk score** ([ml/anomaly_detector.py](../ml/anomaly_detector.py),
  `check_deal`). An isolation forest and nearest-neighbour search over 30
  hand-entered, mostly US historical LBOs plus 500 synthetic deals jittered
  around the successes (`_build_training_data`, `train_detector`), on the five
  inputs of `risk_model_inputs`. Score = 1 + min(max(leverage − 4, 0) × 0.5, 3)
  + min(max(entry multiple − 8, 0) × 0.3, 2) + min(max(2 − coverage, 0) × 2, 2)
  + 2 × anomaly severity, clipped to 1–10, where coverage = 100 / (leverage ×
  rate) and severity = how far the forest's score falls below its threshold,
  scaled to 0–1. The figures are unsourced; the screen shows the sample size
  (`historical_sample`; `detector_is_trained` says whether the files exist).
  PLAN.md 5.2 replaces it.
- **Live sliders** ([ml/surrogate/predict.py](../ml/surrogate/predict.py),
  [core/surrogate.py](../core/surrogate.py)). A small neural network
  (`SurrogatePredictor.predict`, loaded once by `SurrogatePredictor.get_instance`)
  maps eleven inputs (`surrogate_features`) to IRR percentiles, mean, spread,
  the chance of beating 20% and the wipeout rate. It was trained on 100,000
  simulations of one fixed deal (`generate` in
  [ml/surrogate/generate_data.py](../ml/surrogate/generate_data.py): a Latin
  hypercube over the eleven inputs, everything else fixed). `training_terms`
  and `training_term_differences` list where the user's deal differs from that
  training deal (`tax_rules_on` counts the tax rules), and `tail_unreliable`
  flags a predicted wipeout of `TAIL_UNRELIABLE_WIPEOUT` or more.
- **Macro regime** ([ml/macro_regime.py](../ml/macro_regime.py),
  `get_current_regime`). A four-state Gaussian hidden Markov model over US
  FRED series (`fetch_fred_data`, `train_regime_model`; `model_is_trained`).
  It reports the latest month's regime, its probability and each regime's
  average departure in GDP growth, inflation, credit spread and rates. US data
  only; PLAN.md 4.2 and 5.7 replace it.
- **SEC EDGAR figures** ([ml/edgar_extractor.py](../ml/edgar_extractor.py),
  `fetch_financials`; `_get_cik_from_ticker` finds the filer). `_source`
  chooses US GAAP from a 10-K in US dollars when the filer has one, else IFRS
  from a 20-F/40-F in the currency its statements mostly use. For each line,
  `_get_annual_values` takes full-year values from annual forms, the latest
  filing per year, in millions; `_merged` fills gaps from fallback concepts;
  `_single_tag` takes one concept for every year (lease cost) and `_leases`
  reads the lease figures. The fiscal year end comes from the latest annual
  report (`_latest_10k_end`, `_fiscal_year_end`: a 52/53-week year ending in a
  month's first week counts as the month before; `_latest_fiscal_year`).
  `_reconcile_balance_sheet` derives the plugs (other long-term assets, other
  liabilities, common stock) so the opening balance sheet balances, and warns
  when a derived line is negative. Deal inputs: EBITDA = operating profit +
  D&A as the standard reports it. `financials_to_session_state` maps the
  figures onto the forecast's historical fields.

**Regional differences.** EDGAR covers companies filing with the SEC
(US GAAP and IFRS foreign filers); other countries' registries arrive with
PLAN.md 4.1. The line items per standard are `US_GAAP_ITEMS`, `IFRS_ITEMS`
and `LEASE_ITEMS` in [core/accounting.py](../core/accounting.py).

## 14. Figures made for display

Not model outputs, but numbers on screen computed from them
([api/serialize.py](../api/serialize.py)): `histogram` (density over equal-width
bins of the finite values), `percentile_curve` (the 0th–100th percentiles in
steps of 1) and `box_stats` (5th, 25th, 50th, 75th, 95th percentiles).
Formatting — separators, digit grouping (including lakh), currency symbols and
fiscal labels — never changes a value
([web/src/lib/format.ts](../web/src/lib/format.ts)).

### Starting figures

A new deal starts from published data on many companies and economies
(PLAN.md 4.3, [benchmarks/starting.py](../benchmarks/starting.py),
`starting_assumptions`). These are **inputs**, not model logic: once applied
they are stored in the deal like any typed figure, so a saved deal's result
never moves when the data is refreshed, and no engine version changes. Worked
figures below are for a German machinery deal on the January 2026 data.

- **Where an industry figure comes from.** Damodaran's industry averages
  ([benchmarks/catalogue.py](../benchmarks/catalogue.py), read by
  [benchmarks/damodaran.py](../benchmarks/damodaran.py)) are aggregates over
  listed companies: the sum of the group's figure over the sum of its revenue.
  `_pick` walks the country's `chain` -- its own file (US, Japan, China,
  India), then its region (developed Europe; Australia, NZ and Canada;
  emerging markets), then global -- and takes the first group with at least
  `MIN_FIRMS` companies whose row passes the data set's check (`_num`,
  `_margins_usable`, `_multiples_usable`, `_capex_usable`, `_wc_usable`,
  `_debt_usable`); each group passed over is reported with why (thin,
  unusable, missing). `_damodaran` records the group, its companies and the
  workbook's date with each figure; `_pct` writes a fraction as the per cent
  the deal reads. Germany reads developed Europe: 210 machinery companies.
- **Margins** (one group for all three, so they stay consistent): gross
  margin as published (44.68%); opex = gross margin - operating margin
  (44.68 - 11.71 = 32.97%: the engine's opex includes D&A); D&A = EBITDA margin
  - operating margin (12.97 - 11.71 = 1.26%). The engine's EBITDA margin,
  gross margin - opex + D&A, is then the industry's EBITDA / sales.
- **Multiples.** Entry EV / EBITDA is the industry's, over companies with
  positive EBITDA (14.98x); the exit starts equal (no expansion assumed).
- **Capex** = D&A x the industry's capex / D&A (1.26% x 0.79 = 0.99%).
  Damodaran's "net cap ex / sales" adds acquisitions and R&D, so it is not used.
- **Working capital.** Receivable days = receivables / sales x 365; inventory
  and payable days are on cost of sales, as the engine reads them: inventory /
  sales x 365 / (1 - gross margin). The flat change in working capital is
  the yearly change of a constant share w of revenue: w x g / (1 + g), with w
  the industry's non-cash working capital / sales (17.37%) and g the starting
  growth (0.59%).
- **Growth** is the country's nominal GDP growth this year, from the IMF's
  World Economic Outlook projection (`_nominal_growth`): (1 + real growth)(1 +
  inflation) - 1 = 1.008 x 1.027 - 1 = 3.52% for Germany. A country outside
  PLAN.md 4.2's catalogue takes the median over its region's economies, then
  over all of them (`_growth`). Industry growth averages are averages of
  single companies' growth and run far above an industry's (decided with the
  user, 2026-10-07).
- **Tax** is the country's marginal corporate rate (the Tax Foundation's
  survey, published in Damodaran's country tax workbook; Germany 29.93%),
  else the median over its region's countries, then all (`_tax`). The
  workbook repeats the end of its list with older rates, so a country's
  first row wins (`parse_country_tax`).
- **Leverage** is the industry's listed-company debt / EBITDA over the entry
  multiple (1.87 / 14.98 = 12.49% of EV), all senior: no free source
  publishes buyout leverage (decided with the user, 2026-10-07). Damodaran
  counts lease debt in it.
- **The interest rate** is the currency's benchmark at today's level (PLAN.md
  4.2; else the country's policy rate, `_benchmark_level`) plus Damodaran's
  default spread for the rating the industry's interest coverage implies
  (`rating_for_coverage`, §8): EURIBOR 2.64% + 0.40% (coverage 10.6x, AAA) =
  3.04%.

Every figure goes into the answer by `_plain`; `industries` lists what the
stored averages cover. The scheduled refresh reads the workbooks
(`every_table`, `fetch_table`, `fetch_country_tax`, `parse_industries`,
`industry_id`) once a week. The defaults registry
([benchmarks/registry.py](../benchmarks/registry.py)) says where every other
default comes from; `tests/test_defaults_registry.py` keeps it complete.

| Constant | Value | Meaning |
|---|---|---|
| `benchmarks.catalogue.MIN_FIRMS` | 20 | a group with fewer companies hands over to a wider one |
| `benchmarks.starting.MAX_MULTIPLE` | 100.0 | a higher EV / EBITDA is not a starting point |
| `benchmarks.starting.MAX_CAPEX_TO_DA` | 10.0 | a higher capex / D&A is not a starting point |

### The reference library

The optional reference library (PLAN.md 4.5, Library mode) shows published
figures beside a deal; **nothing in the deal model, the simulation or the risk
warnings reads it**, so switching it off (`library/switch.py`) moves no result.

- **Base rates** ([library/base_rates.py](../library/base_rates.py)) are
  transcribed tables, never estimates. Default rates come from S&P Global
  Ratings' 2024 global default study: the annual rate by rating category
  (Table 3) and by grade (Table 1), 1981-2024; the speculative-grade rate by
  region (Table 5: the U.S. and tax havens, Europe, emerging and frontier
  markets, other developed); and the average cumulative rate after 1 to 15
  years by rating, globally (Table 24) and for the U.S., Europe and emerging
  markets (Table 25). Recovery comes from Global Credit Data's 2020 report on
  bank loans to large corporates (2000-2016 defaults, 11,527 borrowers): loss
  given default by seniority and collateral (Table 2), year of default (Table 3)
  and region (Table 4); recovery is 100 less it. `base_rates` lays them out by
  year (`_years`) with their sources (`_source`). `sp_region` reads a country
  into S&P's region from the study's own lists (anything unlisted is an
  emerging or frontier market). Each source records the day its edition was
  last confirmed as the newest one free to read; `recheck_due` is a year later
  (`_add_months`) and `stale` turns true from then, shown on screen and failed
  by the daily production check.
- **Coverage** ([library/coverage.py](../library/coverage.py)) counts the
  library's deals by region (S&P's), size (`size`: enterprise value at entry in
  US dollars, under 100m, 100m-1bn, 1-10bn, over 10bn), sector, era (`era`:
  entry year before 2000, 2000-2007, 2008-2014, 2015-2019, 2020 on) and outcome,
  listing empty buckets (`_band`, `_counts`, `collection`). `example_tags` tags
  each example (its year from its name, `_deal_year`); `coverage` adds the
  base-rate tables' regions, bands, years and observations
  (`base_rates.coverage`); the approved reference transactions are counted
  by their own buckets (`references.tags`), sectors as GICS's eleven.
- **Reference transactions** ([library/references.py](../library/references.py),
  PLAN.md 4.5b) are real buyouts with every figure read from a filing
  (`reference_deals.json`, read once by `_file`; `repository_deals` hands out
  copies). A figure is one number at one place in a filing, or the sum of
  pieces each with its own place (EBITDA as operating income plus D&A, debt
  as its facilities, fees less those that were financing costs). Money is in
  millions of the deal's currency; `value` reads a figure (`_figure`), `closed_year` the
  closing year. `derived` puts the figures together: the entry multiple
  (transaction value / EBITDA), leverage (debt / EBITDA), debt as a share of
  the value, and the three fee Settings as Settings measure them --
  transaction fees per cent of the transaction value, financing fees per cent
  of the debt (`_pct`), and the senior amortisation per cent a year as the
  credit agreement states it. Worked: HCA (2006) paid 33,000 for 2006 EBITDA
  of 4,327 (the management projection in its merger proxy), so 7.63x; its
  financing fees, 568 on 19,964 of new debt, are 2.85%.
  The **inclusion rules** (`problems`) refuse a proposal that cannot be
  checked or does not describe a buyout: a missing required figure
  (transaction value, EBITDA, debt; and a US dollar value for a deal in
  another currency, so `usd_value` can size it), a cited place whose source
  is missing or is not on a filing host (`is_filing_url`: the regulator's,
  register's or exchange's own site, over https), a source nothing cites,
  pieces that do not add up to within 0.05 (`_evidence` walks every cited
  place), a multiple outside 2-40x, debt above the value paid, a fee outside
  0-10% or amortisation outside 0-100% (`_in_range`), a closing in the
  future, an outcome before the closing or an event that does not fit its
  outcome. The **balance rules** (`balance`) are advisory: once the library
  holds six deals, a proposal that would leave any region, size, sector, era
  or outcome holding more than half of it is flagged, and the empty buckets a
  proposal fills are named. `summary` adds `derived` and `tags` to a deal for
  the screens; `content_hash` fingerprints a proposal so the same one is
  never stored twice. A transaction joins the library on the second approval
  by an administrator other than its proposer (library/review.py,
  db/references.py); nothing in the model reads it.
- **Sourced fees and amortisation**
  ([library/fees.py](../library/fees.py)): `sourced` offers `tx_fee_pct`,
  `fin_fee_pct` and `def_senior_amort` as the **median** across the approved
  transactions that give each figure, with how many deals, their lowest and
  highest and the closing years; a figure fewer than three deals give is not
  offered. They are Settings the user applies (Settings -> Fees, "Use sourced
  figures"); until then the factory values below stay in force and labelled
  illustrative, so no saved deal moves. Worked, with the ten repository
  transactions approved: transaction fees are 0.63, 0.64, 0.89, 1.58, 3.29
  and 3.53% of value, median 1.24%; financing fees 1.94-6.00% of debt over
  seven deals, median 3.07%; three term loans amortise 1% a year.

## 15. Regional differences at a glance

| Area | What varies by region | Where |
|---|---|---|
| Currency and unit | Any ISO 4217 currency; thousands, millions or billions; no exchange rates yet | §2 |
| Floating rates | Benchmark label and its curve; floor before margin | §5 |
| Accounting | IFRS 16 leases in EBITDA and net debt (IFRS) or not (US GAAP) | §6 |
| Tax | Rate, interest limit, loss rules, minimum tax per country preset | §7 |
| Risk thresholds | ECB/US leverage guidance; US-based rating and default studies | §8 |
| Base rates | Default rates for the U.S., Europe, emerging and other developed markets; recovery for six regions | §14 |
| Filings | SEC EDGAR only (US GAAP and IFRS filers) | §13 |
| Starting figures | Industry averages by region, growth and tax by country, rate by currency | §14 |
| Monte Carlo ranges | Spreads, correlations and presets from each region's and country's history | §9 |
| Macro | US FRED series only | §13 |
| Fiscal years | Labels follow the year-end month | §1 |

## Known limitations

Open model findings awaiting the user's approval (CLAUDE.md "Model findings"):

- **10. Rounding.** The engine rounds money to two decimals of a million
  (10,000 of the currency), and the old backtest's predicted EBITDA to 0.1M.
  A deal with EBITDA under about 10M loses precision, and the rounding
  builds up: a net income of 30.075 a year left cash 0.015 off a hand answer
  after three years ([tests/reference/](../tests/reference/README.md)).
- **11. Cash shortfalls funded out of nothing.** Closing cash is set to the
  minimum whatever happened (§3.5), unless a revolver can draw. The unfunded
  repayment warning (§8) shows the amount. An explicit tranche list also keeps
  its maturities at every hold of the grid, while the percentage structure
  repays the mezzanine at each hold. Reference case 08 measures it by hand.
- **12. Two-bucket simulation discards surplus cash** once the debt is repaid
  (§9); the tranche path keeps it.
- **13. The simulation ignores minimum cash** (`minimum_cash_pct` is 0), so a
  deal with minimum cash simulates a slightly smaller equity cheque.

Other simplifications, by design and shown on screen where relevant: whole
years and exit at year end; lease figures flat over the hold; a loss earns no
credit unless loss carry-forward is on; interest income at a fixed 0.5% on
minimum cash; the Live sliders' surrogate knows one training deal.

## 17. Constants

The values quoted in this document, checked against the code by
`tests/test_methodology.py`.

| Constant | Value | Meaning |
|---|---|---|
| `core.deal.MAX_INTEREST_PASSES` | 50 | most passes of the interest circularity |
| `core.deal.INTEREST_TOLERANCE` | 0.001 | stop when interest moves less than this (millions) |
| `core.debt.SENSITIVITY_HEADROOM` | 10 | extra years of rate path for the grid |
| `core.risk_sources.LEVERAGE_GUIDANCE_X` | 6.0 | leverage warning threshold (debt / EBITDA) |
| `core.risk_warnings.ROUNDING` | 0.005 | smallest unfunded repayment reported (millions) |
| `core.surrogate.TAIL_UNRELIABLE_WIPEOUT` | 0.02 | predicted wipeout above which the surrogate's tail is flagged |
| `core.model_version.FINGERPRINT_DIGITS` | 12 | hex digits of each fingerprint |
| `core.model_version.RESULT_REL_TOL` | 1e-9 | relative tolerance for "results changed" |
| `core.model_version.RESULT_ABS_TOL` | 1e-12 | absolute tolerance for "results changed" |
| `api.limits.MAX_SIMULATION_PATHS` | 100000 | most Monte Carlo or backtest paths |
| `api.limits.MAX_FORECAST_PATHS` | 200000 | most forecast simulation paths |
| `ml.edgar_extractor._FIRST_WEEK` | 7 | a year ending on or before this day counts as the month before |

## 18. Settings defaults

Every default below is illustrative, not market data, and every one can be
overridden in Settings. Percentages are numbers (`2.3` = 2.3%). The deal
defaults are replaced by sourced starting figures when a deal starts (§14),
the Monte Carlo ones by sourced ranges when applied (§9), and the fees and
senior amortisation by the reference transactions' medians when applied
(§14, the reference library).

| Setting | Default | Used by |
|---|---|---|
| `tx_fee_pct` | 2.3 | transaction fees, % of EV |
| `fin_fee_pct` | 2.6 | financing fees, % of drawn debt |
| `other_uses` | 0.0 | other uses at close (money) |
| `def_ebitda` | 100.0 | a new deal's EBITDA |
| `def_entry_mult` | 10.0 | a new deal's entry multiple |
| `def_exit_mult` | 11.0 | a new deal's exit multiple |
| `def_hold` | 5 | a new deal's holding period |
| `def_growth` | 5.0 | a new deal's revenue growth |
| `def_gross_margin` | 40.0 | a new deal's gross margin |
| `def_opex` | 18.0 | a new deal's opex, % of revenue |
| `def_tax` | 25.0 | a new deal's tax rate |
| `def_da` | 4.0 | a new deal's D&A, % of revenue |
| `def_debt_pct` | 60.0 | a new deal's debt, % of EV |
| `def_senior_pct` | 70.0 | a new deal's senior share of debt |
| `def_base_rate` | 6.5 | a new deal's base rate |
| `def_mezz_spread` | 4.0 | a new deal's mezzanine spread |
| `def_capex` | 4.0 | a new deal's capex, % of revenue |
| `def_nwc` | 1.0 | a new deal's change in NWC, % of revenue |
| `def_mincash` | 0.0 | a new deal's minimum cash |
| `def_senior_amort` | 5.0 | senior amortisation, % of principal a year |
| `sens_em_min` | 6.0 | grid: lowest exit multiple |
| `sens_em_max` | 12.0 | grid: highest exit multiple |
| `sens_em_steps` | 7 | grid: number of exit multiples |
| `sens_hp_min` | 3 | grid: shortest hold |
| `sens_hp_max` | 7 | grid: longest hold |
| `mc_n` | 50000 | simulation paths |
| `mc_growth_mean` | 5.0 | growth mean |
| `mc_growth_std` | 3.0 | growth spread |
| `mc_exit_mean` | 10.0 | exit multiple mean |
| `mc_exit_std` | 1.5 | exit multiple spread |
| `mc_rate_mean` | 6.5 | rate mean |
| `mc_rate_std` | 1.5 | rate spread |
| `mc_gm_mean` | 40.0 | gross margin mean |
| `mc_gm_std` | 3.0 | gross margin spread |
| `mc_hurdle` | 20.0 | hurdle IRR |
| `mc_n_passes` | 2 | interest passes in the simulation |
| `corr_g_em` | 0.60 | correlation: growth and exit multiple |
| `corr_g_ir` | -0.30 | growth and rate |
| `corr_g_gm` | 0.40 | growth and gross margin |
| `corr_g_sh` | 0.50 | growth and EBITDA shock |
| `corr_em_ir` | -0.50 | exit multiple and rate |
| `corr_em_gm` | 0.30 | exit multiple and gross margin |
| `corr_em_sh` | 0.30 | exit multiple and EBITDA shock |
| `corr_ir_gm` | -0.20 | rate and gross margin |
| `corr_ir_sh` | -0.20 | rate and EBITDA shock |
| `corr_gm_sh` | 0.20 | gross margin and EBITDA shock |
| `bull_growth_mult` | 1.50 | bull: growth × |
| `bull_exit_mult` | 1.15 | bull: exit multiple × |
| `bull_rate_mult` | 0.85 | bull: rate × |
| `bull_margin_mult` | 1.05 | bull: gross margin × |
| `rec_growth_adj` | -6.0 | recession: growth + (points) |
| `rec_growth_floor` | -10.0 | recession: growth floor |
| `rec_exit_mult` | 0.80 | recession: exit multiple × |
| `rec_rate_mult` | 1.20 | recession: rate × |
| `rec_margin_mult` | 0.93 | recession: gross margin × |
| `stag_growth_adj` | -3.0 | stagflation: growth + (points) |
| `stag_growth_floor` | -5.0 | stagflation: growth floor |
| `stag_exit_mult` | 0.85 | stagflation: exit multiple × |
| `stag_rate_mult` | 1.40 | stagflation: rate × |
| `stag_margin_mult` | 0.90 | stagflation: gross margin × |

`mc_clip_irr` (on) clips simulated IRRs to −100%…500%.

## Appendix: functions that affect no number

Named here so the coverage test can tell "not described" from "nothing to
describe". They print to a console, build the Burger King reference case used
by the engine's own self-tests, or move data without computing.

- Console printers: `print_operating_model`, `print_cashflow_model`,
  `print_debt_schedule`, `print_returns_summary`, `print_sensitivity_table`,
  `print_transaction_summary`, `print_lbo_summary`,
  `CapitalStructure.print_summary`.
- Engine self-test fixtures (the four inception-era deals are never an input
  to the app's calculations): `build_bk_capital_structure`,
  `build_bk_conservative`, `build_bk_management`,
  `build_bk_cashflow_assumptions`, `build_bk_debt_model_inputs`,
  `_test_generic`, `_test_burger_king`, `_test_simulation_wrapper`, and
  `run_lbo_from_inputs` (an IRR/MOIC shortcut over `run_lbo`).
- Plumbing: `to_json`, `_nullable` and `model_from_dataclass` (serialising),
  `rescale`'s helper above, `TrancheSpec` validation, `coerce`, `commit`,
  `_plain`.
