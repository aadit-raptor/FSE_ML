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
16. [Model validation](#model-validation) and [known limitations](#known-limitations)
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
- **Implied rating** (`_rating_warning`, reading `credit_view`): year-one EBIT / interest is mapped
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

**Coverage and default risk on every deal** (PLAN.md 4.6): `credit_view`
answers the same reading -- coverage, its rating and band, S&P's cumulative
default rate over the hold, whether it is speculative grade, and the sources
(`CREDIT_SOURCES`) -- for every deal, investment grade too, as the deal
answer's `credit` block; without interest every figure is empty. The default
deal: EBIT 88.85 over interest 46.20 in year one is 1.92x, a B+ (1.75 to
2.00), and S&P's B+ row gives 12.88% over five years. The deal summary shows
it beside the IRR, the MOIC and the simulation's probability of loss and
downside (§9).

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
`run_vectorized_simulation_full` runs one simulation; with `credit` it also
keeps each path's yearly EBITDA, EBIT, interest and debt at the start of the
year (`credit_paths`; the tranche path's `run_tranche_schedule` records the
opening debt with `track_debt`) for §13's distress predictor, and changes no
other figure. The older
`run_vectorized_simulation` infers opex and gross margin from an EBITDA margin
(opex = 69% of it) and returns a table, and is kept for the engine's tests.

**Summaries.** `risk_summary` (via `calculate_risk_metrics` in
[analytics/risk_metrics.py](../analytics/risk_metrics.py)): mean and median
IRR, 5th and 95th percentiles, the share of paths above the hurdle, the
wipeout rate, and the probability of loss (`probability_of_loss`: the share
of paths whose MOIC is below 1, which on one investment and one exit is an
IRR below zero; the deal summary shows it with the 5th percentile as the
downside case, PLAN.md 4.6). `analysis_sample` takes up to 50,000 paths (fixed seed) for the
charts: `empirical_correlations` (Pearson, drivers and IRR/MOIC),
`driver_sensitivity` (Spearman's ρ of each driver with IRR, sorted by size) and
`driver_fits` (least-squares line of IRR on each driver, with r).

**Driver explanations** (PLAN.md 5.6;
[analytics/driver_attribution.py](../analytics/driver_attribution.py), tests
[tests/test_driver_explanations.py](../tests/test_driver_explanations.py);
why this and not a fitted model:
[driver-explanations.md](driver-explanations.md)). A path's IRR is a function
of its five draws and nothing else, so the simulation can be asked what it
would have answered had some of them stayed at their means. `tail_paths`
ranks the paths by IRR and takes the worst and the best 5% (`TAIL_SHARE`;
ties, such as wiped-out paths, go by path order, so a tail is never larger
than its share); a tail of more than 2,500 paths (`MAX_TAIL_PATHS`, a run
above 50,000) is read at 2,500 evenly spaced through its ranks
(`evenly_spaced`), and the answer gives both counts. For those paths
`coalition_irr` reruns the simulation's core 32 times, once for every choice
of which drivers keep their draws and which sit at the mean (`mean_draws`:
the four means, and no EBITDA shock), in slices of half the run's size, and
takes each rerun's mean IRR. `shapley_values` turns the 32 means into one
contribution a driver: the average, over every order in which the drivers
could be switched from mean to draw, of what switching it moves,

    φ_i = Σ over S not containing i of |S|! (5 − |S| − 1)! / 5! × (v(S ∪ i) − v(S))

`explain_tails` returns them with the base (all five at the mean) and the
tail's mean IRR. No model is fitted, so they add up exactly:

    IRR at the mean assumptions + the five contributions = the tail's mean IRR

Default deal, 50,000 paths, seed 42 (IRR, percentage points):

| | Base | Growth | Exit multiple | Rate | Gross margin | EBITDA shock | Tail mean |
|---|---|---|---|---|---|---|---|
| Worst 5% | 18.56 | −9.51 | −10.49 | −0.50 | −0.22 | −0.08 | −2.24 |
| Best 5% | 18.56 | +8.29 | +6.96 | +0.40 | +0.12 | +0.08 | 34.41 |

A driver held at its mean is held there whatever the others do, so a
contribution is the driver's own effect through the model, not what it
shares with correlated drivers: the rate's rank correlation with IRR above is
−0.47, almost all of it borrowed from growth and the exit multiple. A driver
that cannot move the deal (the rate, when no facility floats) contributes
exactly zero. The base is the simulation's own IRR at the means, which is not
the deal model's (the simulation's simplifications, findings 12 and 13). A
run that itself took more than 20 seconds (`EXPLAIN_RUN_LIMIT_S`) is answered
without explanations, so they can never be what makes an answer time out.

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
  `assess`; PLAN.md 5.2). The deal against companies and deals like it. It
  reads three of the deal's figures (`DealShape`, from `risk_model_inputs`):
  leverage (drawn debt / EBITDA), the entry multiple and the EBITDA margin,
  and compares each (`compare_one`, for each `Metric` of `METRICS`) with its
  industry's in Damodaran's averages for the deal's region (§14's tables).
  *The peer group* (`peer_group`) is the closest group in the country's chain
  whose industry row has `MIN_FIRMS` = 20 companies and the figure: the
  country's own file (US, Japan, China, India), else its region; **never the
  global group**, so a region with too few companies in the industry says
  "not enough data" (`compare`: `no_peers`; without a country, `no_country`;
  an empty industry is the whole market). *How far off*: `spread` is how
  much the group's industries differ, 1.4826 × the median absolute deviation
  of the figure over its industries with 20 companies or more (at least
  `MIN_INDUSTRIES` = 10 of them), and z = (deal − industry) / spread, the
  margin in per cent; `risk_z` is z signed so that positive is riskier
  (lower margin, higher leverage and price); `position` names it (in line
  under 1, above or below from 1, well above or below from 2). *The score*
  (`score_of`) is the sum of the positive `risk_z` of leverage and the entry
  multiple (the margin is shown, not scored: the transactions it is tested
  on don't all give revenue), none unless both have a peer group; the deal
  is *unusual* (`unusual`) when any figure's `risk_z` is 2 or more. *Shown
  only where its card beats the baseline*: `shown_score` reads the headline
  verdict for the deal's S&P region from the card (`card_result`, `_card`)
  and gives the value only when it is "beats the baseline". *Deals like it*
  (`similar_deals`, only while the reference library is on): the approved
  reference transactions in the same S&P region and GICS sector
  (`sector_of`, §14's industry map), the same size bucket (entry value in US
  dollars) first. `_value` and `_number` read figures. Worked: a German
  machinery deal at 5.0x debt and 11.0x EBITDA with a 15% margin sits
  against Developed Europe's 210 machinery companies (1.87x, 14.98x,
  12.97%); Europe's 65 industries spread 1.81x on leverage, so leverage is
  (5.0 − 1.87) / 1.81 = 1.73 spreads above, the price 1.04 below, the margin
  0.33 above: score 1.73, not unusual, and not shown, because the card has
  one European case.
- **Distress predictor** ([ml/distress_model.py](../ml/distress_model.py);
  PLAN.md 5.3). Each year's chance of default, from published tables only:
  nothing is trained. *Each year's band* (`bands`) is the weaker of two
  reads, both folded to the letter grades S&P's regional tables print
  (`band_of`: 'bb-' is BB, CCC to D are CCC/C; `BANDS`). Coverage
  (`coverage_band`): EBIT / interest through Damodaran's table (§8's
  `rating_for_coverage`, the same bounds); a year with no interest is not
  held back by it. Leverage (`leverage_band`): debt at the start of the year
  over the year's EBITDA, as the deal is priced (with leases on the post
  view, the liability added to debt and the lease cost to EBITDA), through
  S&P's Corporate Methodology (January 2024): Table 17's standard-volatility
  bounds give the financial risk profile (`leverage_profile`: under 1.5x
  minimal, 1.5-2 modest, 2-3 intermediate, 3-4 significant, 4-5 aggressive,
  over 5 or debt against a non-positive EBITDA highly leveraged; no debt is
  minimal, `leverage_of`), and Table 3 combines it
  with the deal's business risk profile into an anchor (`anchor`, the weaker
  where the table prints two; `business_risk` is a deal input,
  `DEFAULT_BUSINESS_RISK` = 4, fair). *The rate* (`probabilities`): S&P's
  2024 study's average cumulative default rates C for the band, Table 25's
  block for the US, Europe and emerging markets, else Table 24's global one
  (`table_for`), turned into forward rates h(t) = (C(t) − C(t−1)) / (100 −
  C(t−1)) (`hazards`), read at the year's age and, past the last year
  printed, at the last; the yearly probability is S(t−1) × h(t) and the
  cumulative 1 − S(t), S being the chance of no default by the end of a
  year. A deal held in one band therefore defaults exactly as the table
  says. *Shown only where its card beats the baseline* (`card_result`,
  `_card`, `_context`): the bands, coverage and leverage are always
  answered; the probabilities only in a region where the card's headline
  verdict is "beats the baseline", which today is none. `deal_view` reads
  the deal model's run (`_figure` drops a non-finite ratio); `simulated_view`
  reads every simulated path (§9's `credit` run, handed over by
  `simulated_distress` in [core/montecarlo.py](../core/montecarlo.py), which
  adds the leases and drops the paths) and answers each year's share of
  paths in each band and the mean probabilities. Worked: the default deal
  in the UK opens year one at 600 / 105 = 5.71x (highly leveraged, 'b' at
  fair business risk) and 1.92x coverage (B+, so B): band B; year three's
  4.66x is aggressive ('bb-') and 2.32x is BB+, so BB. Europe's table gives
  year one 1.75%, year two (4.72 − 1.75) / 98.25 = 3.02% of the survivors.
- **Multiple predictor** ([ml/multiple_predictor.py](../ml/multiple_predictor.py),
  `predict`; PLAN.md 5.4). Entry and exit EV/EBITDA ranges for the deal's
  industry in its region, from Damodaran's averages (§14) and his archive of
  past editions (§14's history): nothing is fitted to deals. *The peer
  group* (`peer_group`) is the closest group in the country's chain whose
  industry has `MIN_FIRMS` = 20 companies and a usable multiple (`usable`:
  above 0 and at most 100x) in its latest year, **never the global group**
  (`no_peers`; without a country `no_country`; an empty industry is the
  whole market). `series` is the industry's usable multiple by year in the
  group, `group_series` every industry's. *The moves* (`moves`): for a
  horizon of h years, every pair of an industry's multiples h years apart
  in the group, as (`gap` in the base year, ln(later / earlier)), where the
  gap is ln(multiple / the group's median industry that year,
  `market_median`, worked out once a year by `market_medians`; `quantile`
  interpolates linearly). *The line* (`fit`,
  a `Fit`): the moves' least-squares line on the gap, a + b x gap (b is
  negative: expensive industries cheapen, cheap ones catch up), and its
  sorted misses; under `MIN_PAIRS` = 30 moves there is no range. *The
  range* (`band`): the latest multiple x exp(a + b x its gap + the misses'
  10th, 50th and 90th percentiles, `QUANTILES`): low, suggestion, high.
  *Horizons* (`ranges`, `entry_horizon`): entry is the year after the
  latest published one (a later year if the stored edition is older),
  exit is entry plus the deal's hold; figures are rounded to two places
  (`_rounded`). *Shown only where its card beats the baseline*: `shown`
  reads each range's verdict from the card's `entry` and `exit` sets
  (`card_result`, `_card`) for the S&P region of the peer group the range
  is built from (`GROUP_REGION`: a Korean deal's peers are Damodaran's
  emerging markets, so emerging's verdict decides), and gives its figures
  only for a horizon the card tested (`TESTED_HORIZONS`: entry 1, exit 4
  to 8, holds of three to seven years) where the verdict is "beats the
  baseline"; `hidden_because` says why not (`untested_horizon`,
  `few_moves`, or the verdict). Only the history of the deal's own groups
  is read (`history_groups`: its chain without the global group). *Comparables*: `by_region`, the industry's latest multiple in
  every group (`_latest`); `same_sector`, the peer group's industries with
  20 companies in the same GICS sector (`sector_of`, §14's map; `_name`
  reads names), cheapest first; and, only while the reference library is on,
  the approved reference transactions like it (the risk score's
  `similar_deals`). `market_band` is the card's baseline: the same
  percentiles across the group's industries in a year. Worked: a German
  machinery deal held five years reads Developed Europe's 210 machinery
  companies at 14.98x (2025) against a market median of 12.03x, a gap of
  ln(14.98 / 12.03) = 0.22; the 801 one-year moves give a = 0.048 and
  b = -0.135, so the entry range for 2026 is 10.92x to 20.50x around 15.43x,
  and the 470 six-year moves an exit range for 2031 of 10.20x to 24.17x
  around 16.12x.
- **Growth calibrator** ([ml/growth_calibrator.py](../ml/growth_calibrator.py),
  `calibrate`; PLAN.md 5.5). Sets the simulation's growth mean and spread
  (§9's `mc_growth_mean`, `mc_growth_std`) from how listed companies'
  revenue really grew, by S&P region and GICS sector, where §9's sourced
  range reads only the economy's growth. *The data* (`load`,
  `data/firm_growth.json`, written by `python -m tests.ml_firm_growth`):
  every SEC filer's yearly revenue from the SEC's XBRL frames (the revenue
  concepts of `US_GAAP_ITEMS` and `IFRS_ITEMS`, every currency), its
  country (business address) and SIC code, the ECB's yearly average
  exchange rates and the IMF's nominal growth by economy. *A case*
  (`spans`, a `Span`): one company's yearly growth over h = 3 to 7 years
  (`HORIZONS`, `Span.horizon`), (end / start)^(1/h) − 1, both ends from the
  highest-priority revenue concept reporting both years in one currency
  (`_pick`, `concept_order`); a company under `FLOOR_USD_M` = 50 million
  US dollars of revenue in the start year (the ECB's average rate that
  year) is left out, as is a financial company or one without a code
  (`sector`, mapping SIC ranges to GICS sectors in `SIC_SECTORS`). *What
  is measured* (`Span.excess`): ln((1 + company) / (1 + economy)), the
  economy's growth compounded over the same years (`compound`) for the
  company's country, else its region's median economy, else every covered
  economy's (`economy_growth`, the starting figures' rule). *The range*
  (`fit`, `fit_group`, `lookup`): the excesses' 10th, 50th and 90th
  percentiles (`quantile`, linear) in the deal's region, sector and hold,
  or the region's every sector when the sector has fewer than
  `MIN_COMPANIES` = 30 companies; none under that in the region. `band`
  applies them to the country's nominal growth this year (the IMF
  projection the starting figures use, `growth_figure` in
  [benchmarks/starting.py](../benchmarks/starting.py)): low, middle, high
  = (1 + growth) × exp(percentile) − 1; the Settings are the normal draw
  whose central 80% is that range, mean = (low + high) / 2 and spread =
  (high − low) / (2 × 1.2816), in % to two places (`_pct`). `ranges` fits
  every group on every case and `write_ranges` stores them in
  `growth_ranges.json`, which the app reads (`served`, `served_model`).
  *Shown only where its card beats the baseline*: `card_result` reads the
  card's verdict for the deal's S&P region, and only holds the card tested
  get a range (`untested_horizon` otherwise); without a country
  `no_country`, without stored economic data `no_growth`. Worked: a US
  software deal held five years rests on 696 US information technology
  companies (3,448 five-year spans, 2008-2025), whose excesses' percentiles
  are −0.103, 0.024 and 0.182; with the economy at 4.00% the range is
  1.04 × e^−0.103 − 1 = −6.20% to 1.04 × e^0.182 − 1 = 24.81%, so the mean
  is 9.31% and the spread 12.10% (a US utility: 6.08% and 8.10%).
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
  never stored twice, numbers read as decimals (`_canonical`) so the
  repository's copy and one sent through the API match. A transaction joins the library on the second approval
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

## Model validation

[validation/cases.py](../validation/cases.py),
[validation/report.py](../validation/report.py),
[validation/tags.py](../validation/tags.py),
[validation/run.py](../validation/run.py). Tests:
[tests/test_validation.py](../tests/test_validation.py). PLAN.md 4.6: does
the model's risk read come true as often as it claims? The nightly
`validation-report` task (`run`, `build`) writes a report the Backtest ->
Validation step shows.

**Three checks.**

- **Default risk** (`default`), on the approved reference transactions (§14,
  only while the library is on). `predicted_default` is what the app answers
  for the deal entered with its filed headline figures only -- EBITDA, entry
  multiple = value paid / EBITDA, debt share = debt / value -- every other
  input and Setting at its default, for the default hold of five years: the
  `credit_view` of §8. `library_case` says what happened: distress (missed
  payment, bankruptcy, restructuring) within the hold of the closing year; an
  exit before then counts as none (so the observed rate is a lower bound); a
  deal still held counts once the hold has passed, not before.
- **IRR range** (`irr_range`) and **loss** (`loss`), on users' deals whose
  owners opted in and which have an exit. `contributed_cases` runs plan vs
  actual (§10) with `PATHS` = 2,000 simulated paths (seed 42): the percentile
  at which the actual IRR fell among the plan's paths, actual minus planned
  IRR, the plan's probability of loss (paths with IRR below zero) and whether
  the actual IRR was below zero. `_read` refuses a deal without an exit, with
  actuals in another currency or that the model refuses; such deals are
  counted in the run log, never in the report.

**Always tested on newer data.** A case is *out of time*, the report's
headline, only if its outcome became known after everything its prediction
was made from. For the default check that is after `fit_until`: the newest
year in the tables `credit_view` reads (S&P's study sample ends 2024;
Damodaran's January 2026 table is built from 2025), so 2025. For a user's
deal, `plan_for` takes the newest version saved **before the deal's actuals
were first saved** (`deals.actuals_first_saved_at`, kept when actuals are
cleared); without one, the working copy is used and the case is in-sample.
In-sample cases are reported apart and never as the headline.

**Groups** (`reference_tags`, `deal_tags`, `_known`): S&P's region, GICS
sector, entry enterprise value in US dollars (the library's size bands) and
era (the library's), every bucket listed and `unknown` for what a deal does
not say. A user's deal's sector comes from its Damodaran industry through
`INDUSTRY_SECTOR`; its size converts entry EBITDA x entry multiple at the
ECB's rate on the day the plan was saved (`_usd_per`: the newest stored rate
when that day is older than the 400 days kept); its era is that day's year.

**Statistics** (`build`, `sample`, `split`, `_cell`, `_stats`; a group
needs `MIN_CASES` = 5 cases):

- Probabilities (`probability_stats`): mean predicted and observed share
  (per cent), expected (sum of p) and observed counts, bias = observed -
  predicted (percentage points), Brier score = mean (p - outcome)^2, and
  z = (observed - expected) / sqrt(sum p(1 - p)), *consistent* when
  |z| <= 1.96.
- Ranges (`range_stats`): for the central 50, 80 and 90% of paths
  (`inside`: from (100 - L)/2 to 100 - (100 - L)/2), the share of actual
  IRRs inside, its 95% Wilson interval (`wilson`: centre
  (p + z^2/2n)/(1 + z^2/n), half-width z sqrt(p(1-p)/n + z^2/4n^2)/(1 + z^2/n)),
  *consistent* when every claimed level lies in its interval; bias = mean
  actual - planned IRR (points) and the mean percentile (50 if unbiased).

**Anonymity.** The report holds counts and rates per group, never a deal,
owner, name or figure. Users' deals count only in groups (`_contributed`,
`_too_few`): a group with 1-4 of them shows neither its statistics nor its
count of them; within a dimension, if the hidden groups hold fewer than
`MIN_CONTRIBUTED` = 5 users' deals together, the smallest shown groups
holding any are hidden too until they do, so the overall figures less the
shown groups never single out fewer than five. Reference transactions are
public filings and always counted. Values are rounded (`_r`).

Worked: the ten repository transactions (all in-sample: every outcome is
before 2026) predict a mean default risk of 16.73% over five years against
one distress within it (Masonite, 2009): 10.0% observed, bias -6.73 points,
Brier 0.1242, z = -0.63, consistent. Toys "R" Us and Gymboree defaulted
after their fifth year, so count as none.

## ML evaluation and model cards

[ml/evaluation/harness.py](../ml/evaluation/harness.py),
[ml/evaluation/deal_risk.py](../ml/evaluation/deal_risk.py),
[ml/evaluation/distress.py](../ml/evaluation/distress.py),
[ml/evaluation/multiples.py](../ml/evaluation/multiples.py),
[ml/evaluation/growth.py](../ml/evaluation/growth.py),
[ml/evaluation/surrogate.py](../ml/evaluation/surrogate.py). Tests:
[tests/test_ml_evaluation.py](../tests/test_ml_evaluation.py),
[tests/test_model_cards.py](../tests/test_model_cards.py). PLAN.md 5.1: every
model the app loads is tested the same way, and the result is its card
(`docs/model-cards/<id>.md`, from `ml/cards/<id>.json`). A card changes no
number the app shows; it says how far to trust one.

**The harness.** `walk_forward` is the time split: for each cutoff year, the
model is fitted on the cases before it and predicts those from it to the
next cutoff (the last fold is open-ended); a fold with nothing to train on or
test is skipped, and a case before the first cutoff or without a date is
never tested. `summarize` computes each statistic for the model and for a
simple baseline on the same cases; `better` decides the headline: the
verdict is *beats the baseline* only when the model is strictly better,
*not enough data* under `MIN_CASES` = 5 cases or when the headline can't be
computed, else *does not beat the baseline*. `evaluate_set` does it overall
and for each of S&P's regions (§14's, as the validation report uses).
Statistics: `auc`, the share of (distressed, not distressed) pairs in which
the distressed case scores higher, ties counting half; `mean_abs`, the mean
absolute error; `share`, the share of true flags. Values are rounded to four
places (`_round`); `metric_specs` records each statistic's direction and the
tolerance the CI gate allows.

**Risk score** (`ml.evaluation.deal_risk`). Cases (`cases`): the ten
sourced reference transactions in the repository (§14), each in its S&P
region with the Damodaran industry of its business (`INDUSTRY`), truth =
distress. Peers: the January 2026 averages as recorded
(`peer_tables`, `data/peer_tables.json`). `predict` gives the score and flag
the app shows (`compare`); a deal the app would answer "not enough data" is
left out with its baseline (`scored`). No time split: the score fits nothing
to outcomes. Baseline (`baseline`): leverage as the score, flagged above the
6.0x of §8. Statistics: `_auc` of the score against distress, `_caught`
(distressed deals flagged) and `_false_alarms` (other deals flagged).
`evaluate` writes the one set. Worked: eight transactions scored (D&B's 15
US information services companies and Masonite's 8 Canadian building
materials companies are too few); in the US, five, two distressed, the score
ranks a distressed deal above another in 4 of 6 pairs (AUC 0.67) against
leverage's 2 of 6 (0.33), so it beats the baseline there and is shown for US
deals; every other region has one case.

**Distress predictor** (`ml.evaluation.distress`). Two sets. The headline,
`reference_deals`: the ten sourced transactions (`cases`), each entered
with its filed EBITDA, multiple and debt share in its own country and
everything else at the defaults (`deal_inputs`, `run`, as the validation
report's default check), truth = distress at any time. `predict` is the
predictor's cumulative chance of default over the hold, read with every
region shown; `baseline` is the deal's year-one default risk (§8's
`credit_view`) as a fraction. Statistics: `_auc` and `_brier`, the mean
squared gap between the chance and what happened. `calibration`
(`calibration_cases`, `calibration_rows`): one case per S&P table, band and
horizon printed, the predictor's cumulative rate for a deal held in the band
(`implied_pct`) against the table's, the baseline reading Table 24 in every
region; `_gaps`, `_mean_gap` and `_max_gap` in percentage points, within
`CALIBRATION_TOLERANCE_PP` = 0.01. `evaluate` writes both. Worked: every
calibration gap is 0.00 (the baseline is off by up to 19.39 points in
emerging markets); on the reference transactions the predictor's AUC is
0.43 against the year-one default risk's 0.52 (US, six deals: 0.25 against
0.375), so it is shown nowhere.

**Multiple predictor** (`ml.evaluation.multiples`). Cases (`cases`): every
industry's multiple in every year of every group but the global one, from
the archive and the January 2026 edition as recorded (`tables`,
`data/multiple_history.json`, every industry), predicted from its multiple h
years earlier, with that year's gap; each group in its S&P region
(`GROUP_REGION`). Two sets: `entry` (h = 1, the headline) and `exit` (h = 4
to 8, holds of three to seven years). Walk-forward, a cutoff each year from
`FIRST_CUTOFF` = 2014 to the newest year stored (`cutoffs`): `fit` draws the line through the moves that ended
before the cutoff, by group and horizon, and `predict` gives the app's
range (`band`); a horizon with too few earlier moves gives none and the case
is left out (`scored`). Baseline: `market_band` in the base year, the
region's whole market. Statistics: `interval_score` (the headline: the
range's width plus 2 / 0.2 = ten times how far the multiple fell outside
it, Gneiting and Raftery's score for a central 80% range), `inside` (the
share inside it), `median_error` (the suggestion's mean distance from the
multiple) and `width`, all in turns of EBITDA. `_set` and `evaluate` write
both sets. Worked: entry, 3,401 industry-years from 2016 to 2025, interval
score 17.48 against the market's 26.82, 73% inside; exit, 10,143, 23.41
against 29.07, 73% inside. Every region beats the baseline in both sets,
so both ranges are shown everywhere a peer group has enough companies.

**Growth calibrator** (`ml.evaluation.growth`). Cases: every company span
the calibrator rests on (`spans`), as `_case`s in the S&P region of the
company's country. Strictly out of time (`scored`): each start year from
`FIRST_START` = 2012, the spans starting that year are predicted from a
`fit` on the spans that had **ended** by then, so nothing a prediction
reads comes after its start. The economy's growth stands in for the IMF
projection the app uses (no past projections are free): the country's
nominal growth compounded over the `FORECAST_YEARS` = 5 years to the start.
Baseline (`baseline_spread`, `_stdev_to`): the range in force from §9's
sourced figures, the same centre with the spread of the country's nominal
growth from year to year since 2000 (at least six years; else its region's
median spread), as the normal draw's 80%. Statistics, in percentage points
of yearly growth: `interval_score` (the headline), `inside`, `centre_error`
(the simulation mean's distance from the growth) and `width`. Worked:
67,933 spans of 4,294 companies starting 2012-2022; 77% fell inside the
calibrated range against 27% for the economy-wide one, interval score 60.2
against 88.9; every region beats the baseline (inside: US 77%, Europe 83%,
emerging 78%, other developed 74%), so the range is offered everywhere a
region has enough companies.

**Live sliders** (`ml.evaluation.surrogate`). Cases (`load_cases`): 230
deals, each economy the app covers with ten industries, whose inputs are
what "Use sourced figures" gives (§9's sourced ranges, §14's starting
figures, `features_from` turning them into the network's fractions) with
the default 60% debt, recorded on the economic data's day. `simulate` runs
§9's simulation on the surrogate's fixed training deal: truth on 10,000
paths, the baseline on 200 (fast enough to run live), each seeded from the
deal's id (`_seed`). Statistics (`_mae_pp`): the mean absolute error of the
median, 5th and 95th percentile IRRs and the wipeout rate, in percentage
points. `in_range` says whether every input lies inside `L_BOUNDS` and
`U_BOUNDS`, the training ranges. Worked: the median IRR is 0.54 points off
over all regions against the 200-path simulation's 0.79, and beats it in
every region; inside the training ranges, 0.36 against 0.55.

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
| `api.limits.EXPLAIN_RUN_LIMIT_S` | 20 | a Monte Carlo run slower than this (seconds) is not explained |
| `analytics.driver_attribution.TAIL_SHARE` | 0.05 | the share of paths in each explained tail |
| `analytics.driver_attribution.MAX_TAIL_PATHS` | 2500 | most paths explained in a tail |
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
