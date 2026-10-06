# Hand-checked reference cases (PLAN.md 3.4)

PLAN.md's sixth principle is that the calculations are proven correct by
hand-checked cases, not by history. These 26 small deals are that proof: each
one is worked out by hand from [docs/methodology.md](../../docs/methodology.md),
never from the engine's code, and
[tests/test_reference_cases.py](../test_reference_cases.py) requires the
engine to give the same answer.

| File | What it is |
|---|---|
| [cases.py](cases.py) | The cases: inputs, the reasoning in words, and the working as one formula per line |
| [reference_cases.xlsx](reference_cases.xlsx) | The same working as a spreadsheet: an index, then one sheet per case with live Excel formulas |
| [workbook.py](workbook.py) | Writes the spreadsheet and works the formulas out the way a spreadsheet does |

## How a case is checked

1. Its formulas are evaluated in order (plain arithmetic, `max` and `min`
   only, so the spreadsheet reads them the same way).
2. The headline figures typed by hand (entry and exit equity) must equal what
   the formulas come to.
3. The deal goes through `POST /api/deal/run` exactly as the deal screen sends
   it, and every line that names an engine figure must match **to 0.01**:
   money in the deal's own unit, IRR and interest rates in percentage points,
   MOIC in turns. A case checks at least ten figures, always including entry
   equity and the returns.
4. The committed workbook must be what `workbook.py` writes from the cases.

Each check was mutation-tested when the cases were written: breaking the floor
rule, the PIK add-back, the sweep share, the loss limit, the minimum tax, the
carried interest, the commitment fee, the lease liability, the working-capital
days, the unit conversion, the financing fee and the interest income each
made at least one case fail.

## The cases

The business is the same throughout unless a case changes it: EBITDA 100 on
revenue of 400 (gross margin 50%, opex 30%, D&A 5%), capex equal to D&A, no
working capital, 25% tax, bought and sold at 8x, held three years, no fees.

| # | Case | What it proves |
|---|---|---|
| 01 | No leverage | Cash piles up when nothing is owed; the exit grid, including a five-year column that is its own run |
| 02 | One tranche, no sweep | Interest on a fixed balance, cash built beside the debt |
| 03 | One tranche, full sweep | Interest on the opening balance each year as the sweep repays it |
| 04 | Amortising term loan | Scheduled repayments of the original principal |
| 05 | Fees only, no debt | Transaction fee and other uses raise the cheque, not the value; the bridge's fee step |
| 06 | All the fees | Transaction, financing and a tranche's own arrangement fee, each its own line |
| 07 | Debt as a share of EV | The percentage structure (no tranche list) with scheduled amortisation |
| 08 | A bullet the cash can't pay | **Known difference, finding 11**: the engine is higher by exactly the "unfunded repayment" it reports |
| 09 | Floating SONIA with a floor | The floor applies to the reference before the margin; a reference path (sterling) |
| 10 | PIK note | PIK interest is an expense, added back to cash, compounding on the balance |
| 11 | Unitranche, half the sweep | A sweep share of all the cash available, including cash kept the year before |
| 12 | Two tranches in order | The sweep waterfall moving to the next facility once the first is repaid |
| 13 | Revolver | Commitment fee on the undrawn part; draws to fund a repayment the cash can't |
| 14 | 30% of EBITDA interest cap | Disallowed interest carried forward |
| 15 | Fixed interest cap | Carried interest used once there is room |
| 16 | Tax loss | A loss carried forward and absorbed |
| 17 | Loss relief limited | An allowance in full plus 60% of the rest (Germany's shape) |
| 18 | Minimum tax | A low rate topped up to a minimum on book profit |
| 19 | IFRS 16, leases as debt | Lease cost added back at entry and exit, the liability counted as debt |
| 20 | IFRS 16 reporter, pre view | The same company valued on EBITDA after leases: case 02 exactly |
| 21 | Euro deal in thousands | Every figure exactly 1,000 times the millions answer |
| 22 | Yen in billions, March year-end | Fiscal years are labels; a negative reference floored at zero |
| 23 | Working capital in days | Growth with receivable, inventory and payable days |
| 24 | Minimum cash | Funded by the sponsor, earning 0.5%, still there at exit |
| 25 | Held one year | IRR = MOIC - 1 |
| 26 | Two facilities sharing the sweep | Each share is of the cash available, not of what the one above left |

## What the cases steer around, and why

The open findings in CLAUDE.md are differences between the engine and a
correct hand answer that the user has not yet approved fixing. A case that hit
one would either fail or pin the wrong number, so:

- **Finding 11** (a cash shortfall funded out of nothing): every case keeps
  its cash flow positive and its repayments covered, except case 08, which
  hits it on purpose and checks the engine's answer is the hand answer *plus*
  the shortfall the risk warning reports. When the finding is fixed, case 08's
  last line goes and its exit equity becomes the hand answer.
- **Finding 10** (money rounded to 0.01 million at every line): the numbers
  are chosen so that rounding never moves a figure by more than 0.01. It
  matters more than "a small deal loses precision" suggests: the first drafts
  of cases 13 and 24 had a half-cent recurring each year (a net income of
  30.075), and the engine's rounding built up to 0.015 by year three.
- **Findings 12 and 13** are in the Monte Carlo simulation, which these cases
  do not run.

## Adding a case

Write a `_cNN()` function in `cases.py` and add it to `CASES`: the inputs as
the API takes them, the same inputs as decimals in `given`, the working as
`Line`s (reuse `operating`, `income`, `entry`, `exit_returns`, `bullet`,
`full_sweep`), the headline answers you worked out on paper, and the reasoning
in a few sentences. Then:

```bash
.venv/Scripts/python.exe -m tests.reference.workbook          # rewrite the spreadsheet
.venv/Scripts/python.exe -m pytest tests/test_reference_cases.py
```

If the engine disagrees, find out which is wrong before changing anything. A
slip in the working is fixed in the working; a real difference in the engine
is a model finding (CLAUDE.md "Model findings"), recorded and not fixed
without the user's approval. Never edit a case's numbers to agree with the
engine.
