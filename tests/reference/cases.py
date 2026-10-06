"""Hand-checked reference cases (PLAN.md 3.4).

Each case is a small deal worked out the way a person would on paper or in a
spreadsheet: inputs, then one line per figure, each a formula over the inputs
and the lines above it. The formulas come from ``docs/methodology.md``, never
from the engine's code, and ``tests/reference/workbook.py`` writes them into
``reference_cases.xlsx`` as live Excel formulas, one sheet per case.

``tests/test_reference_cases.py`` then runs every case through the API
(``POST /api/deal/run``) and requires each line marked with ``check`` to match
the engine to 0.01: money in the deal's own unit, IRR and rates in percentage
points, MOIC in turns.

The numbers are chosen to be easy to follow, not realistic: EBITDA 100 on
revenue of 400 (gross margin 50%, opex 30%, D&A 5%), capex equal to D&A, no
working capital, 25% tax and 8x in and out, unless a case says otherwise. Most
deals are held three years. Fees are off unless the case is about fees.

Formula language: numbers, the names in ``given``, earlier line keys,
``+ - * / **``, brackets, ``max`` and ``min``. Nothing else.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Line:
    key: str
    label: str
    formula: str
    check: str = ""        # path into the /api/deal/run answer, "" = shown only
    kind: str = "money"    # money | pct (decimal shown in %) | x (a multiple)


@dataclass(frozen=True)
class Case:
    id: str
    title: str
    covers: tuple          # which of PLAN.md 3.4's list it proves
    reasoning: str
    given: dict            # the hand's inputs: decimals and money in the deal's unit
    inputs: dict           # the same deal as the API takes it
    lines: tuple
    answers: dict          # headline figures typed by hand: line key -> value
    settings: dict = field(default_factory=dict)


# Fees and amortisation off: a case that is about fees turns them on itself
NO_FEES = {"tx_fee_pct": 0.0, "fin_fee_pct": 0.0, "other_uses": 0.0, "def_senior_amort": 0.0}

# The business every case starts from, as the hand writes it and as the API takes it
BASE_GIVEN = dict(ebitda=100.0, gm=0.50, opex=0.30, da=0.05, g=0.0, capex=0.05, nwc=0.0,
                  tax=0.25, entry_mult=8.0, exit_mult=8.0, hold=3)
BASE_INPUTS = dict(ebitda=100.0, entry_mult=8.0, exit_mult=8.0, hold=3, growth=0.0,
                   gross_margin=50.0, opex=30.0, da=5.0, tax=25.0, capex=5.0, nwc=0.0,
                   debt_pct=0.0, mincash=0.0)


def given(**kw) -> dict:
    return {**BASE_GIVEN, **kw}


def inputs(**kw) -> dict:
    return {**BASE_INPUTS, **kw}


# ---------------------------------------------------------------------------
# Lines every case shares: the operating model, the income statement, returns
# ---------------------------------------------------------------------------
def operating(n: int, ebitda: str = "ebitda", nwc: str = "nwc") -> list:
    """Revenue backed out of EBITDA, grown each year, and the lines down to
    EBITDA, capex and the change in working capital (methodology 3.2, 3.4)."""
    out = [Line("rev_0", "Base revenue = EBITDA / (gross margin - opex + D&A)",
                f"{ebitda} / (gm - opex + da)")]
    for t in range(1, n + 1):
        i = t - 1
        out += [
            Line(f"rev_{t}", f"Y{t} revenue", f"rev_{t - 1} * (1 + g)", f"operating_model.revenue.{i}"),
            Line(f"ebit_{t}", f"Y{t} EBIT = revenue x (gross margin - opex)", f"rev_{t} * (gm - opex)",
                 f"operating_model.ebit.{i}"),
            Line(f"da_{t}", f"Y{t} D&A", f"rev_{t} * da", f"operating_model.da.{i}"),
            Line(f"ebitda_{t}", f"Y{t} EBITDA", f"ebit_{t} + da_{t}", f"operating_model.ebitda.{i}"),
            Line(f"capex_{t}", f"Y{t} capex", f"rev_{t} * capex", f"cash_flow.capex.{i}"),
            Line(f"dnwc_{t}", f"Y{t} change in working capital", f"rev_{t} * {nwc}",
                 f"cash_flow.delta_nwc.{i}"),
        ]
    return out


def income(t: int, interest: str, tax: str = "max(ebt_{t}, 0) * tax", other_income: str = "0",
           pik: str = "") -> list:
    """Profit before tax, tax, net income and levered free cash flow for year t
    (methodology 3.3-3.4). ``tax`` may be a case's own formula (tax rules)."""
    i = t - 1
    fcf = f"ni_{t} + da_{t} - capex_{t} - dnwc_{t}" + (f" + {pik}" if pik else "")
    return [
        Line(f"int_{t}", f"Y{t} interest expense", interest, f"operating_model.interest_expense.{i}"),
        Line(f"ebt_{t}", f"Y{t} profit before tax = EBIT - interest + interest income",
             f"ebit_{t} - int_{t} + {other_income}", f"operating_model.ebt.{i}"),
        Line(f"tax_{t}", f"Y{t} tax", tax.format(t=t), f"operating_model.taxes.{i}"),
        Line(f"ni_{t}", f"Y{t} net income", f"ebt_{t} - tax_{t}", f"operating_model.net_income.{i}"),
        Line(f"fcf_{t}", f"Y{t} levered free cash flow (before repaying debt)", fcf,
             f"cash_flow.levered_fcf.{i}"),
    ]


def entry(debt: str, costs: str = "0", addback: str = "0", liability: str = "0",
          mincash: str = "0") -> list:
    """Entry EV and the sponsor's cheque (methodology 3.1)."""
    return [
        Line("ev", "Entry EV = (EBITDA + lease add-back) x entry multiple",
             f"({_op_ebitda()} + {addback}) * entry_mult"),
        Line("debt", "Debt drawn at close", debt),
        Line("costs", "Entry costs (fees and other uses)", costs),
        Line("equity", "Sponsor equity = EV + costs + minimum cash - debt - lease liability",
             f"ev + costs + {mincash} - debt - {liability}", "returns.entry_equity"),
    ]


def _op_ebitda() -> str:
    return "op_ebitda"


def exit_returns(n: int, net_debt: str, addback: str = "0") -> list:
    """Exit value, equity, MOIC and IRR (methodology 3.7). No interim
    dividends, so IRR = MOIC^(1/n) - 1."""
    return [
        Line("net_debt_exit", "Net debt at exit (debt - cash + lease liability)", net_debt,
             "returns.net_debt_at_exit"),
        Line("exit_ev", "Exit EV = (final-year EBITDA + lease add-back) x exit multiple",
             f"(ebitda_{n} + {addback}) * exit_mult", "returns.exit_ev"),
        Line("exit_equity", "Exit equity = exit EV - net debt", "exit_ev - net_debt_exit",
             "returns.net_exit_equity"),
        Line("moic", "MOIC = exit equity / entry equity", "exit_equity / equity", "returns.moic", "x"),
        Line("irr", "IRR = MOIC ^ (1 / years) - 1", "moic ** (1 / hold) - 1", "returns.irr", "pct"),
    ]


def op_ebitda(formula: str = "ebitda") -> list:
    return [Line("op_ebitda", "Operating EBITDA (what the model grows)", formula)]


def bullet(name: str, n: int, rate: str = "r", debt: str = "debt", other_income: str = "0",
           tax: str = "max(ebt_{t}, 0) * tax") -> list:
    """A facility that neither amortises nor takes the sweep, outstanding
    past the exit: interest on the same balance every year, the cash piles up."""
    out = []
    for t in range(1, n + 1):
        out += income(t, f"{debt} * {rate}", tax=tax, other_income=other_income)
        prev = f"cash_{t - 1}" if t > 1 else "0"
        out.append(Line(f"cash_{t}", f"Y{t} closing cash above the minimum", f"{prev} + fcf_{t}"))
        out.append(Line(f"bal_{t}", f"Y{t} {name} closing balance", debt,
                        f"tranches.{name}.{t - 1}.ending_balance"))
    return out


def full_sweep(name: str, n: int, rate: str = "r", mandatory: str = "0", tax: str = "max(ebt_{t}, 0) * tax") -> list:
    """One facility that takes every spare unit of cash: interest on the
    opening balance, then the mandatory repayment and the sweep."""
    out = []
    for t in range(1, n + 1):
        prev = f"bal_{t - 1}"
        out += [Line(f"opening_{t}", f"Y{t} {name} opening balance", prev,
                     f"tranches.{name}.{t - 1}.beginning_balance")]
        out += income(t, f"{prev} * {rate}", tax=tax)
        out += [
            Line(f"mand_{t}", f"Y{t} mandatory repayment", f"min({mandatory}, {prev})",
                 f"debt_schedule.total_mandatory_repayment.{t - 1}"),
            Line(f"sweep_{t}", f"Y{t} cash sweep = cash left after the mandatory repayment",
                 f"min(max(fcf_{t} - mand_{t}, 0), {prev} - mand_{t})",
                 f"debt_schedule.total_cash_sweep.{t - 1}"),
            Line(f"bal_{t}", f"Y{t} {name} closing balance", f"{prev} - mand_{t} - sweep_{t}",
                 f"tranches.{name}.{t - 1}.ending_balance"),
        ]
    return out


def notes(amount: float, rate: float, kind: str = "senior_notes", name: str = "Notes", **kw) -> dict:
    return dict(name=name, kind=kind, amount=amount, fixed_rate=rate, maturity_years=7, **kw)


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------
def _c01():
    n = 3
    lines = [*op_ebitda(), *entry("0"), *operating(n)]
    for t in range(1, n + 1):
        lines += income(t, "0")
        lines.append(Line(f"cash_{t}", f"Y{t} closing cash: nothing to repay, so it piles up",
                          f"{f'cash_{t - 1}' if t > 1 else '0'} + fcf_{t}", f"debt_schedule.cash_balance.{t - 1}"))
    lines += exit_returns(n, f"0 - cash_{n}")
    lines += [
        Line("grid_6x_3y", "Grid: IRR at 6x after 3 years = ((6 x EBITDA + cash) / equity)^(1/3) - 1",
             f"((6 * ebitda_{n} + cash_{n}) / equity) ** (1 / 3) - 1", "exit_sensitivity.table.0.0", "pct"),
        Line("cash_5y", "Grid, held 5 years: five years of the same cash flow", "5 * fcf_1"),
        Line("grid_8x_5y", "Grid: IRR at 8x after 5 years (its own five-year run)",
             "((8 * ebitda_3 + cash_5y) / equity) ** (1 / 5) - 1", "exit_sensitivity.table.2.2", "pct"),
    ]
    return Case(
        "01_no_leverage", "No leverage", ("no leverage", "zero growth"),
        "An all-equity buyout. Each year EBIT 80 less 25% tax leaves 60 of net income; capex "
        "equals D&A, so 60 of cash a year, which stays on the balance sheet because there is "
        "nothing to repay. Exit: 800 of EV plus 180 of cash for an 800 cheque, MOIC 1.225. "
        "The exit-sensitivity grid is checked twice: a different multiple at the same hold, and "
        "a five-year hold, which is its own model run (finding 7).",
        given(), inputs(), tuple(lines), {"equity": 800.0, "exit_equity": 980.0}, NO_FEES)


def _c02():
    n = 3
    lines = [*op_ebitda(), *entry("400"), *operating(n), *bullet("Notes", n)]
    lines += exit_returns(n, f"bal_{n} - cash_{n}")
    return Case(
        "02_single_bullet", "One tranche, no sweep", ("single tranche", "zero growth"),
        "400 of fixed-rate notes at 10%, due after the exit, not swept. Interest is 40 every "
        "year, net income (80 - 40) x 75% = 30, and the cash piles up to 90. Net debt at exit "
        "400 - 90 = 310, exit equity 490 for a 400 cheque.",
        given(r=0.10), inputs(tranches=[notes(400, 10)]), tuple(lines),
        {"equity": 400.0, "exit_equity": 490.0}, NO_FEES)


def _c03():
    n = 3
    lines = [*op_ebitda(), *entry("400"), *operating(n), Line("bal_0", "Term loan at close", "debt"),
             *full_sweep("Term loan", n)]
    lines += exit_returns(n, f"bal_{n}")
    return Case(
        "03_single_sweep", "One tranche, full cash sweep", ("single tranche", "zero growth"),
        "400 at 10%, no scheduled repayment, every spare unit of cash swept. Year 1: interest "
        "40, net income 30, the loan falls to 370. Year 2: interest 37, net income 32.25, 337.75. "
        "Year 3: interest 33.775, net income 34.66875, 303.08125 left at exit.",
        given(r=0.10),
        inputs(tranches=[dict(name="Term loan", kind="institutional_term_loan", amount=400, floating=False,
                              fixed_rate=10, amort_pct=0, sweep=True, maturity_years=7)]),
        tuple(lines), {"equity": 400.0, "exit_equity": 496.91875}, NO_FEES)


def _c04():
    n = 3
    lines = [*op_ebitda(), *entry("400"), *operating(n), Line("bal_0", "Term loan at close", "debt")]
    for t in range(1, n + 1):
        lines += income(t, f"bal_{t - 1} * r")
        lines += [
            Line(f"mand_{t}", f"Y{t} scheduled repayment = 5% of the original 400", "debt * amort",
                 f"tranches.Term loan.{t - 1}.mandatory_repayment"),
            Line(f"bal_{t}", f"Y{t} closing balance", f"bal_{t - 1} - mand_{t}",
                 f"tranches.Term loan.{t - 1}.ending_balance"),
            Line(f"cash_{t}", f"Y{t} closing cash", f"{f'cash_{t - 1}' if t > 1 else '0'} + fcf_{t} - mand_{t}",
                 f"debt_schedule.cash_balance.{t - 1}"),
        ]
    lines += exit_returns(n, f"bal_{n} - cash_{n}")
    return Case(
        "04_amortising", "Amortising term loan, no sweep", ("single tranche", "zero growth"),
        "400 at 10% repaying 5% of the original principal (20) a year, not swept. Interest "
        "40, 38, 36 on the opening balances; net income 30, 31.5, 33; cash after repayments "
        "10, 21.5, 34.5. Net debt at exit 340 - 34.5 = 305.5.",
        given(r=0.10, amort=0.05),
        inputs(tranches=[dict(name="Term loan", kind="amortising_term_loan", amount=400, floating=False,
                              fixed_rate=10, amort_pct=5, sweep=False, maturity_years=6)]),
        tuple(lines), {"equity": 400.0, "exit_equity": 494.5}, NO_FEES)


def _c05():
    n = 3
    lines = [*op_ebitda(), *entry("0", costs="ev * tx_fee + other_uses"), *operating(n)]
    for t in range(1, n + 1):
        lines += income(t, "0")
        lines.append(Line(f"cash_{t}", f"Y{t} closing cash", f"{f'cash_{t - 1}' if t > 1 else '0'} + fcf_{t}"))
    lines += exit_returns(n, f"0 - cash_{n}")
    lines += [
        Line("bridge_fees", "Bridge: fees step = - entry costs", "0 - costs", "equity_bridge.entry_costs"),
        Line("bridge_delever", "Bridge: deleveraging = net debt at entry - net debt at exit",
             "0 - net_debt_exit", "equity_bridge.deleveraging"),
        Line("bridge_gain", "Bridge: total gain = exit equity - entry equity", "exit_equity - equity",
             "equity_bridge.total_gain"),
    ]
    return Case(
        "05_fees_only", "Fees only, no debt", ("fees only", "no leverage"),
        "Case 01 with a 2.5% transaction fee (20 on an EV of 800) and 10 of other costs at close. "
        "The fees buy no value: the business, its cash and its exit are case 01's, so exit equity "
        "is still 980, but the cheque is 830. The equity bridge shows the 30 as its own step.",
        given(tx_fee=0.025, other_uses=10.0), inputs(), tuple(lines),
        {"equity": 830.0, "exit_equity": 980.0},
        {**NO_FEES, "tx_fee_pct": 2.5, "other_uses": 10.0})


def _c06():
    n = 3
    lines = [*op_ebitda(),
             *entry("400", costs="ev * tx_fee + debt * fin_fee + debt * upfront"),
             *operating(n), *bullet("Notes", n)]
    lines += exit_returns(n, f"bal_{n} - cash_{n}")
    return Case(
        "06_fees_on_debt", "Transaction, financing and arrangement fees", ("fees only",),
        "Case 02 with every fee on: 1% of EV (8), the deal's 2% financing fee on the 400 drawn "
        "(8) and the notes' own 1.5% arrangement fee (6), all paid by the sponsor at close. The "
        "cheque is 422; nothing after close changes, so exit equity is case 02's 490.",
        given(r=0.10, tx_fee=0.01, fin_fee=0.02, upfront=0.015),
        inputs(tranches=[notes(400, 10, upfront_fee_pct=1.5)]), tuple(lines),
        {"equity": 422.0, "exit_equity": 490.0},
        {**NO_FEES, "tx_fee_pct": 1.0, "fin_fee_pct": 2.0})


def _c07():
    n = 3
    lines = [*op_ebitda(), *entry("ev * debt_pct * senior_pct"), *operating(n),
             Line("bal_0", "Senior loan at close", "debt"),
             *full_sweep("Senior Term Loan", n, mandatory="debt * amort")]
    lines += exit_returns(n, f"bal_{n}")
    return Case(
        "07_percentage_structure", "Debt sized as a share of EV", ("single tranche", "zero growth"),
        "The deal's percentage structure instead of a tranche list: debt 50% of EV (400), all "
        "senior at 10%, repaying 5% (20) a year with the rest of the cash swept. Repaying 20 and "
        "sweeping the rest takes the same cash as case 03's full sweep, so the balances match it.",
        given(r=0.10, debt_pct=0.50, senior_pct=1.0, amort=0.05),
        inputs(debt_pct=50.0, senior_pct=100.0, base_rate=10.0, mezz_spread=0.0),
        tuple(lines), {"equity": 400.0, "exit_equity": 496.91875},
        {**NO_FEES, "def_senior_amort": 5.0})


def _c08():
    n = 3
    lines = [*op_ebitda(), *entry("ev * debt_pct"), *operating(n),
             Line("senior_0", "Senior at close = 75% of the debt", "debt * senior_pct"),
             Line("mezz", "Mezzanine = the rest, a bullet due at the exit", "debt - senior_0")]
    for t in range(1, n):
        lines += income(t, f"senior_{t - 1} * r + mezz * (r + spread)")
        lines.append(Line(f"senior_{t}", f"Y{t} senior after the sweep", f"senior_{t - 1} - fcf_{t}",
                          f"tranches.Senior Term Loan.{t - 1}.ending_balance"))
    lines += income(n, f"senior_{n - 1} * r + mezz * (r + spread)")
    lines += [
        Line("shortfall", "Y3 mezzanine due less the cash the year made: nothing pays for this",
             f"mezz - fcf_{n}", "risk_warnings.unfunded_repayment.unfunded_total"),
        Line("hand_net_debt", "Net debt at exit, by hand: the cash held, the mezzanine still owed",
             f"senior_{n - 1} + mezz - fcf_{n}"),
        Line("hand_exit_equity", "Exit equity, by hand", f"ebitda_{n} * exit_mult - hand_net_debt"),
        Line("engine_exit_equity", "What the engine reports: by hand plus the cash it conjures (finding 11)",
             "hand_exit_equity + shortfall", "returns.net_exit_equity"),
    ]
    return Case(
        "08_finding_11", "Known difference: a bullet the cash can't pay (finding 11)",
        ("open finding",),
        "Senior 300 at 10% swept, mezzanine 100 at 14% repaid in one go at the exit. In year 3 "
        "the business makes 31.20 of cash against 100 due. By hand the mezzanine is still owed "
        "(or the cash goes negative, the same thing at exit), so net debt is 312.77 and exit "
        "equity 487.23. The engine resets cash to the minimum whatever happened (CLAUDE.md "
        "finding 11, open, awaiting approval), so it reports 556.03: higher by exactly the "
        "68.80 the 'unfunded repayment' warning names. This case pins the size of the finding "
        "by hand; it changes when the finding is fixed.",
        given(r=0.10, spread=0.04, debt_pct=0.50, senior_pct=0.75),
        inputs(debt_pct=50.0, senior_pct=75.0, base_rate=10.0, mezz_spread=4.0),
        tuple(lines), {"equity": 400.0, "hand_exit_equity": 487.226875}, NO_FEES)


def _c09():
    n = 3
    lines = [*op_ebitda(), *entry("400"), *operating(n)]
    for t in range(1, n + 1):
        lines.append(Line(f"rate_{t}", f"Y{t} rate = max(SONIA, floor) + margin",
                          f"max(sonia_{t}, floor) + margin", f"tranches.Term loan.{t - 1}.interest_rate", "pct"))
    for t in range(1, n + 1):
        lines += income(t, f"debt * rate_{t}")
        lines.append(Line(f"cash_{t}", f"Y{t} closing cash", f"{f'cash_{t - 1}' if t > 1 else '0'} + fcf_{t}"))
    lines += exit_returns(n, f"debt - cash_{n}")
    return Case(
        "09_sonia_floor", "Floating SONIA loan with a floor (sterling)",
        ("floating with floor", "non-USD"),
        "A sterling loan of 400 at SONIA + 4% with a 1% floor on SONIA, SONIA at 0.5%, 1% and 3% "
        "over the three years. The floor applies to the reference before the margin: 5%, 5% "
        "and 7%, so interest 20, 20, 28. Not swept, due after the exit; cash 45, 90, 129.",
        given(sonia_1=0.005, sonia_2=0.01, sonia_3=0.03, floor=0.01, margin=0.04),
        inputs(currency="GBP", tranches=[dict(
            name="Term loan", kind="unitranche", amount=400, floating=True, reference_rate="SONIA",
            reference_path=[0.5, 1.0, 3.0], margin=4.0, floor=1.0, sweep=False, maturity_years=7)]),
        tuple(lines), {"equity": 400.0, "exit_equity": 529.0}, NO_FEES)


def _c10():
    n = 3
    lines = [*op_ebitda(), *entry("400"), *operating(n), Line("bal_0", "PIK notes at close", "debt")]
    for t in range(1, n + 1):
        lines += [Line(f"pik_{t}", f"Y{t} coupon, all of it added to the principal", f"bal_{t - 1} * r",
                       f"tranches.PIK notes.{t - 1}.pik_interest")]
        lines += income(t, f"pik_{t}", pik=f"pik_{t}")
        lines += [
            Line(f"bal_{t}", f"Y{t} PIK notes closing balance", f"bal_{t - 1} + pik_{t}",
                 f"tranches.PIK notes.{t - 1}.ending_balance"),
            Line(f"cash_{t}", f"Y{t} closing cash", f"{f'cash_{t - 1}' if t > 1 else '0'} + fcf_{t}",
                 f"debt_schedule.cash_balance.{t - 1}"),
        ]
    lines += exit_returns(n, f"bal_{n} - cash_{n}")
    return Case(
        "10_pik", "PIK note", ("PIK",),
        "400 of notes at 10%, the whole coupon paid in kind. It is still an expense (40, 44, "
        "48.4), so it lowers tax, but no cash leaves: free cash flow adds it back, 70, 71, 72.1. "
        "The notes compound to 532.4 while 213.1 of cash builds up beside them.",
        given(r=0.10),
        inputs(tranches=[notes(400, 10, kind="pik_notes", name="PIK notes", pik_share=100)]),
        tuple(lines), {"equity": 400.0, "exit_equity": 480.7}, NO_FEES)


def _c11():
    n = 3
    lines = [*op_ebitda(), *entry("400"), *operating(n),
             Line("rate", "Rate = max(reference, floor) + margin", "max(ref, floor) + margin"),
             Line("bal_0", "Unitranche at close", "debt")]
    for t in range(1, n + 1):
        prev_cash = f"cash_{t - 1}" if t > 1 else "0"
        lines += income(t, f"bal_{t - 1} * rate")
        lines += [
            Line(f"avail_{t}", f"Y{t} cash available = cash carried in + free cash flow",
                 f"{prev_cash} + fcf_{t}", f"debt_schedule.available_for_sweep.{t - 1}"),
            Line(f"sweep_{t}", f"Y{t} sweep = half the cash available", f"min(avail_{t} * share, bal_{t - 1})",
                 f"tranches.Unitranche.{t - 1}.cash_sweep"),
            Line(f"bal_{t}", f"Y{t} closing balance", f"bal_{t - 1} - sweep_{t}",
                 f"tranches.Unitranche.{t - 1}.ending_balance"),
            Line(f"cash_{t}", f"Y{t} closing cash = the half not swept", f"avail_{t} - sweep_{t}",
                 f"debt_schedule.cash_balance.{t - 1}"),
        ]
    lines += exit_returns(n, f"bal_{n} - cash_{n}")
    return Case(
        "11_unitranche_half_sweep", "Unitranche taking half the sweep", ("unitranche",),
        "A 400 unitranche at a 5% reference (floor 0) + 5% margin, taking 50% of the cash "
        "available each year. The share is of all the cash available, including what was kept "
        "the year before: year 1 sweeps 15 of 30 and keeps 15; year 2 has 15 + 31.125 and sweeps "
        "23.06; year 3 sweeps 27.96 of 55.92.",
        given(ref=0.05, floor=0.0, margin=0.05, share=0.5),
        inputs(tranches=[dict(name="Unitranche", kind="unitranche", amount=400, floating=True,
                              reference_level=5.0, margin=5.0, sweep=True, sweep_share=50.0, maturity_years=7)]),
        tuple(lines), {"equity": 400.0, "exit_equity": 493.9796875}, NO_FEES)


def _c12():
    n = 3
    lines = [*op_ebitda(), *entry("a_0 + b_0"), *operating(n)]
    lines = [lines[0], Line("a_0", "Term loan B at close", "50"), Line("b_0", "Second lien at close", "350"),
             *lines[1:]]
    for t in range(1, n + 1):
        lines += income(t, f"a_{t - 1} * ra + b_{t - 1} * rb")
        lines += [
            Line(f"a_sweep_{t}", f"Y{t} term loan B takes the cash first", f"min(fcf_{t}, a_{t - 1})",
                 f"tranches.Term loan B.{t - 1}.cash_sweep"),
            Line(f"b_sweep_{t}", f"Y{t} second lien takes what is left", f"min(fcf_{t} - a_sweep_{t}, b_{t - 1})",
                 f"tranches.Second lien.{t - 1}.cash_sweep"),
            Line(f"a_{t}", f"Y{t} term loan B closing", f"a_{t - 1} - a_sweep_{t}",
                 f"tranches.Term loan B.{t - 1}.ending_balance"),
            Line(f"b_{t}", f"Y{t} second lien closing", f"b_{t - 1} - b_sweep_{t}",
                 f"tranches.Second lien.{t - 1}.ending_balance"),
        ]
    lines += exit_returns(n, f"a_{n} + b_{n}")
    return Case(
        "12_two_tranche_waterfall", "Two tranches in sweep order", ("single tranche",),
        "A 50 term loan B at 8% swept first and a 350 second lien at 10% swept second. Year 1's "
        "30.75 all goes to the term loan (19.25 left); year 2's 32.595 repays the last 19.25 and "
        "sends 13.345 to the second lien; year 3's 34.75 all goes to the second lien.",
        given(ra=0.08, rb=0.10),
        inputs(tranches=[
            dict(name="Term loan B", kind="institutional_term_loan", amount=50, floating=False, fixed_rate=8,
                 amort_pct=0, sweep=True, maturity_years=7),
            dict(name="Second lien", kind="second_lien", amount=350, floating=False, fixed_rate=10,
                 sweep=True, maturity_years=8)]),
        tuple(lines), {"equity": 400.0, "exit_equity": 498.095875}, NO_FEES)


def _c13():
    n = 2
    lines = [*op_ebitda(), *entry("400"), *operating(n),
             Line("tl_0", "Term loan at close", "debt"), Line("rv_0", "Revolver drawn at close", "0")]
    for t in range(1, n + 1):
        lines += [
            Line(f"fee_{t}", f"Y{t} commitment fee on the undrawn revolver", f"(commitment - rv_{t - 1}) * fee",
                 f"tranches.Revolver.{t - 1}.commitment_fee"),
        ]
        lines += income(t, f"tl_{t - 1} * r + rv_{t - 1} * rv_rate + fee_{t}")
        lines += [
            Line(f"tl_{t}", f"Y{t} term loan after its 15% repayment", f"tl_{t - 1} - debt * amort",
                 f"tranches.Term loan.{t - 1}.ending_balance"),
            Line(f"draw_{t}", f"Y{t} revolver draw = repayment the cash flow can't cover",
                 f"min(max(debt * amort - fcf_{t}, 0), commitment - rv_{t - 1})",
                 f"tranches.Revolver.{t - 1}.redrawn"),
            Line(f"rv_{t}", f"Y{t} revolver closing balance", f"rv_{t - 1} + draw_{t}",
                 f"tranches.Revolver.{t - 1}.ending_balance"),
        ]
    lines += exit_returns(n, f"tl_{n} + rv_{n}")
    return Case(
        "13_revolver", "Revolver funding a repayment", ("single tranche",),
        "A 400 term loan at 10% repaying 60 a year, beside an undrawn 100 revolver (5% reference "
        "+ 3% margin, 1% a year on what is undrawn). Year 1 makes 29.25 of cash against 60 due, "
        "so the revolver draws 30.75; year 2 pays 8% on that, a smaller fee on the 69.25 still "
        "undrawn, and draws another 27.86. Held two years. Only the drawn part is a source of "
        "funds, so the cheque is 400.",
        given(r=0.10, amort=0.15, commitment=100.0, fee=0.01, rv_rate=0.08, hold=2),
        inputs(hold=2, tranches=[
            dict(name="Term loan", kind="amortising_term_loan", amount=400, floating=False, fixed_rate=10,
                 amort_pct=15, sweep=False, maturity_years=6),
            dict(name="Revolver", kind="revolver", amount=100, drawn_pct=0, floating=True, reference_level=5.0,
                 margin=3.0, commitment_fee_pct=1.0, sweep=True, allow_redraw=True, maturity_years=6)]),
        tuple(lines), {"equity": 400.0, "exit_equity": 461.385625}, NO_FEES)


def _c14():
    n = 3
    tax = "max(ebit_{t} - ded_{t}, 0) * tax"
    lines = [*op_ebitda(), *entry("400"), *operating(n)]
    for t in range(1, n + 1):
        carried = f"carried_{t - 1}" if t > 1 else "0"
        lines += [
            Line(f"cap_{t}", f"Y{t} cap = 30% of EBITDA", f"cap_share * ebitda_{t}"),
            Line(f"ded_{t}", f"Y{t} interest deductible = min(interest + carried, cap)",
                 f"min(debt * r + {carried}, cap_{t})", f"tax.interest_deductible.{t - 1}"),
            Line(f"carried_{t}", f"Y{t} interest carried forward", f"debt * r + {carried} - ded_{t}",
                 f"tax.interest_carried.{t - 1}"),
        ]
        lines += income(t, "debt * r", tax=tax)
        lines.append(Line(f"cash_{t}", f"Y{t} closing cash", f"{f'cash_{t - 1}' if t > 1 else '0'} + fcf_{t}"))
    lines += exit_returns(n, f"debt - cash_{n}")
    return Case(
        "14_interest_cap", "Interest capped at 30% of EBITDA", ("interest cap",),
        "Case 02 under a 30%-of-EBITDA interest limit (as in Germany, the UK or the US). Interest "
        "40 against a cap of 30: 30 is deductible and 10 a year carries forward, never used "
        "because the cap never has room. Tax (80 - 30) x 25% = 12.5 instead of 10, so net income "
        "27.5.",
        given(r=0.10, cap_share=0.30),
        inputs(tranches=[notes(400, 10)], tax_interest_limit="ebitda_share", tax_interest_limit_pct=30.0),
        tuple(lines), {"equity": 400.0, "exit_equity": 482.5}, NO_FEES)


def _c15():
    n = 3
    tax = "max(ebit_{t} - ded_{t}, 0) * tax"
    lines = [*op_ebitda(), *entry("400"), *operating(n), Line("bal_0", "Term loan at close", "debt")]
    for t in range(1, n + 1):
        carried = f"carried_{t - 1}" if t > 1 else "0"
        prev = f"bal_{t - 1}"
        lines += [
            Line(f"ded_{t}", f"Y{t} interest deductible = min(interest + carried, the fixed cap)",
                 f"min({prev} * r + {carried}, cap)", f"tax.interest_deductible.{t - 1}"),
            Line(f"carried_{t}", f"Y{t} interest carried forward", f"{prev} * r + {carried} - ded_{t}",
                 f"tax.interest_carried.{t - 1}"),
        ]
        lines += income(t, f"{prev} * r", tax=tax)
        lines += [
            Line(f"bal_{t}", f"Y{t} closing balance after the sweep", f"{prev} - fcf_{t}",
                 f"tranches.Term loan.{t - 1}.ending_balance"),
        ]
    lines += exit_returns(n, f"bal_{n}")
    return Case(
        "15_interest_cap_used", "Fixed interest cap, carried interest used later", ("interest cap",),
        "A fixed cap of 38 a year on a swept 400 loan at 10%. Year 1: 40 of interest, 38 "
        "allowed, 2 carried. Year 2: 37.05 + 2 claimed, 38 allowed, 1.05 carried. Year 3: "
        "interest has fallen to 33.805, so the 1.05 fits under the cap and is used up.",
        given(r=0.10, cap=38.0),
        inputs(tranches=[dict(name="Term loan", kind="institutional_term_loan", amount=400, floating=False,
                              fixed_rate=10, amort_pct=0, sweep=True, maturity_years=7)],
               tax_interest_limit="fixed", tax_interest_limit_amount=38.0),
        tuple(lines), {"equity": 400.0, "exit_equity": 496.85875}, NO_FEES)


def _loss_lines(n: int, limit: str) -> list:
    """Losses carried forward (methodology 7): a profit absorbs losses up to
    ``limit``, a loss adds to them."""
    lines = []
    for t in range(1, n + 1):
        prev = f"loss_{t - 1}" if t > 1 else "0"
        lines += [
            Line(f"profit_{t}", f"Y{t} profit for tax = EBIT - interest", f"ebit_{t} - debt * r"),
            Line(f"limit_{t}", f"Y{t} most this year's profit may absorb", limit.format(t=t)),
            Line(f"used_{t}", f"Y{t} losses used", f"min({prev}, limit_{t})", f"tax.losses_used.{t - 1}"),
            Line(f"loss_{t}", f"Y{t} losses carried forward", f"{prev} - used_{t} + max(0 - profit_{t}, 0)",
                 f"tax.losses_carried.{t - 1}"),
            Line(f"taxable_{t}", f"Y{t} taxable income", f"max(profit_{t}, 0) - used_{t}",
                 f"tax.taxable_income.{t - 1}"),
        ]
        lines += income(t, "debt * r", tax="taxable_{t} * tax")
        lines.append(Line(f"cash_{t}", f"Y{t} closing cash", f"{f'cash_{t - 1}' if t > 1 else '0'} + fcf_{t}",
                          f"debt_schedule.cash_balance.{t - 1}"))
    return lines


def _c16():
    n = 3
    lines = [*op_ebitda(), *entry("750"), *operating(n), *_loss_lines(n, "max(profit_{t}, 0)")]
    lines += exit_returns(n, f"debt - cash_{n}")
    return Case(
        "16_tax_losses", "Tax loss carried forward", ("tax losses",),
        "Growing 10% a year with capex at 3%, carrying 750 of notes at 12% (90 a year). Year 1 "
        "EBIT 88 is a loss of 2, carried forward. Year 2's profit of 6.8 absorbs it, so tax is "
        "25% of 4.8 = 1.2 instead of 1.7. Year 3 pays in full on 16.48.",
        given(r=0.12, g=0.10, capex=0.03),
        inputs(growth=10.0, capex=3.0, tranches=[notes(750, 12)], tax_loss_carryforward=True),
        tuple(lines), {"equity": 50.0, "exit_equity": 359.888}, NO_FEES)


def _c17():
    n = 3
    limit = "min(max(profit_{t}, 0), allowance) + share * max(max(profit_{t}, 0) - allowance, 0)"
    lines = [*op_ebitda(), *entry("750"), *operating(n), *_loss_lines(n, limit)]
    lines += exit_returns(n, f"debt - cash_{n}")
    return Case(
        "17_loss_limit", "Loss relief limited to an allowance plus 60%", ("tax losses",),
        "Germany's shape of rule: losses offset profit in full up to an allowance (1 here) and "
        "60% of the profit above it. Growing 20% with 750 at 14%: a loss of 9 in year 1; year 2's "
        "profit of 10.2 may absorb 1 + 60% x 9.2 = 6.52, so 2.48 carries on into year 3, which "
        "absorbs it all.",
        given(r=0.14, g=0.20, capex=0.03, allowance=1.0, share=0.60),
        inputs(growth=20.0, capex=3.0, tranches=[notes(750, 14)], tax_loss_carryforward=True,
               tax_loss_limit_amount=1.0, tax_loss_limit_pct=60.0),
        tuple(lines), {"equity": 50.0, "exit_equity": 693.174}, NO_FEES)


def _c18():
    n = 3
    lines = [*op_ebitda(), *entry("400"), *operating(n)]
    for t in range(1, n + 1):
        lines += [
            Line(f"regular_{t}", f"Y{t} tax at the 10% rate", f"max(ebit_{t} - debt * r, 0) * tax",
                 f"tax.regular_tax.{t - 1}"),
            Line(f"topup_{t}", f"Y{t} top-up to the 15% minimum on book profit",
                 f"max(max(ebit_{t} - debt * r, 0) * minimum - regular_{t}, 0)", f"tax.minimum_tax_topup.{t - 1}"),
        ]
        lines += income(t, "debt * r", tax="regular_{t} + topup_{t}")
        lines.append(Line(f"cash_{t}", f"Y{t} closing cash", f"{f'cash_{t - 1}' if t > 1 else '0'} + fcf_{t}"))
    lines += exit_returns(n, f"debt - cash_{n}")
    return Case(
        "18_minimum_tax", "Minimum tax above a low rate", ("tax losses",),
        "Case 02 in a 10% tax system with a 15% minimum tax on book profit (the shape of the "
        "OECD's global minimum). Profit before tax is 40: 4 at the rate, topped up by 2 to 6. "
        "Net income 34 a year.",
        given(r=0.10, tax=0.10, minimum=0.15),
        inputs(tax=10.0, tranches=[notes(400, 10)], tax_minimum_pct=15.0),
        tuple(lines), {"equity": 400.0, "exit_equity": 502.0}, NO_FEES)


def _ifrs(view: str):
    n = 3
    post = view == "post"
    addback, liability = ("lease", "liability") if post else ("0", "0")
    lines = [*op_ebitda("ebitda - lease"),
             *entry("400", addback=addback, liability=liability),
             *operating(n, ebitda="op_ebitda"), *bullet("Notes", n)]
    lines += exit_returns(n, f"bal_{n} - cash_{n} + {liability}", addback=addback)
    if post:
        lines += [
            Line("lease_ev", "Leases block: entry EV on EBITDA before lease costs", "ev", "leases.entry_ev"),
            Line("lease_nd", "Leases block: net debt at entry = debt + lease liability", "debt + liability",
                 "leases.net_debt_at_entry"),
        ]
    return lines


def _c19():
    return Case(
        "19_ifrs16_post", "IFRS 16: leases as debt", ("IFRS 16",),
        "An IFRS reporter with EBITDA of 110 before 10 of lease costs and a lease liability of 60. "
        "The business underneath is case 02's (operating EBITDA 100 after rent, which is cash "
        "either way). Valued post-IFRS 16, the multiple applies to 110 (EV 880) and the 60 counts "
        "as debt, at entry and at exit: cheque 880 - 400 - 60 = 420, exit equity 880 - 310 - 60 = 510.",
        given(r=0.10, ebitda=110.0, lease=10.0, liability=60.0),
        inputs(ebitda=110.0, accounting_standard="ifrs", lease_cost=10.0, lease_liability=60.0,
               tranches=[notes(400, 10)]),
        tuple(_ifrs("post")), {"equity": 420.0, "exit_equity": 510.0}, NO_FEES)


def _c20():
    return Case(
        "20_ifrs16_pre", "IFRS 16 reporter valued pre-IFRS 16", ("IFRS 16",),
        "The same company as case 19 valued the old way: the multiple applies to EBITDA after "
        "lease costs (100) and the lease liability is not debt. It is case 02 exactly: cheque "
        "400, exit equity 490. The difference between 19 and 20 is the view, not the business.",
        given(r=0.10, ebitda=110.0, lease=10.0, liability=60.0),
        inputs(ebitda=110.0, accounting_standard="ifrs", lease_view="pre_ifrs16", lease_cost=10.0,
               lease_liability=60.0, tranches=[notes(400, 10)]),
        tuple(_ifrs("pre")), {"equity": 400.0, "exit_equity": 490.0}, NO_FEES)


def _c21():
    n = 3
    lines = [*op_ebitda(), *entry("400000"), *operating(n),
             Line("rate", "Rate = EURIBOR + margin", "euribor + margin"), *bullet("Term loan", n, rate="rate")]
    lines += exit_returns(n, f"bal_{n} - cash_{n}")
    return Case(
        "21_euro_thousands", "Euro deal counted in thousands", ("non-USD",),
        "Case 02 as a euro deal entered in thousands: EBITDA 100,000 thousand, a 400,000 loan at "
        "3% EURIBOR + 7%. Every figure comes back in thousands, exactly 1,000 times case 02's "
        "millions: exit equity 490,000.",
        given(ebitda=100000.0, euribor=0.03, margin=0.07),
        inputs(ebitda=100000.0, currency="EUR", unit="thousands", tranches=[dict(
            name="Term loan", kind="unitranche", amount=400000, floating=True, reference_rate="EURIBOR",
            reference_level=3.0, margin=7.0, sweep=False, maturity_years=7)]),
        tuple(lines), {"equity": 400000.0, "exit_equity": 490000.0}, NO_FEES)


def _c22():
    n = 3
    lines = [*op_ebitda(), *entry("400"), *operating(n),
             Line("rate", "Rate = max(TONA, floor) + margin: the floor lifts -0.1% to 0", "max(tona, floor) + margin",
                  "tranches.Term loan.0.interest_rate", "pct"),
             *bullet("Term loan", n, rate="rate")]
    lines += exit_returns(n, f"bal_{n} - cash_{n}")
    return Case(
        "22_march_year_end_yen", "Yen deal in billions, March year-end, negative reference rate",
        ("March year-end", "non-USD", "floating with floor"),
        "A Japanese deal in billions of yen with a fiscal year ending in March, first projected "
        "year FY2026/27. Fiscal years are labels: year 1 is still the first full year after "
        "close, so the figures are what a December year-end gives. The loan pays TONA (-0.1%) "
        "floored at 0, plus 4%: 4%, so 16 of interest and 48 of cash a year.",
        given(tona=-0.001, floor=0.0, margin=0.04),
        inputs(currency="JPY", unit="billions", fiscal_year_end_month=3, first_fiscal_year=2027,
               tranches=[dict(name="Term loan", kind="unitranche", amount=400, floating=True,
                              reference_rate="TONA", reference_level=-0.1, floor=0.0, margin=4.0,
                              sweep=False, maturity_years=7)]),
        tuple(lines), {"equity": 400.0, "exit_equity": 544.0}, NO_FEES)


def _c23():
    n = 3
    lines = [*op_ebitda(), *entry("400"),
             Line("k", "Working capital / revenue = (AR days + (1 - GM) x (inventory - AP days)) / 365",
                  "(ar + (1 - gm) * (inv - ap)) / 365"),
             Line("nwc_share", "Change in working capital / revenue = k x g / (1 + g)", "k * g / (1 + g)"),
             *operating(n, nwc="nwc_share"), *bullet("Notes", n)]
    lines += exit_returns(n, f"bal_{n} - cash_{n}")
    return Case(
        "23_growth_working_capital_days", "Growth with working capital in days", ("zero growth",),
        "Case 02 growing 10% a year, with working capital from 36.5 receivable days, 73 inventory "
        "and 36.5 payable days, at a 60% gross margin (opex 40%, so the same EBITDA). That is 14% "
        "of revenue (36.5 + 40% x 36.5, over 365), so each year ties up 14% of the revenue "
        "added: 5.6, 6.16, 6.776. Exit EBITDA 133.1.",
        given(r=0.10, g=0.10, gm=0.60, opex=0.40, ar=36.5, inv=73.0, ap=36.5),
        inputs(growth=10.0, gross_margin=60.0, opex=40.0, wsp_mode=True, ar_days=36.5, inv_days=73.0, ap_days=36.5, tranches=[notes(400, 10)]),
        tuple(lines), {"equity": 400.0, "exit_equity": 774.724}, NO_FEES)


def _c24():
    n = 3
    lines = [*op_ebitda(), *entry("400", mincash="mincash"), *operating(n),
             Line("inc", "Interest income = minimum cash x 0.5%", "mincash * 0.005",
                  "operating_model.interest_income.0")]
    lines += bullet("Notes", n, other_income="inc")
    lines += [Line("cash_total", "Closing cash at exit = minimum + what built up", f"mincash + cash_{n}",
                   f"debt_schedule.cash_balance.{n - 1}")]
    lines += exit_returns(n, f"bal_{n} - cash_total")
    return Case(
        "24_minimum_cash", "Minimum cash funded at close", ("single tranche",),
        "Case 02 keeping 40 of cash in the business from day one. The sponsor funds it (cheque "
        "440, finding 1), it earns the model's 0.5% (0.2 a year) and it is still there at exit, "
        "where it reduces net debt.",
        given(r=0.10, mincash=40.0), inputs(mincash=40.0, tranches=[notes(400, 10)]), tuple(lines),
        {"equity": 440.0, "exit_equity": 530.45}, NO_FEES)


def _c25():
    n = 1
    lines = [*op_ebitda(), *entry("400"), *operating(n), *bullet("Notes", n)]
    lines += exit_returns(n, f"bal_{n} - cash_{n}")
    return Case(
        "25_one_year_hold", "Held one year", ("single tranche",),
        "Case 02 sold after one year: 30 of cash, net debt 370, exit equity 430 for 400. With "
        "one year the IRR is simply MOIC - 1 = 7.5%.",
        given(r=0.10, hold=1), inputs(hold=1, tranches=[notes(400, 10)]), tuple(lines),
        {"equity": 400.0, "exit_equity": 430.0}, NO_FEES)


def _c26():
    n = 3
    lines = [*op_ebitda(), Line("a_0", "Term loan A at close", "200"), Line("b_0", "Term loan B at close", "200"),
             *entry("a_0 + b_0"), *operating(n)]
    for t in range(1, n + 1):
        lines += income(t, f"a_{t - 1} * ra + b_{t - 1} * rb")
        lines += [
            Line(f"a_sweep_{t}", f"Y{t} term loan A takes half the cash available",
                 f"min(fcf_{t} * share, a_{t - 1})", f"tranches.Term loan A.{t - 1}.cash_sweep"),
            Line(f"b_sweep_{t}", f"Y{t} term loan B takes half the cash available, not half of what A left",
                 f"min(fcf_{t} * share, fcf_{t} - a_sweep_{t}, b_{t - 1})", f"tranches.Term loan B.{t - 1}.cash_sweep"),
            Line(f"a_{t}", f"Y{t} term loan A closing", f"a_{t - 1} - a_sweep_{t}",
                 f"tranches.Term loan A.{t - 1}.ending_balance"),
            Line(f"b_{t}", f"Y{t} term loan B closing", f"b_{t - 1} - b_sweep_{t}",
                 f"tranches.Term loan B.{t - 1}.ending_balance"),
            Line(f"cash_{t}", f"Y{t} closing cash: the two halves take it all", f"fcf_{t} - a_sweep_{t} - b_sweep_{t}",
                 f"debt_schedule.cash_balance.{t - 1}"),
        ]
    lines += exit_returns(n, f"a_{n} + b_{n}")
    return Case(
        "26_shared_sweep", "Two facilities sharing the sweep", ("single tranche",),
        "Two 200 term loans, at 8% and 10%, each entitled to 50% of the cash available. The "
        "share is of the cash the business had, not of what the facility above left: year 1's "
        "33 goes 16.5 and 16.5, not 16.5 and 8.25, and nothing is left over.",
        given(ra=0.08, rb=0.10, share=0.5),
        inputs(tranches=[
            dict(name="Term loan A", kind="amortising_term_loan", amount=200, floating=False, fixed_rate=8,
                 amort_pct=0, sweep=True, sweep_share=50.0, maturity_years=6),
            dict(name="Term loan B", kind="institutional_term_loan", amount=200, floating=False, fixed_rate=10,
                 amort_pct=0, sweep=True, sweep_share=50.0, maturity_years=7)]),
        tuple(lines), {"equity": 400.0, "exit_equity": 505.83285625}, NO_FEES)


CASES = tuple(f() for f in (
    _c01, _c02, _c03, _c04, _c05, _c06, _c07, _c08, _c09, _c10, _c11, _c12, _c13,
    _c14, _c15, _c16, _c17, _c18, _c19, _c20, _c21, _c22, _c23, _c24, _c25, _c26,
))

# What PLAN.md 3.4 asks the cases to cover; the test fails if one goes missing
REQUIRED_COVERAGE = (
    "no leverage", "single tranche", "fees only", "zero growth", "floating with floor", "PIK",
    "unitranche", "interest cap", "tax losses", "IFRS 16", "non-USD", "March year-end",
)
