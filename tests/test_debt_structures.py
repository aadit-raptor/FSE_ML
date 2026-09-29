"""Global debt structures and interest rates (PLAN.md 2.4).

Hand-checked cases, worked out on paper and written out in the docstring of
each test, so a reader can check the model rather than the model checking
itself. The arithmetic is chosen to land on exact decimal values: no test
depends on which way a tie rounds.

Three shapes the old two-tranche model could not express:
  * a floating loan priced off a reference rate with a floor (`test_floating_*`),
  * a note whose interest accrues to principal (`test_pik_*`),
  * a single unitranche facility taking a share of the cash sweep
    (`test_unitranche_*`).

Plus the rule that guards everything else: a deal with no tranche list, and a
deal whose tranche list spells out today's senior + mezzanine structure, must
both give exactly today's numbers.
"""
import dataclasses

import pytest

from core.config import resolve_config
from core.deal import DealInputs, build_lbo_params, run_deal
from core.debt import REFERENCE_RATES, TRANCHE_TYPES, TrancheSpec, equivalent_tranches
from lbo_engine.model import run_lbo


def cfg():
    return resolve_config()


def run(tranches, **overrides):
    """The default deal financed by ``tranches`` only."""
    d = DealInputs(tranches=list(tranches), **overrides)
    return run_deal(d, cfg())


def rows(result, name):
    return result.debt_schedule.schedule[name]


# ---------------------------------------------------------------------------
# Case 1 — a floating SONIA term loan with a floor
# ---------------------------------------------------------------------------
# GBP 400M term loan, SONIA + 4.00% margin, 2.50% floor on SONIA, 10% of the
# original principal amortising a year, no cash sweep, maturity beyond the
# five-year hold.
#
#   SONIA path (%)     3.0   2.0   1.0   4.0   5.0
#   floored at 2.50    3.0   2.5   2.5   4.0   5.0
#   + 4.00 margin      7.0   6.5   6.5   8.0   9.0   <- all-in rate
#
#   amortisation 10% x 400 = 40.0 a year
#   opening balance    400   360   320   280   240
#   interest           28.0  23.4  20.8  22.4  21.6
#     400 x 7.0%  = 28.0      360 x 6.5% = 23.4      320 x 6.5% = 20.8
#     280 x 8.0%  = 22.4      240 x 9.0% = 21.6
# ---------------------------------------------------------------------------
SONIA_LOAN = TrancheSpec(
    name="GBP term loan",
    kind="amortising_term_loan",
    amount=400.0,
    floating=True,
    reference_rate="SONIA",
    reference_path=[3.0, 2.0, 1.0, 4.0, 5.0],
    floor=2.5,
    margin=4.0,
    amort_pct=10.0,
    sweep=False,
    maturity_years=7,
)


def test_floating_loan_prices_off_the_reference_path_with_a_floor():
    r = run([SONIA_LOAN])
    assert [t.interest_rate for t in rows(r, "GBP term loan")] == [0.07, 0.065, 0.065, 0.08, 0.09]


def test_floating_loan_balances_and_interest_match_the_hand_check():
    r = run([SONIA_LOAN])
    sched = rows(r, "GBP term loan")
    assert [t.beginning_balance for t in sched] == [400.0, 360.0, 320.0, 280.0, 240.0]
    assert [t.mandatory_repayment for t in sched] == [40.0] * 5
    assert [t.interest_expense for t in sched] == [28.0, 23.4, 20.8, 22.4, 21.6]
    # The only tranche, so it is the whole interest line
    assert r.debt_schedule.total_interest_expense == [28.0, 23.4, 20.8, 22.4, 21.6]


def test_the_floor_is_what_holds_years_2_and_3_up():
    """Without the floor SONIA would be 2.0% and 1.0%, not 2.5%."""
    floorless = dataclasses.replace(SONIA_LOAN, floor=0.0)
    r = run([floorless])
    sched = rows(r, "GBP term loan")
    assert [t.interest_rate for t in sched] == [0.07, 0.06, 0.05, 0.08, 0.09]
    # 360 x 6.0% = 21.6 and 320 x 5.0% = 16.0
    assert [t.interest_expense for t in sched][1:3] == [21.6, 16.0]


def test_a_rate_path_shorter_than_the_hold_holds_its_last_year():
    """The exit-sensitivity grid reruns the deal at holds of 3 to 7 years, so a
    five-year path has to answer for years 6 and 7: it stays at its last year
    (5.0% + 4.00 = 9.0%) rather than falling back to zero."""
    r = run([SONIA_LOAN], hold=7)
    assert [t.interest_rate for t in rows(r, "GBP term loan")] == [0.07, 0.065, 0.065, 0.08, 0.09, 0.09, 0.09]


def test_a_fixed_tranche_ignores_the_reference_path():
    fixed = dataclasses.replace(SONIA_LOAN, floating=False, fixed_rate=7.5)
    r = run([fixed])
    assert [t.interest_rate for t in rows(r, "GBP term loan")] == [0.075] * 5
    # 400 x 7.5% = 30.0, 360 x 7.5% = 27.0 ...
    assert [t.interest_expense for t in rows(r, "GBP term loan")] == [30.0, 27.0, 24.0, 21.0, 18.0]


# ---------------------------------------------------------------------------
# Case 2 — a PIK note
# ---------------------------------------------------------------------------
# 200M PIK note at 10.0% fixed, the whole coupon accruing to principal, bullet
# at maturity (year 8, beyond the hold), no sweep.
#
#   year               1       2       3        4         5
#   opening          200.0   220.0   242.0    266.2     292.82
#   interest @ 10%    20.0    22.0    24.2     26.62      29.282  -> shown 29.28
#   accrued to        220.0   242.0   266.2    292.82    322.102  -> shown 322.1
#
# None of it is paid in cash, so all of it comes back as a non-cash add-back in
# the cash flow: levered FCF is higher by exactly the interest each year, and
# the income statement still carries the full expense.
# ---------------------------------------------------------------------------
PIK_NOTE = TrancheSpec(
    name="PIK notes",
    kind="pik_notes",
    amount=200.0,
    fixed_rate=10.0,
    pik_share=100.0,
    sweep=False,
    maturity_years=8,
)


def test_pik_interest_accrues_to_principal():
    r = run([PIK_NOTE])
    sched = rows(r, "PIK notes")
    assert [t.beginning_balance for t in sched] == [200.0, 220.0, 242.0, 266.2, 292.82]
    assert [t.interest_expense for t in sched] == [20.0, 22.0, 24.2, 26.62, 29.28]
    assert [t.ending_balance for t in sched] == [220.0, 242.0, 266.2, 292.82, 322.1]
    assert [t.pik_interest for t in sched] == [20.0, 22.0, 24.2, 26.62, 29.28]


def test_pik_interest_is_added_back_in_the_cash_flow_but_stays_an_expense():
    r = run([PIK_NOTE])
    cf, op = r.cash_flow, r.operating_model
    # Still an expense in the P&L: the income statement carries the full coupon
    assert op.interest_expense == [20.0, 22.0, 24.2, 26.62, 29.28]
    # And none of it is paid, so all of it is added back
    assert cf.non_cash_interest == [20.0, 22.0, 24.2, 26.62, 29.28]
    for t in range(5):
        expected = (cf.net_income[t] + cf.da[t] + cf.non_cash_interest[t]
                    - cf.capex[t] - cf.delta_nwc[t] - cf.mandatory_repay[t])
        assert cf.levered_fcf[t] == pytest.approx(expected, abs=0.01)


def test_without_the_pik_share_the_note_neither_accrues_nor_adds_back():
    cash_pay = dataclasses.replace(PIK_NOTE, pik_share=0.0)
    r = run([cash_pay])
    sched = rows(r, "PIK notes")
    assert [t.beginning_balance for t in sched] == [200.0] * 5
    assert [t.interest_expense for t in sched] == [20.0] * 5
    assert r.cash_flow.non_cash_interest == [0.0] * 5


def test_half_the_coupon_in_kind_accrues_half():
    """40% cash, 60% in kind: year 1 interest 20.0, of which 12.0 accrues."""
    part = dataclasses.replace(PIK_NOTE, pik_share=60.0)
    r = run([part])
    sched = rows(r, "PIK notes")
    assert sched[0].interest_expense == 20.0
    assert sched[0].pik_interest == 12.0
    assert sched[0].ending_balance == 212.0
    # 212 x 10% = 21.2, of which 60% = 12.72 accrues -> 224.72
    assert sched[1].interest_expense == 21.2
    assert sched[1].pik_interest == 12.72
    assert sched[1].ending_balance == 224.72


# ---------------------------------------------------------------------------
# Case 3 — a unitranche
# ---------------------------------------------------------------------------
# One facility instead of senior + mezzanine: 4.0x the 100M entry EBITDA, so
# 400M, SOFR + 6.00% with a 1.00% floor, 1% of the original principal a year,
# and a 60% excess cash flow sweep (the other 40% stays with the business).
#
#   SOFR path (%)      4.5   4.0   3.5   3.0   3.0
#   floored at 1.00    unchanged (all above the floor)
#   + 6.00 margin     10.5  10.0   9.5   9.0   9.0
#
#   amortisation 1% x 400 = 4.0 a year
#   year 1: opening 400.0, interest 400 x 10.5% = 42.0
# ---------------------------------------------------------------------------
UNITRANCHE = TrancheSpec(
    name="Unitranche",
    kind="unitranche",
    amount=400.0,
    floating=True,
    reference_rate="SOFR",
    reference_path=[4.5, 4.0, 3.5, 3.0, 3.0],
    floor=1.0,
    margin=6.0,
    amort_pct=1.0,
    sweep=True,
    sweep_share=60.0,
    maturity_years=7,
)


def test_unitranche_is_the_whole_structure_at_a_blended_floating_rate():
    r = run([UNITRANCHE])
    assert list(r.capital_structure.tranches and [t.name for t in r.capital_structure.tranches]) == ["Unitranche"]
    assert r.capital_structure.total_debt == 400.0
    sched = rows(r, "Unitranche")
    assert [t.interest_rate for t in sched] == [0.105, 0.10, 0.095, 0.09, 0.09]
    assert sched[0].beginning_balance == 400.0
    assert sched[0].mandatory_repayment == 4.0
    assert sched[0].interest_expense == 42.0


def test_the_sweep_share_splits_excess_cash_between_debt_and_the_balance_sheet():
    """60% of the cash available each year pays down the facility; the rest
    builds on the balance sheet. The old model always swept all of it."""
    r = run([UNITRANCHE], mincash=10.0)
    debt = r.debt_schedule
    sched = rows(r, "Unitranche")
    for t in range(5):
        available = debt.available_for_sweep[t]
        # Only meaningful while the facility is larger than 60% of the cash
        assert available > 0
        assert sched[t].cash_sweep == pytest.approx(round(available * 0.6, 4), abs=0.01)
        # The unswept 40% is retained above the minimum cash floor
        assert debt.cash_balance[t] == pytest.approx(10.0 + available * 0.4, abs=0.01)


def test_a_full_sweep_share_pays_down_faster_than_a_partial_one():
    full = dataclasses.replace(UNITRANCHE, sweep_share=100.0)
    partial = run([UNITRANCHE], mincash=10.0)
    whole = run([full], mincash=10.0)
    assert whole.debt_schedule.total_ending_debt[-1] < partial.debt_schedule.total_ending_debt[-1]
    assert whole.returns.irr > partial.returns.irr


# ---------------------------------------------------------------------------
# A revolver: commitment, drawn amount and a fee on the undrawn part
# ---------------------------------------------------------------------------
def test_a_revolver_charges_a_commitment_fee_on_what_is_undrawn():
    """A 100M commitment drawn 30% at close: 30M of debt, 70M undrawn at a
    0.50% commitment fee = 0.35 a year on top of the interest on the 30M.

      interest 30 x 5.0% = 1.5, plus 70 x 0.50% = 0.35  ->  1.85
    """
    revolver = TrancheSpec(
        name="Revolving credit facility",
        kind="revolver",
        amount=100.0,
        drawn_pct=30.0,
        fixed_rate=5.0,
        commitment_fee_pct=0.5,
        sweep=False,
        maturity_years=6,
    )
    r = run([revolver])
    sched = rows(r, "Revolving credit facility")
    assert sched[0].beginning_balance == 30.0
    assert sched[0].commitment_fee == 0.35
    assert sched[0].interest_expense == 1.85
    # Only the drawn part is a source of funds
    assert r.capital_structure.total_debt == 30.0


# ---------------------------------------------------------------------------
# Nothing changes for a deal that doesn't use tranches
# ---------------------------------------------------------------------------
def test_a_deal_with_no_tranche_list_is_untouched():
    """The default deal's answer, to the cent, with the tranche field present."""
    plain = run_deal(DealInputs(), cfg())
    assert plain.returns.irr == pytest.approx(0.2116, abs=0.0001)
    assert set(plain.debt_schedule.schedule) == {"Senior Term Loan", "Mezzanine"}


def test_spelling_out_todays_structure_as_tranches_gives_the_same_answer():
    """`equivalent_tranches` writes today's senior + mezzanine structure out as
    an explicit tranche list. Running it must give the identical result, or the
    tranche path is not a superset of the old one."""
    d = DealInputs()
    plain = run_deal(d, cfg())
    explicit = run_deal(dataclasses.replace(d, tranches=equivalent_tranches(d, cfg())), cfg())

    assert explicit.returns.irr == plain.returns.irr
    assert explicit.returns.moic == plain.returns.moic
    assert explicit.returns.entry_equity == plain.returns.entry_equity
    assert explicit.debt_schedule.net_debt_at_exit == plain.debt_schedule.net_debt_at_exit
    assert explicit.debt_schedule.total_interest_expense == plain.debt_schedule.total_interest_expense
    assert explicit.cash_flow.levered_fcf == plain.cash_flow.levered_fcf
    for name in plain.debt_schedule.schedule:
        assert ([dataclasses.asdict(r) for r in explicit.debt_schedule.schedule[name]]
                == [dataclasses.asdict(r) for r in plain.debt_schedule.schedule[name]])


@pytest.mark.parametrize("hold", [3, 4, 5, 6, 7])
def test_the_equivalent_structure_matches_at_every_hold(hold):
    """Whatever the hold, the deal's own answer is identical -- returns, debt
    schedule and the sensitivity grid's own column."""
    d = DealInputs(hold=hold)
    plain = run_deal(d, cfg())
    explicit = run_deal(dataclasses.replace(d, tranches=equivalent_tranches(d, cfg())), cfg())
    assert explicit.returns.irr == plain.returns.irr
    assert explicit.debt_schedule.total_ending_debt == plain.debt_schedule.total_ending_debt
    grid = plain.exit_sensitivity
    col = grid["holding_periods"].index(hold)
    assert [row[col] for row in explicit.exit_sensitivity["table"]] == [row[col] for row in grid["table"]]


def test_an_explicit_structure_does_not_stretch_its_maturities_with_the_hold():
    """The sensitivity grid's *other* columns differ between the two paths, on
    purpose.

    The old sizing gives the mezzanine ``maturity_years = hold``, so when the
    grid reruns the deal at three or seven years the bullet moves with it and
    the tranche is always repaid at exit -- out of cash the model never earns
    (CLAUDE.md "Model findings", finding 11). A facility written out
    explicitly matures when its agreement says, whatever the sponsor's hold:
    exiting early leaves it outstanding, and it is deducted as net debt like
    any other borrowing. That is the honest reading, so the two grids agree
    only in the column the deal actually ran.
    """
    d = DealInputs(hold=5)
    plain = run_deal(d, cfg())
    explicit = run_deal(dataclasses.replace(d, tranches=equivalent_tranches(d, cfg())), cfg())
    grid = plain.exit_sensitivity
    three = grid["holding_periods"].index(3)
    # Exiting at three years leaves the mezzanine outstanding, so the explicit
    # structure carries more net debt out and returns less
    assert [row[three] for row in explicit.exit_sensitivity["table"]] != [row[three] for row in grid["table"]]
    assert explicit.exit_sensitivity["table"][0][three] < grid["table"][0][three]


# ---------------------------------------------------------------------------
# Every tranche type runs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", TRANCHE_TYPES)
def test_every_tranche_type_runs_and_carries_debt(kind):
    spec = TrancheSpec(name=kind, kind=kind, amount=300.0, fixed_rate=8.0, margin=5.0,
                       reference_level=3.0, maturity_years=7)
    r = run([spec])
    sched = rows(r, kind)
    assert len(sched) == 5
    assert sched[0].beginning_balance == 300.0
    assert sched[0].interest_expense > 0
    assert r.returns.irr == r.returns.irr  # a number, not NaN


@pytest.mark.parametrize("reference", REFERENCE_RATES)
def test_every_reference_rate_prices_the_same_way(reference):
    """The reference rate is a label on the path, not a calculation: two
    tranches with the same path and margin price identically whichever
    benchmark they name."""
    spec = TrancheSpec(name="Loan", kind="institutional_term_loan", amount=300.0,
                       floating=True, reference_rate=reference, reference_level=3.5,
                       margin=4.0, maturity_years=7)
    r = run([spec])
    # 3.5/100 + 4.0/100 is 0.07500000000000001, not 0.075: the sum of two
    # divided percentages, which is exactly the expression the engine has
    # always used for a rate over a reference. Compare as a number.
    assert [t.interest_rate for t in rows(r, "Loan")] == pytest.approx([0.075] * 5)


# ---------------------------------------------------------------------------
# Several tranches at once, in sweep order
# ---------------------------------------------------------------------------
def test_the_sweep_runs_down_the_priority_order():
    """Two sweeping tranches: the first priority is repaid before the second
    takes anything."""
    first = TrancheSpec(name="Term loan B", kind="institutional_term_loan", amount=120.0,
                        fixed_rate=6.0, sweep=True, sweep_priority=1, maturity_years=7)
    second = TrancheSpec(name="Second lien", kind="second_lien", amount=300.0,
                         fixed_rate=9.0, sweep=True, sweep_priority=2, maturity_years=7)
    r = run([first, second])
    a, b = rows(r, "Term loan B"), rows(r, "Second lien")
    # The junior tranche is untouched until the senior one is gone
    for t in range(5):
        if b[t].cash_sweep > 0:
            assert a[t].ending_balance == 0.0
    assert a[-1].ending_balance == 0.0


def test_a_shareholder_loan_can_sit_behind_everything_and_accrue():
    """A shareholder loan at 8% PIK, junior to a sweeping term loan."""
    senior = TrancheSpec(name="Term loan", kind="amortising_term_loan", amount=300.0,
                         fixed_rate=6.0, amort_pct=5.0, sweep=True, sweep_priority=1,
                         maturity_years=7)
    shareholder = TrancheSpec(name="Shareholder loan", kind="shareholder_loan", amount=100.0,
                              fixed_rate=8.0, pik_share=100.0, sweep=False, maturity_years=10)
    r = run([senior, shareholder])
    sl = rows(r, "Shareholder loan")
    assert [t.beginning_balance for t in sl] == [100.0, 108.0, 116.64, 125.97, 136.05]
    assert r.debt_schedule.total_ending_debt[-1] > 0


# ---------------------------------------------------------------------------
# Money: tranche amounts are in the deal's unit
# ---------------------------------------------------------------------------
def test_tranche_amounts_are_read_in_the_deals_unit():
    """The same deal counted in thousands: every amount is a thousand times
    larger, and after `in_millions` -- the conversion every router does -- the
    engine sees the identical deal."""
    from core.deal import in_millions

    millions = DealInputs(tranches=[SONIA_LOAN])
    thousands = DealInputs(
        unit="thousands", ebitda=100_000.0,
        tranches=[dataclasses.replace(SONIA_LOAN, amount=400_000.0)],
    )
    a = run_deal(*in_millions(millions, cfg()))
    b = run_deal(*in_millions(thousands, cfg()))
    assert a.returns.irr == b.returns.irr
    assert [t.interest_rate for t in rows(b, "GBP term loan")] == [0.07, 0.065, 0.065, 0.08, 0.09]
    # The engine always runs in millions, whatever unit the deal arrived in
    assert rows(b, "GBP term loan")[0].beginning_balance == 400.0


# ---------------------------------------------------------------------------
# Per-tranche upfront fees
# ---------------------------------------------------------------------------
def test_an_upfront_fee_raises_the_equity_cheque_without_buying_value():
    """A 2% upfront fee on a 400M facility is 8M of extra sponsor equity: the
    deal's own financing-fee setting is untouched, so the two are separate
    lines and neither double counts the other."""
    free = run([dataclasses.replace(SONIA_LOAN, upfront_fee_pct=0.0)])
    paid = run([dataclasses.replace(SONIA_LOAN, upfront_fee_pct=2.0)])
    assert paid.returns.entry_equity - free.returns.entry_equity == pytest.approx(8.0, abs=0.01)
    assert paid.returns.irr < free.returns.irr


def test_tranche_fees_are_reported_as_their_own_use_of_funds():
    from core.deal import sources_and_uses_for

    su = sources_and_uses_for(DealInputs(tranches=[dataclasses.replace(SONIA_LOAN, upfront_fee_pct=2.0)]), cfg())
    assert su["tranche_fees"] == pytest.approx(8.0, abs=0.01)
    assert su["tranches"] == [{"name": "GBP term loan", "kind": "amortising_term_loan", "amount": 400.0}]
    assert su["balanced"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_a_tranche_with_an_unknown_type_is_refused():
    with pytest.raises(ValueError, match="kind"):
        TrancheSpec(name="Mystery", kind="not_a_tranche", amount=10.0)


def test_a_tranche_with_an_unknown_reference_rate_is_refused():
    with pytest.raises(ValueError, match="reference_rate"):
        TrancheSpec(name="Loan", kind="unitranche", amount=10.0, floating=True, reference_rate="LIBOR")


def test_build_lbo_params_only_builds_a_structure_when_there_are_tranches():
    assert build_lbo_params(DealInputs(), cfg()).capital_structure is None
    built = build_lbo_params(DealInputs(tranches=[SONIA_LOAN]), cfg()).capital_structure
    assert built is not None and built.total_debt == 400.0
