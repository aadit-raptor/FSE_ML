"""A deal as a workbook of live formulas (PLAN.md 7.3a) and the saved deal's
cells for the Excel add-in (7.3b).

Every check works the downloaded workbook out from its own formulas -- with the
tests' own Excel reader (tests/research_support.py, which shares no code with
the API) and, where it is installed, with LibreOffice -- and compares the answer
with what `POST /api/deal/run` says for the same deal. Nothing here reads a
value the API wrote into a cell: the workbook holds inputs and formulas only.
"""
from __future__ import annotations

import io
import sys

import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook
from sqlalchemy import select

from api.auth import AuthUser, require_user
from api.main import app
from tests.deal_workbook_check import named, row_of, worked_out
from tests.research_support import recalculated_by_libreoffice, soffice_wanted, workbook_values

client = TestClient(app)
SHEET = "Deal model"
# A year's cells read the year before through a dozen rows: deeper than Python's default
sys.setrecursionlimit(max(sys.getrecursionlimit(), 50_000))

# A euro deal in thousands with every debt mechanic at once: a floating term loan whose EURIBOR floor bites from
# year 3, a part-drawn revolver that pays a commitment fee and draws when the senior notes fall due in year 5, PIK
# notes past the hold, a vendor loan on its own schedule, and fixed senior notes; with the interest limit, losses
# carried forward with a limit, a minimum tax and IFRS 16 leases.
EUR_DEAL = {
    "ebitda": 85_000.0, "entry_mult": 9.5, "exit_mult": 10.0, "hold": 6, "growth": 4.0, "gross_margin": 42.0,
    "opex": 22.0, "da": 5.0, "tax": 29.9, "capex": 4.5, "nwc": 0.8, "mincash": 4_000.0,
    "currency": "EUR", "unit": "thousands", "country": "DE",
    "tax_interest_limit": "ebitda_share", "tax_interest_limit_pct": 30.0, "tax_interest_limit_amount": 3_000.0,
    "tax_loss_carryforward": True, "tax_loss_limit_pct": 60.0, "tax_loss_limit_amount": 1_000.0,
    "tax_minimum_pct": 5.0,
    "accounting_standard": "ifrs", "lease_cost": 6_000.0, "lease_liability": 30_000.0,
    "tranches": [
        {"name": "Term loan B", "kind": "institutional_term_loan", "amount": 330_000.0, "floating": True,
         "reference_rate": "EURIBOR", "reference_path": [2.4, 2.1, 1.8, 1.6], "floor": 2.0, "margin": 4.25,
         "maturity_years": 7, "amort_pct": 1.0, "upfront_fee_pct": 2.0, "sweep": True},
        {"name": "Revolver", "kind": "revolver", "amount": 60_000.0, "drawn_pct": 20.0, "floating": True,
         "reference_rate": "EURIBOR", "reference_level": 2.2, "margin": 3.5, "maturity_years": 6,
         "commitment_fee_pct": 0.35, "upfront_fee_pct": 1.0, "sweep": True, "sweep_share": 50.0,
         "allow_redraw": True},
        {"name": "Senior notes", "kind": "senior_notes", "amount": 70_000.0, "fixed_rate": 6.5, "maturity_years": 5},
        {"name": "PIK notes", "kind": "pik_notes", "amount": 40_000.0, "fixed_rate": 11.0, "pik_share": 100.0,
         "maturity_years": 8},
        {"name": "Vendor loan", "kind": "vendor_loan", "amount": 12_000.0, "fixed_rate": 5.0,
         "amort_schedule": [0.0, 2_000.0, 2_000.0, 4_000.0], "maturity_years": 6},
    ],
}


def run(inputs: dict, settings: dict | None = None) -> dict:
    answer = client.post("/api/deal/run", json={"inputs": inputs, "settings": settings or {}})
    assert answer.status_code == 200, answer.text
    return answer.json()


def download(inputs: dict, settings: dict | None = None, grouping: str = "locale", deal_id: str | None = None):
    answer = client.post("/api/export/deal-model" + (f"?deal_id={deal_id}" if deal_id else ""),
                         json={"inputs": inputs, "settings": settings or {}, "name": "Project Kestrel",
                               "grouping": grouping})
    assert answer.status_code == 200, answer.text
    return load_workbook(io.BytesIO(answer.content))


def years(wb, label: str, n: int, get) -> list:
    row = row_of(wb, label)
    return [get("%s%d" % (chr(ord("D") + y), row)) for y in range(n)]


def test_the_workbook_holds_formulas_and_inputs_but_no_answers():
    wb = download(EUR_DEAL)
    ws = wb[SHEET]
    assert ws[named(wb, "irr")].value.startswith("=")
    assert ws[named(wb, "equity")].value.startswith("=")
    assert ws[named(wb, "t1_drawn")].value.startswith("=")
    assert isinstance(ws[named(wb, "ebitda")].value, (int, float))     # an input
    # no formula reaches another sheet: the add-in may rename the sheet
    formulas = [c.value for row in ws.iter_rows() for c in row if isinstance(c.value, str) and c.value.startswith("=")]
    assert len(formulas) > 400 and not any("!" in f for f in formulas)


def test_a_euro_floating_rate_deal_recalculates_to_the_apps_irr():
    app = run(EUR_DEAL)
    wb = download(EUR_DEAL)
    value = worked_out(wb)
    assert value("irr") == pytest.approx(app["returns"]["irr"], abs=1e-4)          # 0.01 percentage points
    assert value("moic") == pytest.approx(app["returns"]["moic"], abs=1e-3)
    # the figures behind it, in the deal's own thousands (the engine rounds to ten thousand euros a line)
    assert value("equity") == pytest.approx(app["equity_bridge"]["entry_equity"], abs=15)
    assert value("net_debt_exit") == pytest.approx(app["returns"]["net_debt_at_exit"], abs=30)
    get = workbook_values(wb, SHEET)
    interest = years(wb, "Interest expense (from the facilities below)", 6, get)
    assert interest == pytest.approx(app["operating_model"]["interest_expense"], abs=30)
    cash = years(wb, "Closing cash (minimum cash plus what no facility could take)", 6, get)
    assert cash == pytest.approx(app["debt_schedule"]["cash_balance"], abs=30)
    taxes = years(wb, "Tax (from the tax lines below)", 6, get)
    assert taxes == pytest.approx(app["operating_model"]["taxes"], abs=30)


def test_every_mechanic_in_the_euro_deal_is_at_work():
    """The case is only a test of every debt type if each mechanic moves something in it."""
    wb = download(EUR_DEAL)
    get = workbook_values(wb, SHEET)
    rate = years(wb, "Rate (floating: the larger of the reference and its floor, plus the margin)", 6, get)
    assert rate[0] == pytest.approx(0.024 + 0.0425) and rate[2] == pytest.approx(0.020 + 0.0425)  # the floor bites
    app = run(EUR_DEAL)
    schedules = app["tranches"]
    assert any(y["redrawn"] > 0 for y in schedules["Revolver"])
    assert all(y["commitment_fee"] > 0 for y in schedules["Revolver"][:1])
    assert schedules["PIK notes"][-1]["ending_balance"] > 40_000
    assert schedules["Vendor loan"][1]["mandatory_repayment"] == pytest.approx(2_000)
    assert max(app["tax"]["interest_carried"]) > 0


# Defined only where LibreOffice is found or CI demands it (FSE_REQUIRE_SOFFICE): CI's ml job fails on a skipped test
if soffice_wanted():
    def test_libreoffice_recalculates_the_euro_deal_to_the_apps_irr():
        app = run(EUR_DEAL)
        wb = download(EUR_DEAL)
        irr_at, moic_at = named(wb, "irr"), named(wb, "moic")
        done = recalculated_by_libreoffice(wb)
        assert done[SHEET][irr_at].value == pytest.approx(app["returns"]["irr"], abs=1e-4)
        assert done[SHEET][moic_at].value == pytest.approx(app["returns"]["moic"], abs=1e-3)


def _one(kind: str, **terms) -> dict:
    return {"name": kind, "kind": kind, "amount": 450.0, "maturity_years": 5, **terms}


DEALS = {
    "percentages (the default deal: finding 11 in year 5)": {},
    "percentages in thousands with minimum cash": {"ebitda": 100_000.0, "unit": "thousands", "mincash": 8_000.0},
    "amortising term loan": {"tranches": [_one("amortising_term_loan", fixed_rate=7.0, amort_pct=15.0, sweep=True)]},
    "unitranche, floating with a floor": {"tranches": [_one(
        "unitranche", floating=True, reference_level=1.0, floor=2.5, margin=6.0, sweep=True)]},
    "second lien behind a swept loan": {"tranches": [
        _one("institutional_term_loan", amount=300.0, floating=True, reference_level=4.0, margin=3.0, amort_pct=1.0, sweep=True),
        _one("second_lien", amount=150.0, fixed_rate=11.0, maturity_years=7, sweep=True, sweep_priority=1)]},
    "a sweep share leaving cash to the next facility": {"tranches": [
        _one("institutional_term_loan", amount=300.0, fixed_rate=7.0, sweep=True, sweep_share=40.0),
        _one("second_lien", amount=150.0, fixed_rate=10.0, maturity_years=7, sweep=True)]},
    "PIK notes, half paid in kind": {"tranches": [
        _one("senior_notes", amount=300.0, fixed_rate=7.0, maturity_years=8),
        _one("pik_notes", amount=150.0, fixed_rate=12.0, pik_share=50.0, maturity_years=4)]},
    "shareholder loan on a schedule": {"tranches": [
        _one("amortising_term_loan", amount=350.0, fixed_rate=6.0, sweep=True),
        _one("shareholder_loan", amount=100.0, fixed_rate=8.0, amort_schedule=[10.0, 10.0, 30.0])]},
    "a revolver covering a bullet": {"mincash": 20.0, "tranches": [
        _one("senior_notes", amount=420.0, fixed_rate=8.0, maturity_years=3),
        _one("revolver", amount=200.0, drawn_pct=10.0, fixed_rate=6.0, commitment_fee_pct=0.5,
             allow_redraw=True, sweep=True)]},
    "a fixed interest cap of zero": {"tax_interest_limit": "fixed", "tax_interest_limit_amount": 0.0},
    "a loss-making deal carrying losses": {"opex": 30.0, "da": 8.0, "debt_pct": 80.0, "base_rate": 8.0,
                                           "growth": 10.0, "hold": 7, "tax_loss_carryforward": True,
                                           "tax_loss_limit_pct": 50.0},
    "a minimum tax above the rate's answer": {"tax": 5.0, "tax_minimum_pct": 15.0},
    "US GAAP leases on the post view": {"accounting_standard": "us_gaap", "lease_view": "post_ifrs16",
                                        "lease_cost": 8.0, "lease_liability": 40.0},
    "rupees in crores with a long hold": {"ebitda": 600.0, "unit": "crores", "currency": "INR", "hold": 9,
                                          "base_rate": 9.5},
}


@pytest.mark.parametrize("case", list(DEALS))
def test_every_structure_recalculates_to_the_apps_irr(case):
    inputs = DEALS[case]
    settings = {"other_uses": 5.0, "tx_fee_pct": 1.5} if "default" in case else {}
    app = run(inputs, settings)
    value = worked_out(download(inputs, settings))
    assert value("irr") == pytest.approx(app["returns"]["irr"], abs=1e-4), case
    assert value("moic") == pytest.approx(app["returns"]["moic"], abs=1e-3), case


def test_the_tax_cases_exercise_their_rules():
    """Losses carried and the minimum tax are each at work in their case, or the case proves nothing."""
    losses = run(DEALS["a loss-making deal carrying losses"])["tax"]
    assert max(losses["losses_carried"]) > 0 and max(losses["losses_used"]) > 0
    assert max(run(DEALS["a minimum tax above the rate's answer"])["tax"]["minimum_tax_topup"]) > 0


def test_a_changed_input_cell_moves_the_irr_as_the_app_does():
    """The formulas live: typing over an input in the workbook gives the app's answer for that input."""
    wb = download(EUR_DEAL)
    ws = wb[SHEET]
    ws[named(wb, "exit_multiple")].value = 12.0
    ws[named(wb, "t1_margin")].value = 0.05
    ws[named(wb, "tax_rate")].value = 0.25
    moved = {**EUR_DEAL, "exit_mult": 12.0, "tax": 25.0,
             "tranches": [{**EUR_DEAL["tranches"][0], "margin": 5.0}, *EUR_DEAL["tranches"][1:]]}
    expected = run(moved)["returns"]["irr"]
    assert expected != pytest.approx(run(EUR_DEAL)["returns"]["irr"], abs=1e-3)
    assert worked_out(wb)("irr") == pytest.approx(expected, abs=1e-4)


def test_the_option_pool_takes_its_share_of_exit_equity():
    """The app never sets a pool (it is the engine's own input), so the cell is checked by hand: a 10% pool
    leaves nine tenths of the exit equity."""
    wb = download(EUR_DEAL)
    value = worked_out(wb)
    gross, equity, hold = value("gross_exit"), value("equity"), 6
    wb[SHEET][named(wb, "option_pool")].value = 0.10
    value = worked_out(wb)
    assert value("net_exit") == pytest.approx(gross * 0.9)
    assert value("irr") == pytest.approx((gross * 0.9 / equity) ** (1 / hold) - 1)


def test_money_is_labelled_in_the_deals_currency_and_unit():
    wb = download(EUR_DEAL)
    ws = wb[SHEET]
    assert ws.cell(row_of(wb, "Revenue"), 2).value == "EUR thousands"
    assert "EUR thousands" in ws["A3"].value
    about = {r[0].value: r[1].value for r in wb["About"].iter_rows()}
    assert about["Currency"] == "EUR" and about["Money unit"] == "thousands" and about["Source"] == "variater.com"
    assert ws[named(wb, "irr")].number_format == "0.00%"


def test_lakh_grouping_sizes_each_money_cell():
    deal = DEALS["rupees in crores with a long hold"]
    plain = download(deal)
    lakh = download(deal, grouping="lakh")
    at = "D%d" % row_of(lakh, "Revenue")              # about 2,600 crores: four digits, a plain pattern
    assert lakh[SHEET][at].number_format == plain[SHEET][at].number_format
    big = {**deal, "unit": "lakhs", "ebitda": 60_000.0}          # 60,000 lakhs: revenue in the lakhs of lakhs
    sized = download(big, grouping="lakh")
    revenue = sized[SHEET]["D%d" % row_of(sized, "Revenue")]
    assert revenue.number_format.startswith("#\\,##\\,##0") or "\\," in revenue.number_format
    assert worked_out(sized)("irr") == pytest.approx(run(big)["returns"]["irr"], abs=1e-4)


def test_a_structure_the_deal_cannot_finance_is_refused_as_the_run_is():
    deal = {"tranches": [_one("amortising_term_loan", amount=5_000.0, fixed_rate=7.0)]}
    assert client.post("/api/deal/run", json={"inputs": deal}).status_code == 422
    answer = client.post("/api/export/deal-model", json={"inputs": deal})
    assert answer.status_code == 422


# ---------------------------------------------------------------------------
# A saved deal, for the Excel add-in (PLAN.md 7.3b)
# ---------------------------------------------------------------------------
def as_user(subject: str) -> None:
    app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject, is_dev=True)


@pytest.fixture
def saved(fresh_db):  # noqa: ARG001
    for subject in ("dev:stranger", "dev:owner"):
        as_user(subject)
        profile = {"country": "DE", "preferred_currency": "EUR", "locale": "de-DE", "time_zone": "Europe/Berlin"}
        assert client.post("/api/account", json=profile).status_code == 200
    made = client.post("/api/deals", json={"name": "Project Kestrel", "inputs": EUR_DEAL, "settings": {}})
    assert made.status_code in (200, 201), made.text
    yield made.json()["id"]
    as_user("dev:owner")


def _without_time(sheets: list) -> list:
    """The sheets less the About sheet's time of writing, which differs between two calls."""
    return [s for s in sheets if s["title"] != "About"]


def test_the_add_in_gets_the_saved_deals_cells_and_the_apps_answer(saved):
    cells = client.get(f"/api/deals/{saved}/model/cells")
    assert cells.status_code == 200, cells.text
    body = cells.json()
    app = run(EUR_DEAL)
    assert body["irr"] == pytest.approx(app["returns"]["irr"], abs=1e-9)
    assert body["name"] == "Project Kestrel" and body["money"]["currency"] == "EUR"
    model = body["sheets"][0]
    assert model["names"]["variater_deal"] == "$B$1"
    marker = next(c for c in model["cells"] if c["row"] == 1 and c["column"] == 2)
    assert marker["text"] == "Variater deal: " + saved


def test_the_add_ins_cells_are_the_downloads_cells(saved):
    """One description: the add-in writes exactly what the download holds."""
    from api.routers.export import deal_model
    from api.schemas import DealInputsIn

    body = client.get(f"/api/deals/{saved}/model/cells").json()
    sheets, *_ = deal_model(DealInputsIn(**EUR_DEAL), {}, name="Project Kestrel", saved=saved)
    expected = _without_time([{**s, "cells": [{k: v for k, v in c.items()} for c in s["cells"]]} for s in sheets])
    got = _without_time(body["sheets"])
    for e, g in zip(expected, got):
        # the schema answers every field; the description leaves out the unset ones (0.0 == False: test `is`)
        compact = [{k: v for k, v in c.items() if v is not None and v is not False} for c in g["cells"]]
        assert compact == e["cells"]
        assert g["names"] == e["names"]


def test_a_refresh_reads_the_deal_as_saved_now(saved):
    before = client.get(f"/api/deals/{saved}/model/cells").json()["irr"]
    changed = {**EUR_DEAL, "exit_mult": 11.5}
    put = client.put(f"/api/deals/{saved}/draft", json={"inputs": changed, "settings": {}})
    assert put.status_code == 200, put.text
    after = client.get(f"/api/deals/{saved}/model/cells").json()
    assert after["irr"] != pytest.approx(before, abs=1e-3)
    assert after["irr"] == pytest.approx(run(changed)["returns"]["irr"], abs=1e-9)


def test_another_account_cannot_get_a_deals_cells(saved):
    as_user("dev:stranger")
    assert client.get(f"/api/deals/{saved}/model/cells").status_code == 404


def test_each_export_is_one_entry_in_the_deals_history(saved):
    from db import engine as db_engine
    from db.models import AuditEvent

    client.get(f"/api/deals/{saved}/model/cells")
    download(EUR_DEAL, deal_id=saved)
    with db_engine.connect() as conn:
        rows = conn.execute(select(AuditEvent.detail).where(AuditEvent.action == "exported")).all()
    assert sorted(r.detail["export"] for r in rows) == ["deal_model", "deal_model_addin"]
