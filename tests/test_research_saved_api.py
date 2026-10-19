"""Saved sector models through the API (PLAN.md R5), on a real database: a
model per covered company with versions; a roll forward whose estimate is
kept beside the results, unchanged by later edits; "Save to Variater" reading
the named cells back from a sheet, and refusing one whose structure changed;
another account's model a 404.
"""
from __future__ import annotations

import copy
import uuid

import pytest
from fastapi.testclient import TestClient

from api.main import app
from db import coverage as coverage_rows
from research.evaluate import evaluate
from tests.test_research_actuals import period, results, table
from tests.test_research_rollforward import AS_OF, EXAMPLES, yatharth
from tests.test_uploads import uploads  # noqa: F401 -- the fixture

client = TestClient(app)


def start(name="Yatharth Hospital"):
    """A covered company with Yatharth's model saved for it, three months of FY27 reported."""
    made = client.post("/api/research/companies", json={"name": name})
    assert made.status_code == 201, made.text
    company_id = made.json()["id"]
    ex = next(e for e in EXAMPLES if e["id"] == "yatharth-hospital")
    saved = client.put(f"/api/research/companies/{company_id}/model",
                       json={"structure": ex["structure"], "company": ex["company"], "as_of": AS_OF})
    assert saved.status_code == 200, saved.text
    return company_id, saved.json()


def accept_half_year(subject, company_id, extra=4100.0):
    _, company = yatharth()
    columns = [("Q2 FY27", "2026-09-30", 3), ("H1 FY27", "2026-09-30", 6), ("FY26", "2026-03-31", 12)]
    stored = period([results(columns, [extra, company.values["rev"]["ytd"] + extra, 12071.72])], unit="millions")
    coverage_rows.save_periods(subject, uuid.UUID(company_id), [{"period_end": "2026-09-30", "months": 6, **stored}],
                               None)


def path(company_id, rest=""):
    return f"/api/research/companies/{company_id}/model{rest}"


# ------------------------------------------------------------------ saving and versions
def test_a_model_is_saved_per_company_and_comes_back_as_saved(uploads):  # noqa: F811
    company_id, saved = start()
    assert saved["changed"] == ["created"] and saved["latest_version"] == 1 and saved["as_of"] == AS_OF
    got = client.get(path(company_id)).json()
    ex = next(e for e in EXAMPLES if e["id"] == "yatharth-hospital")
    assert got["company"]["values"] == ex["company"]["values"] and got["structure"] == ex["structure"]
    assert got["newer_results"] is None and got["company_name"] == "Yatharth Hospital"
    versions = client.get(path(company_id, "/versions")).json()["versions"]
    assert [(v["number"], v["kind"]) for v in versions] == [(1, "created")]


def test_a_company_without_a_model_says_so_and_a_model_that_does_not_build_is_refused(uploads):  # noqa: F811
    made = client.post("/api/research/companies", json={"name": "Empty"}).json()["id"]
    missing = client.get(path(made))
    assert missing.status_code == 404 and missing.json()["detail"]["code"] == "no_model"
    ex = next(e for e in EXAMPLES if e["id"] == "yatharth-hospital")
    wrong = copy.deepcopy(ex["company"])
    wrong["choices"]["line1_past"] = "guessed"         # not one of the choice's options
    refused = client.put(path(made), json={"structure": ex["structure"], "company": wrong})
    assert refused.status_code == 422 and refused.json()["detail"]["code"] == "company_refused"
    assert client.get(path(made)).status_code == 404
    # as a run takes it: a figure typed where the model works one out is left out, a choice not made takes its first
    loose = copy.deepcopy(ex["company"])
    loose["values"]["rev"]["y1"] = 5.0
    del loose["choices"]["line1_end_to_date"]
    kept = client.put(path(made), json={"structure": ex["structure"], "company": loose}).json()
    assert "y1" not in kept["company"]["values"]["rev"] and kept["company"]["choices"]["line1_end_to_date"]


def test_autosave_writes_only_what_changed_and_versions_restore(uploads):  # noqa: F811
    company_id, saved = start()
    body = {"structure": saved["structure"], "company": saved["company"], "as_of": AS_OF}
    again = client.put(path(company_id), json=body).json()
    assert again["changed"] == [] and again["updated_at"] == saved["updated_at"]
    kept = client.post(path(company_id, "/versions"), json={"label": "Before the call"})
    assert kept.status_code == 201 and (kept.json()["number"], kept.json()["label"]) == (1, "Before the call")
    changed = copy.deepcopy(body)
    changed["company"]["values"]["mat_sh"]["y1"] = 0.4
    edited = client.put(path(company_id), json=changed).json()
    assert edited["changed"] == ["mat_sh"]
    restored = client.post(path(company_id, "/versions/1/restore"))
    assert restored.status_code == 200
    assert restored.json()["company"]["values"]["mat_sh"]["y1"] == saved["company"]["values"]["mat_sh"]["y1"]
    kinds = [v["kind"] for v in client.get(path(company_id, "/versions")).json()["versions"]]
    assert kinds[:2] == ["restored", "saved"]          # the edit kept before the restore


# ------------------------------------------------------------------ roll forward
def test_newer_results_are_offered_and_the_roll_sets_the_estimate_beside_them(uploads):  # noqa: F811
    company_id, saved = start()
    accept_half_year("test:alice", company_id)
    got = client.get(path(company_id)).json()
    assert got["newer_results"] == {"ly_end": "2026-03-31", "ytd_end": "2026-09-30", "ytd_months": 6}
    model, company = yatharth()
    before = evaluate(model, company, partial=True).values
    rolled = client.post(path(company_id, "/roll"))
    assert rolled.status_code == 200, rolled.text
    body = rolled.json()
    assert body["kind"] == "quarter" and body["model"]["as_of"]["ytd_months"] == 6
    assert body["model"]["newer_results"] is None
    rev = next(x for x in body["comparison"]["lines"] if x["row"] == "rev")
    assert rev["estimate"] == pytest.approx(before["rev"]["roy"] * 92 / 274, abs=1e-6)
    assert rev["actual"] == pytest.approx(4100, abs=1e-6)
    assert {"row": "s1_occ", "period": "ytd"} in body["carried"]
    again = client.post(path(company_id, "/roll"))
    assert again.status_code == 422 and again.json()["detail"]["code"] == "nothing_newer"


def test_the_estimate_saved_before_a_quarter_is_unchanged_by_later_edits(uploads):  # noqa: F811
    company_id, saved = start()
    accept_half_year("test:alice", company_id)
    body = client.post(path(company_id, "/roll")).json()
    kept = body["comparison"]
    rolled_version = next(v for v in client.get(path(company_id, "/versions")).json()["versions"]
                          if v["kind"] == "rolled")
    assert rolled_version["comparison"] == kept and rolled_version["number"] == body["version"]
    assert rolled_version["as_of"] == AS_OF            # the version is the model as it stood before
    # the analyst changes the assumptions the estimate came from, and the results are accepted again
    model = client.get(path(company_id)).json()
    model["company"]["values"]["mat_sh"]["roy"] = 0.9
    model["company"]["values"]["s1_price"]["roy"] = 1.0
    assert client.put(path(company_id), json={"structure": model["structure"], "company": model["company"],
                                              "as_of": model["as_of"]}).json()["changed"]
    accept_half_year("test:alice", company_id, extra=9999.0)
    later = next(v for v in client.get(path(company_id, "/versions")).json()["versions"] if v["kind"] == "rolled")
    assert later["comparison"] == kept


def test_restoring_the_version_before_a_roll_brings_its_period_back(uploads):  # noqa: F811
    company_id, _ = start()
    accept_half_year("test:alice", company_id)
    version = client.post(path(company_id, "/roll")).json()["version"]
    back = client.post(path(company_id, f"/versions/{version}/restore")).json()
    assert back["as_of"] == AS_OF and back["newer_results"]["ytd_months"] == 6


# ------------------------------------------------------------------ save back from Excel
def as_excel(sheet):
    """A sheet of cells as Excel answers it from A1: words and numbers (a formula's value is not read)."""
    rows = max(c["row"] for c in sheet["cells"])
    columns = max(c["column"] for c in sheet["cells"])
    grid = [[""] * columns for _ in range(rows)]
    for c in sheet["cells"]:
        value = c["text"] if c.get("text") is not None else c.get("number")
        grid[c["row"] - 1][c["column"] - 1] = 0 if c.get("formula") else ("" if value is None else value)
    return grid, dict(sheet["names"])


def type_into(grid, names, name, value):
    address = names[name].replace("$", "")
    letters = "".join(ch for ch in address if ch.isalpha())
    column = 0
    for ch in letters:
        column = column * 26 + ord(ch) - 64
    grid[int(address[len(letters):]) - 1][column - 1] = value


def test_a_workbook_saved_changed_in_excel_and_saved_again_returns_the_changed_assumptions(uploads):  # noqa: F811
    company_id, _ = start()
    cells = client.get(path(company_id, "/cells"))
    assert cells.status_code == 200, cells.text
    sheet = cells.json()["sheets"][0]
    assert any((c.get("text") or "").endswith(company_id) for c in sheet["cells"] if c["row"] == 1)
    grid, names = as_excel(sheet)
    type_into(grid, names, "mat_sh_y1", 0.33)
    first = client.post(path(company_id, "/sheet"), json={"grid": grid, "names": names})
    assert first.status_code == 200, first.text
    assert {"row": "mat_sh", "period": "y1", "after": 0.33} == {
        k: v for k, v in first.json()["changed"][0].items() if k != "before"}
    assert client.get(path(company_id)).json()["company"]["values"]["mat_sh"]["y1"] == 0.33
    # changed again in the sheet the add-in loaded, and saved again
    type_into(grid, names, "mat_sh_y1", 0.35)
    type_into(grid, names, "tax_rate_y2", 0.2)
    second = client.post(path(company_id, "/sheet"), json={"grid": grid, "names": names}).json()
    assert {(c["row"], c["period"], c["after"]) for c in second["changed"]} == {("mat_sh", "y1", 0.35),
                                                                                ("tax_rate", "y2", 0.2)}
    values = client.get(path(company_id)).json()["company"]["values"]
    assert values["mat_sh"]["y1"] == 0.35 and values["tax_rate"]["y2"] == 0.2
    labels = [v["label"] for v in client.get(path(company_id, "/versions")).json()["versions"]]
    assert labels.count("Saved from Excel") == 2


def test_a_restructured_sheet_is_refused_and_nothing_is_saved(uploads):  # noqa: F811
    company_id, saved = start()
    grid, names = as_excel(client.get(path(company_id, "/cells")).json()["sheets"][0])
    type_into(grid, names, "mat_sh_y1", 0.5)
    grid.insert(30, [""] * len(grid[0]))               # a row inserted; Excel moves the names below it
    moved = {}
    for name, address in names.items():
        row = int("".join(ch for ch in address if ch.isdigit()))
        moved[name] = address.replace("$%d" % row, "$%d" % (row + 1)) if row > 30 else address
    refused = client.post(path(company_id, "/sheet"), json={"grid": grid, "names": moved})
    assert refused.status_code == 422 and refused.json()["detail"]["code"] == "restructured"
    assert "row 31" in refused.json()["detail"]["message"]
    assert client.get(path(company_id)).json()["company"]["values"] == saved["company"]["values"]


def test_the_saved_workbook_carries_the_models_reference(uploads):  # noqa: F811
    import io

    from openpyxl import load_workbook
    company_id, _ = start()
    answer = client.get(path(company_id, "/workbook"))
    assert answer.status_code == 200
    wb = load_workbook(io.BytesIO(answer.content))
    assert wb["Model"]["B1"].value == "Saved in Variater: " + company_id


# ------------------------------------------------------------------ whose model
def test_another_accounts_model_is_not_found_and_cannot_be_written(uploads):  # noqa: F811
    company_id, saved = start()
    accept_half_year("test:alice", company_id)
    grid, names = as_excel(client.get(path(company_id, "/cells")).json()["sheets"][0])
    uploads("test:bob")
    body = {"structure": saved["structure"], "company": saved["company"]}
    for answer in (client.get(path(company_id)), client.put(path(company_id), json=body),
                   client.get(path(company_id, "/versions")), client.post(path(company_id, "/roll")),
                   client.post(path(company_id, "/versions"), json={}),
                   client.get(path(company_id, "/cells")),
                   client.post(path(company_id, "/sheet"), json={"grid": grid, "names": names})):
        assert answer.status_code == 404, answer.text
        assert answer.json()["detail"] == "not_found"
    uploads("test:alice")
    assert client.get(path(company_id)).json()["latest_version"] == 1


def test_each_change_writes_one_audit_entry(uploads):  # noqa: F811
    company_id, saved = start()
    accept_half_year("test:alice", company_id)
    client.post(path(company_id, "/roll"))
    client.post(path(company_id, "/versions"), json={"label": "Kept"})
    history = client.get("/api/account/history").json()["entries"]
    actions = [e["action"] for e in history]
    assert actions.count("model_saved") == 1 and actions.count("model_rolled") == 1
    assert actions.count("model_versioned") == 1
    rolled = next(e for e in history if e["action"] == "model_rolled")
    assert rolled["version"] == 2 and rolled["deal_id"] is None
    assert next(e for e in history if e["action"] == "model_saved")["fields"] == ["created"]


def test_a_save_from_excel_onto_a_copy_changed_since_it_was_read_is_refused(uploads, monkeypatch):  # noqa: F811
    from db import research_models as rows
    company_id, saved = start()
    grid, names = as_excel(client.get(path(company_id, "/cells")).json()["sheets"][0])
    type_into(grid, names, "mat_sh_y1", 0.33)
    real = rows.get_model

    def read_then_edited(subject, cid):        # the screen autosaves between the sheet's read and its save
        record = real(subject, cid)
        body = {"structure": saved["structure"], "company": dict(saved["company"], name="Edited meanwhile")}
        assert client.put(path(company_id), json=body).status_code == 200
        return record
    monkeypatch.setattr(rows, "get_model", read_then_edited)
    refused = client.post(path(company_id, "/sheet"), json={"grid": grid, "names": names})
    assert refused.status_code == 409 and refused.json()["detail"]["code"] == "changed_meanwhile"


def test_a_rolled_versions_label_is_kept_when_a_version_is_asked_for(uploads):  # noqa: F811
    company_id, _ = start()
    accept_half_year("test:alice", company_id)
    client.post(path(company_id, "/roll"))
    assert client.post(path(company_id, "/versions"), json={"label": "Mine"}).status_code == 201
    rolled = next(v for v in client.get(path(company_id, "/versions")).json()["versions"] if v["kind"] == "rolled")
    assert rolled["label"].startswith("Before rolling forward to")
