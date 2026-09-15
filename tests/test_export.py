"""Excel downloads: the workbook holds exactly the tables sent, and the Monte
Carlo sample is the simulation behind the on-screen results."""
import io

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _read(resp):
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == XLSX
    return pd.read_excel(io.BytesIO(resp.content), sheet_name=None)


def test_workbook_round_trips_tables_and_sanitises_names():
    body = {"filename": "lbo summary/2026", "sheets": [
        {"name": "P&L", "columns": ["Line", "Y1", "Y2"], "rows": [["Revenue", 250.0, 262.5], ["EBITDA", 65.5, None]]},
        {"name": "Debt: schedule [senior]", "columns": ["Year", "Balance"], "rows": [[1, 420.0], [2]]},
        {"name": "P&L", "columns": ["x"], "rows": []},
    ]}
    r = client.post("/api/export/workbook", json=body)
    books = _read(r)
    assert 'filename="lbo_summary_2026.xlsx"' in r.headers["content-disposition"]
    assert list(books) == ["P&L", "Debt  schedule  senior", "P&L 2"]
    pl = books["P&L"]
    assert pl.columns.tolist() == ["Line", "Y1", "Y2"]
    assert pl.loc[0, "Y2"] == 262.5 and pd.isna(pl.loc[1, "Y2"])
    assert pd.isna(books["Debt  schedule  senior"].loc[1, "Balance"])


def test_workbook_validation():
    assert client.post("/api/export/workbook", json={"sheets": []}).status_code == 422
    assert client.post("/api/export/workbook", json={"sheets": [{"name": "a", "columns": []}]}).status_code == 422


def test_montecarlo_sample_matches_the_seeded_run():
    req = {"mc": {"n": 4000}, "seed": 11}
    run = client.post("/api/montecarlo/run", json=req).json()
    books = _read(client.post("/api/export/montecarlo-sample", json=req))
    paths = books["Paths"]
    assert len(paths) == 4000 and {"IRR", "MOIC", "Growth", "Exit Multiple"} <= set(paths.columns)
    assert abs(paths["IRR"].mean() - run["summary"]["mean_irr"]) < 1e-9
    about = dict(zip(books["About"]["Item"], books["About"]["Value"]))
    assert about["Seed"] == 11 and about["Paths simulated"] == 4000
