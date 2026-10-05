"""Every Excel download says it came from variater.com: in its About sheet and
in the file's own properties (what Excel shows under File > Info > Author)."""
import io
from datetime import datetime, timezone

import openpyxl
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routers.export import SOURCE

client = TestClient(app)
SHEETS = [{"name": "P&L", "columns": ["", "Y1"], "rows": [["Revenue", 1.0]]}]

DOWNLOADS = [
    ("/api/export/workbook", {"sheets": SHEETS}),
    ("/api/export/workbook", {"sheets": SHEETS, "money": {"currency": "EUR", "unit": "thousands"}}),
    ("/api/export/montecarlo-sample", {"mc": {"n": 1000}, "seed": 3}),
]


def download(path, body):
    resp = client.post(path, json=body)
    assert resp.status_code == 200, resp.text
    return resp.content


@pytest.mark.parametrize("path, body", DOWNLOADS)
def test_every_download_names_variater_as_its_author(path, body):
    book = openpyxl.load_workbook(io.BytesIO(download(path, body)))
    assert SOURCE == "variater.com"
    assert book.properties.creator == SOURCE
    assert book.properties.lastModifiedBy == SOURCE


@pytest.mark.parametrize("path, body", DOWNLOADS)
def test_every_about_sheet_says_where_and_when_it_was_made(path, body):
    before = datetime.now(timezone.utc).replace(microsecond=0)
    about = pd.read_excel(io.BytesIO(download(path, body)), sheet_name="About")
    values = dict(zip(about["Item"], about["Value"]))
    assert values["Source"] == "variater.com"
    made = datetime.strptime(values["Generated (UTC)"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    assert before <= made <= datetime.now(timezone.utc)
    # The source comes first, so it is the first thing a reader sees
    assert list(values)[0] == "Source"
