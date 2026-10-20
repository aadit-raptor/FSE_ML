"""Results notes through the API (PLAN.md R8), on a real database: a note drawn from the saved model with every
figure its cell's; what the analyst typed kept and the only rating in it; the firm name and disclosures typed once
for the account; the AI's sentences written, checked and limited; Word and PDF downloads, each in the history;
another account's note a 404.
"""
from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from api.main import app
from research import writer
from research.reader.providers import Answer
from tests.test_research_saved_api import accept_half_year, start
from tests.test_uploads import uploads  # noqa: F401 -- the fixture

client = TestClient(app)


def note(company_id, rest=""):
    return f"/api/research/companies/{company_id}/report{rest}"


def figures(blocks):
    out = []
    for b in blocks:
        cells = (b.get("parts") or []) + [seg for c in (b.get("header") or []) for seg in c] + \
            [seg for row in (b.get("rows") or []) for c in row for seg in c]
        out += [seg for seg in cells if seg["kind"] == "figure"]
    return out


def text_of(blocks) -> str:
    out = []
    for b in blocks:
        out.append("".join(seg["text"] for seg in b.get("parts") or []))
        out += [seg["text"] for row in (b.get("rows") or []) for c in row for seg in c]
    return " ".join(out)


class Writes:
    def __init__(self, answer):
        self.answer, self.sent = answer, []

    def write(self, text):
        self.sent.append(text)
        return Answer(self.answer, "stand-in", "")


@pytest.fixture
def no_ai(monkeypatch):
    monkeypatch.setattr("research.reader.providers.get_provider", lambda: None)


def test_a_note_is_drawn_from_the_saved_model_and_keeps_only_what_the_analyst_typed(uploads, no_ai):  # noqa: F811
    company_id, saved = start()
    first = client.get(note(company_id))
    assert first.status_code == 200, first.text
    got = first.json()
    assert got["fields"] == {"argument": "", "rating": "", "target": None, "note_date": None, "compare_version": None}
    assert got["details"] == {"firm": "", "disclosures": ""}
    revenue = [f for f in figures(got["blocks"]) if f["source"] == "model"]
    assert len(revenue) > 30 and "12,071.7" in [f["text"] for f in revenue]
    assert {s["source"] for s in got["sentences"]} == {"template"}
    typed = client.put(note(company_id), json={"rating": "  Accumulate ", "target": 412.5, "note_date": "2026-10-20",
                                               "argument": "Beds open on time."})
    assert typed.status_code == 200, typed.text
    body = typed.json()
    assert body["fields"]["rating"] == "Accumulate" and body["fields"]["note_date"] == "2026-10-20"
    whole = text_of(body["blocks"])
    assert whole.count("Accumulate") == 1 and "412.50" in whole and "Beds open on time." in whole
    assert client.get(note(company_id)).json()["fields"] == body["fields"]
    cleared = client.put(note(company_id), json={}).json()
    assert "Accumulate" not in text_of(cleared["blocks"])


def test_a_note_is_refused_what_it_cannot_hold(uploads, no_ai):  # noqa: F811
    company_id, _ = start()
    for body in ({"rating": "x" * 41}, {"target": -1}, {"note_date": "yesterday"}, {"compare_version": 0},
                 {"price": 100}):
        assert client.put(note(company_id), json=body).status_code == 422, body
    missing = client.put(note(company_id), json={"compare_version": 9})
    assert missing.status_code == 422 and missing.json()["detail"]["code"] == "no_version"
    empty = client.post("/api/research/companies", json={"name": "No model"}).json()["id"]
    for answer in (client.get(note(empty)), client.put(note(empty), json={}), client.get(note(empty, "/pdf"))):
        assert answer.status_code == 404 and answer.json()["detail"]["code"] == "no_model"


def test_estimates_are_set_against_a_version_and_results_against_the_rolls_estimate(uploads, no_ai):  # noqa: F811
    company_id, _ = start()
    accept_half_year("test:alice", company_id)
    rolled = client.post(f"/api/research/companies/{company_id}/model/roll")
    assert rolled.status_code == 200, rolled.text
    version = rolled.json()["version"]
    got = client.put(note(company_id), json={"compare_version": version}).json()
    sources = {f["source"] for f in figures(got["blocks"])}
    assert sources == {"model", "earlier", "comparison"}
    comparison = rolled.json()["comparison"]["lines"]
    tables = [b for b in got["blocks"] if b["type"] == "table"]
    shown = [f["text"] for f in figures(tables) if f["source"] == "comparison"]
    assert len(shown) == 4 * sum(1 for line in comparison if line["kind"] == "flow")
    assert {s["section"] for s in got["sentences"]} == set(writer.SECTIONS)


def test_the_firm_name_and_disclosures_are_typed_once_for_every_note(uploads, no_ai):  # noqa: F811
    company_id, _ = start()
    other, _ = start("Second Hospital")
    put = client.put("/api/account/report-details",
                     json={"firm": "  Example   Securities ", "disclosures": "SEBI registration INH000000000."})
    assert put.status_code == 200 and put.json()["firm"] == "Example Securities"
    for cid in (company_id, other):
        whole = text_of(client.get(note(cid)).json()["blocks"])
        assert "Example Securities" in whole and "SEBI registration INH000000000." in whole
    assert client.put("/api/account/report-details", json={"firm": "x" * 201}).status_code == 422
    history = client.get("/api/account/history").json()["entries"]
    assert [e["action"] for e in history].count("report_details_saved") == 1
    client.put("/api/account/report-details", json=put.json())           # the same again: no entry
    history = client.get("/api/account/history").json()["entries"]
    assert [e["action"] for e in history].count("report_details_saved") == 1


def test_the_ais_sentences_are_written_checked_kept_and_limited(uploads, monkeypatch):  # noqa: F811
    company_id, _ = start()
    ai = Writes({"outlook": "Revenue rises from {rev_ly} to {rev_cy}.", "valuation": "It is cheap at {price}."})
    monkeypatch.setattr("research.reader.providers.get_provider", lambda: ai)
    written = client.post(note(company_id, "/sentences"))
    assert written.status_code == 200, written.text
    got = written.json()
    by = {s["section"]: s["source"] for s in got["sentences"]}
    assert by == {"outlook": "ai", "valuation": "ai_refused"} and got["written_by"] == "stand-in"
    assert "Start" not in ai.sent[0] and "Yatharth" not in ai.sent[0]
    assert "Revenue rises from 12,071.7 to " in text_of(got["blocks"]) and "cheap" not in text_of(got["blocks"])
    assert client.get(note(company_id)).json()["sentences"] == got["sentences"]       # kept with the note
    for _ in range(4):
        assert client.post(note(company_id, "/sentences")).status_code == 200
    refused = client.post(note(company_id, "/sentences"))
    assert refused.status_code == 429
    # without an AI the templates write, and nothing is counted
    monkeypatch.setattr("research.reader.providers.get_provider", lambda: None)
    off = client.post(note(company_id, "/sentences")).json()
    assert {s["source"] for s in off["sentences"]} == {"ai_off"}


def test_the_word_and_pdf_files_are_the_note_and_each_is_in_the_history(uploads, no_ai):  # noqa: F811
    company_id, _ = start()
    client.put(note(company_id), json={"rating": "Add"})
    shown = [f["text"] for f in figures(client.get(note(company_id)).json()["blocks"]) if f["text"] != "–"]
    word = client.get(note(company_id, "/word"))
    assert word.status_code == 200 and word.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert 'filename="Yatharth_Hospital_results_note.docx"' in word.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(word.content)) as z:
        document = z.read("word/document.xml").decode()
    pdf = client.get(note(company_id, "/pdf"))
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    text = " ".join(" ".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf.content)).pages).split())
    for figure in shown:
        assert figure in document and figure in text, figure
    assert ">Add<" in document and "Add" in text
    exports = [e.get("export") for e in client.get("/api/account/history").json()["entries"]
               if e["action"] == "exported"]
    assert exports == ["report_pdf", "report_word"]


def test_another_accounts_note_is_not_found(uploads, no_ai):  # noqa: F811
    company_id, _ = start()
    client.put(note(company_id), json={"rating": "Add"})
    uploads("test:bob")
    for answer in (client.get(note(company_id)), client.put(note(company_id), json={"rating": "Sell"}),
                   client.post(note(company_id, "/sentences")), client.get(note(company_id, "/word")),
                   client.get(note(company_id, "/pdf"))):
        assert answer.status_code == 404, answer.text
        assert answer.json()["detail"] == "not_found"
    assert client.get("/api/account/report-details").json() == {"firm": "", "disclosures": ""}
    uploads("test:alice")
    assert client.get(note(company_id)).json()["fields"]["rating"] == "Add"


def test_each_change_to_a_note_writes_one_audit_entry_and_none_for_a_repeat(uploads, no_ai):  # noqa: F811
    company_id, _ = start()
    client.put(note(company_id), json={"rating": "Add"})
    client.put(note(company_id), json={"rating": "Add"})
    client.put(note(company_id), json={"rating": "Add", "argument": "More beds."})
    entries = [e for e in client.get("/api/account/history").json()["entries"] if e["action"] == "report_saved"]
    assert [e["fields"] for e in entries] == [["argument"], ["rating"]]


def test_writing_sentences_keeps_the_fields_as_they_stand_and_a_pruned_version_does_not_lock_the_note(
        uploads, no_ai):  # noqa: F811
    import uuid as _uuid

    from db import research_reports

    company_id, _ = start()
    client.put(note(company_id), json={"rating": "Add"})
    research_reports.save_report("test:alice", _uuid.UUID(company_id), sentences={"sections": {}})
    assert client.get(note(company_id)).json()["fields"]["rating"] == "Add"
    # a version chosen to compare with, then gone (versions are pruned): the note still opens, without it
    kept = client.post(f"/api/research/companies/{company_id}/model/versions", json={"label": "Kept"}).json()
    client.put(note(company_id), json={"rating": "Add", "compare_version": kept["number"]})
    from db.engine import transaction
    from db.models import ResearchModelVersion
    from sqlalchemy import delete
    with transaction() as conn:
        conn.execute(delete(ResearchModelVersion).where(ResearchModelVersion.number == kept["number"]))
    got = client.get(note(company_id))
    assert got.status_code == 200 and got.json()["fields"]["compare_version"] is None
    assert client.get(note(company_id, "/pdf")).status_code == 200
