"""Investment memos through the API (PLAN.md 6.3), on a real database: a saved version drafted as a job and drawn
with every figure its run's; the AI's sentences written from nothing of the deal, counted and limited; Word and PDF
downloads in the deal's history; one audit entry per change; another account's deal a 404.
"""
from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from api.auth import AuthUser, require_user
from api.main import app
from core.model_version import ENGINE_VERSION
from jobs import config as jobs_config
from jobs.runner import Runner
from research.reader.providers import Answer

client = TestClient(app)

PROFILE = {"country": "GB", "preferred_currency": "GBP", "locale": "en-GB", "time_zone": "Europe/London"}
NAME = "Kestrel Holdings Ltd (company 01234567)"
INPUTS = {"ebitda": 120.0, "entry_mult": 9.0, "exit_mult": 9.5, "hold": 5, "currency": "GBP", "unit": "millions"}
SETTINGS = {"mc_n": 10000}


@pytest.fixture
def account(fresh_db):  # noqa: ARG001 -- the database
    def as_user(subject: str):
        app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)
        assert client.post("/api/account", json=PROFILE).status_code == 200
    as_user("test:alice")
    return as_user


@pytest.fixture
def no_ai(monkeypatch):
    monkeypatch.setattr("research.reader.providers.get_provider", lambda: None)


def saved_deal(label="Board pack") -> tuple[str, int]:
    deal = client.post("/api/deals", json={"name": NAME, "inputs": INPUTS, "settings": SETTINGS})
    assert deal.status_code == 201, deal.text
    deal_id = deal.json()["id"]
    version = client.post(f"/api/deals/{deal_id}/versions", json={"label": label})
    assert version.status_code in (200, 201), version.text
    return deal_id, version.json()["number"]


def run_job():
    return Runner(jobs_config.get_queue).run_next()


def memos(deal_id, rest=""):
    return f"/api/deals/{deal_id}/memos{rest}"


def drafted(deal_id: str, version: int) -> dict:
    started = client.post(memos(deal_id), json={"version": version})
    assert started.status_code == 202, started.text
    job = started.json()
    assert run_job() == "succeeded"
    done = client.get(memos(deal_id, f"/jobs/{job['id']}"))
    assert done.status_code == 200 and done.json()["status"] == "succeeded", done.text
    assert done.json()["result"]["version"] == version
    got = client.get(memos(deal_id, f"/{version}"))
    assert got.status_code == 200, got.text
    return got.json()


def figures(blocks):
    out = []
    for b in blocks:
        cells = (b.get("parts") or []) + [seg for row in (b.get("rows") or []) for c in row for seg in c]
        out += [seg for seg in cells if seg["kind"] == "figure"]
    return out


def text_of(blocks) -> str:
    out = []
    for b in blocks:
        out.append("".join(seg["text"] for seg in b.get("parts") or []))
        out += [seg["text"] for row in (b.get("rows") or []) for c in row for seg in c]
    return " ".join(out)


def test_a_version_is_drafted_as_a_job_and_drawn_from_its_run(account, no_ai):
    deal_id, version = saved_deal()
    got = drafted(deal_id, version)
    assert got["deal_name"] == NAME and got["version"] == version and got["current"] is True
    assert got["engine_version"] == ENGINE_VERSION and got["language"] == "en"
    # the figures are the screens' answers for the version
    irr = client.post("/api/deal/run", json={"inputs": INPUTS, "settings": SETTINGS}).json()["returns"]["irr"]
    shown = {f["source"]: f for f in figures(got["blocks"])}
    assert shown["run"]["kind"] == "figure"
    assert "%.1f%%" % (irr * 100) in [f["text"] for f in figures(got["blocks"])]
    assert "Version %d: Board pack." % version in text_of(got["blocks"])
    assert {s["source"] for s in got["sentences"]} == {"ai_off"} or {s["source"] for s in got["sentences"]} == \
        {"template"}
    listed = client.get(memos(deal_id)).json()
    assert [m["version"] for m in listed["memos"]] == [version] and listed["memos"][0]["current"] is True


def test_drafting_again_replaces_the_run_and_its_sentences_and_a_moved_engine_is_said(account, no_ai, monkeypatch):
    deal_id, version = saved_deal()
    drafted(deal_id, version)
    from db import memos as rows

    record = rows.get_memo("test:alice", __import__("uuid").UUID(deal_id), version)
    monkeypatch.setattr("api.routers.memos.ENGINE_VERSION", "9.9.9")
    got = client.get(memos(deal_id, f"/{version}")).json()
    assert got["current"] is False and got["engine_version"] == record.engine_version
    assert client.get(memos(deal_id)).json()["memos"][0]["current"] is False


def test_a_missing_version_another_accounts_deal_and_another_kind_of_job_are_not_found(account, no_ai):
    deal_id, version = saved_deal()
    assert client.post(memos(deal_id), json={"version": 99}).status_code == 404
    assert client.get(memos(deal_id, "/1")).status_code == 404            # no memo yet
    drafted(deal_id, version)
    account("test:bob")
    for answer in (client.get(memos(deal_id)), client.post(memos(deal_id), json={"version": version}),
                   client.get(memos(deal_id, f"/{version}")), client.post(memos(deal_id, f"/{version}/sentences")),
                   client.get(memos(deal_id, f"/{version}/word")), client.get(memos(deal_id, f"/{version}/pdf")),
                   client.delete(memos(deal_id, f"/{version}"))):
        assert answer.status_code == 404, answer.text
    other = client.post("/api/jobs", json={"kind": "montecarlo.run", "input": {"mc": {"n": 1000}, "seed": 1}})
    assert client.get(memos(deal_id, f"/jobs/{other.json()['id']}")).status_code == 404
    account("test:alice")
    assert client.get(memos(deal_id, f"/{version}")).status_code == 200


class Writes:
    def __init__(self, answer):
        self.answer, self.sent = answer, []
        self.free_tier = True
        self.name = "gemini"

    def write(self, text):
        self.sent.append(text)
        return Answer(self.answer, "gemini-3-flash-preview", "", {"promptTokenCount": 900, "candidatesTokenCount": 120})


def test_the_ais_sentences_hold_nothing_of_the_deal_are_counted_and_limited(account, monkeypatch):
    deal_id, version = saved_deal(label="Pack for the Kestrel board")
    monkeypatch.setattr("research.reader.providers.get_provider", lambda: None)
    drafted(deal_id, version)
    ai = Writes({"summary": "Equity grows from {entry_equity} to {exit_equity}.",
                 "metrics": "The IRR is cheap at {irr}."})
    monkeypatch.setattr("research.reader.providers.get_provider", lambda: ai)
    written = client.post(memos(deal_id, f"/{version}/sentences"))
    assert written.status_code == 200, written.text
    got = written.json()
    by = {s["section"]: s["source"] for s in got["sentences"]}
    assert by["summary"] == "ai" and by["metrics"] == "ai_refused" and got["written_by"] == "gemini-3-flash-preview"
    sent = ai.sent[0]
    for secret in ("Kestrel", "01234567", "GBP", deal_id, "alice") + tuple(str(x) for x in range(10)):
        assert secret not in sent, secret
    assert "Equity grows from " in text_of(got["blocks"]) and "cheap" not in text_of(got["blocks"])
    assert client.get(memos(deal_id, f"/{version}")).json()["sentences"] == got["sentences"]
    usage = client.get("/api/account/ai-usage").json()
    rows = [r for r in usage["days"] if r["kind"] == "memo"]
    assert rows and rows[0]["requests"] == 1 and rows[0]["input_tokens"] == 900
    for _ in range(4):
        assert client.post(memos(deal_id, f"/{version}/sentences")).status_code == 200
    assert client.post(memos(deal_id, f"/{version}/sentences")).status_code == 429


def test_the_word_and_pdf_files_are_the_memo_and_each_is_in_the_deals_history(account, no_ai):
    deal_id, version = saved_deal()
    shown = [f["text"] for f in figures(drafted(deal_id, version)["blocks"]) if f["text"] != "–"]
    word = client.get(memos(deal_id, f"/{version}/word"))
    assert word.status_code == 200 and word.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert 'filename="Kestrel_Holdings_Ltd_company_01234567_v%d_memo.docx"' % version \
        in word.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(word.content)) as z:
        document, styles = z.read("word/document.xml").decode(), z.read("word/styles.xml").decode()
    assert 'w:val="en-GB"' in styles
    pdf = client.get(memos(deal_id, f"/{version}/pdf"))
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    text = " ".join(" ".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf.content)).pages).split())
    for figure in shown:
        assert figure in document and figure in text, figure
    history = client.get(f"/api/deals/{deal_id}/history").json()["entries"]
    assert [e.get("export") for e in history if e["action"] == "exported"] == ["memo_pdf", "memo_word"]


def test_each_change_writes_one_audit_entry_and_none_for_a_repeat(account, no_ai):
    deal_id, version = saved_deal()
    drafted(deal_id, version)
    drafted(deal_id, version)                       # the same run again: nothing changed
    client.post(memos(deal_id, f"/{version}/sentences"))
    client.post(memos(deal_id, f"/{version}/sentences"))       # the same templates again
    assert client.delete(memos(deal_id, f"/{version}")).status_code == 204
    assert client.delete(memos(deal_id, f"/{version}")).status_code == 404
    history = client.get(f"/api/deals/{deal_id}/history").json()["entries"]
    actions = [(e["action"], e.get("version")) for e in history if e["action"].startswith("memo_")]
    assert actions == [("memo_deleted", version), ("memo_written", version), ("memo_drafted", version)]


def test_a_person_keeps_at_most_max_memos(account, no_ai, monkeypatch):
    monkeypatch.setattr("db.memos.MAX_MEMOS", 1)
    deal_id, version = saved_deal()
    drafted(deal_id, version)
    second = client.post(f"/api/deals/{deal_id}/versions", json={"label": "Second"}).json()["number"]
    # a version kept with nothing changed is the same number; change the deal so a new one is kept
    if second == version:
        client.put(f"/api/deals/{deal_id}/draft", json={"inputs": {**INPUTS, "hold": 6}, "settings": SETTINGS})
        second = client.post(f"/api/deals/{deal_id}/versions", json={"label": "Second"}).json()["number"]
    full = client.post(memos(deal_id), json={"version": second})
    assert full.status_code == 409 and full.json()["detail"] == "memos_full"
    assert client.get(memos(deal_id, f"/{second}")).status_code == 404
    assert [m["version"] for m in client.get(memos(deal_id)).json()["memos"]] == [version]
    # drafting the version it already has is still allowed
    drafted(deal_id, version)


def test_deleting_the_deal_deletes_its_memos(account, no_ai):
    deal_id, version = saved_deal()
    drafted(deal_id, version)
    assert client.delete(f"/api/deals/{deal_id}").status_code in (200, 204)
    from sqlalchemy import func, select

    from db.engine import connect
    from db.models import DealMemo
    with connect() as conn:
        assert conn.execute(select(func.count()).select_from(DealMemo)).scalar() == 0
