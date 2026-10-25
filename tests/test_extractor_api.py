"""A document read for a deal, through the API (PLAN.md 6.2): the job, who may read what, the public confirmation
on a free tier, the deal built only from accepted values, the limits, the AI's use counted, nothing logged.

Uploads go the whole road of PLAN.md 6.1 (tests/test_uploads.py's fixture); the reader replays recorded answers
(``AI_PROVIDER=replay``) or answers Gemini's own HTTP with them (``RecordedGemini``), so no test calls an AI.
"""
from __future__ import annotations

import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

from api.limits import USER_DOCUMENT_PAGES, Rule
from api.main import app
from ml import document_reader as dr
from ml import nlp_extractor as nx
from research.reader import providers
from tests.test_paid_switch import RecordedGemini
from tests.test_research_reader_api import run_reads
from tests.test_uploads import upload, uploads  # noqa: F401 -- the fixture

client = TestClient(app)
RECORDING = "de-mbb-2025"
RECORDED = json.loads((dr.EXTRACTIONS / "evaluation" / f"{RECORDING}.json").read_text(encoding="utf-8"))


def blank_pdf(pages: int) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=595, height=842)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


@pytest.fixture
def documents(uploads, monkeypatch):  # noqa: F811 -- the imported fixture
    monkeypatch.setenv("AI_PROVIDER", "replay")
    monkeypatch.setenv("FSE_EXTRACT_REPLAY_DEFAULT", RECORDING)
    for name in ("FSE_ENV", "AI_FREE_TIER", "FSE_PLAN", "GEMINI_MODELS_A", "GEMINI_MODELS_B"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(providers, "_status", [0.0, None, None])
    return uploads


def clean_pdf(queue, pages: int = RECORDED["document"]["pages"]) -> str:
    found = upload(queue, "annual-report.pdf", blank_pdf(pages))
    assert found["status"] == "clean", found
    return found["id"]


def start(file_id: str, **extra):
    return client.post("/api/documents/reads", json={"file_id": file_id, "public": True, **extra})


def read_done(queue, file_id: str) -> dict:
    started = start(file_id)
    assert started.status_code == 202, started.text
    run_reads(queue)
    done = client.get(f"/api/documents/reads/{started.json()['id']}").json()
    assert done["status"] == "succeeded", done
    return done


def test_a_clean_pdf_is_read_twice_in_a_job_and_every_value_has_its_page_and_quote(documents, job_queue):
    file_id = clean_pdf(job_queue)
    info = client.get("/api/documents/reader").json()
    assert (info["status"], info["provider"], info["confirm_public"], info["open"]) == ("on", "replay", True, True)
    assert info["max_pages"] == dr.MAX_PAGES and info["pages_left"] == USER_DOCUMENT_PAGES.limit
    done = read_done(job_queue, file_id)
    result = done["result"]
    assert result["file_id"] == file_id and result["language"] == "de"
    assert {label["key"]: label["value"] for label in result["labels"]} == {
        "currency": "EUR", "unit": "thousands", "accounting_standard": "ifrs", "fiscal_year_end": "2025-12-31",
        "scope": "consolidated"}
    assert len(result["figures"]) >= 10
    assert all(f["page"] and f["quote"] and f["id"] == f"figure:{f['key']}:{f['period_end']}" for f in result["figures"])
    assert result["reads"]["A"]["model"] != result["reads"]["B"]["model"]
    # a blank copy has no text layer: no quote could be looked up, so none is said to be missing
    assert result["document"]["text_pages"] == []
    assert not any(f["amber"] == "quote_not_on_page" for f in result["figures"])
    assert client.get("/api/documents/reader").json()["pages_left"] == USER_DOCUMENT_PAGES.limit - RECORDED["document"]["pages"]
    listed = client.get("/api/jobs").json()["jobs"]
    assert [j["kind"] for j in listed] == ["document.extract"]


def test_on_a_free_provider_the_caller_must_confirm_the_document_is_public(documents, job_queue, monkeypatch):
    file_id = clean_pdf(job_queue, 3)
    refused = client.post("/api/documents/reads", json={"file_id": file_id})
    assert (refused.status_code, refused.json()["detail"]) == (422, "confirm_public")
    assert client.get("/api/jobs").json()["jobs"] == []
    monkeypatch.setenv("AI_FREE_TIER", "0")
    assert client.get("/api/documents/reader").json()["confirm_public"] is False
    assert client.post("/api/documents/reads", json={"file_id": file_id}).status_code == 202


def test_nothing_reaches_a_deal_unless_accepted_and_only_from_the_reads_own_values(documents, job_queue):
    done = read_done(job_queue, clean_pdf(job_queue))
    deal = f"/api/documents/reads/{done['id']}/deal"
    nothing = client.post(deal, json={"accepted": []}).json()
    assert {k: v for k, v in nothing["inputs"].items() if v is not None} == {} and nothing["used"] == {}
    every = [i["id"] for i in done["result"]["labels"] + done["result"]["figures"]]
    whole = client.post(deal, json={"accepted": every}).json()
    expected = nx.deal_inputs(done["result"], set(every))
    assert {k: v for k, v in whole["inputs"].items() if v is not None} == pytest.approx(expected["inputs"])
    assert whole["inputs"]["currency"] == "EUR" and whole["inputs"]["unit"] == "thousands"
    assert whole["inputs"]["ebitda"] == pytest.approx(217835)            # as MBB prints it, in thousands
    # a margin without its money: currency and unit not accepted, so no EBITDA; the browser can't add a value
    ratios = [i for i in every if not i.startswith("label:")] + ["figure:ebitda:2025-12-31", "figure:made_up:2025-12-31"]
    partial = client.post(deal, json={"accepted": ratios}).json()
    assert partial["inputs"]["ebitda"] is None and partial["inputs"]["currency"] is None
    assert partial["inputs"]["gross_margin"] is None or partial["inputs"]["gross_margin"] > 0
    assert any(o["input"] == "ebitda" and "label:currency" in o["needs"] for o in partial["left_out"])
    refused = client.post(deal, json={"accepted": every, "values": {"ebitda": 1}})
    assert refused.status_code == 422                                     # a value can't be sent at all


def test_only_the_owner_sees_the_read_or_gets_a_deal_from_it(documents, job_queue):
    file_id = clean_pdf(job_queue, 3)
    job = start(file_id).json()
    documents("test:mallory")
    assert start(file_id).status_code == 404
    assert client.get(f"/api/documents/reads/{job['id']}").status_code == 404
    assert client.post(f"/api/documents/reads/{job['id']}/deal", json={"accepted": []}).status_code == 404
    documents("test:alice")
    waiting = client.post(f"/api/documents/reads/{job['id']}/deal", json={"accepted": []})
    assert (waiting.status_code, waiting.json()["detail"]) == (409, "read_not_done")
    record = job_queue.submit("test:alice", "montecarlo.run", {}, max_active_per_owner=4, max_queued=10)
    assert client.get(f"/api/documents/reads/{record.id}").status_code == 404


def test_the_reader_is_off_without_a_provider_and_closed_in_production_until_its_card_meets_the_bar(
        documents, job_queue, monkeypatch, tmp_path):
    file_id = clean_pdf(job_queue, 3)
    monkeypatch.setenv("AI_PROVIDER", "")
    assert client.get("/api/documents/reader").json()["status"] == "off"
    assert (start(file_id).status_code, start(file_id).json()["detail"]) == (503, "reader_off")
    monkeypatch.setenv("AI_PROVIDER", "replay")
    card = json.loads(dr.CARD.read_text())
    card["evaluation"]["sets"]["evaluation"]["meets_bar"] = False
    path = tmp_path / "extractor.json"
    path.write_text(json.dumps(card))
    monkeypatch.setattr(dr, "CARD", path)
    monkeypatch.setattr(dr, "deploy_environment", lambda: "production")
    closed = start(file_id)
    assert (closed.status_code, closed.json()["detail"]) == (503, "reader_not_measured")
    card["evaluation"]["sets"]["evaluation"]["meets_bar"] = True
    path.write_text(json.dumps(card))
    assert start(file_id).status_code == 202


def test_a_persons_document_pages_a_day_are_charged_when_the_job_counts_them(documents, job_queue, monkeypatch):
    file_id = clean_pdf(job_queue)
    monkeypatch.setattr("api.limits.USER_DOCUMENT_PAGES",
                        Rule(USER_DOCUMENT_PAGES.name, "document pages read", "user", RECORDED["document"]["pages"] + 1,
                             86_400))
    jobs = [start(file_id).json()["id"] for _ in range(2)]
    run_reads(job_queue)
    found = [client.get(f"/api/documents/reads/{j}").json() for j in jobs]
    assert [f["status"] for f in found] == ["succeeded", "failed"]
    assert (found[1]["error"], found[1]["error_status"]) == ("page_limit", 429)


def test_a_document_longer_than_the_reader_takes_is_refused_before_any_ai_is_asked(documents, job_queue):
    file_id = clean_pdf(job_queue, dr.MAX_PAGES + 1)
    job = start(file_id).json()
    run_reads(job_queue)
    found = client.get(f"/api/documents/reads/{job['id']}").json()
    assert (found["status"], found["error"]) == ("failed", "too_long")
    assert client.get("/api/account/ai-usage").json()["days"] == []


def test_geminis_own_path_reads_with_the_documents_prompt_and_both_answers_are_counted(documents, job_queue,
                                                                                      monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("FSE_ENV", "staging")
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-KEY-123")
    monkeypatch.setattr(providers.Gemini, "_spent", {})
    session = RecordedGemini(RECORDED)
    sent = []
    original = session.post

    def post(url, headers=None, json=None, **kwargs):  # noqa: A002
        sent.append([p.get("text") for p in json["contents"][0]["parts"] if "text" in p])
        return original(url, headers=headers, json=json, **kwargs)

    session.post = post
    monkeypatch.setattr("requests.Session", lambda: session)
    done = read_done(job_queue, clean_pdf(job_queue))
    texts = [t for prompt in sent for t in prompt]
    assert nx.prompt_text() in texts and nx.prompt_text(pictures=True) in texts
    assert all("quarterly results filing" not in t for t in texts)        # never the results reader's prompt
    assert done["result"]["reads"]["A"]["model"] == RECORDED["reads"]["A"]["model"]
    rows = client.get("/api/account/ai-usage").json()["days"]
    assert [(r["kind"], r["tier"], r["requests"], r["pages"]) for r in rows] == [
        ("extract", "free", 2, RECORDED["document"]["pages"])]
    usage_a, usage_b = RECORDED["reads"]["A"]["usage"], RECORDED["reads"]["B"]["usage"]
    assert rows[0]["input_tokens"] == usage_a["promptTokenCount"] + usage_b["promptTokenCount"]


def test_a_failed_read_still_counts_the_answer_it_got(documents, job_queue, monkeypatch):
    broken = json.loads(json.dumps(RECORDED))
    broken["reads"]["B"]["read"] = "not the object asked for"
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-KEY-123")
    monkeypatch.setattr(providers.Gemini, "_spent", {})
    monkeypatch.setattr("requests.Session", lambda: RecordedGemini(broken))
    file_id = clean_pdf(job_queue)
    job = start(file_id).json()
    run_reads(job_queue)
    found = client.get(f"/api/documents/reads/{job['id']}").json()
    assert (found["status"], found["error"]) == ("failed", "unreadable_answer")
    (row,) = client.get("/api/account/ai-usage").json()["days"]
    assert (row["kind"], row["requests"]) == ("extract", 2)


def test_nothing_from_a_document_reaches_a_log(documents, job_queue, caplog):
    caplog.set_level(logging.DEBUG)
    done = read_done(job_queue, clean_pdf(job_queue))
    logged = "\n".join(r.getMessage() + json.dumps(getattr(r, "__dict__", {}), default=str) for r in caplog.records)
    for found in done["result"]["figures"][:5]:
        assert found["quote"] not in logged
    assert done["result"]["company"] not in logged and "annual-report.pdf" not in logged


def test_a_read_no_model_answered_gives_its_document_pages_back(documents, job_queue, monkeypatch):
    monkeypatch.setenv("FSE_EXTRACT_REPLAY_DEFAULT", "")
    file_id = clean_pdf(job_queue, 3)
    before = client.get("/api/documents/reader").json()["pages_left"]
    job = start(file_id).json()
    run_reads(job_queue)
    found = client.get(f"/api/documents/reads/{job['id']}").json()
    assert (found["status"], found["error"]) == ("failed", "not_recorded")
    assert client.get("/api/documents/reader").json()["pages_left"] == before
