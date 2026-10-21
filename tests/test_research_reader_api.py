"""The results reader through the API (PLAN.md R1): who may read what, the
job, the limits, and that nothing from a document is logged.

Uploads go the whole road of PLAN.md 6.1 (tests/test_uploads.py's fixture);
the reader replays recorded answers (``AI_PROVIDER=replay``), so no test
calls an AI.
"""
from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from api.limits import READS_ALL, USER_READ_PAGES, USER_READS, Rule
from api.main import app
from jobs.runner import Runner
from research.reader import gate, providers
from tests import upload_samples as s
from tests.test_uploads import begin, run_scans, upload, uploads  # noqa: F401 -- the fixture

client = TestClient(app)
RECORDED = "eternal_FY27-Q1"


@pytest.fixture
def reader(uploads, monkeypatch):  # noqa: F811 -- the imported fixture
    monkeypatch.setenv("AI_PROVIDER", "replay")
    monkeypatch.setenv("FSE_READER_REPLAY_DEFAULT", RECORDED)
    monkeypatch.delenv("FSE_ENV", raising=False)
    monkeypatch.setattr(providers, "_status", [0.0, None, None])
    return uploads


def clean_pdf(queue) -> str:
    found = upload(queue, "results.pdf", s.pdf())
    assert found["status"] == "clean"
    return found["id"]


def start(file_id: str, **extra):
    return client.post("/api/research/reads", json={"file_id": file_id, "public": True, **extra})


def run_reads(queue) -> None:
    runner = Runner(lambda: queue)
    while runner.run_next() is not None:
        pass


def test_a_clean_results_pdf_is_read_twice_and_checked_in_a_job(reader, job_queue):
    file_id = clean_pdf(job_queue)
    info = client.get("/api/research/reader").json()
    assert (info["status"], info["provider"], info["confirm_public"], info["open"]) == ("on", "replay", True, True)
    started = start(file_id, quarter_end="2026-06-30")
    assert started.status_code == 202, started.text
    job = started.json()
    assert job["status"] == "queued" and started.headers["Location"].endswith(job["id"])
    run_reads(job_queue)
    done = client.get(f"/api/research/reads/{job['id']}").json()
    assert done["status"] == "succeeded", done
    result = done["result"]
    assert result["file_id"] == file_id and result["quarter_source"] == "given"
    assert result["unit"] in ("crores", "lakhs", "millions") and result["summary"]["figures"] > 100
    assert result["reads"]["A"]["model"] != result["reads"]["B"]["model"]
    assert result["document"]["pages"] == 1
    # the same job is listed among the caller's jobs, without its result
    listed = client.get("/api/jobs").json()["jobs"]
    assert [j["kind"] for j in listed] == ["research.read"] and listed[0]["result"] is None


def test_a_free_provider_needs_the_caller_to_confirm_the_file_is_public(reader, job_queue):
    file_id = clean_pdf(job_queue)
    refused = client.post("/api/research/reads", json={"file_id": file_id})
    assert (refused.status_code, refused.json()["detail"]) == (422, "confirm_public")
    assert client.get("/api/jobs").json()["jobs"] == []


def test_a_paid_provider_asks_for_no_confirmation(reader, job_queue, monkeypatch):
    monkeypatch.setenv("AI_FREE_TIER", "0")
    file_id = clean_pdf(job_queue)
    assert client.get("/api/research/reader").json()["confirm_public"] is False
    assert client.post("/api/research/reads", json={"file_id": file_id}).status_code == 202


def test_only_the_owners_clean_pdf_can_be_read_and_only_the_owner_sees_the_read(reader, job_queue):
    file_id = clean_pdf(job_queue)
    job = start(file_id).json()
    reader("test:mallory")
    assert start(file_id).json()["detail"] == "no_file"
    assert start(file_id).status_code == 404
    assert client.get(f"/api/research/reads/{job['id']}").status_code == 404
    reader("test:alice")
    waiting = begin("waiting.pdf", 10)
    refused = start(waiting["id"])
    assert (refused.status_code, refused.json()["detail"]) == (409, "not_clean")
    document = upload(job_queue, "notes.docx", s.open_xml("docx"))
    assert document["status"] == "clean"
    assert start(document["id"]).json()["detail"] == "not_pdf"


def test_another_kind_of_job_is_not_a_read(reader, job_queue):
    record = job_queue.submit("test:alice", "montecarlo.run", {}, max_active_per_owner=4, max_queued=10)
    assert client.get(f"/api/research/reads/{record.id}").status_code == 404


def test_the_reader_is_off_without_a_provider_and_closed_in_production_until_measured(reader, job_queue,
                                                                                    monkeypatch, tmp_path):
    file_id = clean_pdf(job_queue)
    monkeypatch.setenv("AI_PROVIDER", "")
    assert client.get("/api/research/reader").json()["status"] == "off"
    assert start(file_id).json()["detail"] == "reader_off"
    assert start(file_id).status_code == 503
    monkeypatch.setenv("AI_PROVIDER", "replay")
    unmeasured = json.loads(gate.CARD.read_text())
    unmeasured["evaluation"]["sets"]["held_out"]["meets_bar"] = None
    card = tmp_path / "reader.json"
    card.write_text(json.dumps(unmeasured))
    monkeypatch.setattr(gate, "CARD", card)
    monkeypatch.setattr("research.reader.service.deploy_environment", lambda: "production")
    closed = start(file_id)
    assert (closed.status_code, closed.json()["detail"]) == (503, "reader_not_measured")


def test_a_persons_daily_pages_are_counted_when_the_job_reads_the_file(reader, job_queue, monkeypatch):
    file_id = clean_pdf(job_queue)
    monkeypatch.setattr("api.limits.USER_READ_PAGES", Rule(USER_READ_PAGES.name, "pages read", "user", 2, 86_400))
    jobs = [start(file_id).json()["id"] for _ in range(3)]
    run_reads(job_queue)
    found = [client.get(f"/api/research/reads/{j}").json() for j in jobs]
    assert [f["status"] for f in found] == ["succeeded", "succeeded", "failed"]
    assert (found[2]["error"], found[2]["error_status"]) == ("page_limit", 429)
    assert client.get("/api/research/reader").json()["pages_left"] == 0


def test_reads_a_day_are_limited_per_person_and_for_everyone(reader, job_queue, monkeypatch):
    file_id = clean_pdf(job_queue)
    monkeypatch.setattr("api.limits.USER_READS", Rule(USER_READS.name, "results reads", "user", 1, 86_400))
    assert start(file_id).status_code == 202
    refused = start(file_id)
    assert refused.status_code == 429 and "your account" in refused.json()["detail"]
    reader("test:bob")
    other = clean_pdf(job_queue)
    monkeypatch.setattr("api.limits.READS_ALL", Rule(READS_ALL.name, "results reads", "service", 1, 86_400))
    shared = start(other)
    assert shared.status_code == 429 and "everyone together" in shared.json()["detail"]


def test_a_document_never_reaches_the_log(reader, job_queue, caplog):
    file_id = clean_pdf(job_queue)
    job = start(file_id).json()
    with caplog.at_level(logging.DEBUG):
        run_reads(job_queue)
    result = client.get(f"/api/research/reads/{job['id']}").json()["result"]
    logged = "\n".join(r.getMessage() + json.dumps(getattr(r, "__dict__", {}), default=str) for r in caplog.records)
    assert "reader_read" in logged
    labels = {row["label"] for t in result["tables"] for row in t["rows"] if len(row["label"]) > 6}
    figures = {str(c["value"]) for t in result["tables"] for row in t["rows"] for c in row["cells"]
               if c["value"] and abs(c["value"]) > 1000}
    assert labels and not any(label in logged for label in labels)
    assert figures and not any(figure in logged for figure in figures)
    assert result["company"] not in logged and "results.pdf" not in logged


def test_a_read_that_the_provider_cannot_answer_fails_with_a_code(reader, job_queue, monkeypatch):
    monkeypatch.setenv("FSE_READER_REPLAY_DEFAULT", "")
    file_id = clean_pdf(job_queue)
    job = start(file_id).json()
    run_reads(job_queue)
    found = client.get(f"/api/research/reads/{job['id']}").json()
    assert (found["status"], found["error"], found["error_status"]) == ("failed", "not_recorded", 404)


def test_the_health_check_says_which_provider_and_never_the_key(reader, monkeypatch, signed_out):
    assert client.get("/api/health/reader").json()["provider"] == "replay"
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-not-a-real-key")
    monkeypatch.setattr(providers, "_status", [0.0, None, None])

    class Refused:
        status_code = 400

        def json(self):
            return {}

    class Session:
        def get(self, *args, **kwargs):
            return Refused()

    monkeypatch.setattr("requests.Session", Session)
    answer = client.get("/api/health/reader")
    assert answer.status_code == 503 and answer.json()["key"] == "refused"
    assert "AIza" not in answer.text
