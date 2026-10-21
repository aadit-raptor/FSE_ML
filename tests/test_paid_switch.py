"""Ready for the paid switch (PLAN.md R9): what the AI costs per person, the limits per plan, and the paid
provider's path on recorded answers.

- **Prices** (research/ai_costs.py): a request's cost from its token counts at the provider's paid price list,
  which changes on dated days; every model the reader or the writer may ask is priced.
- **Plans** (api/plans.py): a person's limits come from their plan (``FSE_PLAN``), the service's from the AI's
  tier (``AI_FREE_TIER``); the free plan is exactly the numbers in force before R9.
- **Usage** (db/ai_usage.py): pages read, requests, tokens and cost, per person per day; another account's is
  never shown; nothing from a document.
- **The switch**: ``AI_PROVIDER`` changed on staging changes the provider and nothing else, for the reader and
  for the note writer, with Gemini's real code path answered by recorded HTTP replies.
"""
from __future__ import annotations

import datetime
import io
import json
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from api import plans
from api.limits import READS_ALL, USER_READ_PAGES, USER_READS, USER_WRITES, WRITES_ALL, effective_limit
from api.main import app
from research import ai_costs
from research.reader import providers
from tests.test_research_reader_api import run_reads
from tests.test_uploads import upload, uploads  # noqa: F401 -- the fixture

client = TestClient(app)
RECORDING = "max-healthcare_FY27-Q1"
RECORDED = json.loads((providers.RECORDINGS / "held_out" / f"{RECORDING}.json").read_text())
DAY = datetime.date(2026, 10, 20)


# ------------------------------------------------------------------------------------------------------ prices
def test_a_requests_cost_is_its_tokens_at_the_paid_price_of_its_day():
    usage = {"promptTokenCount": 1_000_000, "candidatesTokenCount": 200_000, "thoughtsTokenCount": 800_000}
    # 0.75 a million in, 3.75 a million out (thinking is output) until the end of 2026
    assert ai_costs.cost("gemini-3.8-flash", usage, DAY) == Decimal("4.500000")
    # the list doubles on 2027-01-01
    assert ai_costs.cost("gemini-3.8-flash", usage, datetime.date(2027, 1, 1)) == Decimal("9.000000")
    assert ai_costs.cost("gemini-3-flash-preview", usage, DAY) == Decimal("3.500000")
    assert ai_costs.cost("a-model-nobody-priced", usage, DAY) is None
    assert ai_costs.tokens({"promptTokenCount": 7, "candidatesTokenCount": 2, "thoughtsTokenCount": 1}) == (7, 3)
    assert ai_costs.tokens({}) == (0, 0)


def test_every_model_the_reader_or_writer_may_ask_is_priced():
    for model in {*providers.MODELS_A, *providers.MODELS_B}:
        assert ai_costs.cost(model, {"promptTokenCount": 1}, DAY) is not None, model


def test_a_recorded_filing_costs_what_its_recorded_tokens_say():
    a, b = RECORDED["reads"]["A"], RECORDED["reads"]["B"]
    expected = ai_costs.cost(a["model"], a["usage"], DAY) + ai_costs.cost(b["model"], b["usage"], DAY)
    assert Decimal("0.01") < expected < Decimal("0.2")


# ------------------------------------------------------------------------------------------------------- plans
def test_the_free_plan_and_the_free_tier_are_the_limits_in_force_before_r9(monkeypatch):
    monkeypatch.delenv("FSE_PLAN", raising=False)
    monkeypatch.delenv("AI_FREE_TIER", raising=False)
    for rule in (USER_READS, USER_READ_PAGES, USER_WRITES, READS_ALL, WRITES_ALL):
        assert effective_limit(rule, "test:alice") == rule.limit
    assert (USER_READS.limit, USER_READ_PAGES.limit, USER_WRITES.limit) == (10, 250, 5)
    assert (READS_ALL.limit, WRITES_ALL.limit) == (30, 10)


def test_a_persons_limits_follow_their_plan_and_the_services_follow_the_ais_tier(monkeypatch):
    monkeypatch.setenv("FSE_PLAN", "research")
    research = plans.PLANS["research"]
    assert effective_limit(USER_READS, "test:alice") == research.reads_per_day > USER_READS.limit
    assert effective_limit(USER_READ_PAGES, "test:alice") == research.pages_per_day > USER_READ_PAGES.limit
    assert effective_limit(USER_WRITES, "test:alice") == research.writes_per_day > USER_WRITES.limit
    assert effective_limit(READS_ALL, "all") == READS_ALL.limit          # the free tier's allowance still binds
    monkeypatch.setenv("AI_FREE_TIER", "0")
    billed = plans.TIERS["billed"]
    assert effective_limit(READS_ALL, "all") == billed.reads_per_day > READS_ALL.limit
    assert effective_limit(WRITES_ALL, "all") == billed.writes_per_day > WRITES_ALL.limit
    monkeypatch.setenv("FSE_PLAN", "no-such-plan")                      # a typo never lifts a limit
    assert effective_limit(USER_READS, "test:alice") == USER_READS.limit


# ------------------------------------------------------------------------------------- recorded HTTP for Gemini
class Reply:
    def __init__(self, status, body):
        self.status_code, self._body, self.headers = status, body, {}
        self.content = json.dumps(body).encode()

    def json(self):
        return self._body


class RecordedGemini:
    """Google's API as it answered when the filing was recorded: the recorded model answers with its recorded read
    and token counts, every other model says "high demand" (which spends nothing), so Gemini's own code moves on to
    the model that read it. Text-only requests are the note writer's: answered with ``writing``."""

    def __init__(self, recording, writing=None):
        self.reads = recording["reads"] if recording else {}
        self.writing, self.calls, self.prompts = writing, [], []

    def post(self, url, headers=None, json=None, timeout=None, **_):  # noqa: A002
        self.calls.append((url, headers))
        model = url.rsplit("/", 1)[1].split(":")[0]
        parts = json["contents"][0]["parts"]
        kinds = {(p.get("inline_data") or {}).get("mime_type") for p in parts} - {None}
        if not kinds:
            self.prompts.append(parts[0]["text"])
            return self._answer(self.writing, {"promptTokenCount": 900, "candidatesTokenCount": 120})
        found = self.reads.get("A" if "application/pdf" in kinds else "B")
        if not found or found["model"] != model:
            return Reply(503, {"error": {"status": "UNAVAILABLE", "message": "high demand"}})
        return self._answer(found["read"], found["usage"])

    @staticmethod
    def _answer(read, usage):
        return Reply(200, {"candidates": [{"content": {"parts": [{"text": json.dumps(read)}]}}],
                           "usageMetadata": usage})

    def get(self, url, params=None, headers=None, timeout=None):
        listed = {*providers.MODELS_A, *providers.MODELS_B}
        return Reply(200, {"models": [{"name": f"models/{m}"} for m in sorted(listed)]})

    def delete(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))


@pytest.fixture
def staging(uploads, monkeypatch):  # noqa: F811 -- the imported fixture
    monkeypatch.setenv("FSE_ENV", "staging")
    monkeypatch.setenv("FSE_READER_REPLAY_DEFAULT", RECORDING)
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-KEY-123")
    for name in ("GEMINI_MODELS_A", "GEMINI_MODELS_B", "GEMINI_MODELS_WRITE", "READER_HIGH_RESOLUTION",
                 "AI_FREE_TIER", "FSE_PLAN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(providers.Gemini, "_spent", {})
    monkeypatch.setattr(providers, "_status", [0.0, None, None])
    session = RecordedGemini(RECORDED)
    session.sign_in = uploads
    monkeypatch.setattr("requests.Session", lambda: session)
    return session


def filing_pdf() -> bytes:
    """A blank PDF as long as the recorded filing, so read B's pages are pages of the file."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(RECORDED["document"]["pages"]):
        writer.add_blank_page(width=595, height=842)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def read_once(queue, *, public=True) -> tuple[dict, dict]:
    found = upload(queue, "results.pdf", filing_pdf())
    assert found["status"] == "clean", found
    started = client.post("/api/research/reads", json={"file_id": found["id"], "public": public,
                                                       "quarter_end": "2026-06-30"})
    assert started.status_code == 202, started.text
    run_reads(queue)
    done = client.get(f"/api/research/reads/{started.json()['id']}").json()
    assert done["status"] == "succeeded", done
    return done["result"], client.get("/api/research/reader").json()


def without(result: dict, *keys) -> dict:
    out = json.loads(json.dumps(result))
    for key in keys:
        out.pop(key, None)
    for read in (out.get("reads") or {}).values():
        if isinstance(read, dict):
            read.pop("seconds", None)
    out.get("reads", {}).pop("seconds", None)
    return out


def usage_today() -> dict:
    answer = client.get("/api/account/ai-usage")
    assert answer.status_code == 200, answer.text
    return answer.json()


def limits() -> dict:
    return {rule.name: effective_limit(rule, "test:alice")
            for rule in (USER_READS, USER_READ_PAGES, USER_WRITES, READS_ALL, WRITES_ALL)}


# ----------------------------------------------------------------------------------------------------- the switch
def test_switching_ai_provider_on_staging_changes_the_provider_and_nothing_else(staging, job_queue, monkeypatch):
    seen = {}
    for choice in ("replay", "gemini"):
        monkeypatch.setenv("AI_PROVIDER", choice)
        result, info = read_once(job_queue)
        seen[choice] = (without(result, "file_id"), info, limits())
    (replay, replay_info, replay_limits), (gemini, gemini_info, gemini_limits) = seen["replay"], seen["gemini"]
    assert (replay_info.pop("provider"), gemini_info.pop("provider")) == ("replay", "gemini")
    replay_info.pop("pages_left"), gemini_info.pop("pages_left")     # the first read spent some
    assert replay_info == gemini_info and replay_limits == gemini_limits
    assert replay == gemini                                          # every figure, check, amber reason and model
    assert gemini["reads"]["A"]["model"] == RECORDED["reads"]["A"]["model"]
    # Gemini was really asked, with the key in a header and never in an address
    assert any(":generateContent" in url for url, _ in staging.calls)
    assert all("SECRET" not in url and headers["x-goog-api-key"] == "SECRET-KEY-123" for url, headers in staging.calls)
    # and both reads were counted the same way
    rows = usage_today()["days"]
    assert len(rows) == 1 and rows[0]["kind"] == "read" and rows[0]["requests"] == 4
    assert rows[0]["pages"] == 2 * RECORDED["document"]["pages"]


def test_the_paid_tier_reads_without_the_public_confirmation_and_counts_what_it_costs(staging, job_queue,
                                                                                     monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("AI_FREE_TIER", "0")
    result, info = read_once(job_queue, public=False)
    assert info["confirm_public"] is False and info["free_tier"] is False
    assert info["reads_per_day"] == min(USER_READS.limit, plans.TIERS["billed"].reads_per_day)
    assert result["summary"]["figures"] > 50
    found = usage_today()
    (row,) = found["days"]
    a, b = RECORDED["reads"]["A"], RECORDED["reads"]["B"]
    expected = ai_costs.cost(a["model"], a["usage"], datetime.date.today()) + \
        ai_costs.cost(b["model"], b["usage"], datetime.date.today())
    assert (row["kind"], row["tier"], row["requests"], row["pages"]) == ("read", "billed", 2, 15)
    assert Decimal(str(row["cost"])) == expected
    assert row["input_tokens"] == a["usage"]["promptTokenCount"] + b["usage"]["promptTokenCount"]
    assert found["money"]["currency"] == "USD" and found["plan"]["name"] == "free"
    assert Decimal(str(found["total_cost"])) == expected
    # another account sees none of it
    staging.sign_in("test:mallory")
    assert usage_today()["days"] == []


def test_on_the_free_tier_the_cost_is_what_it_would_cost_and_says_so(staging, job_queue, monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "replay")
    read_once(job_queue)
    (row,) = usage_today()["days"]
    assert row["tier"] == "free" and Decimal(str(row["cost"])) > 0


def test_switching_ai_provider_changes_nothing_in_a_results_note(staging, monkeypatch, tmp_path):
    from tests.test_research_saved_api import start

    company_id, _ = start()
    staging.writing = {"outlook": "Revenue moves from {rev_ly} to {rev_cy}."}
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    path = f"/api/research/companies/{company_id}/report/sentences"
    by_gemini = client.post(path)
    assert by_gemini.status_code == 200, by_gemini.text
    # the same answer, recorded for replay by the prompt Gemini was sent
    monkeypatch.setattr(providers, "WRITINGS", tmp_path)
    (prompt,) = staging.prompts
    model = by_gemini.json()["written_by"]
    (tmp_path / f"{providers.sha256(prompt.encode())}.json").write_text(json.dumps(
        {"answer": staging.writing, "model": model, "usage": {"promptTokenCount": 900, "candidatesTokenCount": 120}}))
    monkeypatch.setenv("AI_PROVIDER", "replay")
    by_replay = client.post(path)
    assert by_replay.status_code == 200, by_replay.text
    assert by_gemini.json() == by_replay.json()
    assert {s["source"] for s in by_gemini.json()["sentences"]} >= {"ai"}
    (row,) = usage_today()["days"]
    assert (row["kind"], row["requests"], row["pages"], row["input_tokens"]) == ("write", 2, 0, 1800)


def test_the_note_writer_counts_nothing_when_no_ai_is_set_up(staging, monkeypatch):
    from tests.test_research_saved_api import start

    company_id, _ = start()
    monkeypatch.setenv("AI_PROVIDER", "")
    assert client.post(f"/api/research/companies/{company_id}/report/sentences").status_code == 200
    assert usage_today()["days"] == []


# -------------------------------------------------------------------------------------------- storage and the night
def test_usage_is_one_row_a_person_day_kind_and_tier_and_the_night_sums_and_sweeps_it(uploads):  # noqa: F811
    from db import ai_usage

    answer = providers.Answer({}, "gemini-3.8-flash", "r1",
                              {"promptTokenCount": 1000, "candidatesTokenCount": 100, "thoughtsTokenCount": 100})
    old = datetime.date.today() - datetime.timedelta(days=ai_usage.KEEP_DAYS + 1)
    ai_usage.record("test:alice", "read", "free", pages=3, answers=[answer], day=old)
    ai_usage.record("test:alice", "read", "free", pages=12, answers=[answer, answer])
    ai_usage.record("test:alice", "read", "free", pages=0, answers=[])
    ai_usage.record("test:nobody-signed-up", "read", "free", pages=5, answers=[answer])   # no account: nothing kept
    (row,) = [r for r in usage_today()["days"] if r["day"] == datetime.date.today().isoformat()]
    assert (row["requests"], row["pages"], row["input_tokens"], row["output_tokens"]) == (2, 12, 2000, 400)
    assert Decimal(str(row["cost"])) == 2 * ai_costs.cost("gemini-3.8-flash", answer.usage, datetime.date.today())
    totals = ai_usage.totals(datetime.date.today())
    assert (totals["requests"], totals["pages"], totals["people"]) == (2, 12, 1)
    assert ai_usage.sweep() == 1
    from jobs.scheduled import TASKS

    summary = TASKS["job-maintenance"].run()
    assert {"ai_usage_rows_deleted", "ai_requests_yesterday", "ai_pages_yesterday", "ai_cost_usd_yesterday",
            "ai_people_yesterday"} <= set(summary)
    assert all(not isinstance(v, (list, dict)) for v in summary.values())
