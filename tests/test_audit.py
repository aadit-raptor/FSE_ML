"""Audit history (PLAN.md 3.3).

Real rows in a real database through the real endpoints. Each action a user
takes on a deal or their settings writes exactly one entry, in the same
transaction as the action; an action that changes nothing, or fails, writes
none. Nothing in the API can change or remove an entry, and the API's own
database role can't either: it may only add and read them. Old edit entries
are merged one per deal and day by ``audit_compact``, which the nightly
job-maintenance task runs.
"""
import json
import uuid
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.exc import ProgrammingError

from api.auth import AuthUser, require_user
from api.main import app
from db import audit
from db import deals as store
from db import engine as db_engine
from db import health as db_health
from db.models import AuditEvent, utc_now
from jobs import scheduled
from tests.test_deals import INPUTS, PROFILE, SETTINGS
from tests.test_security import app_role, as_role  # noqa: F401 - the fixture

client = TestClient(app)

ACTUALS = {"currency": "EUR", "unit": "thousands", "years": [{"ebitda": 251.0}]}
SHEET = {"name": "Returns", "columns": ["Item", "Value"], "rows": [["IRR", 0.2]]}


@pytest.fixture
def sign_in():
    def as_user(subject: str):
        app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)
        assert client.post("/api/account", json=PROFILE).status_code == 200
    yield as_user


def rows() -> list[dict]:
    """Every entry in the table, oldest first, read as the owner."""
    with db_engine.connect() as conn:
        found = conn.execute(select(AuditEvent).order_by(AuditEvent.id)).all()
    return [dict(r._mapping) for r in found]


def history(deal_id) -> list[dict]:
    resp = client.get(f"/api/deals/{deal_id}/history")
    assert resp.status_code == 200, resp.text
    return resp.json()["entries"]


def account() -> list[dict]:
    resp = client.get("/api/account/history")
    assert resp.status_code == 200, resp.text
    return resp.json()["entries"]


class Exactly:
    """Run an action and check it wrote exactly one entry, the expected one."""

    def __init__(self):
        self.seen = len(rows())

    def one(self, action: str, deal_id=None, **detail) -> dict:
        now = rows()
        assert len(now) == self.seen + 1, [r["action"] for r in now[self.seen:]]
        self.seen = len(now)
        entry = now[-1]
        assert entry["action"] == action
        assert entry["deal_id"] == (uuid.UUID(deal_id) if deal_id else None)
        assert entry["count"] == 1
        for key, value in detail.items():
            assert entry["detail"].get(key) == value, (key, entry["detail"])
        return entry

    def none(self) -> None:
        now = rows()
        assert len(now) == self.seen, [r["action"] for r in now[self.seen:]]


# ---------------------------------------------------------------------------
# One entry per action
# ---------------------------------------------------------------------------
def test_each_action_on_a_deal_writes_exactly_one_entry(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    log = Exactly()

    deal = client.post("/api/deals", json={"name": "Alpine", "inputs": INPUTS, "settings": SETTINGS}).json()
    did = deal["id"]
    log.one("created", did)

    # Autosave: one entry per save that changes the deal, naming the fields
    url = f"/api/deals/{did}/draft"
    assert client.put(url, json={"inputs": {**INPUTS, "exit_mult": 12.75}, "settings": SETTINGS}).status_code == 200
    log.one("edited", did, fields=["exit_mult"])
    client.put(url, json={"inputs": {**INPUTS, "exit_mult": 12.75}, "settings": SETTINGS})
    log.none()
    client.put(url, json={"inputs": {**INPUTS, "exit_mult": 12.75, "hold": 7}, "settings": {"tx_fee_pct": 2.0}})
    log.one("edited", did, fields=["hold", "settings.tx_fee_pct"])

    # Keeping a version; again with nothing changed is nothing
    number = client.post(f"/api/deals/{did}/versions", json={}).json()["number"]
    log.one("versioned", did, version=number)
    client.post(f"/api/deals/{did}/versions", json={})
    log.none()

    # A restore is one action, though it may write two versions
    client.put(url, json={"inputs": INPUTS, "settings": SETTINGS})
    log.one("edited", did)
    assert client.post(f"/api/deals/{did}/versions/1/restore").status_code == 200
    log.one("restored", did, version=1)

    client.patch(f"/api/deals/{did}", json={"name": "Alpine II"})
    log.one("renamed", did)
    client.patch(f"/api/deals/{did}", json={"name": "Alpine II"})
    log.none()
    client.patch(f"/api/deals/{did}", json={"archived": True})
    log.one("archived", did)
    client.patch(f"/api/deals/{did}", json={"archived": True})
    log.none()
    client.patch(f"/api/deals/{did}", json={"archived": False})
    log.one("unarchived", did)

    assert client.put(f"/api/deals/{did}/actuals", json=ACTUALS).status_code == 200
    log.one("actuals_saved", did)
    client.delete(f"/api/deals/{did}/actuals")
    log.one("actuals_cleared", did)

    # Exports name the deal they come from; one that names none is not a deal's
    resp = client.post("/api/export/workbook", params={"deal_id": did}, json={"sheets": [SHEET]})
    assert resp.status_code == 200, resp.text
    log.one("exported", did, export="workbook")
    client.post("/api/export/workbook", json={"sheets": [SHEET]})
    log.none()
    resp = client.post("/api/export/montecarlo-sample", params={"deal_id": did},
                       json={"mc": {"n": 1000}, "seed": 3})
    assert resp.status_code == 200, resp.text
    log.one("exported", did, export="simulation_sample")

    # A duplicate is the new deal's creation, pointing back at its source
    copy = client.post(f"/api/deals/{did}/duplicate").json()
    log.one("created", copy["id"], source_deal=did)

    assert client.delete(f"/api/deals/{did}").status_code == 204
    log.one("deleted", did)

    assert [e["action"] for e in history(copy["id"])] == ["created"]
    # The deal is gone, so its own history is too; the account's keeps it
    assert client.get(f"/api/deals/{did}/history").status_code == 404
    actions = [e["action"] for e in account() if e["deal_id"] == did]
    assert actions == ["deleted", "exported", "exported", "actuals_cleared", "actuals_saved",
                       "unarchived", "archived", "renamed", "restored", "edited", "versioned",
                       "edited", "edited", "created"]
    deleted = account()[0]
    assert deleted["deal_name"] is None      # deleted for good: its name is gone too


def test_each_settings_change_writes_exactly_one_entry(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    log = Exactly()
    client.put("/api/account/settings", json={"settings": {"tx_fee_pct": 3.0, "mc_clip_irr": False}})
    entry = log.one("settings_changed", fields=["mc_clip_irr", "tx_fee_pct"])
    assert entry["deal_id"] is None
    client.put("/api/account/settings", json={"settings": {"tx_fee_pct": 3.0, "mc_clip_irr": False}})
    log.none()
    client.put("/api/account/settings", json={"settings": {"tx_fee_pct": 3.0}})
    log.one("settings_changed", fields=["mc_clip_irr"])
    assert [e["action"] for e in account()] == ["settings_changed", "settings_changed"]


def test_an_action_that_fails_writes_nothing(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    deal = client.post("/api/deals", json={"name": "Alpine", "inputs": INPUTS}).json()
    log = Exactly()
    bad = client.put(f"/api/deals/{deal['id']}/draft", json={"inputs": {**INPUTS, "hold": 40}})
    assert bad.status_code == 422
    assert client.post(f"/api/deals/{deal['id']}/versions/99/restore").status_code == 404
    assert client.patch(f"/api/deals/{deal['id']}", json={"name": ""}).status_code == 422
    assert client.put("/api/account/settings", json={"settings": {"no_such": 1}}).status_code == 422
    # Exporting "from" a deal that isn't the caller's is refused, file and all
    stranger = client.post("/api/export/workbook", params={"deal_id": str(uuid.uuid4())},
                           json={"sheets": [SHEET]})
    assert stranger.status_code == 404
    log.none()


def test_a_deal_saved_by_the_store_records_with_the_time_it_was_given(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    then = utc_now() - timedelta(days=3)
    deal = store.create_deal("user_anna", "Alpine", INPUTS, now=then)
    [entry] = rows()
    assert entry["occurred_at"] == then and entry["deal_id"] == deal.id


# ---------------------------------------------------------------------------
# Whose history, and what it holds
# ---------------------------------------------------------------------------
def test_history_is_newest_first_and_says_what_happened(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    did = client.post("/api/deals", json={"name": "Alpine", "inputs": INPUTS}).json()["id"]
    client.put(f"/api/deals/{did}/draft", json={"inputs": {**INPUTS, "growth": 9.0}})
    client.post(f"/api/deals/{did}/versions", json={"label": "Board case"})
    entries = history(did)
    assert [e["action"] for e in entries] == ["versioned", "edited", "created"]
    assert entries[0]["version"] == 2 and entries[1]["fields"] == ["growth"]
    assert all(e["count"] == 1 and e["until"] is None and e["deal_id"] == did for e in entries)
    # The account's view names the deal as it is called now
    assert {e["deal_name"] for e in account()} == {"Alpine"}


def test_users_cant_read_each_others_history(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    did = client.post("/api/deals", json={"name": "Alpine", "inputs": INPUTS}).json()["id"]
    sign_in("user_ben")
    assert client.get(f"/api/deals/{did}/history").status_code == 404
    assert client.get(f"/api/deals/{uuid.uuid4()}/history").status_code == 404
    assert account() == []
    # Nor write into it by exporting "from" it
    assert client.post("/api/export/workbook", params={"deal_id": did},
                       json={"sheets": [SHEET]}).status_code == 404
    sign_in("user_anna")
    assert [e["action"] for e in history(did)] == ["created"]


def test_an_entry_holds_no_figures_names_or_labels(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    did = client.post("/api/deals", json={"name": "Project Zebedee", "inputs": INPUTS}).json()["id"]
    client.put(f"/api/deals/{did}/draft", json={"inputs": {**INPUTS, "exit_mult": 12.75}})
    client.post(f"/api/deals/{did}/versions", json={"label": "Quixotic label"})
    client.patch(f"/api/deals/{did}", json={"name": "Project Yarrow"})
    client.put("/api/account/settings", json={"settings": {"tx_fee_pct": 3.25}})
    stored = json.dumps([r["detail"] for r in rows()])
    for secret in ("Zebedee", "Yarrow", "Quixotic", "12.75", "3.25", "240"):
        assert secret not in stored


def test_an_entry_is_small(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    did = client.post("/api/deals", json={"name": "Alpine", "inputs": INPUTS}).json()["id"]
    every_field = {k: (v + 1 if isinstance(v, float) else v) for k, v in INPUTS.items()}
    client.put(f"/api/deals/{did}/draft", json={"inputs": every_field})
    with db_engine.connect() as conn:
        sizes = conn.execute(text("SELECT pg_column_size(a.*) FROM audit_events a")).scalars().all()
    assert max(sizes) < 600, sizes          # an edit naming every money field
    assert min(sizes) < 120, sizes          # a plain entry


# ---------------------------------------------------------------------------
# Append-only
# ---------------------------------------------------------------------------
def test_the_api_has_no_way_to_change_an_entry():
    paths = app.openapi()["paths"]
    history_paths = [p for p in paths if p.endswith("/history")]
    assert sorted(history_paths) == ["/api/account/history", "/api/deals/{deal_id}/history"]
    for path in history_paths:
        assert set(paths[path]) == {"get"}, path
    for path, methods in paths.items():
        if "audit" in path or "history" in path:
            assert set(methods) == {"get"}, path


@pytest.mark.parametrize("method", ["put", "patch", "delete", "post"])
def test_writing_to_a_history_is_refused(fresh_db, sign_in, method):  # noqa: ARG001
    sign_in("user_anna")
    did = client.post("/api/deals", json={"name": "Alpine", "inputs": INPUTS}).json()["id"]
    for url in (f"/api/deals/{did}/history", "/api/account/history"):
        assert getattr(client, method)(url).status_code == 405
    assert len(rows()) == 1


@pytest.mark.parametrize("sql", [
    "UPDATE audit_events SET action = 'created'",
    "DELETE FROM audit_events",
    "TRUNCATE audit_events",
])
def test_the_apis_database_role_cant_change_or_remove_entries(app_role, sql):  # noqa: F811
    with pytest.raises(ProgrammingError, match="permission denied"):
        as_role(app_role, sql)


def test_the_apis_database_role_can_add_read_and_compact_entries(app_role, sign_in):  # noqa: F811
    sign_in("user_anna")
    did = client.post("/api/deals", json={"name": "Alpine", "inputs": INPUTS}).json()["id"]
    assert [e["action"] for e in history(did)] == ["created"]
    assert as_role(app_role, "SELECT audit_compact(now())").scalar() == 0


def test_the_health_check_flags_a_role_that_could_rewrite_the_log(app_role, fresh_db):  # noqa: F811
    assert client.get("/api/health/database").json()["role"] == {"status": "restricted", "privileges": []}
    as_role(fresh_db, "GRANT UPDATE ON audit_events TO fse_app")
    db_health.reset()
    role = client.get("/api/health/database").json()["role"]
    assert role == {"status": "privileged", "privileges": ["audit_log_writable"]}


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------
def test_old_edits_are_merged_one_per_deal_and_day(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    now = utc_now().replace(hour=12, minute=0, second=0, microsecond=0)
    old_day = now - timedelta(days=10)
    deal = store.create_deal("user_anna", "Alpine", INPUTS, now=old_day)
    other = store.create_deal("user_anna", "Other", INPUTS, now=old_day)

    def edit(deal_id, at, **change):
        current = store.get_deal("user_anna", deal_id)
        store.save_draft("user_anna", deal_id, {**current.inputs, **change}, current.settings, now=at)

    # Five edits on one old day, two on the next, and one of the other deal's
    for minute, change in enumerate([{"growth": 1.0}, {"growth": 2.0}, {"hold": 4},
                                     {"growth": 3.0}, {"exit_mult": 9.0}]):
        edit(deal.id, old_day + timedelta(minutes=minute), **change)
    store.save_version("user_anna", deal.id, now=old_day + timedelta(minutes=10))
    edit(deal.id, old_day + timedelta(days=1), growth=4.0)
    edit(deal.id, old_day + timedelta(days=1, hours=2), opex=18.0)
    edit(other.id, old_day + timedelta(minutes=3), hold=3)
    # Recent edits stay one entry each
    edit(deal.id, now - timedelta(days=1), growth=5.0)
    edit(deal.id, now - timedelta(days=1, minutes=-1), growth=6.0)
    before = rows()

    summary = scheduled.job_maintenance()
    assert summary["audit_rows_compacted"] == 5     # 5 -> 1 and 2 -> 1; singles stay

    after = rows()
    assert len(after) == len(before) - 5
    edits = sorted((r for r in after if r["action"] == "edited" and r["deal_id"] == deal.id),
                   key=lambda r: r["occurred_at"])
    first, second, *recent = edits
    assert first["count"] == 5 and first["detail"]["fields"] == ["exit_mult", "growth", "hold"]
    assert first["occurred_at"] == old_day and first["last_at"] == old_day + timedelta(minutes=4)
    assert second["count"] == 2 and second["detail"]["fields"] == ["growth", "opex"]
    assert second["last_at"] == old_day + timedelta(days=1, hours=2)
    assert [r["count"] for r in recent] == [1, 1] and all(r["last_at"] is None for r in recent)
    # Nothing but edits merges, and no action is lost from the count
    assert [r["action"] for r in after if r["action"] != "edited"] == \
        [r["action"] for r in before if r["action"] != "edited"]
    assert sum(r["count"] for r in after) == sum(r["count"] for r in before)

    # Merged entries read back with their span; compacting again changes nothing
    entry = next(e for e in history(deal.id) if e["count"] == 5)
    assert entry["until"] is not None and entry["fields"] == ["exit_mult", "growth", "hold"]
    assert audit.compact() == 0 and rows() == after


def test_compaction_keeps_the_newest_week(fresh_db, sign_in):  # noqa: ARG001
    assert audit.COMPACT_AFTER_DAYS == 7
    sign_in("user_anna")
    deal = store.create_deal("user_anna", "Alpine", INPUTS)
    at = utc_now() - timedelta(days=6, hours=23)
    for growth in (1.0, 2.0, 3.0):
        store.save_draft("user_anna", deal.id, {**INPUTS, "growth": growth}, {}, now=at)
    assert audit.compact() == 0
    with db_engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(AuditEvent)).scalar() == 4
