"""Saved deals, versions and account settings (PLAN.md 1.5).

Real rows in a real database (``fresh_db``) through the real endpoints, and
real model runs: "reopens identically" and "restores exactly" are checked by
running the deal model on what comes back, not by comparing JSON alone.
"""
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

from api.auth import AuthUser, require_user
from api.main import app
from db import deals as store
from db import engine as db_engine
from db.models import DealVersion, utc_now

client = TestClient(app)

PROFILE = {"country": "DE", "preferred_currency": "EUR", "locale": "de-DE",
           "time_zone": "Europe/Berlin"}
# A deal nothing like the defaults, with settings that move its IRR
INPUTS = {"ebitda": 240.0, "entry_mult": 9.0, "exit_mult": 10.5, "hold": 6, "growth": 7.5,
          "gross_margin": 44.0, "opex": 17.0, "tax": 29.0, "da": 3.5, "debt_pct": 55.0,
          "senior_pct": 75.0, "base_rate": 5.25, "mezz_spread": 4.5, "capex": 3.0, "nwc": 0.5,
          "mincash": 15.0, "wsp_mode": False, "ar_days": 45.0, "inv_days": 30.0, "ap_days": 60.0,
          "currency": "EUR", "unit": "thousands",
          "fiscal_year_end_month": 12, "first_fiscal_year": None}
# Every field of DealInputsIn, as the API answers: the fiscal year labels
# (PLAN.md 2.3a) are stored only when set but always come back
SETTINGS = {"tx_fee_pct": 3.0}


@pytest.fixture
def sign_in():
    """Sign in as ``subject`` (with a finished account) for the next requests."""
    def as_user(subject: str, *, profile: bool = True):
        app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)
        if profile:
            assert client.post("/api/account", json=PROFILE).status_code == 200
    yield as_user


def irr(inputs: dict, settings: dict) -> float:
    resp = client.post("/api/deal/run", json={"inputs": inputs, "settings": settings})
    assert resp.status_code == 200, resp.text
    return resp.json()["returns"]["irr"]


def create(name="Project Alpine", inputs=INPUTS, settings=SETTINGS) -> dict:
    resp = client.post("/api/deals", json={"name": name, "inputs": inputs, "settings": settings})
    assert resp.status_code == 201, resp.text
    return resp.json()


def version_rows(deal_id) -> int:
    with db_engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(DealVersion)
                            .where(DealVersion.deal_id == deal_id)).scalar()


# ---------------------------------------------------------------------------
# No database needed
# ---------------------------------------------------------------------------
def test_the_settings_in_the_test_deal_really_change_its_irr():
    """Otherwise the reopen and restore tests below couldn't tell settings were kept."""
    assert irr(INPUTS, SETTINGS) != pytest.approx(irr(INPUTS, {}), abs=1e-4)


@pytest.mark.parametrize("settings, why", [
    ({"no_such_setting": 1}, "unknown key"),
    ({"tx_fee_pct": True}, "a switch where a number belongs"),
    ({"tx_fee_pct": "3"}, "text"),
    ({"tx_fee_pct": float("inf")}, "not finite"),
])
def test_impossible_settings_are_refused(settings, why):
    with pytest.raises(store.InvalidDeal):
        store.clean_settings(settings)


def test_settings_equal_to_the_defaults_are_not_stored():
    from core.config import DEFAULTS
    assert store.clean_settings({"tx_fee_pct": DEFAULTS["tx_fee_pct"], **SETTINGS}) == SETTINGS


def test_stored_inputs_are_complete_even_when_a_caller_sends_a_few():
    cleaned = store.clean_inputs({"ebitda": 50})
    assert cleaned["ebitda"] == 50 and cleaned["exit_mult"] == 11.0 and len(cleaned) == len(INPUTS)


@pytest.mark.parametrize("name", ["", "   ", "x" * 121])
def test_names_must_be_real(name):
    with pytest.raises(store.InvalidDeal):
        store.clean_name(name)


def test_without_a_database_saved_deals_say_so(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    resp = client.get("/api/deals")
    assert resp.status_code == 503 and "DATABASE_URL" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Done when: a saved deal reopens identically elsewhere
# ---------------------------------------------------------------------------
def test_a_saved_deal_reopens_identically_elsewhere(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    expected = irr(INPUTS, SETTINGS)
    deal = create()

    # "Elsewhere": every pooled connection closed, a new client, the same account
    db_engine.dispose_engines()
    other_device = TestClient(app)
    listed = other_device.get("/api/deals").json()["deals"]
    assert [d["name"] for d in listed] == ["Project Alpine"]

    opened = other_device.get(f"/api/deals/{listed[0]['id']}").json()
    assert opened["inputs"] == INPUTS and opened["settings"] == SETTINGS
    assert irr(opened["inputs"], opened["settings"]) == expected
    assert opened["latest_version"] == 1 and deal["id"] == opened["id"]


def test_autosave_is_what_reopens(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    deal = create()
    edited = {**INPUTS, "exit_mult": 12.0}
    saved = client.put(f"/api/deals/{deal['id']}/draft", json={"inputs": edited, "settings": {}})
    assert saved.status_code == 200, saved.text

    db_engine.dispose_engines()
    opened = client.get(f"/api/deals/{deal['id']}").json()
    assert opened["inputs"] == edited and opened["settings"] == {}
    assert irr(opened["inputs"], opened["settings"]) == irr(edited, {})
    assert irr(opened["inputs"], opened["settings"]) != irr(INPUTS, SETTINGS)


# ---------------------------------------------------------------------------
# Done when: restoring a version brings back its exact IRR
# ---------------------------------------------------------------------------
def test_restoring_a_version_brings_back_its_exact_irr(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    deal = create()
    original_irr = irr(INPUTS, SETTINGS)

    # Edit it well away from the first version
    edited = {**INPUTS, "exit_mult": 8.0, "hold": 4}
    client.put(f"/api/deals/{deal['id']}/draft", json={"inputs": edited, "settings": {}})
    now = client.get(f"/api/deals/{deal['id']}").json()
    assert irr(now["inputs"], now["settings"]) != pytest.approx(original_irr, abs=1e-3)

    restored = client.post(f"/api/deals/{deal['id']}/versions/1/restore")
    assert restored.status_code == 200, restored.text
    reopened = client.get(f"/api/deals/{deal['id']}").json()
    assert irr(reopened["inputs"], reopened["settings"]) == original_irr

    # Nothing was lost: the edits are a version, and so is the restore
    versions = client.get(f"/api/deals/{deal['id']}/versions").json()["versions"]
    assert [(v["number"], v["kind"], v["label"]) for v in versions] == [
        (3, "restored", "Restored version 1"),
        (2, "saved", "Before restoring version 1"),
        (1, "created", None),
    ]
    before = client.get(f"/api/deals/{deal['id']}/versions/2").json()
    assert irr(before["inputs"], before["settings"]) == irr(edited, {})


def test_a_named_version_restores_after_later_edits(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    deal = create()
    step = {**INPUTS, "growth": 2.0}
    client.put(f"/api/deals/{deal['id']}/draft", json={"inputs": step, "settings": SETTINGS})
    kept = client.post(f"/api/deals/{deal['id']}/versions", json={"label": "IC draft"})
    assert kept.status_code == 201 and kept.json()["number"] == 2 and kept.json()["label"] == "IC draft"

    client.put(f"/api/deals/{deal['id']}/draft", json={"inputs": INPUTS, "settings": {}})
    client.post(f"/api/deals/{deal['id']}/versions/2/restore")
    reopened = client.get(f"/api/deals/{deal['id']}").json()
    assert reopened["inputs"] == step
    assert irr(reopened["inputs"], reopened["settings"]) == irr(step, SETTINGS)


def test_restoring_a_missing_version_changes_nothing(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    deal = create()
    assert client.post(f"/api/deals/{deal['id']}/versions/9/restore").status_code == 404
    assert client.get(f"/api/deals/{deal['id']}/versions/9").status_code == 404
    assert client.get(f"/api/deals/{deal['id']}").json()["latest_version"] == 1


# ---------------------------------------------------------------------------
# Done when: users can't open each other's deals
# ---------------------------------------------------------------------------
def test_users_cant_open_each_others_deals(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    deal = create()
    path = f"/api/deals/{deal['id']}"

    sign_in("user_ben")
    assert client.get("/api/deals").json()["deals"] == []
    assert client.get("/api/deals", params={"archived": True}).json()["deals"] == []
    attempts = [
        ("GET", path, None),
        ("PATCH", path, {"name": "Mine now"}),
        ("PATCH", path, {"archived": True}),
        ("PUT", f"{path}/draft", {"inputs": {**INPUTS, "ebitda": 1.0}, "settings": {}}),
        ("POST", f"{path}/duplicate", {}),
        ("GET", f"{path}/versions", None),
        ("POST", f"{path}/versions", {"label": "hijack"}),
        ("GET", f"{path}/versions/1", None),
        ("POST", f"{path}/versions/1/restore", None),
        ("DELETE", path, None),
    ]
    for method, url, body in attempts:
        resp = client.request(method, url, json=body)
        # 404, not 403: another account's deal looks exactly like no deal
        assert resp.status_code == 404, f"{method} {url} answered {resp.status_code}"
        assert resp.json() == {"detail": "Deal not found"}
    assert client.get("/api/deals").json()["deals"] == []  # the duplicate wasn't made

    sign_in("user_anna")
    untouched = client.get(path).json()
    assert untouched["name"] == "Project Alpine" and not untouched["archived"]
    assert untouched["inputs"] == INPUTS and untouched["latest_version"] == 1
    assert version_rows(deal["id"]) == 1


def test_a_deal_id_that_never_existed_is_the_same_404(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    resp = client.get("/api/deals/00000000-0000-4000-8000-000000000000")
    assert resp.status_code == 404 and resp.json() == {"detail": "Deal not found"}
    assert client.get("/api/deals/not-a-uuid").status_code == 422


def test_saving_needs_a_finished_account(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_new", profile=False)
    resp = client.post("/api/deals", json={"name": "Early", "inputs": INPUTS})
    assert resp.status_code == 409 and "account" in resp.json()["detail"]
    assert client.get("/api/deals").json()["deals"] == []


# ---------------------------------------------------------------------------
# The deal list: rename, duplicate, archive, delete
# ---------------------------------------------------------------------------
def test_rename_archive_unarchive_and_the_list(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    first = create("First")
    second = create("Second")
    # Most recently edited first
    assert [d["name"] for d in client.get("/api/deals").json()["deals"]] == ["Second", "First"]

    renamed = client.patch(f"/api/deals/{first['id']}", json={"name": "  Renamed   deal "})
    assert renamed.json()["name"] == "Renamed deal"
    assert [d["name"] for d in client.get("/api/deals").json()["deals"]] == ["Renamed deal", "Second"]

    client.patch(f"/api/deals/{second['id']}", json={"archived": True})
    assert [d["name"] for d in client.get("/api/deals").json()["deals"]] == ["Renamed deal"]
    everything = client.get("/api/deals", params={"archived": True}).json()["deals"]
    assert {(d["name"], d["archived"]) for d in everything} == {("Renamed deal", False), ("Second", True)}
    # Archived deals still open, with their content
    assert client.get(f"/api/deals/{second['id']}").json()["inputs"] == INPUTS

    client.patch(f"/api/deals/{second['id']}", json={"archived": False})
    assert len(client.get("/api/deals").json()["deals"]) == 2
    # Renaming doesn't add history
    assert client.get(f"/api/deals/{first['id']}").json()["latest_version"] == 1


@pytest.mark.parametrize("body", [{"name": ""}, {"name": "   "}, {"name": "x" * 121}, {"owner": "user_ben"}])
def test_bad_renames_are_refused(fresh_db, sign_in, body):  # noqa: ARG001
    sign_in("user_anna")
    deal = create()
    assert client.patch(f"/api/deals/{deal['id']}", json=body).status_code == 422
    assert client.get(f"/api/deals/{deal['id']}").json()["name"] == "Project Alpine"


def test_duplicate_copies_the_working_copy_not_the_history(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    deal = create()
    edited = {**INPUTS, "hold": 7}
    client.put(f"/api/deals/{deal['id']}/draft", json={"inputs": edited, "settings": SETTINGS})
    client.post(f"/api/deals/{deal['id']}/versions", json={"label": "v2"})

    copy = client.post(f"/api/deals/{deal['id']}/duplicate")
    assert copy.status_code == 201, copy.text
    copy = copy.json()
    assert copy["id"] != deal["id"] and copy["name"] == "Project Alpine (copy)"
    assert copy["inputs"] == edited and copy["settings"] == SETTINGS
    assert irr(copy["inputs"], copy["settings"]) == irr(edited, SETTINGS)
    assert [v["number"] for v in client.get(f"/api/deals/{copy['id']}/versions").json()["versions"]] == [1]

    named = client.post(f"/api/deals/{deal['id']}/duplicate", json={"name": "Alpine downside"}).json()
    assert named["name"] == "Alpine downside"
    # Editing the copy leaves the original alone
    client.put(f"/api/deals/{copy['id']}/draft", json={"inputs": INPUTS, "settings": {}})
    assert client.get(f"/api/deals/{deal['id']}").json()["inputs"] == edited


def test_delete_removes_the_deal_and_every_version(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    deal = create()
    keep = create("Keep me")
    client.put(f"/api/deals/{deal['id']}/draft", json={"inputs": {**INPUTS, "hold": 3}, "settings": {}})
    client.post(f"/api/deals/{deal['id']}/versions")
    assert version_rows(deal["id"]) == 2

    assert client.delete(f"/api/deals/{deal['id']}").status_code == 204
    assert client.get(f"/api/deals/{deal['id']}").status_code == 404
    assert client.delete(f"/api/deals/{deal['id']}").status_code == 404
    assert version_rows(deal["id"]) == 0
    assert [d["id"] for d in client.get("/api/deals").json()["deals"]] == [keep["id"]]


@pytest.mark.parametrize("body", [
    {"name": "Bad", "inputs": {**INPUTS, "hold": 40}},
    {"name": "Bad", "inputs": INPUTS, "settings": {"no_such_setting": 1}},
    {"name": "Bad", "inputs": INPUTS, "settings": {"tx_fee_pct": True}},
    {"name": "", "inputs": INPUTS},
    {"inputs": INPUTS},
])
def test_impossible_deals_are_not_saved(fresh_db, sign_in, body):  # noqa: ARG001
    sign_in("user_anna")
    assert client.post("/api/deals", json=body).status_code == 422
    assert client.get("/api/deals").json()["deals"] == []


# ---------------------------------------------------------------------------
# Versions stay compact (the free plan has 0.5 GB)
# ---------------------------------------------------------------------------
def test_autosave_overwrites_the_draft_without_adding_versions(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    deal = create()
    for ebitda in range(101, 131):  # thirty edits in a row
        client.put(f"/api/deals/{deal['id']}/draft",
                   json={"inputs": {**INPUTS, "ebitda": float(ebitda)}, "settings": SETTINGS})
    assert version_rows(deal["id"]) == 1
    assert client.get(f"/api/deals/{deal['id']}").json()["inputs"]["ebitda"] == 130.0


def test_autosave_adds_a_checkpoint_at_most_once_per_window_and_keeps_a_few(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    subject = "user_anna"
    start = utc_now()
    deal = store.create_deal(subject, "Timed", INPUTS, SETTINGS, now=start)
    window = timedelta(seconds=store.AUTO_CHECKPOINT_S)

    store.save_draft(subject, deal.id, {**INPUTS, "hold": 4}, SETTINGS, now=start + window / 2)
    assert [v.kind for v in store.list_versions(subject, deal.id)] == ["created"]
    store.save_draft(subject, deal.id, {**INPUTS, "hold": 7}, SETTINGS, now=start + window)
    assert [v.kind for v in store.list_versions(subject, deal.id)] == ["auto", "created"]
    # Unchanged content: no write and no checkpoint, however late
    same = store.save_draft(subject, deal.id, {**INPUTS, "hold": 7}, SETTINGS, now=start + 5 * window)
    assert same.latest_version == 2 and same.updated_at == start + window

    for i in range(store.KEEP_AUTO_VERSIONS + 5):
        store.save_draft(subject, deal.id, {**INPUTS, "ebitda": 300.0 + i}, SETTINGS,
                         now=start + (i + 2) * window)
    kinds = [v.kind for v in store.list_versions(subject, deal.id)]
    assert kinds.count("auto") == store.KEEP_AUTO_VERSIONS and kinds[-1] == "created"
    newest = store.get_version(subject, deal.id, store.list_versions(subject, deal.id)[0].number)
    assert newest.inputs["ebitda"] == 300.0 + store.KEEP_AUTO_VERSIONS + 4


def test_saving_without_changes_adds_no_version(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    deal = create()
    first = client.post(f"/api/deals/{deal['id']}/versions").json()
    again = client.post(f"/api/deals/{deal['id']}/versions").json()
    assert first["number"] == again["number"] == 1
    named = client.post(f"/api/deals/{deal['id']}/versions", json={"label": "Signed"}).json()
    assert (named["number"], named["label"]) == (1, "Signed")
    assert version_rows(deal["id"]) == 1


def test_a_version_is_a_few_hundred_bytes(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    deal = create()
    with db_engine.connect() as conn:
        row_bytes = conn.execute(text(
            "SELECT pg_column_size(v.*) FROM deal_versions v WHERE deal_id = :id"), {"id": deal["id"]}
        ).scalar()
    # 0.5 GB holds on the order of a million versions at this size
    assert row_bytes < 900, row_bytes


# ---------------------------------------------------------------------------
# Settings follow the account
# ---------------------------------------------------------------------------
def test_settings_follow_the_account_not_the_browser(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    assert client.get("/api/account/settings").json() == {"settings": {}}
    saved = client.put("/api/account/settings", json={"settings": {"tx_fee_pct": 5.0, "mc_hurdle": 25}})
    assert saved.status_code == 200 and saved.json()["settings"] == {"tx_fee_pct": 5.0, "mc_hurdle": 25}

    db_engine.dispose_engines()
    assert TestClient(app).get("/api/account/settings").json()["settings"] == {
        "tx_fee_pct": 5.0, "mc_hurdle": 25}

    sign_in("user_ben")
    assert client.get("/api/account/settings").json() == {"settings": {}}
    sign_in("user_anna")
    client.put("/api/account/settings", json={"settings": {}})
    assert client.get("/api/account/settings").json() == {"settings": {}}


def test_impossible_account_settings_are_refused(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_anna")
    client.put("/api/account/settings", json={"settings": SETTINGS})
    assert client.put("/api/account/settings", json={"settings": {"nope": 1}}).status_code == 422
    assert client.get("/api/account/settings").json()["settings"] == SETTINGS


def test_settings_need_a_finished_account(fresh_db, sign_in):  # noqa: ARG001
    sign_in("user_new", profile=False)
    assert client.get("/api/account/settings").status_code == 409
    assert client.put("/api/account/settings", json={"settings": SETTINGS}).status_code == 409
