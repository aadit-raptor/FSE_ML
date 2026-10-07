"""The optional reference library and its admin switch (PLAN.md 4.5).

With the library off every other endpoint answers exactly as with it on;
the library's own endpoints say it is off and hold nothing. Only an
administrator (``FSE_ADMINS``) may switch it, the choice is stored per
environment and audited once per change, and the server's own
``FSE_EXAMPLE_LIBRARY=0`` keeps it off whatever was chosen.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from api.auth import AuthUser, require_user
from api.main import app
from db import engine as db_engine
from db.models import AppFlag, AuditEvent
from library import coverage as cov
from library import switch
from tests.test_deals import PROFILE

client = TestClient(app)
ADMIN = "user_admin"
DEAL = {"ebitda": 100.0, "entry_mult": 10.0, "exit_mult": 10.0, "hold": 5}


@pytest.fixture
def as_user():
    def sign_in(subject: str, *, profile: bool = True):
        app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)
        if profile:
            assert client.post("/api/account", json=PROFILE).status_code == 200
    yield sign_in


@pytest.fixture
def admins(monkeypatch):
    monkeypatch.setenv("FSE_ADMINS", f" {ADMIN} , user_other_admin")


def library_answers() -> dict:
    return {
        "examples": client.get("/api/backtesting/examples").json(),
        "deals": client.get("/api/backtesting/deals").json(),
        "base_rates": client.get("/api/library/base-rates").json(),
        "coverage": client.get("/api/library/coverage").json(),
    }


def everything_else() -> dict:
    """Answers from outside the library, which must not depend on it."""
    deal = client.post("/api/deal/run", json={"inputs": DEAL, "settings": {}})
    assert deal.status_code == 200, deal.text
    answer = deal.json()
    return {"irr": answer["returns"]["irr"], "moic": answer["returns"]["moic"],
            "presets": client.get("/api/deal/tax-presets").status_code}


def switch_to(enabled: bool) -> dict:
    resp = client.put("/api/library/switch", json={"enabled": enabled})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_on_by_default_with_the_examples_base_rates_and_coverage():
    assert client.get("/api/library").json() == {
        "enabled": True, "locked_off": False, "can_switch": False, "updated_at": None}
    on = library_answers()
    assert on["examples"]["enabled"] and len(on["examples"]["examples"]) == 4
    assert len(on["deals"]) == 5   # the four examples and the blank template
    assert on["base_rates"]["enabled"] and len(on["base_rates"]["sources"]) == 2
    assert on["coverage"]["enabled"] and len(on["coverage"]["base_rates"]) == 8


def test_with_the_server_switch_off_the_library_is_empty_and_nothing_else_moves(monkeypatch):
    before = everything_else()
    monkeypatch.setenv("FSE_EXAMPLE_LIBRARY", "0")
    assert client.get("/api/library").json()["enabled"] is False
    assert library_answers() == {
        "examples": {"enabled": False, "examples": []},
        "deals": [],
        "base_rates": {"enabled": False, "sources": [], "tables": [], "country": None, "sp_region": None,
                       "default": None, "recovery": None},
        "coverage": {"enabled": False, "collections": [], "base_rates": []},
    }
    assert everything_else() == before


def test_the_base_rates_name_the_deals_region():
    answer = client.get("/api/library/base-rates", params={"country": "in"}).json()
    assert answer["sp_region"] == "emerging"
    assert client.get("/api/library/base-rates", params={"country": "IND"}).status_code == 422


def test_only_an_administrator_may_switch(admins, as_user):  # noqa: ARG001
    as_user("user_someone", profile=False)
    assert client.get("/api/library").json()["can_switch"] is False
    assert client.put("/api/library/switch", json={"enabled": False}).status_code == 403


def test_without_a_database_an_administrator_is_told_why(admins, as_user, monkeypatch):  # noqa: ARG001
    monkeypatch.delenv("DATABASE_URL", raising=False)
    as_user(ADMIN, profile=False)
    assert client.get("/api/library").json()["can_switch"] is False
    assert client.put("/api/library/switch", json={"enabled": False}).status_code == 503


def test_the_switch_only_takes_a_boolean(admins, as_user):  # noqa: ARG001
    as_user(ADMIN, profile=False)
    assert client.put("/api/library/switch", json={"enabled": "no"}).status_code == 422


def test_admins_are_read_from_the_server_setting(monkeypatch):
    monkeypatch.setenv("FSE_ADMINS", "user_a,, user_b ")
    assert switch.admins() == {"user_a", "user_b"}
    assert switch.is_admin("user_b") and not switch.is_admin("user_c") and not switch.is_admin(None)
    monkeypatch.delenv("FSE_ADMINS")
    assert switch.admins() == frozenset()


def audit_rows() -> list:
    with db_engine.connect() as conn:
        return conn.execute(select(AuditEvent.action, AuditEvent.detail)
                            .where(AuditEvent.action == "library_switched").order_by(AuditEvent.id)).all()


def test_an_administrator_hides_it_for_everyone_and_shows_it_again(fresh_db, admins, as_user):  # noqa: ARG001
    as_user(ADMIN)
    before = everything_else()
    assert client.get("/api/library").json()["can_switch"] is True

    off = switch_to(False)
    assert off["enabled"] is False and off["updated_at"]
    as_user("user_someone")
    assert client.get("/api/library").json() == {**off, "can_switch": False}
    hidden = library_answers()
    assert hidden["examples"] == {"enabled": False, "examples": []}
    assert hidden["base_rates"]["enabled"] is False and hidden["coverage"]["collections"] == []
    assert everything_else() == before

    as_user(ADMIN)
    assert switch_to(True)["enabled"] is True
    assert library_answers()["examples"]["enabled"] is True
    with db_engine.connect() as conn:
        stored = conn.execute(select(AppFlag.name, AppFlag.enabled)).all()
    assert [tuple(r) for r in stored] == [("library", True)]


def test_each_change_is_one_audit_entry_and_a_repeat_is_none(fresh_db, admins, as_user):  # noqa: ARG001
    as_user(ADMIN)
    # Already on by default: nothing changes, nothing is written, not even a row
    assert switch_to(True)["updated_at"] is None
    assert audit_rows() == []
    with db_engine.connect() as conn:
        assert conn.execute(select(AppFlag.name)).all() == []
    first = switch_to(False)
    # A repeat changes nothing at all: not even when it was last switched
    assert switch_to(False)["updated_at"] == first["updated_at"]
    switch_to(True)
    assert [tuple(r) for r in audit_rows()] == [("library_switched", {"enabled": False}),
                                                ("library_switched", {"enabled": True})]
    entries = client.get("/api/account/history").json()["entries"]
    assert [(e["action"], e["enabled"], e["deal_id"]) for e in entries] == [
        ("library_switched", True, None), ("library_switched", False, None)]


def test_the_server_setting_keeps_it_off_whatever_was_chosen(fresh_db, admins, as_user, monkeypatch):  # noqa: ARG001
    as_user(ADMIN)
    switch_to(True)
    monkeypatch.setenv("FSE_EXAMPLE_LIBRARY", "off")
    state = client.get("/api/library").json()
    assert state["enabled"] is False and state["locked_off"] is True and state["can_switch"] is False
    assert client.put("/api/library/switch", json={"enabled": True}).status_code == 409
    assert library_answers()["examples"]["enabled"] is False


def test_an_administrator_without_an_account_is_asked_to_finish_it(fresh_db, admins, as_user):  # noqa: ARG001
    as_user(ADMIN, profile=False)
    assert client.put("/api/library/switch", json={"enabled": False}).status_code == 409


def test_the_stored_choice_is_read_once_a_minute(fresh_db, admins, as_user, monkeypatch):  # noqa: ARG001
    as_user(ADMIN)
    switch_to(False)
    switch.reset_cache()
    reads = []
    real = switch.flags.get
    monkeypatch.setattr(switch.flags, "get", lambda name: reads.append(name) or real(name))
    for _ in range(5):
        assert switch.enabled() is False
    assert reads == ["library"]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
def counts(collection: dict, dimension: str) -> dict:
    return {b["bucket"]: b["count"] for b in collection["dimensions"][dimension]}


def test_coverage_counts_the_examples_in_every_dimension_with_empty_buckets_shown():
    answer = client.get("/api/library/coverage").json()
    refs, examples = answer["collections"]
    assert (refs["id"], refs["count"], refs["sourced"]) == ("reference_deals", 0, True)
    assert (examples["id"], examples["count"], examples["sourced"]) == ("examples", 4, False)
    assert counts(examples, "region") == {"us": 4, "europe": 0, "emerging": 0, "other_developed": 0}
    # Entry EV (US dollar millions): BK 3,894; Freescale 17,640; Hilton 25,900; Dell 24,790
    assert counts(examples, "size") == {"under_100m": 0, "100m_1bn": 0, "1bn_10bn": 1, "over_10bn": 3}
    assert counts(examples, "era") == {"before_2000": 0, "2000_2007": 2, "2008_2014": 2, "2015_2019": 0, "2020_on": 0}
    assert counts(examples, "outcome") == {"success": 3, "distress": 1, "held": 0}
    assert sum(counts(examples, "sector").values()) == 4
    assert all(b["count"] == 0 for d in refs["dimensions"].values() for b in d if d)


@pytest.mark.parametrize("year,band", [(1999, "before_2000"), (2000, "2000_2007"), (2007, "2000_2007"),
                                       (2008, "2008_2014"), (2019, "2015_2019"), (2020, "2020_on"), (None, None)])
def test_eras_are_bounded_by_entry_year(year, band):
    assert cov.era(year) == band


@pytest.mark.parametrize("ev,band", [(99.9, "under_100m"), (100.0, "100m_1bn"), (999.9, "100m_1bn"),
                                     (1_000.0, "1bn_10bn"), (10_000.0, "over_10bn"), (None, None)])
def test_sizes_are_bounded_by_entry_value(ev, band):
    assert cov.size(ev) == band


def test_the_base_rate_coverage_counts_regions_bands_and_years():
    rows = {r["table"]: r for r in client.get("/api/library/coverage").json()["base_rates"]}
    assert (rows["speculative_by_region"]["regions"], rows["speculative_by_region"]["first_year"]) == (4, 1981)
    assert rows["annual_by_rating"]["bands"] == 7 and rows["annual_by_rating"]["observations"] == 23831
    assert rows["cumulative_by_region"]["regions"] == 3
    assert rows["lgd_by_region"]["regions"] == 6 and rows["lgd_by_region"]["observations"] == 11527
    assert (rows["lgd_by_year"]["first_year"], rows["lgd_by_year"]["last_year"]) == (2000, 2016)


def test_a_database_outage_leaves_the_library_on_rather_than_failing(fresh_db, monkeypatch):  # noqa: ARG001
    from db import DatabaseUnavailable

    def down(name):
        raise DatabaseUnavailable("down")
    monkeypatch.setattr(switch.flags, "get", down)
    assert client.get("/api/library").json()["enabled"] is True
    assert len(client.get("/api/backtesting/examples").json()["examples"]) == 4
