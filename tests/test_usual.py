"""Personalized defaults (PLAN.md 5.9): a user's usual figures.

The task's two lines are tested through the real endpoints and a real
database: suggestions come from the caller's own deals and nobody else's,
and the off switch stops them. The arithmetic (which deals count, the
median, how many are needed) is tested on plain dictionaries.
"""
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from api.auth import AuthUser, require_user
from api.main import app
from benchmarks import usual
from db import deals as store
from db import engine as db_engine
from db.models import AuditEvent
from tests.test_deals import INPUTS, PROFILE

client = TestClient(app)
URL = "/api/account/suggestions"


def deal(country="DE", industry="machinery", **fields) -> dict:
    return {**INPUTS, "country": country, "industry": industry, **fields}


def by_field(answer: dict) -> dict:
    return {f["field"]: f for f in answer["figures"]}


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------
def test_the_usual_figure_is_the_median_of_the_group_with_its_range():
    deals = [deal(gross_margin=g) for g in (30.0, 50.0, 34.0)]
    answer = usual.usual_figures(deals, "DE", "machinery")
    assert answer["status"] == "ok"
    assert (answer["region"], answer["sector"], answer["deals"]) == ("europe", "industrials", 3)
    assert by_field(answer)["gross_margin"] == {"field": "gross_margin", "value": 34.0, "deals": 3,
                                                "low": 30.0, "high": 50.0}


def test_a_group_is_the_same_region_and_sector_not_the_same_country_or_industry():
    deals = [deal("DE", "machinery", opex=10.0),
             deal("FR", "trucking", opex=12.0),          # Europe, industrials: counts
             deal("GB", "aerospace_defense", opex=14.0),  # Europe, industrials: counts
             deal("US", "machinery", opex=90.0),          # another region
             deal("DE", "software_system_application", opex=90.0),  # another sector
             deal("", "machinery", opex=90.0)]            # never sourced for a country
    answer = usual.usual_figures(deals, "IT", "machinery")
    assert answer["deals"] == 3
    assert by_field(answer)["opex"]["value"] == 12.0


def test_fewer_deals_than_the_minimum_suggest_nothing():
    deals = [deal(gross_margin=30.0)] * (usual.MIN_DEALS - 1)
    answer = usual.usual_figures(deals, "DE", "machinery")
    assert (answer["status"], answer["deals"], answer["figures"]) == ("not_enough", usual.MIN_DEALS - 1, [])
    assert answer["min_deals"] == usual.MIN_DEALS


def test_no_country_means_no_group():
    answer = usual.usual_figures([deal()] * 5, "", "machinery")
    assert (answer["status"], answer["region"], answer["figures"]) == ("no_country", None, [])


def test_an_industry_without_a_sector_is_its_own_group():
    deals = [deal("DE", "all", tax=20.0), deal("FR", "", tax=22.0), deal("NL", "all", tax=24.0),
             deal("DE", "machinery", tax=90.0)]
    answer = usual.usual_figures(deals, "DE", "")
    assert (answer["sector"], answer["industry"], answer["deals"]) == (None, "all", 3)
    assert by_field(answer)["tax"]["value"] == 22.0


def test_a_deal_that_lists_its_facilities_says_nothing_about_the_percentage_debt_fields():
    listed = deal(debt_pct=10.0, senior_pct=10.0, base_rate=1.0, growth=3.0,
                  tranches=[{"name": "Term loan", "amount": 100.0}])
    deals = [listed, listed, deal(debt_pct=60.0, growth=5.0), deal(debt_pct=62.0, growth=5.0),
             deal(debt_pct=64.0, growth=7.0)]
    figures = by_field(usual.usual_figures(deals, "DE", "machinery"))
    assert (figures["debt_pct"]["value"], figures["debt_pct"]["deals"]) == (62.0, 3)
    assert (figures["growth"]["value"], figures["growth"]["deals"]) == (5.0, 5)


def test_only_figures_with_enough_values_are_suggested_and_bad_values_are_dropped():
    deals = [deal(capex=3.0, nwc=float("nan")), deal(capex=4.0, nwc=None), deal(capex=5.0, nwc="1"),
             deal(capex=True)]
    figures = by_field(usual.usual_figures(deals, "DE", "machinery"))
    assert (figures["capex"]["value"], figures["capex"]["deals"]) == (4.0, 3)
    assert "nwc" not in figures


def test_days_keep_one_decimal_and_other_figures_two():
    deals = [deal(ar_days=d, entry_mult=m) for d, m in ((40.11, 8.111), (40.22, 8.222), (40.33, 8.333), (40.44, 8.444))]
    figures = by_field(usual.usual_figures(deals, "DE", "machinery"))
    assert figures["ar_days"]["value"] == 40.3
    assert figures["entry_mult"]["value"] == 8.28


# ---------------------------------------------------------------------------
# Through the API and the database
# ---------------------------------------------------------------------------
@pytest.fixture
def sign_in():
    def as_user(subject: str, *, profile: bool = True):
        app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)
        if profile:
            assert client.post("/api/account", json=PROFILE).status_code == 200
    yield as_user


def save(name: str, **fields) -> str:
    resp = client.post("/api/deals", json={"name": name, "inputs": deal(**fields)})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def ask(**params) -> dict:
    resp = client.get(URL, params={"country": "DE", "industry": "machinery", **params})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_suggestions_use_only_the_callers_own_deals(fresh_db, sign_in):
    sign_in("user_other")
    for i in range(5):
        save(f"theirs {i}", gross_margin=80.0, entry_mult=20.0)
    sign_in("user_me")
    for i, margin in enumerate((30.0, 32.0, 34.0)):
        save(f"mine {i}", gross_margin=margin, entry_mult=8.0)

    mine = ask()
    assert (mine["enabled"], mine["status"], mine["deals"]) == (True, "ok", 3)
    figures = by_field(mine)
    assert figures["gross_margin"] == {"field": "gross_margin", "value": 32.0, "deals": 3,
                                       "low": 30.0, "high": 34.0}
    assert figures["entry_mult"]["value"] == 8.0
    assert all(f["deals"] == 3 for f in mine["figures"])

    sign_in("user_other")
    theirs = ask()
    assert theirs["deals"] == 5 and by_field(theirs)["gross_margin"]["value"] == 80.0


def test_a_new_account_with_no_deals_sees_nothing_of_anyone(fresh_db, sign_in):
    sign_in("user_other")
    for i in range(4):
        save(f"theirs {i}")
    sign_in("user_new")
    answer = ask()
    assert (answer["status"], answer["deals"], answer["figures"]) == ("not_enough", 0, [])
    sign_in("user_unfinished", profile=False)
    answer = ask()
    assert (answer["enabled"], answer["deals"], answer["figures"]) == (True, 0, [])


def test_the_open_deal_does_not_count_towards_its_own_suggestion(fresh_db, sign_in):
    sign_in("user_me")
    ids = [save(f"mine {i}", gross_margin=m) for i, m in enumerate((30.0, 32.0, 34.0, 90.0))]
    assert ask()["deals"] == 4
    answer = ask(deal_id=ids[3])
    assert answer["deals"] == 3 and by_field(answer)["gross_margin"]["value"] == 32.0
    # Naming a deal that isn't the caller's, or doesn't exist, leaves nothing out and shows nothing more
    assert ask(deal_id=str(uuid.uuid4()))["deals"] == 4


def test_off_stops_suggestions_and_on_brings_them_back(fresh_db, sign_in):
    sign_in("user_me")
    for i in range(3):
        save(f"mine {i}")
    assert ask()["status"] == "ok"

    assert client.put(URL, json={"enabled": False}).json() == {"enabled": False}
    off = ask()
    assert off == {"enabled": False, "status": "off", "region": None, "sector": None, "industry": None,
                   "deals": 0, "min_deals": usual.MIN_DEALS, "figures": []}
    assert store.own_deal_inputs("user_me") == (False, [])   # off: the deals are not even read

    assert client.put(URL, json={"enabled": True}).json() == {"enabled": True}
    assert ask()["status"] == "ok"


def test_one_accounts_switch_is_not_anothers(fresh_db, sign_in):
    sign_in("user_other")
    for i in range(3):
        save(f"theirs {i}")
    sign_in("user_me")
    client.put(URL, json={"enabled": False})
    sign_in("user_other", profile=False)
    assert ask()["status"] == "ok"


def test_each_change_of_the_switch_is_one_history_entry_and_a_repeat_is_none(fresh_db, sign_in):
    sign_in("user_me")
    for enabled in (False, False, True, True, False):
        assert client.put(URL, json={"enabled": enabled}).status_code == 200
    with db_engine.connect() as conn:
        rows = conn.execute(select(AuditEvent.action, AuditEvent.detail, AuditEvent.deal_id)
                            .order_by(AuditEvent.id)).all()
    assert [tuple(r) for r in rows] == [("suggestions_switched", {"enabled": False}, None),
                                        ("suggestions_switched", {"enabled": True}, None),
                                        ("suggestions_switched", {"enabled": False}, None)]
    history = client.get("/api/account/history").json()["entries"]
    assert [(e["action"], e["enabled"]) for e in history] == [
        ("suggestions_switched", False), ("suggestions_switched", True), ("suggestions_switched", False)]


def test_the_switch_needs_a_finished_account_and_a_valid_request(fresh_db, sign_in):
    sign_in("user_unfinished", profile=False)
    assert client.put(URL, json={"enabled": False}).status_code == 409
    sign_in("user_me")
    assert client.put(URL, json={"enabled": "maybe"}).status_code == 422
    assert client.get(URL, params={"country": "Germany"}).status_code == 422


def test_a_retired_country_code_is_read_as_its_current_country(fresh_db, sign_in):
    sign_in("user_me")
    for i in range(3):
        save(f"mine {i}")
    assert ask(country="UK")["region"] == "europe"


def test_without_a_database_there_is_nothing_to_suggest(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert client.get(URL, params={"country": "DE"}).status_code == 503
