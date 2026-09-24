"""Locale: digit grouping, fiscal years (PLAN.md 2.3a).

Numbers and dates are formatted in the browser from the account's locale; the
API's part is to store the account's digit-grouping choice, to carry each
deal's fiscal year-end (never calculated with, like its currency) and to read
a filer's fiscal year-end from its 10-K. Excel number formats are in
test_export.py.
"""
import pytest
from fastapi.testclient import TestClient

from api.auth import AuthUser, require_user
from api.main import app
from db import users as user_store

client = TestClient(app)

MUMBAI = {"country": "IN", "preferred_currency": "INR", "locale": "en-IN",
          "time_zone": "Asia/Kolkata"}


@pytest.fixture
def as_user():
    def sign_in(subject: str):
        app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)
    yield sign_in


# ---------------------------------------------------------------------------
# Digit grouping on the account
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("grouping", ["locale", "thousands", "lakh"])
def test_every_digit_grouping_is_accepted(grouping):
    assert user_store.clean(**MUMBAI, digit_grouping=grouping).digit_grouping == grouping


def test_a_profile_without_a_grouping_follows_its_locale():
    assert user_store.clean(**MUMBAI).digit_grouping == "locale"


@pytest.mark.parametrize("grouping", ["indian", "", "LAKH ", "crore"])
def test_an_unknown_grouping_is_refused(grouping):
    with pytest.raises(user_store.InvalidProfile, match="digit grouping"):
        user_store.clean(**MUMBAI, digit_grouping=grouping)


def test_the_grouping_is_stored_with_the_account(fresh_db, as_user):  # noqa: ARG001
    as_user("user_lakh")
    saved = client.post("/api/account", json={**MUMBAI, "digit_grouping": "lakh"})
    assert saved.status_code == 200, saved.text
    assert client.get("/api/account").json()["profile"]["digit_grouping"] == "lakh"
    # Changing only the grouping changes only the grouping
    client.post("/api/account", json={**MUMBAI, "digit_grouping": "thousands"})
    assert client.get("/api/account").json()["profile"] == {**MUMBAI, "digit_grouping": "thousands"}


def test_a_client_that_never_sends_a_grouping_gets_the_locales(fresh_db, as_user):  # noqa: ARG001
    """The web app deployed before this change still saves profiles."""
    as_user("user_old_client")
    assert client.post("/api/account", json=MUMBAI).status_code == 200
    assert client.get("/api/account").json()["profile"]["digit_grouping"] == "locale"


def test_the_endpoint_refuses_an_unknown_grouping(fresh_db, as_user):  # noqa: ARG001
    as_user("user_bad")
    assert client.post("/api/account", json={**MUMBAI, "digit_grouping": "crore"}).status_code == 422
    assert client.get("/api/account").json()["profile"] is None


# ---------------------------------------------------------------------------
# Fiscal year-end per deal
# ---------------------------------------------------------------------------
MARCH_DEAL = {"fiscal_year_end_month": 3, "first_fiscal_year": 2027}


def _run(inputs):
    resp = client.post("/api/deal/run", json={"inputs": inputs})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_a_deal_defaults_to_a_december_year_end_with_no_first_year():
    schema = client.get("/api/openapi.json").json()["components"]["schemas"]["DealInputsIn"]["properties"]
    assert schema["fiscal_year_end_month"]["default"] == 12
    assert schema["first_fiscal_year"].get("default") is None


def test_the_fiscal_year_changes_no_number():
    """Labels only: a March year-end deal answers exactly as a December one."""
    plain, march = _run({}), _run(MARCH_DEAL)
    assert march["returns"]["irr"] == plain["returns"]["irr"] and plain["returns"]["irr"] > 0
    assert march["returns"] == plain["returns"]
    assert march["operating_model"] == plain["operating_model"]
    assert march["debt_schedule"] == plain["debt_schedule"]


@pytest.mark.parametrize("bad", [
    {"fiscal_year_end_month": 0}, {"fiscal_year_end_month": 13}, {"fiscal_year_end_month": 2.5},
    {"first_fiscal_year": 1899}, {"first_fiscal_year": 2201},
])
def test_an_impossible_fiscal_year_is_refused(bad):
    assert client.post("/api/deal/run", json={"inputs": bad}).status_code == 422


def test_every_endpoint_that_takes_a_deal_accepts_its_fiscal_year():
    """Each router builds the engine's DealInputs from the request, so the
    dataclass must carry the new fields too (DealInputs(**model_dump()))."""
    deal = {**MARCH_DEAL}
    assert client.post("/api/montecarlo/run", json={"mc": {"n": 1000}, "deal": deal, "seed": 1}).status_code == 200
    assert client.post("/api/export/montecarlo-sample",
                       json={"mc": {"n": 1000}, "deal": deal, "seed": 1}).status_code == 200


def test_a_saved_deal_keeps_its_fiscal_year(fresh_db, as_user):  # noqa: ARG001
    as_user("user_fiscal")
    client.post("/api/account", json=MUMBAI)
    created = client.post("/api/deals", json={"name": "March", "inputs": MARCH_DEAL})
    assert created.status_code == 201, created.text
    got = client.get(f"/api/deals/{created.json()['id']}").json()
    assert got["inputs"]["fiscal_year_end_month"] == 3
    assert got["inputs"]["first_fiscal_year"] == 2027


def test_a_deal_without_a_fiscal_year_stores_no_fiscal_keys():
    """Stored only when set: deals saved before 2.3a stay byte-identical (no
    extra version on their next autosave), and a rollback to an API without
    these fields only trips over deals that really use them."""
    from db.deals import clean_inputs
    assert "fiscal_year_end_month" not in clean_inputs({})
    assert "first_fiscal_year" not in clean_inputs({"fiscal_year_end_month": 12, "first_fiscal_year": None})
    march = clean_inputs({"fiscal_year_end_month": 3})
    assert march["fiscal_year_end_month"] == 3 and "first_fiscal_year" not in march
    assert clean_inputs(MARCH_DEAL)["first_fiscal_year"] == 2027


def test_an_old_deal_gets_no_new_version_from_the_new_fields(fresh_db, as_user):  # noqa: ARG001
    as_user("user_old_deal")
    client.post("/api/account", json=MUMBAI)
    deal = client.post("/api/deals", json={"name": "Old", "inputs": {"ebitda": 120.0}}).json()
    client.post(f"/api/deals/{deal['id']}/versions", json={"label": "v1"})
    before = client.get(f"/api/deals/{deal['id']}/versions").json()
    # The new web app sends the defaults explicitly; nothing changed
    client.put(f"/api/deals/{deal['id']}/draft",
               json={"inputs": {"ebitda": 120.0, "fiscal_year_end_month": 12, "first_fiscal_year": None}})
    client.post(f"/api/deals/{deal['id']}/versions", json={})
    assert client.get(f"/api/deals/{deal['id']}/versions").json() == before


# ---------------------------------------------------------------------------
# A filer's fiscal year-end, from its 10-K
# ---------------------------------------------------------------------------
def _facts(*ends):
    rows = [{"end": end, "val": 1e9, "form": "10-K", "fp": "FY", "filed": f"{end[:4]}-06-01"} for end in ends]
    return {"facts": {"us-gaap": {"Assets": {"units": {"USD": rows}}}}}


@pytest.mark.parametrize("ends, year_end", [
    (("2023-03-31", "2024-03-31", "2025-03-31"), (2025, 3)),  # March year-end (e.g. Indian and Japanese filers)
    (("2024-06-30", "2025-06-30"), (2025, 6)),                # June (MSFT)
    (("2024-12-31", "2025-12-31"), (2025, 12)),
    (("2020-12-31", "2024-03-31", "2025-03-31"), (2025, 3)),  # the latest filing decides after a change
    (("2024-09-28", "2025-10-04"), (2025, 9)),                # 52/53-week year: first days of a month
    (("2024-01-01", "2025-01-03"), (2024, 12)),               # ... count as the month before, even across a year
    (("2025-02-01",), (2025, 1)),                             # a retailer's late-January year
])
def test_the_fiscal_year_end_comes_from_the_latest_10k(ends, year_end):
    from ml.edgar_extractor import _fiscal_year_end
    assert _fiscal_year_end(_facts(*ends)) == year_end


def test_no_balance_sheet_means_no_fiscal_year_end():
    from ml.edgar_extractor import _fiscal_year_end
    assert _fiscal_year_end({"facts": {}}) is None


def test_edgar_answers_with_the_filers_fiscal_year_end(monkeypatch):
    import ml.edgar_extractor as ee

    def fake_fetch(ticker, n_years=3):
        return ee.ExtractedFinancials(ticker=ticker, company_name="MARCH CORP", years=[2023, 2024, 2025],
                                      data={}, warnings=[], fiscal_year_end_month=3)

    monkeypatch.setattr(ee, "fetch_financials", fake_fetch)
    monkeypatch.setattr(ee, "financials_to_session_state", lambda extracted: {"hist_h_rev_0": 1.0})
    body = client.get("/api/edgar/mar").json()
    assert body["years"] == [2023, 2024, 2025]
    assert body["fiscal_year_end_month"] == 3
