"""Settings, capabilities, optional ML features and SEC EDGAR."""
import importlib.util
import os
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException

from api.deps import resolve_settings
from api.schemas import (
    Capabilities, DealRiskRequest, DealRiskResponse, EdgarResponse, MultiplesRequest, MultiplesResponse, GrowthRequest, GrowthResponse, SettingsResponse, SettingsValidateRequest,
    SurrogateRequest, SurrogateResponse,
)
from api.serialize import to_json
from core.model_version import stamp
from core.config import DEFAULTS, build_corr_matrix, is_valid_corr
from core.deal import DealInputs, risk_model_inputs
from core.money import to_millions
from db.engine import is_configured
from core.montecarlo import MCInputs
from core.surrogate import surrogate_features, tail_unreliable, training_term_differences, training_terms

router = APIRouter(tags=["settings & integrations"])

SURROGATE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "surrogate")


def _installed(*pkgs):
    try:
        return all(importlib.util.find_spec(p) is not None for p in pkgs)
    except (ImportError, ValueError):
        return False


def _unavailable(feature, why):
    raise HTTPException(status_code=503, detail=f"{feature} is unavailable: {why}")


# ---------------------------------------------------------------------------
@router.get("/settings/defaults", response_model=SettingsResponse)
def get_settings_defaults():
    m = build_corr_matrix(DEFAULTS)
    return {"defaults": DEFAULTS, "correlation_matrix": m.tolist(),
            "correlation_valid": is_valid_corr(m)}


@router.post("/settings/validate", response_model=SettingsResponse)
def post_settings_validate(req: SettingsValidateRequest):
    """Apply overrides and report whether the correlation settings are usable."""
    cfg = resolve_settings(req.settings)
    m = build_corr_matrix(cfg)
    return {"defaults": cfg, "correlation_matrix": m.tolist(), "correlation_valid": is_valid_corr(m)}


@router.get("/capabilities", response_model=Capabilities)
def get_capabilities():
    """Which optional features this server can run."""
    # Checked by file, not by importing ml.surrogate.predict: that loads torch,
    # which takes seconds.
    surrogate = _installed("torch", "joblib") and all(
        os.path.exists(os.path.join(SURROGATE_DIR, f))
        for f in ("model.pt", "scaler_X.pkl", "scaler_y.pkl"))
    regime_installed = _installed("sklearn", "joblib", "hmmlearn", "fredapi")
    regime_trained = False
    if regime_installed:
        try:
            from ml.macro_regime import model_is_trained
            regime_trained = model_is_trained()
        except (ImportError, OSError):
            pass
    # The deal risk score reads stored averages and needs no ML packages (PLAN.md 5.2)
    return {"anomaly_detector": True, "surrogate": surrogate,
            "macro_regime_installed": regime_installed, "macro_regime_trained": regime_trained,
            "edgar": _installed("requests")}


# ---------------------------------------------------------------------------
def _peer_data(currency: str, history_groups: Optional[list] = None) -> tuple[dict, Optional[list], Optional[float]]:
    """What the deal risk score and the multiple predictor read from
    storage: the industry averages (with ``history_groups``' history), the
    approved reference transactions (None while the library is off) and US
    dollars per unit of ``currency`` (None when not known). A server without
    a database has none of them, and they say "not enough data"."""
    if not is_configured():
        return {}, None, None
    from db import benchmarks, economy
    from db import references as stored_references
    from library import switch
    tables, _ = benchmarks.all_tables(history_groups=history_groups)
    library = stored_references.approved() if switch.enabled() else None
    if currency == "USD":
        return tables, library, 1.0
    # The ECB's newest rates are units per euro
    day = economy.fx_on()
    per_euro = {**day.rates, "EUR": 1.0} if day else {}
    usd, units = per_euro.get("USD"), per_euro.get(currency)
    return tables, library, (usd / units if usd and units else None)


@router.post("/ml/deal-risk", response_model=DealRiskResponse)
def post_deal_risk(req: DealRiskRequest):
    """The deal against companies and deals like it in its region, sector
    and size (PLAN.md 5.2): its leverage, price and margin beside its
    industry's, the score where its card says it beats leverage alone, and
    the reference transactions like it when the library is on."""
    from ml.anomaly_detector import DealShape, assess
    d = DealInputs(**req.inputs.model_dump())
    kw = risk_model_inputs(d, req.senior_x, req.mezz_x)
    tables, library, usd_per = _peer_data(d.currency)
    ev_usd_m = d.entry_mult * to_millions(d.ebitda, d.unit) * usd_per if usd_per else None
    answer = assess(DealShape(country=d.country, industry=d.industry, leverage=kw["leverage"],
                              entry_multiple=kw["entry_mult"], ebitda_margin=kw["ebitda_margin"]),
                    tables, library, ev_usd_m)
    # The score reads no Settings
    return {**answer, "inputs": to_json(kw), "model": stamp(None)}


def _today() -> date:
    return datetime.now(timezone.utc).date()


@router.post("/ml/multiples", response_model=MultiplesResponse)
def post_multiples(req: MultiplesRequest):
    """Entry and exit EV/EBITDA ranges for the deal's industry in its region
    (PLAN.md 5.4), each where its card says it beats the region's whole
    market, with the comparables: the industry in every group, the region's
    industries in the same sector and, when the library is on, the
    reference transactions like it."""
    from ml.multiple_predictor import history_groups, predict
    d = DealInputs(**req.inputs.model_dump())
    # Only the history of the groups the deal can be ranged in: the rest is never read
    tables, library, usd_per = _peer_data(d.currency, history_groups(d.country))
    ev_usd_m = d.entry_mult * to_millions(d.ebitda, d.unit) * usd_per if usd_per else None
    answer = predict(d.country, d.industry, int(d.hold), tables, _today(), library, ev_usd_m)
    # The ranges read no Settings
    return {**answer, "model": stamp(None)}


@router.post("/ml/growth", response_model=GrowthResponse)
def post_growth(req: GrowthRequest):
    """The revenue growth range for the deal's sector and region over its
    hold (PLAN.md 5.5), and the Monte Carlo Settings that draw it, where
    its card says it beats the economy-wide range."""
    from benchmarks.starting import growth_figure
    from ml.growth_calibrator import calibrate
    d = DealInputs(**req.inputs.model_dump())
    anchor = None
    if d.country and is_configured():
        from db import economy
        series, _ = economy.all_series()
        anchor = growth_figure(series, d.country, _today())
    # The range reads no Settings
    return {**calibrate(d.country, d.industry, int(d.hold), anchor), "model": stamp(None)}


@router.post("/ml/surrogate", response_model=SurrogateResponse)
def post_surrogate(req: SurrogateRequest):
    """Instant IRR distribution estimate for the live sliders."""
    if not _installed("torch", "joblib"):
        _unavailable("live mode", "install requirements-ml.txt")
    try:
        from ml.surrogate.generate_data import TRAINING_FIXED
        from ml.surrogate.predict import SurrogatePredictor
    except (ImportError, OSError) as e:
        _unavailable("live mode", str(e))
    surrogate = SurrogatePredictor.get_instance()
    if surrogate is None:
        _unavailable("live mode", "the surrogate model is not trained")
    cfg = resolve_settings(req.settings)
    mc, deal = MCInputs(**req.mc.model_dump()), DealInputs(**req.deal.model_dump())
    pred = surrogate.predict(**surrogate_features(mc, deal, **req.sliders.model_dump()))
    return {
        "prediction": to_json(pred),
        "tail_unreliable": tail_unreliable(pred.p_wipeout),
        "term_differences": to_json(training_term_differences(mc, deal, cfg, TRAINING_FIXED)),
        "training_deal": to_json(training_terms(mc, deal, cfg, TRAINING_FIXED)),
        "model": stamp(cfg),
    }


@router.post("/ml/macro-regime")
def post_macro_regime():
    """Classify the current macro regime from FRED data."""
    if not _installed("sklearn", "joblib", "hmmlearn", "fredapi"):
        _unavailable("macro regime detection", "install requirements-ml.txt")
    from ml.macro_regime import get_current_regime, model_is_trained
    if not model_is_trained():
        _unavailable("macro regime detection",
                     "set FRED_API_KEY and run `python -m ml.macro_regime`")
    try:
        return to_json(get_current_regime())
    except Exception as e:      # network or API-key failures
        raise HTTPException(status_code=502, detail=f"could not classify the regime: {e}") from None


# ---------------------------------------------------------------------------
@router.get("/edgar/{ticker}", response_model=EdgarResponse)
def get_edgar(ticker: str):
    """Historical financials of a company filing with the SEC, as forecasting and deal
    inputs: a US filer's 10-K (US GAAP) or a foreign filer's 20-F or 40-F (IFRS),
    in the statements' own currency."""
    if not _installed("requests"):
        _unavailable("EDGAR autofill", "install requests")
    from ml.edgar_extractor import fetch_financials, financials_to_session_state
    ticker = ticker.strip().upper()
    if not ticker.isalnum() and not ticker.replace(".", "").replace("-", "").isalnum():
        raise HTTPException(422, "invalid ticker")
    try:
        extracted = fetch_financials(ticker, n_years=3)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except Exception as e:      # network failures, SEC outages
        raise HTTPException(status_code=502, detail=f"SEC EDGAR request failed: {e}") from None
    flat = financials_to_session_state(extracted)
    history = {}
    for key, value in flat.items():                   # hist_h_rev_0 -> h_rev[0]
        field, year = key[len("hist_"):].rsplit("_", 1)
        history.setdefault(field, {})[int(year)] = value
    history = {f: [by_year[j] for j in sorted(by_year)] for f, by_year in history.items()}
    # The extractor reads the statements in their own currency (US dollars for
    # a 10-K) and divides by a million
    money = {"currency": extracted.currency, "unit": "millions"}
    leases = extracted.leases
    warnings = list(extracted.warnings)
    latest_liability = (leases.get("lease_liability") or [0.0])[-1]
    if not latest_liability:
        warnings.append(f"The filing tags no lease liability for FY{extracted.years[-1] if extracted.years else ''}: "
                        "enter it from the annual report if the company leases")
    latest_cost = (leases.get("lease_cost") or [0.0])[-1]
    if latest_cost and extracted.accounting_standard == "ifrs" and not extracted.lease_interest_tagged:
        warnings.append("The lease cost is the principal repaid on lease liabilities only: the filing "
                        "tags no interest on them, so add it from the annual report's lease note")
    if extracted.lease_cost_basis == "LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths":
        warnings.append("The filing tags no lease cost: the lease cost shown is each year's lease "
                        "payments due over the next twelve months")
    data = extracted.data
    deal_inputs = {
        # EBITDA as the standard reports it: before lease costs under IFRS 16
        "ebitda": (data.get("operating_income") or [0.0])[-1]
                  + (data.get("depreciation_amortization") or [0.0])[-1],
        **money, "accounting_standard": extracted.accounting_standard,
        "lease_cost": (leases.get("lease_cost") or [0.0])[-1], "lease_liability": latest_liability,
    }
    return {"ticker": ticker, "company_name": extracted.company_name,
            "years": list(extracted.years), "history": to_json(history),
            "warnings": warnings,
            "fiscal_year_end_month": extracted.fiscal_year_end_month,
            "money": money, "accounting_standard": extracted.accounting_standard,
            "leases": to_json(leases), "deal_inputs": to_json(deal_inputs)}
