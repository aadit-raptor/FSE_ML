"""Request and response schemas.

Input bounds mirror the Streamlit input widgets. Percentages are numbers like
60.0 (not 0.60) and money is a plain number in the deal's currency and unit
(``Money``; US dollar millions unless the request says otherwise), matching the
model's inputs; engine outputs keep the engine's own units (IRR 0.157 = 15.7%).
Every response that holds money says which currency and unit it is in.
"""
from datetime import date, datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from api.limits import MAX_FORECAST_PATHS, MAX_SIMULATION_PATHS

from api.serialize import model_from_dataclass
from core.money import DEFAULT_CURRENCY, DEFAULT_UNIT
from core.forecasting import ForecastYear, HistoricalYear
from lbo_engine.cashflow_model import CashFlowResult
from lbo_engine.operating_model import OperatingModelResult
from lbo_engine.returns import ReturnsResult
from lbo_engine.tax import TaxSchedule

SettingValue = Union[float, int, bool]


class Strict(BaseModel):
    # JSON lets a caller write Infinity or NaN, and Pydantic accepts them as
    # floats by default. They survive every bound (NaN compares false against
    # ge and le) and only fail deep inside the model, as a 500 whose traceback
    # carries the deal's own figures into the log. Refuse them at the edge.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _path_cap(maximum: int):
    """At most ``maximum`` simulation paths, refused with a sentence that says so
    (api/limits.py explains the numbers)."""
    def check(n: int) -> int:
        if n > maximum:
            raise PydanticCustomError(
                "too_many_paths",
                "This server runs at most {maximum} paths in one simulation; this one asks for "
                "{n}. Use {maximum} or fewer.",
                {"maximum": f"{maximum:,}", "n": f"{n:,}"})
        return n
    # The schema still states the maximum, so clients can bound their inputs
    return Annotated[int, AfterValidator(check), Field(json_schema_extra={"maximum": maximum})]


SimulationPaths = _path_cap(MAX_SIMULATION_PATHS)
ForecastPaths = _path_cap(MAX_FORECAST_PATHS)


# ---------------------------------------------------------------------------
# Money (PLAN.md 2.2)
# ---------------------------------------------------------------------------
MoneyUnit = Literal["thousands", "millions", "billions"]
CurrencyCode = Annotated[str, Field(pattern=r"^[A-Z]{3}$", min_length=3, max_length=3,
                                    description="ISO 4217, e.g. EUR")]


# How long numbers are grouped (PLAN.md 2.3a); db/users.py DIGIT_GROUPINGS
DigitGrouping = Literal["locale", "thousands", "lakh"]
# Accounting standard and lease view (PLAN.md 2.6, core/accounting.py); "" = not stated
AccountingStandard = Literal["", "ifrs", "us_gaap"]
LeaseView = Literal["", "pre_ifrs16", "post_ifrs16"]


class Money(Strict):
    """What money figures are counted in. Any ISO 4217 currency; the model never
    calculates with the code, so the same numbers give the same results in any
    currency."""
    currency: CurrencyCode = DEFAULT_CURRENCY
    unit: MoneyUnit = Field(DEFAULT_UNIT, description="Money figures are thousands, millions or billions")


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------
TrancheKind = Literal[
    "amortising_term_loan", "institutional_term_loan", "unitranche", "second_lien",
    "senior_notes", "pik_notes", "vendor_loan", "revolver", "shareholder_loan",
]
ReferenceRate = Literal[
    "SOFR", "SONIA", "ESTR", "EURIBOR", "TONA", "SARON", "BBSY", "MIBOR", "custom",
]
# A rate can be below zero: ESTR and SARON have both spent years there, and a
# floor of zero is a term someone negotiates, not a law of arithmetic. Bounding
# these at zero would make the model wrong outside the dollar and the pound.
Rate = Annotated[float, Field(ge=-5, le=50)]
# Long enough for any facility's life; also bounds what a stored deal can hold
MAX_TRANCHE_YEARS = 30
# Larger than any deal anyone will model, in any currency and unit, and small
# enough that the arithmetic never runs out of floating-point range
MAX_MONEY = 1e15

# ---------------------------------------------------------------------------
# Which model produced a result (PLAN.md 3.1, core/model_version.py)
# ---------------------------------------------------------------------------
_HEX = r"^[0-9a-f]{1,64}$"


class ModelStamp(Strict):
    """Which model produced a result: on every result, and sent back with an
    export so the workbook says which model made the figures it holds."""
    engine_version: str = Field(max_length=20, pattern=r"^\d+\.\d+\.\d+$",
                                description="Raised whenever any deal's numbers would move; "
                                            "see MODEL_CHANGELOG.md")
    commit: Optional[str] = Field(None, max_length=64, pattern=_HEX,
                                  description="Git commit the API was built from; null locally")
    settings_fingerprint: Optional[str] = Field(
        None, max_length=64, pattern=_HEX,
        description="Fingerprint of the resolved Settings the run used; null when it used none")
    data_vintage: str = Field(max_length=10, pattern=r"^\d{4}(-\d{2}){0,2}$",
                              description="Newest edition date among the published data sets")
    data_fingerprint: str = Field(max_length=64, pattern=_HEX)
    # Written into an export's About sheet: ids and dates only, so nothing a
    # caller sends can start with "=" and become a formula
    data_sets: Dict[Annotated[str, Field(max_length=40, pattern=r"^[a-z0-9_]+$")],
                    Annotated[str, Field(max_length=10, pattern=r"^\d{4}(-\d{2}){0,2}$")]] = Field(
        default_factory=dict, max_length=30,
        description="Each published data set the model reads, with its edition date")


class SavedModel(BaseModel):
    """The stamp a saved deal or version keeps, with what the deal gave then."""
    engine_version: str
    commit: Optional[str] = None
    settings_fingerprint: Optional[str] = None
    data_vintage: str
    data_fingerprint: str
    content: Optional[str] = Field(
        None, description="Fingerprint of the deal content the stamp was made for")
    irr: Optional[float] = Field(None, description="Engine units: 0.157 = 15.7%; null if unfundable")
    moic: Optional[float] = None


class ModelCheck(BaseModel):
    """Whether a saved deal's results have changed since it was saved."""
    status: Literal["changed", "unchanged", "unknown"] = Field(
        description="changed: IRR or MOIC differs today; unknown: saved before stamps were kept")
    saved: Optional[SavedModel]
    now: SavedModel
    causes: List[Literal["engine_version", "data", "settings"]] = Field(
        description="What differs between the two stamps")



class TrancheIn(Strict):
    """One facility in the deal's debt structure (PLAN.md 2.4, core/debt.py).

    ``amount`` is the facility's full size, in the deal's currency and unit.
    For a revolver that is the commitment, of which ``drawn_pct`` is drawn at
    close: only the drawn part is a source of funds, and only the undrawn part
    pays a commitment fee.

    Pricing is fixed (``fixed_rate``) or floating, where the all-in rate for a
    year is ``max(reference, floor) + margin`` -- the floor bites on the
    reference before the margin, as a loan agreement writes it.
    ``reference_path`` gives the reference year by year and repeats its last
    year when the deal runs longer; empty, the flat ``reference_level`` stands
    for every year.
    """
    name: str = Field(min_length=1, max_length=60)
    kind: TrancheKind
    amount: float = Field(0.0, ge=0, le=MAX_MONEY,
                          description="Facility size (in the deal's currency and unit)")
    drawn_pct: float = Field(100.0, ge=0, le=100, description="% of the facility drawn at close")
    floating: bool = False
    fixed_rate: Rate = Field(0.0, description="All-in rate (%) when not floating")
    reference_rate: ReferenceRate = "custom"
    reference_level: Rate = Field(0.0, description="The reference rate (%) when no path is given")
    reference_path: List[Rate] = Field(default_factory=list, max_length=MAX_TRANCHE_YEARS,
                                       description="The reference rate (%) year by year")
    margin: float = Field(0.0, ge=0, le=50, description="Margin over the reference (%)")
    floor: Rate = Field(0.0, description="Floor on the reference (%)")
    maturity_years: int = Field(7, ge=1, le=MAX_TRANCHE_YEARS, strict=True)
    amort_pct: float = Field(0.0, ge=0, le=100, description="% of the original principal repaid a year")
    amort_schedule: List[Annotated[float, Field(ge=0, le=MAX_MONEY)]] = Field(
        default_factory=list, max_length=MAX_TRANCHE_YEARS,
        description="Repayment a year; overrides amort_pct")
    upfront_fee_pct: float = Field(0.0, ge=0, le=10, description="Arrangement fee at close (%)")
    commitment_fee_pct: float = Field(0.0, ge=0, le=5, description="Yearly fee on the undrawn commitment (%)")
    sweep: bool = Field(False, description="Takes the cash sweep")
    sweep_share: float = Field(100.0, ge=0, le=100, description="% of the cash available it may take")
    pik_share: float = Field(0.0, ge=0, le=100, description="% of the coupon that accrues to principal")
    sweep_priority: int = Field(0, ge=0, le=99, strict=True, description="0 = its position in the list")
    allow_redraw: bool = Field(False, description="Draws to cover a cash shortfall (revolver)")


# Each tranche multiplies the work of a run: the exit-sensitivity grid reruns
# the whole model once per holding period, so the schedule is built five times
# over. Twelve facilities is more than any real structure and keeps a run well
# inside the timeout (api/limits.py).
MAX_TRANCHES = 12


class DealInputsIn(Strict):
    """Deal wizard inputs.

    debt_pct and senior_pct default to the model's stored defaults (60% / 70%).
    The Streamlit wizard instead derives them on load from its financing
    inputs -- 3.4x senior + 0.8x mezz EBITDA, i.e. 42% and ~81% -- which is why
    its default deal shows a lower IRR. Use /deal/sources-and-uses to derive
    them from debt multiples the same way.
    """
    ebitda: float = Field(100.0, gt=0, description="LTM EBITDA (in currency and unit)")
    entry_mult: float = Field(10.0, gt=0, description="Entry EV / EBITDA (x)")
    exit_mult: float = Field(11.0, gt=0, description="Exit EV / EBITDA (x)")
    hold: int = Field(5, ge=1, le=15, description="Holding period (years)")
    growth: float = Field(5.0, ge=-50, le=100, description="Revenue growth (%)")
    gross_margin: float = Field(40.0, ge=0, le=100, description="Gross margin (%)")
    opex: float = Field(18.0, ge=0, le=100, description="OpEx / revenue (%)")
    tax: float = Field(25.0, ge=0, le=100, description="Tax rate (%)")
    da: float = Field(4.0, ge=0, le=100, description="D&A / revenue (%)")
    debt_pct: float = Field(60.0, ge=0, le=99, description="Total debt / EV (%)")
    senior_pct: float = Field(70.0, ge=0, le=100, description="Senior / total debt (%)")
    base_rate: float = Field(6.5, ge=0, le=50, description="Senior interest rate (%)")
    mezz_spread: float = Field(4.0, ge=0, le=50, description="Mezz spread over senior (%)")
    capex: float = Field(4.0, ge=0, le=100, description="Capex / revenue (%)")
    nwc: float = Field(1.0, ge=-100, le=100, description="Change in NWC / revenue (%)")
    mincash: float = Field(0.0, ge=0, description="Minimum cash (in currency and unit)")
    wsp_mode: bool = Field(False, description="Use AR/inventory/AP days instead of flat NWC")
    ar_days: float = Field(45.0, ge=0, le=365)
    inv_days: float = Field(30.0, ge=0, le=365)
    ap_days: float = Field(60.0, ge=0, le=365)
    currency: CurrencyCode = Field(DEFAULT_CURRENCY, description="The deal's currency (ISO 4217)")
    unit: MoneyUnit = Field(DEFAULT_UNIT, description="The deal's money figures are thousands, millions or billions")
    # Labels only (PLAN.md 2.3a): the model counts years from 1 whatever these say
    fiscal_year_end_month: int = Field(12, ge=1, le=12, strict=True,
                                       description="Month the deal's fiscal year ends (12 = December)")
    first_fiscal_year: Optional[int] = Field(
        None, ge=1900, le=2200, strict=True,
        description="Fiscal year of the first projected year, named by the year it ends in; "
                    "none labels years Y1, Y2 ...")
    # Labels only (PLAN.md 4.3): where the deal's starting figures were
    # sourced for. The model never reads them; stored only when set
    country: str = Field("", pattern=r"^([A-Z]{2})?$",
                         description="The deal's country (ISO 3166-1 alpha-2) its starting figures were "
                                     "sourced for, a label only")
    industry: str = Field("", pattern=r"^[a-z0-9_]{0,60}$",
                          description="The industry its starting figures were sourced for "
                                      "(/api/benchmarks/industries), a label only")
    tranches: List[TrancheIn] = Field(
        default_factory=list, max_length=MAX_TRANCHES,
        description="The deal's debt, facility by facility (PLAN.md 2.4). Empty keeps the "
                    "two-tranche sizing above; a list replaces debt_pct, senior_pct, base_rate "
                    "and mezz_spread entirely, and is swept in the order it is given.")
    # Tax rules beyond the flat rate (PLAN.md 2.5, core/tax.py). Every default
    # is off; they are stored and sent only when set (db/deals.py).
    # A label, so any two-letter code is accepted: validating it against
    # today's presets would make every saved deal naming a preset retired
    # later unreadable
    tax_preset: str = Field(
        "", pattern=r"^([A-Z]{2})?$",
        description="The country preset last applied (ISO 3166 code), a label only; "
                    "the rules below are what the model reads")
    tax_interest_limit: Literal["none", "ebitda_share", "fixed"] = Field(
        "none", description="Interest deductibility: none, a share of EBITDA (never below "
                            "tax_interest_limit_amount), or a fixed amount (tax_interest_limit_amount "
                            "a year; left at 0, no interest is deductible)")
    tax_interest_limit_pct: float = Field(30.0, ge=0, le=100, description="Share of EBITDA (%)")
    tax_interest_limit_amount: float = Field(
        0.0, ge=0, le=MAX_MONEY,
        description="The fixed cap, or the allowance always deductible (in currency and unit)")
    tax_loss_carryforward: bool = Field(False, description="Carry tax losses forward")
    tax_loss_limit_pct: float = Field(
        100.0, ge=0, le=100, description="Share of profit above the allowance losses may offset (%)")
    tax_loss_limit_amount: float = Field(
        0.0, ge=0, le=MAX_MONEY, description="Profit losses may offset in full each year (in currency and unit)")
    tax_minimum_pct: float = Field(0.0, ge=0, le=100, description="Minimum tax on book profit (%)")
    accounting_standard: AccountingStandard = Field(
        "", description="The standard the EBITDA follows (PLAN.md 2.6): 'ifrs' (before lease costs, IFRS 16), "
                        "'us_gaap' (after operating lease costs), or '' for not stated")
    lease_view: LeaseView = Field(
        "", description="How the deal is priced: 'pre_ifrs16' (EBITDA after lease costs, leases not debt), "
                        "'post_ifrs16' (EBITDA before lease costs, the lease liability counted with net debt), "
                        "or '' for the standard's own view")
    lease_cost: float = Field(0.0, ge=0, le=MAX_MONEY, description="What the leases cost a year (in currency and unit)")
    lease_liability: float = Field(
        0.0, ge=0, le=MAX_MONEY, description="The lease liability at close (in currency and unit)")
    # PLAN.md 5.3: read only by the distress predictor; stored only when set
    business_risk: int = Field(
        4, ge=1, le=6, description="S&P's business risk profile, 1 (excellent) to 6 (vulnerable); with "
                                   "leverage it gives each year's rating band (Corporate Methodology, Table 3)")

    @model_validator(mode="after")
    def _only_a_committed_line_is_partly_drawn(self) -> "DealInputsIn":
        """A term loan or a bond is drawn in full on day one; it is a revolver
        that leaves part of itself undrawn and pays to keep it available. A
        term loan with drawn_pct below 100 is almost always someone meaning to
        make the facility smaller, so say that rather than quietly funding
        less than the deal needs."""
        for tranche in self.tranches:
            if tranche.drawn_pct < 100 and tranche.kind != "revolver":
                raise ValueError(
                    f"{tranche.name}: only a revolving credit facility can be partly drawn at "
                    f"close. Set drawn_pct to 100, or reduce the facility's size.")
        return self

    def money(self) -> "Money":
        return Money(currency=self.currency, unit=self.unit)


class MCInputsIn(Strict):
    n: SimulationPaths = Field(50000, ge=1000, description="Scenarios to simulate")
    ebitda: float = Field(100.0, ge=1)
    entry_mult: float = Field(10.0, ge=1)
    hold: int = Field(5, ge=1, le=15)
    hurdle: float = Field(20.0, ge=0, description="Hurdle IRR (%)")
    growth_mean: float = 5.0
    growth_std: float = Field(3.0, ge=0.1)
    exit_mean: float = Field(10.0, ge=1)
    exit_std: float = Field(1.5, ge=0.1)
    rate_mean: float = Field(6.5, ge=0)
    rate_std: float = Field(1.5, ge=0.1)
    gm_mean: float = Field(40.0, ge=1, le=99)
    gm_std: float = Field(3.0, ge=0.1)


Scenario = Literal["recession", "base", "bull", "stagflation"]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
class SettingsResponse(BaseModel):
    defaults: Dict[str, SettingValue]
    correlation_matrix: List[List[float]]
    correlation_valid: bool


class SettingsValidateRequest(Strict):
    settings: Dict[str, SettingValue] = {}


# ---------------------------------------------------------------------------
# Deal
# ---------------------------------------------------------------------------
class SourcesUsesRequest(Strict):
    ebitda: float = Field(100.0, gt=0)
    entry_mult: float = Field(10.0, gt=0)
    senior_x: float = Field(3.4, ge=0, description="Senior debt (x EBITDA)")
    mezz_x: float = Field(0.8, ge=0, description="Mezz debt (x EBITDA)")
    mincash: float = Field(0.0, ge=0, description="Minimum cash left on the balance sheet")
    tranches: List[TrancheIn] = Field(
        default_factory=list, max_length=MAX_TRANCHES,
        description="The deal's facilities (PLAN.md 2.4). Given, they are the sources of debt "
                    "and senior_x / mezz_x are ignored.")
    accounting_standard: AccountingStandard = Field(
        "", description="The standard the EBITDA follows (PLAN.md 2.6): 'ifrs' (before lease costs, IFRS 16), "
                        "'us_gaap' (after operating lease costs), or '' for not stated")
    lease_view: LeaseView = Field(
        "", description="How the deal is priced: 'pre_ifrs16' (EBITDA after lease costs, leases not debt), "
                        "'post_ifrs16' (EBITDA before lease costs, the lease liability counted with net debt), "
                        "or '' for the standard's own view")
    lease_cost: float = Field(0.0, ge=0, le=MAX_MONEY, description="What the leases cost a year (in currency and unit)")
    lease_liability: float = Field(
        0.0, ge=0, le=MAX_MONEY, description="The lease liability at close (in currency and unit)")
    settings: Dict[str, SettingValue] = {}
    money: Money = Money()


class TrancheSource(BaseModel):
    name: str
    kind: str
    amount: float


class SourcesUsesResponse(BaseModel):
    senior_debt: Optional[float] = Field(None, description="Only when the deal is sized by percentages")
    mezz_debt: Optional[float] = Field(None, description="Only when the deal is sized by percentages")
    tranches: List[TrancheSource] = Field(
        default_factory=list, description="One source per facility, when the deal lists its tranches")
    total_debt: Optional[float] = Field(None, description="Debt drawn at close, when listed by tranche")
    tranche_fees: float = Field(0.0, description="Arrangement fees the facilities charge at close")
    sponsor_equity: float
    total_sources: float
    equity_purchase_price: float = Field(description="Entry EV less any lease liability taken over")
    lease_liability: float = Field(0.0, description="Lease liability taken over with the business, when counted as debt")
    transaction_fees: float
    financing_fees: float
    other_uses: float
    cash_to_balance_sheet: float
    total_uses: float
    check: float
    balanced: bool
    debt_pct: Optional[float] = Field(None, description="Implied total debt / EV (%)")
    senior_pct: Optional[float] = Field(None, description="Implied senior / total debt (%)")
    money: Money
    model: ModelStamp


class DealRunRequest(Strict):
    inputs: DealInputsIn = DealInputsIn()
    settings: Dict[str, SettingValue] = {}


class BridgeStep(BaseModel):
    key: str
    label: str
    value: Optional[float]
    is_total: bool
    pct_of_gain: Optional[float]


class EquityBridge(BaseModel):
    entry_equity: Optional[float]
    entry_costs: Optional[float]
    ebitda_growth: Optional[float]
    multiple_expansion: Optional[float]
    deleveraging: Optional[float]
    exit_equity: Optional[float]
    total_gain: Optional[float]
    residual: Optional[float]
    entry_costs_pct: Optional[float]
    ebitda_growth_pct: Optional[float]
    multiple_expansion_pct: Optional[float]
    deleveraging_pct: Optional[float]


class ExitSensitivity(BaseModel):
    metric: str
    exit_multiples: List[float]
    holding_periods: List[int]
    table: List[List[Optional[float]]] = Field(description="rows = exit multiples, cols = holding periods")


class TrancheYear(BaseModel):
    year: int
    tranche_name: str
    beginning_balance: Optional[float]
    mandatory_repayment: Optional[float]
    cash_sweep: Optional[float]
    ending_balance: Optional[float]
    interest_expense: Optional[float]
    interest_rate: Optional[float] = Field(description="The year's all-in rate, as a decimal")
    # Global debt structures (PLAN.md 2.4). Zero for a facility that pays its
    # whole coupon in cash and has nothing undrawn.
    cash_interest: Optional[float] = Field(None, description="The part of the coupon actually paid")
    pik_interest: Optional[float] = Field(None, description="The part that accrues to principal")
    commitment_fee: Optional[float] = Field(None, description="Charged on the undrawn commitment")
    undrawn: Optional[float] = Field(None, description="Commitment nobody has taken down")
    redrawn: Optional[float] = Field(None, description="Drawn in the year to cover a cash shortfall")


class TrancheSummary(BaseModel):
    """One row per facility, for the capital-structure table."""
    name: str
    kind: str
    amount: float = Field(description="Drawn at close, in the deal's currency and unit")
    commitment: Optional[float] = Field(None, description="Facility limit, when part of it is undrawn")
    x_ebitda: float
    rate: float = Field(description="Year-one all-in rate, as a decimal")
    floating: bool
    reference_rate: Optional[str]
    maturity_years: int
    sweep_share: float = Field(description="% of the cash sweep it takes; 0 when it does not sweep")
    pik_share: float = Field(description="% of the coupon that accrues to principal")


class LeaseSummary(BaseModel):
    accounting_standard: str
    view: LeaseView
    counted_as_debt: bool = Field(description="Whether the lease liability is counted with net debt")
    operating_ebitda: float = Field(description="EBITDA after lease costs: what the operating model grows")
    valuation_ebitda: float = Field(description="EBITDA the entry and exit multiples are applied to")
    lease_cost: float
    lease_liability: float
    entry_ev: float
    net_debt_at_entry: float = Field(description="Debt at close less minimum cash, plus leases when counted")


class RiskSample(BaseModel):
    count: int
    what: str = Field(description="What was counted, e.g. issuers")
    first_year: int
    last_year: int


class RiskSource(BaseModel):
    """Where a warning's figures come from (core/risk_sources.py)."""
    id: str
    publisher: str
    title: str
    published: Optional[str] = Field(None, description="Publication date, ISO, as precise as the source gives")
    detail: str = Field(description="The table or passage the figure is read from")
    url: Optional[str] = None
    sample: Optional[RiskSample] = Field(None, description="How many observations stand behind it, when the source says")


class RiskWarning(BaseModel):
    """A risk warning computed from the deal and published data (PLAN.md 2.8).

    Numbers only: the words are the web app's (``warnings.<id>``). Every figure
    is computed from the deal's model run or read from a source listed in
    ``sources``. Money figures (``unfunded``, ``unfunded_total``,
    ``repayment_due``) are in the deal's unit; years count from 1."""
    id: Literal["leverage_above_guidance", "implied_rating", "interest_exceeds_ebitda", "unfunded_repayment"]
    figures: Dict[str, Optional[float]]
    labels: Dict[str, str] = Field(default_factory=dict, description="Ratings, as the sources name them")
    sources: List[RiskSource]


class CreditView(BaseModel):
    """Year-one interest coverage and the default risk it implies, shown on
    every deal (PLAN.md 4.6). The same reading as the ``implied_rating``
    warning, which is raised only for a speculative grade."""
    coverage: Optional[float] = Field(description="Year-one EBIT / interest; none without interest")
    rating: Optional[str] = Field(description="Damodaran's coverage band's rating")
    study_row: Optional[str] = Field(description="The row of S&P's study read for it")
    band_low: Optional[float]
    band_high: Optional[float]
    years: int = Field(description="The deal's hold, the horizon of the default rate")
    default_pct: Optional[float] = Field(description="S&P's average cumulative default rate (%) over the hold")
    speculative: Optional[bool]
    sources: List[RiskSource]


# The distress predictor (PLAN.md 5.3, ml/distress_model.py)
RatingBand = Literal["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C"]


class DistressCard(BaseModel):
    """What the predictor's card says for the deal's region."""
    verdict: Literal["beats_baseline", "does_not_beat_baseline", "not_enough_data"]
    cases: int
    model: Optional[float] = Field(None, description="The card's headline statistic (AUC) for the predictor")
    baseline: Optional[float] = Field(None, description="The same for the year-one default risk")


class DistressContext(BaseModel):
    region: Optional[Literal["us", "europe", "emerging", "other_developed"]] = Field(
        description="S&P's region for the deal's country; none without a country")
    table: Literal["us", "europe", "emerging", "global"] = Field(
        description="The block of S&P's study read: Table 25's for the region, else Table 24 (global)")
    business_risk: int = Field(description="The business risk profile read (1 excellent .. 6 vulnerable)")
    shown: bool = Field(description="Whether the card lets the probabilities be shown in this region; "
                                    "they are null when not")
    card: DistressCard
    sources: List[RiskSource]


class DealDistressYear(BaseModel):
    year: int
    coverage: Optional[float] = Field(description="EBIT / interest; none without interest")
    leverage: Optional[float] = Field(description="Debt at the start of the year / the year's EBITDA (0 without "
                                                  "debt); none for debt against an EBITDA that is not positive")
    coverage_band: RatingBand
    leverage_band: RatingBand
    leverage_profile: int = Field(description="S&P's financial risk profile for the leverage, 1-6 (Table 17)")
    band: RatingBand = Field(description="The weaker of the two reads")
    probability: Optional[float] = Field(description="Chance of default in this year (fraction); null where "
                                                     "not shown")
    cumulative: Optional[float] = Field(description="Chance of default by the end of this year; null where "
                                                    "not shown")


class DealDistress(DistressContext):
    """Each year's rating band and, where the card allows, chance of default."""
    years: List[DealDistressYear]


class SimulatedDistressYear(BaseModel):
    year: int
    band_shares: Dict[RatingBand, float] = Field(description="Share of paths in each band")
    probability: Optional[float] = Field(description="Mean chance of default in this year over the paths; "
                                                     "null where not shown")
    cumulative: Optional[float] = Field(description="Mean chance of default by the end of this year; null "
                                                    "where not shown")


class SimulatedDistress(DistressContext):
    """The distress predictor on every simulated path."""
    years: List[SimulatedDistressYear]


class DealRunResponse(BaseModel):
    returns: model_from_dataclass(ReturnsResult)
    operating_model: model_from_dataclass(OperatingModelResult)
    cash_flow: model_from_dataclass(CashFlowResult)
    debt_schedule: Dict[str, object] = Field(description="Totals per year; per-tranche detail in tranches")
    tranches: Dict[str, List[TrancheYear]]
    capital_structure: List[TrancheSummary] = Field(
        default_factory=list,
        description="One row per facility, when the deal lists its tranches (PLAN.md 2.4); "
                    "empty for a deal sized by debt_pct and senior_pct")
    equity_bridge: EquityBridge
    bridge_steps: List[BridgeStep]
    exit_sensitivity: ExitSensitivity
    interest_converged: bool
    tax: Optional[model_from_dataclass(TaxSchedule)] = Field(
        None, description="The tax computation year by year, when the deal has tax rules "
                          "(PLAN.md 2.5); none for a flat rate on positive profit")
    leases: Optional[LeaseSummary] = Field(
        None, description="What the deal's leases did to its value (PLAN.md 2.6); none without leases")
    risk_warnings: List[RiskWarning] = Field(
        default_factory=list,
        description="Risk warnings the deal's own figures raise, each with its sources (PLAN.md 2.8)")
    credit: CreditView
    distress: DealDistress
    money: Money
    model: ModelStamp


class TaxPresetOut(BaseModel):
    """A country's headline tax rules, as a starting point (core/tax.py).
    Percentages as numbers like 25.0; amounts in millions of ``currency``."""
    code: str = Field(description="ISO 3166-1 alpha-2 country code")
    currency: str = Field(description="The currency of the amounts (ISO 4217)")
    rate: float
    interest_limit: str
    interest_limit_pct: float
    interest_limit_amount: float
    loss_carryforward: bool
    loss_limit_pct: float
    loss_limit_amount: float
    minimum_pct: float
    source: str
    note: str
    as_of: str = Field(description="When the numbers were last checked (YYYY-MM)")


class TaxPresetsResponse(BaseModel):
    presets: List[TaxPresetOut]


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
class MonteCarloRequest(Strict):
    mc: MCInputsIn = MCInputsIn()
    deal: DealInputsIn = DealInputsIn()
    settings: Dict[str, SettingValue] = {}
    scenario: Optional[Scenario] = None
    seed: Optional[int] = Field(None, description="Fix the random draws for reproducible results")
    histogram_bins: int = Field(80, ge=10, le=400)
    scatter_points: int = Field(2000, ge=0, le=20000)


class Histogram(BaseModel):
    edges: List[float]
    density: List[float]


class PercentileCurve(BaseModel):
    percentiles: List[float]
    values: List[float]


class RiskSummary(BaseModel):
    mean_irr: Optional[float]
    median_irr: Optional[float]
    p5_irr: Optional[float]
    p95_irr: Optional[float]
    p_above_hurdle: Optional[float]
    wipeout_rate: Optional[float]
    p_loss: Optional[float] = Field(None, description="Share of paths with MOIC below 1 (PLAN.md 4.6)")
    hurdle: float


class DriverSensitivity(BaseModel):
    driver: str
    spearman_rho: Optional[float]


class DriverFit(BaseModel):
    slope: Optional[float]
    intercept: Optional[float]
    r: Optional[float]


class DriverContribution(BaseModel):
    driver: str
    irr: Optional[float] = Field(description="What the driver adds to the case's IRR, as a fraction")


class ExplainedCase(BaseModel):
    case: Literal["downside", "upside"]
    paths: int = Field(description="How many simulated paths the figures average")
    tail_paths: int = Field(description="How many paths the tail holds; more than `paths` when it is sampled")
    irr: Optional[float] = Field(description="The mean IRR of those `paths`: base_irr plus the contributions")
    contributions: List[DriverContribution]


class DriverExplanations(BaseModel):
    """Why the worst and best simulated paths are where they are (PLAN.md
    5.6): each tail's mean IRR, split exactly between the drivers."""
    share: float = Field(description="The share of paths in each case")
    base_irr: Optional[float] = Field(description="The simulation's IRR with every driver at its mean")
    cases: List[ExplainedCase]


class Heatmap(BaseModel):
    growth: List[float]
    exit_multiple: List[float]
    irr: List[List[Optional[float]]] = Field(description="rows = exit multiples, cols = growth")
    note: str


class MonteCarloResponse(BaseModel):
    n: int
    elapsed_ms: float
    scenario: Optional[Scenario]
    params: Dict[str, object]
    summary: RiskSummary
    irr_histogram: Histogram
    moic_histogram: Histogram
    irr_cdf: PercentileCurve
    drivers: List[DriverSensitivity]
    driver_fits: Dict[str, DriverFit]
    correlations: Dict[str, object]
    scatter: Dict[str, List[Optional[float]]]
    heatmap: Heatmap
    distress: SimulatedDistress
    explanations: Optional[DriverExplanations] = Field(
        None, description="Left out when the run itself took too long to explain within the timeout")
    money: Money
    model: ModelStamp


class ScenarioStats(BaseModel):
    mean_irr: Optional[float]
    median_irr: Optional[float]
    p5_irr: Optional[float]
    p95_irr: Optional[float]
    p_above_hurdle: Optional[float]
    wipeout_rate: Optional[float]
    irr_box: Dict[str, Optional[float]]
    moic_box: Dict[str, Optional[float]]


class ScenariosRequest(Strict):
    mc: MCInputsIn = MCInputsIn(n=50000)
    deal: DealInputsIn = DealInputsIn()
    settings: Dict[str, SettingValue] = {}
    seed: Optional[int] = None


class ScenariosResponse(BaseModel):
    hurdle: float
    scenarios: Dict[str, ScenarioStats]
    money: Money
    model: ModelStamp


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
class HistoricalField(BaseModel):
    key: str
    default_latest: float
    step: float


class ForecastDefaultsResponse(BaseModel):
    n_hist: int
    n_fwd: int
    fields: List[HistoricalField]
    history: Dict[str, List[float]]
    assumption_keys: List[str]
    seeded_assumptions: Dict[str, float]
    money: Money = Field(description="What the default history is counted in")


class HistoryRequest(Strict):
    history: Dict[str, List[float]] = Field(description="field key -> one value per historical year, oldest first")
    money: Money = Field(Money(), description="The company's reporting currency and the unit its figures are in")
    accounting_standard: AccountingStandard = Field(
        "", description="The standard the company reports under (PLAN.md 2.6): a label, echoed back")


class HistoricalMetrics(BaseModel):
    revenue: Optional[float]
    revenue_growth: Optional[float]
    gross_margin: Optional[float]
    rd_pct: Optional[float]
    sga_pct: Optional[float]
    ebitda_margin: Optional[float]
    adj_ebitda_margin: Optional[float]


class SeedResponse(BaseModel):
    ltm: model_from_dataclass(HistoricalYear)
    historical_metrics: List[HistoricalMetrics]
    seeded_assumptions: Dict[str, float]
    money: Money
    accounting_standard: AccountingStandard = ""


class ForecastRunRequest(HistoryRequest):
    assumptions: Dict[str, List[float]] = Field(description="assumption key -> one value per forecast year")
    simulate: bool = False
    n_sim: ForecastPaths = Field(30000, ge=1000)


class Bands(BaseModel):
    p5: List[float]
    p25: List[float]
    p50: List[float]
    p75: List[float]
    p95: List[float]


class FinalStats(BaseModel):
    mean: float
    median: float
    p5: float
    p25: float
    p75: float
    p95: float
    deterministic: float


class TargetProbability(BaseModel):
    target: float
    probability: float
    scenario: str


class ForecastSimulation(BaseModel):
    n: int
    revenue_bands: Bands
    ebitda_bands: Bands
    revenue_final: FinalStats
    ebitda_final: FinalStats
    target_probabilities: List[TargetProbability]
    growth_final_mean: float


class ForecastRunResponse(BaseModel):
    ltm: model_from_dataclass(HistoricalYear)
    years: List[model_from_dataclass(ForecastYear)]
    revenue_cagr: Optional[float]
    opening_balance_gap: float
    forecast_balance_gaps: List[float] = Field(description="Gap each year introduced by the forecast itself")
    balanced: bool
    simulation: Optional[ForecastSimulation]
    money: Money
    model: ModelStamp
    accounting_standard: AccountingStandard = ""


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
class BacktestEntry(Strict):
    entry_ebitda: float = Field(gt=0)
    entry_multiple: float = Field(gt=0)
    exit_multiple: float = Field(gt=0)
    holding_period: int = Field(ge=1, le=15)
    debt_pct: float = Field(ge=0, le=99)
    senior_pct: float = Field(ge=0, le=100)
    base_rate: float = Field(ge=0, le=50)
    mezz_spread: float = Field(ge=0, le=50)
    revenue_growth: float
    gross_margin: float = Field(ge=0, le=100)
    opex_pct: float = Field(ge=0, le=100)
    da_pct: float = Field(ge=0, le=100)
    tax_rate: float = Field(ge=0, le=100)
    capex_pct: float = Field(ge=0, le=100)
    nwc_pct: float


class BacktestActuals(Strict):
    revenue: List[float]
    ebitda: List[float]
    net_income: List[float]
    fcf: List[float]
    total_debt: List[float]


class BacktestActualExit(Strict):
    exit_ev: float = Field(ge=0)
    net_debt_at_exit: float = Field(ge=0)
    sponsor_equity_entry: float = Field(ge=0)
    moic: float = Field(ge=0)
    irr: float = Field(description="Actual IRR (%)")


class PreloadedDeal(BaseModel):
    name: str
    description: Optional[str] = None
    sector: Optional[str] = None
    geography: Optional[str] = None
    outcome: Optional[str] = None
    entry: Dict[str, float]
    actual_years: List[int] = []
    actual: Dict[str, List[float]] = {}
    actual_exit: Dict[str, float] = {}
    money: Money


class BacktestRequest(Strict):
    entry: BacktestEntry
    actual: BacktestActuals
    actual_exit: BacktestActualExit
    settings: Dict[str, SettingValue] = {}
    n: SimulationPaths = Field(30000, ge=1000)
    histogram_bins: int = Field(80, ge=10, le=400)
    money: Money = Money()


class BacktestYear(BaseModel):
    year_index: int
    predicted_ebitda: float
    actual_ebitda: float
    ebitda_variance: float
    actual_revenue: float
    actual_fcf: float
    actual_total_debt: float


class BacktestResponse(BaseModel):
    predicted_irr_mean: float
    predicted_irr_p5: float
    predicted_irr_p95: float
    predicted_moic: float
    predicted_equity_entry: float
    predicted_exit_equity: float
    predicted_ebitda: List[float]
    actual_irr: float
    actual_moic: float
    actual_equity_entry: float
    actual_exit_equity: float
    actual_percentile: float
    actual_ebitda_margin: List[float]
    predicted_ebitda_margin: float
    predicted_net_debt_at_exit: float
    actual_exit_multiple: float
    attribution: Dict[str, float] = Field(
        description="Exact split of actual minus predicted exit equity: exit_ebitda, exit_multiple, net_debt")
    irr_histogram: Histogram
    years: List[BacktestYear]
    money: Money
    model: ModelStamp


# ---------------------------------------------------------------------------
# Plan vs actual (PLAN.md 2.7): any deal against what happened
# ---------------------------------------------------------------------------
ActualFigure = Optional[Annotated[float, Field(ge=-MAX_MONEY, le=MAX_MONEY)]]


class ActualYearIn(Strict):
    """One year's reported results, on the same basis as the deal's EBITDA. A
    figure left out (null) is not known; its variance is then left out too."""
    revenue: ActualFigure = None
    ebitda: ActualFigure = None
    net_income: ActualFigure = None
    fcf: ActualFigure = Field(None, description="Free cash flow after interest and tax, before debt repayment")
    total_debt: ActualFigure = Field(None, description="Debt at the year's end")


class ActualExitIn(Strict):
    exit_ev: float = Field(ge=0, le=MAX_MONEY, description="Enterprise value at exit")
    net_debt_at_exit: float = Field(ge=-MAX_MONEY, le=MAX_MONEY,
                                    description="Net debt at exit, counted as the deal counts it (negative: net cash)")
    sponsor_equity_entry: float = Field(gt=0, le=MAX_MONEY, description="The sponsor's equity cheque at entry")
    moic: Optional[float] = Field(None, ge=0, le=1000,
                                  description="Actual MOIC; left out, exit equity over the equity cheque")
    irr: Optional[float] = Field(None, ge=-100, le=10000,
                                 description="Actual IRR (%); left out, computed from the MOIC over the years held")


class DealActuals(Strict):
    """What happened to a deal: results for its first years (as many as are
    known, up to its hold) and, once it has been sold, the exit."""
    currency: CurrencyCode = Field(description="The currency the figures are in; must be the deal's")
    unit: MoneyUnit = Field(description="thousands, millions or billions")
    years: List[ActualYearIn] = Field(min_length=1, max_length=15, description="From the plan's year 1")
    exit: Optional[ActualExitIn] = Field(None, description="Left out while the deal is still held")


class PlanActualRequest(Strict):
    plan: DealInputsIn = Field(description="The deal that is the plan: a saved deal's inputs")
    settings: Dict[str, SettingValue] = Field({}, description="The plan's Settings overrides")
    actuals: DealActuals
    n: SimulationPaths = Field(30000, ge=1000)
    histogram_bins: int = Field(60, ge=10, le=400)


class PlanActualYear(BaseModel):
    year_index: int
    plan_revenue: float
    plan_ebitda: float
    plan_net_income: float
    plan_fcf: float = Field(description="Before debt repayment, like a reported free cash flow")
    plan_total_debt: float
    actual_revenue: Optional[float]
    actual_ebitda: Optional[float]
    actual_net_income: Optional[float]
    actual_fcf: Optional[float]
    actual_total_debt: Optional[float]
    variance_revenue: Optional[float]
    variance_ebitda: Optional[float]
    variance_net_income: Optional[float]
    variance_fcf: Optional[float]
    variance_total_debt: Optional[float]


class PlanReturns(BaseModel):
    """The plan's returns for an exit in the actual exit year (the plan's own
    hold while the deal is held). IRRs are fractions."""
    irr: float
    moic: float
    entry_equity: float
    exit_ebitda: float = Field(description="The EBITDA the exit multiple is applied to (with any lease add-back)")
    exit_multiple: float
    exit_ev: float
    net_debt_at_exit: float
    exit_equity: float
    irr_mean: float = Field(description="Mean IRR of the plan's simulated paths")
    irr_p5: float
    irr_p95: float


class ActualReturns(BaseModel):
    irr: Optional[float]
    moic: Optional[float]
    irr_given: bool = Field(description="True when the IRR was entered, false when computed")
    moic_given: bool
    entry_equity: float
    exit_ebitda: float = Field(description="The exit year's EBITDA plus the plan's lease add-back")
    exit_multiple: Optional[float] = Field(description="None when the exit EBITDA is not positive")
    exit_ev: float
    net_debt_at_exit: float
    exit_equity: float
    percentile: Optional[float] = Field(description="Share of the plan's simulated paths below the actual IRR (%)")


class PlanActualAttribution(BaseModel):
    exit_ebitda: float
    exit_multiple: float
    net_debt: float


class PlanActualResponse(BaseModel):
    hold: int = Field(description="The plan's holding period")
    years_compared: int
    exit_year: Optional[int] = Field(description="The year the deal was sold; None while held")
    years: List[PlanActualYear]
    plan: PlanReturns
    actual: Optional[ActualReturns]
    attribution: Optional[PlanActualAttribution] = Field(
        description="Exact split of actual minus plan exit equity; None while the deal is held")
    lease_addback: float = Field(description="What leases add to the EBITDA a multiple is applied to")
    plan_ebitda_margin: List[Optional[float]]
    actual_ebitda_margin: List[Optional[float]]
    irr_histogram: Histogram
    money: Money
    model: ModelStamp


class ExampleDeal(BaseModel):
    name: str
    description: str
    sector: str
    geography: str
    outcome: str
    plan: DealInputsIn
    actuals: DealActuals


class ExampleLibrary(BaseModel):
    enabled: bool = Field(description="False when the example library is switched off")
    examples: List[ExampleDeal]


class StoredActuals(BaseModel):
    actuals: Optional[DealActuals] = Field(description="None until actuals are saved for the deal")
    updated_at: Optional[str] = Field(None, description="UTC, ISO 8601")


class ValidationConsent(Strict):
    """Whether the deal's plan-vs-actual result counts, anonymised, in the
    model validation report (PLAN.md 4.6)."""
    opt_in: bool = Field(description="True to count it; false (the default) keeps it out")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
Cell = Union[float, int, str, bool, None]


# How a cell's number shows in Excel (PLAN.md 2.3a): api/routers/export.py NUMBER_FORMATS
CellFormat = Literal["money", "percent", "multiple", "integer", "number", "text"]


class WorkbookSheet(Strict):
    name: str = Field(min_length=1, max_length=100)
    columns: List[str] = Field(min_length=1, max_length=200)
    rows: List[List[Cell]] = Field(max_length=50_000)
    column_formats: Optional[List[Optional[CellFormat]]] = Field(
        None, max_length=200, description="Number format per column; percent cells hold fractions")
    row_formats: Optional[List[Optional[CellFormat]]] = Field(
        None, max_length=50_000, description="Number format per row; wins over the column's")


class WorkbookRequest(Strict):
    filename: str = Field("export.xlsx", max_length=120)
    sheets: List[WorkbookSheet] = Field(min_length=1, max_length=30)
    grouping: DigitGrouping = Field("locale", description="Lakh and crore patterns when 'lakh'")
    money: Optional[Money] = Field(
        None, description="What the money columns are counted in; written on an About sheet")
    model: Optional[ModelStamp] = Field(
        None, description="The stamp of the result the sheets hold, written on the About sheet; "
                          "without one the About sheet shows this API's own")


# ---------------------------------------------------------------------------
# ML and EDGAR
# ---------------------------------------------------------------------------
class Capabilities(BaseModel):
    anomaly_detector: bool = Field(description="The deal risk score (ml/anomaly_detector.py): always true "
                                               "since PLAN.md 5.2, which needs no ML packages")
    surrogate: bool
    macro_regime_installed: bool
    macro_regime_trained: bool
    edgar: bool


class DealRiskRequest(Strict):
    inputs: DealInputsIn = DealInputsIn()
    senior_x: float = Field(3.4, ge=0)
    mezz_x: float = Field(0.8, ge=0)


class SurrogateSliders(Strict):
    growth_mean: float = Field(5.0, ge=-5, le=20, description="Revenue growth mean (%)")
    exit_mean: float = Field(10.0, ge=4, le=20)
    interest_mean: float = Field(6.5, ge=1, le=15, description="(%)")
    gross_margin_mean: float = Field(40.0, ge=10, le=80, description="(%)")
    debt_pct: float = Field(60.0, ge=20, le=90, description="Debt / EV (%)")
    exit_std: float = Field(1.5, ge=0.3, le=5)


class SurrogateRequest(Strict):
    mc: MCInputsIn = MCInputsIn()
    deal: DealInputsIn = DealInputsIn()
    settings: Dict[str, SettingValue] = {}
    sliders: SurrogateSliders = SurrogateSliders()


class TermDifference(BaseModel):
    term: str
    value: float
    model_value: float
    unit: str
    decimals: int


class SurrogateResponse(BaseModel):
    prediction: Dict[str, Optional[float]]
    tail_unreliable: bool
    term_differences: List[TermDifference]
    training_deal: List[TermDifference] = Field(
        description="Every term held fixed in training (model_value), beside this deal's value")
    model: ModelStamp


class EdgarDealInputs(BaseModel):
    """The latest year of a filing as deal inputs: EBITDA as the standard
    reports it (operating profit plus D&A) and the leases, in the filing's
    currency and in millions."""
    ebitda: float
    currency: str
    unit: MoneyUnit
    accounting_standard: AccountingStandard
    lease_cost: float
    lease_liability: float


class EdgarResponse(BaseModel):
    ticker: str
    company_name: str
    years: List[int]
    history: Dict[str, List[float]] = Field(description="Forecasting history keys, oldest year first")
    warnings: List[str]
    fiscal_year_end_month: Optional[int] = Field(
        None, ge=1, le=12, description="Month the filer's fiscal year ends; years are named by the year they end in")
    money: Money = Field(description="The filing's own currency (US dollars for a 10-K), in millions")
    accounting_standard: AccountingStandard = Field(
        "us_gaap", description="The standard the statements follow: us_gaap (10-K) or ifrs (20-F, 40-F)")
    leases: Dict[str, List[float]] = Field(
        default_factory=dict, description="lease_cost and lease_liability, one value per year, oldest first")
    deal_inputs: EdgarDealInputs = Field(description="The latest year as deal inputs (PLAN.md 2.6)")


# ---------------------------------------------------------------------------
# Account (PLAN.md 1.4)
# ---------------------------------------------------------------------------
class AccountProfile(Strict):
    """How one person wants money, dates and numbers shown.

    Any country and any currency: the codes are checked for shape and the time
    zone against the IANA database (db/users.py), never against a list of
    "supported" places.
    """
    country: str = Field(description="ISO 3166-1 alpha-2, e.g. GB", min_length=2, max_length=2)
    preferred_currency: str = Field(description="ISO 4217, e.g. EUR", min_length=3, max_length=3)
    locale: str = Field(description="BCP 47 language tag, e.g. en-GB", min_length=2, max_length=35)
    time_zone: str = Field(description="IANA time zone, e.g. Europe/London", min_length=1, max_length=64)
    digit_grouping: DigitGrouping = Field(
        "locale", description="Group long numbers as the locale does, in thousands, or in lakh and crore")


class AccountResponse(BaseModel):
    """The signed-in account. ``profile`` is null until sign-up finishes it."""
    subject: str = Field(description="The identity provider's user id")
    profile: Optional[AccountProfile] = None


class AccountSettings(Strict):
    """The account's Settings overrides: only keys that differ from the defaults."""
    settings: Dict[str, SettingValue] = {}


# ---------------------------------------------------------------------------
# Saved deals (PLAN.md 1.5)
# ---------------------------------------------------------------------------
class DealContent(Strict):
    """Everything that decides a deal's numbers: its inputs and the Settings
    overrides in effect."""
    inputs: DealInputsIn
    settings: Dict[str, SettingValue] = {}


class DealCreate(DealContent):
    name: str = Field(min_length=1, max_length=120, description="Shown in the deal list")


class DealPatch(Strict):
    """Rename, archive or unarchive; fields left out stay as they are."""
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    archived: Optional[bool] = None


class DealDuplicate(Strict):
    name: Optional[str] = Field(None, min_length=1, max_length=120,
                                description="Defaults to the original's name with (copy)")


class DealSummary(BaseModel):
    id: str
    name: str
    archived: bool
    latest_version: int = Field(description="Number of the newest version")
    created_at: str = Field(description="UTC, ISO 8601")
    updated_at: str = Field(description="UTC, ISO 8601")


class DealDetail(DealSummary):
    inputs: DealInputsIn
    settings: Dict[str, SettingValue]
    model: Optional[SavedModel] = Field(
        None, description="The model stamp and results the working copy was saved with; "
                          "null for a deal saved before they were kept")
    model_check: Optional[ModelCheck] = Field(
        None, description="On opening or restoring: whether results changed since saved")


class DealList(BaseModel):
    deals: List[DealSummary]


class VersionSave(Strict):
    label: Optional[str] = Field(None, min_length=1, max_length=120)


class VersionSummary(BaseModel):
    number: int
    kind: Literal["created", "saved", "auto", "restored"]
    label: Optional[str]
    created_at: str = Field(description="UTC, ISO 8601")


class VersionDetail(VersionSummary):
    inputs: DealInputsIn
    settings: Dict[str, SettingValue]
    model: Optional[SavedModel] = None


class VersionList(BaseModel):
    versions: List[VersionSummary]


AuditAction = Literal["created", "edited", "renamed", "archived", "unarchived", "versioned", "restored",
                      "actuals_saved", "actuals_cleared", "exported", "deleted", "settings_changed",
                      "shared", "library_switched", "reference_proposed", "reference_reviewed",
                      "validation_opted_in", "validation_opted_out"]


class AuditEntry(BaseModel):
    """One thing done to a deal or the account's settings (PLAN.md 3.3). Holds
    what was touched, never a figure, a name or a label."""
    id: int
    action: AuditAction
    at: str = Field(description="UTC, ISO 8601; the first of the merged actions when count > 1")
    until: Optional[str] = Field(None, description="The last of the merged actions (UTC), when count > 1")
    count: int = Field(description="How many actions this entry stands for: old edits are merged one per day")
    deal_id: Optional[str] = Field(None, description="Null for a settings change")
    deal_name: Optional[str] = Field(
        None, description="The deal's name now (account history only); null once it is deleted")
    fields: Optional[List[str]] = Field(
        None, description="What an edit or settings change touched: input field names, settings.<key> for "
                          "a deal's settings, setting keys for the account's")
    version: Optional[int] = Field(None, description="The version kept or restored")
    source_deal: Optional[str] = Field(None, description="The deal a duplicate was made from")
    export: Optional[Literal["workbook", "simulation_sample"]] = None
    enabled: Optional[bool] = Field(None, description="Whether the reference library was shown or hidden")
    reference: Optional[str] = Field(None, description="The reference transaction proposed or reviewed")
    verdict: Optional[Literal["approve", "reject"]] = Field(None, description="The review given")


class AuditHistory(BaseModel):
    entries: List[AuditEntry] = Field(description="Newest first")


# ---------------------------------------------------------------------------
# Background jobs (PLAN.md 1.9)
# ---------------------------------------------------------------------------
class MonteCarloJob(Strict):
    kind: Literal["montecarlo.run"]
    input: MonteCarloRequest


class ScenariosJob(Strict):
    kind: Literal["montecarlo.scenarios"]
    input: ScenariosRequest


class BacktestJob(Strict):
    kind: Literal["backtesting.run"]
    input: BacktestRequest


class ForecastJob(Strict):
    kind: Literal["forecasting.run"]
    input: ForecastRunRequest


# What to run: the same request the matching endpoint takes, e.g. kind
# "montecarlo.run" with the body of POST /api/montecarlo/run
JobSubmit = Annotated[Union[MonteCarloJob, ScenariosJob, BacktestJob, ForecastJob],
                      Field(discriminator="kind")]

JobKindName = Literal["montecarlo.run", "montecarlo.scenarios", "backtesting.run", "forecasting.run"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]


class JobOut(BaseModel):
    id: str
    kind: JobKindName
    status: JobStatus
    progress: float = Field(description="0 to 1; moves as the run passes its stages")
    stage: Optional[str] = Field(None, description="What the run is doing now, for people")
    ahead: Optional[int] = Field(None, description="Queued jobs ahead of this one (queued only)")
    attempts: int = Field(description="Runs so far; above 1 means it was resumed after a restart")
    cancel_requested: bool
    created_at: str = Field(description="UTC, ISO 8601")
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = Field(None, description="Why it failed, for people")
    error_status: Optional[int] = Field(
        None, description="The HTTP status the same request would have got (422: bad input)")
    result_expired: bool = Field(
        False, description="It succeeded, but the result has been deleted; run it again")
    result: Optional[dict] = Field(
        None, description="Once succeeded: exactly what the matching endpoint answers "
                          "(MonteCarloResponse for montecarlo.run, and so on)")


class JobList(BaseModel):
    jobs: List[JobOut]


# ---------------------------------------------------------------------------
# Company filings (PLAN.md 4.1, companies/)
# ---------------------------------------------------------------------------
CompanySource = Literal["sec", "esef", "companies_house", "edinet"]
CompanyStandard = Literal["ifrs", "us_gaap", "uk_gaap", "jgaap"]


class CompanySourceOut(BaseModel):
    id: CompanySource
    coverage: List[str] = Field(description="ISO country codes, \"EU\" for every member state, \"*\" anywhere")
    searchable_by: List[str] = Field(description="name, ticker, lei, isin, company_number, cik, edinet_code, jcn")
    needs_key: bool
    configured: bool = Field(description="Whether this server can load companies from it")
    licence: str


class CompanySourcesResponse(BaseModel):
    sources: List[CompanySourceOut]
    fallback: Literal["document_upload"] = Field(
        description="What to offer for a company no source covers (PLAN.md 6.2)")


class CompanyRefOut(BaseModel):
    source: CompanySource
    id: str = Field(description="The company's id at its source: CIK, LEI, company number or EDINET code")
    name: str
    local_name: Optional[str] = Field(None, description="The name in the register's own language")
    country: Optional[str] = Field(None, description="ISO 3166-1 alpha-2, when the source says")
    identifiers: Dict[str, str] = Field(description="lei, isin, ticker, cik, company_number, edinet_code, "
                                                    "sec_code, jcn: whichever are known")
    loadable: bool = Field(description="This server has the source's key")
    stored: bool = Field(description="Its figures are already stored")


class CompanyUnavailable(BaseModel):
    source: str
    reason: str = Field(description="not_configured, unreachable, refused, rate_limited, failed ...")


class CompanySearchResponse(BaseModel):
    matched: Optional[Literal["lei", "isin"]] = Field(
        None, description="The query was read as this identifier (its check digits passed)")
    results: List[CompanyRefOut]
    unavailable: List[CompanyUnavailable]
    fallback: Literal["document_upload"]


class CompanyFigures(BaseModel):
    """Summary figures for one fiscal year, millions of the company's currency;
    null where the filing reports none."""
    revenue: Optional[float] = None
    operating_income: Optional[float] = None
    depreciation_amortization: Optional[float] = None
    ebitda: Optional[float] = Field(None, description="Operating income plus D&A, as the standard reports them")
    net_income: Optional[float] = None
    income_tax_expense: Optional[float] = None
    interest_expense: Optional[float] = None
    capital_expenditures: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    accounts_receivable: Optional[float] = None
    inventories: Optional[float] = None
    accounts_payable: Optional[float] = None
    total_debt: Optional[float] = None
    total_assets: Optional[float] = None
    total_equity: Optional[float] = None
    lease_cost: Optional[float] = None
    lease_liability: Optional[float] = None


class CompanyFilingOut(BaseModel):
    url: str = Field(description="The filing itself, at its source")
    filed_on: Optional[date] = Field(None, description="Filed (ESEF: added to filings.xbrl.org)")
    form: str
    id: str


class CompanyYearOut(BaseModel):
    fiscal_year: int = Field(description="Named by the calendar year it ends in")
    period_end: date
    figures: CompanyFigures
    filing: CompanyFilingOut


class CompanyWarningOut(BaseModel):
    code: str = Field(description="missing_figure, scanned_accounts, not_indexed_yet, summary_only, "
                                  "no_annual_figures")
    field: Optional[str] = None


class CompanyDealInputs(BaseModel):
    """The latest year as a deal's inputs (companies/use.py), millions of the
    company's currency."""
    ebitda: float = Field(description="Operating income plus D&A, as the standard reports them")
    currency: str
    unit: MoneyUnit
    accounting_standard: AccountingStandard = Field(
        description="The company's standard when a deal knows it (IFRS, US GAAP), else \"\"")
    lease_cost: float = Field(description="0 when the filing reports none or the deal doesn't know the standard")
    lease_liability: float
    fiscal_year: int = Field(description="The year these come from")
    notes: List[Literal["standard_not_in_deal", "lease_cost_missing"]]


class CompanyResponse(BaseModel):
    company: CompanyRefOut
    money: Money = Field(description="The filing's own currency, in millions")
    accounting_standard: CompanyStandard
    fiscal_year_end_month: Optional[int] = Field(None, ge=1, le=12)
    years: List[CompanyYearOut] = Field(description="Oldest first")
    warnings: List[CompanyWarningOut]
    refreshed_at: Optional[datetime] = Field(None, description="When the stored figures were read (UTC)")
    licence: str
    deal_inputs: Optional[CompanyDealInputs] = Field(
        None, description="The latest year as deal inputs; null without an EBITDA")
    forecast_history: Optional[Dict[str, List[float]]] = Field(
        None, description="Forecasting history rows from the summary (companies/use.py), oldest year first; "
                          "null when a year has no revenue")


# ---------------------------------------------------------------------------
# Economic data by country, reference rates, exchange rates (PLAN.md 4.2)
# ---------------------------------------------------------------------------
EconomicSource = Literal["imf", "worldbank", "oecd", "bis", "ecb", "fred"]
EconomicBasis = Literal["projection", "actual", "observed", "euro_area", "computed", "benchmark",
                        "policy_rate", "interbank_3m"]


class EconomicFigure(BaseModel):
    value: float = Field(description="Per cent (3.75 = 3.75%), or percentage points for a spread")
    period: str = Field(description="YYYY, YYYY-MM or YYYY-MM-DD: the period the value is for")
    source: EconomicSource
    source_series: str = Field(description="The series' id at its source")
    url: str = Field(description="Where a person can check it")
    basis: EconomicBasis = Field(description="projection (IMF WEO, this year), actual, observed, euro_area "
                                             "(a euro member's policy rate is the ECB's), computed (a "
                                             "spread), benchmark (the reference rate itself), or what "
                                             "stands in for a benchmark with no free source: policy_rate, "
                                             "interbank_3m")
    current: bool = Field(description="Recent enough to count as today's (economy/views.py)")


class EconomyArea(BaseModel):
    area: str = Field(description="ISO 3166-1 alpha-2, or XM for the euro area")
    currency: str
    current: bool = Field(description="Growth, inflation and the policy rate are all current")
    figures: Dict[str, EconomicFigure] = Field(
        description="gdp_growth, gdp_growth_actual, inflation, inflation_actual, policy_rate, "
                    "bond_yield_10y, short_rate_3m, sovereign_spread: whichever are known")


class EconomySourceOut(BaseModel):
    id: EconomicSource
    name: str
    licence: str
    needs_key: bool
    configured: bool


class EconomyResponse(BaseModel):
    as_of: date = Field(description="The day 'current' was judged on (UTC)")
    refreshed_at: Optional[datetime] = Field(None, description="The latest refresh (UTC); null before the first")
    areas: List[EconomyArea]
    sources: List[EconomySourceOut]


class ReferenceRatesResponse(BaseModel):
    as_of: date
    refreshed_at: Optional[datetime] = None
    rates: Dict[str, EconomicFigure] = Field(
        description="Each benchmark's current level, by code (core/debt.py REFERENCE_RATES); a benchmark "
                    "with no current figure is left out")
    currency_benchmarks: Dict[str, ReferenceRate] = Field(
        description="The benchmark a new floating facility starts on, by the deal's currency")


class ExchangeRatesResponse(BaseModel):
    base: str
    published_on: date = Field(description="The ECB publication day the rates are from")
    rates: Dict[str, float] = Field(description="Units of each currency for one unit of base")
    source: Literal["ecb"] = "ecb"
    url: str


BenchmarkLevel = Literal["country", "region", "global"]
BenchmarkArea = Literal["us", "japan", "china", "india", "europe", "aus_nz_canada", "emerging", "global"]
StartingField = Literal["growth", "tax", "gross_margin", "opex", "da", "entry_mult", "exit_mult", "capex",
                        "ar_days", "inv_days", "ap_days", "nwc", "debt_pct", "senior_pct", "base_rate"]


class BenchmarkSkipped(BaseModel):
    area: str = Field(description="A group (benchmarks/catalogue.py REGIONS) or a country or currency code")
    reason: Literal["thin", "unusable", "missing"] = Field(
        description="thin: fewer companies than min_firms; unusable: a figure the model can't start "
                    "from; missing: the group has no figure")
    sample: Optional[int] = Field(None, description="How many companies the group has, when known")


class StartingFigure(BaseModel):
    field: StartingField = Field(description="The deal input this figure fills")
    value: float = Field(description="As the deal input reads it: per cent, a multiple or days")
    source: Literal["damodaran", "imf", "tax_foundation", "benchmark", "choice"]
    dataset: str = Field(description="The table or series read")
    area: str = Field(description="The group or country the figure is for")
    level: BenchmarkLevel
    sample: Optional[int] = Field(None, description="Companies, economies or countries behind the figure")
    sample_kind: Optional[Literal["companies", "economies", "countries"]] = None
    as_of: Optional[str] = Field(None, description="The publisher's date, or the period the figure is for")
    url: Optional[str] = None
    skipped: List[BenchmarkSkipped] = Field(description="Closer groups passed over, and why")
    detail: Dict[str, Any] = Field(description="The published figures it was worked out from")


class StartingMissing(BaseModel):
    field: StartingField
    skipped: List[BenchmarkSkipped]


class BenchmarkSourceOut(BaseModel):
    id: Literal["damodaran"]
    publisher: str
    title: str
    url: str
    basis: str
    usage: str


class StartingAssumptionsResponse(BaseModel):
    country: str
    industry: str
    industry_name: Optional[str] = Field(None, description="The industry as the source names it; null before "
                                                           "the first refresh")
    currency: str
    region: BenchmarkArea = Field(description="The country's region, where its figures are pooled when it has none")
    chain: List[BenchmarkArea] = Field(description="The groups looked in, closest first")
    min_firms: int = Field(description="A group with fewer companies hands over to the next")
    inputs: Dict[StartingField, float] = Field(description="The starting value of each deal input found")
    figures: List[StartingFigure]
    missing: List[StartingMissing] = Field(description="Inputs with no sourced figure: the deal keeps its own")
    notes: List[Literal["size_not_split", "exit_equals_entry", "listed_company_leverage"]]
    source: BenchmarkSourceOut
    refreshed_at: Optional[datetime] = Field(None, description="When the stored averages were last read (UTC)")


# PLAN.md 4.4: the Monte Carlo Settings worked out from published history (benchmarks/risk.py SETTINGS)
RiskSetting = Literal[
    "mc_growth_mean", "mc_exit_mean", "mc_rate_mean", "mc_gm_mean",
    "mc_growth_std", "mc_exit_std", "mc_rate_std", "mc_gm_std",
    "corr_g_em", "corr_g_ir", "corr_g_gm", "corr_g_sh", "corr_em_ir", "corr_em_gm", "corr_em_sh",
    "corr_ir_gm", "corr_ir_sh", "corr_gm_sh",
    "bull_growth_mult", "bull_exit_mult", "bull_rate_mult", "bull_margin_mult",
    "rec_growth_adj", "rec_growth_floor", "rec_exit_mult", "rec_rate_mult", "rec_margin_mult",
    "stag_growth_adj", "stag_growth_floor", "stag_exit_mult", "stag_rate_mult", "stag_margin_mult",
]
RiskScenarioId = Literal["recession", "stagflation", "bull"]


class RiskSkipped(BaseModel):
    area: str = Field(description="A group (benchmarks/catalogue.py REGIONS), a country or a currency code")
    reason: Literal["thin", "unusable", "missing", "short", "base_not_positive", "no_history_in_years"] = Field(
        description="short: fewer usable years than min_years; base_not_positive: a multiplier needs a "
                    "positive mean to scale; no_history_in_years: the industry's history has none of the "
                    "scenario's years; the rest as for the starting figures")
    sample: Optional[int] = Field(None, description="Companies (or usable years, when short), when known")


class RiskFigure(BaseModel):
    field: RiskSetting = Field(description="The Setting this figure fills")
    value: float = Field(description="As the Setting reads it: per cent, a multiple, a multiplier or a correlation")
    source: Literal["damodaran", "imf", "bis", "benchmark", "tax_foundation", "choice", "damodaran+imf+bis"]
    dataset: str = Field(description="The table or series read")
    area: str = Field(description="The group or country the figure is for")
    level: BenchmarkLevel
    sample: Optional[int] = Field(None, description="Companies, economies, years or industry-years behind it")
    sample_kind: Optional[Literal["companies", "economies", "countries", "years", "industry_years"]] = None
    as_of: Optional[str] = Field(None, description="The publisher's date, when there is one")
    url: Optional[str] = None
    skipped: List[RiskSkipped] = Field(description="Closer groups passed over, and why")
    detail: Dict[str, Any] = Field(description="The years and published figures it was worked out from")


class RiskMissing(BaseModel):
    field: RiskSetting
    skipped: List[RiskSkipped]


class RiskPeriod(BaseModel):
    start: int = Field(description="The period's first year")
    end: int = Field(description="Its last year (the same for a single year)")


class RiskScenario(BaseModel):
    id: RiskScenarioId
    rule: Literal["weakest_growth", "highest_inflation", "strongest_growth"] = Field(
        description="The fifth of the economy's years the preset is built from: lowest real GDP growth, "
                    "highest inflation, highest real GDP growth")
    area: str = Field(description="The country, or the group whose economies' median stands in for it")
    level: BenchmarkLevel
    economies: int
    years: List[int]
    periods: List[RiskPeriod] = Field(description="The years, consecutive ones joined")
    of_years: int = Field(description="How many years they were chosen from")
    window: Optional[str] = Field(None, description="First and last year looked at")


class RiskCorrelation(BaseModel):
    group: BenchmarkArea
    level: BenchmarkLevel
    labels: List[str]
    matrix: List[List[float]] = Field(description="The valid matrix the Settings take")
    observations: List[List[int]] = Field(description="What each correlation rests on: industry-years, or "
                                                      "years for growth and the rate")
    rows: int = Field(description="Industry-years in the panel")
    economies: int
    period: Optional[str] = None
    shrink: float = Field(description="How far the measured matrix was shrunk toward no correlation to be "
                                      "valid (0 = not at all)")


class RiskAssumptionsResponse(BaseModel):
    country: str
    industry: str
    industry_name: Optional[str] = Field(None, description="The industry as the source names it; null before "
                                                           "the first refresh")
    currency: str
    region: BenchmarkArea
    settings: Dict[RiskSetting, float] = Field(description="The value of each Setting found")
    figures: List[RiskFigure]
    missing: List[RiskMissing] = Field(description="Settings with no sourced figure: they keep their value")
    scenarios: Dict[RiskScenarioId, RiskScenario]
    correlation: RiskCorrelation
    window_start: int = Field(description="The first year of economic history read")
    min_years: int = Field(description="A spread needs at least this many usable years")
    notes: List[Literal["size_not_split", "growth_economy_wide", "rate_policy_only", "industry_aggregates"]]
    source: BenchmarkSourceOut
    refreshed_at: Optional[datetime] = Field(None, description="When the stored averages were last read (UTC)")


class BenchmarkIndustry(BaseModel):
    id: str = Field(description="all, or the industry's name as an id (machinery, oil_gas_integrated ...)")
    name: str = Field(description="The industry as the source names it")
    firms: int = Field(description="Listed companies in it worldwide")


class BenchmarkIndustriesResponse(BaseModel):
    industries: List[BenchmarkIndustry] = Field(description="Empty until the first refresh")
    published: Optional[date] = Field(None, description="The averages' date, as their publisher gives it")
    refreshed_at: Optional[datetime] = None
    source: BenchmarkSourceOut


class CompanyLoadRequest(Strict):
    source: CompanySource
    id: str = Field(min_length=1, max_length=20)
    years: int = Field(3, ge=2, le=5, strict=True, description="Fiscal years to read, newest last")


# ---------------------------------------------------------------------------
# Reference library (PLAN.md 4.5)
# ---------------------------------------------------------------------------
class LibraryState(BaseModel):
    enabled: bool = Field(description="Whether the library is shown to everyone")
    locked_off: bool = Field(description="The server's configuration forces it off (FSE_EXAMPLE_LIBRARY=0)")
    can_switch: bool = Field(description="The caller is an administrator and may switch it")
    updated_at: Optional[datetime] = Field(None, description="When an administrator last switched it (UTC)")


class LibrarySwitch(Strict):
    enabled: bool = Field(strict=True)


class BaseRateSample(BaseModel):
    count: int
    what: str
    first_year: int
    last_year: int


class BaseRateSource(BaseModel):
    id: Literal["sp_default_study_2024", "gcd_lgd_2020"]
    kind: Literal["default", "recovery"]
    publisher: str
    title: str
    published: str = Field(description="ISO date, or year and month")
    url: str
    sample: BaseRateSample
    note: str
    checked_on: str = Field(description="When this edition was last confirmed as the newest free to read")
    recheck_due: str = Field(description="A year after checked_on")
    stale: bool = Field(description="True once recheck_due has passed: a newer edition may exist")


BaseRateTableId = Literal["annual_by_rating", "annual_by_grade", "speculative_by_region", "cumulative_global",
                          "cumulative_by_region", "lgd_by_seniority", "lgd_by_year", "lgd_by_region"]
SpRegion = Literal["us", "europe", "emerging", "other_developed"]
Rating = Literal["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C"]


class BaseRateTable(BaseModel):
    id: BaseRateTableId
    source: str
    table: str = Field(description="The table's number in its source")
    title: str = Field(description="The table's title as its source prints it")


class AnnualByRating(BaseModel):
    years: List[int]
    rates: Dict[Rating, List[float]] = Field(description="Per cent of issuers rated at the start of the year")


class AnnualByGrade(BaseModel):
    years: List[int]
    defaults: List[int] = Field(description="Defaults that year, including issuers no longer rated")
    rates: Dict[Literal["all_rated", "investment_grade", "speculative_grade"], List[float]]


class SpeculativeByRegion(BaseModel):
    years: List[int]
    regions: List[SpRegion]
    rates: Dict[SpRegion, List[Optional[float]]] = Field(description="Null where the source has no figure")


class CumulativeRates(BaseModel):
    region: Literal["global", "us", "europe", "emerging"]
    table: BaseRateTableId
    horizons: int = Field(description="Years after the rating: rates[k][i] is after i + 1 years")
    rates: Dict[str, List[float]] = Field(description="By rating, and investment_grade, speculative_grade, all_rated")


class DefaultBaseRates(BaseModel):
    ratings: List[Rating]
    annual_by_rating: AnnualByRating
    annual_by_grade: AnnualByGrade
    speculative_by_region: SpeculativeByRegion
    cumulative: List[CumulativeRates]


class LgdRow(BaseModel):
    key: str
    defaults: int = Field(description="Defaulted borrowers")
    lgd_pct: float = Field(description="Loss given default, per cent of exposure; recovery is 100 less this")


class LgdYear(BaseModel):
    year: int
    defaults: int
    lgd_pct: float


class RecoveryBaseRates(BaseModel):
    by_seniority: List[LgdRow]
    by_year: List[LgdYear]
    by_region: List[LgdRow]


class BaseRatesResponse(BaseModel):
    enabled: bool = Field(description="False when the library is switched off: everything else is empty")
    sources: List[BaseRateSource]
    tables: List[BaseRateTable]
    country: Optional[str] = Field(None, description="The country asked about")
    sp_region: Optional[SpRegion] = Field(None, description="S&P's region for that country")
    default: Optional[DefaultBaseRates] = None
    recovery: Optional[RecoveryBaseRates] = None


class CoverageBucket(BaseModel):
    bucket: str
    count: int


class CoverageCollection(BaseModel):
    id: Literal["reference_deals", "examples"]
    count: int
    awaiting_review: Optional[int] = Field(None, description="Reference transactions proposed, not yet decided")
    sourced: bool = Field(description="Whether every figure in its deals carries a source")
    dimensions: Dict[Literal["region", "size", "sector", "era", "outcome"], List[CoverageBucket]]


class BaseRateCoverage(BaseModel):
    table: BaseRateTableId
    source: str
    regions: int
    bands: int = Field(description="Rating bands, grades or seniority classes")
    first_year: int
    last_year: int
    observations: Optional[int] = Field(None, description="Issuers or borrowers behind the table")


class CoverageResponse(BaseModel):
    enabled: bool
    collections: List[CoverageCollection]
    base_rates: List[BaseRateCoverage]


# ---------------------------------------------------------------------------
# Reference transactions and their review (PLAN.md 4.5b, library/references.py)
# ---------------------------------------------------------------------------
ReferenceSector = Literal["communication_services", "consumer_discretionary", "consumer_staples", "energy",
                          "financials", "health_care", "industrials", "information_technology", "materials",
                          "real_estate", "utilities"]
SourceId = Annotated[str, Field(pattern=r"^[a-z0-9_]{1,60}$")]
Words = Annotated[str, Field(min_length=1, max_length=300)]


class ReferenceSource(Strict):
    filer: Annotated[str, Field(min_length=1, max_length=200)]
    form: Annotated[str, Field(min_length=1, max_length=20, description="The filing's form, e.g. 10-K, 424B4")]
    filed: date
    url: Annotated[str, Field(pattern=r"^https://", max_length=500,
                              description="The filing on its regulator's or exchange's own site")]


class ReferencePart(Strict):
    label: Words
    value: float
    source: SourceId
    where: Words
    quote: Optional[Words] = None


class _Cited(Strict):
    source: Optional[SourceId] = None
    where: Optional[Words] = None
    quote: Optional[Words] = None
    note: Optional[Annotated[str, Field(max_length=400)]] = None


class ReferenceFigure(_Cited):
    """A figure read from one place in a filing, or the sum of ``parts``
    read from several (each with its own source)."""
    value: float
    parts: Optional[Annotated[List[ReferencePart], Field(min_length=1, max_length=12)]] = None

    @model_validator(mode="after")
    def _cited(self):
        if not self.parts and not (self.source and self.where):
            raise ValueError("a figure needs a source and where in it, or parts that each have one")
        return self


class ReferenceValue(ReferenceFigure):
    basis: Literal["stated", "equity_value", "uses_less_fees", "funds_needed"]


class ReferenceEbitda(ReferenceFigure):
    basis: Literal["reported", "adjusted", "stated", "projection"]
    year: Annotated[int, Field(ge=1950, le=2100, description="The fiscal year it covers")]


class ReferenceFigures(Strict):
    transaction_value: ReferenceValue
    transaction_value_usd: Optional[ReferenceFigure] = Field(
        None, description="Needed when the deal is not in US dollars, for its size")
    ebitda: ReferenceEbitda
    debt: ReferenceFigure
    equity: Optional[ReferenceFigure] = None
    transaction_fees: Optional[ReferenceFigure] = None
    financing_fees: Optional[ReferenceFigure] = None
    senior_amort_pct: Optional[ReferenceFigure] = Field(None, description="Per cent of the original principal a year")


class ReferenceClosed(Strict):
    date: date
    source: SourceId
    where: Words
    quote: Optional[Words] = None


class ReferenceOutcome(Strict):
    kind: Literal["success", "distress", "held"]
    event: Literal["ipo", "sale", "relisted", "missed_payment", "bankruptcy", "restructuring", "held"]
    year: Annotated[int, Field(ge=1950, le=2100)]
    source: SourceId
    where: Words
    quote: Optional[Words] = None
    note: Optional[Annotated[str, Field(max_length=400)]] = None


class ReferenceDealIn(Strict):
    """A reference transaction as proposed: money in millions of ``currency``."""
    key: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{2,79}$", description="e.g. hca-2006")]
    target: Annotated[str, Field(min_length=1, max_length=200)]
    country: Annotated[str, Field(pattern=r"^[A-Z]{2}$", description="Where the business is, ISO 3166-1")]
    sector: ReferenceSector
    kind: Literal["take_private", "carve_out", "recapitalization", "secondary"]
    sponsors: Annotated[List[Annotated[str, Field(min_length=1, max_length=100)]], Field(min_length=1, max_length=12)]
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")]
    closed: ReferenceClosed
    figures: ReferenceFigures
    outcome: ReferenceOutcome
    sources: Annotated[Dict[SourceId, ReferenceSource], Field(min_length=1, max_length=20)]


class ReferenceDerived(BaseModel):
    entry_multiple: Optional[float] = Field(None, description="Transaction value over EBITDA")
    leverage: Optional[float] = Field(None, description="Debt over EBITDA")
    debt_share_pct: Optional[float] = Field(None, description="Debt, per cent of the transaction value")
    tx_fee_pct: Optional[float] = Field(None, description="Transaction fees, per cent of the transaction value")
    fin_fee_pct: Optional[float] = Field(None, description="Financing fees, per cent of the debt")
    def_senior_amort: Optional[float] = Field(None, description="Senior amortisation, per cent a year")


class ReferenceTags(BaseModel):
    region: Optional[SpRegion] = None
    size: Optional[str] = None
    sector: str
    era: Optional[str] = None
    outcome: str


class ReferenceDealView(BaseModel):
    """A reference transaction with what its figures say together and its coverage buckets."""
    key: str
    target: str
    country: str
    sector: ReferenceSector
    kind: str
    sponsors: List[str]
    currency: str
    closed: ReferenceClosed
    figures: ReferenceFigures
    outcome: ReferenceOutcome
    sources: Dict[str, ReferenceSource]
    derived: ReferenceDerived
    tags: ReferenceTags
    approved_at: Optional[datetime] = None


class ReferenceDealsResponse(BaseModel):
    enabled: bool
    deals: List[ReferenceDealView] = Field(description="The approved reference transactions")
    awaiting_review: int = Field(description="Proposed and not yet decided")


RuleCode = Literal["no_sponsor", "missing_figure", "unknown_figure", "missing_source", "not_a_filing",
                   "unused_source", "parts_dont_add_up", "not_positive", "multiple_out_of_range",
                   "debt_exceeds_value", "fee_out_of_range", "amortisation_out_of_range", "closed_in_future",
                   "outcome_before_close", "event_doesnt_match_outcome"]
RejectReason = Literal["figure_wrong", "source_wrong", "not_a_buyout", "duplicate", "other"]
BalanceDimension = Literal["region", "size", "sector", "era", "outcome"]


class RuleProblem(BaseModel):
    code: RuleCode
    at: Optional[str] = Field(None, description="The figure, source or field it concerns")


class BalanceOver(BaseModel):
    dimension: BalanceDimension
    bucket: Optional[str] = None
    share_pct: float = Field(description="The bucket's share of the library with this deal added")


class BalanceFill(BaseModel):
    dimension: BalanceDimension
    bucket: Optional[str] = None


class Balance(BaseModel):
    over: List[BalanceOver] = Field(description="Buckets this deal would push past half the library")
    fills: List[BalanceFill] = Field(description="Empty buckets this deal would fill")


class ReviewProposal(BaseModel):
    id: str
    key: str
    origin: Literal["repository", "user"]
    status: Literal["proposed", "approved", "rejected", "superseded"]
    proposed_at: datetime
    decided_at: Optional[datetime] = None
    approvals: int
    approvals_needed: int
    rejections: int
    reasons: List[RejectReason]
    mine: bool = Field(description="The caller proposed it, so may not review it")
    my_verdict: Optional[Literal["approve", "reject"]] = None
    replaces_approved: bool = Field(description="An approved version of this transaction is in the library")
    deal: ReferenceDealView
    problems: List[RuleProblem] = Field(description="The inclusion rules' findings; any blocks approval")
    balance: Balance


class ReviewQueue(BaseModel):
    enabled: bool
    proposals: List[ReviewProposal] = Field(description="Awaiting review, oldest first")
    decided: List[ReviewProposal] = Field(description="The latest decided, newest first")
    library_size: int


class ReviewVerdict(Strict):
    verdict: Literal["approve", "reject"]
    reason: Optional[RejectReason] = None

    @model_validator(mode="after")
    def _reason(self):
        if self.verdict == "reject" and self.reason is None:
            raise ValueError("a rejection needs a reason")
        return self


FeeSetting = Literal["tx_fee_pct", "fin_fee_pct", "def_senior_amort"]


class SourcedFee(BaseModel):
    value: float = Field(description="The median across the deals, per cent")
    n: int
    low: float
    high: float
    first_year: int
    last_year: int
    deals: List[str] = Field(description="The reference transactions' keys")


class SourcedFeesResponse(BaseModel):
    enabled: bool
    min_deals: int = Field(description="Deals a figure needs before it is offered")
    library_size: int
    settings: Dict[FeeSetting, Optional[SourcedFee]]


# ---------------------------------------------------------------------------
# Model validation (PLAN.md 4.6, validation/)
# ---------------------------------------------------------------------------
class ProbabilityStats(BaseModel):
    """A predicted probability against what happened, for one group."""
    predicted_pct: Optional[float] = Field(description="Mean predicted probability, per cent")
    observed_pct: Optional[float] = Field(description="Share that happened, per cent")
    expected: Optional[float] = Field(description="Sum of the predicted probabilities")
    observed: int
    bias_pp: Optional[float] = Field(description="Observed minus predicted, percentage points")
    brier: Optional[float] = Field(description="Mean squared error of the probabilities (0 is perfect)")
    z: Optional[float] = Field(description="(observed - expected) / its standard deviation")
    consistent: Optional[bool] = Field(description="|z| at most 1.96")


class RangeLevel(BaseModel):
    claimed_pct: int = Field(description="The central share of simulated paths the range holds")
    inside_pct: Optional[float] = Field(description="How often the actual IRR fell inside it, per cent")
    interval_pct: List[Optional[float]] = Field(description="95% Wilson interval of inside_pct")
    consistent: bool


class RangeStats(BaseModel):
    """The plan's IRR ranges against actual IRRs, for one group."""
    levels: List[RangeLevel]
    bias_pp: Optional[float] = Field(description="Mean of actual minus planned IRR, percentage points")
    mean_percentile: Optional[float] = Field(description="Where actual IRRs fell among the paths (50 = unbiased)")
    consistent: bool


class ValidationCell(BaseModel):
    """One group. ``n`` and ``contributed_n`` are null when suppressed: too few
    users' deals to show without risking one being singled out."""
    n: Optional[int]
    library_n: int
    contributed_n: Optional[int]
    suppressed: bool
    enough: bool = Field(description="At least the report's min_cases, so statistics are shown")
    stats: Optional[Union[RangeStats, ProbabilityStats]]


class ValidationBucket(ValidationCell):
    bucket: str


class ValidationSample(BaseModel):
    overall: ValidationCell
    splits: Dict[str, List[ValidationBucket]] = Field(description="region, sector, size and era")


class ValidationSamples(BaseModel):
    out_of_time: ValidationSample = Field(description="Outcomes newer than the prediction's data: the headline")
    in_sample: ValidationSample


class ValidationCheck(BaseModel):
    id: Literal["default", "irr_range", "loss"]
    samples: ValidationSamples


class ValidationRules(BaseModel):
    min_cases: int
    min_contributed: int
    levels: List[int]
    z: float


class ValidationCases(BaseModel):
    library: int
    contributed_deals: Optional[int] = Field(description="Null when fewer than min_contributed")


class ValidationReportOut(BaseModel):
    generated_at: str = Field(description="UTC, ISO 8601")
    engine_version: str
    library_included: bool
    fit_until: Dict[str, int] = Field(description="Per check, the newest year of data its predictions read")
    rules: ValidationRules
    cases: ValidationCases
    dimensions: List[str]
    checks: List[ValidationCheck]


class ValidationReportResponse(BaseModel):
    report: Optional[ValidationReportOut] = Field(description="The newest report; null before the first")


# ---------------------------------------------------------------------------
# The deal risk score (PLAN.md 5.2, ml/anomaly_detector.py)
SpRegion = Literal["us", "europe", "emerging", "other_developed"]
Verdict = Literal["beats_baseline", "does_not_beat_baseline", "not_enough_data"]


class PeerSkipped(BaseModel):
    area: str = Field(description="A group (benchmarks/catalogue.py REGIONS)")
    reason: Literal["thin", "unusable", "missing", "few_industries"] = Field(
        description="thin: fewer companies than min_firms; few_industries: too few industries in the group "
                    "to say how much they differ; the rest as for the starting figures")
    sample: Optional[int] = Field(None, description="Companies (industries, for few_industries), when known")


class PeerComparison(BaseModel):
    metric: Literal["leverage", "entry_multiple", "ebitda_margin"]
    scored: bool = Field(description="Part of the score; the margin is shown but not scored")
    deal: Optional[float] = Field(description="The deal's figure: a multiple, or per cent for the margin")
    status: Literal["ok", "not_enough_data"]
    peer: Optional[float] = Field(None, description="The industry's figure in the group, in the deal's terms")
    spread: Optional[float] = Field(None, description="How much the group's industries differ (robust spread)")
    z: Optional[float] = Field(None, description="(deal - peer) / spread")
    risk_z: Optional[float] = Field(None, description="z signed so that positive is riskier")
    position: Optional[Literal["well_below", "below", "in_line", "above", "well_above"]] = None
    group: Optional[BenchmarkArea] = Field(None, description="The group the industry's figure is from; never global")
    level: Optional[BenchmarkLevel] = None
    firms: Optional[int] = Field(None, description="Companies behind the industry's figure")
    industries: Optional[int] = Field(None, description="Industries the spread is measured over")
    published: Optional[date] = None
    url: Optional[str] = None
    skipped: List[PeerSkipped] = Field(description="Closer groups passed over, and why")


class PeerSample(BaseModel):
    firms: int = Field(description="The fewest companies behind any figure compared")
    group: BenchmarkArea


class DealRiskScore(BaseModel):
    shown: bool = Field(description="Whether the score's card says it beats the baseline in the deal's region")
    value: Optional[float] = Field(None, description="The score, only when shown: the sum of how far leverage "
                                                     "and the entry multiple sit above the industry's, in spreads")
    region: Optional[SpRegion] = None
    verdict: Verdict = Field(description="The card's verdict for the region (docs/model-cards/deal_risk.md)")
    cases: int = Field(description="Reference transactions the card tested in the region")
    model: Optional[float] = Field(None, description="The card's headline statistic (AUC) for the score there")
    baseline: Optional[float] = Field(None, description="The same for leverage alone")


class SimilarDeal(BaseModel):
    key: str
    target: str
    country: str
    sector: str
    size: Optional[str] = None
    year: int = Field(description="The year it closed")
    outcome: Literal["success", "distress", "held"]
    event: str
    entry_multiple: Optional[float] = None
    leverage: Optional[float] = None
    same_size: bool


class SimilarDeals(BaseModel):
    enabled: bool = Field(description="Whether the reference library is on; off, no deals are listed")
    region: Optional[SpRegion] = None
    sector: Optional[str] = Field(None, description="The deal industry's GICS sector")
    size: Optional[str] = Field(None, description="The deal's size bucket (entry value in US dollars), when known")
    deals: List[SimilarDeal] = Field(description="Approved reference transactions in the same region and sector")


class DealRiskResponse(BaseModel):
    status: Literal["ok", "not_enough_data"]
    reason: Optional[Literal["no_country", "no_peers"]] = None
    country: str
    industry: str
    industry_name: Optional[str] = None
    region: Optional[SpRegion] = Field(None, description="The country's S&P region (the card's regions)")
    min_firms: int = Field(description="A peer group needs this many companies")
    sample: Optional[PeerSample] = Field(None, description="\"Based on N companies in [group, industry]\"")
    inputs: Dict[str, float] = Field(description="What the score read of the deal")
    comparisons: List[PeerComparison]
    unusual: bool = Field(description="A figure sits two spreads or more on the risky side of its industry's")
    score: DealRiskScore
    deals: SimilarDeals
    notes: List[Literal["size_not_split", "listed_company_figures"]]
    source: BenchmarkSourceOut
    model: ModelStamp


# ---------------------------------------------------------------------------
# Multiple predictor (PLAN.md 5.4)
# ---------------------------------------------------------------------------
class MultiplesRequest(Strict):
    inputs: DealInputsIn = DealInputsIn()


class MultipleBand(BaseModel):
    low: float = Field(description="The 10th percentile: the range's low end, x EBITDA")
    median: float = Field(description="The 50th percentile: the suggestion")
    high: float = Field(description="The 90th percentile: the range's high end")
    pairs: int = Field(description="Industry moves over this many years in the group the range rests on")


class MultipleCardResult(BaseModel):
    region: Optional[SpRegion] = Field(None, description="The S&P region whose cases the card tested: the peer "
                                                         "group's")
    verdict: Verdict = Field(description="The card's verdict for this range in the region "
                                         "(docs/model-cards/multiples.md)")
    cases: int = Field(description="Industry-years the card tested in the region")
    model: Optional[float] = Field(None, description="The card's headline statistic (interval score) for the range")
    baseline: Optional[float] = Field(None, description="The same for the region's whole market")


class MultipleRange(BaseModel):
    horizon: int = Field(description="Years after the latest published multiple")
    year: int = Field(description="The year the range is for")
    shown: bool = Field(description="Whether the card tested this horizon and says the range beats the baseline "
                                    "in the region of the group it is built from")
    hidden: Optional[Literal["untested_horizon", "few_moves", "does_not_beat_baseline", "not_enough_data"]] = Field(
        None, description="Why it is not shown: a horizon the card didn't test, too few moves over it in the "
                          "group, or the card's verdict")
    range: Optional[MultipleBand] = Field(None, description="Only when shown")
    card: MultipleCardResult


class MultipleByGroup(BaseModel):
    group: BenchmarkArea
    level: Literal["country", "region", "global"]
    used: bool = Field(description="The peer group the ranges are built in")
    enough: bool = Field(description="At least min_firms companies and a usable multiple")
    year: int
    firms: Optional[int] = None
    multiple: Optional[float] = None


class MultipleSectorPeer(BaseModel):
    industry: str
    name: str
    this: bool = Field(description="The deal's own industry")
    year: int
    firms: Optional[int] = None
    multiple: float


class MultiplesResponse(BaseModel):
    status: Literal["ok", "not_enough_data"]
    reason: Optional[Literal["no_country", "no_peers"]] = None
    country: str
    industry: str
    industry_name: Optional[str] = None
    region: Optional[SpRegion] = Field(None, description="The country's S&P region (the card's regions)")
    hold: int
    min_firms: int
    min_pairs: int = Field(description="A range needs this many industry moves in the group")
    quantiles: List[float]
    group: Optional[BenchmarkArea] = Field(None, description="The peer group: never the global one")
    level: Optional[Literal["country", "region"]] = None
    firms: Optional[int] = Field(None, description="Companies behind the industry's latest multiple in the group")
    latest_year: Optional[int] = Field(None, description="The year the latest published multiple describes")
    latest: Optional[float] = Field(None, description="The industry's latest multiple in the group")
    published: Optional[date] = None
    url: Optional[str] = None
    skipped: List[PeerSkipped]
    entry: Optional[MultipleRange] = None
    exit: Optional[MultipleRange] = None
    regions: List[MultipleByGroup] = Field(description="The industry's latest multiple in every group")
    sector: List[MultipleSectorPeer] = Field(description="The group's industries in the same GICS sector")
    deals: SimilarDeals
    notes: List[Literal["size_not_split", "listed_company_figures"]]
    source: BenchmarkSourceOut
    model: ModelStamp


# ---------------------------------------------------------------------------
# Growth calibrator (PLAN.md 5.5)
# ---------------------------------------------------------------------------
class GrowthRequest(Strict):
    inputs: DealInputsIn = DealInputsIn()


class GrowthBand(BaseModel):
    low: float = Field(description="The 10th percentile of yearly revenue growth over the hold, %")
    median: float = Field(description="The 50th percentile, %")
    high: float = Field(description="The 90th percentile, %")


class GrowthObserved(BaseModel):
    low: float = Field(description="The 10th percentile of companies' yearly growth over their economy's, as a "
                                   "rate: (1 + company) / (1 + economy) - 1, %")
    median: float
    high: float
    companies: int = Field(description="Companies the range rests on")
    cases: int = Field(description="Company spans of the hold's length")
    first: int = Field(description="The first year of revenue read")
    last: int = Field(description="The last year of revenue read")


class GrowthSettings(BaseModel):
    mc_growth_mean: float = Field(description="The simulation's growth mean, %: the range's middle")
    mc_growth_std: float = Field(description="Its spread, %: the normal draw whose central 80% is the range")


class GrowthCardResult(BaseModel):
    region: Optional[SpRegion] = None
    verdict: Verdict = Field(description="The card's verdict in the region (docs/model-cards/growth.md)")
    cases: int = Field(description="Company spans the card tested in the region")
    model: Optional[float] = Field(None, description="The card's interval score for the range, points")
    baseline: Optional[float] = Field(None, description="The same for the economy-wide range in force")


class GrowthSource(BaseModel):
    publisher: str
    title: str
    url: str


class GrowthResponse(BaseModel):
    status: Literal["ok", "not_enough_data"]
    reason: Optional[Literal["no_country", "no_growth"]] = Field(
        None, description="No country chosen, or no economic data stored for it")
    country: str
    industry: str
    region: Optional[SpRegion] = Field(None, description="The country's S&P region (the card's regions)")
    sector: Optional[str] = Field(None, description="The industry's GICS sector")
    group_sector: Optional[str] = Field(None, description="The sector the range rests on; null when the "
                                                          "region's every sector stands in")
    hold: int
    horizons: List[int] = Field(description="The holds the card tested")
    quantiles: List[float]
    min_companies: int
    floor_usd_m: float = Field(description="Companies with less revenue than this in the start year are left out, "
                                           "millions of US dollars")
    shown: bool
    hidden: Optional[Literal["untested_horizon", "does_not_beat_baseline", "not_enough_data"]] = None
    anchor: Optional[StartingFigure] = Field(None, description="The country's nominal growth the range is centred on")
    range: Optional[GrowthBand] = Field(None, description="Only when shown")
    settings: Optional[GrowthSettings] = Field(None, description="Only when shown")
    observed: Optional[GrowthObserved] = None
    card: GrowthCardResult
    years: List[int] = Field(description="The first and last year of revenue in the data")
    data_written_on: date
    sources: Dict[str, GrowthSource]
    notes: List[Literal["listed_companies", "survivors_only", "acquisitions_included", "size_floor"]]
    model: ModelStamp
