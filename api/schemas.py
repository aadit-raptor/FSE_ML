"""Request and response schemas.

Input bounds mirror the Streamlit input widgets. Percentages are numbers like
60.0 (not 0.60) and money is a plain number in the deal's currency and unit
(``Money``; US dollar millions unless the request says otherwise), matching the
model's inputs; engine outputs keep the engine's own units (IRR 0.157 = 15.7%).
Every response that holds money says which currency and unit it is in.
"""
from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from api.limits import MAX_FORECAST_PATHS, MAX_SIMULATION_PATHS

from api.serialize import model_from_dataclass
from core.money import DEFAULT_CURRENCY, DEFAULT_UNIT
from core.forecasting import ForecastYear, HistoricalYear
from lbo_engine.cashflow_model import CashFlowResult
from lbo_engine.operating_model import OperatingModelResult
from lbo_engine.returns import ReturnsResult

SettingValue = Union[float, int, bool]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


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
    amount: float = Field(0.0, ge=0, description="Facility size (in the deal's currency and unit)")
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
    amort_schedule: List[float] = Field(default_factory=list, max_length=MAX_TRANCHE_YEARS,
                                        description="Repayment a year; overrides amort_pct")
    upfront_fee_pct: float = Field(0.0, ge=0, le=10, description="Arrangement fee at close (%)")
    commitment_fee_pct: float = Field(0.0, ge=0, le=5, description="Yearly fee on the undrawn commitment (%)")
    sweep: bool = Field(False, description="Takes the cash sweep")
    sweep_share: float = Field(100.0, ge=0, le=100, description="% of the cash available it may take")
    pik_share: float = Field(0.0, ge=0, le=100, description="% of the coupon that accrues to principal")
    sweep_priority: int = Field(0, ge=0, le=99, strict=True, description="0 = its position in the list")
    allow_redraw: bool = Field(False, description="Draws to cover a cash shortfall (revolver)")

    @field_validator("amort_schedule")
    @classmethod
    def _repayments_are_not_negative(cls, value: List[float]) -> List[float]:
        if any(x < 0 for x in value):
            raise ValueError("a repayment cannot be negative")
        return value


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
    tranches: List[TrancheIn] = Field(
        default_factory=list, max_length=MAX_TRANCHES,
        description="The deal's debt, facility by facility (PLAN.md 2.4). Empty keeps the "
                    "two-tranche sizing above; a list replaces debt_pct, senior_pct, base_rate "
                    "and mezz_spread entirely, and is swept in the order it is given.")

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
    equity_purchase_price: float
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
    money: Money


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
    hurdle: float


class DriverSensitivity(BaseModel):
    driver: str
    spearman_rho: Optional[float]


class DriverFit(BaseModel):
    slope: Optional[float]
    intercept: Optional[float]
    r: Optional[float]


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
    money: Money


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


# ---------------------------------------------------------------------------
# ML and EDGAR
# ---------------------------------------------------------------------------
class Capabilities(BaseModel):
    anomaly_detector: bool
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


class EdgarResponse(BaseModel):
    ticker: str
    company_name: str
    years: List[int]
    history: Dict[str, List[float]] = Field(description="Forecasting history keys, oldest year first")
    warnings: List[str]
    fiscal_year_end_month: Optional[int] = Field(
        None, ge=1, le=12, description="Month the filer's fiscal year ends; years are named by the year they end in")
    money: Money = Field(description="SEC filings are read in US dollars, in millions")


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


class VersionList(BaseModel):
    versions: List[VersionSummary]


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
