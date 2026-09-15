"""Request and response schemas.

Input bounds mirror the Streamlit input widgets. Percentages are numbers like
60.0 (not 0.60) and money is $M, matching the model's inputs; engine outputs
keep the engine's own units (IRR 0.157 = 15.7%).
"""
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from api.serialize import model_from_dataclass
from core.forecasting import ForecastYear, HistoricalYear
from lbo_engine.cashflow_model import CashFlowResult
from lbo_engine.operating_model import OperatingModelResult
from lbo_engine.returns import ReturnsResult

SettingValue = Union[float, int, bool]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------
class DealInputsIn(Strict):
    """Deal wizard inputs.

    debt_pct and senior_pct default to the model's stored defaults (60% / 70%).
    The Streamlit wizard instead derives them on load from its financing
    inputs -- 3.4x senior + 0.8x mezz EBITDA, i.e. 42% and ~81% -- which is why
    its default deal shows a lower IRR. Use /deal/sources-and-uses to derive
    them from debt multiples the same way.
    """
    ebitda: float = Field(100.0, gt=0, description="LTM EBITDA ($M)")
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
    mincash: float = Field(0.0, ge=0, description="Minimum cash ($M)")
    wsp_mode: bool = Field(False, description="Use AR/inventory/AP days instead of flat NWC")
    ar_days: float = Field(45.0, ge=0, le=365)
    inv_days: float = Field(30.0, ge=0, le=365)
    ap_days: float = Field(60.0, ge=0, le=365)


class MCInputsIn(Strict):
    n: int = Field(50000, ge=1000, le=1_000_000, description="Scenarios to simulate")
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
    mincash: float = Field(0.0, ge=0, description="Minimum cash left on the balance sheet ($M)")
    settings: Dict[str, SettingValue] = {}


class SourcesUsesResponse(BaseModel):
    senior_debt: float
    mezz_debt: float
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
    interest_rate: Optional[float]


class DealRunResponse(BaseModel):
    returns: model_from_dataclass(ReturnsResult)
    operating_model: model_from_dataclass(OperatingModelResult)
    cash_flow: model_from_dataclass(CashFlowResult)
    debt_schedule: Dict[str, object] = Field(description="Totals per year; per-tranche detail in tranches")
    tranches: Dict[str, List[TrancheYear]]
    equity_bridge: EquityBridge
    bridge_steps: List[BridgeStep]
    exit_sensitivity: ExitSensitivity
    interest_converged: bool


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


class HistoryRequest(Strict):
    history: Dict[str, List[float]] = Field(description="field key -> one value per historical year, oldest first")


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


class ForecastRunRequest(HistoryRequest):
    assumptions: Dict[str, List[float]] = Field(description="assumption key -> one value per forecast year")
    simulate: bool = False
    n_sim: int = Field(30000, ge=1000, le=200_000)


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


class BacktestRequest(Strict):
    entry: BacktestEntry
    actual: BacktestActuals
    actual_exit: BacktestActualExit
    settings: Dict[str, SettingValue] = {}
    n: int = Field(30000, ge=1000, le=200_000)
    histogram_bins: int = Field(80, ge=10, le=400)


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
        description="Exact split of actual minus predicted exit equity ($M): exit_ebitda, exit_multiple, net_debt")
    irr_histogram: Histogram
    years: List[BacktestYear]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
Cell = Union[float, int, str, bool, None]


class WorkbookSheet(Strict):
    name: str = Field(min_length=1, max_length=100)
    columns: List[str] = Field(min_length=1, max_length=200)
    rows: List[List[Cell]] = Field(max_length=50_000)


class WorkbookRequest(Strict):
    filename: str = Field("export.xlsx", max_length=120)
    sheets: List[WorkbookSheet] = Field(min_length=1, max_length=30)


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
