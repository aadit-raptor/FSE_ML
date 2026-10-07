"""Monte Carlo: simulation parameters, scenario presets and result analytics.

Moved from app.py (page_monte_carlo and _get_scenario_params_cfg). Operating
and capital-structure assumptions come from the deal wizard inputs, exactly as
on the Monte Carlo page; percentages are numbers like 5.0.
"""
import copy
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from scipy.stats import spearmanr

from analytics.risk_metrics import calculate_risk_metrics
from core.config import build_corr_matrix
from core.deal import DealInputs, entry_costs
from core.tax import rules_from_deal
from core.accounting import lease_terms
from core.debt import (
    build_capital_structure, check_specs, financing_fees, shift_references, simulation_tranches,
    total_debt as tranche_debt,
)
from simulation.vectorized_simulation import SimulationParams, run_vectorized_simulation_full

SCENARIOS = ["recession", "base", "bull", "stagflation"]


@dataclass
class MCInputs:
    """Monte Carlo page inputs. Defaults match the page's first load."""
    n: int = 50000
    ebitda: float = 100.0        # entry EBITDA (money, in the deal's currency and unit)
    entry_mult: float = 10.0
    hold: int = 5
    hurdle: float = 20.0         # %
    growth_mean: float = 5.0
    growth_std: float = 3.0
    exit_mean: float = 10.0
    exit_std: float = 1.5
    rate_mean: float = 6.5
    rate_std: float = 1.5
    gm_mean: float = 40.0
    gm_std: float = 3.0


# Money in a simulation answer: the parameters it ran with
MC_MONEY_KEYS = frozenset({"entry_ebitda", "other_uses"})


def mc_in_millions(mc: MCInputs, deal: DealInputs, cfg: Mapping):
    """Simulation inputs with money in millions (core.deal.in_millions)."""
    from core.deal import in_millions
    from core.money import to_millions
    mc = replace(mc, ebitda=to_millions(mc.ebitda, deal.unit))
    deal, cfg = in_millions(deal, cfg)
    return mc, deal, cfg


def build_sim_params(mc: MCInputs, deal: DealInputs, cfg: Mapping) -> SimulationParams:
    """The simulation's parameters for a deal, its Monte Carlo inputs and
    Settings.

    A deal that lists its tranches is simulated with them (PLAN.md 2.4b): the
    same structure the deal screen shows, refused the same way when it raises
    more than the deal costs. ``debt_pct``, ``senior_pct`` and ``mezz_spread``
    are still filled in, and ignored by the tranche path.
    """
    # The rail's EBITDA is as the deal's standard reports it; the simulation
    # grows it after lease costs and values it the way the deal is priced
    # (PLAN.md 2.6). No leases: the rail's EBITDA and two zeros.
    leases = lease_terms(mc.ebitda, deal.accounting_standard, deal.lease_view,
                         deal.lease_cost, deal.lease_liability)
    tranches = ()
    if deal.tranches:
        entry_ev = (leases.operating_ebitda + leases.valuation_addback) * mc.entry_mult
        check_specs(deal.tranches, entry_ev - leases.debt_like,
                    entry_costs(entry_ev, tranche_debt(deal.tranches), cfg)
                    + financing_fees(deal.tranches))
        tranches = simulation_tranches(deal.tranches, int(mc.hold))
    return SimulationParams(
        lease_ebitda_addback=leases.valuation_addback, lease_liability=leases.debt_like,
        n=int(mc.n), entry_ebitda=leases.operating_ebitda, entry_multiple=mc.entry_mult,
        holding_period=int(mc.hold),
        growth_mean=mc.growth_mean/100, growth_std=mc.growth_std/100,
        exit_mean=mc.exit_mean, exit_std=mc.exit_std,
        interest_mean=mc.rate_mean/100, interest_std=mc.rate_std/100,
        gross_margin_mean=mc.gm_mean/100, gross_margin_std=mc.gm_std/100,
        opex_pct=deal.opex/100, da_pct=deal.da/100, tax_rate=deal.tax/100,
        capex_pct=deal.capex/100, nwc_pct=deal.nwc/100,
        debt_pct=deal.debt_pct/100, senior_pct=deal.senior_pct/100,
        mezz_spread=deal.mezz_spread/100,
        transaction_fees_pct=cfg['tx_fee_pct']/100,
        financing_fees_pct=cfg['fin_fee_pct']/100,
        other_uses=cfg['other_uses'],
        senior_amort_pct=cfg['def_senior_amort']/100,
        n_interest_passes=int(cfg['mc_n_passes']),
        clip_irr=bool(cfg['mc_clip_irr']),
        corr_matrix=build_corr_matrix(cfg),
        tranches=tranches,
        tax_rules=rules_from_deal(deal),
    )


def apply_scenario(scenario: str, base_params: SimulationParams, cfg: Mapping) -> SimulationParams:
    """Shift distribution means by a scenario preset's settings multipliers.

    For a deal listing its tranches, the rate draw is a shock around
    ``interest_mean`` to each floating facility's reference rate, so a
    scenario that moves the mean would otherwise move nothing. The change in
    the mean is added to every floating reference instead: stagflation's
    higher rates reach floating debt, and fixed debt does not care.
    """
    p = copy.deepcopy(base_params)
    if scenario == "bull":
        p.growth_mean       *= cfg["bull_growth_mult"]
        p.exit_mean         *= cfg["bull_exit_mult"]
        p.interest_mean     *= cfg["bull_rate_mult"]
        p.gross_margin_mean *= cfg["bull_margin_mult"]
    elif scenario == "base":
        pass
    elif scenario == "recession":
        p.growth_mean        = max(p.growth_mean + cfg["rec_growth_adj"]/100,
                                   cfg["rec_growth_floor"]/100)
        p.exit_mean         *= cfg["rec_exit_mult"]
        p.interest_mean     *= cfg["rec_rate_mult"]
        p.gross_margin_mean *= cfg["rec_margin_mult"]
    elif scenario == "stagflation":
        p.growth_mean        = max(p.growth_mean + cfg["stag_growth_adj"]/100,
                                   cfg["stag_growth_floor"]/100)
        p.exit_mean         *= cfg["stag_exit_mult"]
        p.interest_mean     *= cfg["stag_rate_mult"]
        p.gross_margin_mean *= cfg["stag_margin_mult"]
    if p.tranches:
        p.tranches = shift_references(p.tranches, p.interest_mean - base_params.interest_mean)
    return p


def risk_summary(sim, hurdle_pct):
    """Headline risk metrics shown above the Monte Carlo tabs."""
    target = hurdle_pct / 100
    metrics = calculate_risk_metrics(sim.df, target)
    return {
        "mean_irr": metrics["Mean IRR"],
        "median_irr": metrics["Median IRR"],
        "p5_irr": metrics["5% Downside IRR"],
        "p95_irr": metrics["95% Upside IRR"],
        "p_above_hurdle": metrics["Probability IRR > Target"],
        "wipeout_rate": sim.wipeout_rate,
        "p_loss": probability_of_loss(sim.moic),
        "hurdle": target,
    }


def probability_of_loss(moic) -> float:
    """Share of paths returning less than the equity put in (MOIC below 1),
    the deal summary's probability of loss (PLAN.md 4.6)."""
    moic = np.asarray(moic, dtype=float)
    return float(np.mean(moic < 1.0)) if moic.size else float("nan")


def analysis_sample(sim):
    """The sample the scatter, correlation and driver analyses are computed on."""
    df = sim.df
    return df.sample(min(50_000, len(df)), random_state=42)


def empirical_correlations(sample):
    corr_cols = ["IRR", "MOIC", "Growth", "Exit Multiple", "Interest", "Gross Margin"]
    return sample[corr_cols].corr()


def driver_sensitivity(sample):
    """Spearman rho of each driver with IRR, largest magnitude first."""
    drv = ["Growth", "Exit Multiple", "Interest", "Gross Margin", "EBITDA Shock"]
    rhos = []
    for d in drv:
        if d in sample.columns:
            rho, _ = spearmanr(sample[d], sample["IRR"])
            rhos.append((d, rho))
    rhos.sort(key=lambda x: abs(x[1]), reverse=True)
    return rhos


def driver_fits(sample):
    """Per-driver linear fit and correlation for the IRR scatter plots."""
    out = {}
    for col in ["Growth", "Exit Multiple", "Interest", "Gross Margin"]:
        x = sample[col].values; y = sample["IRR"].values * 100
        m_fit, b_fit = np.polyfit(x, y, 1)
        corr = np.corrcoef(x, y)[0, 1]
        out[col] = {"slope": m_fit, "intercept": b_fit, "r": corr}
    return out


def growth_exit_heatmap(params: SimulationParams, mc: MCInputs, deal: DealInputs):
    """IRR grid over growth x exit multiple, as on the Sensitivity tab.

    Each cell is a full deterministic deal-model run (fees, debt paydown and
    the interest loop included) at the simulation's mean assumptions for
    that growth and exit multiple. The Streamlit version was a fee-free
    closed form with exit debt assumed at 70% of entry debt (finding 3).

    A deal that lists its tranches runs on them, with a scenario's rate stress
    on the floating ones (the move in ``interest_mean`` from the rail's own
    rate mean). Left out, every cell would quietly revert to the percentages.
    """
    from core.deal import INTEREST_TOLERANCE, MAX_INTEREST_PASSES
    from lbo_engine.model import LBOParams, run_lbo

    g_vals = np.linspace(
        max(params.growth_mean - 3*params.growth_std, -0.10),
        params.growth_mean + 3*params.growth_std, 8)
    em_vals = np.linspace(
        max(params.exit_mean - 2*params.exit_std, 2.0),
        params.exit_mean + 2*params.exit_std, 7)
    structure, fees = None, 0.0
    if deal.tranches:
        shift = params.interest_mean - mc.rate_mean / 100
        structure = build_capital_structure(deal.tranches, params.entry_ebitda,
                                            int(params.holding_period), reference_shift=shift)
        fees = financing_fees(deal.tranches)
    irr_grid = np.zeros((len(em_vals), len(g_vals)))
    for i, em in enumerate(em_vals):
        for j, g in enumerate(g_vals):
            r = run_lbo(LBOParams(
                capital_structure=structure, tranche_fees=fees, tax_rules=params.tax_rules,
                lease_ebitda_addback=params.lease_ebitda_addback,
                lease_liability=params.lease_liability,
                entry_ebitda=params.entry_ebitda, entry_multiple=params.entry_multiple,
                exit_multiple=float(em), holding_period=int(params.holding_period),
                debt_pct=params.debt_pct, senior_pct=params.senior_pct,
                mezz_spread=params.mezz_spread, interest_rate=params.interest_mean,
                senior_amort_pct=params.senior_amort_pct,
                revenue_growth=float(g), gross_margin=params.gross_margin_mean,
                opex_pct=params.opex_pct, da_pct=params.da_pct, tax_rate=params.tax_rate,
                capex_pct=params.capex_pct, nwc_pct=params.nwc_pct,
                transaction_fees_pct=params.transaction_fees_pct,
                financing_fees_pct=params.financing_fees_pct,
                other_uses=params.other_uses,
                n_iterations=MAX_INTEREST_PASSES, interest_tolerance=INTEREST_TOLERANCE,
                compute_sensitivity=False,
            ))
            irr_grid[i, j] = r.returns.irr
    return g_vals, em_vals, irr_grid


def run_scenarios(params: SimulationParams, cfg: Mapping, seed=None):
    """Run every scenario preset from the same base parameters."""
    return {sc: run_vectorized_simulation_full(apply_scenario(sc, params, cfg), seed=seed)
            for sc in SCENARIOS}


def scenario_stats(scenario_results, hurdle_pct):
    target = hurdle_pct / 100
    out = {}
    for sc in SCENARIOS:
        sr = scenario_results[sc]
        irr_s = sr.irr
        out[sc] = {
            "mean_irr": float(irr_s.mean()),
            "median_irr": float(np.median(irr_s)),
            "p5_irr": float(np.percentile(irr_s, 5)),
            "p95_irr": float(np.percentile(irr_s, 95)),
            "p_above_hurdle": float((irr_s > target).mean()),
            "wipeout_rate": float(sr.wipeout_rate),
        }
    return out
