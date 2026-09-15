"""Monte Carlo endpoints."""
import time

from fastapi import APIRouter

from api.deps import resolve_settings
from api.observability import model_timer
from api.schemas import MonteCarloRequest, MonteCarloResponse, ScenariosRequest, ScenariosResponse
from api.serialize import box_stats, histogram, percentile_curve, to_json
from core.deal import DealInputs
from core.montecarlo import (
    MCInputs, analysis_sample, apply_scenario, build_sim_params, driver_fits,
    driver_sensitivity, empirical_correlations, growth_exit_heatmap, risk_summary,
    run_scenarios, scenario_stats,
)
from simulation.vectorized_simulation import run_vectorized_simulation_full

router = APIRouter(prefix="/montecarlo", tags=["monte carlo"])

SCATTER_COLUMNS = ["IRR", "MOIC", "Growth", "Exit Multiple", "Interest", "Gross Margin"]


@router.post("/run", response_model=MonteCarloResponse)
def post_run(req: MonteCarloRequest):
    """Simulate the deal and return chart-ready distributions and analytics.

    Raw paths are not returned: histograms, a CDF, a scatter sample and the
    same analytics the Streamlit tabs show are computed server-side.
    """
    cfg = resolve_settings(req.settings, check_correlations=True)
    mc = MCInputs(**req.mc.model_dump())
    deal = DealInputs(**req.deal.model_dump())
    params = build_sim_params(mc, deal, cfg)
    if req.scenario:
        params = apply_scenario(req.scenario, params, cfg)

    t0 = time.perf_counter()
    with model_timer("montecarlo.run"):
        sim = run_vectorized_simulation_full(params, seed=req.seed)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    sample = analysis_sample(sim)
    corr = empirical_correlations(sample)
    scatter = sample[SCATTER_COLUMNS].head(req.scatter_points)
    g_vals, em_vals, grid = growth_exit_heatmap(params, mc, deal)
    params_json = to_json(params)
    return {
        "n": params.n,
        "elapsed_ms": elapsed_ms,
        "scenario": req.scenario,
        "params": params_json,
        "summary": to_json(risk_summary(sim, mc.hurdle)),
        "irr_histogram": histogram(sim.irr, req.histogram_bins),
        "moic_histogram": histogram(sim.moic, req.histogram_bins),
        "irr_cdf": percentile_curve(sim.irr),
        "drivers": [{"driver": d, "spearman_rho": to_json(rho)} for d, rho in driver_sensitivity(sample)],
        "driver_fits": to_json(driver_fits(sample)),
        "correlations": {"labels": list(corr.columns), "matrix": to_json(corr.values)},
        "scatter": {col: to_json(scatter[col].values) for col in SCATTER_COLUMNS},
        "heatmap": {
            "growth": to_json(g_vals), "exit_multiple": to_json(em_vals), "irr": to_json(grid),
            "note": "Each cell is a full deal-model run at the simulation's mean "
                    "assumptions for that growth and exit multiple.",
        },
    }


@router.post("/scenarios", response_model=ScenariosResponse)
def post_scenarios(req: ScenariosRequest):
    """Run all four scenario presets from the same inputs."""
    cfg = resolve_settings(req.settings, check_correlations=True)
    mc = MCInputs(**req.mc.model_dump())
    params = build_sim_params(mc, DealInputs(**req.deal.model_dump()), cfg)
    with model_timer("montecarlo.scenarios"):
        results = run_scenarios(params, cfg, seed=req.seed)
    stats = scenario_stats(results, mc.hurdle)
    return {
        "hurdle": mc.hurdle / 100,
        "scenarios": {
            sc: {**to_json(stats[sc]),
                 "irr_box": to_json(box_stats(results[sc].irr)),
                 "moic_box": to_json(box_stats(results[sc].moic))}
            for sc in results
        },
    }
