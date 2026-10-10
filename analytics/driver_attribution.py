"""Why a deal's best and worst simulated outcomes are where they are (PLAN.md 5.6).

A path's IRR is a function of its five draws and nothing else, so the
simulation can be asked what it would have answered had some of them stayed
at their means. Each driver's contribution is its Shapley value over those
32 answers: the average, over every order the drivers could be switched from
their mean to the path's draw, of what switching it moves. That needs no
fitted model, so the contributions add up to the simulated IRR exactly:

    IRR at the mean assumptions + the five contributions = the paths' mean IRR

Nothing here changes a simulated number: it reruns the simulation's own core
on draws it has already made.
"""
import copy
import itertools
import math

import numpy as np

from simulation.vectorized_simulation import SimulationParams, _run_vectorized_core

# (the draw's name in the simulation, its column in the result), in the simulation's order
DRIVERS = (
    ("growth", "Growth"),
    ("exit_multiple", "Exit Multiple"),
    ("interest", "Interest"),
    ("gross_margin", "Gross Margin"),
    ("ebitda_shock", "EBITDA Shock"),
)
# The share of paths in each tail: the paths at and beyond the summary's 5th and 95th percentiles
TAIL_SHARE = 0.05
# The most paths explained in a tail: all of it up to a 50,000-path run. A
# larger tail is read at this many paths evenly spaced through its ranks,
# which keeps the 32 reruns of each to about the cost of the run itself.
MAX_TAIL_PATHS = 2_500


def mean_draws(params: SimulationParams) -> dict:
    """Every driver at its mean, with no EBITDA shock: the path with nothing
    uncertain in it. The means are taken as given: one outside the bounds the
    simulation clips its draws to (a rate mean under 1%, say) is not a value
    any path has, and the contributions still add up from it."""
    return {
        "growth": params.growth_mean,
        "exit_multiple": params.exit_mean,
        "interest": params.interest_mean,
        "gross_margin": params.gross_margin_mean,
        "ebitda_shock": 0.0,
    }


def coalition_irr(params: SimulationParams, paths: dict) -> dict:
    """The paths' mean IRR for every choice of which drivers keep their draws
    (1) and which are held at the mean (0), keyed by that choice.

    Runs in slices of half the simulation's size, so explaining a run holds
    fewer paths at once than the run did.
    """
    base = mean_draws(params)
    count = len(next(iter(paths.values())))
    masks = list(itertools.product((0, 1), repeat=len(DRIVERS)))
    size = max(1, params.n // (2 * len(masks)))
    totals = np.zeros(len(masks))
    for start in range(0, count, size):
        part = {key: values[start:start + size] for key, values in paths.items()}
        k = len(part[DRIVERS[0][0]])
        draws = {
            key: np.concatenate([part[key] if mask[j] else np.full(k, float(base[key])) for mask in masks])
            for j, (key, _) in enumerate(DRIVERS)
        }
        sliced = copy.copy(params)
        sliced.n = k * len(masks)
        totals += _run_vectorized_core(sliced, draws)["IRR"].reshape(len(masks), k).sum(axis=1)
    return dict(zip(masks, totals / count))


def shapley_values(values: dict) -> list:
    """Each player's Shapley value, from the worth of every coalition (keyed
    by a tuple of 0s and 1s). They sum to the worth of everyone less the
    worth of no one."""
    m = len(next(iter(values)))
    out = [0.0] * m
    for mask, worth in values.items():
        size = sum(mask)
        if size == m:
            continue
        weight = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
        for j in range(m):
            if not mask[j]:
                joined = mask[:j] + (1,) + mask[j + 1:]
                out[j] += weight * (values[joined] - worth)
    return out


def tail_paths(irr, share: float = TAIL_SHARE) -> dict:
    """Which paths are each tail: the worst and the best ``share`` of them by
    IRR, at least one. Ties (wiped-out paths all return -100%) go by path
    order, so a tail is never larger than its share."""
    irr = np.asarray(irr, dtype=float)
    order = np.argsort(irr, kind="stable")
    k = min(len(order), max(1, int(round(len(order) * share))))
    return {"downside": order[:k], "upside": order[len(order) - k:]}


def evenly_spaced(rows, limit: int = MAX_TAIL_PATHS):
    """At most ``limit`` of the rows, evenly spaced from the first to the last."""
    if len(rows) <= limit:
        return rows
    return rows[np.linspace(0, len(rows) - 1, limit).round().astype(int)]


def explain_tails(sim, share: float = TAIL_SHARE) -> dict:
    """Each tail's mean IRR split into the five drivers' contributions, from
    the simulation's IRR with every driver at its mean.

    ``tail_paths`` is how many paths the tail holds and ``paths`` how many of
    them the figures average: all of them, or ``MAX_TAIL_PATHS``."""
    df, params = sim.df, sim.params
    cases, base_irr = [], None
    for case, tail in tail_paths(df["IRR"].values, share).items():
        rows = evenly_spaced(tail)
        paths = {key: df[column].values[rows] for key, column in DRIVERS}
        values = coalition_irr(params, paths)
        nobody, everybody = (0,) * len(DRIVERS), (1,) * len(DRIVERS)
        base_irr = float(values[nobody])
        cases.append({
            "case": case,
            "paths": int(len(rows)),
            "tail_paths": int(len(tail)),
            "irr": float(values[everybody]),
            "contributions": [
                {"driver": column, "irr": float(value)}
                for (_, column), value in zip(DRIVERS, shapley_values(values))
            ],
        })
    return {"share": share, "base_irr": base_irr, "cases": cases}
