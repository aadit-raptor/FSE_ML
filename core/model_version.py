"""Which model produced a result (PLAN.md 3.1).

Every result the API answers carries a **stamp**: the engine version, the git
commit the API was built from, a fingerprint of the Settings it ran with and
the vintage of the published data this build carries. Saved deals and their
versions store the stamp beside the deal's IRR and MOIC, so reopening an old
deal can say whether its results have changed since it was saved, and why.

- ``ENGINE_VERSION`` changes whenever any deal's numbers would move. Each
  version is described in ``MODEL_CHANGELOG.md`` (its newest heading must be
  this version), and ``tests/test_model_version.py`` pins reference deals'
  results to it, so a change that moves a number fails until the version is
  raised and the new results recorded.
- The **settings fingerprint** is over the Settings *resolved* against
  today's defaults: an untouched deal stores no overrides, so a changed
  default changes its fingerprint even though nothing it stored did.
- The **data vintage** is the newest edition among the published data sets
  below; ``data_fingerprint`` changes when any one of them does.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Mapping, Optional

ENGINE_VERSION = "1.0.0"

# How long the fingerprints are, in hex digits: short enough to keep a saved
# version small, long enough that two settings never share one by chance
FINGERPRINT_DIGITS = 12
# Results equal within this are the same result: the engine is deterministic,
# but numpy on another platform may differ in the last bits
RESULT_REL_TOL = 1e-9
RESULT_ABS_TOL = 1e-12


def _published_data() -> dict[str, str]:
    """Each published data set the model reads, by id, with its edition date."""
    from core.risk_sources import SOURCES
    from core.tax import PRESETS

    data = {key: src["published"] for key, src in SOURCES.items() if src.get("published")}
    data["tax_presets"] = max(p.as_of for p in PRESETS)
    return dict(sorted(data.items()))


def _digest(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:FINGERPRINT_DIGITS]


def data_sets() -> dict[str, str]:
    return _published_data()


def data_vintage() -> str:
    """The newest edition date among the data sets ("2026-01")."""
    return max(data_sets().values())


def data_fingerprint() -> str:
    return _digest(data_sets())


def settings_fingerprint(cfg: Optional[Mapping]) -> Optional[str]:
    """A short fingerprint of resolved Settings, or None for a result that
    reads none. Whole numbers and their float twins (3 and 3.0) agree."""
    if cfg is None:
        return None

    def plain(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return v
        return float(v)

    return _digest({k: plain(v) for k, v in cfg.items()})


def commit() -> Optional[str]:
    """The git commit this API was built from (Render sets it); None locally."""
    return os.environ.get("RENDER_GIT_COMMIT") or None


def stamp(cfg: Optional[Mapping] = None) -> dict:
    """The stamp every result carries. ``cfg`` is the resolved Settings the
    run used, in the unit the caller sent them (None when it used none)."""
    return {
        "engine_version": ENGINE_VERSION,
        "commit": commit(),
        "settings_fingerprint": settings_fingerprint(cfg),
        "data_vintage": data_vintage(),
        "data_fingerprint": data_fingerprint(),
        "data_sets": data_sets(),
    }


# ---------------------------------------------------------------------------
# Saved deals: the stamp plus what the deal gave, and the check on reopening
# ---------------------------------------------------------------------------
def _finite(x) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def content_fingerprint(inputs: Mapping, settings: Mapping) -> str:
    """A short fingerprint of a saved deal's content (its stored inputs and
    overrides). A stamp carries it so that a stamp left behind by a release
    that didn't write stamps -- one still running during a deploy, or a
    rollback -- is recognised as describing other content."""
    return _digest([dict(inputs), dict(settings)])


def saved_stamp(deal, cfg: Mapping) -> dict:
    """What a saved deal stores: the stamp (without the data set list, which
    the fingerprint stands for) and the deal's IRR and MOIC now.

    ``deal`` is a ``core.deal.DealInputs`` in its own unit and ``cfg`` its
    resolved Settings. A structure the sponsor could not fund has no IRR to
    keep: both results are stored as None.
    """
    from dataclasses import replace

    from core.debt import UnfinanceableStructure
    from core.deal import build_lbo_params, in_millions
    from lbo_engine.model import run_lbo

    out = {k: v for k, v in stamp(cfg).items() if k != "data_sets"}
    # Twelve digits name a commit as well as forty and keep a version small
    out["commit"] = out["commit"][:FINGERPRINT_DIGITS] if out["commit"] else None
    try:
        deal_m, cfg_m = in_millions(deal, cfg)
        # The sensitivity grid reruns the deal 25 times and moves neither figure
        result = run_lbo(replace(build_lbo_params(deal_m, cfg_m), compute_sensitivity=False))
    except UnfinanceableStructure:
        return {**out, "irr": None, "moic": None}
    return {**out, "irr": _finite(result.returns.irr), "moic": _finite(result.returns.moic)}


# What can differ between the stamp a deal was saved with and today's, and the
# word the answer uses for each
STAMP_CAUSES = (
    ("engine_version", "engine_version"),
    ("data_fingerprint", "data"),
    ("settings_fingerprint", "settings"),
)


def _same(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=RESULT_REL_TOL, abs_tol=RESULT_ABS_TOL)


def compare(saved: Optional[Mapping], now: Mapping) -> dict:
    """Whether a saved deal's results have changed since it was saved.

    ``status`` is ``changed`` when its IRR or MOIC differs today,
    ``unchanged`` when both agree, and ``unknown`` for a deal saved before
    stamps were recorded. ``causes`` lists what differs between the two
    stamps (``engine_version``, ``data``, ``settings``) -- possibly several,
    possibly none even when results changed (a commit that forgot to raise
    the version), and possibly some while results agree.
    """
    if not saved or saved.get("content") != now.get("content"):
        # Saved before stamps were kept, or the content changed without its
        # stamp (an older release saved it): nothing to compare with
        return {"status": "unknown", "saved": dict(saved) if saved else None, "now": dict(now),
                "causes": []}
    causes = [word for key, word in STAMP_CAUSES if saved.get(key) != now.get(key)]
    same = _same(saved.get("irr"), now.get("irr")) and _same(saved.get("moic"), now.get("moic"))
    return {"status": "unchanged" if same else "changed", "saved": dict(saved), "now": dict(now),
            "causes": causes}
