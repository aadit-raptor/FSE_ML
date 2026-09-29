"""Shared request helpers."""
from typing import Mapping, Optional

from fastapi import HTTPException

from core.config import build_corr_matrix, is_valid_corr, resolve_config


# Interest passes a simulation may run: the Settings screen offers the same range
MIN_PASSES, MAX_PASSES = 1, 10


def resolve_settings(overrides: Optional[Mapping], *, check_correlations: bool = False) -> dict:
    """Settings with overrides applied; 422 for unknown keys or an invalid matrix."""
    try:
        cfg = resolve_config(overrides)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e.args[0])) from None
    passes = cfg.get("mc_n_passes")
    if not (isinstance(passes, (int, float)) and not isinstance(passes, bool)
            and float(passes).is_integer() and MIN_PASSES <= passes <= MAX_PASSES):
        # Each pass reruns the whole simulation; a thousand of them would hold
        # the one simulation slot for an hour (PLAN.md 2.4b security review)
        raise HTTPException(
            status_code=422,
            detail=f"mc_n_passes must be a whole number from {MIN_PASSES} to {MAX_PASSES}")
    if check_correlations and not is_valid_corr(build_corr_matrix(cfg)):
        raise HTTPException(
            status_code=422,
            detail="correlation settings do not form a valid (positive semi-definite) matrix")
    return cfg
