"""Shared request helpers."""
from typing import Mapping, Optional

from fastapi import HTTPException

from core.config import build_corr_matrix, is_valid_corr, resolve_config


def resolve_settings(overrides: Optional[Mapping], *, check_correlations: bool = False) -> dict:
    """Settings with overrides applied; 422 for unknown keys or an invalid matrix."""
    try:
        cfg = resolve_config(overrides)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e.args[0])) from None
    if check_correlations and not is_valid_corr(build_corr_matrix(cfg)):
        raise HTTPException(
            status_code=422,
            detail="correlation settings do not form a valid (positive semi-definite) matrix")
    return cfg
