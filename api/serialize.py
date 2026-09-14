"""Turn model results into JSON, and generate response schemas from them."""
import dataclasses
import math
import typing
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, create_model


def to_json(x):
    """Dataclasses, numpy and pandas values -> JSON-safe Python.

    NaN and infinity become None: JSON has no representation for them, and the
    models can produce NaN (for example IRR when equity is wiped out).
    """
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: to_json(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_json(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


_MODELS: Dict[type, type] = {}


def _nullable(tp):
    """Map a dataclass field type to a response type that tolerates None."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is float:
        return Optional[float]
    if dataclasses.is_dataclass(tp):
        return Optional[model_from_dataclass(tp)]
    if origin in (list, List):
        return List[_nullable(args[0])] if args else List[Any]
    if origin in (dict, Dict):
        return Dict[str, Any]
    if origin is Union:
        return Optional[Union[tuple(_nullable(a) for a in args if a is not type(None))]]
    return tp


def model_from_dataclass(dc: type) -> type:
    """A pydantic model mirroring an engine dataclass, for response schemas.

    Generated rather than hand-written so the API schema cannot drift from the
    model's own result types.
    """
    if dc not in _MODELS:
        hints = typing.get_type_hints(dc)
        fields = {f.name: (_nullable(hints[f.name]), None) for f in dataclasses.fields(dc)}
        _MODELS[dc] = create_model(dc.__name__, __base__=BaseModel, **fields)
    return _MODELS[dc]


def histogram(values, bins):
    """Density histogram: bin edges (len bins+1) and densities (len bins)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    density, edges = np.histogram(values, bins=bins, density=True)
    return {"edges": edges.tolist(), "density": density.tolist()}


def percentile_curve(values, step=1.0):
    """Points (percentile, value) for a CDF chart, from 0 to 100."""
    values = np.asarray(values, dtype=float)
    qs = np.arange(0.0, 100.0 + step / 2, step)
    return {"percentiles": qs.tolist(), "values": np.percentile(values, qs).tolist()}


def box_stats(values):
    values = np.asarray(values, dtype=float)
    return {q: float(np.percentile(values, int(q[1:]))) for q in ("p5", "p25", "p50", "p75", "p95")}
