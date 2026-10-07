"""Model validation (PLAN.md 4.6): the newest report, read from storage.

The nightly ``validation-report`` task writes it (``validation/run.py``);
this only reads it, so it is not a run. A server without a database has
none, and answers ``report: null`` like one before the first run.
"""
from fastapi import APIRouter, HTTPException

from api.schemas import ValidationReportResponse
from db import DatabaseUnavailable
from db.engine import is_configured as database_configured

router = APIRouter(prefix="/validation", tags=["validation"])

UNREACHABLE = "The validation report's database can't be reached just now; try again in a minute."


@router.get("/report", response_model=ValidationReportResponse)
def get_report():
    """The newest model validation report: calibration and bias of the
    model's ranges and default risk, overall and by region, sector, size and
    era, out of time and in-sample. Holds no deal, owner or figure."""
    if not database_configured():
        return {"report": None}
    from db import validation as store
    try:
        return {"report": store.latest_report()}
    except DatabaseUnavailable:
        raise HTTPException(503, UNREACHABLE) from None
