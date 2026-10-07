"""The optional reference library (PLAN.md 4.5).

``GET /library`` says whether it is shown and whether the caller may switch
it; ``PUT /library/switch`` is the administrators' switch. The base rates and
the coverage counts answer ``enabled: false`` and nothing else while it is
off. They read tables kept in code (``library/``), so they are not runs and
cost no database round trip beyond the switch's.
"""
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import AuthUser, require_user
from api.schemas import BaseRatesResponse, CoverageResponse, LibraryState, LibrarySwitch
from db.flags import NoAccount
from library import base_rates, switch
from library.coverage import coverage

router = APIRouter(prefix="/library", tags=["library"])


@router.get("", response_model=LibraryState)
def get_state(user: AuthUser = Depends(require_user)):
    """Whether the library is shown, and whether the caller may switch it."""
    return switch.state(user.subject)


@router.put("/switch", response_model=LibraryState)
def put_switch(body: LibrarySwitch, user: AuthUser = Depends(require_user)):
    """Show or hide the library for everyone (administrators only)."""
    try:
        return switch.switch(user.subject, body.enabled)
    except switch.NotAdmin:
        raise HTTPException(403, "Only an administrator can show or hide the reference library.") from None
    except switch.LockedOff:
        raise HTTPException(409, "The server's configuration keeps the reference library off "
                                 "(FSE_EXAMPLE_LIBRARY); change it there.") from None
    except switch.NoStore:
        raise HTTPException(503, "Switching the library needs a database: set DATABASE_URL.") from None
    except NoAccount:
        raise HTTPException(409, "Finish setting up your account first.") from None


@router.get("/base-rates", response_model=BaseRatesResponse)
def get_base_rates(country: Optional[str] = Query(None, pattern="^[A-Za-z]{2}$",
                                                  description="ISO country code, to name its S&P region")):
    """Published default and recovery rates by region, rating band and year, each with its source."""
    if not switch.enabled():
        return {"enabled": False, "sources": [], "tables": []}
    return {"enabled": True, **base_rates.base_rates(date.today(), country)}


@router.get("/coverage", response_model=CoverageResponse)
def get_coverage():
    """What the library covers, in counts by region, size, sector, era and outcome."""
    if not switch.enabled():
        return {"enabled": False, "collections": [], "base_rates": []}
    return {"enabled": True, **coverage()}
