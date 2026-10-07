"""The optional reference library (PLAN.md 4.5).

``GET /library`` says whether it is shown and whether the caller may switch
it; ``PUT /library/switch`` is the administrators' switch. The base rates,
the coverage counts, the reference transactions and the sourced fees answer
``enabled: false`` and nothing else while it is off. The base rates are
tables kept in code; the reference transactions are the approved ones in the
database, so a server without a database has none. None of it is a run.

``/library/review`` is Library -> Review (PLAN.md 4.5b): administrators
propose reference transactions and approve or reject them, two approvals
from administrators other than the proposer admitting one
(``library/review.py``, ``db/references.py``).
"""
import uuid
from contextlib import contextmanager
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import AuthUser, require_user
from api.schemas import (
    BaseRatesResponse, CoverageResponse, LibraryState, LibrarySwitch, ReferenceDealIn, ReferenceDealsResponse,
    ReviewProposal, ReviewQueue, ReviewVerdict, SourcedFeesResponse,
)
from db import DatabaseUnavailable
from db import references as store
from db.engine import is_configured as database_configured
from db.flags import NoAccount
from library import base_rates, fees, references, review, switch
from library.coverage import coverage

router = APIRouter(prefix="/library", tags=["library"])

UNREACHABLE = "The reference library's database can't be reached just now; try again in a minute."
NO_DATABASE = "Reviewing reference transactions needs a database: set DATABASE_URL."
OFF = "The reference library is switched off."


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


def _library() -> tuple[list[dict], int]:
    """The approved reference transactions and how many await review (the
    repository's proposals queued first, so they count before anyone opens
    Review); none without a database."""
    if not database_configured():
        return [], 0
    try:
        review.sync()
        return store.library()
    except DatabaseUnavailable:
        raise HTTPException(503, UNREACHABLE) from None


@router.get("/coverage", response_model=CoverageResponse)
def get_coverage():
    """What the library covers, in counts by region, size, sector, era and outcome."""
    if not switch.enabled():
        return {"enabled": False, "collections": [], "base_rates": []}
    deals, awaiting = _library()
    return {"enabled": True, **coverage(deals, awaiting)}


@router.get("/references", response_model=ReferenceDealsResponse)
def get_references():
    """The approved reference transactions, every figure with its filing."""
    if not switch.enabled():
        return {"enabled": False, "deals": [], "awaiting_review": 0}
    deals, awaiting = _library()
    return {"enabled": True, "deals": [references.summary(d) for d in deals], "awaiting_review": awaiting}


@router.get("/fees", response_model=SourcedFeesResponse)
def get_fees():
    """Transaction fees, financing fees and senior amortisation from the
    approved reference transactions' filings, offered as Settings."""
    if not switch.enabled():
        return {"enabled": False, "min_deals": fees.MIN_DEALS, "library_size": 0,
                "settings": {k: None for k in fees.SETTINGS}}
    deals, _ = _library()
    return {"enabled": True, "min_deals": fees.MIN_DEALS, "library_size": len(deals),
            "settings": fees.sourced(deals)}


@contextmanager
def _review_errors():
    if not switch.enabled():
        raise HTTPException(409, OFF)
    if not database_configured():
        raise HTTPException(503, NO_DATABASE)
    try:
        yield
    except review.NotAdmin:
        raise HTTPException(403, "Only an administrator can propose or review reference transactions.") from None
    except NoAccount:
        raise HTTPException(409, "Finish setting up your account first.") from None
    except store.NotFound:
        raise HTTPException(404, "No such proposal.") from None
    except store.AlreadyDecided as exc:
        raise HTTPException(409, f"This proposal was already {exc}.") from None
    except store.OwnProposal:
        raise HTTPException(403, "You proposed this transaction: two other administrators review it.") from None
    except store.AlreadyReviewed:
        raise HTTPException(409, "You have already reviewed this proposal.") from None
    except store.Duplicate:
        raise HTTPException(409, "This transaction is already proposed or in the library as it stands.") from None
    except store.TooMany:
        raise HTTPException(409, "The library holds as many proposals as it can; decide some first.") from None
    except store.Blocked as exc:
        raise HTTPException(409, {"message": "The inclusion rules refuse this transaction as it stands.",
                                  "problems": exc.problems}) from None
    except DatabaseUnavailable:
        raise HTTPException(503, UNREACHABLE) from None


@router.get("/review", response_model=ReviewQueue)
def get_review(user: AuthUser = Depends(require_user)):
    """Proposals awaiting review, with the inclusion and balance rules' findings (administrators only)."""
    if not switch.enabled():
        return {"enabled": False, "proposals": [], "decided": [], "library_size": 0}
    with _review_errors():
        return {"enabled": True, **review.queue(user.subject)}


@router.post("/review", response_model=ReviewProposal, status_code=201)
def post_proposal(body: ReferenceDealIn, user: AuthUser = Depends(require_user)):
    """Propose a reference transaction (administrators only); two other administrators decide it."""
    with _review_errors():
        return review.propose(user.subject, body.model_dump(mode="json", exclude_none=True))


@router.post("/review/{reference_id}", response_model=ReviewProposal)
def post_verdict(reference_id: uuid.UUID, body: ReviewVerdict, user: AuthUser = Depends(require_user)):
    """Approve or reject a proposal (administrators other than its proposer, once each)."""
    with _review_errors():
        return review.decide(user.subject, reference_id, body.verdict, body.reason)
