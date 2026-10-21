"""One filing read twice and checked (PLAN.md R1).

1. The page count and each page's scan resolution (``pages.info``, in a
   child process).
2. **Read A**: the whole PDF to the first model. It finds the statement
   pages and reads every table on them.
3. **Read B**: pictures of exactly those pages to a **different** model.
4. ``assemble``: read A's figures with every disagreement, failed check and
   poorly scanned figure amber.

The provider is passed in (``providers.get_provider()`` in the API, the
recordings in tests), so this module never decides who is called.
"""
from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

from research.reader import assemble as asm
from research.reader import pages
from research.reader.providers import UNREADABLE, Answer, Document, ProviderError, sha256


class TooLong(ValueError):
    """More pages than a results filing has (``pages.MAX_PAGES``)."""


def _usage(answer: Answer) -> dict:
    """Token counts only: never the answer's text."""
    usage = answer.usage or {}
    keep = ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount")
    return {"model": answer.model, "prompt_version": answer.prompt_version, "seconds": answer.seconds,
            "tokens": {k: int(usage[k]) for k in keep if isinstance(usage.get(k), (int, float))}}


def _unwrap(answer: Answer) -> dict:
    try:
        return asm.unwrap(answer.read)
    except asm.Unreadable:
        raise ProviderError(UNREADABLE) from None


def read_filing(path: Path, provider, *, today: datetime.date, quarter: Optional[datetime.date] = None,
                stored: Sequence[Mapping] = (), info: Optional[pages.Info] = None,
                kept: Sequence[Mapping] = (), spent: Optional[list] = None) -> dict:
    """The checked statements of the PDF at ``path``. Each read is a timed
    section (``reader.read_a``, ``reader.read_b``), which is what moves a
    job's stage and progress (jobs/kinds.py). ``kept``: a covered company's
    stored periods (``db.coverage.company_figures``), compared with this
    filing's figures in the unit it prints (PLAN.md R2). ``spent`` gets each
    answer as it arrives, so a read that fails later is still counted
    (PLAN.md R9, ``db.ai_usage``)."""
    from research.coverage import stored_for_read

    from api.limits import holding_simulation_slot
    from api.observability import model_timer

    data = path.read_bytes()
    info = info or pages.info(path)
    if info.pages > pages.MAX_PAGES:
        raise TooLong(info.pages)
    digest = sha256(data)
    started = time.time()
    with model_timer("reader.read_a"):
        a = provider.read("A", Document(digest, pdf=data))
    if spent is not None:
        spent.append(a)
    read_a = _unwrap(a)
    stored = list(stored) + stored_for_read(kept, read_a["unit"])
    sent = asm.table_pages(read_a)
    if provider.needs_pictures:
        # a page number past the end of the file is a model's guess: never drawn
        sent = [p for p in sent if p <= info.pages]
    sent = sent[:pages.MAX_PICTURES]
    read_b, b = None, None
    if sent:
        with model_timer("reader.read_b"):
            pictures = []
            if provider.needs_pictures:
                # drawing pages is the read's one memory-heavy moment: it takes
                # the slot simulations and scans share (api/limits.py)
                with holding_simulation_slot():
                    pictures = pages.render(path, sent)
            b = provider.read("B", Document(digest, pictures=pictures), exclude=frozenset({a.model}))
        if spent is not None:
            spent.append(b)
        read_b = asm.picture_pages(_unwrap(b), sent)
    result = asm.assemble(read_a, read_b, today=today, info=info, quarter=quarter, stored=stored)
    if not sent:
        result["problems"].append({"statement": None, "column": None, "code": "no_tables", "detail": ""})
    result["document"] = {"pages": info.pages, "poor_pages": [p for p in range(1, info.pages + 1) if info.poor(p)],
                          "sha256": digest}
    result["reads"] = {"A": _usage(a), "B": _usage(b) if b else None,
                       "seconds": round(time.time() - started, 1)}
    return result
