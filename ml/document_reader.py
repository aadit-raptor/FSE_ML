"""A document read for a deal, as the API runs it (PLAN.md 6.2): who may read what, the job, and the recorder.

``start`` (``POST /api/documents/reads``) checks what the results reader checks (``research/reader/service.py``):
the reader is configured, the card lets it read for people (``open_to_people``), the person **confirmed the
document is public** when the provider is a free tier, the file is theirs, a PDF and scanned clean, and the day's
reads are not used up (a document is two requests to the same allowance, so it counts as a read). Then it queues
job ``document.extract`` under the person's name.

``extract_file`` is the job: the file fetched to a private spool file, its pages counted in a child process and
charged against the person's **document pages** a day, then ``read_document``: read A the whole PDF, read B
pictures of the pages A cited (a different model), the cited pages' text layer to look each quote up in, and
``ml.nlp_extractor.assemble``. The answer is the job's result, kept six hours like every job's; **nothing reaches
a deal until the person accepts each value** (``POST /api/documents/reads/{id}/deal``, ``deal_inputs``). Every
answer the provider gives is counted in ``ai_usage`` (kind ``extract``, PLAN.md R9), a failed read too.

``python -m ml.document_reader record`` reads documents on a real key into ``tests/fixtures/extract/<set>/``,
which is what CI, the browser tests and the model card replay (``AI_PROVIDER=replay``). **Nothing from a
document reaches a log**: events carry counts, models and timings only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from api.limits import holding_simulation_slot
from api.observability import deploy_environment, log_event, model_timer
from ml import nlp_extractor as nx
from research.reader import gate, pages
from research.reader import providers as ai
from research.reader.read import usage_of
from research.reader.service import clean_pdf, count_usage

KIND = "document.extract"
MAX_PAGES = 500                 # the evaluation set's longest is 448 (L'Oréal's universal registration document)
EXTRACTIONS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "extract"
CARD = Path(__file__).resolve().parent / "cards" / "extractor.json"
INSTRUCTIONS = ai.Instructions(nx.PROMPT_VERSION, nx.prompt_text(), nx.prompt_text(pictures=True))
TEXT_PAGE = 40                  # characters below which a page has no text layer worth looking a quote up in

OFF, NOT_MEASURED, CONFIRM_PUBLIC, TOO_LONG = "reader_off", "reader_not_measured", "confirm_public", "too_long"
PAGE_LIMIT, STORE_DOWN, NO_FILE = "page_limit", "store_unreachable", "no_file"


def provider():
    """The configured provider, with the document reader's recordings for ``replay``; None when off."""
    found = ai.get_provider()
    if isinstance(found, ai.Replay):
        return ai.Replay(EXTRACTIONS, default=os.environ.get("FSE_EXTRACT_REPLAY_DEFAULT", ""))
    return found


def measured(path: Optional[Path] = None) -> dict:
    return gate.measured(CARD if path is None else path)


def open_to_people(environment: str, path: Optional[Path] = None) -> bool:
    """Production reads documents only while the card's headline set meets its bar (phase R rule 2, held here
    too); staging and local copies read anyway, the card's state shown."""
    return gate.open_to_people(environment, CARD if path is None else path)


def _reader():
    found = provider()
    if found is None:
        raise HTTPException(503, OFF)
    return found


def start(subject: str, file_id: uuid.UUID, *, public: bool):
    from api.limits import READS_ALL, USER_READS, enforce
    from api.routers.jobs import submit_kind

    reader = _reader()
    if not open_to_people(deploy_environment()):
        raise HTTPException(503, NOT_MEASURED)
    if reader.free_tier and not public:
        raise HTTPException(422, CONFIRM_PUBLIC)
    clean_pdf(subject, file_id)
    enforce([(USER_READS, subject), (READS_ALL, "all")])
    return submit_kind(subject, KIND, {"owner": subject, "file_id": str(file_id)})


def _charge_pages(subject: str, count: int) -> None:
    from api.limits import USER_DOCUMENT_PAGES, check_for
    from api.usage import counters

    check = check_for(USER_DOCUMENT_PAGES, subject, time.time())
    if counters().usage(check.key) + count > check.limit:
        raise HTTPException(429, PAGE_LIMIT)
    counters().add(check, count)


def pages_left(subject: str) -> int:
    from api.limits import USER_DOCUMENT_PAGES, check_for
    from api.usage import counters

    check = check_for(USER_DOCUMENT_PAGES, subject, time.time())
    return max(0, check.limit - counters().usage(check.key))


def _unwrap(answer: ai.Answer) -> dict:
    try:
        return nx.unwrap(answer.read)
    except nx.Unreadable:
        raise ai.ProviderError(ai.UNREADABLE) from None


def quote_lookup(texts: dict[int, str]):
    """``quote_on_page`` for ``assemble``: None for a page without a text layer (a scan), else whether the
    quote is on it."""
    def on_page(page: int, quote: str) -> Optional[bool]:
        text = texts.get(page) or ""
        if len(text.strip()) < TEXT_PAGE:
            return None
        return nx.quote_on_text(quote, text)
    return on_page


def read_document(path: Path, reader, *, info: pages.Info, spent: Optional[list] = None) -> dict:
    """The two reads of the PDF at ``path``, checked. ``spent`` gets each answer as it arrives (PLAN.md R9)."""
    data = path.read_bytes()
    digest = ai.sha256(data)
    started = time.time()
    with model_timer("document.read_a"):
        a = reader.read("A", ai.Document(digest, pdf=data), instructions=INSTRUCTIONS)
    if spent is not None:
        spent.append(a)
    read_a = _unwrap(a)
    sent = nx.cited_pages(read_a)
    if reader.needs_pictures:
        sent = [p for p in sent if p <= info.pages]       # a page past the end is a model's guess: never drawn
    read_b, b = None, None
    if sent:
        with model_timer("document.read_b"):
            pictures = []
            if reader.needs_pictures:
                with holding_simulation_slot():
                    pictures = pages.render(path, sent)
            b = reader.read("B", ai.Document(digest, pictures=pictures), exclude=frozenset({a.model}),
                            instructions=INSTRUCTIONS)
        if spent is not None:
            spent.append(b)
        # replayed or drawn, read B numbers what it was shown 1, 2 ...: the cited pages, in order
        read_b = nx.picture_pages(_unwrap(b), sent)
    cited = {p for p in nx.cited_pages(read_a) + nx.cited_pages(read_b or {"figures": [], "labels": {}})
             if p <= info.pages}
    texts = pages.texts(path, sorted(cited)) if cited else {}
    result = nx.assemble(read_a, read_b, quote_on_page=quote_lookup(texts), poor=info.poor)
    result["document"] = {"pages": info.pages, "sha256": digest,
                          "text_pages": sorted(p for p, t in texts.items() if len(t.strip()) >= TEXT_PAGE)}
    result["reads"] = {"A": usage_of(a), "B": usage_of(b) if b else None, "seconds": round(time.time() - started, 1)}
    return result


def extract_file(owner: str, file_id: uuid.UUID) -> dict:
    """The job: fetch, count and charge the pages, read twice, check."""
    from uploads import rules, service as uploads
    from uploads.store import NotStored, StorageError

    reader = _reader()
    record = clean_pdf(owner, file_id)
    path = uploads.spool() / f"document-{uuid.uuid4().hex}.pdf"
    t0 = time.perf_counter()
    charged, spent = 0, []
    try:
        try:
            uploads.store().fetch(record.key, path, rules.MAX_FILE_BYTES)
        except NotStored:
            raise HTTPException(404, NO_FILE) from None
        except StorageError:
            raise HTTPException(503, STORE_DOWN) from None
        try:
            with holding_simulation_slot():
                info = pages.info(path, MAX_PAGES)
        except pages.PagesError:
            raise HTTPException(422, "unreadable_pdf") from None
        if info.pages > MAX_PAGES:
            raise HTTPException(422, TOO_LONG)
        _charge_pages(owner, info.pages)
        charged = info.pages
        try:
            result = read_document(path, reader, info=info, spent=spent)
        except ai.ProviderError as exc:
            log_event("document_read_failed", logging.WARNING, provider=reader.name, code=exc.code,
                      http_status=exc.status)
            raise HTTPException({ai.SPENT: 429, ai.BUSY: 503, ai.NOT_RECORDED: 404}.get(exc.code, 502),
                                exc.code) from None
        except pages.PagesError:
            raise HTTPException(422, "unreadable_pdf") from None
    finally:
        uploads.discard(path)
        count_usage(owner, "extract", reader, pages=charged, answers=spent)
    log_event("document_read", provider=reader.name, pages=info.pages, figures=result["summary"]["figures"],
              amber=result["summary"]["amber"], model_a=result["reads"]["A"]["model"],
              model_b=(result["reads"]["B"] or {}).get("model"),
              duration_ms=round((time.perf_counter() - t0) * 1000))
    result["file_id"] = str(file_id)
    return result


# ---------------------------------------------------------------- the recorder

def plain_hash(quote: str) -> str:
    """How a recording keeps which quotes are on their page without keeping the page's text."""
    return hashlib.sha256(nx.plain(quote).encode("utf-8")).hexdigest()[:16]


def recorded_lookup(record: dict):
    """``quote_on_page`` from a recording: the text pages, and the quotes found on each (as hashes)."""
    found = {int(k): set(v) for k, v in (record.get("quotes") or {}).items()}
    text_pages = set(record["document"].get("text_pages") or [])

    def on_page(page: int, quote: str) -> Optional[bool]:
        if page not in text_pages:
            return None
        return plain_hash(quote) in found.get(page, set())
    return on_page


def recorded_reads(record: dict) -> tuple[dict, Optional[dict]]:
    a = nx.unwrap(record["reads"]["A"]["read"])
    b = record["reads"].get("B")
    if b is None:
        return a, None
    sent = [p for p in nx.cited_pages(a) if p <= record["document"]["pages"]]
    return a, nx.picture_pages(nx.unwrap(b["read"]), sent)


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _quotes(pdf: Path, record: dict) -> None:
    """Which quotes of both reads are on the page they cite, kept as hashes (no page text is committed)."""
    a, b = recorded_reads(record)
    items = [(f["page"], f["quote"]) for f in a["figures"]] + [(v["page"], v["quote"]) for v in a["labels"].values()]
    if b:
        items += [(f["page"], f["quote"]) for f in b["figures"]] + [(v["page"], v["quote"]) for v in b["labels"].values()]
    wanted = sorted({p for p, _ in items if p and p <= record["document"]["pages"]})
    texts = pages.texts(pdf, wanted) if wanted else {}
    record["document"]["text_pages"] = sorted(p for p, t in texts.items() if len(t.strip()) >= TEXT_PAGE)
    found: dict = {}
    for page, quote in items:
        if page in texts and quote and nx.quote_on_text(quote, texts[page]):
            found.setdefault(str(page), set()).add(plain_hash(quote))
    record["quotes"] = {k: sorted(v) for k, v in sorted(found.items(), key=lambda kv: int(kv[0]))}


def record_one(reader, pdf: Path, target: Path, set_name: str) -> str:
    """Read one document, saving after each answer. What happened, in words."""
    record = json.loads(target.read_text(encoding="utf-8")) if target.exists() else None
    if record is not None and record["prompt"] != nx.PROMPT_VERSION:
        return f"recorded with prompt {record['prompt']}, not {nx.PROMPT_VERSION}: left as it is"
    data = pdf.read_bytes()
    if record is None:
        info = pages.info(pdf, MAX_PAGES)
        record = {"document": {"name": target.stem, "sha256": ai.sha256(data), "pages": info.pages,
                               "dpi": list(info.dpi), "set": set_name},
                  "prompt": nx.PROMPT_VERSION, "reads": {}}
    reads = record["reads"]
    for which in ("A", "B"):
        if which in reads:
            continue
        if which == "A":
            document, exclude = ai.Document(record["document"]["sha256"], pdf=data), frozenset()
        else:
            sent = [p for p in nx.cited_pages(nx.unwrap(reads["A"]["read"])) if p <= record["document"]["pages"]]
            if not sent:
                return "read A cited no pages"
            document = ai.Document(record["document"]["sha256"], pictures=pages.render(pdf, sent))
            exclude = frozenset({reads["A"]["model"]})
        answer = reader.read(which, document, exclude=exclude, instructions=INSTRUCTIONS)
        reads[which] = {"model": answer.model, "read": answer.read, "usage": dict(answer.usage),
                        "seconds": answer.seconds}
        _write(target, record)
    _quotes(pdf, record)
    _write(target, record)
    return f"A by {reads['A']['model']}, B by {reads['B']['model']}"


def main(argv=None) -> int:
    from research.reader.record import api_key

    parser = argparse.ArgumentParser(prog="python -m ml.document_reader")
    sub = parser.add_subparsers(dest="command", required=True)
    live = sub.add_parser("record")
    live.add_argument("--set", default="evaluation")
    live.add_argument("--env-file", type=Path)
    live.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    reader = ai.Gemini(api_key(args.env_file))
    for pdf in args.files:
        target = EXTRACTIONS / args.set / f"{pdf.stem}.json"
        try:
            print(f"{pdf.name}: {record_one(reader, pdf, target, args.set)}", flush=True)
        except ai.ProviderError as exc:
            print(f"{pdf.name}: {exc}", flush=True)
            if exc.code == ai.SPENT:
                print("Every model's allowance for today is spent: run the same command again tomorrow.")
                return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
