"""The job drill (PLAN.md 1.9 "done when"): many simulations at once, on staging.

``POST /api/scheduled/drill`` queues ``count`` identical seeded Monte Carlo
jobs as the owner ``system:drill``; ``ops/job_drill.py`` (run by staging.yml
after each deploy) watches them while it polls ``/api/health``, and checks
that every job succeeded, that the health check kept answering, and that each
result's summary equals ``EXPECTED_SUMMARY`` -- the synchronous endpoint's
answer for the same seed, pinned here and re-checked against the live code by
``tests/test_jobs.py``. So "seeded results are unchanged" is checked on the
server that ran them, not only on a laptop.

Then ``POST /api/scheduled/drill/live`` (PLAN.md 5.8 "done when") runs a
session of the live sliders in the same process, which the ten simulations
have just stretched, and answers how long the estimates took and how much
memory the API used at its peak against what the service has
(``live_drill``; ops/scheduled.py ``live_problems`` judges the answer).

Refused in production: the drill is load on purpose.
"""
from __future__ import annotations

DRILL_OWNER = "system:drill"
MAX_DRILL_JOBS = 10
# The app's default deal and simulation size, with a fixed seed
DRILL_REQUEST = {"mc": {"n": 50000}, "seed": 20260918}
# What POST /api/montecarlo/run answers for DRILL_REQUEST (summary only).
# Compared at 1e-9 relative, like the golden snapshot: Linux numpy can differ
# in the last bits.
EXPECTED_SUMMARY: dict[str, float] = {
    "mean_irr": 0.17943005567295175,
    "median_irr": 0.18450687170702174,
    "p5_irr": 0.0269408373574059,
    "p95_irr": 0.31468050755760724,
    "p_above_hurdle": 0.42886,
    "wipeout_rate": 0.0,
    "hurdle": 0.2,
}
RELATIVE_TOLERANCE = 1e-9


def summary_mismatches(summary: dict) -> list[str]:
    """Keys of ``summary`` that differ from ``EXPECTED_SUMMARY``; empty when all match."""
    bad = []
    for key, want in EXPECTED_SUMMARY.items():
        got = summary.get(key)
        if not isinstance(got, (int, float)) or abs(got - want) > RELATIVE_TOLERANCE * max(abs(want), 1e-12):
            bad.append(key)
    return bad


# ---------------------------------------------------------------------------
# The live sliders on the server (PLAN.md 5.8 "done when")
# ---------------------------------------------------------------------------
# Render's free web service; a container that says less (cgroup) is believed
SERVICE_MEMORY_MB = 512.0
LIVE_MOVES = 40
# What POST /api/montecarlo/live answers for the default deal with no
# sliders (core.live.LIVE_SEED on LIVE_PATHS paths), re-checked against the
# live code by tests/test_jobs.py
EXPECTED_LIVE: dict[str, float] = {
    "irr_p5": 0.026343774265902644,
    "irr_p50": 0.18570548595366276,
    "irr_p95": 0.31428766513083767,
    "p_wipeout": 0.000125,
}
# Twelve facilities over fifteen years: the largest deal the API takes
_LOAN = {"kind": "institutional_term_loan", "amount": 30.0, "floating": True, "reference_level": 4.0,
         "margin": 3.5, "amort_pct": 1.0, "sweep": True, "maturity_years": 7}
_NOTES = {"kind": "senior_notes", "amount": 30.0, "floating": False, "fixed_rate": 9.0, "maturity_years": 8}
LARGEST_DEAL = {"hold": 15, "tax_interest_limit": "ebitda_share", "tax_loss_carryforward": True,
                "tranches": [{**_LOAN, "name": f"Loan {i}"} for i in range(6)]
                + [{**_NOTES, "name": f"Notes {i}"} for i in range(6)]}


def _megabytes(path: str) -> float | None:
    """A cgroup memory file's figure in MB; None when absent or unlimited."""
    try:
        with open(path, encoding="ascii") as f:
            text = f.read().strip()
    except OSError:
        return None
    return int(text) / 1e6 if text.isdigit() else None


def process_memory() -> dict[str, float | None]:
    """The API process's resident memory now and at its peak, and the
    container's limit (MB). Linux reads /proc and the cgroup; elsewhere the
    peak comes from getrusage and the rest is unknown."""
    now = peak = None
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            fields = dict(line.split(":", 1) for line in f if ":" in line)
        now = int(fields["VmRSS"].split()[0]) / 1000
        peak = int(fields["VmHWM"].split()[0]) / 1000
    except (OSError, KeyError, ValueError):
        import resource
        import sys
        used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = used / 1e6 if sys.platform == "darwin" else used / 1000      # bytes on macOS, KB elsewhere
    limit = _megabytes("/sys/fs/cgroup/memory.max") or _megabytes("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    return {"rss_mb": now, "peak_mb": peak,
            "limit_mb": min(limit, SERVICE_MEMORY_MB) if limit else SERVICE_MEMORY_MB}


def live_drill(moves: int = LIVE_MOVES, memory=None) -> dict:
    """A session of the live sliders in this process: the default deal as it
    stands, ``moves`` slider positions, and the largest deal the API takes.
    Answers what it measured, flat, for the workflow to judge
    (ops/scheduled.py ``live_problems``)."""
    import time
    from dataclasses import replace

    from core.config import resolve_config
    from core.deal import DealInputs
    from core.live import live_estimate, sliders_of
    from core.montecarlo import MCInputs

    memory = memory or process_memory
    before = memory()
    mc, deal, cfg = MCInputs(), DealInputs(), resolve_config({})
    own = live_estimate(mc, deal, cfg)
    start, timings, medians = sliders_of(mc, deal), [], set()
    for i in range(moves):
        moved = replace(start, growth_mean=start.growth_mean + 0.1 * (i + 1), debt_pct=start.debt_pct - 0.5 * i)
        t0 = time.perf_counter()
        medians.add(live_estimate(mc, deal, cfg, moved)["prediction"]["irr_p50"])
        timings.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    largest = live_estimate(MCInputs(hold=15), DealInputs(**LARGEST_DEAL), cfg)
    largest_ms = (time.perf_counter() - t0) * 1000
    after = memory()
    added = (after["rss_mb"] - before["rss_mb"]) if after["rss_mb"] is not None and before["rss_mb"] is not None else None
    return {
        "live_paths": own["paths"], "live_moves": moves, "live_distinct": len(medians),
        "live_mismatched": ",".join(k for k, want in EXPECTED_LIVE.items()
                                    if abs(own["prediction"][k] - want) > RELATIVE_TOLERANCE * max(abs(want), 1e-12)),
        "live_median_irr": own["prediction"]["irr_p50"],
        "live_typical_ms": round(sorted(timings)[len(timings) // 2], 2), "live_slowest_ms": round(max(timings), 2),
        "live_largest_paths": largest["paths"], "live_largest_ms": round(largest_ms, 2),
        "live_added_mb": None if added is None else round(added, 1),
        "memory_peak_mb": None if after["peak_mb"] is None else round(after["peak_mb"], 1),
        "memory_limit_mb": round(after["limit_mb"], 1),
    }


# ---------------------------------------------------------------------------
# Uploaded files (PLAN.md 6.1): one file's whole road through the real store
# ---------------------------------------------------------------------------
def drill_pdf(extra: bytes = b"") -> bytes:
    """A small well-formed PDF; ``extra`` becomes a fourth object."""
    bodies = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
              b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>"] + ([extra] if extra else [])
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    table = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(bodies) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(bodies) + 1, table)
    return bytes(out)


def scanner_drill() -> dict:
    """The scanner as this server really runs it: the child process, with its
    limits and the packages this image has, passes a plain PDF and refuses a
    scripted one. Needs no store, so it is checked wherever the API runs."""
    import tempfile
    from pathlib import Path

    from uploads.scan import scan_path

    with tempfile.TemporaryDirectory(prefix="fse-drill-") as folder:
        plain, scripted = Path(folder, "plain"), Path(folder, "scripted")
        plain.write_bytes(drill_pdf())
        scripted.write_bytes(drill_pdf(b"<< /S /JavaScript /JS (app.alert\\(1\\)) >>"))
        passed, refused = scan_path(plain, "pdf"), scan_path(scripted, "pdf")
    if not passed.clean:
        return {"files_scanner": f"failed:plain:{passed.reason}"}
    if refused.reason != "active_content":
        return {"files_scanner": f"failed:scripted:{'clean' if refused.clean else refused.reason}"}
    return {"files_scanner": "ok"}


def files_drill(fetch=None) -> dict:
    """Store a file, read it back, scan it in the child process, download it
    through a signed link and delete it, in the store this server is set up
    with. Flat, for the workflow to judge: ``files_scanner`` is ``ok`` or
    ``failed:...`` (``scanner_drill``); ``files_drill`` is ``ok``, ``none``
    (no store set up) or ``failed:<step>``. The object belongs to user 0,
    which is nobody, so no row and no account is touched."""
    import tempfile
    import time
    import urllib.request
    import uuid
    from pathlib import Path

    from uploads import rules, store as stores
    from uploads.scan import scan_path

    scanner = scanner_drill()
    files = stores.get_store()
    if files is None:
        return {**scanner, "files_drill": "none"}
    key, sample, step, t0 = stores.object_key(0, uuid.uuid4()), drill_pdf(), "start", time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fse-drill-") as folder:
        source, back = Path(folder, "source"), Path(folder, "back")
        source.write_bytes(sample)
        try:
            step = "bucket"
            if hasattr(files, "ensure_bucket"):
                files.ensure_bucket()
            step = "store"
            files.put(key, source, rules.KINDS["pdf"].content_type)
            step = "size"
            assert files.size(key) == len(sample)
            step = "fetch"
            assert files.fetch(key, back, rules.MAX_FILE_BYTES) == len(sample) and back.read_bytes() == sample
            step = "scan"
            assert scan_path(back, "pdf").clean
            step = "link"
            url = files.download_url(key, "drill.pdf", rules.LINK_S)
            if url.startswith("https://"):
                step = "download"
                with (fetch or urllib.request.urlopen)(url, timeout=30) as answer:
                    assert answer.read(len(sample) + 1) == sample
            step = "delete"
            files.delete([key])
            assert files.size(key) is None
        except Exception as exc:  # noqa: BLE001 - reported by its step and kind; nothing from the store is quoted
            try:
                files.delete([key])
            except Exception:  # noqa: BLE001
                pass
            # The store's own short reason, when it gave one (uploads/store.py
            # ``StorageError.detail``: the key is removed from it)
            return {**scanner, "files_drill": f"failed:{step}", "files_drill_error": type(exc).__name__,
                    "files_drill_detail": str(getattr(exc, "detail", ""))[:200], "files_drill_store": files.name,
                    # The sort of key the server holds (secret, publishable ...), never the key
                    "files_drill_key": str(getattr(files, "key_kind", "none"))}
    return {**scanner, "files_drill": "ok", "files_drill_store": files.name,
            "files_drill_ms": round((time.perf_counter() - t0) * 1000)}
