"""The scheduled refresh of economic data (``economy-refresh``, nightly).

Run by GitHub Actions through the API (jobs/scheduled.py, scheduled.yml),
on both environments, with a database only. Each run reads every source once
(about a dozen paced calls, connectors.py), stores the series, then adds the
ECB's exchange rates published since the last stored day (a week back, for
revisions; ``KEEP_FX_DAYS`` on the first run) and drops the oldest.

A source that fails leaves its stored series as they were, and their age
shows (economy.views); a source answering a shape its connector doesn't
know counts as ``unreadable``. The run itself fails, so it alerts, once
everything it read is stored, when the FRED key is refused, or when fewer than ``MIN_CURRENT_ECONOMIES`` economies have
current growth, inflation and a policy rate after it (PLAN.md 4.2's "done
when": at least 20).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from api.observability import log_event
from companies import http
from economy import connectors, views
from economy.catalogue import COUNTRIES

MIN_CURRENT_ECONOMIES = 20
FX_REVISION_DAYS = 7


class NotCurrent(RuntimeError):
    """Too few economies are current after a refresh."""


class KeyRefused(RuntimeError):
    """FRED refused the key: raised after everything else read was stored."""


def _read(today) -> tuple[list, dict]:
    calls = (("imf", lambda: connectors.imf(today)), ("worldbank", connectors.worldbank),
             ("bis", connectors.bis), ("oecd", connectors.oecd), ("ecb", connectors.ecb),
             ("fred", connectors.fred))
    found, problems = [], {}
    for name, call in calls:
        try:
            found += call()
        except http.NotConfigured:
            problems[name] = "not_configured"
        except http.SourceError as exc:
            log_event("economy_source_failed", logging.WARNING, source=name, reason=exc.reason)
            problems[name] = exc.reason
        except (AttributeError, TypeError, KeyError, ValueError, IndexError):
            # Valid JSON or CSV of a shape the connector doesn't know: this
            # source is unreadable today, the others still count
            log_event("economy_source_failed", logging.WARNING, source=name, reason="unreadable")
            problems[name] = "unreadable"
    return found, problems


def run(now: Optional[datetime] = None) -> dict:
    from db import economy as store
    from db.engine import is_configured
    if not is_configured():
        return {"skipped_no_database": True}
    now = now or datetime.now(timezone.utc)
    today = now.date()
    found, problems = _read(today)
    summary: dict = {"series_saved": store.save_series(found), "source_problems": problems}

    latest = store.latest_fx_day()
    since = latest - timedelta(days=FX_REVISION_DAYS) if latest else today - timedelta(days=store.KEEP_FX_DAYS)
    try:
        summary["fx_days_saved"] = store.save_fx(connectors.ecb_fx(since))
    except http.SourceError as exc:
        log_event("economy_source_failed", logging.WARNING, source="ecb_fx", reason=exc.reason)
        problems["ecb_fx"] = exc.reason
    summary["fx_days_pruned"] = store.prune_fx(today)

    series, _ = store.all_series()
    current = views.current_economies(series, today)
    fx = store.fx_on()
    use = store.usage()
    summary.update(
        economies_current=len(current), economies_not_current=sorted(set(COUNTRIES) - set(current)),
        reference_rates=sorted(views.reference_rates(series, today)),
        fx_date=fx.day.isoformat() if fx else None, fx_currencies=len(fx.rates) if fx else 0,
        economy_bytes=use["bytes"], economy_budget_warning=use["warning"])
    log_event("economy_refreshed", **{k: v for k, v in summary.items() if isinstance(v, (int, str, bool))})
    if problems.get("fred") == "refused":              # the key: fail the run, so it alerts
        raise KeyRefused("FRED refused the key (FRED_API_KEY); every other source was stored")
    if len(current) < MIN_CURRENT_ECONOMIES:
        raise NotCurrent(f"{len(current)} economies current, fewer than {MIN_CURRENT_ECONOMIES}")
    return summary
