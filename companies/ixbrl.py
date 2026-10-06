"""Inline XBRL (iXBRL) numbers out of an XHTML accounts document.

Companies House accounts filed electronically are inline XBRL: an XHTML page
whose figures are wrapped in ``ix:nonFraction`` tags naming the concept,
the context (period, and any dimension) and the unit. This reads those
tags only -- nothing else on the page -- into ``Fact``s:

- the context's period gives the dates; a context with a ``segment`` or
  ``scenario`` (a dimension) is not the entity's total and is skipped;
- the value is the tag's text read with its ``format`` (``num-dot-decimal``,
  ``num-comma-decimal``, ``zerodash`` ...), times 10^``scale``, negated when
  ``sign="-"``;
- concepts get canonical prefixes (``companies.items.canonical_prefix``).

Parsed with the standard library's expat-based parser, which never fetches
external entities and refuses entity-expansion bombs (expat 2.4+).
"""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from datetime import date
from typing import Optional

from companies.facts import Fact
from companies.items import canonical_prefix
from companies.model import FilingLink

IX_NAMESPACES = ("http://www.xbrl.org/2013/inlineXBRL", "http://www.xbrl.org/2008/inlineXBRL")
XBRLI = "http://www.xbrl.org/2003/instance"
XBRLDI_SEGMENT = (f"{{{XBRLI}}}segment", f"{{{XBRLI}}}scenario")


class Unreadable(ValueError):
    pass


# A filing is someone else's document: refuse what no real accounts need
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_SCALE = 12              # 10^12: trillions; a scale beyond is not a real figure


def _text(elem: ET.Element) -> str:
    return "".join(elem.itertext()).strip()


def parse_number(text: str, fmt: str) -> Optional[float]:
    """A displayed number read the way its ``format`` says."""
    name = fmt.rsplit(":", 1)[-1].replace("-", "").lower()
    if name in ("zerodash", "fixedzero", "nocontent") or text in ("-", "–", "—", ""):
        return 0.0
    if name in ("numcommadecimal", "numdotcomma", "numspacecomma"):
        text = text.replace(".", "").replace(" ", "").replace(" ", "").replace(",", ".")
    else:
        text = text.replace(",", "").replace(" ", "").replace(" ", "")
    text = text.strip("()")
    if not re.fullmatch(r"\d+(\.\d+)?", text):
        return None
    return float(text)


def _contexts(root: ET.Element) -> dict[str, tuple[Optional[date], date]]:
    """Context id -> (start or None, end), for contexts without dimensions."""
    out = {}
    for ctx in root.iter(f"{{{XBRLI}}}context"):
        if any(ctx.find(f".//{tag}") is not None for tag in XBRLDI_SEGMENT):
            continue
        period = ctx.find(f"{{{XBRLI}}}period")
        if period is None:
            continue
        instant = period.findtext(f"{{{XBRLI}}}instant")
        start = period.findtext(f"{{{XBRLI}}}startDate")
        end = period.findtext(f"{{{XBRLI}}}endDate")
        try:
            if instant:
                out[ctx.get("id")] = (None, date.fromisoformat(instant.strip()[:10]))
            elif start and end:
                out[ctx.get("id")] = (date.fromisoformat(start.strip()[:10]), date.fromisoformat(end.strip()[:10]))
        except ValueError:
            continue
    return out


def _units(root: ET.Element) -> dict[str, str]:
    """Unit id -> ISO currency code, for currency units."""
    out = {}
    for unit in root.iter(f"{{{XBRLI}}}unit"):
        measures = [m.text or "" for m in unit.iter(f"{{{XBRLI}}}measure")]
        if len(measures) == 1 and measures[0].strip().split(":")[-1].isupper() \
                and "iso4217" in measures[0].lower():
            out[unit.get("id")] = measures[0].strip().split(":")[-1]
    return out


def read(content: bytes, link: FilingLink) -> tuple[list[Fact], set[str]]:
    """(money facts, every dimension member the contexts name). The members
    say what the accounts are, such as the standard they follow (``FRS102``).
    Text facts are never read: they include officers' names."""
    if len(content) > MAX_DOCUMENT_BYTES:
        raise Unreadable("larger than any accounts this reads")
    namespaces: dict[str, str] = {}
    try:
        parser = ET.iterparse(io.BytesIO(content), events=("start-ns",))
        for _event, (prefix, uri) in parser:
            namespaces.setdefault(prefix, uri)
        root = parser.root                  # parsed once
    except ET.ParseError:
        raise Unreadable("not well-formed XHTML") from None
    canon = {p: canonical_prefix(uri) for p, uri in namespaces.items()}
    contexts, units = _contexts(root), _units(root)
    members = {(_text(m).rsplit(":", 1)[-1]) for m in root.iter("{http://xbrl.org/2006/xbrldi}explicitMember")}
    facts = []
    for ns in IX_NAMESPACES:
        for elem in root.iter(f"{{{ns}}}nonFraction"):
            concept = _concept(elem.get("name", ""), canon)
            period = contexts.get(elem.get("contextRef", ""))
            currency = units.get(elem.get("unitRef", ""))
            if concept is None or period is None or currency is None:
                continue
            if elem.get("{http://www.w3.org/2001/XMLSchema-instance}nil") == "true":
                continue
            value = parse_number(_text(elem), elem.get("format", ""))
            if value is None:
                continue
            scale = _scale(elem.get("scale", "0"))
            if scale is None:
                continue
            value *= 10.0 ** scale
            if elem.get("sign") == "-":
                value = -value
            facts.append(Fact(concept, value, currency, period[1], period[0], link))
    return facts, members


def _scale(text: str) -> Optional[int]:
    """A fact's power of ten, or None when it is not a believable one."""
    try:
        scale = int((text or "0").strip())
    except ValueError:
        return None
    return scale if -MAX_SCALE <= scale <= MAX_SCALE else None


def _concept(qname: str, canon: dict) -> Optional[str]:
    prefix, _, name = qname.partition(":")
    canonical = canon.get(prefix)
    return f"{canonical}:{name}" if canonical and name else None
