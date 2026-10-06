"""Record real responses from the filing sources as test fixtures.

    python -m companies.record --out tests/fixtures/companies sec_mcd esef_tesco
    python -m companies.record --list

Each case runs the same calls the app makes (searches and loads) through a
recording transport, then trims every response to what the code reads, so
a fixture is kilobytes, not the megabytes of a whole filing:

- company facts and xBRL-JSON keep only the concepts the summary maps name;
- name lists (the SEC's tickers, EDINET's code list) keep only the rows a
  case's searches can match;
- EDINET's day lists keep the annual reports; its CSV archives keep the
  mapped elements and the DEI labels.

Keys are never recorded: requests are named without their secret
parameters, and the run fails if a key's value appears in anything written.
Companies House and EDINET need their keys in the environment; the
``record-filings`` workflow runs those cases with the repository secrets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from companies import companies_house, edinet, http, sec, sources
from companies.items import mapped_concepts

INDEX = "index.json"




# ---------------------------------------------------------------------------
# Replay (tests) and recording
# ---------------------------------------------------------------------------
def replay(directory: Path) -> http.Transport:
    """A transport answering from a recorded fixture directory; a request
    that wasn't recorded is a 404, as the source would say."""
    index = json.loads((directory / INDEX).read_text(encoding="utf-8"))

    def transport(req: http.Request) -> http.Response:
        entry = index.get(req.key())
        if entry is None:
            return http.Response(404, b"")
        return http.Response(entry["status"], (directory / entry["file"]).read_bytes(), entry.get("type", ""))
    return transport


@dataclass
class Recorder:
    real: http.Transport
    paced: bool = True                                 # calls go to the real sources
    responses: dict = field(default_factory=dict)      # key -> (Request, Response)

    def __call__(self, req: http.Request) -> http.Response:
        resp = self.real(req)
        self.responses[req.key()] = (req, resp)
        return resp


def _name(key: str, content_type: str) -> str:
    """``<host>_<last path part>_<hash>.<ext>``: readable and unique."""
    host, _, path = key.split("://", 1)[-1].partition("/")
    last = path.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    stem = re.sub(r"[^A-Za-z0-9]+", "_", f"{host.split('.')[-3] if host.count('.') >= 2 else host}_{last}")[:60]
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    ext = ".zip" if "zip" in content_type or key.endswith(".zip") else (
        ".json" if "json" in content_type else ".xhtml" if "html" in content_type else ".bin")
    return f"{stem}_{digest}{ext}"


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------
def trim(req: http.Request, content: bytes, needles: list[str]) -> bytes:
    url = req.url
    if url == "https://www.sec.gov/files/company_tickers.json":
        rows = json.loads(content)
        keep = {k: r for k, r in rows.items() if _matches(r.get("title", ""), r.get("ticker", ""), needles)}
        return json.dumps(keep).encode()
    if "/api/xbrl/companyfacts/" in url:
        data = json.loads(content)
        wanted = mapped_concepts()
        facts = {}
        for ns, concepts in data.get("facts", {}).items():
            kept = {}
            for name, body in concepts.items():
                if f"{ns}:{name}" not in wanted:
                    continue
                units = {u: [r for r in rows if r.get("fp") == "FY" and r.get("form") in
                             ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A")]
                         for u, rows in body.get("units", {}).items()}
                kept[name] = {"units": {u: r for u, r in units.items() if r}}
            facts[ns] = kept
        return json.dumps({"cik": data.get("cik"), "entityName": data.get("entityName"), "facts": facts}).encode()
    if url.startswith("https://data.sec.gov/submissions/"):
        data = json.loads(content)
        return json.dumps({k: data.get(k) for k in ("cik", "name", "tickers", "addresses", "fiscalYearEnd")}).encode()
    if url.startswith("https://filings.xbrl.org/") and url.endswith(".json") and "/api/" not in url:
        doc = json.loads(content)
        prefixes = doc.get("documentInfo", {}).get("namespaces", {})
        wanted = {c.split(":", 1)[1] for c in mapped_concepts() if c.startswith("ifrs-full:")}
        facts = {k: f for k, f in doc.get("facts", {}).items()
                 if (f.get("dimensions") or {}).get("concept", "").partition(":")[2] in wanted
                 and prefixes.get((f.get("dimensions") or {}).get("concept", "").partition(":")[0], "")
                 .endswith("ifrs-full")}
        return json.dumps({"documentInfo": doc.get("documentInfo"), "facts": facts}).encode()
    if url == edinet.CODE_LIST_URL:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            name = z.namelist()[0]
            lines = z.read(name).decode("cp932").splitlines()
        kept = lines[:2] + [ln for ln in lines[2:] if _code_row_matches(ln, needles)]
        return _zip({name: ("\r\n".join(kept) + "\r\n").encode("cp932")})
    if url.endswith("/documents.json"):
        data = json.loads(content)
        data["results"] = [r for r in data.get("results") or [] if r.get("docTypeCode") in ("120", "130")
                           and (not needles or r.get("edinetCode") in needles)]
        return json.dumps(data, ensure_ascii=False).encode()
    if url.startswith(edinet.API + "/documents/"):
        return _trim_edinet_csv(content)
    if url.startswith("https://document-api.company-information.service.gov.uk/") and url.endswith("/content"):
        return _trim_ixbrl(content)
    return content


def _matches(name: str, ticker: str, needles: list[str]) -> bool:
    plain = re.sub(r"[^A-Z0-9]", "", name.upper())
    for n in needles:
        up = n.upper()
        if ticker.upper() == up or up in name.upper() or re.sub(r"[^A-Z0-9]", "", up) == plain:
            return True
    return False


def _code_row_matches(line: str, needles: list[str]) -> bool:
    row = next(csv.reader([line]), [])
    if len(row) < 13:
        return False
    plain = edinet._plain
    for n in needles:
        q = plain(n)
        if q in (row[0], row[12], row[11][:4]) or q in plain(row[6]) or q in plain(row[7]):
            return True
    return False


def _zip(files: dict[str, bytes]) -> bytes:
    """A zip that is the same bytes wherever it is made: fixed timestamps, no
    compression (zlib builds differ) and the same "made on" system."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for name, data in files.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 0
            info.external_attr = 0
            z.writestr(info, data)
    return buf.getvalue()


def _trim_ixbrl(content: bytes) -> bytes:
    """An inline XBRL document cut down to its contexts, units and the
    figures the maps read, each in its own paragraph, with the original
    namespace declarations so every concept still resolves."""
    import xml.etree.ElementTree as ET

    from companies.items import canonical_prefix
    from companies.ixbrl import IX_NAMESPACES, XBRLI
    namespaces: dict[str, str] = {}
    for _, (prefix, uri) in ET.iterparse(io.BytesIO(content), events=("start-ns",)):
        namespaces.setdefault(prefix, uri)
    for prefix, uri in namespaces.items():
        ET.register_namespace(prefix, uri)
    root = ET.fromstring(content)
    wanted = mapped_concepts()
    canon = {p: canonical_prefix(u) for p, u in namespaces.items()}

    def keep(elem) -> bool:
        prefix, _, local = elem.get("name", "").partition(":")
        return f"{canon.get(prefix)}:{local}" in wanted

    resources = [e for tag in ("context", "unit") for e in root.iter(f"{{{XBRLI}}}{tag}")]
    figures = [e for ns in IX_NAMESPACES for e in root.iter(f"{{{ns}}}nonFraction") if keep(e)]
    for e in resources + figures:
        e.tail = None
    ix = next(ns for ns in IX_NAMESPACES if ns in namespaces.values())
    ix_prefix = next(p for p, u in namespaces.items() if u == ix)
    declarations = " ".join(f'xmlns{":" + p if p else ""}="{u}"' for p, u in namespaces.items())
    parts = [f'<?xml version="1.0" encoding="UTF-8"?>\n<html {declarations}><body>'
             f"<{ix_prefix}:header><{ix_prefix}:resources>"]
    parts += [ET.tostring(e, encoding="unicode") for e in resources]
    parts.append(f"</{ix_prefix}:resources></{ix_prefix}:header>")
    parts += [f"<p>{ET.tostring(e, encoding='unicode')}</p>" for e in figures]
    parts.append("</body></html>\n")
    return "\n".join(parts).encode("utf-8")


def _trim_edinet_csv(content: bytes) -> bytes:
    wanted = mapped_concepts()
    out = {}
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in z.namelist():
            if not re.search(r"XBRL_TO_CSV/jpcrp[^/]*\.csv$", name):
                continue
            rows = list(csv.reader(io.StringIO(z.read(name).decode("utf-16")), delimiter="\t"))
            kept = rows[:1] + [r for r in rows[1:] if len(r) >= 9 and (
                r[0].startswith("jpdei_cor:") or
                f"{edinet.canonical(r[0].partition(':')[0])}:{r[0].partition(':')[2]}" in wanted)]
            buf = io.StringIO()
            csv.writer(buf, delimiter="\t", lineterminator="\r\n").writerows(kept)
            out[name] = buf.getvalue().encode("utf-16")
    return _zip(out)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    name: str
    run: Callable[[], None]
    needles: tuple = ()
    needs: tuple = ()        # environment variables


def _edinet_case(code: str, reports: int = 2) -> Callable[[], None]:
    """Find the company's latest annual reports by walking back through the
    day lists (not recorded), index them, then load the company."""
    def run():
        recorder = http._transport[0]
        http.use_transport(recorder.real, paced=recorder.paced)
        try:
            found, day = 0, date.today()
            while found < reports and day > date.today() - timedelta(days=800):
                day -= timedelta(days=1)
                if day.weekday() >= 5:          # EDINET takes filings on working days
                    continue
                before = len(edinet.index().reports(code))
                try:
                    edinet.scan_day(day)
                except http.NotFound:
                    continue
                if len(edinet.index().reports(code)) > before:
                    found += 1
                    _found_days.append(day)
        finally:
            http.use_transport(recorder, paced=recorder.paced)
        for d in _found_days:
            edinet.scan_day(d)               # recorded this time
        edinet.search(code)
        sources.fetch("edinet", code)
    return run


_found_days: list = []

CASES = {c.name: c for c in (
    Case("sec_mcd", lambda: (sources.search("MCD"), sources.fetch("sec", "63908")), ("MCD", "MCDONALDS")),
    Case("esef_tesco", lambda: (sources.search("GB00BLGZ9862"), sources.fetch("esef", "2138002P5RNKC5W2JZ46")),
         ("TESCO",)),
    Case("esef_heineken", lambda: (sources.search("Heineken"), sources.fetch("esef", "724500K5PTPSST86UQ23")),
         ("HEINEKEN",)),
    Case("search_names", lambda: (sources.search("Toyota"), sources.search("2138002P5RNKC5W2JZ46")),
         ("TOYOTA", "TESCO")),
    Case("ch_cambridge_united", lambda: (sources.search("00482197"), sources.fetch("companies_house", "00482197")),
         ("00482197",), (companies_house.KEY_ENV,)),
    Case("edinet_toyota", _edinet_case("E02144"), ("E02144",), (edinet.KEY_ENV,)),
    Case("edinet_nintendo", _edinet_case("E02367"), ("E02367",), (edinet.KEY_ENV,)),
)}


def record(case: Case, out: Path, real: http.Transport = http._requests_transport,
           raw: Path | None = None) -> Path:
    """Run ``case`` and write its trimmed responses to ``out/<case>``; with
    ``raw``, the untrimmed ones too (to read a filing's concepts when a map
    is missing some -- never committed)."""
    directory = out / case.name
    directory.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(real, paced=real is http._requests_transport)
    http.use_transport(recorder, paced=recorder.paced)
    edinet.use_index(None)
    sec.reset_cache()
    edinet.reset_cache()
    _found_days.clear()
    try:
        case.run()
    finally:
        http.use_transport(None)
    secrets = [os.environ[k].encode() for k in (companies_house.KEY_ENV, edinet.KEY_ENV) if os.environ.get(k)]
    files, index = {}, {}
    for key, (req, resp) in recorder.responses.items():
        body = trim(req, resp.content, list(case.needles)) if resp.status == 200 else resp.content
        if any(s in body or s in resp.content or s in key.encode() for s in secrets):
            raise SystemExit(f"{case.name}: a key appeared in a recorded response; nothing written")
        name = _name(key, resp.content_type or "")
        files[name] = (body, resp.content)
        index[key] = {"file": name, "status": resp.status, "type": resp.content_type}
    for name, (body, original) in files.items():
        (directory / name).write_bytes(body)
        if raw is not None:
            (raw / case.name).mkdir(parents=True, exist_ok=True)
            (raw / case.name / name).write_bytes(original)
    (directory / INDEX).write_text(json.dumps(index, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return directory


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--out", default="tests/fixtures/companies")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--raw", help="also write the untrimmed responses here (not for committing)")
    args = ap.parse_args(argv)
    if args.list or not args.cases:
        for c in CASES.values():
            print(c.name, "(needs " + ", ".join(c.needs) + ")" if c.needs else "")
        return 0
    for name in args.cases:
        case = CASES[name]
        missing = [k for k in case.needs if not os.environ.get(k)]
        if missing:
            print(f"{name}: skipped, set {', '.join(missing)}", file=sys.stderr)
            continue
        print(f"{name}: {record(case, Path(args.out), raw=Path(args.raw) if args.raw else None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
