"""The scanner: is this file what it says it is, and can it do anything?

**Why not an antivirus.** ClamAV, the free signature scanner, loads 3.6
million signatures: scanning one 60-byte PDF peaked at 1,687 MB (measured
2026-10-11, ClamAV 1.5.4), and the free server has 512 MB. So this is a
**structural scanner**: it does not look for known malware, it refuses every
document that carries something able to *act* -- which is how a document
does harm in the first place -- and every document it cannot see into.

**Parsers, not patterns.** A file is read the way the program that opens it
reads it, because a check that reads it differently can be told one thing
while the program is told another:

- **PDF**: pypdf (open source, BSD) resolves the cross-reference table,
  filters, predictors and object streams, and every object it finds is
  walked for the keys and names that make a PDF active. Beside it, the raw
  bytes are searched for the same names however they are spelt (pdfid's
  method), and anything that inflates to an object stream is searched too:
  two readers, and either one's finding refuses the file.
- **Office Open XML** (.xlsx, .docx): the package's relationships, content
  types and Word's field instructions are parsed with expat, so an attribute
  written with character references, in UTF-16 or with a ``>`` inside it is
  read as Office reads it. A part with a document type declaration or that
  is not well-formed is refused. On top: parts that are OLE objects or
  nested packages, ActiveX, macro sheets, archives that expand out of
  proportion; and oletools' verdict on macros.
- **Older Office files** (.xls, .doc): oletools (open source, BSD) for VBA
  and Excel 4 macros, DDE and encryption, plus the container's own streams.

Refused: JavaScript, launch actions, embedded files, rich media, form
submission, XFA forms, encryption (PDF); macros, embedded objects, DDE,
resources fetched from elsewhere, imported chunks, encryption (Office); and
anything that is not the kind its name says.

**In a process of its own.** ``scan_path`` runs the scan in a child process
with a CPU limit, a memory limit and a wall-clock deadline, and with none of
this server's secrets in its environment. A file built to make a parser spin
or swell kills that process, not the API, and is refused.

**What it is not.** It can't tell a harmless macro from a harmful one (both
are refused), and it does not detect an exploit of a reader's own bugs in a
file with no active content. Two readers can still disagree about a file
built for it; files are private to the person who uploaded them, so the
reader at risk is that person and this server's own parser (PLAN.md 6.2),
which must treat every file as hostile regardless.
``docs/security/threat-model.md`` records this; a signature scan needs a
worker with memory (PLAN.md 12.1).

``scan`` never raises and never logs anything from a file: an unexpected
failure is the verdict ``scan_failed``, and the file is refused.
"""
from __future__ import annotations

import io
import logging
import os
import re
import struct
import subprocess
import sys
import warnings
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional
from xml.parsers import expat

from uploads.rules import KINDS, MAX_FILE_BYTES, MB, REASONS

# An object stream inflates to at most this, and all of a file's to this
OBJECT_STREAM_MAX = 16 * MB
OBJECT_STREAMS_TOTAL = 96 * MB
PDF_MAX_OBJECTS = 500_000
# Places in a PDF where a stream may begin that are looked at for an object stream
PDF_MAX_STREAMS = 200_000
# An Office file's parts together; one XML part read through; and all the XML
# of a file that is parsed (every element costs a callback: 16 MB of empty
# elements is 4 seconds on a laptop)
ARCHIVE_MAX_BYTES = 512 * MB
ARCHIVE_MAX_MEMBERS = 10_000
PART_MAX_BYTES = 64 * MB
XML_BUDGET_BYTES = 96 * MB
NESTED_MAX_BYTES = MAX_FILE_BYTES
# The child process: CPU seconds, address space (Linux), and the deadline. A
# scan holds the slot simulations share, so a file built to keep the scanner
# busy must not hold it long: a 20 MB PDF takes under a second on a laptop.
# The price is that a very large file of the older formats, or a workbook
# with hundreds of megabytes of sheets, can run out of time on the free
# server's fraction of a processor and is refused as ``scan_failed``
SCAN_CPU_S = 45
SCAN_MEMORY_BYTES = 700 * MB
SCAN_TIMEOUT_S = 60.0

OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
ZIP_MAGIC = b"PK\x03\x04"


@dataclass(frozen=True)
class Verdict:
    clean: bool
    reason: Optional[str] = None


CLEAN = Verdict(True)
FAILED = Verdict(False, "scan_failed")


class Refused(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def scan(data: bytes, kind: str) -> Verdict:
    """Check ``data`` as a file of ``kind`` (a key of ``rules.KINDS``), in
    this process. The API itself goes through ``scan_path``."""
    try:
        if kind not in KINDS:
            raise Refused("wrong_type")
        if not data:
            raise Refused("empty")
        if len(data) > MAX_FILE_BYTES:
            raise Refused("too_large")
        _CHECKS[kind](data)
        return CLEAN
    except Refused as refused:
        return Verdict(False, refused.reason)
    except BaseException as exc:  # noqa: BLE001 - a scanner that fails refuses; the message may quote the file
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return FAILED


def scan_path(path: Path, kind: str, *, timeout: Optional[float] = None) -> Verdict:
    """Scan the file at ``path`` in a child process, and wait at most
    ``timeout`` seconds for it. A child that dies, overruns or answers
    anything unexpected is a file refused."""
    root = Path(__file__).resolve().parents[1]
    # Nothing of this server's goes with it: no database address, no keys
    environment = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(root),
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        done = subprocess.run([sys.executable, "-m", "uploads.scan", kind, str(path)],
                              stdin=subprocess.DEVNULL, capture_output=True, cwd=root, env=environment,
                              timeout=SCAN_TIMEOUT_S if timeout is None else timeout, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return FAILED
    word = done.stdout.decode("ascii", "replace").strip()
    if done.returncode == 0 and word == "clean":
        return CLEAN
    if done.returncode == 3 and word in REASONS:
        return Verdict(False, word)
    return FAILED


def _limit_this_process() -> None:
    import resource

    limits = [(resource.RLIMIT_CPU, SCAN_CPU_S), (resource.RLIMIT_CORE, 0)]
    if sys.platform.startswith("linux"):       # macOS does not enforce an address-space limit
        limits.append((resource.RLIMIT_AS, SCAN_MEMORY_BYTES))
    for which, value in limits:
        try:
            resource.setrlimit(which, (value, value))
        except (ValueError, OSError):
            pass


def main(argv: list[str]) -> int:
    """The child: ``python -m uploads.scan <kind> <path>`` prints ``clean`` or
    the reason, and nothing else."""
    _limit_this_process()
    if len(argv) != 2:
        return 2
    try:
        with open(argv[1], "rb") as source:
            data = source.read(MAX_FILE_BYTES + 1)
    except OSError:
        print("scan_failed")
        return 3
    verdict = scan(data, argv[0])
    print("clean" if verdict.clean else verdict.reason)
    return 0 if verdict.clean else 3


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
_PDF_DELIMITER = rb"(?=[\x00\t\n\x0c\r /()<>\[\]{}%]|\Z)"


def _pdf_name(word: str) -> bytes:
    """A regex for the name ``/word`` however it is spelt: any character may
    be written ``#xx`` (``/J#53`` is ``/JS``)."""
    parts = [b"(?:%s|#%02x)" % (re.escape(ch.encode()), ord(ch)) for ch in word]
    return b"/" + b"".join(parts)


def _pdf_names(words: tuple[str, ...]) -> "re.Pattern[bytes]":
    # Longest first, so /EmbeddedFiles is not read as /EmbeddedFile + "s"
    ordered = sorted(words, key=len, reverse=True)
    body = b"|".join(b"(?P<%s>%s)" % (w.encode(), _pdf_name(w)) for w in ordered)
    return re.compile(b"(?:" + body + b")" + _PDF_DELIMITER, re.IGNORECASE)


# What makes a PDF active. In the raw bytes only the names long enough that
# compressed data never spells them by chance are looked for (20 MB of it
# spells "/JS(" in about one file in two hundred); the short ones are found
# by the parser, and in the text of object streams
_ACTIVE_LONG = ("JavaScript", "Launch", "EmbeddedFile", "EmbeddedFiles", "RichMedia", "Rendition",
                "SubmitForm", "ImportData", "GoToE")
_ACTIVE_SHORT = ("JS", "XFA")
_RAW = _pdf_names(_ACTIVE_LONG + ("Encrypt",))
_TEXT = _pdf_names(_ACTIVE_LONG + _ACTIVE_SHORT + ("Encrypt",))
# Read by the parser only (ordinary words, or too short for the raw bytes):
# files attached to the document, portfolios, links that open another file
# (a network path gives away the reader's Windows sign-in), and media and 3D
# content, whose own readers have been the way in before
_ACTIVE_PARSED = ("EF", "RF", "FileAttachment", "Collection", "GoToR", "Movie", "Sound", "Screen",
                  "3D", "3DD", "3DV")
_ACTIVE_NAMES = frozenset("/" + name for name in _ACTIVE_LONG + _ACTIVE_SHORT + _ACTIVE_PARSED)
_OBJECT_STREAM_NAME = _pdf_names(("ObjStm",))
_STREAM = re.compile(rb"stream(?:\r\n|\n|\r)")
# An object stream starts with pairs of integers: object number, offset
_OBJECT_STREAM_HEAD = re.compile(rb"\s*\d+\s+\d+\s")


def _note(name: str) -> None:
    raise Refused("encrypted" if name.lower() == "encrypt" else "active_content")


def _is_zlib(data, at: int = 0) -> bool:
    return len(data) - at >= 8 and data[at] & 0x0F == 8 and not ((data[at] << 8) | data[at + 1]) % 31


def _inflate(data, start: int, limit: int) -> Optional[bytes]:
    """The zlib stream at ``start``, inflated to at most ``limit`` bytes. It
    is fed a piece at a time: handing zlib the rest of the file would copy
    the rest of the file for every place a stream may begin."""
    inflater, out, at, piece = zlib.decompressobj(), bytearray(), start, 512
    try:
        while not inflater.eof and at < len(data):
            out += inflater.decompress(bytes(data[at:at + piece]), limit + 1 - len(out))
            if len(out) > limit or inflater.unconsumed_tail:
                raise Refused("unreadable")
            at, piece = at + piece, 64 * 1024
    except zlib.error:
        return None
    return bytes(out)


def _object_stream_text(data: bytes, start: int, budget: int) -> Optional[bytes]:
    """What stands at ``start`` if it inflates to an object stream, through
    up to three layers of compression; None if it is anything else."""
    limit = min(OBJECT_STREAM_MAX, budget)
    if not _is_zlib(data, start):
        return None
    try:
        head = zlib.decompressobj().decompress(bytes(data[start:start + 512]), 256)
    except zlib.error:
        return None
    if not (_OBJECT_STREAM_HEAD.match(head) or _is_zlib(head)):
        return None
    text = _inflate(data, start, limit)
    for _ in range(2):
        if text is None or _OBJECT_STREAM_HEAD.match(text):
            return text
        if not _is_zlib(text):
            return None
        text = _inflate(text, 0, limit)
    return text if text is not None and _OBJECT_STREAM_HEAD.match(text) else None


def _check_pdf_bytes(data: bytes) -> None:
    """The raw bytes: the long names anywhere, and every name in whatever
    inflates to an object stream, wherever a stream may begin."""
    for match in _RAW.finditer(data):
        _note(match.lastgroup)
    budget = OBJECT_STREAMS_TOTAL
    for seen, keyword in enumerate(_STREAM.finditer(data)):
        if seen >= PDF_MAX_STREAMS:
            raise Refused("unreadable")
        text = _object_stream_text(data, keyword.end(), budget)
        if text is None:
            continue
        budget -= len(text)
        for match in _TEXT.finditer(text):
            _note(match.lastgroup)


def _check_pdf_objects(data: bytes) -> None:
    """Every object the file's own cross-reference table leads to, read as a
    PDF reader reads it, and walked for what makes a PDF active."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pypdf import PdfReader
        from pypdf.generic import (ArrayObject, DictionaryObject, IndirectObject, NameObject, NullObject,
                                   StreamObject)

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise Refused("encrypted")
        numbers = {(number, generation) for generation, table in reader.xref.items() for number in table}
        inside = set(reader.xref_objStm)
        numbers |= {(number, 0) for number in inside}
        if len(numbers) > PDF_MAX_OBJECTS:
            raise Refused("unreadable")
        pending, holders = [reader.trailer], 0
        for number, generation in sorted(numbers):
            item = reader.get_object(IndirectObject(number, generation, reader))
            if isinstance(item, NullObject) and number in inside:
                # An object the parser could not read inside an object stream
                # is one nobody checked; another reader may make more of it
                raise Refused("unreadable")
            if isinstance(item, StreamObject) and str(dict.get(item, "/Type")) == "/ObjStm":
                # The objects packed in it, decoded as a reader decodes them
                # (filters, predictors), whether or not the table lists them
                holders += 1
                for match in _TEXT.finditer(item.get_data()[:OBJECT_STREAM_MAX]):
                    _note(match.lastgroup)
            pending.append(item)
        # An object stream the bytes declare and the table does not lead to
        # is one only a reader in repair mode would open: not vouched for
        if sum(1 for _ in _OBJECT_STREAM_NAME.finditer(data)) > holders:
            raise Refused("unreadable")
        seen = 0
        while pending:
            item = pending.pop()
            seen += 1
            if seen > 20 * PDF_MAX_OBJECTS:
                raise Refused("unreadable")
            if isinstance(item, NameObject):
                if str(item) in _ACTIVE_NAMES:
                    raise Refused("active_content")
            elif isinstance(item, DictionaryObject):
                # dict.items: the entries as written, without following references
                for key, value in dict.items(item):
                    if str(key) == "/Encrypt":
                        raise Refused("encrypted")
                    if str(key) in _ACTIVE_NAMES:
                        raise Refused("active_content")
                    pending.append(value)
            elif isinstance(item, ArrayObject):
                pending.extend(list.__iter__(item))
    except Refused:
        raise
    except Exception:  # noqa: BLE001 - a PDF the parser can't follow is one nobody checked
        raise Refused("unreadable") from None


def _check_pdf(data: bytes) -> None:
    if not data.startswith(b"%PDF-"):
        raise Refused("wrong_type")
    _silence_parsers()
    _check_pdf_bytes(data)
    _check_pdf_objects(data)


# ---------------------------------------------------------------------------
# Office Open XML (.xlsx, .docx): a zip of XML parts
# ---------------------------------------------------------------------------
_MAIN_PART = {
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
}
_REQUIRED_PART = {"xlsx": "xl/workbook.xml", "docx": "word/document.xml"}
_NESTED = {".xlsx": "xlsx", ".docx": "docx"}
_MACRO_WORDS = ("macroenabled", "vbaproject", "macrosheet", "dialogsheet", "vbadata")
# Relationship types, by the last part of their address, lower case
_MACRO_RELATIONSHIPS = frozenset({"vbaproject", "wordvbadata", "xlmacrosheet", "xlintlmacrosheet", "xldialogsheet"})
# Things that run, embed any file at all, import a part in another format, or
# fetch data from elsewhere when the file is opened
_ACTIVE_RELATIONSHIPS = frozenset({
    "oleobject", "control", "activexcontrolbinary", "afchunk", "webextension", "webextensiontaskpanes",
    "attachedtemplate", "frame", "subdocument", "mailmergesource", "mailmergeheadersource",
    "connections", "querytable", "extensibility", "recipientdata"})
# Outside targets that fetch nothing by themselves: a link to click, and a
# reference to another workbook's values (Excel asks before following it)
_EXTERNAL_ALLOWED = frozenset({"hyperlink", "externallinkpath", "xlpathmissing", "xlstartup",
                               "xlalternatestartup", "xllibrary"})
# Folders whose parts fetch or run by themselves, whatever their relationships say
_ACTIVE_FOLDERS = frozenset({"activex", "customui", "webextensions", "querytables"})
_ACTIVE_PARTS = frozenset({"xl/connections.xml"})
# Word fields that fetch or run: DDE, and the ones that pull in a file or a
# picture from wherever their instruction says
_ACTIVE_FIELD = re.compile(
    r"\s*\"?\s*(?:DDEAUTO|DDE|INCLUDETEXT|INCLUDEPICTURE|LINK|IMPORT|DATABASE)(?![A-Za-z])", re.IGNORECASE)
_FIELD_ENDS = frozenset({"begin", "separate", "end"})
# Worksheet functions that fetch, call out or run; a | outside a text value is DDE
_ACTIVE_FUNCTION = re.compile(r"(?<![A-Z0-9_.])(?:_XLFN\.)?(?:WEBSERVICE|FILTERXML|RTD|CALL|REGISTER|EXEC)\s*\(")
_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")
_VML = ("urn:schemas-microsoft-com:vml", "urn:schemas-microsoft-com:office:office")


def _local(name: str) -> str:
    """An element's or attribute's own name, without its namespace."""
    return name.rsplit("}", 1)[-1]


def _values(attributes: dict, name: str) -> list[str]:
    """Every value written for an attribute called ``name``, in any letter
    case and any namespace. Office reads one of them; which one is not
    something to guess, so a check holds for all of them or the file is
    refused (``Type="...oleObject" type="...hyperlink"`` is two attributes)."""
    wanted = name.lower()
    return [value for key, value in attributes.items() if _local(key).lower() == wanted]


class _Budget:
    """How much XML one file may have parsed, all its parts together."""

    def __init__(self, left: int):
        self.left = left

    def take(self, size: int) -> None:
        self.left -= size
        if self.left < 0:
            raise Refused("unreadable")


def _parse_part(archive: zipfile.ZipFile, info: zipfile.ZipInfo, budget: _Budget, *,
                start: Callable[[str, dict, "expat.XMLParserType"], None],
                end: Optional[Callable[[str, "expat.XMLParserType"], None]] = None,
                limit: Optional[int] = None) -> None:
    """Read an XML part through expat, as Office does: character references
    decoded, any encoding, attributes whatever their order or contents.
    ``start`` gets the element's own name, its attributes under their full
    names, and the parser (to switch a text handler on while it wants text).
    A document type declaration (the only way to define an entity) and
    anything not well-formed are refused; so is a part past ``limit``, or a
    file past its budget of XML."""
    limit = PART_MAX_BYTES if limit is None else limit
    parser = expat.ParserCreate(namespace_separator="}")
    parser.buffer_text = True

    def forbidden(*_):
        raise Refused("unreadable")

    parser.StartDoctypeDeclHandler = forbidden
    parser.EntityDeclHandler = forbidden
    parser.StartElementHandler = lambda name, attributes: start(name, attributes, parser)
    if end is not None:
        parser.EndElementHandler = lambda name: end(name, parser)
    read = 0
    try:
        with archive.open(info) as part:
            while True:
                chunk = part.read(MB)
                if not chunk:
                    break
                read += len(chunk)
                budget.take(len(chunk))
                if read > limit:
                    raise Refused("unreadable")
                parser.Parse(chunk, False)
        parser.Parse(b"", True)
    except expat.ExpatError:
        raise Refused("unreadable") from None
    except (zipfile.BadZipFile, zlib.error, NotImplementedError, RuntimeError, EOFError):
        raise Refused("unreadable") from None


def _check_relationship(name: str, attributes: dict, parser=None) -> None:
    if _local(name) != "Relationship":
        return
    kinds = [value.rstrip("/").rsplit("/", 1)[-1].lower() for value in _values(attributes, "Type")]
    if any(kind in _MACRO_RELATIONSHIPS for kind in kinds):
        raise Refused("macros")
    if any(kind in _ACTIVE_RELATIONSHIPS for kind in kinds):
        raise Refused("active_content")
    outside = any(value.strip().lower() == "external" for value in _values(attributes, "TargetMode"))
    if outside and not (kinds and all(kind in _EXTERNAL_ALLOWED for kind in kinds)):
        raise Refused("active_content")


def _check_link(name: str, attributes: dict, parser=None) -> None:
    if _local(name) in ("ddeLink", "oleLink"):
        raise Refused("active_content")


def _check_custom_xml(name: str, attributes: dict, parser=None) -> None:
    if _local(name) == "DataMashup":          # Power Query: fetches from wherever its queries say
        raise Refused("active_content")


class _Text:
    """Collects the text of the elements called ``wanted``, switching the
    parser's text handler on only while inside one (a callback for every
    piece of text in a 60 MB part is seconds of nothing)."""

    wanted = ""
    limit = 8192

    def __init__(self):
        self.parts: list[str] = []
        self.size = 0

    def open(self, parser) -> None:
        parser.CharacterDataHandler = self.text

    def text(self, data: str) -> None:
        if self.size < self.limit:
            self.parts.append(data)
            self.size += len(data)

    def end(self, name: str, parser) -> None:
        if _local(name) == self.wanted:
            parser.CharacterDataHandler = None
            self.closed()

    def closed(self) -> None:
        pass

    def taken(self) -> str:
        text, self.parts, self.size = "".join(self.parts), [], 0
        return text


class _WordFields(_Text):
    """Word's field instructions, joined across the runs they are split over
    (``DD`` in one run, ``EAUTO`` in the next); and pictures a drawing
    fetches from elsewhere without a relationship."""

    wanted = "instrText"

    def start(self, name: str, attributes: dict, parser) -> None:
        local = _local(name)
        if local == "instrText":
            self.open(parser)
        elif local == "fldSimple":
            for instruction in _values(attributes, "instr"):
                self.check(instruction)
        elif local == "fldChar" and any(v.lower() in _FIELD_ENDS for v in _values(attributes, "fldCharType")):
            self.flush()
        if name.startswith(_VML) or any(key.startswith(_VML) for key in attributes):
            if any(_values(attributes, which) for which in ("href", "src")):
                raise Refused("active_content")

    def flush(self) -> None:
        self.check(self.taken())

    @staticmethod
    def check(instruction: str) -> None:
        if _ACTIVE_FIELD.match(instruction):
            raise Refused("active_content")


class _Formulas(_Text):
    """A worksheet's formulas: the functions that fetch, call out or run, and
    DDE written straight into a cell (``=cmd|'/c calc'!A1``)."""

    wanted = "f"

    def start(self, name: str, attributes: dict, parser) -> None:
        if _local(name) == "f":
            self.open(parser)

    def closed(self) -> None:
        formula = self.taken().upper()
        if _ACTIVE_FUNCTION.search(formula) or "|" in _QUOTED.sub("", formula):
            raise Refused("active_content")


def _content_types(archive: zipfile.ZipFile, info: zipfile.ZipInfo, budget: _Budget) -> list[str]:
    found: list[str] = []
    _parse_part(archive, info, budget, limit=4 * MB, start=lambda name, attributes, parser: found.extend(
        value.strip().lower() for value in _values(attributes, "ContentType")))
    return found


def _check_open_xml(data: bytes, kind: str, *, nested: bool = False, budget: Optional[_Budget] = None) -> None:
    budget = _Budget(XML_BUDGET_BYTES) if budget is None else budget
    if data.startswith(OLE_MAGIC):
        # A password-protected Open XML file is an OLE container around it
        raise Refused("encrypted" if b"E\x00n\x00c\x00r\x00y\x00p\x00t\x00e\x00d\x00" in data
                      else "wrong_type")
    if not data.startswith(ZIP_MAGIC):
        raise Refused("wrong_type")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, ValueError, NotImplementedError):
        raise Refused("wrong_type") from None
    with archive:
        members = archive.infolist()
        if any(m.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) for m in members):
            raise Refused("unreadable")
        if len(members) > ARCHIVE_MAX_MEMBERS or sum(m.file_size for m in members) > ARCHIVE_MAX_BYTES:
            raise Refused("archive_bomb")
        names = [m.filename.lower().replace("\\", "/") for m in members]
        if len(set(names)) != len(names):
            # Two parts under one name (in any letter case): which one a
            # program opens is its own business, so nobody can vouch for it
            raise Refused("unreadable")
        by_name = {m.filename: m for m in members}
        if "[Content_Types].xml" not in by_name or _REQUIRED_PART[kind] not in by_name:
            raise Refused("wrong_type")
        types = _content_types(archive, by_name["[Content_Types].xml"], budget)
        if any(word in declared for declared in types for word in _MACRO_WORDS):
            raise Refused("macros")
        if _MAIN_PART[kind] not in types:
            raise Refused("wrong_type")
        for info in members:
            _check_member(archive, info, kind, nested, budget)


def _check_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, kind: str, nested: bool,
                  budget: _Budget) -> None:
    if info.is_dir():
        return
    if info.flag_bits & 0x1:
        raise Refused("encrypted")
    name = info.filename.lower().replace("\\", "/")
    if any(word in name for word in _MACRO_WORDS):
        raise Refused("macros")
    folders = name.split("/")[:-1]
    if _ACTIVE_FOLDERS.intersection(folders) or name in _ACTIVE_PARTS:
        raise Refused("active_content")
    try:
        with archive.open(info) as part:
            head = part.read(8)
    except (zipfile.BadZipFile, zlib.error, NotImplementedError, RuntimeError, EOFError):
        raise Refused("unreadable") from None
    if head.startswith(OLE_MAGIC):
        # An OLE object inside the package: a macro project or an embedded
        # object (which can carry any file at all), whatever it is called
        raise Refused("active_content")
    if head.startswith(ZIP_MAGIC):
        # A package inside the package, wherever it sits: a chart's own
        # workbook is checked like any other, anything else is refused
        inner = _NESTED.get(name[name.rfind("."):]) if "embeddings" in folders else None
        if inner is None or nested or info.file_size > NESTED_MAX_BYTES:
            raise Refused("active_content")
        with archive.open(info) as part:
            _check_open_xml(part.read(NESTED_MAX_BYTES), inner, nested=True, budget=budget)
    elif "embeddings" in folders:
        raise Refused("active_content")
    elif name.endswith(".rels"):
        _parse_part(archive, info, budget, start=_check_relationship)
    elif name == "[content_types].xml":
        return
    elif "customxml" in folders and name.endswith(".xml"):
        _parse_part(archive, info, budget, start=_check_custom_xml)
    elif kind == "xlsx" and "externallinks" in folders and name.endswith(".xml"):
        _parse_part(archive, info, budget, start=_check_link)
    elif kind == "xlsx" and name.endswith(".xml") and folders[:1] == ["xl"] and len(folders) > 1 \
            and folders[1] in ("worksheets", "chartsheets", "macrosheets", "dialogsheets"):
        formulas = _Formulas()
        _parse_part(archive, info, budget, start=formulas.start, end=formulas.end)
    elif kind == "docx" and name.endswith(".xml"):
        fields = _WordFields()
        _parse_part(archive, info, budget, start=fields.start, end=fields.end)
        fields.flush()


def _silence_parsers() -> None:
    """The parsers' own logs may quote a file. Nothing from a file is logged."""
    for name in ("olevba", "oletools", "olefile", "msoffcrypto", "msodde", "pcodedmp", "oleobj",
                 "xls_parser", "record_base", "ftguess", "crypto", "pypdf"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL + 1)
        logger.propagate = False


def _macros(data: bytes) -> bool:
    """oletools' verdict: VBA or Excel 4 macros anywhere in the file."""
    _silence_parsers()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # olevba's grammar uses names pyparsing has renamed
        from oletools import olevba

    _silence_parsers()      # importing olevba creates its loggers
    parser = None
    try:
        parser = olevba.VBA_Parser("upload", data=data)
        return bool(parser.detect_vba_macros() or parser.detect_xlm_macros())
    except olevba.FileOpenError:
        raise Refused("wrong_type") from None
    finally:
        if parser is not None:
            parser.close()


def _check_office_xml(kind: str):
    def check(data: bytes) -> None:
        _check_open_xml(data, kind)
        if _macros(data):
            raise Refused("macros")
    return check


# ---------------------------------------------------------------------------
# The older binary formats (.xls, .doc): OLE compound files
# ---------------------------------------------------------------------------
_MAIN_STREAM = {"xls": ("workbook", "book"), "doc": ("worddocument",)}
# Storages that hold embedded objects, in Word and in Excel
_OBJECT_STORAGES = ("objectpool",)
_OBJECT_STREAMS = ("\x01ole10native", "\x03objinfo", "ctls")


def _check_binary(kind: str):
    def check(data: bytes) -> None:
        if not data.startswith(OLE_MAGIC):
            raise Refused("wrong_type")
        _silence_parsers()
        import olefile
        from oletools import crypto

        _silence_parsers()
        try:
            ole = olefile.OleFileIO(io.BytesIO(data))
        except Exception:  # noqa: BLE001 - olefile raises several kinds for a broken container
            raise Refused("wrong_type") from None
        with ole:
            paths = [[part.lower() for part in path] for path in ole.listdir(streams=True, storages=True)]
            if not any(len(p) == 1 and p[0] in _MAIN_STREAM[kind] for p in paths):
                raise Refused("wrong_type")
            if any(p[-1] in ("encryptedpackage", "encryptioninfo") for p in paths) \
                    or crypto.is_encrypted(ole):
                raise Refused("encrypted")
            for path in paths:
                if any(part in _OBJECT_STORAGES or part.startswith("mbd") for part in path) \
                        or path[-1] in _OBJECT_STREAMS:
                    raise Refused("active_content")
            if kind == "doc" and _doc_dde(ole):
                raise Refused("active_content")
            if kind == "xls":
                _check_workbook_streams(ole)
        if _macros(data):
            raise Refused("macros")
    return check


# A field's start mark and its instruction, in one-byte or two-byte text
# (oletools reads the first; Word stores the second whenever the text is not
# plain ASCII). The space, quote or backslash after the word keeps a chance
# run of bytes in a picture from reading as one
_DOC_FIELD = re.compile(
    rb"\x13\x00?(?:[ \"]\x00?){0,8}(?:" + b"|".join(
        b"".join(b"%c\x00?" % ch for ch in word) for word in (
            b"DDEAUTO", b"DDE", b"INCLUDETEXT", b"INCLUDEPICTURE", b"LINK", b"IMPORT", b"DATABASE"))
    + rb")[ \"\\]", re.IGNORECASE)


def _doc_dde(ole) -> bool:
    from oletools import msodde

    _silence_parsers()
    if msodde.process_doc(ole).strip():
        return True
    for path in ole.listdir(streams=True, storages=False):
        if _DOC_FIELD.search(ole.openstream(path).read(MAX_FILE_BYTES)):
            return True
    return False


_BIFF_FILEPASS, _BIFF_SUPBOOK = 0x002F, 0x01AE


def _biff_records(stream: bytes) -> Iterator[tuple[int, bytes]]:
    """A workbook stream's records: type, then that many bytes. Stops at the
    first one that runs off the end (writers pad the stream with zeros)."""
    at = 0
    while at + 4 <= len(stream):
        kind, size = struct.unpack_from("<HH", stream, at)
        at += 4
        if at + size > len(stream):
            return
        yield kind, stream[at:at + size]
        at += size


def _is_data_link(record: bytes) -> bool:
    """Whether a SupBook record names a DDE or OLE data source ([MS-XLS]
    2.4.271): no sheet names, and a path that is a real string -- not the
    markers for this workbook, an add-in, an unused entry or the same sheet."""
    if len(record) < 4:
        return False
    sheets, length = struct.unpack_from("<HH", record, 0)
    if sheets != 0 or not 0 < length <= 0xFF or len(record) < 6:
        return False
    wide = record[4] & 0x01
    first = record[5] | (record[6] << 8 if wide and len(record) > 6 else 0)
    return not (length == 1 and first in (0x00, 0x20))


def _check_workbook_streams(ole) -> None:
    for name in ("Workbook", "Book"):
        if not ole.exists(name):
            continue
        for kind, record in _biff_records(ole.openstream(name).read(MAX_FILE_BYTES)):
            if kind == _BIFF_FILEPASS:
                raise Refused("encrypted")
            if kind == _BIFF_SUPBOOK and _is_data_link(record):
                raise Refused("active_content")


_CHECKS = {
    "pdf": _check_pdf,
    "xlsx": _check_office_xml("xlsx"),
    "docx": _check_office_xml("docx"),
    "xls": _check_binary("xls"),
    "doc": _check_binary("doc"),
}

__all__ = ["CLEAN", "FAILED", "Verdict", "scan", "scan_path"]

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
