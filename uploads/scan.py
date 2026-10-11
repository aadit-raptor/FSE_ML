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
# An Office file's parts together, and one XML part read through
ARCHIVE_MAX_BYTES = 512 * MB
ARCHIVE_MAX_MEMBERS = 10_000
PART_MAX_BYTES = 64 * MB
NESTED_MAX_BYTES = MAX_FILE_BYTES
# The child process: CPU seconds, address space (Linux), and the deadline.
# Generous, because the free server has a fraction of a processor: a 20 MB
# file takes about 3 seconds on a laptop
SCAN_CPU_S = 90
SCAN_MEMORY_BYTES = 700 * MB
SCAN_TIMEOUT_S = 150.0

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
_ACTIVE_NAMES = frozenset("/" + name for name in _ACTIVE_LONG + _ACTIVE_SHORT)
_STREAM = re.compile(rb"stream(?:\r\n|\n|\r)")
# An object stream starts with pairs of integers: object number, offset
_OBJECT_STREAM_HEAD = re.compile(rb"\s*\d+\s+\d+\s")


def _note(name: str) -> None:
    raise Refused("encrypted" if name.lower() == "encrypt" else "active_content")


def _is_zlib(data, at: int = 0) -> bool:
    return len(data) - at >= 8 and data[at] & 0x0F == 8 and not ((data[at] << 8) | data[at + 1]) % 31


def _object_stream_text(data: bytes, start: int, budget: int) -> Optional[bytes]:
    """What stands at ``start`` if it inflates to an object stream, through
    up to three layers of compression; None if it is anything else."""
    body, limit = memoryview(data)[start:], min(OBJECT_STREAM_MAX, budget)
    for _ in range(3):
        if not _is_zlib(body):
            return None
        try:
            head = zlib.decompressobj().decompress(body, 256)
            if not (_OBJECT_STREAM_HEAD.match(head) or _is_zlib(head)):
                return None
            text = zlib.decompressobj().decompress(body, limit + 1)
        except zlib.error:
            return None
        if len(text) > limit:
            raise Refused("unreadable")
        if _OBJECT_STREAM_HEAD.match(text):
            return text
        body = memoryview(text)
    return None


def _check_pdf_bytes(data: bytes) -> None:
    """The raw bytes: the long names anywhere, and every name in whatever
    inflates to an object stream, wherever a stream may begin."""
    for match in _RAW.finditer(data):
        _note(match.lastgroup)
    budget = OBJECT_STREAMS_TOTAL
    for keyword in _STREAM.finditer(data):
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
        from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject, NameObject

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise Refused("encrypted")
        numbers = {(number, generation) for generation, table in reader.xref.items() for number in table}
        numbers |= {(number, 0) for number in reader.xref_objStm}
        if len(numbers) > PDF_MAX_OBJECTS:
            raise Refused("unreadable")
        pending = [reader.trailer]
        pending += [reader.get_object(IndirectObject(number, generation, reader))
                    for number, generation in sorted(numbers)]
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
# Things that run, embed any file at all, or import a part in another format
_ACTIVE_RELATIONSHIPS = frozenset({
    "oleobject", "control", "activexcontrolbinary", "afchunk", "webextension", "webextensiontaskpanes",
    "attachedtemplate", "frame", "subdocument", "mailmergesource", "mailmergeheadersource"})
# Outside targets that fetch nothing by themselves: a link to click, and a
# reference to another workbook's values (asked about on opening)
_EXTERNAL_ALLOWED = frozenset({"hyperlink", "externallinkpath", "xlpathmissing", "xlstartup",
                               "xlalternatestartup", "xllibrary"})
_DDE_FIELD = re.compile(r"\s*\"?\s*DDE(?:AUTO)?(?![A-Za-z])", re.IGNORECASE)
_FIELD_ENDS = frozenset({"begin", "separate", "end"})


def _local(name: str) -> str:
    """An element's or attribute's own name, without its namespace."""
    return name.rsplit("}", 1)[-1]


def _parse_part(archive: zipfile.ZipFile, info: zipfile.ZipInfo, *, start: Callable[[str, dict], None],
                text: Optional[Callable[[str], None]] = None, end: Optional[Callable[[str], None]] = None,
                limit: Optional[int] = None) -> None:
    """Read an XML part through expat, as Office does: character references
    decoded, any encoding, attributes whatever their order or contents. A
    document type declaration (the only way to define an entity) and
    anything not well-formed are refused; so is a part past ``limit``."""
    limit = PART_MAX_BYTES if limit is None else limit
    parser = expat.ParserCreate(namespace_separator="}")
    parser.buffer_text = True

    def forbidden(*_):
        raise Refused("unreadable")

    parser.StartDoctypeDeclHandler = forbidden
    parser.EntityDeclHandler = forbidden
    parser.StartElementHandler = lambda name, attributes: start(
        _local(name), {_local(k).lower(): v for k, v in attributes.items()})
    if text is not None:
        parser.CharacterDataHandler = text
    if end is not None:
        parser.EndElementHandler = lambda name: end(_local(name))
    read = 0
    try:
        with archive.open(info) as part:
            while True:
                chunk = part.read(MB)
                if not chunk:
                    break
                read += len(chunk)
                if read > limit:
                    raise Refused("unreadable")
                parser.Parse(chunk, False)
        parser.Parse(b"", True)
    except expat.ExpatError:
        raise Refused("unreadable") from None
    except (zipfile.BadZipFile, zlib.error, NotImplementedError, RuntimeError, EOFError):
        raise Refused("unreadable") from None


def _check_relationship(name: str, attributes: dict) -> None:
    if name != "Relationship":
        return
    kind = (attributes.get("type") or "").rstrip("/").rsplit("/", 1)[-1].lower()
    if kind in _MACRO_RELATIONSHIPS:
        raise Refused("macros")
    if kind in _ACTIVE_RELATIONSHIPS:
        raise Refused("active_content")
    if (attributes.get("targetmode") or "").strip().lower() == "external" and kind not in _EXTERNAL_ALLOWED:
        raise Refused("active_content")


def _check_link(name: str, attributes: dict) -> None:
    if name in ("ddeLink", "oleLink"):
        raise Refused("active_content")


class _WordFields:
    """Word's field instructions, joined across the runs they are split over
    (``DD`` in one run, ``EAUTO`` in the next)."""

    def __init__(self):
        self.parts: list[str] = []
        self.reading = False

    def start(self, name: str, attributes: dict) -> None:
        if name == "instrText":
            self.reading = True
        elif name == "fldSimple":
            self.check(attributes.get("instr") or "")
        elif name == "fldChar" and (attributes.get("fldchartype") or "").lower() in _FIELD_ENDS:
            self.flush()

    def text(self, data: str) -> None:
        if self.reading and sum(map(len, self.parts)) < 4096:
            self.parts.append(data)

    def end(self, name: str) -> None:
        if name == "instrText":
            self.reading = False

    def flush(self) -> None:
        self.check("".join(self.parts))
        self.parts = []

    @staticmethod
    def check(instruction: str) -> None:
        if _DDE_FIELD.match(instruction):
            raise Refused("active_content")


def _content_types(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> list[str]:
    found: list[str] = []
    _parse_part(archive, info, start=lambda name, attributes: found.append(
        (attributes.get("contenttype") or "").strip().lower()), limit=4 * MB)
    return found


def _check_open_xml(data: bytes, kind: str, *, nested: bool = False) -> None:
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
        by_name = {m.filename: m for m in members}
        if "[Content_Types].xml" not in by_name or _REQUIRED_PART[kind] not in by_name:
            raise Refused("wrong_type")
        types = _content_types(archive, by_name["[Content_Types].xml"])
        if any(word in declared for declared in types for word in _MACRO_WORDS):
            raise Refused("macros")
        if _MAIN_PART[kind] not in types:
            raise Refused("wrong_type")
        for info in members:
            _check_member(archive, info, kind, nested)


def _check_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, kind: str, nested: bool) -> None:
    if info.is_dir():
        return
    if info.flag_bits & 0x1:
        raise Refused("encrypted")
    name = info.filename.lower()
    if any(word in name for word in _MACRO_WORDS):
        raise Refused("macros")
    folders = name.split("/")[:-1]
    if "activex" in folders:
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
            _check_open_xml(part.read(NESTED_MAX_BYTES), inner, nested=True)
    elif "embeddings" in folders:
        raise Refused("active_content")
    elif name.endswith(".rels"):
        _parse_part(archive, info, start=_check_relationship)
    elif name == "[content_types].xml":
        return
    elif kind == "xlsx" and "externallinks" in folders and name.endswith(".xml"):
        _parse_part(archive, info, start=_check_link)
    elif kind == "docx" and name.endswith(".xml"):
        fields = _WordFields()
        _parse_part(archive, info, start=fields.start, text=fields.text, end=fields.end)
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


def _doc_dde(ole) -> bool:
    from oletools import msodde

    _silence_parsers()
    return bool(msodde.process_doc(ole).strip())


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
