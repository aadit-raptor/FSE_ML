"""Files for the upload tests, built in memory: nothing here is a real document.

Each builder returns bytes. The "bad" ones carry the *structure* the scanner
refuses (a macro project's streams, a JavaScript action, a remote template),
never working code: there is nothing in them that could run.
"""
from __future__ import annotations

import io
import struct
import zipfile
import zlib

# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
_PAGE = (b"<< /Type /Catalog /Pages 2 0 R >>",
         b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
         b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>")
OBJECT_STREAM = b"/Type /ObjStm /Filter /FlateDecode"


def _stream(dictionary: bytes, body: bytes) -> bytes:
    return b"<< %s /Length %d >>\nstream\n" % (dictionary, len(body)) + body + b"\nendstream"


def pdf(*objects, compressed: tuple = (), padding: int = 0, trailer: bytes = b"",
        object_stream: bytes = OBJECT_STREAM, encode=zlib.compress, before: bytes = b"") -> bytes:
    """A well-formed one-page PDF (cross-reference stream and all).

    ``objects`` are added as objects 5 onwards: each a dictionary's bytes, or
    ``(dictionary, body)`` for a stream. ``compressed`` are dictionaries put
    inside an object stream, whose own dictionary and encoding can be
    overridden. ``padding`` adds that many bytes of incompressible image-like
    data; ``trailer`` adds entries to the trailer; ``before`` is written
    straight after the header."""
    content = (b"/Filter /FlateDecode", zlib.compress(b"BT /F1 12 Tf (Annual report) Tj ET"))
    plain = [*_PAGE, content, *objects]
    if padding:
        import random
        plain.append((b"/Type /XObject /Subtype /Image /Filter /DCTDecode",
                      b"\xff\xd8\xff" + random.Random(7).randbytes(padding)))
    out, offsets = bytearray(b"%PDF-1.7\n" + before), {}
    for number, item in enumerate(plain, start=1):
        offsets[number] = len(out)
        body = _stream(*item) if isinstance(item, tuple) else item
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    inside = {}
    if compressed:
        holder = len(plain) + 1
        first = holder + 1
        index = b" ".join(b"%d %d" % (first + i, sum(len(c) + 1 for c in compressed[:i]))
                          for i in range(len(compressed))) + b" "
        packed = encode(index + b" ".join(compressed))
        offsets[holder] = len(out)
        out += b"%d 0 obj\n" % holder + _stream(
            object_stream + b" /N %d /First %d" % (len(compressed), len(index)), packed) + b"\nendobj\n"
        inside = {first + i: (holder, i) for i in range(len(compressed))}
    table = max([*offsets, *inside]) + 1
    offsets[table] = len(out)
    rows = bytearray(struct.pack(">BIH", 0, 0, 65535))
    for number in range(1, table + 1):
        if number in inside:
            rows += struct.pack(">BIH", 2, *inside[number])
        else:
            rows += struct.pack(">BIH", 1, offsets[number], 0)
    out += b"%d 0 obj\n" % table + _stream(
        b"/Type /XRef /W [1 4 2] /Size %d /Root 1 0 R %s" % (table + 1, trailer), bytes(rows)) + b"\nendobj\n"
    out += b"startxref\n%d\n%%%%EOF\n" % offsets[table]
    return bytes(out)


JAVASCRIPT = b"<< /S /JavaScript /JS (app.alert\\(1\\)) >>"

# ---------------------------------------------------------------------------
# Office Open XML
# ---------------------------------------------------------------------------
_TYPES = {
    "xlsx": ("xl/workbook.xml", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"),
    "docx": ("word/document.xml",
             "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"),
}
_BODY = {
    "xlsx": b'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheets/></workbook>',
    "docx": (b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
             b"<w:body><w:p><w:r><w:t>Annual report</w:t></w:r></w:p></w:body></w:document>"),
}
RELS = ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        "{}</Relationships>")


def relationship(kind: str, target: str, *, external: bool = False) -> str:
    mode = ' TargetMode="External"' if external else ""
    return (f'<Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            f'relationships/{kind}" Target="{target}"{mode}/>')


def open_xml(kind: str, parts: dict | None = None, *, content_types: str | None = None,
             main: bytes | None = None) -> bytes:
    """A minimal .xlsx or .docx, plus ``parts`` (name -> bytes or str)."""
    main_part, main_type = _TYPES[kind]
    types = content_types or (
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        f'<Override PartName="/{main_part}" ContentType="{main_type}"/></Types>')
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", types)
        archive.writestr("_rels/.rels", RELS.format(relationship("officeDocument", main_part)))
        archive.writestr(main_part, _BODY[kind] if main is None else main)
        for name, body in (parts or {}).items():
            archive.writestr(name, body)
    return out.getvalue()


def word_document(*fields: tuple) -> bytes:
    """A document with one field per tuple, each field's instruction split
    over the runs given."""
    def field(runs: tuple) -> str:
        inner = "".join(f"<w:r><w:instrText>{run}</w:instrText></w:r>" for run in runs)
        return ('<w:r><w:fldChar w:fldCharType="begin"/></w:r>' + inner +
                '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>result</w:t></w:r>'
                '<w:r><w:fldChar w:fldCharType="end"/></w:r>')
    return ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p>'
            + "".join(field(runs) for runs in fields) + "</w:p></w:body></w:document>").encode()


def word_fields(*runs: str) -> bytes:
    """A document whose one field's instruction is split over ``runs``."""
    return word_document(runs)


# ---------------------------------------------------------------------------
# OLE compound files (.doc, .xls)
# ---------------------------------------------------------------------------
_SECTOR = 512
_FREE, _END, _FAT, _NONE = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFF
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def compound_file(streams: dict[str, bytes]) -> bytes:
    """A compound file with ``streams`` (``"Macros/VBA/dir"`` -> bytes).

    Every stream is padded to 4,096 bytes, so none needs the mini stream; each
    storage's children are chained through their right-hand links.
    """
    tree: dict = {}
    for path, body in streams.items():
        node = tree
        *folders, leaf = path.split("/")
        for folder in folders:
            node = node.setdefault(folder, {})
        node[leaf] = body.ljust(4096, b"\0")

    entries: list[dict] = [{"name": "Root Entry", "type": 5, "body": None}]

    def add(children: dict) -> int:
        """Add ``children`` as siblings; the index of the first."""
        first = previous = _NONE
        for name in sorted(children, key=lambda n: (len(n), n.upper())):
            value = children[name]
            index = len(entries)
            entries.append({"name": name, "type": 1 if isinstance(value, dict) else 2,
                            "body": None if isinstance(value, dict) else value})
            if previous == _NONE:
                first = index
            else:
                entries[previous]["right"] = index
            previous = index
            if isinstance(value, dict):
                entries[index]["child"] = add(value)
        return first

    entries[0]["child"] = add(tree)
    directory_sectors = -(-len(entries) // 4)
    data_sectors = sum(len(e["body"]) // _SECTOR + (len(e["body"]) % _SECTOR > 0)
                       for e in entries if e["body"] is not None)
    fat_sectors = 1
    while fat_sectors * 128 < fat_sectors + directory_sectors + data_sectors:
        fat_sectors += 1
    fat = [_FAT] * fat_sectors
    first_directory = len(fat)
    fat += [first_directory + i + 1 for i in range(directory_sectors - 1)] + [_END]
    data = b""
    for entry in entries:
        if entry["body"] is None:
            continue
        count = -(-len(entry["body"]) // _SECTOR)
        entry["start"] = len(fat)
        fat += [len(fat) + i + 1 for i in range(count - 1)] + [_END]
        data += entry["body"].ljust(count * _SECTOR, b"\0")
    fat += [_FREE] * (fat_sectors * 128 - len(fat))

    header = OLE_MAGIC + b"\0" * 16 + struct.pack(
        "<HHHHH6xIIIIIIIII", 0x3E, 3, 0xFFFE, 9, 6, 0, fat_sectors, first_directory, 0, 4096,
        _END, 0, _END, 0)
    header += b"".join(struct.pack("<I", i if i < fat_sectors else _FREE) for i in range(109))
    directory = b""
    for entry in entries:
        name = entry["name"].encode("utf-16-le")
        body = entry["body"]
        directory += (name.ljust(64, b"\0") + struct.pack(
            "<HBBIII", len(name) + 2, entry["type"], 1, _NONE, entry.get("right", _NONE),
            entry.get("child", _NONE)) + b"\0" * 36 + struct.pack(
            "<IQ", entry.get("start", _END if entry["type"] == 5 else 0), len(body) if body else 0))
    directory = directory.ljust(directory_sectors * _SECTOR, b"\0")
    table = b"".join(struct.pack("<I", value) for value in fat)
    return header + table + directory + data


def word_binary(extra: dict[str, bytes] | None = None) -> bytes:
    """Something shaped like a .doc: the stream Word's format is named for."""
    return compound_file({"WordDocument": b"\xec\xa5\xc1\x00" + b"\0" * 64, "1Table": b"\0" * 64,
                          **(extra or {})})


MACRO_PROJECT = {"Macros/PROJECT": b"ID=\"{0}\"\r\nModule=Module1\r\n",
                 "Macros/VBA/dir": b"\x01\x00\xb0\x00", "Macros/VBA/_VBA_PROJECT": b"\xcc\x61\xff\xff",
                 "Macros/VBA/Module1": b"\0" * 16}
