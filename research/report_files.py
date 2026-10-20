"""A results note as a Word file and as a PDF (PLAN.md R8).

Both are written here from a **checked** draft (research/report.py
``check``), with no library beyond Python's own: a Word file is a zip of a
few XML parts, a PDF a few hundred lines of drawing operators. So nothing new
is installed on the free server, and nothing but the draft's own segments
reaches either file.

**Word**: headings, paragraphs and tables in Word's own styles, and each
chart a **native Word chart** (DrawingML) holding the figures as its data,
so it can be restyled in Word. **PDF**: A4, the standard Helvetica fonts
(WinAnsi encoding: a character those fonts lack, such as the rupee sign, is
written "Rs" or "?"), tables ruled and charts drawn as bars, every label and
figure the draft's.
"""
from __future__ import annotations

import datetime
import io
import re
import zipfile
from xml.sax.saxutils import escape

from research.report import Chart, Draft, Figure, Heading, Paragraph, Table, Typed, plain

SOURCE = "variater.com"
_BAD_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")
COLOURS = ("2F7F95", "8A9AA3", "58B28A", "D9A54A")       # the Tape accent, darkened for paper; context; gain


def _x(text: str) -> str:
    return escape(_BAD_XML.sub("", text))


def _lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


# ===================================================================================================== Word
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
C = "http://schemas.openxmlformats.org/drawingml/2006/chart"

STYLES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri" w:eastAsia="Calibri"/>
<w:sz w:val="20"/><w:szCs w:val="20"/><w:lang w:val="en-GB"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
<w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="120" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr>
<w:rPr><w:b/><w:color w:val="1F3A44"/><w:sz w:val="36"/><w:szCs w:val="36"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
<w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="80"/><w:outlineLvl w:val="1"/></w:pPr>
<w:rPr><w:b/><w:color w:val="2F7F95"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Note"><w:name w:val="Note"/><w:basedOn w:val="Normal"/>
<w:rPr><w:color w:val="5C6B73"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Analyst"><w:name w:val="Analyst"/><w:basedOn w:val="Normal"/>
<w:pPr><w:ind w:left="0"/></w:pPr></w:style>
<w:style w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/>
<w:tblPr><w:tblInd w:w="0" w:type="dxa"/><w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/>
<w:bottom w:w="0" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>
</w:styles>"""


def _runs(parts, bold: bool = False) -> str:
    out = []
    for seg in parts:
        rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
        lines = _lines(seg.text)
        body = "<w:br/>".join('<w:t xml:space="preserve">%s</w:t>' % _x(line) for line in lines)
        out.append("<w:r>%s%s</w:r>" % (rpr, body))
    return "".join(out)


def _para(parts, style: str = "", align: str = "", bold: bool = False, spacing: bool = True) -> str:
    ppr = ""
    if style or align or not spacing:
        ppr = "<w:pPr>%s%s%s</w:pPr>" % ('<w:pStyle w:val="%s"/>' % style if style else "",
                                         '<w:spacing w:after="0"/>' if not spacing else "",
                                         '<w:jc w:val="%s"/>' % align if align else "")
    return "<w:p>%s%s</w:p>" % (ppr, _runs(parts, bold))


def _paragraphs(block: Paragraph) -> str:
    style = {"note": "Note", "analyst": "Analyst"}.get(block.kind, "")
    if len(block.parts) == 1 and isinstance(block.parts[0], Typed):     # the analyst's paragraphs, kept as typed
        return "".join(_para((type(block.parts[0])(block.parts[0].field, line),), style)
                       for line in _lines(block.parts[0].text))
    return _para(block.parts, style)


def _is_number(cell) -> bool:
    return bool(cell) and all(isinstance(seg, Figure) for seg in cell)


def _table(block: Table) -> str:
    columns = len(block.header)
    first, rest = 3600, max(900, (9000 - 3600) // max(1, columns - 1))
    widths = [first] + [rest] * (columns - 1)
    border = ('<w:tblBorders><w:top w:val="single" w:sz="6" w:color="1F3A44"/><w:bottom w:val="single" w:sz="6" '
              'w:color="1F3A44"/><w:insideH w:val="single" w:sz="2" w:color="C9D2D6"/></w:tblBorders>')
    grid = "".join('<w:gridCol w:w="%d"/>' % w for w in widths)
    out = ['<w:tbl><w:tblPr><w:tblW w:w="%d" w:type="dxa"/>%s<w:tblLayout w:type="fixed"/></w:tblPr>'
           '<w:tblGrid>%s</w:tblGrid>' % (sum(widths), border, grid)]
    for r, row in enumerate((block.header,) + tuple(block.rows)):
        cells = []
        for c, cell in enumerate(row):
            align = "right" if c > 0 else ""
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="EEF2F4"/>' if r == 0 else ""
            cells.append('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>%s</w:tcPr>%s</w:tc>'
                         % (widths[c], shade, _para(cell, align=align, bold=r == 0, spacing=False)))
        out.append("<w:tr>%s</w:tr>" % "".join(cells))
    out.append("</w:tbl>")
    return "".join(out) + "<w:p/>"


def chart_xml(block: Chart) -> str:
    cats = [seg.text for seg in block.categories]

    def lit(values, numeric):
        tag = "numLit" if numeric else "strLit"
        points = "".join('<c:pt idx="%d"><c:v>%s</c:v></c:pt>' % (i, _x(repr(float(v)) if numeric else v))
                         for i, v in enumerate(values) if v is not None)
        code = "<c:formatCode>General</c:formatCode>" if numeric else ""
        return "<c:%s>%s<c:ptCount val=\"%d\"/>%s</c:%s>" % (tag, code, len(values), points, tag)

    series = []
    for i, (name, figures) in enumerate(block.series):
        series.append(
            '<c:ser><c:idx val="%d"/><c:order val="%d"/><c:tx><c:v>%s</c:v></c:tx>'
            '<c:spPr><a:solidFill><a:srgbClr val="%s"/></a:solidFill></c:spPr><c:invertIfNegative val="0"/>'
            '<c:cat>%s</c:cat><c:val>%s</c:val></c:ser>'
            % (i, i, _x(plain(name)), COLOURS[i % len(COLOURS)], lit(cats, False), lit([f.value for f in figures], True)))
    fmt = "#,##0.00" if block.unit == "rupees2" else "#,##0"
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<c:chartSpace xmlns:c="{C}" xmlns:a="{A}" xmlns:r="{R}"><c:roundedCorners val="0"/><c:chart>'
            f'<c:title><c:tx><c:rich><a:bodyPr/><a:p><a:pPr><a:defRPr sz="1100" b="1"/></a:pPr><a:r><a:rPr lang="en-GB" '
            f'sz="1100" b="1"/><a:t>{_x(plain(block.title))}</a:t></a:r></a:p></c:rich></c:tx><c:overlay val="0"/>'
            f'</c:title><c:autoTitleDeleted val="0"/><c:plotArea><c:layout/><c:barChart><c:barDir val="col"/>'
            f'<c:grouping val="clustered"/><c:varyColors val="0"/>{"".join(series)}<c:gapWidth val="80"/>'
            f'<c:axId val="101"/><c:axId val="102"/></c:barChart>'
            f'<c:catAx><c:axId val="101"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:delete val="0"/>'
            f'<c:axPos val="b"/><c:numFmt formatCode="General" sourceLinked="0"/><c:majorTickMark val="none"/>'
            f'<c:minorTickMark val="none"/><c:tickLblPos val="low"/><c:crossAx val="102"/><c:crosses val="autoZero"/>'
            f'<c:auto val="1"/><c:lblAlgn val="ctr"/><c:lblOffset val="100"/><c:noMultiLvlLbl val="0"/></c:catAx>'
            f'<c:valAx><c:axId val="102"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:delete val="0"/>'
            f'<c:axPos val="l"/><c:majorGridlines><c:spPr><a:ln w="3175"><a:solidFill><a:srgbClr val="D9E0E3"/>'
            f'</a:solidFill></a:ln></c:spPr></c:majorGridlines><c:numFmt formatCode="{fmt}" sourceLinked="0"/>'
            f'<c:majorTickMark val="none"/><c:minorTickMark val="none"/><c:tickLblPos val="nextTo"/>'
            f'<c:crossAx val="101"/><c:crosses val="autoZero"/><c:crossBetween val="between"/></c:valAx></c:plotArea>'
            f'<c:legend><c:legendPos val="b"/><c:overlay val="0"/></c:legend><c:plotVisOnly val="1"/>'
            f'<c:dispBlanksAs val="gap"/></c:chart></c:chartSpace>')


def _chart_ref(number: int) -> str:
    return (f'<w:p><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="5760000" cy="2880000"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{number}" name="Chart {number}"/><wp:cNvGraphicFramePr/>'
            f'<a:graphic xmlns:a="{A}"><a:graphicData uri="{C}"><c:chart xmlns:c="{C}" r:id="rIdChart{number}"/>'
            f'</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


def word(draft: Draft, title: str, now: datetime.datetime | None = None) -> bytes:
    """The checked draft as a .docx file."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    body, charts = [], []
    for block in draft.blocks:
        if isinstance(block, Heading):
            body.append(_para(block.parts, "Heading%d" % min(2, block.level)))
        elif isinstance(block, Paragraph):
            body.append(_paragraphs(block))
        elif isinstance(block, Table):
            body.append(_table(block))
        elif isinstance(block, Chart):
            charts.append(chart_xml(block))
            body.append(_chart_ref(len(charts)))
    section = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" '
               'w:left="1134" w:header="567" w:footer="567" w:gutter="0"/></w:sectPr>')
    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{W}" xmlns:r="{R}" '
                f'xmlns:wp="{WP}" xmlns:a="{A}"><w:body>{"".join(body)}{section}</w:body></w:document>')
    chart_rels = "".join(f'<Relationship Id="rIdChart{i}" Type="{R}/chart" Target="charts/chart{i}.xml"/>'
                         for i in range(1, len(charts) + 1))
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-'
            'package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.'
            'wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/docProps/core.xml" '
            'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            + "".join('<Override PartName="/word/charts/chart%d.xml" ContentType="application/vnd.openxmlformats-'
                      'officedocument.drawingml.chart+xml"/>' % i for i in range(1, len(charts) + 1)) + "</Types>"),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.'
            'openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.'
            'openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/'
            'core-properties" Target="docProps/core.xml"/></Relationships>'),
        "word/_rels/document.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.'
            f'openxmlformats.org/package/2006/relationships"><Relationship Id="rIdStyles" Type="{R}/styles" '
            f'Target="styles.xml"/>{chart_rels}</Relationships>'),
        "word/document.xml": document,
        "word/styles.xml": STYLES,
        "docProps/core.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.'
            'openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f'<dc:title>{_x(title)}</dc:title><dc:creator>{SOURCE}</dc:creator><cp:lastModifiedBy>{SOURCE}'
            f'</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{stamp}</dcterms:created>'
            f'<dcterms:modified xsi:type="dcterms:W3CDTF">{stamp}</dcterms:modified></cp:coreProperties>'),
    }
    for i, xml in enumerate(charts, 1):
        files["word/charts/chart%d.xml" % i] = xml
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            info = zipfile.ZipInfo(name, date_time=now.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, text.encode("utf-8"))
    return buf.getvalue()


# ===================================================================================================== PDF
# The standard fonts' widths, a thousandth of the size, for characters 32 to 126 (Adobe's AFM files)
_HELVETICA = [278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556, 556,
              556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556, 1015, 667, 667, 722, 722, 667, 611, 778, 722, 278,
              500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469,
              556, 333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500,
              278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584]
_BOLD = [278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278, 556, 556, 556, 556, 556, 556,
         556, 556, 556, 556, 333, 333, 584, 584, 584, 611, 975, 722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722,
         611, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333, 278, 333, 584, 556, 333, 556,
         611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611, 611, 611, 389, 556, 333, 611, 556, 778,
         556, 556, 500, 389, 280, 389, 584]
PAGE_W, PAGE_H, MARGIN = 595.0, 842.0, 50.0


def _pdf_text(text: str) -> str:
    """Text as the standard fonts can write it: the rupee sign as Rs, anything else they lack as '?'."""
    text = text.replace("₹", "Rs ").replace("–", "-").replace("—", "-")
    return text.encode("cp1252", "replace").decode("cp1252")


def width(text: str, size: float, bold: bool = False) -> float:
    table = _BOLD if bold else _HELVETICA
    return sum(table[ord(ch) - 32] if 32 <= ord(ch) <= 126 else 556 for ch in text) * size / 1000


def wrap(text: str, size: float, room: float, bold: bool = False) -> list[str]:
    out = []
    for line in _lines(text):
        words, now = line.split(" "), ""
        for w in words:
            trial = (now + " " + w) if now else w
            if width(trial, size, bold) <= room or not now:
                now = trial
            else:
                out.append(now)
                now = w
            while width(now, size, bold) > room and len(now) > 1:      # a word wider than the room
                cut = len(now)
                while cut > 1 and width(now[:cut], size, bold) > room:
                    cut -= 1
                out.append(now[:cut])
                now = now[cut:]
        out.append(now)
    return out


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class _Pages:
    def __init__(self):
        self.pages: list[list[str]] = [[]]
        self.y = PAGE_H - MARGIN

    @property
    def ops(self) -> list[str]:
        return self.pages[-1]

    def room(self, height: float) -> None:
        if self.y - height < MARGIN:
            self.pages.append([])
            self.y = PAGE_H - MARGIN

    def text(self, x: float, y: float, text: str, size: float, bold: bool = False, grey: bool = False) -> None:
        colour = "0.36 0.42 0.45 rg" if grey else "0.1 0.14 0.16 rg"
        self.ops.append("BT %s /%s %.1f Tf %.2f %.2f Td (%s) Tj ET" % (colour, "F2" if bold else "F1", size, x, y,
                                                                      _esc(text)))

    def rule(self, x1, y1, x2, y2, grey: float = 0.75, w: float = 0.5) -> None:
        self.ops.append("%.2f G %.2f w %.2f %.2f m %.2f %.2f l S" % (grey, w, x1, y1, x2, y2))

    def box(self, x, y, w, h, rgb) -> None:
        self.ops.append("%.3f %.3f %.3f rg %.2f %.2f %.2f %.2f re f" % (*rgb, x, y, w, h))


def _rgb(hexcode: str) -> tuple:
    return tuple(int(hexcode[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _pdf_paragraph(p: _Pages, text: str, size: float, bold=False, grey=False, after=6.0) -> None:
    lead = size * 1.3
    for line in wrap(_pdf_text(text), size, PAGE_W - 2 * MARGIN, bold):
        p.room(lead)
        p.y -= lead
        p.text(MARGIN, p.y + 2, line, size, bold, grey)
    p.y -= after


def _pdf_table(p: _Pages, block: Table) -> None:
    columns = len(block.header)
    room = PAGE_W - 2 * MARGIN
    first = room * (0.4 if columns > 2 else 0.6)
    rest = (room - first) / max(1, columns - 1)
    xs = [MARGIN] + [MARGIN + first + rest * i for i in range(columns - 1)]
    ws = [first] + [rest] * (columns - 1)
    size = 8.0
    for r, row in enumerate((block.header,) + tuple(block.rows)):
        cells = [wrap(_pdf_text(plain(cell)), size, ws[c] - 6, r == 0) for c, cell in enumerate(row)]
        height = max(len(c) for c in cells) * size * 1.25 + 4
        p.room(height)
        if r == 0:
            p.box(MARGIN, p.y - height, room, height, _rgb("EEF2F4"))
        for c, lines in enumerate(cells):
            for i, line in enumerate(lines):
                ty = p.y - 2 - size * 1.25 * (i + 1) + 2
                tx = xs[c] + 3 if c == 0 else xs[c] + ws[c] - 3 - width(line, size, r == 0)
                p.text(tx, ty, line, size, r == 0)
        p.y -= height
        p.rule(MARGIN, p.y, MARGIN + room, p.y, 0.8 if r else 0.2, 0.4)
    p.y -= 10


def _pdf_chart(p: _Pages, block: Chart) -> None:
    height, room = 190.0, PAGE_W - 2 * MARGIN
    p.room(height + 30)
    _pdf_paragraph(p, plain(block.title), 9, bold=True, after=2)
    top, bottom = p.y - 8, p.y - height + 24
    values = [f.value for _, figs in block.series for f in figs if f.value is not None]
    high, low = max(values + [0.0]), min(values + [0.0])
    span = (high - low) or 1.0
    zero = bottom + (0 - low) / span * (top - bottom)
    left = MARGIN + 6
    groups = max(1, len(block.categories))
    group_w = (room - 12) / groups
    bar_w = group_w * 0.7 / max(1, len(block.series))
    p.rule(left, zero, left + room - 12, zero, 0.3, 0.6)
    for g, cat in enumerate(block.categories):
        gx = left + g * group_w + group_w * 0.15
        for s, (_, figs) in enumerate(block.series):
            f = figs[g]
            if f.value is None:
                continue
            y = bottom + (f.value - low) / span * (top - bottom)
            p.box(gx + s * bar_w, min(y, zero), bar_w * 0.92, abs(y - zero), _rgb(COLOURS[s % len(COLOURS)]))
            label = _pdf_text(f.text)
            lw = width(label, 6.0)
            p.text(gx + s * bar_w + (bar_w * 0.92 - lw) / 2, max(y, zero) + 2, label, 6.0)
        name = _pdf_text(cat.text)
        for i, line in enumerate(wrap(name, 7.0, group_w - 4)[:2]):
            p.text(left + g * group_w + (group_w - width(line, 7.0)) / 2, bottom - 10 - 8 * i, line, 7.0, grey=True)
    lx = left
    for s, (name, _) in enumerate(block.series):
        p.box(lx, bottom - 30, 7, 7, _rgb(COLOURS[s % len(COLOURS)]))
        text = _pdf_text(plain(name))
        p.text(lx + 10, bottom - 29, text, 7.0)
        lx += 20 + width(text, 7.0)
    p.y = bottom - 40


def pdf(draft: Draft, title: str, now: datetime.datetime | None = None) -> bytes:
    """The checked draft as a PDF."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    p = _Pages()
    for block in draft.blocks:
        if isinstance(block, Heading):
            size = 17.0 if block.level == 1 else 11.5
            p.room(size * 3)
            p.y -= 6 if block.level == 1 else 8
            _pdf_paragraph(p, plain(block.parts), size, bold=True, after=3)
        elif isinstance(block, Paragraph):
            _pdf_paragraph(p, plain(block.parts), 8.0 if block.kind == "note" else 9.5, grey=block.kind == "note")
        elif isinstance(block, Table):
            _pdf_table(p, block)
        elif isinstance(block, Chart):
            _pdf_chart(p, block)
    total = len(p.pages)
    for n, ops in enumerate(p.pages, 1):
        footer = _pdf_text("%s   %s %d/%d" % (SOURCE, "page", n, total))
        ops.append("BT 0.5 0.5 0.5 rg /F1 7 Tf %.2f %.2f Td (%s) Tj ET" % (MARGIN, MARGIN / 2, _esc(footer)))
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"]
    kids = []
    for ops in p.pages:
        stream = "\n".join(ops).encode("cp1252", "replace")
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        objects.append("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Resources << /Font << /F1 3 0 R /F2 4 0 R "
                       ">> >> /Contents %d 0 R >>" % (PAGE_W, PAGE_H, len(objects)))
        kids.append(len(objects))
    objects[1] = "<< /Type /Pages /Kids [%s] /Count %d >>" % (" ".join("%d 0 R" % k for k in kids), len(kids))
    stamp = now.strftime("D:%Y%m%d%H%M%SZ")
    objects.append("<< /Title (%s) /Producer (%s) /Creator (%s) /CreationDate (%s) >>"
                   % (_esc(_pdf_text(title)), SOURCE, SOURCE, stamp))
    info = len(objects)
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(out.tell())
        body = obj if isinstance(obj, bytes) else obj.encode("cp1252", "replace")
        out.write(b"%d 0 obj\n" % i + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for off in offsets:
        out.write(b"%010d 00000 n \n" % off)
    out.write(b"trailer\n<< /Size %d /Root 1 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, info,
                                                                                               xref))
    return out.getvalue()


__all__ = ["SOURCE", "chart_xml", "pdf", "width", "word", "wrap"]
