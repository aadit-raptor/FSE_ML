"""A results PDF's pages: how many, how well scanned, and pictures of them.

The file is an upload (PLAN.md 6.1): it was scanned clean, but it is still
someone else's bytes, so it is opened **only in a child process** with a
CPU and memory limit and an environment holding no secret, as the upload
scanner does (``uploads/scan.py`` ``scan_path``). The child is
``python -m research.reader.pages <command> <pdf> [...]`` and answers one
line of JSON on its standard output:

- ``info``: ``{"pages": n, "dpi": [lowest picture resolution per page, or
  null for a page with no picture]}``. A scan below ``POOR_DPI`` is a poor
  scan: the reader marks its figures amber (R0 found the misread digits on
  faint, dense pages).
- ``render``: draws the pages asked for at ``DPI`` (R0's read B used 170)
  as PNG files in a folder the parent made, and answers their names.
- ``text``: the text layer of the pages asked for (``{"texts": {page: str}}``),
  each cut to ``MAX_TEXT`` characters: how a document's quotes are checked
  against the page they cite (PLAN.md 6.2, ``ml/nlp_extractor.py``).

``info`` takes an optional page limit (``MAX_PAGES`` for a results filing; an
annual report read for a deal may be longer, ``ml/document_reader.py``).

PDFium (``pypdfium2``) draws the pages; the PNG files are written here with
``zlib`` so no imaging library is needed.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

DPI = 170                 # R0's read B: enough for small digits, 1-3 MB a page
POOR_DPI = 100            # below this a scanned page's digits are guesswork
MAX_PAGES = 300           # a filing this long is not a results filing (R0's longest: 193)
MAX_PICTURES = 12         # pages sent as pictures in one request (R0's limit)
CHILD_TIMEOUT_S = 90.0
CHILD_CPU_S = 60
CHILD_MEMORY_BYTES = 700 * 1024 * 1024    # as the upload scanner's: the free server has 512 MB in all
# The most pixels one page is drawn with: an A4 page at 170 dpi is 4 million.
# A page whose declared size is huge is drawn smaller rather than filling memory
MAX_PIXELS = 12_000_000
MAX_TEXT = 20_000          # characters of one page's text layer (a dense statement page holds about 6,000)


class PagesError(RuntimeError):
    """The child could not read the file, or ran out of time or memory."""


@dataclass(frozen=True)
class Info:
    pages: int
    dpi: tuple[Optional[int], ...]

    def poor(self, page: int) -> bool:
        """Whether a 1-based page is a scan below ``POOR_DPI``."""
        if not 1 <= page <= len(self.dpi):
            return False
        found = self.dpi[page - 1]
        return found is not None and found < POOR_DPI


def _child(args: Sequence[str], timeout: float = CHILD_TIMEOUT_S) -> dict:
    root = Path(__file__).resolve().parents[2]
    environment = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(root),
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        done = subprocess.run([sys.executable, "-m", "research.reader.pages", *args],
                              stdin=subprocess.DEVNULL, capture_output=True, cwd=root, env=environment,
                              timeout=timeout, check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise PagesError(type(exc).__name__) from None
    if done.returncode != 0:
        raise PagesError(f"exit {done.returncode}")
    try:
        answer = json.loads(done.stdout.decode("utf-8"))
    except ValueError:
        raise PagesError("unreadable answer") from None
    if not isinstance(answer, dict):
        raise PagesError("unreadable answer")
    return answer


def info(path: Path, limit: int = MAX_PAGES) -> Info:
    answer = _child(["info", str(path), str(limit)])
    pages, dpi = answer.get("pages"), answer.get("dpi")
    if not isinstance(pages, int) or not isinstance(dpi, list) or len(dpi) != pages:
        raise PagesError("unreadable answer")
    return Info(pages, tuple(d if isinstance(d, int) else None for d in dpi))


def render(path: Path, pages: Sequence[int]) -> list[bytes]:
    """PNG pictures of the 1-based ``pages`` (at most ``MAX_PICTURES``), in order."""
    wanted = [p for p in pages if isinstance(p, int) and p >= 1][:MAX_PICTURES]
    if not wanted:
        return []
    with tempfile.TemporaryDirectory(prefix="fse-pages-") as folder:
        answer = _child(["render", str(path), folder, ",".join(map(str, wanted))],
                        timeout=CHILD_TIMEOUT_S * 2)
        names = answer.get("files")
        if not isinstance(names, list) or len(names) != len(wanted):
            raise PagesError("unreadable answer")
        out = []
        for name in names:
            target = Path(folder) / Path(str(name)).name      # never outside the folder
            out.append(target.read_bytes())
        return out


def texts(path: Path, pages: Sequence[int]) -> dict[int, str]:
    """The text layer of the 1-based ``pages`` (an empty string for a page with none)."""
    wanted = sorted({p for p in pages if isinstance(p, int) and p >= 1})
    if not wanted:
        return {}
    answer = _child(["text", str(path), ",".join(map(str, wanted))])
    found = answer.get("texts")
    if not isinstance(found, dict):
        raise PagesError("unreadable answer")
    return {int(k): str(v)[:MAX_TEXT] for k, v in found.items() if str(k).isdigit() and int(k) in wanted}


# ---------------------------------------------------------------------------
# The child
# ---------------------------------------------------------------------------
def png(pixels, width: int, height: int, channels: int) -> bytes:
    """A PNG file from rows of 8-bit pixels (grey or RGB)."""
    colour = {1: 0, 3: 2}[channels]
    stride = width * channels
    raw = b"".join(b"\x00" + bytes(pixels[y * stride:(y + 1) * stride]) for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, colour, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) \
        + chunk(b"IEND", b"")


def _page_dpi(page) -> Optional[int]:
    """The lowest resolution of the large pictures on a page (a scan is one
    picture covering it), or None when the page has none."""
    import pypdfium2.raw as raw

    width_in, height_in = page.get_width() / 72.0, page.get_height() / 72.0
    lowest = None
    for item in page.get_objects(filter=(raw.FPDF_PAGEOBJ_IMAGE,), max_depth=2):
        left, bottom, right, top = item.get_pos()
        shown_w, shown_h = (right - left) / 72.0, (top - bottom) / 72.0
        # a logo or a stamp says nothing about how the figures were scanned
        if shown_w < width_in * 0.5 or shown_h < height_in * 0.3:
            continue
        size = item.get_size()
        dpi = int(min(size[0] / shown_w, size[1] / shown_h))
        lowest = dpi if lowest is None else min(lowest, dpi)
    return lowest


def _limit_this_process() -> None:
    import resource

    limits = [(resource.RLIMIT_CPU, CHILD_CPU_S), (resource.RLIMIT_CORE, 0)]
    if sys.platform.startswith("linux"):       # macOS does not enforce an address-space limit
        limits.append((resource.RLIMIT_AS, CHILD_MEMORY_BYTES))
    for which, value in limits:
        try:
            resource.setrlimit(which, (value, value))
        except (ValueError, OSError):
            pass


def main(argv: list[str]) -> int:
    """The child's work. The limits are set where the child starts (below),
    never here: a test calling this in-process must not limit its runner."""
    import pypdfium2 as pdfium

    if len(argv) < 2 or argv[0] not in ("info", "render", "text"):
        return 2
    try:
        document = pdfium.PdfDocument(argv[1])
    except Exception:  # noqa: BLE001 -- any failure to open is "unreadable", nothing more
        return 3
    count = len(document)
    limit = int(argv[2]) if argv[0] == "info" and len(argv) > 2 and argv[2].isdigit() else MAX_PAGES
    if argv[0] == "info" and count > limit:
        print(json.dumps({"pages": count, "dpi": [None] * count}))
        return 0
    if argv[0] == "info":
        print(json.dumps({"pages": count, "dpi": [_page_dpi(document[i]) for i in range(count)]}))
        return 0
    if argv[0] == "text":
        found = {}
        for number in [int(p) for p in argv[2].split(",") if p.isdigit()]:
            if 1 <= number <= count:
                found[str(number)] = document[number - 1].get_textpage().get_text_range()[:MAX_TEXT]
        print(json.dumps({"texts": found}, ensure_ascii=False))
        return 0
    folder, wanted = Path(argv[2]), [int(p) for p in argv[3].split(",") if p]
    names = []
    for number in wanted:
        if not 1 <= number <= count:
            return 4
        page = document[number - 1]
        scale = DPI / 72.0
        area = page.get_width() * page.get_height() * scale * scale
        if area > MAX_PIXELS:
            scale *= (MAX_PIXELS / area) ** 0.5
        bitmap = page.render(scale=scale)
        array = bitmap.to_numpy()
        height, width = array.shape[0], array.shape[1]
        channels = array.shape[2] if array.ndim == 3 else 1
        if channels == 4:                       # BGRx: drop the padding, put red first
            array = array[:, :, [2, 1, 0]]
            channels = 3
        elif channels == 3:                     # BGR
            array = array[:, :, ::-1]
        name = f"page-{number}.png"
        (folder / name).write_bytes(png(array.tobytes(), width, height, channels))
        names.append(name)
    print(json.dumps({"files": names}))
    return 0


if __name__ == "__main__":
    _limit_this_process()
    sys.exit(main(sys.argv[1:]))
