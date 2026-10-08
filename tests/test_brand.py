"""The logo files match the mark they are drawn from.

web/src/components/brand/mark.json is the one source of the Variater mark;
web/scripts/brand.mjs draws every icon and image from it. These tests fail
when a file is missing, has the wrong size, or still shows an older mark
(mark.json changed and the script wasn't rerun).
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
MARK = json.loads((WEB / "src/components/brand/mark.json").read_text(encoding="utf-8"))
APP = WEB / "src/app"
BRAND = WEB / "public/brand"

PNG_SIZES = {
    APP / "apple-icon.png": (180, 180),
    APP / "opengraph-image.png": (1200, 630),
    BRAND / "icon-192.png": (192, 192),
    BRAND / "icon-512.png": (512, 512),
    BRAND / "icon-maskable-512.png": (512, 512),
    BRAND / "logo-120.png": (120, 120),
    BRAND / "avatar.png": (400, 400),
    BRAND / "avatar-cyan.png": (400, 400),
}

LOGOS = [
    "logo-horizontal-dark.png",
    "logo-horizontal-light.png",
    "logo-stacked-dark.png",
    "logo-stacked-light.png",
    "logo-black.png",
    "logo-white.png",
    "logo-email.png",
]


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    return struct.unpack(">II", data[16:24])


def rects(svg: str) -> list[dict[str, str]]:
    return [dict(re.findall(r'([\w-]+)="([^"]*)"', r)) for r in re.findall(r"<rect\b[^>]*>", svg)]


@pytest.mark.parametrize("tone", ["dark", "light", "black", "white"])
def test_each_mark_svg_draws_the_bars_in_its_tones_colours(tone):
    drawn = rects((BRAND / f"mark-{tone}.svg").read_text(encoding="utf-8"))
    colours = MARK["tones"][tone]
    assert len(drawn) == len(MARK["bars"])
    for bar, rect in zip(MARK["bars"], drawn):
        outlined = bar["role"] == "dip" and colours.get("dipOutline")
        inset = 1.5 if outlined else 0
        assert float(rect["x"]) == bar["x"] + inset
        assert float(rect["y"]) == bar["y"] + inset
        assert float(rect["width"]) == bar["w"] - 2 * inset
        assert float(rect["height"]) == bar["h"] - 2 * inset
        colour = rect["stroke"] if outlined else rect["fill"]
        assert colour == colours[bar["role"]]


def test_the_browser_tab_icon_is_the_mark_and_follows_the_browser_theme():
    svg = (APP / "icon.svg").read_text(encoding="utf-8")
    drawn = rects(svg)
    side = MARK["width"] + 2
    top = (side - MARK["height"]) / 2
    assert [(float(r["x"]) - 1, float(r["y"]) - top, float(r["width"]), float(r["height"])) for r in drawn] == [
        (b["x"], b["y"], b["w"], b["h"]) for b in MARK["bars"]
    ]
    # Dark bars on a light tab, light bars on a dark one
    light, dark = svg.split("@media (prefers-color-scheme: dark)")
    for role in ("start", "dip", "rise", "end"):
        assert f".{role}{{fill:{MARK['tones']['light'][role]}}}" in light
        assert f".{role}{{fill:{MARK['tones']['dark'][role]}}}" in dark


@pytest.mark.parametrize("path, size", list(PNG_SIZES.items()), ids=lambda v: getattr(v, "name", str(v)))
def test_each_icon_has_the_size_its_platform_asks_for(path, size):
    assert png_size(path) == size


@pytest.mark.parametrize("name", LOGOS)
def test_each_logo_with_the_name_is_wide_enough_to_print(name):
    width, height = png_size(BRAND / name)
    assert width >= 400 and height >= 100


def test_favicon_ico_holds_16_32_and_48():
    data = (APP / "favicon.ico").read_bytes()
    reserved, kind, count = struct.unpack("<HHH", data[:6])
    assert (reserved, kind) == (0, 1)
    sizes = [data[6 + 16 * i] for i in range(count)]
    assert sizes == [16, 32, 48]
    for i in range(count):
        length, offset = struct.unpack("<II", data[6 + 16 * i + 8 : 6 + 16 * i + 16])
        assert data[offset : offset + 8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", data[offset + 16 : offset + 24]) == (sizes[i], sizes[i])
        # RGBA (colour type 6): Next.js's build refuses an .ico with RGB PNGs
        assert data[offset + 25] == 6, f"{sizes[i]} px image is not RGBA"


def test_the_manifest_names_icons_that_exist():
    manifest = (APP / "manifest.ts").read_text(encoding="utf-8")
    sources = re.findall(r'src: "(/brand/[^"]+)"', manifest)
    assert sources
    for src in sources:
        assert (WEB / "public" / src.lstrip("/")).is_file(), src


def test_the_link_preview_has_alt_text_naming_the_product():
    alt = (APP / "opengraph-image.alt.txt").read_text(encoding="utf-8")
    messages = json.loads((WEB / "messages/en.json").read_text(encoding="utf-8"))
    assert alt.startswith(messages["app"]["brand"])
