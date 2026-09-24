"""No hardcoded number or date locale in the web app (PLAN.md 2.3a).

Figures on screen follow the account's locale and digit grouping
(web/src/lib/locale.ts, used by lib/format.ts), so a formatter pinned to a
language, or a number turned into text with `toFixed` or `toLocaleString`, is
a US-format assumption. This fails on:
  - a quoted language tag such as "en-US" passed to Intl or toLocaleString;
  - `.toLocaleString(` and `.toLocaleDateString(` (they ignore the account);
  - `.toFixed(`, unless it's rounding a number that stays a number:
    `Number(x.toFixed(6))`.
A line that really needs one ends with the comment ``locale-ok``, saying why.
lib/locale.ts, the one place that turns numbers into text, is exempt.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "src"
SKIP = {"web/src/lib/api/schema.d.ts", "web/src/lib/locale.ts"}

LOCALE_TAG = re.compile(r"""(["'`])[a-z]{2,3}-[A-Z]{2}\1""")
TO_LOCALE = re.compile(r"\.toLocale(?:Date|Time)?String\(")
TO_FIXED = re.compile(r"\.toFixed\(")
ROUNDED = re.compile(r"Number\([^()]*(?:\([^()]*\))?[^()]*\.toFixed\(\d+\)\)")


def locale_hits(line: str) -> bool:
    if "locale-ok" in line:
        return False
    if LOCALE_TAG.search(line) or TO_LOCALE.search(line):
        return True
    return bool(TO_FIXED.search(ROUNDED.sub("", line)))


def offenders() -> list[str]:
    found = []
    for path in WEB.rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix not in {".ts", ".tsx"} or rel in SKIP or "node_modules" in path.parts:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if locale_hits(line):
                found.append(f"{rel}:{n}: {line.strip()[:120]}")
    return found


def test_no_hardcoded_locale_in_the_web_app():
    found = offenders()
    assert not found, (
        "Numbers or dates formatted outside the account's locale. Use lib/format.ts "
        "(fmtMoney, fmtRate, fmtNumber, fmtCount ...) or lib/locale.ts:\n" + "\n".join(found))


CAUGHT = ["return v.toFixed(d)", 'new Intl.NumberFormat("en-US", {})', "n.toLocaleString('en-US')", "x.toLocaleString()",
          "d.toLocaleDateString()", "`${(v * 100).toFixed(1)}%`", "{v.toFixed(2)}"]
FINE = ["ticks.push(Number(t.toFixed(6)))", "Number(x.toFixed(6))", "Number((value + dir * spec.step).toFixed(6))",
        'new Intl.NumberFormat(style.locale, {})', 'const ok = "en";', "{fmtNumber(v, 2)}",
        'selectOption("de-DE") // locale-ok: a test value', "Number(seniorX.toFixed(6))"]


def test_the_check_catches_hardcoded_formats_and_passes_the_rest():
    assert [s for s in CAUGHT if not locale_hits(s)] == []
    assert [s for s in FINE if locale_hits(s)] == []
