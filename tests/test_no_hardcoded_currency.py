"""No hardcoded dollar signs (PLAN.md 2.2).

Every money figure is shown in its deal's currency and unit (core/money.py,
web/src/lib/money.ts), so a dollar sign used as currency in the code is a US
assumption. This scans the application code and fails on one: "$5", "$M",
"$bn", "$ " and a lone "$" string, a literal "$" before a JavaScript template
value (`$${v}`) and a Python f-string's "${". Regex anchors, template
interpolation and SQL dollar quoting are not currency and pass. A line that
really needs a dollar sign can end with the comment ``currency-ok``.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["api", "core", "lbo_engine", "simulation", "analytics", "jobs", "db", "ml", "ops", "web/src"]
SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".css", ".json", ".md"}
SKIP = {"web/src/lib/api/schema.d.ts"}   # generated from the API's own descriptions

CURRENCY = re.compile(
    r"\$(?=\d|\s|(?:M|B|K|k|bn|mm|MM)\b)"   # $5  $ 5  $M  $bn
    r"|(?P<q>[\"'`])\$(?P=q)"                # "$"
    r"|US\$"
    r"|\$\$\{")                               # `$${value}`: a dollar sign, then a template value
PY_FSTRING_DOLLAR = re.compile(r"\$\{")      # f"${x}": in Python "${" is a dollar sign, not interpolation


def currency_dollars(line: str, python: bool = False) -> list[int]:
    if "currency-ok" in line:
        return []
    hits = [m.start() for m in CURRENCY.finditer(line)]
    if python:
        hits += [m.start() for m in PY_FSTRING_DOLLAR.finditer(line)]
    return hits


def offenders() -> list[str]:
    found = []
    for source in SOURCES:
        base = ROOT / source
        if not base.exists():
            continue
        for path in base.rglob("*"):
            rel = path.relative_to(ROOT).as_posix()
            if path.suffix not in SUFFIXES or rel in SKIP or "node_modules" in path.parts:
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if currency_dollars(line, python=path.suffix == ".py"):
                    found.append(f"{rel}:{n}: {line.strip()[:120]}")
    return found


def test_no_hardcoded_dollar_signs():
    found = offenders()
    assert not found, (
        "Hardcoded dollar signs. Show money with its deal's currency and unit "
        "(web: useMoney() or moneyLabel(); Python: a Money in the response):\n" + "\n".join(found))


CURRENCY_LINES = ['sub="$M"', "label: 'Value ($M)'", "costs $5", "`$${v}`", "US$ 3", "fees in $bn",
                  'return "$" + v', "a $ 5 fee", 'unit="by tranche, $M"']
PYTHON_CURRENCY_LINES = ['f"${x:,.0f}M"', "print(f'  ${x}')"]
FINE_LINES = ['r"^[A-Z]{3}$"', "/^[1-9]$/.test(k)", "`${a} ${b}`", "DO $$", "$$",
              'rstrip("$")  # currency-ok', 'decoded.replace(/\\$$/, "")', "$env:DATABASE_URL",
              "x = 1  # $5 currency-ok", r"(\d{4})\)$/)", "${message}", "`${fmtMoney(v)} ${mu}`"]


def test_the_check_catches_currency_and_passes_everything_else():
    assert [s for s in CURRENCY_LINES if not currency_dollars(s)] == []
    assert [s for s in PYTHON_CURRENCY_LINES if not currency_dollars(s, python=True)] == []
    assert [s for s in FINE_LINES if currency_dollars(s)] == []
