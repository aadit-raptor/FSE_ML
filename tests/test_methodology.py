"""docs/methodology.md covers every calculation (PLAN.md 3.2).

"Done when: every function affecting a number is covered." These tests make
that a rule the build keeps rather than a promise the document made once:

- every function and method in the model packages, and in the modules behind
  the ML panels, is named in the document (a function that affects no number,
  such as a console printer, is still named, in the appendix that says so);
- every link in the document points at a file that exists;
- every module of the model packages is linked at least once;
- the constants the document quotes are the code's own values.

A new function in ``core/`` therefore fails the build until someone writes down
what it computes.
"""
import ast
import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "methodology.md"

# Every number the app shows comes out of these.
MODEL_PACKAGES = ("core", "lbo_engine", "simulation", "analytics")
# Modules outside them that compute a number the app shows.
EXTRA_MODULES = (
    "api/serialize.py",              # histogram, percentile curve
    "ml/anomaly_detector.py",        # risk score
    "ml/surrogate/predict.py",       # Live sliders
    "ml/surrogate/generate_data.py",
    "ml/macro_regime.py",            # macro regime
    "ml/edgar_extractor.py",         # SEC EDGAR figures
    "benchmarks/starting.py",        # a new deal's starting figures
    "benchmarks/risk.py",            # the sourced Monte Carlo ranges (PLAN.md 4.4)
    "benchmarks/history.py",         # the history they are measured on
    "library/base_rates.py",         # published default and recovery rates (PLAN.md 4.5)
    "library/coverage.py",           # the reference library's coverage counts
    "library/references.py",         # reference transactions and their rules (PLAN.md 4.5b)
    "library/fees.py",               # fees and amortisation from them
)
# Only object plumbing: they validate or expand inputs, never compute a figure.
SKIPPED_METHODS = {"__init__", "__post_init__", "__repr__", "forward"}


def model_files():
    files = [p for pkg in MODEL_PACKAGES for p in sorted((ROOT / pkg).glob("*.py"))]
    return files + [ROOT / m for m in EXTRA_MODULES]


def defined_names(path: Path):
    """Top-level functions and class methods, as ``name`` and ``Class.method``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name not in SKIPPED_METHODS):
                    yield f"{node.name}.{item.name}"


def doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


def code_spans(text: str) -> set:
    return set(re.findall(r"`([^`\n]+)`", text))


def test_the_methodology_exists():
    assert DOC.is_file(), "docs/methodology.md is missing (PLAN.md 3.2)"


def test_every_function_that_can_affect_a_number_is_named():
    spans = code_spans(doc_text())
    # A span may carry a call or a qualifier: `run_lbo(params)`, `core.deal.run_deal`.
    named = set()
    for s in spans:
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", s):
            named.add(token)
            parts = token.split(".")
            named.update(parts)
            named.update(".".join(parts[i:i + 2]) for i in range(len(parts) - 1))
    missing = []
    for path in model_files():
        for name in defined_names(path):
            if name not in named:
                missing.append(f"{path.relative_to(ROOT).as_posix()}: {name}")
    assert not missing, ("docs/methodology.md does not name these functions; "
                         "describe what each computes (or list it in the appendix "
                         "of functions that affect no number):\n  " + "\n  ".join(missing))


def links(text: str):
    return re.findall(r"\]\(([^)\s]+)\)", text)


def test_every_link_points_at_a_file_that_exists():
    broken = []
    for target in links(doc_text()):
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            continue
        path = (DOC.parent / target.split("#")[0]).resolve()
        if not path.exists():
            broken.append(target)
    assert not broken, f"broken links in docs/methodology.md: {broken}"


def test_every_model_module_is_linked():
    linked = {(DOC.parent / t.split("#")[0]).resolve() for t in links(doc_text())
              if not re.match(r"[a-z]+:", t)}
    unlinked = [p.relative_to(ROOT).as_posix() for p in model_files()
                if p.name != "__init__.py" and p.resolve() not in linked]
    assert not unlinked, f"link these modules from docs/methodology.md: {unlinked}"


def quoted_constants(text: str):
    """Rows of the constants table: | `module.NAME` | value |"""
    return re.findall(r"^\|\s*`([a-z_.]+)\.([A-Z][A-Z0-9_]*)`\s*\|\s*`?([^|`]+?)`?\s*\|",
                      text, flags=re.M)


def test_the_document_quotes_constants():
    assert len(quoted_constants(doc_text())) >= 10


@pytest.mark.parametrize("module,name,value", quoted_constants(
    DOC.read_text(encoding="utf-8") if DOC.is_file() else ""))
def test_each_quoted_constant_is_the_codes_value(module, name, value):
    actual = getattr(importlib.import_module(module), name)
    if isinstance(actual, range):
        actual = len(actual)
    assert float(value) == pytest.approx(float(actual)), (
        f"docs/methodology.md says {module}.{name} is {value}; the code says {actual}")


def test_the_settings_defaults_quoted_are_the_codes():
    from core.config import DEFAULTS
    rows = re.findall(r"^\|\s*`([a-z_]+)`\s*\|\s*([-0-9.]+)\s*\|", doc_text(), flags=re.M)
    quoted = {k: v for k, v in rows if k in DEFAULTS}
    assert len(quoted) >= 20, "the Settings table should list the defaults"
    wrong = {k: (v, DEFAULTS[k]) for k, v in quoted.items() if float(v) != float(DEFAULTS[k])}
    assert not wrong, f"Settings defaults in docs/methodology.md differ from core/config.py: {wrong}"
