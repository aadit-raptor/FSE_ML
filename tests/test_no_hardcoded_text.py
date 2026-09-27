"""No interface text outside the translation files (PLAN.md 2.3b).

Every word the app shows lives in ``web/messages/<language>.json`` and reaches
the screen through next-intl (``useTranslations``), so another language is a
new file rather than a hunt through the components. This fails on English
written straight into ``web/src``:

  - a JSX text node with a word of three letters or more:
    ``<span>Saved deals</span>``;
  - such a word in a quoted value of a prop people read --- ``title``,
    ``label``, ``aria-label``, ``placeholder``, ``caption`` and the rest of
    ``TEXT_PROPS``;
  - such a word in a quoted value of an object property people read:
    ``{ label: "EBITDA" }``, which is how the field and navigation tables
    used to hold their labels.

Symbols and units that are the same in every language pass: ``x``, ``%``,
``·``, ``→``, ``FY``, a lone letter. A line that really needs English ends
with the comment ``text-ok`` and a reason (as ``currency-ok`` and
``locale-ok`` do in the checks beside this one).

Excel sheet names, chart labels and screen-reader labels count as interface
text: they are read by people.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "src"
SKIP = {"web/src/lib/api/schema.d.ts"}

#: Props whose value a person reads (on screen, or through a screen reader).
#: `name` is left out: a `name` prop identifies a field (`<DealField name=`,
#: `<input name=`), it is not shown.
TEXT_PROPS = (
    "title|label|aria-label|aria-valuetext|placeholder|caption|alt|summary|"
    "sub|heading|hint|yearLabel|of|unit|message|description|note"
)
#: Object properties that used to carry labels; the same words, one nesting in.
#: `unit` is left out here: as a property it holds a FieldSpec's unit token
#: ("%", "x", "millions"), which is data, not a word on screen.
TEXT_KEYS = (
    "label|title|summary|caption|heading|hint|message|description|note|sub|"
    "placeholder|text|name"
)

# A word of three letters or more: what a translator would have to change.
WORD = re.compile(r"[A-Za-z]{3,}")
# >Saved deals</span>  (one line; nothing but text between the tags). The
# lookbehind skips `=>` and the lookahead skips `>= x <= y`, so arrow
# functions, comparisons and TypeScript generics are not mistaken for text.
JSX_TEXT = re.compile(r"(?<![=!<>+\-*/&|])>([^<>{}\n]*?)<(?=[/A-Za-z])")
PROP_VALUE = re.compile(rf"""\b(?:{TEXT_PROPS})\s*=\s*(?:\{{\s*)?(["'`])([^"'`]*)\1""")
KEY_VALUE = re.compile(rf"""\b(?:{TEXT_KEYS})\s*:\s*(["'`])([^"'`]*)\1""")

# Words that are code, not language: they read the same in every locale, or
# they name something in the API or the DOM rather than something on screen.
CODE_WORDS = {
    "true", "false", "null", "undefined", "none", "auto", "div", "span", "svg", "img",
    "url", "http", "https", "api", "utf", "json", "xlsx", "csv", "html", "css",
    "ltr", "rtl", "col", "row", "img", "text", "money", "percent", "multiple", "integer",
    "number", "date", "time", "page", "status", "alert", "note", "combobox", "listbox",
    "option", "radio", "radiogroup", "switch", "progressbar", "group", "navigation",
    "table", "columnheader", "rowheader", "cell", "form", "button", "link", "heading",
    "always", "path", "sans", "serif", "monospace", "normal", "bold", "medium", "light",
    "uppercase", "lowercase", "capitalize", "inherit", "currentColor", "transparent",
    "flex", "grid", "block", "inline", "hidden", "visible", "absolute", "relative",
    "fixed", "sticky", "static", "spring", "linear", "user", "reduce", "decimal",
    "numeric", "off", "one", "dev", "clerk", "sample", "edgar", "idle", "loading",
    "error", "success", "failed", "queued", "running", "succeeded", "cancelled",
    "created", "saved", "saving", "unsaved", "restored", "checking", "down", "ok",
    # Keys as they are printed on the keyboard, the same in every language
    "ctrl", "alt", "shift", "cmd", "esc", "tab", "enter",
}


def _quoted_hit(value: str) -> bool:
    """A quoted value that is English rather than a code word or a format."""
    words = [w.lower() for w in WORD.findall(value)]
    if not words:
        return False
    # Interpolation only, e.g. `${of} currency` in a template: the words still
    # count, so only values made entirely of code words pass.
    return any(w not in CODE_WORDS for w in words)


def text_hits(line: str) -> bool:
    if "text-ok" in line:
        return False
    stripped = line.strip()
    # Comments and the file's own documentation are for whoever reads the code
    if stripped.startswith(("//", "*", "/*")):
        return False
    for m in JSX_TEXT.finditer(line):
        if _quoted_hit(m.group(1)):
            return True
    for pattern in (PROP_VALUE, KEY_VALUE):
        for m in pattern.finditer(line):
            if _quoted_hit(m.group(2)):
                return True
    return False


def offenders() -> list[str]:
    found = []
    for path in sorted(WEB.rglob("*")):
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix not in {".ts", ".tsx"} or rel in SKIP or "node_modules" in path.parts:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if text_hits(line):
                found.append(f"{rel}:{n}: {line.strip()[:120]}")
    return found


def test_no_interface_text_outside_the_translation_files():
    found = offenders()
    assert not found, (
        "Interface text written into the components. Put the words in "
        "web/messages/en.json and read them with useTranslations (PLAN.md 2.3b):\n"
        + "\n".join(found)
    )


CAUGHT = [
    "<span>Saved deals</span>",
    '<Tile span={6} title="Sources" unit={mu}>',
    '  { key: "ebitda", label: "EBITDA", unit: MONEY },',
    '<DataTable caption="Income statement by year" columns={years} rows={rows} />',
    '<input aria-label="Deal name" value={name} />',
    '  <p className="type-body">No saved deals yet.</p>',
    '<MoneySelects money={money} onChange={setMoney} of="Deal" />',
    '      { slug: "inputs", label: "Deal inputs" },',
    '<button title={`Alt ${i + 1} opens this mode`}>',
    "aria-label={`${of} currency`}",
]
FINE = [
    "<span>{t('deal.saved')}</span>",
    '<Tile span={6} title={t("sources")} unit={mu}>',
    '  { key: "ebitda", unit: MONEY, step: 5 },',
    '<span className="chip text-accent">{t("open")}</span>',
    "<span>{fmtMoney(v)}</span>",
    '<td className="px-2 py-1 text-right">',
    "  1.00",
    '  <option key={c} value={c}>',
    '<span aria-hidden>·</span>',
    '  { unit: "%", step: 1, decimals: 1 },',
    '  { unit: "x", step: 0.5 },',
    '  { name: "money", label: "Money" }, // text-ok: Excel format names, not words',
    '<DealField name="ebitda" />',
    '  <kbd>Ctrl K</kbd>',
    '  attention: { bg: "bg-[#1b1710]", titleClass: "type-alert" },',
    '  export const DEFAULT_MONEY: Money = { currency: "USD", unit: "millions" };',
    "// A label like <span>Saved deals</span> in a comment is fine",
    '  role="radiogroup"',
    '<svg role="img" aria-label={t("chart.irr")}>',
    '  <Notice tone="loss" title={t("errors.notSaved")} role="alert">',
    '    <p style={{ fontFamily: "system-ui, sans-serif" }}>',
]


def test_the_check_catches_written_in_text_and_passes_the_rest():
    assert [s for s in CAUGHT if not text_hits(s)] == []
    assert [s for s in FINE if text_hits(s)] == []
