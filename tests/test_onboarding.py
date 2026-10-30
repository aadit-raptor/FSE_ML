"""Onboarding and in-app help (PLAN.md 7.1), the parts a machine can check without a browser.

- **Every input has a tip.** The shared inputs (``NumberField``, ``CellInput``, ``Switch``,
  ``MoneySelects``, ``FiscalSelects``) take a required ``tip`` key, so TypeScript finds a use without
  one; this test finds the rest: every ``<input>``, ``<select>`` and ``<textarea>`` written straight
  into a component, and every ``role="switch"`` button, must point ``aria-describedby`` at its tip.
  Every tip key written in the code, and every key of the field tables whose tips are looked up at
  run time (the deal's ``FIELDS``, the Settings tables, the simulation's ``SIM_FIELDS``), must have
  its words in ``web/messages/en.json`` under ``tips``. ``web/e2e/tips.spec.ts`` checks the same on
  the screens, where the tip's text is read through the input's description.
- **Sample deals** come from several regions, each a country the economic data covers, an industry
  the industry tables hold and the country's own currency.
- **Help articles** are ``docs/methodology.md``'s sections, written by ``python -m ops.help_articles``
  into ``web/src/lib/help/articles.json``; the committed file must be what the document gives today.
- **What's new** lists dated entries, newest first, each with its words in the catalogue.
"""
import json
import re
from datetime import date
from pathlib import Path

from economy.catalogue import AREAS, EURO_MEMBERS
from library.base_rates import sp_region
from ops import help_articles

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "src"
MESSAGES = json.loads((ROOT / "web" / "messages" / "en.json").read_text(encoding="utf-8"))
TIPS = MESSAGES.get("tips", {})
ONBOARDING = json.loads((WEB / "lib" / "onboarding" / "data.json").read_text(encoding="utf-8"))
WHATS_NEW = json.loads((WEB / "lib" / "help" / "whats-new.json").read_text(encoding="utf-8"))
INDUSTRIES = {i["id"] for i in json.loads((ROOT / "web" / "e2e" / "fixtures" / "benchmarks.json").read_text())["industries"]["industries"]}

#: The opening of a form control written straight into JSX, up to the end of its tag.
CONTROL = re.compile(r"<(input|select|textarea)\b")
SWITCH_ROLE = re.compile(r"""role=["']switch["']""")
#: A tip key written in the code: tip="deal_hold", tip={"x"}, useTip("x"), tipKey: "x"
TIP_LITERAL = re.compile(r"""(?:\btip\s*=\s*\{?\s*|\buseTip\(\s*|\btip(?:Key)?\s*:\s*)["']([A-Za-z0-9_]+)["']""")
#: The keys of a useTips(["a", "b"]) call
TIPS_LIST = re.compile(r"""\buseTips\(\s*\[([^\]]*)\]""")


def components() -> list[Path]:
    return [p for p in sorted(WEB.rglob("*.tsx")) if "node_modules" not in p.parts]


def tag_at(text: str, start: int) -> str:
    """The JSX opening tag starting at ``start``: up to the first ``>`` outside braces."""
    depth = 0
    for i in range(start + 1, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == ">" and depth == 0:
            return text[start : i + 1]
    return text[start:]


def untipped_controls(text: str) -> list[str]:
    found = []
    for m in CONTROL.finditer(text):
        tag = tag_at(text, m.start())
        if 'type="hidden"' in tag or "aria-describedby" in tag:
            continue
        found.append(" ".join(tag.split())[:140])
    for m in SWITCH_ROLE.finditer(text):
        # The button holding the role: back to its "<button"
        start = text.rfind("<button", 0, m.start())
        tag = tag_at(text, start)
        if "aria-describedby" not in tag:
            found.append(" ".join(tag.split())[:140])
    return found


def test_every_form_control_written_into_a_component_points_at_its_tip():
    missing = [
        f"{p.relative_to(ROOT).as_posix()}: {tag}" for p in components() for tag in untipped_controls(p.read_text(encoding="utf-8"))
    ]
    assert not missing, (
        "A form control without its tip (PLAN.md 7.1). Give it `aria-describedby={tip.id}` from "
        "`useTip(...)` and show the tip with `<TipLabel tip={tip}>` (components/ui/Tip.tsx):\n" + "\n".join(missing)
    )


def test_the_control_check_catches_a_control_without_a_tip():
    assert untipped_controls('<select value={v} onChange={(e) => go(e.target.value)} className="x">')
    assert untipped_controls('<button type="button" role="switch" aria-checked={on}>')
    assert not untipped_controls('<select aria-describedby={tip.id} value={v} onChange={(e) => go(e.target.value)}>')
    assert not untipped_controls('<input type="hidden" name="x" />')


def tip_keys(text: str) -> list[str]:
    keys = TIP_LITERAL.findall(text)
    for m in TIPS_LIST.finditer(text):
        keys += re.findall(r"""["']([A-Za-z0-9_]+)["']""", m.group(1))
    return keys


def test_every_tip_key_written_in_the_code_has_its_words():
    used = {(p.relative_to(ROOT).as_posix(), k) for p in components() for k in tip_keys(p.read_text(encoding="utf-8"))}
    used |= {(p.relative_to(ROOT).as_posix(), k) for p in sorted(WEB.rglob("*.ts")) for k in tip_keys(p.read_text(encoding="utf-8"))}
    assert len(used) > 40, "the tip check found almost no tips: has the way tips are written changed?"
    missing = sorted(f"{f}: tips.{k}" for f, k in used if k not in TIPS)
    assert not missing, "Tips with no words in web/messages/en.json:\n" + "\n".join(missing)


def object_keys(path: Path, const: str) -> list[str]:
    """The keys of a TypeScript object literal ``export const NAME ... = { key: ..., }`` (one per line)."""
    text = path.read_text(encoding="utf-8")
    body = text[text.index(f"const {const}") :]
    body = body[body.index("{", body.index("=")) :]
    depth, end = 0, 0
    for i, c in enumerate(body):
        depth += c == "{"
        depth -= c == "}"
        if depth == 0:
            end = i
            break
    return re.findall(r"^\s{2}(\w+):", body[:end], re.M)


def test_every_deal_field_has_a_tip():
    keys = object_keys(WEB / "lib" / "deal" / "fields.ts", "FIELDS")
    assert "ebitda" in keys and "business_risk" in keys
    assert not [k for k in keys if f"field_{k}" not in TIPS], [k for k in keys if f"field_{k}" not in TIPS]


def test_every_simulation_field_has_a_tip():
    keys = object_keys(WEB / "components" / "montecarlo" / "MonteCarloProvider.tsx", "SIM_FIELDS")
    assert "n" in keys and "hurdle" in keys
    assert not [k for k in keys if f"mc_{k}" not in TIPS], [k for k in keys if f"mc_{k}" not in TIPS]


def test_every_setting_has_a_tip():
    text = (WEB / "components" / "settings" / "SettingsSteps.tsx").read_text(encoding="utf-8")
    keys = set(re.findall(r'\bkey: "(\w+)"', text)) | set(re.findall(r'\["(\w+)", "(?:×|pts)"\]', text))
    keys |= set(re.findall(r'floor: "(\w+)"', text))
    assert {"def_ebitda", "tx_fee_pct", "bull_growth_mult", "rec_growth_floor"} <= keys
    assert not sorted(k for k in keys if f"settings_{k}" not in TIPS), sorted(k for k in keys if f"settings_{k}" not in TIPS)


def test_every_facility_actual_and_forecast_row_has_a_tip():
    families = {
        "tranche_": object_keys(WEB / "components" / "deal" / "steps" / "TrancheList.tsx", "SPECS"),
        "actual_": re.findall(r'key: "(\w+)"', (WEB / "lib" / "backtest" / "actuals.ts").read_text().split("EXIT_FIELDS")[0]),
        "actualExit_": re.findall(r'key: "(\w+)"', (WEB / "lib" / "backtest" / "actuals.ts").read_text().split("EXIT_FIELDS", 1)[1]),
        "forecast_": re.findall(r'"(\w+)"', "".join(re.findall(r"rows: \[([^\]]*)\]", (WEB / "lib" / "forecast.ts").read_text()))),
    }
    assert all(len(keys) >= 5 for keys in families.values()), families
    missing = [f"{prefix}{k}" for prefix, keys in families.items() for k in keys if f"{prefix}{k}" not in TIPS]
    assert not missing, missing


def test_no_tip_is_empty_or_repeats_its_label():
    fields = MESSAGES["fields"]
    for key, text in TIPS.items():
        assert len(text.split()) >= 5, f"tips.{key} is too short to help: {text!r}"
        if key.startswith("field_") and key[6:] in fields:
            assert text.strip().rstrip(".").lower() != fields[key[6:]].lower(), f"tips.{key} only repeats the label"


# --- Sample deals and the guided deal's currency ---


def test_sample_deals_come_from_several_regions_in_their_own_currency():
    samples = ONBOARDING["samples"]
    assert len(samples) >= 5
    assert len({sp_region(s["country"]) for s in samples}) >= 4, "samples from at least four of S&P's regions"
    for s in samples:
        assert s["country"] in AREAS, s
        assert s["industry"] in INDUSTRIES, s
        assert s["currency"] == AREAS[s["country"]].currency, s
        assert s["unit"] in {"thousands", "lakhs", "millions", "crores", "billions"}, s
        assert s["ebitda"] > 0, s
        assert f"sample_{s['id']}" in MESSAGES["onboarding"], f"onboarding.sample_{s['id']} names it"
        assert f"sample_{s['id']}Note" in MESSAGES["onboarding"], f"onboarding.sample_{s['id']}Note describes it"


def test_the_guided_deal_offers_each_country_its_own_currency():
    expected = {code: a.currency for code, a in AREAS.items() if code != "XM"}
    expected |= {code: "EUR" for code in EURO_MEMBERS}
    assert ONBOARDING["currencies"] == expected


# --- Help articles and what's new ---


def test_help_articles_are_the_methodology_as_it_is_today():
    committed = json.loads(help_articles.OUT.read_text(encoding="utf-8"))
    assert committed == help_articles.build(), "web/src/lib/help/articles.json is stale: run `python -m ops.help_articles`"
    index = json.loads(help_articles.INDEX.read_text(encoding="utf-8"))
    assert index == help_articles.index(committed), "web/src/lib/help/index.json is stale: run `python -m ops.help_articles`"


def test_every_methodology_section_is_an_article_and_every_link_leads_somewhere():
    articles = help_articles.build()["articles"]
    slugs = {a["slug"] for a in articles}
    headings = [line[3:].strip() for line in help_articles.SOURCE.read_text(encoding="utf-8").splitlines() if line.startswith("## ")]
    assert len(articles) == len([h for h in headings if h not in help_articles.LEFT_OUT])
    assert {"the-deal-model", "monte-carlo-simulation", "tax-rules"} <= slugs
    assert not [s for s in slugs if f"summary_{s}" not in MESSAGES["help"]], "each article has its summary in help.summary_<slug>"

    def inlines(blocks):
        for b in blocks:
            for part in (b.get("inlines") or []) + [i for item in b.get("items", []) for i in item] + [
                i for row in b.get("rows", []) + ([b["header"]] if b.get("header") else []) for cell in row for i in cell
            ]:
                yield part

    for a in articles:
        assert a["blocks"], a["slug"]
        for part in inlines(a["blocks"]):
            if part.get("article"):
                assert part["article"] in slugs, (a["slug"], part)
            if part.get("href"):
                assert part["href"].startswith("https://"), (a["slug"], part)


def test_an_article_keeps_the_methodologys_formulas_and_tables():
    articles = {a["slug"]: a for a in help_articles.build()["articles"]}
    deal = articles["the-deal-model"]
    kinds = {b["type"] for b in deal["blocks"]}
    assert {"heading", "paragraph", "code"} <= kinds
    assert any("entry EV" in b.get("text", "") for b in deal["blocks"] if b["type"] == "code")
    assert any(b["type"] == "table" for a in articles.values() for b in a["blocks"])


def test_whats_new_is_dated_newest_first_and_worded():
    entries = WHATS_NEW["entries"]
    assert len(entries) >= 5
    dates = [date.fromisoformat(e["date"]) for e in entries]
    assert dates == sorted(dates, reverse=True)
    assert len({e["id"] for e in entries}) == len(entries)
    for e in entries:
        assert f"{e['id']}Title" in MESSAGES["whatsNew"], e
        assert f"{e['id']}Body" in MESSAGES["whatsNew"], e
        assert e["href"].startswith("/"), e
        assert not re.search(r"PLAN\.md|\bR\d\b", MESSAGES["whatsNew"][f"{e['id']}Body"]), "written for people, not for the plan"
