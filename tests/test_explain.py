"""Plain-English explanations and the glossary (PLAN.md 6.4): the catalogue's side.

The browser test (``web/e2e/explain.spec.ts``) proves what a person sees: every
figure an explanation quotes is a figure the screen shows, character for
character, and every label on the explained tiles carries a glossary entry.
This test holds the catalogue to the rules that make that possible, so a
translator can't break them unseen:

  - an explanation's words hold **no digit**: every number reaches it as an
    argument, formatted by the tile that shows it, inside an ``<f>`` tag (the
    tag is what the screen marks as a quote);
  - the only tag in ``explain`` is ``<f>``, and every ``<f>`` holds exactly one
    argument, so a quote can never be a number the tile does not show;
  - every glossary id the components name exists, and so does every entry the
    screens look up at run time (a tranche kind, an engine label, a money unit);
  - the regional terms the task names are there (crores, lakhs, Ind AS,
    EBITDA after lease costs, the IFRS words for interest and net income).
"""
import json
import re
from pathlib import Path

from tests.test_translations import catalogue, sources

ROOT = Path(__file__).resolve().parent.parent
MESSAGES = ROOT / "web" / "messages" / "en.json"

#: `term="irr"` in JSX, `term: "irr"` in an object, `<Term id="irr">`
TERM_LITERAL = re.compile(r"""\bterm(?:=|:\s*)["'](\w+)["']|<Term\s+id=["'](\w+)["']""")
TAG = re.compile(r"</?([\w-]+)>")
QUOTE = re.compile(r"<f>(.*?)</f>")
ARGUMENT = re.compile(r"\{(\w+)\}")


def _namespace(name: str) -> dict[str, str]:
    return {k.split(".", 1)[1]: v for k, v in catalogue("en").items() if k.startswith(f"{name}.")}


def _camel(name: str) -> str:
    """The engine label's key, as web/src/lib/i18n/useEngineText.ts makes it."""
    return "".join(p[0].upper() + p[1:] for p in re.split(r"[^A-Za-z0-9]+", name) if p)


def test_the_two_namespaces_follow_regime():
    keys = list(json.loads(MESSAGES.read_text(encoding="utf-8")))
    at = keys.index("regime")
    assert keys[at + 1 : at + 3] == ["explain", "glossary"]


def test_an_explanation_holds_no_digit_of_its_own():
    for key, message in _namespace("explain").items():
        words = ARGUMENT.sub("", message)                      # an argument's name may hold a digit ({p5})
        assert not re.search(r"\d", words), f"explain.{key} writes a number into its words: {message!r}"


def test_every_quote_is_one_argument_and_f_is_the_only_tag():
    for key, message in _namespace("explain").items():
        tags = set(TAG.findall(message))
        assert tags <= {"f"}, f"explain.{key} uses tags {tags - {'f'}}; only <f> marks a quote"
        assert message.count("<f>") == message.count("</f>"), f"explain.{key}: unbalanced <f>"
        for inner in QUOTE.findall(message):
            assert re.fullmatch(r"\{\w+\}", inner), f"explain.{key}: <f>{inner}</f> must hold one argument and nothing else"


def test_every_glossary_entry_defines_its_term():
    glossary = _namespace("glossary")
    assert len(glossary) >= 80
    for key, text in glossary.items():
        assert len(text.split()) >= 6, f"glossary.{key} is too short to define anything: {text!r}"
        assert "{" not in text, f"glossary.{key} takes no argument: it is read on hover, with nothing to fill"


def test_every_term_a_component_names_is_in_the_glossary():
    glossary = _namespace("glossary")
    named: dict[str, str] = {}
    for path in sources():
        for a, b in TERM_LITERAL.findall(path.read_text(encoding="utf-8")):
            named.setdefault(a or b, path.relative_to(ROOT).as_posix())
    assert len(named) >= 60, "the explained tiles name their terms"
    missing = {term: where for term, where in named.items() if term not in glossary}
    assert not missing, f"glossary entries missing for {missing}"


def test_every_term_looked_up_at_run_time_is_in_the_glossary():
    """web/src/lib/glossary.ts: a facility's kind, the engine's tranche, bridge and driver labels, a money unit."""
    glossary = _namespace("glossary")
    wanted = [f"kind_{k}" for k in _namespace("trancheKinds")]
    for key in _namespace("engine"):
        for kind in ("tranche", "bridgeAxis", "driver"):
            if key.startswith(kind):
                wanted.append(f"{kind}_{key[len(kind):]}")
    wanted += [f"unit_{u}" for u in ("thousands", "lakhs", "millions", "crores", "billions")]
    missing = [w for w in wanted if w not in glossary]
    assert not missing, f"glossary entries missing for {missing}"
    assert _camel("Exit Multiple") == "ExitMultiple"


def test_the_regional_terms_are_defined():
    glossary = _namespace("glossary")
    for term in ("unit_crores", "unit_lakhs", "indAs", "ebitdaAfterLeases", "financeCosts", "profitForYear", "ifrs", "usGaap"):
        assert term in glossary, f"glossary.{term} is missing"
    assert "10 million" in glossary["unit_crores"] or "ten million" in glossary["unit_crores"]
    assert "100,000" in glossary["unit_lakhs"] or "hundred thousand" in glossary["unit_lakhs"]


def test_the_research_explanations_and_terms_hold_no_recommendation():
    """Phase R's rule (PLAN.md R6): the tool's words never name a rating or a call, the valuation's included."""
    from tests.test_research_valuation import calls_in

    explain = _namespace("explain")
    glossary = _namespace("glossary")
    research = [v for k, v in explain.items() if k.startswith(("value_", "model_"))]
    research += [glossary[k] for k in ("valueDcf", "valueMultiple", "priceTyped", "valuation", "sectorModel", "modelRow", "modelUnit", "indAs")]
    assert [text for text in research if calls_in(text)] == []
