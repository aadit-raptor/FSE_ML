"""Every message the web app asks for exists, and every message is used (PLAN.md 2.3b).

``tests/test_no_hardcoded_text.py`` stops English being written into the
components; this is the other half. A ``t("kpiIrr")`` whose key is missing
renders the key itself, which no test looking at figures would notice, so:

  - every key a component asks for is in ``web/messages/en.json``;
  - every key in the catalogue is asked for somewhere, so a screen's rewrite
    doesn't leave dead English behind;
  - every language file has exactly the keys English has, and the same
    placeholders in each message, so a translation can't silently drop a
    figure (PLAN.md 7.8 adds the languages).

Keys built at run time (``t(SAVE_KEY[state])``, ``t(spec.unit)``) can't be read
statically. The module that owns them declares them in a comment,
``i18n-keys: nav.*``, ``i18n-keys: settings.def_*`` or
``i18n-keys: deal.kindAuto, deal.kindSaved``. A token ending in ``*`` covers
every key that starts with what comes before it; a bare ``*`` is refused, so a
declaration wrapped onto a second comment line can't turn into "everything".
Each line of a comment repeats ``i18n-keys:``.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "src"
MESSAGES = ROOT / "web" / "messages"

#: `const t = useTranslations("deal")`, and the same for any other name.
SCOPE = re.compile(r"""\b(?:const|let)\s+(\w+)\s*=\s*useTranslations\(\s*["'](\w[\w.]*)["']\s*\)""")
#: `t("kpiIrr")`, `t.rich("note")`, `t.has("trancheMezzanine")`
CALL = re.compile(r"""\b(\w+)(?:\.(?:rich|has|markup))?\(\s*["']([\w.\-]+)["']""")
#: `i18n-keys: fields.*` or `i18n-keys: deal.kindAuto, deal.kindSaved`, in a comment
DECLARED = re.compile(r"i18n-keys:[ 	]*([\w.*\-]+(?:[ 	]*,[ 	]*[\w.*\-]+)*)")

#: Message keys read from the catalogue outside a component (page titles).
DIRECT = re.compile(r"DEFAULT_MESSAGES\.(\w+)\.(\w+)|messages\.(\w+)\.(\w+)")


def catalogue(language: str) -> dict[str, str]:
    """Flat "namespace.key" -> message for one language file."""
    data = json.loads((MESSAGES / f"{language}.json").read_text(encoding="utf-8"))
    flat: dict[str, str] = {}

    def walk(node: dict, prefix: str) -> None:
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, f"{prefix}{key}.")
            else:
                flat[f"{prefix}{key}"] = value

    walk(data, "")
    return flat


def languages() -> list[str]:
    return sorted(p.stem for p in MESSAGES.glob("*.json"))


def sources() -> list[Path]:
    return [p for p in sorted(WEB.rglob("*")) if p.suffix in {".ts", ".tsx"} and p.name != "schema.d.ts"]


def requested() -> tuple[set[str], set[str], list[str]]:
    """(exact keys asked for, declared wildcard prefixes, where each came from)."""
    exact: set[str] = set()
    prefixes: set[str] = set()
    where: list[str] = []
    for path in sources():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT).as_posix()
        scopes = {name: ns for name, ns in SCOPE.findall(text)}
        # `nav(mode.labelKey)` and friends: the key is a variable, so only the
        # declared ones below can be checked
        for name, key in CALL.findall(text):
            if name not in scopes:
                continue
            full = f"{scopes[name]}.{key}"
            exact.add(full)
            where.append(f"{rel}: {full}")
        for group in DECLARED.findall(text):
            for token in (t.strip() for t in group.split(",")):
                assert token != "*", f"{rel}: `i18n-keys: *` would cover every message; name a prefix"
                if token.endswith("*"):
                    prefixes.add(token[:-1])
                elif token:
                    exact.add(token)
        for a, b, c, d in DIRECT.findall(text):
            exact.add(f"{a or c}.{b or d}")
    return exact, prefixes, where


def covered(key: str, exact: set[str], prefixes: set[str]) -> bool:
    return key in exact or any(key.startswith(p) for p in prefixes)


def test_every_key_the_web_app_asks_for_exists():
    exact, _, where = requested()
    known = set(catalogue("en"))
    missing = sorted({f for f in exact if f not in known})
    assert not missing, (
        "Messages asked for but not in web/messages/en.json:\n"
        + "\n".join(f"  {k}   ({next((w for w in where if w.endswith(k)), '?')})" for k in missing)
    )


def test_every_message_is_used():
    exact, prefixes, _ = requested()
    unused = sorted(k for k in catalogue("en") if not covered(k, exact, prefixes))
    assert not unused, (
        "Messages in web/messages/en.json that no screen asks for. Remove them, "
        "or declare a run-time key with an `i18n-keys:` comment (see this test's docstring):\n"
        + "\n".join(f"  {k}" for k in unused)
    )


PLACEHOLDER = re.compile(r"\{(\w+)")


def test_every_language_has_the_same_keys_and_placeholders():
    english = catalogue("en")
    for language in languages():
        if language == "en":
            continue
        other = catalogue(language)
        assert set(other) == set(english), (
            f"web/messages/{language}.json doesn't have English's keys: "
            f"missing {sorted(set(english) - set(other))}, extra {sorted(set(other) - set(english))}"
        )
        for key, text in english.items():
            assert set(PLACEHOLDER.findall(other[key])) == set(PLACEHOLDER.findall(text)), (
                f"web/messages/{language}.json: {key} has different placeholders from English "
                f"({other[key]!r} vs {text!r})"
            )


# Labels the engine itself produces and the web app translates on the way in
# (web/src/lib/i18n/useEngineText.ts). Tranche names come from
# lbo_engine/capital_structure.py, the driver columns from
# core/montecarlo.py's `empirical_correlations` and `driver_sensitivity`.
ENGINE_TRANCHES = ["Senior Term Loan", "Mezzanine", "Senior Notes"]
ENGINE_DRIVERS = ["IRR", "MOIC", "Growth", "Exit Multiple", "Interest", "Gross Margin", "EBITDA Shock"]


def camel(name: str) -> str:
    """The key `useEngineText.ts` builds from an engine label."""
    return "".join(part[0].upper() + part[1:] for part in re.split(r"[^A-Za-z0-9]+", name) if part)


def engine_bridge_labels() -> tuple[list[str], list[str]]:
    """(axis labels, table labels) for every step core/deal.py can produce."""
    from core.deal import bridge_steps

    bridge = {
        "entry_equity": 100.0,
        # Non-zero, so the optional "Fees" step is included too
        "entry_costs": -5.0,
        "ebitda_growth": 20.0,
        "multiple_expansion": 10.0,
        "deleveraging": 30.0,
        "exit_equity": 155.0,
    }
    steps = bridge_steps(bridge)
    return [axis for axis, *_ in steps], [label for _, label, *_ in steps]


def test_every_label_the_engine_produces_has_a_key():
    known = set(catalogue("en"))
    axis, rows = engine_bridge_labels()
    expected = (
        [f"engine.tranche{camel(n)}" for n in ENGINE_TRANCHES]
        + [f"engine.driver{camel(n)}" for n in ENGINE_DRIVERS]
        + [f"engine.bridgeAxis{camel(n)}" for n in axis]
        + [f"engine.bridgeRow{camel(n)}" for n in rows]
    )
    missing = sorted(k for k in expected if k not in known)
    assert not missing, (
        "The engine produces labels the catalogue can't translate, so they would "
        "show in English whatever the account's language (web/messages/en.json, "
        "`engine`):\n" + "\n".join(f"  {k}" for k in missing)
    )
