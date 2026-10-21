"""Reading a document for a deal (PLAN.md 6.2): the answer's shape, the checks, the deal it gives.

Pure logic of ``ml/nlp_extractor.py``; the job and the endpoints are tests/test_extractor_api.py.
No test calls an AI: the answers here are written by hand, the recorded ones are replayed.
"""
from __future__ import annotations

import copy

import pytest

from ml import nlp_extractor as nx

LATEST, PRIOR = "2025-12-31", "2024-12-31"


def label(value, page=3, quote=None):
    return {"value": value, "page": page, "quote": quote or f"in {value}"}


def figure(key, value, period=LATEST, page=40, quote=None):
    printed = f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return {"key": key, "period_end": period, "value": value, "page": page,
            "quote": quote if quote is not None else f"{key} {printed}"}


def answer(**changes):
    found = {
        "company": "Muster AG", "language": "de", "document": "annual_report",
        "labels": {"currency": label("EUR", quote="in Mio. EUR"), "unit": label("millions", quote="in Mio. EUR"),
                   "accounting_standard": label("ifrs", page=60, quote="nach IFRS, wie sie in der EU anzuwenden sind"),
                   "fiscal_year_end": label(LATEST, quote="31. Dezember 2025"),
                   "scope": label("consolidated", quote="Konzern-Gewinn- und Verlustrechnung")},
        "figures": [figure("revenue", 1627.9), figure("revenue", 2035.9, PRIOR),
                    figure("gross_profit", 700.0), figure("operating_income", 120.5),
                    figure("depreciation_amortization", 80.0, page=41), figure("profit_before_tax", 100.0),
                    figure("income_tax", 25.0), figure("net_income", 75.0),
                    figure("capital_expenditures", 60.0, page=44), figure("total_assets", 2500.0, page=42)],
    }
    found.update(changes)
    return found


def no_text(page, quote):
    return None                                 # a page with no text layer: nothing to look the quote up in


def assembled(a=None, b="same", *, on_page=no_text, poor=lambda page: False):
    read_a = nx.unwrap(a if a is not None else answer())
    read_b = nx.unwrap(copy.deepcopy(a if a is not None else answer())) if b == "same" else \
        (None if b is None else nx.unwrap(b))
    return nx.assemble(read_a, read_b, quote_on_page=on_page, poor=poor)


def item(result, key, period=LATEST):
    if key in nx.LABELS:
        return next(i for i in result["labels"] if i["key"] == key)
    return next(i for i in result["figures"] if i["key"] == key and i["period_end"] == period)


def every_id(result):
    return {i["id"] for i in result["labels"] + result["figures"]}


# ---------------------------------------------------------------- the answer

def test_an_answer_is_cleaned_to_the_keys_asked_for_and_anything_else_is_refused():
    a = answer()
    a["figures"].append({"key": "made_up", "period_end": LATEST, "value": 1, "page": 1, "quote": "x"})
    a["figures"].append({"key": "revenue", "period_end": "last year", "value": 9, "page": 1, "quote": "x"})
    a["figures"].append({"key": "net_income", "period_end": PRIOR, "value": "lots", "page": True, "quote": "x"})
    a["labels"]["unit"] = label("gazillions")
    a["labels"]["currency"] = label("euro")
    found = nx.unwrap([a])                           # a model may wrap it in a list
    assert {f["key"] for f in found["figures"]} <= set(nx.FIGURES)
    assert all(f["period_end"] in (LATEST, PRIOR) for f in found["figures"])
    assert found["labels"]["unit"]["value"] is None and found["labels"]["currency"]["value"] is None
    noisy = next(f for f in found["figures"] if f["key"] == "net_income" and f["period_end"] == PRIOR)
    assert noisy["value"] is None and noisy["page"] is None
    for bad in ("text", [], [answer(), answer()], {"labels": {}}):
        with pytest.raises(nx.Unreadable):
            nx.unwrap(bad)


def test_quotes_are_cut_to_size_and_printable_and_a_repeated_figure_keeps_the_first():
    a = answer()
    a["figures"][0]["quote"] = "Umsatz\x00 " + "1.627,9 " * 100
    a["figures"].append(figure("revenue", 9999.0))
    found = nx.unwrap(a)
    revenue = [f for f in found["figures"] if f["key"] == "revenue" and f["period_end"] == LATEST]
    assert len(revenue) == 1 and revenue[0]["value"] == 1627.9
    assert len(revenue[0]["quote"]) <= nx.MAX_QUOTE and "\x00" not in revenue[0]["quote"]


def test_read_b_is_shown_the_pages_read_a_cites_figures_first_and_gets_its_pages_back():
    found = nx.unwrap(answer())
    assert nx.cited_pages(found) == [40, 41, 42, 44, 3, 60]
    b = nx.unwrap(answer(figures=[figure("revenue", 1627.9, page=1), figure("total_assets", 2500.0, page=3)],
                         labels={"currency": label("EUR", page=5), "unit": label("millions", page=9)}))
    back = nx.picture_pages(b, [40, 41, 42, 44, 3])
    assert [f["page"] for f in back["figures"]] == [40, 42]
    assert back["labels"]["currency"]["page"] == 3 and back["labels"]["unit"]["page"] is None


def test_the_prompt_asks_for_any_language_quotes_verbatim_and_pdf_pages():
    text = nx.prompt_text()
    for words in ("any language", "verbatim", "1-based page of the PDF", "consolidated"):
        assert words in text
    for key in nx.FIGURES + nx.LABELS:
        assert f'"{key}"' in text
    assert nx.prompt_text(pictures=True).startswith("These are pictures")


# ---------------------------------------------------------------- the checks

def test_two_reads_that_agree_on_a_digital_page_leave_nothing_amber():
    result = assembled(on_page=lambda page, quote: True)
    assert result["summary"]["amber"] == 0
    assert all(i["amber"] is None for i in result["labels"] + result["figures"])
    assert item(result, "revenue")["id"] == f"figure:revenue:{LATEST}"
    assert item(result, "currency")["id"] == "label:currency"


def test_a_figure_the_second_read_gives_differently_is_amber_with_its_value():
    b = answer()
    b["figures"][0] = figure("revenue", 1672.9)
    result = assembled(b=b)
    assert item(result, "revenue")["amber"] == "reads_differ"
    assert item(result, "revenue")["other"] == 1672.9
    assert item(result, "revenue", PRIOR)["amber"] is None


def test_a_figure_or_label_only_one_read_gives_is_amber():
    b = answer()
    del b["figures"][-1]
    b["labels"]["accounting_standard"] = label(None)
    result = assembled(b=b)
    assert item(result, "total_assets")["amber"] == "one_read_only"
    assert item(result, "accounting_standard")["amber"] == "one_read_only"
    alone = assembled(b=None)
    assert all(i["amber"] == "one_read_only" for i in alone["figures"])


def test_labels_the_reads_disagree_on_are_amber():
    b = answer()
    b["labels"]["unit"] = label("thousands")
    assert item(assembled(b=b), "unit")["amber"] == "reads_differ"


def test_a_quote_not_on_its_page_is_amber_and_so_is_a_number_its_quote_does_not_hold():
    seen = []

    def on_page(page, quote):
        seen.append(page)
        return "1.627,9" not in quote                # every quote is on its page but revenue's

    result = assembled(on_page=on_page)
    assert item(result, "revenue")["amber"] == "quote_not_on_page"
    assert 40 in seen
    a = answer()
    a["figures"][2]["quote"] = "Bruttoergebnis 7.000,0"
    assert item(assembled(a=a, b="same"), "gross_profit")["amber"] == "number_not_in_quote"


def test_a_number_is_found_in_its_quote_however_the_page_groups_and_signs_it():
    for value, quote in ((1627.9, "Umsatzerlöse 1.627,9"), (-1234.0, "Ergebnis (1,234)"), (1170008, "1 170 008"),
                         (12.3, "12.30"), (5.0, "5"), (163990.0, "１６３，９９０")):
        assert nx.number_in_quote(value, quote), (value, quote)
    for value, quote in ((1627.9, "Umsatzerlöse 1.672,9"), (5.0, "Seite 12"), (12.3, "")):
        assert not nx.number_in_quote(value, quote), (value, quote)


def test_a_quote_is_matched_to_its_page_whatever_spaces_and_full_width_forms_it_holds():
    text = "売上高　　１，６７１，８６５\n営業利益 ３７２，３１６"
    assert nx.quote_on_text("売上高 1,671,865", text)
    assert nx.quote_on_text("Revenue  1,627.9", "REVENUE\n1,627.9 2,035.9")
    assert not nx.quote_on_text("売上高 1,671,866", text)


def test_a_figure_on_a_poor_scan_is_amber_even_when_both_reads_agree():
    result = assembled(poor=lambda page: page == 41)
    assert item(result, "depreciation_amortization")["amber"] == "poor_scan"
    assert item(result, "revenue")["amber"] is None


# ---------------------------------------------------------------- the deal

def test_nothing_enters_a_deal_that_was_not_accepted():
    result = assembled()
    assert nx.deal_inputs(result, set())["inputs"] == {}
    only_revenue = nx.deal_inputs(result, {f"figure:revenue:{LATEST}", f"figure:revenue:{PRIOR}"})
    assert set(only_revenue["inputs"]) == {"growth"}
    assert any(o["input"] == "ebitda" for o in only_revenue["left_out"])


def test_accepted_figures_give_the_deals_inputs_by_the_engines_definitions():
    result = assembled()
    patch = nx.deal_inputs(result, every_id(result))
    got = patch["inputs"]
    assert got["currency"] == "EUR" and got["unit"] == "millions"
    assert got["ebitda"] == pytest.approx(120.5 + 80.0)                    # operating income + D&A
    assert got["gross_margin"] == pytest.approx(700 / 1627.9 * 100)
    assert got["opex"] == pytest.approx((700 - 120.5) / 1627.9 * 100)        # EBIT = gross profit - opex
    assert got["da"] == pytest.approx(80 / 1627.9 * 100)
    assert got["capex"] == pytest.approx(60 / 1627.9 * 100)
    assert got["tax"] == pytest.approx(25.0)
    assert got["growth"] == pytest.approx((1627.9 / 2035.9 - 1) * 100)
    assert got["accounting_standard"] == "ifrs"
    assert got["fiscal_year_end_month"] == 12 and got["first_fiscal_year"] == 2026
    assert patch["used"]["ebitda"] == [f"figure:operating_income:{LATEST}",
                                       f"figure:depreciation_amortization:{LATEST}", "label:currency", "label:unit"]


def test_money_needs_its_currency_and_unit_accepted_and_ratios_do_not():
    result = assembled()
    accepted = every_id(result) - {"label:currency"}
    patch = nx.deal_inputs(result, accepted)
    assert "ebitda" not in patch["inputs"] and "currency" not in patch["inputs"]
    assert "gross_margin" in patch["inputs"]
    left = next(o for o in patch["left_out"] if o["input"] == "ebitda")
    assert "label:currency" in left["needs"]


def test_a_printed_ebitda_is_taken_as_printed_and_figures_in_units_become_thousands():
    a = answer()
    a["labels"]["unit"] = label("units")
    a["figures"].append(figure("ebitda", 250_000.0))
    result = assembled(a=a)
    got = nx.deal_inputs(result, every_id(result))["inputs"]
    assert got["unit"] == "thousands" and got["ebitda"] == pytest.approx(250.0)


def test_a_standard_the_deal_does_not_model_is_a_note_not_an_input():
    a = answer()
    a["labels"]["accounting_standard"] = label("ind_as")
    a["labels"]["fiscal_year_end"] = label("2025-03-31")
    result = assembled(a=a)
    patch = nx.deal_inputs(result, every_id(result))
    assert "accounting_standard" not in patch["inputs"] and "standard_not_in_deal" in patch["notes"]
    assert patch["inputs"]["fiscal_year_end_month"] == 3 and patch["inputs"]["first_fiscal_year"] == 2026


def test_figures_a_deal_cannot_take_are_left_out_not_bent_to_fit():
    a = answer()
    a["figures"] = [f for f in a["figures"] if f["key"] != "gross_profit"]
    a["figures"].append(figure("ebitda", -40.0))
    result = assembled(a=a)
    patch = nx.deal_inputs(result, every_id(result))
    assert "gross_margin" not in patch["inputs"] and "opex" not in patch["inputs"]
    assert "ebitda" not in patch["inputs"] and "ebitda_not_positive" in patch["notes"]


def test_an_id_that_names_nothing_in_the_read_is_ignored():
    result = assembled()
    patch = nx.deal_inputs(result, {"figure:revenue:1999-12-31", "label:nonsense", "x"})
    assert patch["inputs"] == {}


def test_an_input_the_document_cannot_give_is_not_listed_as_waiting_for_acceptance():
    a = answer()
    a["figures"] = [f for f in a["figures"] if f["key"] != "capital_expenditures"]
    result = assembled(a=a)
    patch = nx.deal_inputs(result, set())
    assert "capex" not in {o["input"] for o in patch["left_out"]}
    assert {"gross_margin", "da", "tax"} <= {o["input"] for o in patch["left_out"]}


def test_a_tax_printed_as_a_negative_expense_gives_a_positive_rate():
    a = answer()
    a["figures"] = [f for f in a["figures"] if f["key"] != "income_tax"] + [figure("income_tax", -25.0)]
    result = assembled(a=a)
    assert nx.deal_inputs(result, every_id(result))["inputs"]["tax"] == pytest.approx(25.0)


# ---------------------------------------------------------------- the recorder and the card

class _Recorded:
    """A provider that answers with a recording's reads and counts what it is asked."""
    needs_pictures = True

    def __init__(self, record, fail_b=None):
        self.record, self.asked, self.fail_b = record, [], fail_b

    def read(self, which, document, *, exclude=frozenset(), instructions=None):
        from research.reader.providers import Answer

        self.asked.append((which, len(document.pictures), instructions.version if instructions else None))
        if which == "B" and self.fail_b:
            raise self.fail_b
        found = self.record["reads"][which]
        return Answer(found["read"], found["model"], "x1", found["usage"], 1.0)


def _pdf_with_text(path, pages, lines):
    """A PDF whose page ``pages`` holds the given lines as real text, the others blank."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    for number in range(1, pages[1] + 1):
        page = writer.add_blank_page(width=595, height=842)
        if number == pages[0]:
            body = "BT /F1 10 Tf 40 800 Td 14 TL " + " ".join(f"({line}) '" for line in lines) + " ET"
            stream = DecodedStreamObject()
            stream.set_data(body.encode("latin-1"))
            page[NameObject("/Contents")] = writer._add_object(stream)
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)})})
    with open(path, "wb") as out:
        writer.write(out)


def test_the_recorder_saves_each_answer_keeps_quotes_only_as_hashes_and_carries_on(tmp_path):
    from ml import document_reader as dr
    from research.reader.providers import SPENT, ProviderError

    source = {"reads": {
        "A": {"model": "gemini-3.8-flash", "usage": {"promptTokenCount": 10}, "read": answer(
            figures=[figure("revenue", 1627.9, page=2, quote="Umsatzerlöse 1.627,9"),
                     figure("net_income", 75.0, page=2, quote="Ergebnis 7.500,0")],
            labels={"currency": label("EUR", page=2, quote="in Mio. EUR")})},
        "B": {"model": "gemini-3.7-flash", "usage": {"promptTokenCount": 5}, "read": answer(
            figures=[figure("revenue", 1627.9, page=1, quote="Umsatzerlöse 1.627,9")],
            labels={"currency": label("EUR", page=1, quote="in Mio. EUR")})}}}
    pdf = tmp_path / "muster.pdf"
    _pdf_with_text(pdf, (2, 3), ["Konzern-Gewinn- und Verlustrechnung in Mio. EUR", "Umsatzerl\xf6se 1.627,9",
                                 "Ergebnis nach Steuern 75,0"])
    target = tmp_path / "evaluation" / "muster.json"
    stopped = _Recorded(source, fail_b=ProviderError(SPENT))
    with pytest.raises(ProviderError):
        dr.record_one(stopped, pdf, target, "evaluation")
    saved = __import__("json").loads(target.read_text(encoding="utf-8"))
    assert set(saved["reads"]) == {"A"} and saved["prompt"] == nx.PROMPT_VERSION      # A kept for tomorrow
    again = _Recorded(source)
    assert dr.record_one(again, pdf, target, "evaluation") == "A by gemini-3.8-flash, B by gemini-3.7-flash"
    assert again.asked == [("B", 1, nx.PROMPT_VERSION)]                                  # only B asked again
    saved = __import__("json").loads(target.read_text(encoding="utf-8"))
    assert saved["document"]["text_pages"] == [2]
    text = target.read_text(encoding="utf-8")
    assert "Konzern-Gewinn" not in text and "Ergebnis nach Steuern" not in text           # no page text kept
    on_page = dr.recorded_lookup(saved)
    assert on_page(2, "Umsatzerlöse 1.627,9") is True and on_page(2, "Ergebnis 7.500,0") is False
    assert on_page(3, "anything") is None                                                  # a page with no text
    a, b = dr.recorded_reads(saved)
    result = nx.assemble(a, b, quote_on_page=on_page, poor=lambda page: False)
    assert item(result, "revenue")["amber"] is None
    assert item(result, "net_income")["amber"] == "one_read_only"


def test_the_recorder_command_stops_for_the_day_when_every_model_is_spent(tmp_path, monkeypatch, capsys):
    from ml import document_reader as dr
    from research.reader.providers import SPENT, ProviderError

    class Spent:
        def __init__(self, key):
            assert key == "K"

        def read(self, *args, **kwargs):
            raise ProviderError(SPENT)

    monkeypatch.setattr(dr.ai, "Gemini", Spent)
    monkeypatch.setattr(dr, "EXTRACTIONS", tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "K")
    pdf = tmp_path / "x.pdf"
    _pdf_with_text(pdf, (1, 1), ["nothing"])
    assert dr.main(["record", str(pdf), str(pdf)]) == 0
    said = capsys.readouterr().out
    assert said.count("allowance_spent") == 1 and "run the same command again tomorrow" in said


def test_a_recording_made_with_another_prompt_is_left_as_it_is(tmp_path):
    from ml import document_reader as dr

    target = tmp_path / "old.json"
    target.write_text('{"document": {"name": "old"}, "prompt": "x0", "reads": {}}', encoding="utf-8")
    assert dr.record_one(None, tmp_path / "missing.pdf", target, "evaluation") == \
        "recorded with prompt x0, not x1: left as it is"
