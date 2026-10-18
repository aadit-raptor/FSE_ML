"""A sector model put together from its lines of revenue (PLAN.md R4): the
analyst's last changes -- add a line, drop one, swap its block -- never leave
a formula naming a row that is gone, and model 1 built from blocks is the
model solved by hand, row for row.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from research import blocks, compose
from research.definition import ASSUME, FIGURE, Company, Heading, InputError, ModelError, cells, label
from research.evaluate import EvaluationError, evaluate
from research.models import CATALOGUE, general, occupancy
from research.workbook import layout, write
from tests import research_support as support

GENERAL = general.DEFAULT
MODEL1 = occupancy.DEFAULT


def rendered(model, words):
    d = model.to_dict()
    out = []
    for i, r in enumerate(model.rows):
        if isinstance(r, Heading):
            out.append({"heading": label(r.label, words)})
        else:
            out.append({**d["rows"][i], "label": label(r.label, words)})
    return out


def test_model1_built_from_blocks_is_the_model_solved_by_hand_row_for_row():
    """`model1_definition.json` is model 1 as it stood before the blocks were shared: every row, rule, label
    (with each sector's words), constant and note is still the same."""
    before = json.loads((support.FIXTURES / "model1_definition.json").read_text(encoding="utf-8"))
    model = occupancy.MODEL
    assert model.to_dict()["periods"] == before["periods"]
    assert list(model.notes) == before["notes"]
    for sector, words in model.sectors.items():
        assert {k: label(v, words) for k, v in model.constants.items()} == before["constants"][sector]
        assert rendered(model, words) == before["rows"][sector], sector


def every_assumption(model, company_values=None, value=None):
    """A company with every typed cell filled: past figures and assumptions alike, by unit."""
    plan = cells(model, Company("x", "Rs lakh", sector=next(iter(model.sectors)), constants={"unit_rupees": 100000},
                                choices={c: sorted(o)[0] for c, o in model.choices().items()}), allow_missing=True)
    pick = {"share": 0.07, "rupees": 2500.0, "count": 40.0, "money": 900.0, "days": 0.0, "count3": 12.0}
    units = {r.key: r.unit for r in model.data_rows()}
    values = {}
    for (row, period), cell in plan.items():
        if cell.typed is None:
            continue
        v = {"ly": 365, "ytd": 183, "roy": 182}.get(period, 365) if units[row] == "days" else pick[units[row]]
        values.setdefault(row, {})[period] = v * (1.1 if period == "ytd" else 1.0)
    values["days"]["ytd"] = 183
    return values


ALL_BLOCKS = compose.Structure("occupancy", (
    compose.Segment("s1", "capacity"), compose.Segment("s2", "units"), compose.Segment("s3", "price_volume"),
    compose.Segment("s4", "subscribers"), compose.Segment("s5", "people"), compose.Segment("s6", "order_book"),
    compose.Segment("s7", "growth"), compose.Segment("s8", "share", of="s1", rest_of_revenue=True),
), places="s1", forecast_years=3)


def all_blocks():
    model = CATALOGUE.build(ALL_BLOCKS)
    choices = {c: sorted(o)[0] for c, o in model.choices().items()}
    company = Company("All blocks", "Rs lakh", sector="hotels", constants={"unit_rupees": 100000}, choices=choices,
                      values=every_assumption(model))
    return model, company


def test_one_model_can_hold_all_eight_blocks_and_works_out_whole():
    model, company = all_blocks()
    values = evaluate(model, company).values
    assert {s.block for s in ALL_BLOCKS.segments} == set(blocks.BLOCKS)
    revenue = sum(values["s%d_rev" % i]["y3"] for i in range(1, 9))
    assert values["rev"]["y3"] == pytest.approx(revenue)
    for period in ("cy", "y1", "y2", "y3"):              # the filler's opening balance sheet is out by its own gap,
        assert values["chk_bs"][period] == pytest.approx(values["chk_bs"]["ly"])     # and the cash flow adds none


def test_the_workbook_of_all_eight_blocks_gives_the_servers_figures():
    model, company = all_blocks()
    where, values = layout(model), evaluate(model, company).values
    get = support.workbook_values(write(model, company))
    wrong = [(r.key, p.name) for r in model.data_rows() for p in model.periods
             if (values[r.key][p.name] is None) != (get(where.cell(r.key, p.name)) is None)
             or (values[r.key][p.name] is not None and abs(values[r.key][p.name] - get(where.cell(r.key, p.name))) > 0.01)]
    assert wrong == []


if support.soffice_wanted():
    def test_libreoffice_works_all_eight_blocks_out_to_the_servers_figures():
        model, company = all_blocks()
        where, values = layout(model), evaluate(model, company).values
        sheet = support.recalculated_by_libreoffice(write(model, company))["Model"]
        wrong = [(r.key, p.name, values[r.key][p.name], sheet[where.cell(r.key, p.name)].value)
                 for r in model.data_rows() for p in model.periods
                 if values[r.key][p.name] is not None and abs(values[r.key][p.name] - (sheet[where.cell(r.key, p.name)].value or 0)) > 0.01]
        assert wrong == []


@pytest.mark.parametrize("block", blocks.BLOCKS)
@pytest.mark.parametrize("where", ["s1", "s2"])
def test_every_block_in_every_place_builds_and_works_out(block, where):
    first = compose.Segment("s1", "growth") if where == "s2" else None
    seg = compose.Segment(where, block, of="s1" if block == "share" else None)
    if block == "share" and where == "s1":
        seg = compose.Segment("s1", "share", formula="days * 0 + 100")
    structure = compose.normalise(compose.Structure("general", tuple(s for s in (first, seg) if s), "s1"))
    model = CATALOGUE.build(structure)
    company = Company("x", "Rs lakh", sector="pharma", constants={"unit_rupees": 100000},
                      choices={c: sorted(o)[0] for c, o in model.choices().items()}, values=every_assumption(model))
    values = evaluate(model, company).values
    assert values["rev"]["y2"] is not None
    assert values["chk_bs"]["y2"] == pytest.approx(values["chk_bs"]["ly"])


def test_add_drop_and_swap_never_break_a_formula():
    structure = MODEL1
    for block in blocks.BLOCKS:
        structure = compose.add_segment(structure, block) if len(structure.segments) < 8 else structure
        CATALOGUE.build(structure)
    assert len(structure.segments) == 8
    with pytest.raises(ModelError, match="at most 8"):
        compose.add_segment(structure, "growth")
    for key in ("s1", "s3", "s2"):                               # drop the line others follow, then more
        structure = compose.drop_segment(structure, key)
        CATALOGUE.build(structure)
    for seg in structure.segments:
        for block in blocks.BLOCKS:
            CATALOGUE.build(compose.swap_block(structure, seg.key, block))
    while len(structure.segments) > 1:
        structure = compose.drop_segment(structure, structure.segments[-1].key)
        CATALOGUE.build(structure)
    with pytest.raises(ModelError, match="at least one"):
        compose.drop_segment(structure, structure.segments[0].key)


def test_a_dropped_line_takes_what_followed_it_along_with_it():
    dropped = compose.drop_segment(MODEL1, "s1")
    assert [s.key for s in dropped.segments] == ["s2", "s3"]
    assert dropped.segment("s3").of == "s2"                      # the share follows the new first line
    assert dropped.places == "s2"                                # costs per place follow the next line that counts
    swapped = compose.swap_block(MODEL1, "s1", "growth")
    assert swapped.places == "s2"
    none_count = compose.swap_block(swapped, "s2", "growth")
    assert none_count.places is None
    model = CATALOGUE.build(none_count)
    assert "emp_pd" not in {r.key for r in model.data_rows()}  # costs are a share of revenue only now
    assert compose.swap_block(MODEL1, "s1", "share").segments[0].block == "growth"   # line 1 can follow nothing


@pytest.mark.parametrize("bad, why", [
    (replace(MODEL1, model="nope"), "no model"),
    (replace(MODEL1, segments=(compose.Segment("s1", "share", of="s2"), compose.Segment("s2", "growth"))), "line before"),
    (replace(MODEL1, segments=(compose.Segment("s1", "growth"), compose.Segment("s1", "growth"))), "each once"),
    (replace(MODEL1, segments=(compose.Segment("s9", "growth"),)), "each once"),
    (replace(MODEL1, segments=(compose.Segment("s1", "magic"),)), "no block"),
    (replace(MODEL1, segments=(compose.Segment("s1", "growth", of="s1"),)), "only a share"),
    (replace(MODEL1, segments=(compose.Segment("s1", "growth"), compose.Segment("s2", "share", formula="rev * 2"))), "may name only"),
    (replace(MODEL1, segments=(compose.Segment("s1", "growth"), compose.Segment("s2", "share", formula="s3_rev"))), "may name only"),
    (replace(MODEL1, segments=(compose.Segment("s1", "growth"), compose.Segment("s2", "share", formula="s1_rev +"))), "formula of line s2"),
    (replace(MODEL1, segments=(compose.Segment("s1", "growth"), compose.Segment("s2", "share", of="s1", rest_of_revenue=True),
                               compose.Segment("s3", "share", of="s1", rest_of_revenue=True))), "at most one"),
    (replace(MODEL1, places="s3"), "counts places"),
    (replace(MODEL1, forecast_years=11), "forecast years"),
    (replace(MODEL1, segments=()), "between 1 and 8"),
])
def test_a_structure_that_would_not_build_is_refused_with_its_reason(bad, why):
    with pytest.raises(ModelError, match=why):
        compose.validate(bad, CATALOGUE.specs)


def test_read_refuses_a_misshapen_structure():
    with pytest.raises(ModelError):
        compose.read({"model": "occupancy"}, CATALOGUE.specs)
    with pytest.raises(ModelError):
        compose.read({"model": "occupancy", "segments": [{"block": "growth"}]}, CATALOGUE.specs)
    assert compose.read(MODEL1.to_dict(), CATALOGUE.specs) == MODEL1


def test_a_changed_model_says_so_in_its_notes_and_keeps_the_sectors_words_only_where_they_fit():
    assert compose.changes(occupancy.SPEC, MODEL1) == []
    swapped = compose.swap_block(MODEL1, "s2", "growth")
    model = CATALOGUE.build(swapped)
    assert any("not among the companies the model was solved by hand for" in n for n in model.notes)
    words = model.sectors["hotels"]
    assert words["s1_places"] == "Rooms"                        # line 1 kept its block and its words
    assert words["s2"] == "Line 2"                              # "Fees on managed rooms" no longer describes it
    assert "s2_units" not in words


def test_fit_leaves_out_what_a_changed_model_has_no_place_for():
    company = support.company(support.load("hotels"))
    swapped = CATALOGUE.build(compose.swap_block(MODEL1, "s2", "growth"))
    with pytest.raises(InputError):
        cells(swapped, company)
    fitted, dropped = compose.fit(swapped, company)
    assert ("s2_add", "y1") in dropped and ("s2_end", "ly") in dropped
    assert ("s2_rev", "ly") not in dropped                       # still a typed figure in the past
    assert fitted.values["s2_rev"]["ly"] == company.values["s2_rev"]["ly"]
    cells(swapped, fitted, allow_missing=True)
    added = CATALOGUE.build(compose.add_segment(MODEL1, "capacity"))
    fitted, _ = compose.fit(added, company)
    assert fitted.choices["line4_past"] == "drivers"             # a new line's choice takes its first option


def test_partial_evaluation_works_the_past_out_before_any_assumption_is_typed():
    company = support.company(support.load("hotels"))
    past_only = replace(company, values={k: {p: v for p, v in by.items() if p in ("ly", "ytd")}
                                         for k, by in company.values.items()})
    with pytest.raises(InputError, match="assumptions not given"):
        evaluate(occupancy.MODEL, past_only)
    partial = evaluate(occupancy.MODEL, past_only, partial=True)
    whole = evaluate(occupancy.MODEL, company).values
    for row in ("rev", "pbt", "pat_own", "s1_rev", "ebitda"):
        for period in ("ly", "ytd"):
            assert partial.values[row][period] == pytest.approx(whole[row][period]), (row, period)
        assert partial.values[row]["y1"] is None
    assert ("s1_occ", "y1") in partial.missing and ("s1_add", "roy") in partial.missing
    assert all(period not in ("ly", "ytd") for _, period in partial.missing)


def test_partial_evaluation_leaves_a_division_by_a_blank_figure_blank_and_says_so():
    empty = Company("x", "Rs lakh", sector="hotels", constants={"unit_rupees": 100000},
                    choices={"line1_past": "drivers", "line1_end_to_date": "typed"})
    with pytest.raises(EvaluationError):
        evaluate(occupancy.MODEL, replace(empty, values=every_assumption(occupancy.MODEL) | {"rev": {"ly": None}}))
    result = evaluate(occupancy.MODEL, empty, partial=True)
    assert ("rev_g", "cy") in result.undefined or result.values["rev_g"]["cy"] is None
    assert result.values["rev"]["ly"] is None and result.values["ebitda"]["ly"] == 0


def test_the_general_model_is_built_but_not_offered_until_solved_by_hand():
    spec = CATALOGUE.spec("general")
    assert spec.offered is False and spec.vetted is False
    assert occupancy.SPEC.offered is True and occupancy.SPEC.vetted is False
    model = CATALOGUE.build(GENERAL)
    assert {r.key for r in model.data_rows()} >= {"s1_g", "s2_g", "mat_sh", "rev", "chk_bs"}
    assert "mat_pd" not in {r.key for r in model.data_rows()}
    assert CATALOGUE.sectors() == {"hospitals": "occupancy", "hotels": "occupancy", "pharma": "general",
                                   "auto_components": "general"}


def test_line_one_cannot_be_switched_off_and_every_other_line_can():
    model = CATALOGUE.build(ALL_BLOCKS)
    assert model.segments() == {"s%d" % i for i in range(2, 9)}
    assert all(r.segment is None for r in model.data_rows() if r.key.startswith("s1_"))


def test_typed_and_assumed_cells_follow_the_rules_each_block_states():
    model = CATALOGUE.build(ALL_BLOCKS)
    plan = cells(model, all_blocks()[1])
    assert plan[("s7_g", "cy")].typed is ASSUME and plan[("s7_rev", "roy")].tree is not None
    assert plan[("s6_ob_in", "ytd")].typed is FIGURE and plan[("s6_exec", "y1")].typed is ASSUME
    assert plan[("s4_churn", "roy")].typed is ASSUME and plan[("s4_cls", "y2")].tree is not None
