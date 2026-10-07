"""The CI check on the defaults registry (PLAN.md 4.3 "done when": no
unsourced default remains).

Every deal input and every Settings default must say where it comes from
(benchmarks/registry.py); nothing a new deal starts from may be illustrative;
the illustrative Settings left are pinned below, each owned by a later task,
and the list may only shrink.
"""
from __future__ import annotations

from typing import get_args

from api.schemas import DealInputsIn, FeeSetting, RiskSetting, StartingField
from benchmarks import registry, risk
from core.config import DEFAULTS
from library import fees

# The illustrative Settings defaults left. Every one is sourced since 4.5b
# (fees and amortisation from the reference transactions' filings); never add one.
PENDING_PINNED: frozenset = frozenset()


def test_every_deal_input_says_where_its_starting_value_comes_from():
    assert set(registry.DEAL_INPUTS) == set(DealInputsIn.model_fields)


def test_every_setting_says_where_its_default_comes_from():
    assert set(registry.SETTINGS) == set(DEFAULTS)


def test_every_entry_has_a_known_basis_and_says_why():
    for name, entry in {**registry.DEAL_INPUTS, **registry.SETTINGS}.items():
        assert entry["basis"] in registry.BASES, name
        if entry["basis"] != "sourced":
            assert entry.get("reason"), f"{name} needs a reason"
        if entry["basis"] == "pending":
            assert entry.get("task"), f"{name} needs the task that sources it"


def test_nothing_a_new_deal_starts_from_is_illustrative():
    assert not [k for k, v in registry.DEAL_INPUTS.items() if v["basis"] in ("pending", "template")]


def test_every_sourced_input_is_one_the_starting_figures_produce():
    sourced = {k for k, v in registry.DEAL_INPUTS.items() if v["basis"] == "sourced"}
    assert sourced == set(get_args(StartingField))


def test_every_sourced_setting_is_one_the_risk_figures_or_the_reference_fees_produce():
    sourced = {k for k, v in registry.SETTINGS.items() if v["basis"] == "sourced"}
    assert set(risk.SETTINGS) == set(get_args(RiskSetting))
    assert set(fees.SETTINGS) == set(get_args(FeeSetting))
    assert sourced == set(risk.SETTINGS) | set(fees.SETTINGS)
    for key in fees.SETTINGS:
        assert registry.SETTINGS[key]["by"] == "library/fees.py"


def test_the_illustrative_defaults_left_can_only_shrink():
    pending = {k for k, v in registry.SETTINGS.items() if v["basis"] == "pending"}
    assert pending <= PENDING_PINNED, f"new illustrative defaults: {sorted(pending - PENDING_PINNED)}"
    assert pending == PENDING_PINNED, f"sourced now, remove from PENDING_PINNED: {sorted(PENDING_PINNED - pending)}"
