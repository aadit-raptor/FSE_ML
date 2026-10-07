"""The CI check on the defaults registry (PLAN.md 4.3 "done when": no
unsourced default remains).

Every deal input and every Settings default must say where it comes from
(benchmarks/registry.py); nothing a new deal starts from may be illustrative;
the illustrative Settings left are pinned below, each owned by a later task,
and the list may only shrink.
"""
from __future__ import annotations

from typing import get_args

from api.schemas import DealInputsIn, StartingField
from benchmarks import registry
from core.config import DEFAULTS

# The illustrative Settings defaults left after 4.3. Remove a key when its
# task sources it; never add one.
PENDING_PINNED = frozenset({
    # 4.5: fees and amortisation from the reference transactions' filings
    "tx_fee_pct", "fin_fee_pct", "def_senior_amort",
    # 4.4: ranges, correlations and scenarios by region
    "mc_growth_mean", "mc_growth_std", "mc_exit_mean", "mc_exit_std", "mc_rate_mean", "mc_rate_std",
    "mc_gm_mean", "mc_gm_std",
    "corr_g_em", "corr_g_ir", "corr_g_gm", "corr_g_sh", "corr_em_ir", "corr_em_gm", "corr_em_sh",
    "corr_ir_gm", "corr_ir_sh", "corr_gm_sh",
    "bull_growth_mult", "bull_exit_mult", "bull_rate_mult", "bull_margin_mult",
    "rec_growth_adj", "rec_growth_floor", "rec_exit_mult", "rec_rate_mult", "rec_margin_mult",
    "stag_growth_adj", "stag_growth_floor", "stag_exit_mult", "stag_rate_mult", "stag_margin_mult",
})


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


def test_the_illustrative_defaults_left_can_only_shrink():
    pending = {k for k, v in registry.SETTINGS.items() if v["basis"] == "pending"}
    assert pending <= PENDING_PINNED, f"new illustrative defaults: {sorted(pending - PENDING_PINNED)}"
    assert pending == PENDING_PINNED, f"sourced now, remove from PENDING_PINNED: {sorted(PENDING_PINNED - pending)}"
