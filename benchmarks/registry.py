"""The defaults registry: where every default number comes from (PLAN.md 4.3).

Every deal input a new deal starts with, and every Settings default
(core/config.py ``DEFAULTS``), is listed here with its **basis**:

- ``sourced``: published data on many companies or economies, worked out by
  benchmarks/starting.py; the screen shows the source, the sample and the
  date beside it;
- ``deal``: the deal's own figure or label, asked when it starts or entered
  (its size, currency, structure, rules switched on);
- ``choice``: a modelling choice rather than a market figure, with the
  reason (a hold period, how many simulation paths, a rule left off);
- ``template``: Settings' starting values, the user's own to set and
  applied to a deal only when the user asks; a new deal never starts from
  them, and the screen labels their factory values illustrative;
- ``pending``: still illustrative, with the PLAN.md task that sources it.

``tests/test_defaults_registry.py`` is the CI check: every input and every
setting is listed, nothing a new deal starts from is ``pending``, each
``sourced`` entry names a figure benchmarks/starting.py produces, and the
``pending`` list can only shrink (the test pins it).
"""
from __future__ import annotations

from typing import Mapping

BASES = ("sourced", "deal", "choice", "template", "pending")

# What a new deal starts with, input by input
DEAL_INPUTS: Mapping[str, dict] = {
    **{k: {"basis": "sourced"} for k in (
        "growth", "tax", "gross_margin", "opex", "da", "entry_mult", "exit_mult", "capex", "nwc",
        "ar_days", "inv_days", "ap_days", "debt_pct", "senior_pct", "base_rate")},
    "ebitda": {"basis": "deal", "reason": "the deal's size, asked when it starts"},
    "currency": {"basis": "deal", "reason": "asked when the deal starts (the account's by default)"},
    "unit": {"basis": "deal", "reason": "how the deal's money is written"},
    "country": {"basis": "deal", "reason": "asked when the deal starts; a label"},
    "industry": {"basis": "deal", "reason": "asked when the deal starts; a label"},
    "fiscal_year_end_month": {"basis": "deal", "reason": "a label"},
    "first_fiscal_year": {"basis": "deal", "reason": "a label"},
    "tranches": {"basis": "deal", "reason": "the deal's own facilities, when listed"},
    "accounting_standard": {"basis": "deal", "reason": "a label, set from a filing or by hand"},
    "lease_view": {"basis": "deal", "reason": "the standard's own view unless chosen"},
    "lease_cost": {"basis": "deal", "reason": "none unless the deal has leases"},
    "lease_liability": {"basis": "deal", "reason": "none unless the deal has leases"},
    "tax_preset": {"basis": "deal", "reason": "a label"},
    **{k: {"basis": "choice", "reason": "tax rules are off until switched on; a country preset "
                                         "brings each rule's statute (core/tax.py)"} for k in (
        "tax_interest_limit", "tax_interest_limit_pct", "tax_interest_limit_amount", "tax_loss_carryforward",
        "tax_loss_limit_pct", "tax_loss_limit_amount", "tax_minimum_pct")},
    "hold": {"basis": "choice", "reason": "the sponsor's plan, not a market figure"},
    "mincash": {"basis": "choice", "reason": "no cash kept back unless the deal sets some"},
    "mezz_spread": {"basis": "choice", "reason": "applies only to a mezzanine tranche; the starting "
                                                 "structure is all senior"},
    "wsp_mode": {"basis": "choice", "reason": "flat working capital unless days are chosen"},
}

_TEMPLATE = {"basis": "template", "reason": "Settings' starting values: applied only when asked"}
_PENDING_44 = {"basis": "pending", "task": "4.4", "reason": "ranges, correlations and scenarios by region"}
_PENDING_45 = {"basis": "pending", "task": "4.5",
               "reason": "no free source publishes buyout fees or amortisation; the reference "
                         "transactions record them from filings"}

SETTINGS: Mapping[str, dict] = {
    "tx_fee_pct": _PENDING_45, "fin_fee_pct": _PENDING_45, "def_senior_amort": _PENDING_45,
    "other_uses": {"basis": "choice", "reason": "none unless the deal has other uses"},
    **{k: _TEMPLATE for k in (
        "def_ebitda", "def_entry_mult", "def_exit_mult", "def_hold", "def_growth", "def_gross_margin",
        "def_opex", "def_tax", "def_da", "def_debt_pct", "def_senior_pct", "def_base_rate", "def_mezz_spread",
        "def_capex", "def_nwc", "def_mincash")},
    **{k: {"basis": "choice", "reason": "the grid's range, not a market figure"} for k in (
        "sens_em_min", "sens_em_max", "sens_em_steps", "sens_hp_min", "sens_hp_max")},
    "mc_n": {"basis": "choice", "reason": "how many paths to simulate"},
    "mc_n_passes": {"basis": "choice", "reason": "how many times the simulation reruns"},
    "mc_clip_irr": {"basis": "choice", "reason": "how extreme paths are shown"},
    "mc_hurdle": {"basis": "choice", "reason": "the sponsor's own return target"},
    **{k: _PENDING_44 for k in (
        "mc_growth_mean", "mc_growth_std", "mc_exit_mean", "mc_exit_std", "mc_rate_mean", "mc_rate_std",
        "mc_gm_mean", "mc_gm_std",
        "corr_g_em", "corr_g_ir", "corr_g_gm", "corr_g_sh", "corr_em_ir", "corr_em_gm", "corr_em_sh",
        "corr_ir_gm", "corr_ir_sh", "corr_gm_sh",
        "bull_growth_mult", "bull_exit_mult", "bull_rate_mult", "bull_margin_mult",
        "rec_growth_adj", "rec_growth_floor", "rec_exit_mult", "rec_rate_mult", "rec_margin_mult",
        "stag_growth_adj", "stag_growth_floor", "stag_exit_mult", "stag_rate_mult", "stag_margin_mult")},
}
