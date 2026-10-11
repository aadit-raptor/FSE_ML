"""The regional deals the live sliders' card is measured on (PLAN.md 5.1, 5.8).

    python -m tests.ml_regional_deals       # rewrites ml/evaluation/data/regional_deals.json

For every economy the app covers and each industry in
``ml.evaluation.surrogate.INDUSTRIES``: the Monte Carlo inputs "Use sourced
figures" gives (PLAN.md 4.3, 4.4), from the recorded industry workbooks,
their history and the economic series (tests/fixtures/benchmarks,
tests/fixtures/economy), judged on the day the series were recorded, with
the deal's default debt share (``deals``: the eleven inputs the retired
network took, on its one fixed deal).

``varied`` (PLAN.md 5.8) is the same economies and industries as whole
deals, as the API takes them: everything "Use sourced figures" sets (starting
figures, Monte Carlo ranges, correlations), the country's tax preset where
there is one, and terms varied deal by deal -- entry multiple, hold, fees,
amortisation, leases and five debt structures in turn, from percentages to
five listed facilities. ``vary`` draws them from a generator seeded by the
deal's id, so the file is the same every time. tests/test_model_cards.py
fails when the file is stale.
"""
from __future__ import annotations

import json
import random
import zlib
from pathlib import Path

from benchmarks import risk, starting
from core.deal import DealInputs
from core.tax import PRESETS_BY_CODE, apply_preset
from economy.catalogue import AREAS, COUNTRIES
from library.base_rates import sp_region
from ml.evaluation.surrogate import INDUSTRIES, REGIONAL_DEALS, features_from
from tests.e2e_benchmarks import recorded_sources


HOLDS = (3, 4, 5, 6, 7, 8, 10)
AMORTISATION = (0.0, 1.0, 2.5, 5.0, 10.0)
STRUCTURES = ("percentages", "unitranche_revolver", "loans_second_lien", "loan_notes_pik", "five_facilities")
EBITDA = 100.0
LEASE_COST, LEASE_LIABILITY = 8.0, 45.0
TAX_FIELDS = ("tax", "tax_preset", "tax_interest_limit", "tax_interest_limit_pct", "tax_interest_limit_amount",
              "tax_loss_carryforward", "tax_loss_limit_pct", "tax_loss_limit_amount", "tax_minimum_pct")


def _facilities(structure: str, debt: float, ev: float, reference: float, rng: random.Random) -> list[dict]:
    """``structure`` as the API's facility list: ``debt`` drawn in all,
    floating facilities on ``reference`` (%), every margin and coupon drawn."""
    def floating(name, kind, share, margin, **more):
        return {"name": name, "kind": kind, "amount": round(debt * share, 2), "floating": True,
                "reference_level": reference, "margin": round(margin, 2), "floor": rng.choice((0.0, 0.5)),
                "upfront_fee_pct": rng.choice((0.0, 1.0, 2.5)), **more}

    def fixed(name, kind, share, coupon, **more):
        return {"name": name, "kind": kind, "amount": round(debt * share, 2), "floating": False,
                "fixed_rate": round(coupon, 2), **more}

    loan = dict(amort_pct=1.0, sweep=True, maturity_years=7)
    margin = rng.uniform(2.5, 4.5)
    if structure == "unitranche_revolver":
        return [floating("Unitranche", "unitranche", 1.0, margin + 2.0, sweep=True, maturity_years=7),
                {"name": "RCF", "kind": "revolver", "amount": round(ev * 0.08, 2), "drawn_pct": 0.0,
                 "floating": True, "reference_level": reference, "margin": round(margin, 2),
                 "commitment_fee_pct": 0.5, "sweep": True, "allow_redraw": True, "maturity_years": 6}]
    if structure == "loans_second_lien":
        return [floating("Term loan A", "amortising_term_loan", 0.35, margin, amort_pct=5.0, sweep=True,
                         maturity_years=6),
                floating("Term loan B", "institutional_term_loan", 0.45, margin + 0.5, **loan),
                floating("Second lien", "second_lien", 0.20, margin + 3.5, maturity_years=8)]
    if structure == "loan_notes_pik":
        return [floating("Term loan B", "institutional_term_loan", 0.60, margin, **loan),
                fixed("Senior notes", "senior_notes", 0.25, reference + margin + 2.0, maturity_years=8),
                fixed("PIK notes", "pik_notes", 0.15, reference + margin + 5.0, pik_share=100.0, maturity_years=8)]
    return [{"name": "RCF", "kind": "revolver", "amount": round(debt * 0.15, 2), "drawn_pct": 33.0,
             "floating": True, "reference_level": reference, "margin": round(margin, 2),
             "commitment_fee_pct": 0.5, "sweep": True, "allow_redraw": True, "maturity_years": 6},
            floating("Term loan B", "institutional_term_loan", 0.50, margin, sweep_share=75.0, **loan),
            fixed("Senior notes", "senior_notes", 0.25, reference + margin + 2.0, maturity_years=8),
            fixed("Vendor loan", "vendor_loan", 0.10, 8.0, pik_share=100.0, maturity_years=8),
            fixed("Shareholder loan", "shareholder_loan", 0.10, 10.0, pik_share=100.0, maturity_years=10)]


def vary(index: int, deal_id: str, country: str, currency: str, start: dict, ranges: dict) -> dict:
    """One sourced regional deal with its own terms: the request the Live
    step would send for it (``mc``, ``deal``, ``settings``)."""
    rng = random.Random(zlib.crc32(deal_id.encode()))
    structure = STRUCTURES[index % len(STRUCTURES)]
    entry = round(start["entry_mult"] * rng.uniform(0.8, 1.15), 1)
    hold = rng.choice(HOLDS)
    mc = {"ebitda": EBITDA, "entry_mult": entry, "hold": hold,
          "growth_mean": ranges["mc_growth_mean"], "growth_std": ranges["mc_growth_std"],
          "exit_mean": ranges["mc_exit_mean"], "exit_std": ranges["mc_exit_std"],
          "rate_mean": ranges["mc_rate_mean"], "rate_std": ranges["mc_rate_std"],
          "gm_mean": ranges["mc_gm_mean"], "gm_std": ranges["mc_gm_std"]}
    deal = {"ebitda": EBITDA, "entry_mult": entry, "exit_mult": start["exit_mult"], "hold": hold,
            "currency": currency, "country": country, "growth": start["growth"],
            "gross_margin": start["gross_margin"], "opex": start["opex"], "da": start["da"],
            "capex": start["capex"], "nwc": start["nwc"], "tax": start["tax"], "base_rate": start["base_rate"]}
    if country in PRESETS_BY_CODE:
        ruled, _ = apply_preset(DealInputs(currency=currency), country)
        deal.update({k: getattr(ruled, k) for k in TAX_FIELDS})
    valued_on = EBITDA
    if index % 7 == 3:
        view = rng.choice(("pre_ifrs16", "post_ifrs16"))
        deal.update(accounting_standard="ifrs", lease_view=view, lease_cost=LEASE_COST,
                    lease_liability=LEASE_LIABILITY)
        valued_on = EBITDA if view == "post_ifrs16" else EBITDA - LEASE_COST
    share = round(rng.uniform(30.0, 70.0), 1)
    if structure == "percentages":
        deal.update(debt_pct=share, senior_pct=round(rng.uniform(50.0, 100.0), 1),
                    mezz_spread=round(rng.uniform(2.0, 7.0), 2))
    else:
        ev = valued_on * entry
        reference = round(max(ranges["mc_rate_mean"] - 1.5, 0.25), 2)
        deal["tranches"] = _facilities(structure, ev * share / 100, ev, reference, rng)
    settings = {k: v for k, v in ranges.items() if k.startswith("corr_")}
    settings.update(tx_fee_pct=round(rng.uniform(0.5, 4.0), 2), fin_fee_pct=round(rng.uniform(0.5, 4.0), 2),
                    def_senior_amort=rng.choice(AMORTISATION))
    return {"structure": structure, "mc": mc, "deal": deal, "settings": settings}


def build() -> dict:
    tables, series, today = recorded_sources()
    debt_pct = DealInputs().debt_pct
    deals, varied, skipped = [], [], []
    for country in COUNTRIES:
        currency = AREAS[country].currency
        for industry in INDUSTRIES:
            start = starting.starting_assumptions(country, industry, currency, tables, series, today)
            ranges = risk.risk_assumptions(country, industry, currency, tables, series, today)
            sourced = {f["field"]: f["value"] for f in ranges["figures"]}
            features = features_from(sourced, start["inputs"], debt_pct)
            deal_id = f"{country}|{industry}"
            if features is None:
                skipped.append(deal_id)
                continue
            deals.append({"id": deal_id, "country": country, "industry": industry, "currency": currency,
                          "region": sp_region(country), "features": features})
            varied.append({"id": deal_id, "country": country, "industry": industry, "region": sp_region(country),
                           **vary(len(varied), deal_id, country, currency, start["inputs"], sourced)})
    return {"recorded_on": today.isoformat(), "debt_pct": debt_pct, "skipped": skipped, "deals": deals,
            "varied": varied}


def text() -> str:
    return json.dumps(build(), indent=1, sort_keys=True) + "\n"


def main() -> None:
    Path(REGIONAL_DEALS).parent.mkdir(parents=True, exist_ok=True)
    REGIONAL_DEALS.write_text(text(), encoding="utf-8", newline="\n")
    print(f"wrote {REGIONAL_DEALS}")


if __name__ == "__main__":
    main()
