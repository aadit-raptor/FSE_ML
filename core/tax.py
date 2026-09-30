"""Tax rules by country (PLAN.md 2.5).

A deal's tax is a rate plus, optionally, the rules that matter most to a
leveraged buyout: an interest limit, losses carried forward, a minimum tax.
The mechanics are ``lbo_engine/tax.py``; this module turns a deal's inputs
(percentages like 30.0, money in the deal's unit) into its ``TaxRules``, and
holds the **country presets** a user can start from.

The presets are a starting point, not advice. Each one says where it comes
from and when it was checked, is simplified to the headline rules (no state
or local taxes unless the rate says so, no group relief, no expiry of
losses), and every field stays editable once applied. The screen says
"check with a tax adviser" beside them, and so does this docstring.

Money amounts in a preset are in its own currency, in millions. A deal in
another currency gets the preset's rates and shares but not its amounts:
there is no exchange rate to convert them with until PLAN.md 4.2, and a
pound allowance quietly read as dollars would be wrong.
"""
from dataclasses import dataclass, replace
from typing import Optional, Tuple

from core.money import in_unit
from lbo_engine.tax import INTEREST_LIMITS, TaxRules

# The deal fields that carry tax rules, with their defaults: the rules off.
# Stored only when set (db/deals.py OMIT_WHEN_DEFAULT), so a deal from before
# 2.5 keeps its exact bytes and an API from before 2.5 still opens it.
RULE_DEFAULTS = {
    "tax_preset": "",
    "tax_interest_limit": "none",
    "tax_interest_limit_pct": 30.0,
    "tax_interest_limit_amount": 0.0,
    "tax_loss_carryforward": False,
    "tax_loss_limit_pct": 100.0,
    "tax_loss_limit_amount": 0.0,
    "tax_minimum_pct": 0.0,
}
# Money among them, in the deal's unit
TAX_MONEY_INPUTS = ("tax_interest_limit_amount", "tax_loss_limit_amount")


def rules_from_deal(d) -> Optional[TaxRules]:
    """The deal's tax rules for the engine (decimals; money as the deal holds
    it, which is millions once ``core.deal.in_millions`` has run), or None
    when none is switched on."""
    rules = TaxRules(
        interest_limit=d.tax_interest_limit,
        interest_limit_share=d.tax_interest_limit_pct / 100,
        interest_limit_amount=d.tax_interest_limit_amount,
        loss_carryforward=bool(d.tax_loss_carryforward),
        loss_limit_share=d.tax_loss_limit_pct / 100,
        loss_limit_amount=d.tax_loss_limit_amount,
        minimum_tax_rate=d.tax_minimum_pct / 100,
    )
    return rules if rules.active else None


# ---------------------------------------------------------------------------
# Country presets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaxPreset:
    """One country's headline rules for a company's profits. Percentages are
    numbers like 25.0; amounts are millions of ``currency``."""

    code: str                        # ISO 3166-1 alpha-2; the screen names the country
    currency: str                    # ISO 4217, the currency of the amounts
    rate: float                      # headline corporate rate, %
    interest_limit: str = "none"
    interest_limit_pct: float = 30.0
    interest_limit_amount: float = 0.0
    loss_carryforward: bool = True
    loss_limit_pct: float = 100.0
    loss_limit_amount: float = 0.0
    minimum_pct: float = 0.0
    source: str = ""                 # the law the numbers come from
    note: str = ""                   # what the preset leaves out or simplifies
    as_of: str = "2026-01"           # when the numbers were last checked

    def __post_init__(self):
        if self.interest_limit not in INTEREST_LIMITS:
            raise ValueError(f"{self.code}: unknown interest limit {self.interest_limit!r}")


# Checked against the legislation named in each source, as of the date given.
# Headline rules only: see each note. Check with a tax adviser.
PRESETS: Tuple[TaxPreset, ...] = (
    TaxPreset(
        code="US", currency="USD", rate=21.0,
        interest_limit="ebitda_share", interest_limit_pct=30.0,
        loss_limit_pct=80.0,
        source="IRC s.11(b) (21%); s.163(j) as amended by Pub. L. 119-21 (30% of EBITDA-based "
               "adjusted taxable income); s.172(a)(2) (losses offset up to 80%)",
        note="Federal only: add state tax to the rate. The 15% corporate alternative minimum "
             "tax applies only above USD 1bn of book income, so it is left at 0.",
    ),
    TaxPreset(
        code="GB", currency="GBP", rate=25.0,
        interest_limit="ebitda_share", interest_limit_pct=30.0, interest_limit_amount=2.0,
        loss_limit_pct=50.0, loss_limit_amount=5.0,
        source="CTA 2010 s.3 (main rate 25% from April 2023); TIOPA 2010 Part 10 (corporate "
               "interest restriction, GBP 2m de minimis); CTA 2010 Part 7ZA (GBP 5m allowance, "
               "then 50%)",
        note="The small profits rate and the group ratio election are left out.",
    ),
    TaxPreset(
        code="DE", currency="EUR", rate=29.83,
        interest_limit="ebitda_share", interest_limit_pct=30.0, interest_limit_amount=3.0,
        loss_limit_pct=70.0, loss_limit_amount=1.0,
        source="KStG s.23 and SolZG (15.825%) plus GewStG at a 400% multiplier (14%); EStG "
               "s.4h (interest barrier, 30% of EBITDA); EStG s.10d (EUR 1m, then 70% for "
               "2024 to 2027)",
        note="The EUR 3m interest threshold is modelled as an allowance; in law, interest above "
             "it loses the exemption entirely. The loss share returns to 60% from 2028, and "
             "the corporate rate falls a point a year from 2028 to 10% in 2032: the rate here "
             "is today's, for every year. Trade tax depends on the municipality.",
    ),
    TaxPreset(
        code="FR", currency="EUR", rate=25.0,
        interest_limit="ebitda_share", interest_limit_pct=30.0, interest_limit_amount=3.0,
        loss_limit_pct=50.0, loss_limit_amount=1.0,
        source="CGI art. 219 (25%); art. 212 bis (30% of tax EBITDA or EUR 3m, the higher); "
               "art. 209 (EUR 1m, then 50%)",
        note="The social contribution and temporary surtaxes on large companies are left out, "
             "and so is the 5% a year by which disallowed interest carried forward is reduced.",
    ),
    TaxPreset(
        code="NL", currency="EUR", rate=25.8,
        interest_limit="ebitda_share", interest_limit_pct=24.5, interest_limit_amount=1.0,
        loss_limit_pct=50.0, loss_limit_amount=1.0,
        source="Wet Vpb 1969 art. 22 (25.8% top rate); art. 15b (24.5% of EBITDA from 2025, "
               "EUR 1m threshold); art. 20 (EUR 1m, then 50%)",
        note="The 19% rate on the first EUR 200,000 of profit is left out.",
    ),
    TaxPreset(
        code="IE", currency="EUR", rate=12.5,
        interest_limit="ebitda_share", interest_limit_pct=30.0, interest_limit_amount=3.0,
        source="TCA 1997 s.21 (12.5% trading rate); Part 35D (interest limitation, 30% of "
               "EBITDA, EUR 3m de minimis); s.396 (trading losses carried forward)",
        note="Losses are set against the same trade only. The 15% minimum rate for large "
             "groups (Pillar Two) is left at 0.",
    ),
    TaxPreset(
        code="IN", currency="INR", rate=25.17,
        source="Income-tax Act 1961 s.115BAA (22% plus surcharge and cess); s.72 (business "
               "losses carried forward)",
        note="s.94B limits interest to 30% of EBITDA only on debt from associated "
             "enterprises, so no limit is set. Losses expire after eight years, which the "
             "model does not track.",
    ),
    TaxPreset(
        code="JP", currency="JPY", rate=29.74,
        loss_limit_pct=50.0,
        source="Corporation Tax Act (23.2%) with local taxes (standard effective rate 29.74%, "
               "Ministry of Finance); CTA art. 57 (losses offset up to 50% for large companies)",
        note="Tokyo's excess local rates bring the effective rate to about 30.6%, and the "
             "defence special corporation tax from April 2026 adds to it. The earnings "
             "stripping rule (20% of adjusted income) reaches only interest to related "
             "parties or debt they fund or guarantee, so no limit is set. Losses expire "
             "after ten years, which the model does not track.",
    ),
    TaxPreset(
        code="AU", currency="AUD", rate=30.0,
        interest_limit="ebitda_share", interest_limit_pct=30.0,
        source="Income Tax Rates Act 1986 s.23 (30%); ITAA 1997 Div 820 (fixed ratio test, 30% "
               "of tax EBITDA, from July 2023); ITAA 1997 Div 36 (losses)",
        note="The AUD 2m de minimis is on total debt deductions, so it is left out; losses "
             "depend on the continuity tests.",
    ),
    TaxPreset(
        code="CA", currency="CAD", rate=26.5,
        interest_limit="ebitda_share", interest_limit_pct=30.0,
        source="ITA s.123 and 124 (15% federal) plus Ontario (11.5%); ITA s.18.2 (EIFEL, 30% "
               "of adjusted taxable income); ITA s.111 (non-capital losses)",
        note="Provincial rates differ: Ontario is shown. Losses expire after 20 years.",
    ),
    TaxPreset(
        code="SG", currency="SGD", rate=17.0,
        source="Income Tax Act 1947 s.43 (17%); s.37 (losses carried forward)",
        note="No general interest limit. Partial exemptions for the first SGD 200,000 of "
             "profit are left out.",
    ),
)
PRESETS_BY_CODE = {p.code: p for p in PRESETS}


def apply_preset(d, code: str):
    """The deal with ``code``'s rules, and whether its amounts were applied.

    Amounts come across only in the preset's own currency, converted to the
    deal's unit; in any other currency they are left at 0 and the answer says
    so, for the screen to tell the user.
    """
    p = PRESETS_BY_CODE[code]
    same_currency = d.currency == p.currency
    amount = (lambda m: in_unit(m, d.unit)) if same_currency else (lambda m: 0.0)  # noqa: E731
    return replace(
        d,
        tax=p.rate,
        tax_preset=p.code,
        tax_interest_limit=p.interest_limit,
        tax_interest_limit_pct=p.interest_limit_pct,
        tax_interest_limit_amount=amount(p.interest_limit_amount),
        tax_loss_carryforward=p.loss_carryforward,
        tax_loss_limit_pct=p.loss_limit_pct,
        tax_loss_limit_amount=amount(p.loss_limit_amount),
        tax_minimum_pct=p.minimum_pct,
    ), same_currency or not (p.interest_limit_amount or p.loss_limit_amount)
