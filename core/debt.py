"""Debt structures and interest rates, for any market (PLAN.md 2.4).

A deal's debt is a list of :class:`TrancheSpec`. Each one says what it is, how
big it is, how it is priced, how it amortises, what share of the cash sweep it
takes, and how much of its coupon is paid in kind. The nine kinds in
``TRANCHE_TYPES`` are **presets over the same fields**, not nine code paths:
only three mechanics are genuinely new against the old senior + mezzanine
model -- interest that accrues to principal (PIK), a revolving facility's
undrawn commitment and its fee, and a sweep the tranche takes only a share of.

Units follow the deal wizard's: percentages are numbers like ``6.5`` and money
is in the deal's own unit (``core/money.py``). ``in_millions`` converts the
amounts before the engine sees them.

**This module owns the market conventions; the engine does not.** A floating
tranche is resolved here into an all-in rate for each year, and
``lbo_engine`` is handed a plain list of rates. So a reference rate is a label
plus a curve: PLAN.md 4.2 can fill ``reference_path`` from market data without
touching a line of ``lbo_engine/``.
"""
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence

from core.money import DEFAULT_CURRENCY, to_millions
from lbo_engine.capital_structure import CapitalStructure, Tranche

# The instruments a leveraged deal is financed with. Every one of them is the
# same dataclass with different defaults (PRESETS); none is a separate code path.
TRANCHE_TYPES = (
    "amortising_term_loan",      # TLA: bank paper, real amortisation
    "institutional_term_loan",   # TLB: 1% a year, bullet at maturity
    "unitranche",                # one blended facility instead of senior + mezz
    "second_lien",
    "senior_notes",              # high-yield bonds: fixed, non-amortising
    "pik_notes",
    "vendor_loan",
    "revolver",
    "shareholder_loan",
)

# Floating-rate benchmarks by market. The code is a label and a choice of
# curve, never a calculation: two tranches with the same path and margin price
# identically whichever benchmark they name.
REFERENCE_RATES = (
    "SOFR",      # US dollar
    "SONIA",     # sterling
    "ESTR",      # euro, overnight
    "EURIBOR",   # euro, term
    "TONA",      # yen
    "SARON",     # Swiss franc
    "BBSY",      # Australian dollar
    "MIBOR",     # Indian rupee
    "custom",    # anything else, or a rate the user types in themselves
)


@dataclass(frozen=True)
class TrancheSpec:
    """One facility in a deal's debt structure.

    ``amount`` is the facility's full size. For a revolver that is the
    commitment, of which ``drawn_pct`` is drawn at close; only the drawn part
    is a source of funds and only the undrawn part pays a commitment fee. For
    everything else ``drawn_pct`` is 100 and the two are the same.

    Pricing is either fixed (``fixed_rate``) or floating, in which case the
    all-in rate for a year is ``max(reference, floor) + margin``: the floor
    applies to the reference before the margin, as a loan agreement writes it.
    ``reference_path`` is the reference rate year by year; when it is empty the
    flat ``reference_level`` stands for every year.
    """

    name: str
    kind: str
    amount: float = 0.0                 # money, in the deal's unit
    drawn_pct: float = 100.0            # % of the facility drawn at close
    floating: bool = False
    fixed_rate: float = 0.0             # %, when floating is False
    reference_rate: str = "custom"
    reference_level: float = 0.0        # %, the flat reference when no path is given
    reference_path: tuple = ()          # % a year; the last year stands for the rest
    margin: float = 0.0                 # %, over the reference
    floor: float = 0.0                  # %, on the reference
    maturity_years: int = 7
    amort_pct: float = 0.0              # % of the original principal a year
    amort_schedule: tuple = ()          # money a year; overrides amort_pct
    upfront_fee_pct: float = 0.0        # % of the facility, paid at close
    commitment_fee_pct: float = 0.0     # % a year on what is undrawn
    sweep: bool = False                 # takes the cash sweep
    sweep_share: float = 100.0          # % of the cash available it may take
    pik_share: float = 0.0              # % of the coupon that accrues to principal
    sweep_priority: int = 0             # 0 = position in the list
    allow_redraw: bool = False          # a revolver funds a cash shortfall
    currency: str = ""                  # "" = the deal's own currency

    def __post_init__(self):
        if self.kind not in TRANCHE_TYPES:
            raise ValueError(
                f"kind must be one of {', '.join(TRANCHE_TYPES)}, not {self.kind!r}")
        if self.reference_rate not in REFERENCE_RATES:
            raise ValueError(
                f"reference_rate must be one of {', '.join(REFERENCE_RATES)}, "
                f"not {self.reference_rate!r}")
        if self.currency and self.currency != DEFAULT_CURRENCY and not self.currency.isupper():
            raise ValueError("currency must be a three-letter ISO 4217 code")
        # Sequences arrive as lists from the API and have to be hashable here
        object.__setattr__(self, "reference_path", tuple(float(x) for x in self.reference_path))
        object.__setattr__(self, "amort_schedule", tuple(float(x) for x in self.amort_schedule))

    # -- derived ---------------------------------------------------------
    @property
    def drawn(self) -> float:
        """What is drawn at close, and so a source of funds (money)."""
        return self.amount * self.drawn_pct / 100

    @property
    def undrawn(self) -> float:
        """The commitment nobody has taken down: fee-paying, not a source."""
        return self.amount - self.drawn

    @property
    def upfront_fee(self) -> float:
        """The arrangement fee paid at close (money)."""
        return self.amount * self.upfront_fee_pct / 100

    def rate_in_year(self, year: int) -> float:
        """The all-in rate for ``year`` (1-indexed), as a decimal.

        A path shorter than the deal's hold **repeats its last year** rather
        than running out. That matters because the exit-sensitivity grid reruns
        the whole model at holds of three to seven years
        (``lbo_engine/model.py``), so a five-year path has to answer for years
        six and seven. Repeating is the honest answer: extrapolating a trend
        would make the grid depend on a curve nobody supplied.
        """
        if not self.floating:
            return self.fixed_rate / 100
        path = self.reference_path or (self.reference_level,)
        reference = path[min(year, len(path)) - 1] / 100
        return max(reference, self.floor / 100) + self.margin / 100

    def in_millions(self, unit: str) -> "TrancheSpec":
        """The same tranche with its money in millions, the unit the engine
        runs in (``core/money.py``)."""
        if unit == "millions":
            return self
        return replace(
            self,
            amount=to_millions(self.amount, unit),
            amort_schedule=tuple(to_millions(x, unit) for x in self.amort_schedule),
        )


# ---------------------------------------------------------------------------
# What each kind is, when the user picks one
# ---------------------------------------------------------------------------
# Only the fields that differ from TrancheSpec's own defaults. These are
# starting points a user edits, not constraints: any field stays editable
# whatever the kind, because the kinds shade into each other in real deals
# (a unitranche with a PIK strip, a second lien that sweeps).
PRESETS: Mapping[str, dict] = {
    "amortising_term_loan": dict(
        floating=True, amort_pct=5.0, sweep=True, maturity_years=6),
    "institutional_term_loan": dict(
        floating=True, amort_pct=1.0, sweep=True, maturity_years=7),
    "unitranche": dict(
        floating=True, sweep=True, maturity_years=7),
    "second_lien": dict(
        floating=True, maturity_years=8),
    "senior_notes": dict(
        floating=False, maturity_years=8),
    "pik_notes": dict(
        floating=False, pik_share=100.0, maturity_years=8),
    "vendor_loan": dict(
        floating=False, pik_share=100.0, maturity_years=8),
    "revolver": dict(
        floating=True, drawn_pct=0.0, commitment_fee_pct=0.5, sweep=True,
        allow_redraw=True, maturity_years=6),
    # Beyond any normal hold: it rides through the exit with the sponsor
    "shareholder_loan": dict(
        floating=False, pik_share=100.0, maturity_years=10),
}


def spec_from_kind(kind: str, name: str, **overrides) -> TrancheSpec:
    """A tranche of ``kind`` with that kind's usual shape, then ``overrides``."""
    if kind not in TRANCHE_TYPES:
        raise ValueError(f"kind must be one of {', '.join(TRANCHE_TYPES)}, not {kind!r}")
    return TrancheSpec(name=name, kind=kind, **{**PRESETS[kind], **overrides})


# ---------------------------------------------------------------------------
# Specs -> the engine's capital structure
# ---------------------------------------------------------------------------
def unique_names(specs: Sequence[TrancheSpec]) -> list:
    """Each tranche's name, made unique.

    The debt schedule is keyed by name, so two facilities called "Term loan"
    would otherwise overwrite each other and the second one's schedule would
    vanish. The second becomes "Term loan (2)".
    """
    taken: set = set()
    names = []
    for spec in specs:
        base = spec.name.strip() or spec.kind
        name, n = base, 1
        # Counting occurrences is not enough: a user who has already named a
        # facility "Term loan (2)" would collide with the second "Term loan".
        while name in taken:
            n += 1
            name = f"{base} ({n})"
        taken.add(name)
        names.append(name)
    return names


def to_tranche(spec: TrancheSpec, name: str, hold: int, priority: int) -> Tranche:
    """``spec`` as the engine's ``Tranche``: rates resolved year by year, and
    the amortisation written the way ``debt_model`` reads it."""
    if spec.amort_schedule:
        amort_type, amort_pct = "custom", 0.0
    elif spec.amort_pct:
        amort_type, amort_pct = "amortizing", spec.amort_pct / 100
    else:
        amort_type, amort_pct = "bullet", 0.0
    # One rate per year of the run. Longer than the hold so the sensitivity
    # grid's longer holds are covered without rebuilding the structure.
    years = max(int(hold), 1) + len(SENSITIVITY_HEADROOM)
    return Tranche(
        name=name,
        amount=spec.drawn,
        interest_rate=spec.rate_in_year(1),
        rate_path=tuple(spec.rate_in_year(y) for y in range(1, years + 1)),
        maturity_years=int(spec.maturity_years),
        amort_type=amort_type,
        amort_pct=amort_pct,
        amort_schedule=list(spec.amort_schedule),
        fee_pct=spec.upfront_fee_pct / 100,
        is_cash_sweep=bool(spec.sweep),
        sweep_share=spec.sweep_share / 100,
        sweep_priority=spec.sweep_priority or priority,
        commitment=spec.amount if spec.undrawn > 0 else None,
        commitment_fee_pct=spec.commitment_fee_pct / 100,
        pik_share=spec.pik_share / 100,
        allow_redraw=bool(spec.allow_redraw),
        currency=spec.currency or DEFAULT_CURRENCY,
    )


# Years of rate path built beyond the deal's hold, so the exit-sensitivity
# grid's longer holds have rates without rebuilding the structure. The grid
# runs to `sens_hp_max`, 7 by default; ten is room to spare, and the path
# repeats its last year anyway.
SENSITIVITY_HEADROOM = range(10)


def build_capital_structure(specs: Sequence[TrancheSpec], ebitda: float, hold: int) -> CapitalStructure:
    """The engine's capital structure for a list of tranche specs, in the order
    the user put them in: first in the list is swept first."""
    names = unique_names(specs)
    return CapitalStructure(
        tranches=[to_tranche(spec, name, hold, i + 1)
                  for i, (spec, name) in enumerate(zip(specs, names))],
        ltm_ebitda=ebitda,
    )


def total_debt(specs: Sequence[TrancheSpec]) -> float:
    """Debt drawn at close (money): a revolver's undrawn commitment is not
    money anyone has."""
    return sum(spec.drawn for spec in specs)


def financing_fees(specs: Sequence[TrancheSpec]) -> float:
    """The arrangement fees the individual facilities charge (money).

    This is **its own use of funds**, beside the deal's flat financing fee
    from Settings -- not a replacement for it. Two independent lines is what
    keeps either one from double counting the other, and it means adding a
    tranche fee never silently switches off a percentage the user set. Both
    appear on the sources & uses table.

    A tranche's fee is never seeded from the Settings percentage. That would
    move the deal's numbers in the last bits, because ``420 x 2.6% + 180 x
    2.6%`` and ``600 x 2.6%`` are not the same float -- and a user who writes
    today's structure out as tranches must see nothing move at all.
    """
    return sum(spec.upfront_fee for spec in specs)


def blended_rate(specs: Sequence[TrancheSpec]) -> float:
    """Year-one rate across the structure, weighted by what is drawn (decimal)."""
    drawn = total_debt(specs)
    if drawn <= 0:
        return 0.0
    return sum(spec.drawn * spec.rate_in_year(1) for spec in specs) / drawn


class UnfinanceableStructure(ValueError):
    """The tranches ask for more than the deal costs.

    A refusal the caller can act on, not a fault: the API answers 422 with the
    message (api/main.py). It carries the deal's own figures, so it goes to
    the person who typed them and **never to a log** -- CLAUDE.md's rule that
    deal contents stay out of logs and Sentry.
    """


def check_specs(specs: Sequence[TrancheSpec], entry_ev: float, entry_costs: float) -> None:
    """Refuse a structure the sponsor could not fund, with a sentence saying
    what to change.

    Sizing debt as a share of EV was capped at 99%; an explicit tranche list
    has no such cap, so a structure can quietly ask for more debt than the
    business is worth and leave a negative equity cheque.
    """
    drawn = total_debt(specs)
    needed = entry_ev + entry_costs
    if drawn > needed:
        raise UnfinanceableStructure(
            f"The tranches raise more than the deal costs: {drawn:,.1f} of debt against "
            f"{needed:,.1f} of enterprise value and fees, which would leave the sponsor "
            f"with a negative equity cheque. Reduce a tranche's size, or raise the entry "
            f"multiple or EBITDA.")


# ---------------------------------------------------------------------------
# Today's two-tranche structure, written out
# ---------------------------------------------------------------------------
def equivalent_tranches(deal, cfg: Mapping) -> list:
    """The senior + mezzanine structure a deal's percentages imply, as an
    explicit tranche list.

    Running this gives **exactly** the answer the percentages give -- returns,
    debt schedule and the sensitivity grid's own column -- because it is what
    the deal screen's "use explicit tranches" button starts from, and a user
    who converts and changes nothing should see their deal unchanged. Two
    details make that exact rather than nearly exact:

    * the mezzanine is written as a **margin over the senior rate**, which is
      what it has always been (``interest_rate + mezz_spread`` in the engine),
      so the arithmetic is the same expression rather than a rounded copy of
      its result;
    * neither tranche carries an upfront fee, so no arrangement fee is added
      and the deal's flat financing-fee setting still applies on its own.

    The one place the two paths part company is the exit-sensitivity grid's
    *other* columns. Sizing by percentages gives the mezzanine a maturity
    equal to the hold, so the bullet follows the grid to three or seven years
    and the tranche is always repaid at exit; a facility written out
    explicitly matures when its agreement says. See
    ``tests/test_debt_structures.py`` and CLAUDE.md "Model findings", 11.
    """
    entry_ev = deal.ebitda * deal.entry_mult
    debt = entry_ev * deal.debt_pct / 100
    senior = round(debt * deal.senior_pct / 100, 2)
    mezz = round(debt * (1 - deal.senior_pct / 100), 2)
    hold = int(deal.hold)
    return [
        TrancheSpec(
            name="Senior Term Loan", kind="amortising_term_loan", amount=senior,
            fixed_rate=deal.base_rate, amort_pct=cfg["def_senior_amort"],
            sweep=True, sweep_priority=1, maturity_years=hold,
        ),
        TrancheSpec(
            # A margin over the senior rate, not a rate of its own: the engine
            # has always priced it as base + spread
            name="Mezzanine", kind="second_lien", amount=mezz,
            floating=True, reference_rate="custom", reference_level=deal.base_rate,
            margin=deal.mezz_spread, sweep=True, sweep_priority=2, maturity_years=hold,
        ),
    ]


def coerce(tranches) -> tuple:
    """A deal's tranche list as ``TrancheSpec``s, whatever it arrives as.

    Routers build ``DealInputs(**model_dump())``, so tranches come in as plain
    dicts; ``dataclasses.replace`` then runs this again on the way to millions,
    so it has to leave specs it has already built alone.
    """
    out = []
    for t in tranches or ():
        out.append(t if isinstance(t, TrancheSpec) else TrancheSpec(**dict(t)))
    return tuple(out)


# Money in a tranche, for the answer's money keys (core/deal.py)
TRANCHE_MONEY_KEYS = frozenset({"amount", "drawn", "undrawn", "commitment", "amort_schedule"})


def summary(specs: Sequence[TrancheSpec], names: Sequence[str], ebitda: float) -> list:
    """One row per tranche for the screen: what it is, how big, how priced."""
    return [
        {
            "name": name,
            "kind": spec.kind,
            "amount": spec.drawn,
            "commitment": spec.amount if spec.undrawn > 0 else None,
            "x_ebitda": spec.drawn / ebitda if ebitda else 0.0,
            "rate": spec.rate_in_year(1),
            "floating": spec.floating,
            "reference_rate": spec.reference_rate if spec.floating else None,
            "maturity_years": int(spec.maturity_years),
            "sweep_share": spec.sweep_share if spec.sweep else 0.0,
            "pik_share": spec.pik_share,
        }
        for spec, name in zip(specs, names)
    ]
