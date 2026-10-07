"""Model validation (PLAN.md 4.6).

Does the model's risk read come true as often as it claims? Three checks,
each on the evidence that can test it:

* ``default`` -- the deal summary's default risk (S&P's cumulative default
  rate for the coverage-implied rating) against what happened to the
  approved reference transactions (``library/``);
* ``irr_range`` -- the plan's simulated IRR range against the actual IRR of
  deals whose owners agreed to count them (plan vs actual, PLAN.md 2.7);
* ``loss`` -- the plan's probability of loss against whether those deals
  lost money.

Each is reported overall and split by region, sector, size and era, and
always tested on newer data: a case counts as **out of time** only when its
outcome became known after everything the prediction was made from
(``cases.py``); the rest are reported apart as in-sample. Users' deals are
anonymised (``report.py``): only counts and rates per group, a group holding
fewer than ``report.MIN_CONTRIBUTED`` of them shows neither. The nightly
``validation-report`` task writes the report (``run.py``), the Backtest ->
Validation step shows it, and docs/methodology.md describes it.
"""
