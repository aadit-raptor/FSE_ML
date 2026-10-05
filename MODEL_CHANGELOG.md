# MODEL_CHANGELOG.md — what changed in the model, version by version

Every result Variater answers carries the **engine version** it was computed
with (PLAN.md 3.1, `core/model_version.py`), and every saved deal keeps the
version, IRR and MOIC it was saved with. When a saved deal's results differ
today, the deal screens say "results changed since saved" and name the cause:
a new engine version (this file says what changed), new editions of the
published data, or changed Settings defaults.

## When the version changes

- **Raise it whenever any deal's numbers would move**: a change in `core/`,
  `lbo_engine/` or `simulation/` that changes an IRR, a MOIC, a cash flow, a
  simulated distribution, or which deals the model accepts.
  `tests/test_model_version.py` pins reference deals' results per version, so
  such a change fails until the version is raised and its results recorded
  (`python -m tests.test_model_version`). Never edit an older version's
  recorded results.
- **Major** (2.0.0): most existing deals' numbers move (a fix to a finding
  below, a new default convention).
- **Minor** (1.1.0): a new mechanic or option; deals that don't use it are
  unchanged to the last digit.
- **Patch** (1.0.1): a fix that moves only the deals it corrects.
- A refactor that moves no number keeps the version, and needs no entry.
- **Data** is versioned separately: a new edition of a published table
  (`core/risk_sources.py`, the tax presets) changes the result stamp's
  `data_vintage` and `data_fingerprint`, not the engine version.
- **Settings defaults** (`core/config.py`) are versioned by the result's
  settings fingerprint, which is taken over the Settings resolved against
  today's defaults; change a default only with an entry here.

Each entry says what moved, why, by how much on the default deal, and the
pull request.

---

## 1.0.0 — 2026-10-04

The first recorded version: the model as it stood when versions began
(PLAN.md 3.1, after 2.8). Default deal: IRR 21.16%, MOIC 2.61x (US dollar
millions, `tests/model_version_pins.json`).

What it contains, for reference:

- The deterministic deal model (`lbo_engine/`), pinned to the retired
  Streamlit app's outputs (`tests/golden/golden.json`), with the fixes to
  model findings 1-5 and 7-9 (CLAUDE.md "Model findings"): sponsor-funded
  minimum cash, Settings the model used to ignore, full-run heatmap and exit
  grid, deal-model backtest predictions, interest iterated to convergence,
  seeded simulations.
- Money in any currency and unit, run in millions (PLAN.md 2.2).
- Debt as a list of facilities: nine kinds, floating rates with floors, PIK,
  revolvers, sweep shares, per-facility fees (PLAN.md 2.4).
- Tax rules: interest limitation, loss carry-forward, minimum tax
  (PLAN.md 2.5).
- Accounting standards and IFRS 16 leases (PLAN.md 2.6).
- Plan vs actual (PLAN.md 2.7) and computed risk warnings (PLAN.md 2.8).

Known open findings that will change numbers when fixed (each waits for the
user's approval, and each will be its own version): 10, rounding money to
two decimals of a million; 11, a cash shortfall funded out of nothing; 12,
the two-bucket simulation discarding cash once debt is repaid; 13, the
simulation ignoring minimum cash.
