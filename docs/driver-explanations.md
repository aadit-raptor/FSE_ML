# Driver explanations: the comparison (PLAN.md 5.6)

The task: compare `ml/shap_attribution.py` with the Monte Carlo Drivers view,
ship per-deal explanations if they add value and are light enough for the
free server, otherwise remove the module.

**Decision.** Per-deal explanations ship, computed from the simulation itself
([analytics/driver_attribution.py](../analytics/driver_attribution.py),
[methodology](methodology.md#9-monte-carlo-simulation)). The XGBoost and SHAP
module is removed: what it did well (contributions in IRR points that add
up) the simulation can do exactly, for any deal, with no trained model and no
new package.

Measured on 2026-10-10 (Apple Silicon Mac, default deal, 50,000 paths, seed
42 unless stated).

## The three candidates

| | Drivers view before 5.6 | `ml/shap_attribution.py` | Shipped |
|---|---|---|---|
| What it is | Spearman rank correlation of each draw with IRR; a scatter with a fitted line; the correlation table | An XGBoost model of median IRR fitted to 30,000 simulated deals, explained with SHAP's tree explainer | Shapley values of the simulation's own IRR over its five draws |
| The question it answers | Which draws rise and fall with IRR across the paths? | Why does a model's prediction for this deal differ from its average prediction for random deals? | Why are this deal's worst and best 5% of paths where they are, starting from its IRR at the mean assumptions? |
| Unit | A correlation, −1 to 1 | IRR points | IRR points |
| Adds up to | Nothing | The model's prediction, not the simulation's answer | The simulated figure, exactly |
| Deals it covers | Any | One shape: entry 10x, five years, opex 18%, tax 25%, no fees, senior and mezzanine by percentage | Any the simulation runs: tranches, tax rules, leases, scenarios |
| Separates correlated drivers | No | Not applicable (its inputs are the deal's assumptions, drawn independently) | Yes |
| Cost per run | Included | About 0.1 s, 375 MB peak memory | 0.03 s; no extra peak memory |
| Needs | Nothing | `xgboost`, `shap`, `matplotlib`, a trained model | Nothing |

## What the old module did, measured

It was never wired in: neither `xgboost` nor `shap` is in a requirements
file, and no trained file was committed, so the app could not import it. To
judge it fairly it was trained as written (xgboost 3.4.1, shap 0.53.0) in a
throwaway environment.

- **Accuracy.** Validation error 0.61 points of median IRR. For the default
  assumptions it predicts 20.78%, and the simulation of *its* fixed deal
  gives 20.70%.
- **It explains another deal.** It knows eleven assumptions and nothing
  else. Entering at 8x and holding seven years, the simulation says 23.30%;
  with 2% transaction and 2% financing fees, 18.85%; the app's own default
  deal, 18.48%. The module answers 20.78% for all of them, and cannot see a
  tranche list, a tax rule or a lease at all.
- **Its reference point is arbitrary.** SHAP explains the gap to the
  model's average prediction over a background sample, here 35.09%: the mean
  over deals drawn uniformly from wide ranges (exit multiples of 5-20x on a
  10x entry, growth of 0-15%). So the default deal's "exit multiple −7.8
  points" means "against a random deal exiting at 12.5x", which is nobody's
  question.
- **Its chart did not add up.** The values sum to the prediction less
  35.09% (to 1.6e-7), but the waterfall started from a hard-coded 18%, so its
  bars ended at 3.7% beside a prediction line at 20.8%.
- **Weight.** 1.9 s to import, 250 MB of memory once imported and 375 MB at
  peak after one explanation. The free API has 512 MB in all, and the
  simulation's size caps are set to it (`api/limits.py`).

## What the Drivers view could not say

Rank correlation is unitless and does not add up, and it credits a driver
with whatever its correlated neighbours do. On the default deal:

| Driver | Rank correlation with IRR | Its own effect on the worst 5% (IRR points) | on the best 5% |
|---|---|---|---|
| Growth | +0.89 | −9.51 | +8.29 |
| Exit multiple | +0.88 | −10.49 | +6.96 |
| Interest rate | −0.47 | −0.50 | +0.40 |
| EBITDA shock | +0.44 | −0.08 | +0.08 |
| Gross margin | +0.41 | −0.22 | +0.12 |

The rate, the margin and the one-year shock look like real drivers in the
first column. They are correlated with growth and the exit multiple (the
Settings' correlation matrix), and by themselves move the tails by well under
a point. With no floating facility the rate's own effect is exactly zero
while its rank correlation stays about −0.4. The view stays on the screen,
because co-movement is real information; the explanation sits beside it.

## What ships

For the worst and the best 5% of paths: the simulation's IRR with every
driver at its mean, each driver's contribution, and the tail's mean IRR,

    18.56 − 9.51 − 10.49 − 0.50 − 0.22 − 0.08 = −2.24   (worst 5%)
    18.56 + 8.29 +  6.96 + 0.40 + 0.12 + 0.08 = 34.41   (best 5%)

- **Exact.** The 32 answers a Shapley value needs come from rerunning the
  simulation's core on the draws it already made, so there is no fitted
  model and no error term. `tests/test_driver_explanations.py` holds the sum
  to 1e-12 for a deal sized by percentages, five tranche structures, tax
  rules and leases, and checks the explained figure against the paths.
- **Light.** At the caps (100,000 paths, 15 years, 12 facilities, every tax
  rule, 10 interest passes) the run took 1.6 s and the explanation 2.3 s,
  with a lower peak in memory (163 MB traced against the run's 234 MB): it
  works in slices of half the run's size and reads at most 2,500 paths a
  tail. The default run's explanation takes 0.03 s. The worst ratio is a
  50,000-path run (two whole tails of 2,500 paths, 32 times each: 3.2 times
  the run), so a run that itself took over 20 seconds is answered without
  explanations (`api/limits.py`), and they can never cause a timeout.
- **No model logic changed.** It calls the simulation's core with chosen
  draws; the pinned simulation and the golden snapshot are untouched, and no
  engine version moves (a new output only).

## Limits, said on the screen or here

- It explains the simulation's five uncertain drivers, not the deal's
  choices (leverage, price, capex). Those are the exit sensitivity grid, the
  heatmap and the Live sliders.
- Where two drivers interact (a low exit multiple hurts more on a company
  that also shrank), Shapley's rule shares the joint part equally between
  them. The total is exact; the split of an interaction is a convention.
- The starting point is the simulation's IRR at the mean assumptions, which
  differs a little from the deal model's IRR (findings 12 and 13).
- Above 50,000 paths a tail is read at 2,500 evenly spaced paths; the answer
  gives both counts, and the mean of those paths is within a tenth of a point
  of the whole tail's (tested).
