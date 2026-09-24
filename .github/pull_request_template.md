<!-- The cycle behind each section: docs/WORKFLOW.md -->

## Task
PLAN.md **X.Y** — <name>. Needs done: <list>. "You first" in place: <yes / what>.

## Plan
- Changes:
- Must not change: <model logic, golden snapshot, …>
- Risks (free limits, migrations, the deploy still running):

## Done when — evidence
- [ ] <"Done when" line> — <test name, CI run, staging check, screenshot>

## Tests
- [ ] New tests failed before the change (red), or were mutation-checked
- [ ] `pytest`, `lint`, `typecheck`, `build`, `test:e2e` pass locally
- [ ] CI gates green: coverage floor (raised if the job suggests it), 80% of changed lines, PR title with an ECC type
- [ ] `web/openapi.json` and `schema.d.ts` refreshed (if schemas changed)

## Review
- [ ] `/code-review` in a fresh context: no CRITICAL or HIGH left
- [ ] `/security-review` (sign-in, input, database, files, outside calls, money, limits): <result or "not needed: why">

## Same-PR upkeep
- [ ] PLAN.md ticked, CLAUDE.md status updated
- [ ] New gotchas and decisions recorded in CLAUDE.md
- [ ] DEPLOY.md updated (new variables, services or steps)

## For the user to check
-
