# WORKFLOW.md — the development cycle

How every change to FSE_ML is made, by Claude or by you. It merges the
framework from **ECC (Everything Claude Code,
https://github.com/affaan-m/ECC)** with the rules this project already
proved: PLAN.md's one task per session and handoff, CLAUDE.md's working
rules (parity, verify by output, mutation checks).

Where ECC and this project disagree, this file says which wins and why.
CLAUDE.md imports this file, so every session reads it.

ECC is **optional tooling**: the cycle works with or without it installed.
Where an ECC command or agent helps, it's named in brackets, e.g.
[`ecc:planner`]. Without ECC, use the built-in equivalent listed next to it.

---

## The cycle

ECC's cycle is research → plan → test → build → review → verify → remember.
Here each step has a concrete output, so nothing lives only in chat.

| # | Step | Output | Tools |
|---|---|---|---|
| 0 | **Pick the task** | The PLAN.md task number | PLAN.md "How to use this plan" |
| 1 | **Research and reuse** | A list of what exists already | Code search, vendor docs, package registries |
| 2 | **Plan** | A task plan at the top of the PR | Plan mode [`ecc:planner`, `ecc:architect`] |
| 3 | **Test first** | Failing tests (red) | [`tdd-workflow` skill, `ecc:tdd-guide`] |
| 4 | **Build** | Code that makes them pass (green), then tidied | — |
| 5 | **Review** | Findings fixed or recorded | `/code-review`, `/security-review` [`ecc:code-reviewer`, `ecc:python-reviewer`, `ecc:typescript-reviewer`, `ecc:security-reviewer`] |
| 6 | **Verify** | Evidence for every "Done when" line | Tests, CI, staging, the browser |
| 7 | **Ship** | PR merged, staging brought level, production checked | PR template, DEPLOY.md |
| 8 | **Remember** | CLAUDE.md, PLAN.md and docs updated; handoff prompt | [`/learn-eval`] |

### 0. Pick the task
One PLAN.md task per session and per PR, lowest open number whose "Needs"
are done. A bug found in passing is recorded (CLAUDE.md "Model findings", or
a spawned task), not fixed on the side.

### 1. Research and reuse (ECC "Research & Reuse")
Before writing new code:
- **This repo first.** Most things have a pattern here already: money keys,
  `RUN_PATHS`, jobs, migrations, `type-*` classes. Search before inventing.
- **Vendor docs second.** For Next.js 16, read `web/node_modules/next/dist/docs/`,
  because it differs from what models remember. For other libraries, read the
  version pinned in the lock files.
- **Proven libraries over hand-rolled code**, *if* they fit the free plans,
  the security policy (CSP hosts, no new trackers) and the licence.

### 2. Plan (ECC "Plan First")
Write a short plan before code, and put it in the PR description:
- the files that change and the files that must not (model logic, golden
  snapshot);
- the risks: free limits, migrations, anything the deployed copy still running
  must survive;
- the tests that will prove each "Done when" line.

ECC suggests a PRD, architecture doc, system design, tech doc and task list
for every feature. **This project uses PLAN.md's task entry as the PRD and the
task list**, and `docs/ARCHITECTURE.md` (PLAN.md 11.4) for design decisions.
A task that needs a real design decision writes it there, or asks the user.

### 3. Test first (ECC TDD: red → green → improve)
- Write the test, run it, **see it fail for the right reason**. That is this
  project's mutation check done up front. Keep the mutation check for tests
  written after the code.
- Test what the user sees change: IRR, MOIC, stored rows, response fields.
  **Verify by output, not by render** (CLAUDE.md).
- Name tests for the behaviour: `test_deal_in_thousands_matches_millions_times_1000`.
- Arrange, act, assert, in that order.
- Where a test goes: unit tests beside the module's other tests in `tests/`;
  API and database tests through the real app and a throwaway database
  (`tests/conftest.py`); user flows in `web/e2e/`.

**Model logic is the exception.** `core/`, `lbo_engine/` and `simulation/`
are pinned by the golden snapshot. A change there needs the user's approval
first (CLAUDE.md "Keep model logic as is"), then an explicit test pinning the
new numbers.

### 4. Build
ECC's style rules, applied to **new and changed code**:
- Keep it simple; no abstractions before a second use needs them.
- Functions under about 50 lines; files under 800 lines (soft ceiling; tests
  and generated files may exceed it with a reason); nesting no deeper than 4.
- Handle errors explicitly; no silent `except: pass`; user-facing messages say
  what to do.
- Validate at the edges: Pydantic schemas on the API, typed client on the web.
- No secrets, no debug prints, no `console.log`, no hardcoded currency sign.
- Prefer returning new values over changing inputs in place.

**Not applied to existing model code.** `lbo_engine/model.py` (929 lines) and
`simulation/vectorized_simulation.py` (907) exceed ECC's ceiling, but
splitting them is a logic-adjacent change that risks parity. Leave them until
a task already rewrites them.

**Rejected from ECC:** its "consistent API response envelope"
(`{success, data, error}`). This API answers the typed result directly and the
web client is generated from that schema; wrapping it would break every
screen for no gain. Errors already use FastAPI's standard `detail` body.

### 5. Review (ECC "Code Review", fresh context)
Review in a fresh context, not in the session that wrote the code:
`/code-review` [or `ecc:code-reviewer`, plus `ecc:python-reviewer` or
`ecc:typescript-reviewer` for their files].

**Also run a security review** (`/security-review` [`ecc:security-reviewer`])
when the change touches any of: sign-in, accounts or ownership, user input,
database queries or migrations, files, outside APIs, cryptography or backups,
money or limits. That covers most PLAN.md tasks.

Severity (ECC):

| Level | Meaning | Action |
|---|---|---|
| CRITICAL | Security hole or data loss | Blocks the merge |
| HIGH | A bug or a real quality problem | Fix before merge |
| MEDIUM | Maintainability | Fix if cheap, or record it |
| LOW | Style | Optional |

Also check the project's own list: money keys for new money figures,
`RUN_PATHS` for new model endpoints, `PUBLIC_PATHS` never widened by
accident, nothing personal or deal-related in logs, CSP hosts for new
third-party calls, the three places a new Python package goes (CLAUDE.md
"Gotchas").

### 6. Verify (ECC "verification loop")
Every line of the task's "Done when" gets evidence: test output, a CI run, a
staging check, a screenshot or a `javascript_tool` read. The local gates, in
order:

```bash
.venv/Scripts/python.exe -m pytest
npm --prefix web run lint
npm --prefix web run typecheck
npm --prefix web run build
PW_CHANNEL=msedge FSE_AUTH_DEV=1 npm --prefix web run test:e2e
```

If the API schemas changed: refresh `web/openapi.json` and `schema.d.ts`
(CLAUDE.md "Commands"). If a build or CI fails, fix the cause
[`/build-fix`, `ecc:build-error-resolver`]. Never skip a hook, loosen a test
or regenerate the golden file to go green.

**CI gates (PLAN.md 0.3).** On every pull request a machine checks what it
can of this cycle:

| Gate | Where | Fails when |
|---|---|---|
| Python coverage floor | `core` and `ml` jobs (`ops/coverage_gate.py`) | the job's total is below its line in `.coverage-floor`, or the PR lowers a floor |
| Changed lines (ECC's 80%) | `ml` job (`diff-cover`) | under 80% of the Python lines the PR adds or changes in the app's packages are covered |
| PR title | `pr.yml` (`ops/pr_title.py`) | the title doesn't start with an ECC type (step 7) |

Each job's summary shows the coverage table and the changed-line report.
When coverage rises well above the floor, the job prints the figure to
raise it to: raise `.coverage-floor` in the same PR (never lower it; if code
is deleted and the total drops, add tests elsewhere rather than lowering).
Locally:

```bash
.venv/Scripts/python.exe -m pytest --cov --cov-report=term    # totals per file
.venv/Scripts/python.exe -m coverage xml
.venv/Scripts/python.exe -m diff_cover.diff_cover_tool coverage.xml --compare-branch=origin/main --fail-under=80
```

The local total differs from CI's when tests skip (no database, no
Upstash, no ML packages); the floors are CI's figures. The web has no unit
tests: the browser tests are its proof, and a unit-test framework comes
with the first task whose web logic is worth unit-testing. Mutation checks
still apply to every new test: coverage says a line ran, not that a test
would notice it breaking.

### 7. Ship
- **Branch** from `main`, named `<type>/<short-name>`: `feat/locale`,
  `fix/scheduler-retries`, `docs/dev-cycle`.
- **Commits** start with an ECC type, `feat:`, `fix:`, `refactor:`, `docs:`,
  `test:`, `chore:`, `perf:` or `ci:`, then a sentence. The body explains
  *why* and *how it was verified* (the project's rule, kept). Keep the
  attribution trailer: ECC's installer turns it off by default, and this
  project keeps it on.
- **PR** from `.github/pull_request_template.md`: the task, the plan, the
  "Done when" evidence, the review result, the user's checks. CI green, then
  merge with "Create a merge commit".
- **PR titles** use the same types (`pr.yml` checks them): the title becomes
  the merge commit's message on `main`.
- **After the merge:** bring staging level with a pull request from `main`
  to `staging`, titled `chore: bring staging level with main`, which the
  user merges (Claude can't push to `staging` or merge). Wait for
  `staging.yml`, then check production (`/api/health` shows the new commit).
- Rollback is DEPLOY.md "Rollback".

### 8. Remember (ECC "remember and improve")
ECC keeps lessons in hook-written "instincts" under `~/.claude`. **This
project keeps them in the repo instead**, so they survive a new computer and
you can read them:
- a trap that cost time → CLAUDE.md "Gotchas";
- a decision the user shouldn't be asked again → CLAUDE.md, in its section;
- a model finding → CLAUDE.md "Model findings";
- the task ticked in PLAN.md and CLAUDE.md's status updated, **in the same PR**;
- the handoff: tell the user to start a new session, with the ready-to-paste
  prompt for the next task and its "You first" items.

ECC's `/learn-eval` may be used on top, but anything it learns that matters
for this project is copied into CLAUDE.md.

---

## Installing ECC (optional, the user does this)

Claude doesn't install or run third-party code on your machine. You do it
once, in an interactive `claude` terminal (not the desktop chat):

```bash
/plugin marketplace add https://github.com/affaan-m/ECC
/plugin install ecc@ecc
```

Then start a new session. Choices that fit this project:
- **Plugin only.** It brings agents, skills and commands. Don't also run ECC's
  `install.sh`; ECC warns against stacking install methods.
- **No hooks for now.** ECC lists Windows defects in its continuous-learning
  observer and memory vault. **ECC's hooks are on by default**
  (`ECC_HOOKS_ENABLED` defaults to `true` in its `scripts/lib/hook-flags.js`),
  so installing the plugin alone turns them on. This repository turns them
  off in the committed `.claude/settings.json`: `"env": {"ECC_HOOKS_ENABLED":
  "false", "ECC_SESSION_START_CONTEXT": "off"}` (the second stops its
  session-start context injection). Keep both when editing that file, and
  start a new session after changing them.
- **No global rules copy.** ECC's rules would apply to every project on the
  machine. This file already carries what this project uses from them.
- **Commit attribution stays on**, whatever ECC's settings say.
- To remove it later, uninstall the plugin from Claude Code's plugin manager.

**Built-in equivalents** if ECC isn't installed: plan mode (plan),
`/code-review` (review), `/security-review` (security), the `Explore` and
`Plan` agents (research), `/simplify` (tidy).

## Parallel work (ECC "worktrees")

Sessions stay one task each. Two independent tasks may run at once in
separate git worktrees (`git worktree add ../FSE_ML-2 -b feat/x`), each with
its own session and PR. Never run two sessions on one migration number:
migrations are numbered and must stay linear.

<!-- PLAN.md 0.3 demo: throwaway change for the PR title check; do not merge. -->
