"""Check a pull request title starts with an ECC type (PLAN.md 0.3).

    PR_TITLE="feat: saved deals" python -m ops.pr_title

The title comes from the environment, never the command line, so a title
can't inject shell (pr.yml passes it as ``env``). The types are
docs/WORKFLOW.md step 7's list; a test keeps the two in step. A merge
commit's message comes from the title, so this keeps ``main``'s history typed.
Standard library only.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Optional

TYPES = ("feat", "fix", "refactor", "docs", "test", "chore", "perf", "ci")

# type, optional (scope), optional ! for a breaking change, ": ", then text
PATTERN = re.compile(rf"^({'|'.join(TYPES)})(\([a-z0-9._/-]+\))?!?: \S")


def problem(title: str) -> Optional[str]:
    """Why the title fails the check, or None when it passes."""
    if PATTERN.match(title):
        return None
    allowed = ", ".join(f"{t}:" for t in TYPES)
    return (f"PR title {title!r} must start with an ECC type ({allowed}), optionally with a "
            "scope, e.g. 'fix(api): retry gateway errors'. Edit the title; the check reruns.")


def main() -> int:
    found = problem(os.environ.get("PR_TITLE", ""))
    if found:
        print(f"::error::{found}")
        return 1
    print("PR title ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
