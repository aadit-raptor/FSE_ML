"""CI fails if a model gets worse (PLAN.md 5.1).

    python -m ops.model_gate --base origin/main

Compares every model card (``ml/cards/<id>.json``) with the base branch's
copy of it. A pull request fails when, in any evaluation set, overall or in
any region, a statistic moves the wrong way by more than the tolerance the
base card gives it, or a group that had results loses them. A new model has
nothing to compare with and passes; a model dropped from ml/registry.json
is reported, not failed (a deliberate removal says why in its pull
request); a registered model without a card fails. The cards themselves are
kept current by tests/test_model_cards.py, so a retrained model can't skip
this by leaving its old card. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional

ROOT = Path(__file__).resolve().parent.parent
CARDS = "ml/cards"
REGISTRY = "ml/registry.json"


def worse(better: str, old: float, new: float, tolerance: float) -> bool:
    return new < old - tolerance if better == "higher" else new > old + tolerance


def _group_problems(where: str, old: Mapping, new: Optional[Mapping], specs: Mapping[str, Mapping]) -> list[str]:
    if "model" not in old:
        return []                                   # nothing to keep
    if new is None or "model" not in new:
        return [f"{where}: had results on {old['cases']} cases, now none"]
    out = []
    for metric, before in old["model"].items():
        spec = specs.get(metric)
        after = new["model"].get(metric)
        if spec is None or before is None:
            continue
        if after is None:
            out.append(f"{where} {metric}: was {before}, now missing")
        elif worse(spec["better"], before, after, spec["tolerance"]):
            out.append(f"{where} {metric}: {before} -> {after} ({spec['better']} is better, "
                       f"tolerance {spec['tolerance']})")
    return out


def compare(model_id: str, old: Mapping, new: Mapping) -> list[str]:
    """Why ``new`` is worse than ``old``; empty when it isn't. Tolerances
    are the base card's, so a pull request can't widen its own."""
    specs = {m["id"]: m for m in old["evaluation"]["metrics"]}
    found = []
    for set_name, old_set in old["evaluation"]["sets"].items():
        new_set = new["evaluation"]["sets"].get(set_name)
        if new_set is None:
            found.append(f"{model_id} {set_name}: set removed")
            continue
        found += _group_problems(f"{model_id} {set_name} overall", old_set["overall"], new_set["overall"], specs)
        for region, group in old_set["by_region"].items():
            found += _group_problems(f"{model_id} {set_name} {region}", group,
                                     new_set["by_region"].get(region), specs)
    return found


def _git_show(ref: str, path: str) -> Optional[str]:
    done = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8")
    return done.stdout if done.returncode == 0 else None


def _ids(registry_text: Optional[str]) -> list[str]:
    return [m["id"] for m in json.loads(registry_text)["models"]] if registry_text else []


def gate(base: str, show=_git_show, read=lambda p: (ROOT / p).read_text(encoding="utf-8")) -> tuple[list[str], list[str]]:
    """``(problems, notes)`` for every card against the base ref."""
    problems, notes = [], []
    base_ids, ids = _ids(show(base, REGISTRY)), _ids(read(REGISTRY))
    for model_id in base_ids:
        old_text = show(base, f"{CARDS}/{model_id}.json")
        if model_id not in ids:
            notes.append(f"{model_id}: removed from the registry")
            continue
        if old_text is None:
            notes.append(f"{model_id}: no card on {base} to compare with")
            continue
        try:
            new_text = read(f"{CARDS}/{model_id}.json")
        except FileNotFoundError:
            problems.append(f"{model_id}: registered but has no card")
            continue
        problems += compare(model_id, json.loads(old_text), json.loads(new_text))
    notes += [f"{i}: new model, nothing to compare with" for i in ids if i not in base_ids]
    return problems, notes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ops.model_gate")
    parser.add_argument("--base", default="origin/main", help="git ref to compare with")
    args = parser.parse_args(argv)
    problems, notes = gate(args.base)
    for n in notes:
        print(n)
    for p in problems:
        print(f"::error::{p}")
    if problems:
        print("A model card got worse than on the base branch. Improve the model, or keep the files "
              "that gave the better card.")
        return 1
    print("No model card got worse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
