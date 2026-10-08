"""Train and evaluate the registered models (PLAN.md 5.1).

    python -m ml.evaluation evaluate [--model ID]          # rewrite the cards from the committed files
    python -m ml.evaluation evaluate --check               # fail when a card is not current
    python -m ml.evaluation train ID --out DIR [--samples N] [--epochs N]

``train`` never touches the committed files: it trains into ``DIR`` and
writes the new card beside them (DIR/cards, DIR/docs), so a person reviews
the card before committing the files. GitHub Actions runs both (ml.yml).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from ml.evaluation import card

# The surrogate's full training run: generate_data's defaults
SURROGATE_SAMPLES = 100_000
SURROGATE_PATHS = 2_000


def evaluation_module(model_id: str):
    return importlib.import_module(card.entry(model_id)["evaluation"])


def model_card(model_id: str, base: Path | None = None) -> dict:
    """The card for the files in the repository (or in ``base``)."""
    module = evaluation_module(model_id)
    evaluation = module.evaluate(str(base)) if base else module.evaluate()
    return card.build(card.entry(model_id), module.CARD, evaluation, base)


def differences(old, new, tolerance: float = 1e-3, where: str = "") -> list[str]:
    """Where two cards differ: numbers by more than ``tolerance`` (another
    platform's numpy may move the last digits), anything else at all."""
    if isinstance(old, bool) or isinstance(new, bool) or not (
            isinstance(old, (int, float)) and isinstance(new, (int, float))):
        if isinstance(old, dict) and isinstance(new, dict):
            out = []
            for k in sorted(set(old) | set(new)):
                if k not in old or k not in new:
                    out.append(f"{where}.{k}: only in the {'new' if k in new else 'committed'} card")
                else:
                    out += differences(old[k], new[k], tolerance, f"{where}.{k}")
            return out
        if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
            return [d for i, (a, b) in enumerate(zip(old, new)) for d in differences(a, b, tolerance, f"{where}[{i}]")]
        return [] if old == new else [f"{where}: {old!r} -> {new!r}"]
    return [] if abs(old - new) <= tolerance else [f"{where}: {old} -> {new}"]


def stale(model_id: str) -> list[str]:
    """Why the committed card (JSON or Markdown) is not what the files give."""
    fresh = model_card(model_id)
    json_path, md_path = card.paths(model_id)
    if not json_path.exists():
        return [f"{json_path} is missing"]
    committed = json.loads(json_path.read_text(encoding="utf-8"))
    found = differences(committed, fresh)
    markdown = md_path.read_text(encoding="utf-8") if md_path.exists() else None
    if markdown != card.render(committed):
        found.append(f"{md_path} is not ml/cards/{model_id}.json rendered")
    return found


def train(model_id: str, out: Path, samples: int, epochs: int) -> dict:
    """Train ``model_id`` into ``out`` and return the card for what it made."""
    out.mkdir(parents=True, exist_ok=True)
    if model_id == "anomaly_detector":
        from ml.anomaly_detector import train_detector
        train_detector(str(out))
    elif model_id == "surrogate":
        from ml.surrogate.generate_data import generate
        from ml.surrogate.train import train as train_network
        generate(n_samples=samples, n_per_call=SURROGATE_PATHS, out_dir=str(out))
        train_network(base=str(out), max_epochs=epochs)
    else:
        raise SystemExit(f"no training recipe for {model_id}")
    new = model_card(model_id, out)
    card.write(new, out / "cards", out / "docs")
    return new


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ml.evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--model", action="append")
    ev.add_argument("--check", action="store_true")
    tr = sub.add_parser("train")
    tr.add_argument("model")
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--samples", type=int, default=SURROGATE_SAMPLES)
    tr.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args(argv)

    if args.command == "train":
        new = train(args.model, args.out, args.samples, args.epochs)
        print(card.render(new))
        return 0
    ids = args.model or [m["id"] for m in card.registry()["models"]]
    if args.check:
        problems = [f"{i}: {p}" for i in ids for p in stale(i)]
        for p in problems:
            print(p)
        if problems:
            print("A model card is not current: run python -m ml.evaluation evaluate and commit the cards.")
        return 1 if problems else 0
    for i in ids:
        card.write(model_card(i))
        print(f"wrote the {i} card")
    return 0


if __name__ == "__main__":
    sys.exit(main())
