#!/usr/bin/env python3
"""Generate data-split manifests from downloaded benchmark datasets.

Produces two files under resources/split_manifests/:
  id_manifest.json   — in-distribution train / dev_tune / dev_select / eval
  ood_manifest.json  — unseen-benchmark sample lists

Run after downloading all datasets described in README.md:
    set -a; source .env; set +a
    python scripts/generate_splits.py [--seed 20260810]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from selfmorph.benchmarks import load_benchmark  # noqa: E402

# ── BBH subtasks (BIG-Bench Hard) ──────────────────────────────────────────

BBH_SUBTASKS = [
    "date_understanding", "disambiguation_qa", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "penguins_in_a_table",
    "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
]

# ── ID dataset spec: source splits and target sizes per split ──────────────
# eval / dev_tune / dev_select counts come from the paper; rest → train.

ID_SPEC = {
    "arc_challenge": dict(src=["train", "test", "validation"], eval=150, dev_tune=100, dev_select=100),
    "openbookqa":    dict(src=["train", "test", "validation"], eval=150, dev_tune=100, dev_select=100),
    "gsm8k":         dict(src=["train", "test"],               eval=250, dev_tune=250, dev_select=250),
    "math500":       dict(src=["test"],                        eval=100, dev_tune=25,  dev_select=25),
}
BBH_PER_SUBTASK = dict(eval=10, dev_tune=7, dev_select=7)

# ── OOD dataset spec ───────────────────────────────────────────────────────

OOD_SPEC = {
    "twoWikiMultihop": dict(src="validation", n=100, domain="Multi-hop QA"),
    "hellaswag":       dict(src="validation", n=100, domain="Commonsense Reasoning"),
    "truthfulqa_mc1":  dict(src=None,         n=100, domain="Truthfulness"),
    "gpqa_diamond":    dict(src=None,         n=100, domain="Graduate Science"),
}


# ── helpers ────────────────────────────────────────────────────────────────

def _load(dataset: str, source_split: str | None = None):
    try:
        return list(load_benchmark(dataset, split=source_split)) if source_split else list(load_benchmark(dataset))
    except TypeError:
        return list(load_benchmark(dataset))


def _collect_id_rows(dataset: str, source_splits: list[str]) -> list[dict]:
    rows = []
    for sp in source_splits:
        for s in _load(dataset, sp):
            rows.append(dict(dataset=dataset, source_split=sp, loader_qid=s.qid, max_new=32))
    return rows


def _allocate(rows: list[dict], n_eval: int, n_dev_tune: int, n_dev_select: int, rng: random.Random):
    rng.shuffle(rows)
    needed = n_eval + n_dev_tune + n_dev_select
    if len(rows) < needed:
        raise ValueError(f"need {needed} samples but only {len(rows)} available")
    i = 0
    out = {}
    for name, n in [("eval", n_eval), ("dev_tune", n_dev_tune), ("dev_select", n_dev_select)]:
        out[name] = rows[i:i + n]; i += n
    # The raw train pool is intentionally large. In practice we filter it after
    # oracle collection to balance the action distribution across morphology types.
    out["train"] = rows[i:]
    return out


# ── generators ─────────────────────────────────────────────────────────────

def generate_id(seed: int) -> dict:
    rng = random.Random(seed)
    splits: dict[str, list] = {"train": [], "dev_tune": [], "dev_select": [], "eval": []}

    for ds, spec in ID_SPEC.items():
        print(f"  loading {ds} ...", flush=True)
        rows = _collect_id_rows(ds, spec["src"])
        parts = _allocate(rows, spec["eval"], spec["dev_tune"], spec["dev_select"], rng)
        for k in splits:
            splits[k].extend(parts[k])

    for sub in BBH_SUBTASKS:
        ds = f"bbh:{sub}"
        print(f"  loading {ds} ...", flush=True)
        rows = _collect_id_rows(ds, ["test"])
        parts = _allocate(rows, BBH_PER_SUBTASK["eval"], BBH_PER_SUBTASK["dev_tune"],
                          BBH_PER_SUBTASK["dev_select"], rng)
        for k in splits:
            splits[k].extend(parts[k])

    return {"seed": seed, "splits": splits}


def generate_ood(seed: int) -> dict:
    rng = random.Random(seed)
    splits: dict[str, list] = {}
    for ds, spec in OOD_SPEC.items():
        print(f"  loading {ds} ...", flush=True)
        samples = _load(ds, spec["src"])
        rng.shuffle(samples)
        splits[ds] = [
            dict(dataset=ds, loader_qid=s.qid, max_new=1024, paper_domain=spec["domain"])
            for s in samples[:spec["n"]]
        ]
    return {"seed": seed, "splits": splits}


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "resources" / "split_manifests")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating ID manifest ...", flush=True)
    id_data = generate_id(args.seed)
    id_path = args.output_dir / "id_manifest.json"
    with id_path.open("w", encoding="utf-8") as f:
        json.dump(id_data, f, indent=2, ensure_ascii=False)
    counts = {k: len(v) for k, v in id_data["splits"].items()}
    print(f"  → {id_path}  ({sum(counts.values())} samples: {counts})")

    print("Generating OOD manifest ...", flush=True)
    ood_data = generate_ood(args.seed)
    ood_path = args.output_dir / "ood_manifest.json"
    with ood_path.open("w", encoding="utf-8") as f:
        json.dump(ood_data, f, indent=2, ensure_ascii=False)
    n = sum(len(v) for v in ood_data["splits"].values())
    print(f"  → {ood_path}  ({n} samples)")


if __name__ == "__main__":
    main()
