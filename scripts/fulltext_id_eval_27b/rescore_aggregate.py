#!/usr/bin/env python3
"""Aggregate completed-response scores.

Scores each saved generation with the whole-output scorer and prints the
per-benchmark mean for the five policies.

Usage:
    python scripts/fulltext_id_eval_27b/rescore_aggregate.py \
        [artifacts/fulltext_id_eval_27b]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from selfmorph.benchmarks import load_benchmark  # noqa: E402
from selfmorph.split_manifest import dataset_source_splits, lookup_split  # noqa: E402

import scorers_fulltext as S  # noqa: E402

POLS = ["base_lm", "fixed_rep", "fixed_br", "benchmark_prior", "selfmorph"]

def _eval_samples(ds: str):
    out = []
    for src_split in sorted(dataset_source_splits(ds)):
        try:
            rows = load_benchmark(ds, split=src_split)
        except TypeError:
            rows = load_benchmark(ds)
        for s in rows:
            if lookup_split(ds, src_split, s.qid) == "eval":
                out.append((ds, s))
    return out

def main(root: str) -> None:
    from selfmorph.split_manifest import manifest_datasets

    samples: dict[str, object] = {}
    for ds in sorted(manifest_datasets()):
        if ds == "arc_challenge" or ds == "openbookqa" or ds.startswith("bbh:"):
            for d, s in _eval_samples(ds):
                samples[f"{d}::{s.qid}"] = s

    rows = []
    for fp in sorted(Path(root).glob("shard*.jsonl")):
        for line in open(fp):
            rows.append(json.loads(line))

    agg: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    n_scored = 0
    for r in rows:
        pol = r["policy"]
        unit = r["unit"]
        ds = r["dataset"]
        sample = samples.get(r["context_id"])
        if sample is None:
            print(f"missing sample for {r['context_id']}")
            continue
        n_scored += 1
        agg[unit][pol].append(S.score_sample_fulltext(r["text"], sample, ds))

    units = ["arc_challenge", "bbh", "openbookqa"]
    print(f"scored {n_scored} rows from {root}\n")
    print(f"{'policy':16s} {'arc':>7s} {'bbh':>7s} {'obqa':>7s} {'macro':>7s}")
    for pol in POLS:
        if pol not in agg["arc_challenge"]:
            continue
        scores = {u: sum(agg[u][pol]) / len(agg[u][pol]) for u in units}
        macro = sum(scores.values()) / len(units)
        print(f"{pol:16s} {scores['arc_challenge']:.4f} {scores['bbh']:.4f} "
              f"{scores['openbookqa']:.4f} {macro:.4f}")

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "artifacts/fulltext_id_eval_27b"
    main(root)
