#!/usr/bin/env python3
"""Per-benchmark aggregation of unseen-benchmark evaluation results.

Evaluates the 4 unseen benchmarks (`twoWikiMultihop`, `hellaswag`, `truthfulqa_mc1`,
`gpqa_diamond`, 100 samples each = 400 samples per backbone) across the 4 policies
(`base_lm`, `fixed_rep`, `fixed_br`, `selfmorph`).

Usage:
    python scripts/aggregate_ood_eval.py \
        --results-root artifacts/ood_eval \
        --models qwen35_27b,qwen35_4b,gemma4_12b \
        --out artifacts/results/ood_eval_per_benchmark.json
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

POLICIES = ["base_lm", "fixed_rep", "fixed_br", "selfmorph"]
BENCH_ROW_ORDER = [
    "twoWikiMultihop", "hellaswag", "truthfulqa_mc1", "gpqa_diamond",
]

def _mean(xs: list[float]) -> float:
    return st.mean(xs)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="artifacts/ood_eval")
    parser.add_argument("--models", default="qwen35_27b,qwen35_4b,gemma4_12b")
    parser.add_argument("--out", default="artifacts/results/ood_eval_per_benchmark.json")
    args = parser.parse_args()

    out: dict = {"policies": POLICIES, "models": {}}
    for model in args.models.split(","):
        rows = []
        for fp in sorted(glob.glob(f"{args.results_root}/{model}/shard*.jsonl")):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("group") in BENCH_ROW_ORDER:
                        rows.append(rec)

        acc: dict[str, dict[str, dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {"n": 0, "score_sum": 0.0, "reward_sum": 0.0})
        )
        for r in rows:
            a = acc[r["group"]][r["policy"]]
            a["n"] += 1
            a["score_sum"] += r["score"]
            a["reward_sum"] += r["reward"]

        def stats(group: str, policy: str) -> dict:
            a = acc[group][policy]
            return {
                "n": a["n"],
                "mean_score": a["score_sum"] / a["n"] if a["n"] else None,
                "mean_reward": a["reward_sum"] / a["n"] if a["n"] else None,
            }

        bench_rows = {
            g: {p: stats(g, p) for p in POLICIES}
            for g in BENCH_ROW_ORDER if g in acc
        }

        macro = {}
        for p in POLICIES:
            macro[p] = {
                "mean_score": _mean([bench_rows[g][p]["mean_score"] for g in bench_rows]),
                "mean_reward": _mean([bench_rows[g][p]["mean_reward"] for g in bench_rows]),
            }

        overall = {}
        for p in POLICIES:
            xs = [r["score"] for r in rows if r["policy"] == p]
            yr = [r["reward"] for r in rows if r["policy"] == p]
            overall[p] = {"n": len(xs), "mean_score": _mean(xs), "mean_reward": _mean(yr)}

        n_per_policy = {p: sum(1 for r in rows if r["policy"] == p) for p in POLICIES}
        out["models"][model] = {
            "n_rows": len(rows), "n_contexts": len(rows) // len(POLICIES),
            "n_per_policy": n_per_policy,
            "benchmark_rows": bench_rows,
            "group_macro_mean": macro, "overall_sample_mean": overall,
        }
        print(f"=== {model} ===  rows={len(rows)} contexts={len(rows)//len(POLICIES)}")
        hdr = f"{'benchmark':20s} {'n':>3s} " + "".join(f"{p:>32s}" for p in POLICIES)
        print(hdr)
        for g in BENCH_ROW_ORDER:
            if g not in bench_rows:
                continue
            line = f"{g:20s} {bench_rows[g]['base_lm']['n']:3d} "
            for p in POLICIES:
                s = bench_rows[g][p]
                line += f"score {s['mean_score']:.4f} rew {s['mean_reward']:.4f}   "
            print(line)
        print(f"{f'{len(bench_rows)}-row macro':20s} {'':3s} " + "".join(
            f"score {macro[p]['mean_score']:.4f} rew {macro[p]['mean_reward']:.4f}   " for p in POLICIES
        ))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {args.out}")

if __name__ == "__main__":
    main()
