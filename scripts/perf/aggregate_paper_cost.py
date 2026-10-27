#!/usr/bin/env python3
"""Merge sharded cost measurements into one JSON and Markdown table.

Reads ``paper_cost_shard*.json`` from ``bench_paper_cost.py`` and pools the
per-sample rows. Every ratio is ``policy / base_lm``
computed **per sample and then averaged** (paired), so prompt-length and
generation-length differences between samples cancel out.

Usage
-----
    python scripts/perf/aggregate_paper_cost.py \\
        --inputs 'artifacts/results/inference_speedup/paper_cost_shard*.json' \\
        --out-json artifacts/results/inference_speedup/paper_cost_eval840.json \\
        --out-md   artifacts/results/inference_speedup/paper_cost_eval840.md
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

POLICY_ORDER = ["base_lm", "fixed_rep", "fixed_br", "benchmark_prior", "selfmorph"]
DISPLAY = {
    "base_lm": "Base LM (BASE)",
    "fixed_rep": "Fixed REPEAT",
    "fixed_br": "Fixed BR",
    "benchmark_prior": "Benchmark Prior",
    "selfmorph": "SelfMorph",
}

def _s(vals: list) -> dict:
    v = [x for x in vals if x is not None]
    if not v:
        return {"n": 0}
    return {
        "n": len(v),
        "mean": statistics.fmean(v),
        "median": statistics.median(v),
        "std": statistics.pstdev(v) if len(v) > 1 else 0.0,
        "min": min(v),
        "max": max(v),
    }

def _aggregate(rows: list[dict], policies: list[str]) -> dict:
    per_policy: dict[str, dict] = {}
    for p in policies:
        sub = [r for r in rows if r["policy"] == p]
        if not sub:
            continue
        per_policy[p] = {
            "n": len(sub),
            "anchors_used": sorted({r["anchor_id"] for r in sub}),
            "anchor_types": sorted({r["anchor_type"] for r in sub}),
            "gen_tokens": _s([float(r["gen_tokens"]) for r in sub]),
            "prompt_len": _s([float(r["prompt_len"]) for r in sub]),
            "hit_cap_rate": statistics.fmean([1.0 if r["hit_cap"] else 0.0 for r in sub]),
            "tps_decode": _s([r["tps_decode"] for r in sub]),
            "tps_total": _s([r["tps_total"] for r in sub]),
            "prefill_s": _s([r["prefill_s"] for r in sub]),
            "decode_s": _s([r["decode_s"] for r in sub]),
            "total_s": _s([r["total_s"] for r in sub]),
            "peak_mem_gb": _s([r["peak_mem_gb"] for r in sub]),
            "avg_mem_gb": _s([r["avg_mem_gb"] for r in sub]),
            "kv_mb": _s([r["kv_mb"] for r in sub]),
            "analytic_cost": _s([r["analytic_cost"] for r in sub]),
        }

    paired: dict[str, dict] = {}
    base_by = {r["context_id"]: r for r in rows if r["policy"] == "base_lm"}
    for p in policies:
        if p == "base_lm" or not base_by:
            continue
        sp = [(base_by[r["context_id"]], r) for r in rows
              if r["policy"] == p and r["context_id"] in base_by]
        if not sp:
            continue
        paired[p] = {
            "n_pairs": len(sp),
            "total_time_ratio": statistics.fmean([t["total_s"] / b["total_s"] for b, t in sp]),
            "tps_decode_ratio": statistics.fmean([t["tps_decode"] / b["tps_decode"] for b, t in sp]),
            "tps_total_ratio": statistics.fmean([t["tps_total"] / b["tps_total"] for b, t in sp]),
            "prefill_ratio": statistics.fmean([t["prefill_s"] / b["prefill_s"] for b, t in sp]),
            "gen_tokens_ratio": statistics.fmean([t["gen_tokens"] / max(1, b["gen_tokens"]) for b, t in sp]),
            "peak_mem_ratio": statistics.fmean([t["peak_mem_gb"] / b["peak_mem_gb"] for b, t in sp]),
            "avg_mem_ratio": statistics.fmean([t["avg_mem_gb"] / b["avg_mem_gb"] for b, t in sp]),
            "kv_ratio": statistics.fmean([t["kv_mb"] / b["kv_mb"] for b, t in sp]),
            "analytic_cost_ratio": statistics.fmean([t["analytic_cost"] / b["analytic_cost"] for b, t in sp]),
        }
    for p, r in paired.items():
        r["measured_over_analytic__kv"] = r["kv_ratio"] / r["analytic_cost_ratio"]
        r["measured_over_analytic__tps"] = r["tps_decode_ratio"] / r["analytic_cost_ratio"]

    per_dataset: dict[str, dict] = {}
    for ds in sorted({r["dataset"] for r in rows}):
        per_dataset[ds] = {}
        for p in policies:
            sub = [r for r in rows if r["policy"] == p and r["dataset"] == ds]
            if not sub:
                continue
            per_dataset[ds][p] = {
                "n": len(sub),
                "tps_decode": _s([r["tps_decode"] for r in sub]),
                "avg_mem_gb": _s([r["avg_mem_gb"] for r in sub]),
                "gen_tokens": _s([float(r["gen_tokens"]) for r in sub]),
                "hit_cap_rate": statistics.fmean([1.0 if r["hit_cap"] else 0.0 for r in sub]),
            }
    return {"per_policy": per_policy, "paired_vs_base_lm": paired, "per_dataset": per_dataset}

def _markdown(payload: dict, policies: list[str]) -> str:
    pp = payload["per_policy"]
    pd_ = payload["paired_vs_base_lm"]
    fp = payload.get("model_footprint", {})

    def f(v, nd=2):
        return "n/a" if v is None else f"{v:.{nd}f}"

    L: list[str] = []
    L.append("# Inference cost: time and memory")
    L.append("")
    L.append(f"- Rows measured: **{payload.get('n_rows', '?')}** (samples times policies)")
    L.append("- Execution: `selfmorph.executor.execute_anchor`, greedy decoding")
    L.append("- Policies: `selfmorph.final_eval.policy_actions`")
    if fp:
        L.append(f"- Backbone: **{fp['n_params'] / 1e9:.3f}B parameters, bf16 weights {fp['weights_gb']:.2f} GB**"
                 f" (resident after load {fp['resident_after_load_gb']:.2f} GB)")
    L.append("")
    L.append("Decode speed is `(generated length - 1) / decode time`. "
             "End-to-end time includes prompt prefill. "
             "Ratios are means of per-sample pairs (`policy / base`).")
    L.append("")
    L.append("| Policy | Actions | n | Generated length | Prompt length | **Decode speed (tok/s)** | Peak memory (GB) | Average memory (GB) | KV (MB) | End-to-end (s) | analytic_cost |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in policies:
        if p not in pp:
            continue
        v = pp[p]
        an = ",".join(f"a{a}" for a in v["anchors_used"])
        L.append(
            f"| {DISPLAY.get(p, p)} | {an} | {v['n']} | {v['gen_tokens']['mean']:.1f} | "
            f"{v['prompt_len']['mean']:.0f} | **{v['tps_decode']['mean']:.2f}** | "
            f"{v['peak_mem_gb']['mean']:.3f} | {v['avg_mem_gb']['mean']:.3f} | "
            f"{v['kv_mb']['mean']:.1f} | {v['total_s']['mean']:.3f} | {v['analytic_cost']['mean']:.4f} |"
        )
    L.append("")
    L.append("## Paired ratios against the base model (1.000 = same as base)")
    L.append("")
    L.append("| Policy | Decode-speed ratio | End-to-end time ratio | Prefill time ratio | Generated-length ratio | "
             "Peak-memory ratio | Average-memory ratio | **KV ratio** | analytic_cost ratio | **Measured KV / analytic** |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in policies:
        if p not in pd_:
            continue
        v = pd_[p]
        L.append(
            f"| {DISPLAY.get(p, p)} | {v['tps_decode_ratio']:.4f} | {v['total_time_ratio']:.4f} | "
            f"{v['prefill_ratio']:.4f} | {v['gen_tokens_ratio']:.4f} | {v['peak_mem_ratio']:.4f} | "
            f"{v['avg_mem_ratio']:.4f} | **{v['kv_ratio']:.4f}** | {v['analytic_cost_ratio']:.4f} | "
            f"**{v['measured_over_analytic__kv']:.4f}** |"
        )
    L.append("")
    L.append(f"Paired samples: {', '.join(f'{p}={pd_[p]['n_pairs']}' for p in policies if p in pd_)}")
    L.append("")
    L.append("Notes:")
    L.append("")
    L.append("1. End-to-end time mixes speed with generated length. Read it together with the length ratio. "
             "Decode speed is normalized by length.")
    L.append("2. Average memory is dominated by frozen weights "
             f"({fp.get('weights_gb', 0):.1f} GB). KV bytes are the sensitive term.")
    L.append("3. Truncation rates: "
             + ", ".join(f"{DISPLAY.get(p, p)}={pp[p]['hit_cap_rate']:.2f}" for p in policies if p in pp)
             + ".")
    L.append("")
    return "\n".join(L)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.inputs))
    if not files:
        raise SystemExit(f"no shard matched {args.inputs!r}")
    rows: list[dict] = []
    footprint = {}
    flags_seen: set = set()
    n_eval_total = None
    skipped: dict[str, int] = {}
    wanted: dict[str, int] = {}
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        rows.extend(d["rows"])
        footprint = footprint or d.get("model_footprint", {})
        if d.get("br_kv_dedup") is not None:
            flags_seen.add(bool(d["br_kv_dedup"]))
        n_eval_total = n_eval_total or d.get("n_eval840_total")
        cov = d.get("coverage", {})
        for k, v in cov.get("skipped_by_dataset", {}).items():
            skipped[k] = skipped.get(k, 0) + v
        for k, v in cov.get("wanted_by_dataset", {}).items():
            wanted[k] = wanted.get(k, 0) + v
    policies = [p for p in POLICY_ORDER if any(r["policy"] == p for r in rows)]
    payload = {
        "schema_version": "selfmorph-paper-cost-merged-v1",
        "shards": [str(f) for f in files],
        "n_rows": len(rows),
        "n_eval840": n_eval_total,
        "n_samples": len({r["context_id"] for r in rows}),
        "model_footprint": footprint,
        "br_kv_dedup": (flags_seen.pop() if len(flags_seen) == 1 else None),
        "coverage": {
            "wanted_by_dataset": dict(sorted(wanted.items())),
            "skipped_by_dataset": dict(sorted(skipped.items())),
            "n_skipped": sum(skipped.values()),
            "full_coverage": (sum(skipped.values()) == 0),
        },
    }
    payload.update(_aggregate(rows, policies))
    payload["policies"] = policies
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = _markdown(payload, policies)
    Path(args.out_md).write_text(md, encoding="utf-8")
    print(f"[merge] {len(files)} shards, {len(rows)} rows, {payload['n_samples']} samples")
    print(f"[merge] wrote {args.out_json} and {args.out_md}")
    for p in policies:
        v = payload["per_policy"][p]
        r = payload["paired_vs_base_lm"].get(p, {})
        print(f"  {DISPLAY.get(p, p):<18} n={v['n']:<4} tps={v['tps_decode']['mean']:6.2f} "
              f"gen={v['gen_tokens']['mean']:5.1f} peak={v['peak_mem_gb']['mean']:7.3f} "
              f"avg={v['avg_mem_gb']['mean']:7.3f} kv={v['kv_mb']['mean']:7.1f} "
              f"| tps_ratio={r.get('tps_decode_ratio', float('nan')):.4f} "
              f"kv_ratio={r.get('kv_ratio', float('nan')):.4f} "
              f"analytic_ratio={r.get('analytic_cost_ratio', float('nan')):.4f}")

if __name__ == "__main__":
    main()
