#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from selfmorph.action_space import build_action_space  # noqa: E402
from selfmorph.benchmarks import (  # noqa: E402
    eval_max_new_tokens, load_benchmark, model_wants_qwen_think_seed, render_prompt,
)
from selfmorph.executor import execute_anchor  # noqa: E402
from selfmorph.core.br_kv_dedup import br_kv_dedup, kv_cache_bytes  # noqa: E402
from selfmorph.final_eval import (  # noqa: E402
    compute_benchmark_prior, compute_domain_prior, compute_fixed_anchor, policy_actions,
)
from selfmorph.model import ModelConfig, load_frozen_model  # noqa: E402
from selfmorph.selector import build_selector  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

LOAD_KWARGS: dict[str, dict] = {
    "arc_challenge": {"split": "train"},
    "gsm8k": {"split": "train"},
    "openbookqa": {"split": "train"},
}

class _Meter:
    """Instrument ``model.forward`` for prefill/decode split + memory sampling.

    Only ONE ``torch.cuda.synchronize()`` is taken, at the prefill->decode
    boundary, plus one at the very end. The decode loop is otherwise left
    async, so the measured decode throughput is the real one.
    """

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self) -> None:
        self.n_forward = 0
        self.prefill_s = 0.0
        self.mem_samples: list[int] = []
        self.cache = None

    def __enter__(self) -> "_Meter":
        self._orig = self.model.forward

        def patched(*a, **kw):
            out = self._orig(*a, **kw)
            if self.n_forward == 0 and torch.cuda.is_available():

                torch.cuda.synchronize()
                self.prefill_s = time.perf_counter() - self._t0
            self.n_forward += 1
            pkv = kw.get("past_key_values")
            if pkv is not None:
                self.cache = pkv
            if torch.cuda.is_available():
                self.mem_samples.append(int(torch.cuda.memory_allocated()))
            return out

        patched.__wrapped__ = self._orig  # type: ignore[attr-defined]
        self.model.forward = patched  # type: ignore[method-assign]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.total_s = time.perf_counter() - self._t0
        else:
            self.total_s = time.perf_counter() - self._t0
        self.model.forward = self._orig  # type: ignore[method-assign]

def _load_policy_actions(config, training_pt: Path, checkpoint_path: Path | None, anchors, policies):
    """Return {policy: {sample_index: anchor_id}} from the loaded selector."""
    payload = torch.load(training_pt, map_location="cpu", weights_only=False)
    hidden = config["model"]["hidden_size"]
    ntl = payload["trajectory"].shape[1]
    sel = None
    if checkpoint_path is not None:
        sel = build_selector(config["selector"], anchors, hidden, ntl).to("cpu")
        st = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sel.load_state_dict(st["model_state"])
    dp = compute_domain_prior(payload, anchors)
    bp = compute_benchmark_prior(payload, anchors)
    fr = compute_fixed_anchor(payload, "REPEAT", anchors)
    fb = compute_fixed_anchor(payload, "BRANCH", anchors)
    acts = policy_actions(payload, "eval", sel, anchors, dp, fr, fb,
                          device="cpu", benchmark_prior=bp) if sel is not None else {}
    mask = [s == "eval" for s in payload["splits"]]
    idx_full = [i for i, m in enumerate(mask) if m]
    out: dict[str, dict[int, int]] = {}
    for p in policies:
        if p not in acts:
            raise SystemExit(f"policy {p!r} unavailable (need --selector-checkpoint for {p})")

        out[p] = {idx_full[k]: int(v) for k, v in enumerate(acts[p]) if v is not None}
    meta = {
        "n_eval840": len(idx_full),
        "fixed_rep_id": int(fr),
        "fixed_br_id": int(fb),
        "benchmark_prior": {k: int(v) for k, v in bp.actions.items()},
    }
    return payload, out, meta

def _summarise(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/runtime.json")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--training-data-pt",
                    default="artifacts/training_data/qwen35_27b/train_dev.pt")
    ap.add_argument("--selector-checkpoint",
                    default="artifacts/checkpoints/qwen35_27b/best_checkpoint.pt")
    ap.add_argument("--policies", default="base_lm,fixed_rep,fixed_br,selfmorph")
    ap.add_argument("--n-per-dataset", type=int, default=0,
                    help="0 uses the full eval split; >0 keeps a stratified subset")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeats", type=int, default=1,
                    help="timed repeats per (sample, policy); the row stores the mean over "
                         "repeats plus a `std` dict and `reps` for auditability")
    ap.add_argument("--warmup", type=int, default=0,
                    help="untimed warm-up runs per (sample, policy) before the timed repeats")
    ap.add_argument("--br-kv-dedup", dest="br_kv_dedup", action="store_true", default=True,
                    help="lossless KV cache batch deduplication for BRANCH anchors outside their "
                         "branch span")
    ap.add_argument("--no-br-kv-dedup", dest="br_kv_dedup", action="store_false",
                    help="disable branch KV cache deduplication")
    ap.add_argument("--rotate-order", action="store_true", default=True,
                    help="rotate the policy measurement order per sample so no policy is "
                         "systematically measured first (enabled by default)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ["SELFMORPH_BR_KV_DEDUP"] = "1" if args.br_kv_dedup else "0"

    config = json.loads((PACKAGE_ROOT / args.config).read_text(encoding="utf-8"))
    num_layers = config["model"]["num_layers"]
    anchors = build_action_space(config)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    payload, acts, sel_meta = _load_policy_actions(
        config, PACKAGE_ROOT / args.training_data_pt,
        PACKAGE_ROOT / args.selector_checkpoint if args.selector_checkpoint else None, anchors, policies,
    )
    ctx_ids = payload["context_ids"]
    datasets = payload["datasets"]

    eval_idx = sorted({i for p in policies for i in acts[p]})
    if args.n_per_dataset > 0:
        per: dict[str, list[int]] = {}
        for i in eval_idx:
            per.setdefault(datasets[i], []).append(i)
        keep: list[int] = []
        for ds, lst in sorted(per.items()):
            stride = max(1, len(lst) // args.n_per_dataset)
            keep.extend(lst[::stride][: args.n_per_dataset])
        eval_idx = sorted(keep)
    eval_idx = [i for k, i in enumerate(eval_idx) if k % args.num_shards == args.shard_id]

    print(f"[cost] policies={policies} eval samples(this shard)={len(eval_idx)} "
          f"fixed_rep=a{sel_meta['fixed_rep_id']} fixed_br=a{sel_meta['fixed_br_id']} "
          f"n_selectors={sel_meta['n_selectors']}", flush=True)

    model, tokenizer = load_frozen_model(
        device=args.device,
        model_config=ModelConfig(
            model_path=args.model_path, num_layers=num_layers,
            hidden_size=config["model"]["hidden_size"],
            torch_dtype=config["model"].get("torch_dtype", "bfloat16"),
            trust_remote_code=config["model"].get("trust_remote_code", False),
        ),
    )
    w_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    footprint = {
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "weights_gb": w_bytes / 2**30,
        "resident_after_load_gb": (torch.cuda.memory_allocated() / 2**30) if torch.cuda.is_available() else None,
    }
    print(f"[cost] base model: {footprint['n_params']/1e9:.3f}B params, "
          f"weights={footprint['weights_gb']:.2f}GB, resident_after_load="
          f"{footprint['resident_after_load_gb']:.2f}GB", flush=True)
    append_think_seed = model_wants_qwen_think_seed(model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    bench_cache: dict[str, dict] = {}
    rows: list[dict] = []
    n_skipped = 0
    skipped_by_ds: dict[str, int] = {}
    wanted_by_ds: dict[str, int] = {}
    for i in eval_idx:
        wanted_by_ds[datasets[i]] = wanted_by_ds.get(datasets[i], 0) + 1
    for i in eval_idx:
        ds = datasets[i]
        cid = ctx_ids[i]
        qid = cid.split("::", 1)[1] if "::" in cid else cid
        if ds not in bench_cache:
            bench_cache[ds] = {s.qid: s for s in load_benchmark(ds, **LOAD_KWARGS.get(ds, {}))}
        sample = bench_cache[ds].get(qid)
        if sample is None:

            n_skipped += 1
            skipped_by_ds[ds] = skipped_by_ds.get(ds, 0) + 1
            continue
        prompt = render_prompt(tokenizer, sample.messages, append_think_seed=append_think_seed)
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        P = int(enc["input_ids"].shape[1])
        max_new = eval_max_new_tokens(ds, config.get("manifest_default_max_new", 32))

        order = policies
        if args.rotate_order and len(policies) > 1:
            k = i % len(policies)
            order = policies[k:] + policies[:k]

        for p in order:
            aid = acts[p].get(i)
            if aid is None:
                continue
            a = anchors.by_id[aid]
            reps: list[dict] = []

            use_dedup = bool(args.br_kv_dedup and a.type == "BRANCH")
            span = (int(a.layer_span[0]), int(a.layer_span[1])) if use_dedup else (0, 0)
            for rep_i in range(args.warmup + args.repeats):

                gc.collect()
                ctx = (br_kv_dedup(K=int(a.K), span=span) if use_dedup
                       else contextlib.nullcontext({}))
                with ctx as dstat, _Meter(model) as m:
                    execute_anchor(model, tokenizer, enc, a, num_layers=num_layers,
                                   action_cfg=config["action_space"], max_new_tokens=max_new,
                                   use_br_kv_dedup=False)
                if use_dedup and dstat.get("fired", 0) == 0:
                    raise SystemExit(
                        f"--br-kv-dedup was requested but never fired for anchor a{aid} "
                        f"(type={a.type} span={a.layer_span} K={a.K}) -- refusing to emit a "
                        f"table that silently mixes a patched and an unpatched path.")
                if rep_i < args.warmup:
                    continue

                gen = int(m.n_forward)
                decode_steps = max(0, gen - 1)
                decode_s = max(0.0, m.total_s - m.prefill_s)
                kv = None
                if m.cache is not None:
                    try:
                        kv = int(kv_cache_bytes(m.cache))
                    except Exception:
                        kv = None
                avg_mem = (statistics.fmean(m.mem_samples) if m.mem_samples else None)
                peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
                reps.append({
                    "gen_tokens": float(gen), "decode_steps": float(decode_steps),
                    "n_forward": float(m.n_forward),
                    "hit_cap": bool(gen >= max_new),
                    "prefill_s": m.prefill_s, "decode_s": decode_s, "total_s": m.total_s,
                    "tps_decode": (decode_steps / decode_s) if (decode_steps > 0 and decode_s > 0) else None,
                    "tps_total": (gen / m.total_s) if (gen > 0 and m.total_s > 0) else None,
                    "prefill_tps": (P / m.prefill_s) if m.prefill_s > 0 else None,
                    "peak_mem_gb": (peak / 2**30) if peak else None,
                    "avg_mem_gb": (avg_mem / 2**30) if avg_mem else None,
                    "kv_mb": (kv / 2**20) if kv is not None else None,
                })
            if not reps:
                continue

            def _avg(key):
                vals = [x[key] for x in reps if x[key] is not None]
                return statistics.fmean(vals) if vals else None

            def _sd(key):
                vals = [x[key] for x in reps if x[key] is not None]
                return statistics.pstdev(vals) if len(vals) > 1 else 0.0

            KEYS = ("gen_tokens", "decode_steps", "n_forward", "prefill_s", "decode_s", "total_s",
                    "tps_decode", "tps_total", "prefill_tps",
                    "peak_mem_gb", "avg_mem_gb", "kv_mb")
            row = {
                "policy": p, "context_id": cid, "dataset": ds,
                "anchor_id": aid, "anchor_type": a.type,
                "span": list(a.layer_span) if a.layer_span else None, "r": a.r, "K": a.K,
                "analytic_cost": float(payload["costs"][i][aid]),
                "prompt_len": P, "max_new_tokens": max_new,
                "hit_cap": reps[-1]["hit_cap"],
                "n_repeats": len(reps),
                "reps": reps,
                "std": {k: _sd(k) for k in KEYS},
                "rel_std_tps_decode": (_sd("tps_decode") / _avg("tps_decode")) if _avg("tps_decode") else None,
                "rel_std_total_s": (_sd("total_s") / _avg("total_s")) if _avg("total_s") else None,
            }
            for k in KEYS:
                row[k] = _avg(k)
            row["n_forward"] = int(round(row["n_forward"]))
            row["gen_tokens"] = int(round(row["gen_tokens"]))
            row["decode_steps"] = int(round(row["decode_steps"]))
            rows.append(row)
        if (len(rows) % (len(policies) * 25)) < len(policies):
            r = rows[-1]
            print(f"[cost] {len(rows)} rows | {ds}::{qid[:28]} P={P} M={max_new} "
                  f"tps_dec={r['tps_decode']:.1f} avg_mem={r['avg_mem_gb']:.2f}GB", flush=True)

    per_policy: dict[str, dict] = {}
    for p in policies:
        sub = [r for r in rows if r["policy"] == p]
        if not sub:
            continue
        per_policy[p] = {
            "n": len(sub),
            "anchors_used": sorted({r["anchor_id"] for r in sub}),
            "tps_decode": _summarise([r["tps_decode"] for r in sub]),
            "tps_total": _summarise([r["tps_total"] for r in sub]),
            "prefill_tps": _summarise([r["prefill_tps"] for r in sub]),
            "total_s": _summarise([r["total_s"] for r in sub]),
            "prefill_s": _summarise([r["prefill_s"] for r in sub]),
            "gen_tokens": _summarise([float(r["gen_tokens"]) for r in sub]),
            "prompt_len": _summarise([float(r["prompt_len"]) for r in sub]),
            "peak_mem_gb": _summarise([r["peak_mem_gb"] for r in sub]),
            "avg_mem_gb": _summarise([r["avg_mem_gb"] for r in sub]),
            "kv_mb": _summarise([r["kv_mb"] for r in sub]),

            "n_repeats": (sub[0].get("n_repeats") if sub else None),
            "within_cell_rel_std_tps_decode": _summarise([r.get("rel_std_tps_decode") for r in sub]),
            "within_cell_rel_std_total_s": _summarise([r.get("rel_std_total_s") for r in sub]),
            "analytic_cost": _summarise([r["analytic_cost"] for r in sub]),
            "hit_cap_rate": statistics.fmean([1.0 if r["hit_cap"] else 0.0 for r in sub]),
        }

    paired: dict[str, dict] = {}
    if "base_lm" in policies and any(r["policy"] == "base_lm" for r in rows):
        base_by = {(r["context_id"]): r for r in rows if r["policy"] == "base_lm"}
        for p in policies:
            if p == "base_lm":
                continue
            sp = [(base_by[r["context_id"]], r) for r in rows
                  if r["policy"] == p and r["context_id"] in base_by]
            if not sp:
                continue

            paired[p] = {
                "n_pairs": len(sp),
                "mean_total_time_ratio": statistics.fmean([t["total_s"] / b["total_s"] for b, t in sp]),
                "mean_tps_decode_ratio": statistics.fmean([t["tps_decode"] / b["tps_decode"] for b, t in sp]),
                "mean_tps_total_ratio": statistics.fmean([t["tps_total"] / b["tps_total"] for b, t in sp]),
                "mean_prefill_ratio": statistics.fmean([t["prefill_s"] / b["prefill_s"] for b, t in sp]),
                "mean_gen_tokens_ratio": statistics.fmean([t["gen_tokens"] / max(1, b["gen_tokens"]) for b, t in sp]),
                "mean_peak_mem_ratio": statistics.fmean([t["peak_mem_gb"] / b["peak_mem_gb"] for b, t in sp]),
                "mean_avg_mem_ratio": statistics.fmean([t["avg_mem_gb"] / b["avg_mem_gb"] for b, t in sp]),
                "mean_kv_ratio": statistics.fmean([t["kv_mb"] / b["kv_mb"] for b, t in sp]),
                "mean_analytic_cost_ratio": statistics.fmean([t["analytic_cost"] / b["analytic_cost"] for b, t in sp]),
            }
    per_dataset: dict[str, dict] = {}
    for ds in sorted({r["dataset"] for r in rows}):
        per_dataset[ds] = {p: {
            "n": len([r for r in rows if r["policy"] == p and r["dataset"] == ds]),
            "tps_decode": _summarise([r["tps_decode"] for r in rows
                                      if r["policy"] == p and r["dataset"] == ds]),
            "avg_mem_gb": _summarise([r["avg_mem_gb"] for r in rows
                                      if r["policy"] == p and r["dataset"] == ds]),
        } for p in policies}

    out = {
        "schema_version": "selfmorph-paper-cost-v1",
        "protocol": ("eval-840 sample set, production execution path "
                     "(selfmorph.executor.execute_anchor), production budget (eval_max_new_tokens), "
                     "greedy decoding"),
        "selection_source": "selfmorph.final_eval.policy_actions (identical to "
                            "scripts/eval_five_policy_840.py) on the persisted trajectory tensor",
        "n_eval840_total": sel_meta["n_eval840"],
        "n_samples_this_shard": len(eval_idx),
        "n_rows": len(rows),
        "n_skipped_unloadable": n_skipped,
        "coverage": {
            "wanted_by_dataset": dict(sorted(wanted_by_ds.items())),
            "skipped_by_dataset": dict(sorted(skipped_by_ds.items())),
            "full_coverage": (n_skipped == 0),
        },
        "shard": {"id": args.shard_id, "num_shards": args.num_shards},
        "n_per_dataset": args.n_per_dataset,
        "selection_meta": sel_meta,
        "br_kv_dedup": bool(args.br_kv_dedup),
        "model_footprint": footprint,
        "per_policy": per_policy,
        "paired_vs_base_lm": paired,
        "per_dataset": per_dataset,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[cost] wrote {args.out} ({len(rows)} rows, skipped={n_skipped})", flush=True)
    for p, v in per_policy.items():
        print(f"[cost] {p:<14} n={v['n']:<4} anchors={v['anchors_used']} "
              f"tps_decode={v['tps_decode']['mean']:.2f} total_s={v['total_s']['mean']:.3f} "
              f"avg_mem={v['avg_mem_gb']['mean']:.3f}GB kv={v['kv_mb']['mean']:.1f}MB", flush=True)

if __name__ == "__main__":
    main()
