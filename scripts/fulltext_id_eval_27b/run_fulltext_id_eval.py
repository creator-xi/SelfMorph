#!/usr/bin/env python3
"""Completed-response evaluation on Qwen3.5-27B.

Runs Base, fixed REPEAT, fixed BRANCH, Benchmark Prior, and SelfMorph on
ARC-Challenge, BBH, and OpenBookQA, then scores the full generation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from selfmorph.action_space import build_action_space  # noqa: E402
from selfmorph.benchmarks import load_benchmark, render_prompt  # noqa: E402
from selfmorph.benchmarks.prompts import model_wants_qwen_think_seed  # noqa: E402
from selfmorph.executor import execute_anchor  # noqa: E402
from selfmorph.final_eval import compute_benchmark_prior, compute_fixed_anchor  # noqa: E402
from selfmorph.model import ModelConfig, load_frozen_model  # noqa: E402
from selfmorph.selector import build_selector  # noqa: E402
from selfmorph.selection import _batched_selector_argmax, benchmark_unit_for  # noqa: E402
from selfmorph.split_manifest import dataset_source_splits, lookup_split  # noqa: E402
from selfmorph.trajectory import load_trajectory_spec, probe_with_prefix, stack_trajectory  # noqa: E402

from scorers_fulltext import score_sample_fulltext  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

POLICIES = ["base_lm", "fixed_rep", "fixed_br", "benchmark_prior", "selfmorph"]

def _eval_samples(ds: str) -> list[tuple[str, object]]:
    """Enumerate the (ds, EvalSample) rows whose frozen split == "eval"."""
    out: list[tuple[str, object]] = []
    for src_split in sorted(dataset_source_splits(ds)):
        try:
            rows = load_benchmark(ds, split=src_split)
        except TypeError:
            rows = load_benchmark(ds)
        for s in rows:
            if lookup_split(ds, src_split, s.qid) == "eval":
                out.append((ds, s))
    return out

def _collect_eval_rows() -> list[tuple[str, object]]:
    """arc_challenge + openbookqa + every BBH subtask present in the frozen
    manifest (the 19-subtask ID pool)."""
    from selfmorph.split_manifest import manifest_datasets

    targets = []
    for ds in sorted(manifest_datasets()):
        if ds in ("arc_challenge", "openbookqa") or ds.startswith("bbh:"):
            targets.append(ds)
    rows: list[tuple[str, object]] = []
    for ds in targets:
        rows.extend(_eval_samples(ds))
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--train-dev-pt", required=True,
                        help="Oracle tensor used to choose Fixed-REPEAT, Fixed-BR, and Bench Prior")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new", type=int, default=1024)
    args = parser.parse_args()

    config = json.load(open(args.config))
    all_rows = _collect_eval_rows()
    my_rows = [r for i, r in enumerate(all_rows) if i % args.num_shards == args.shard_id]
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(my_rows)}/{len(all_rows)} eval contexts", flush=True)

    model_cfg = ModelConfig(model_path=args.model_path, num_layers=config["model"]["num_layers"],
                            hidden_size=config["model"]["hidden_size"],
                            torch_dtype=config["model"].get("torch_dtype", "bfloat16"),
                            trust_remote_code=config["model"].get("trust_remote_code", False))
    model, tokenizer = load_frozen_model(device=args.device, model_config=model_cfg)
    append_think_seed = model_wants_qwen_think_seed(model)

    anchors = build_action_space(config)
    action_cfg = config["action_space"]
    lambda_cost = action_cfg["lambda_cost"]
    num_layers = config["model"]["num_layers"]
    base_id = anchors.base_id
    train_payload = torch.load(args.train_dev_pt, map_location="cpu", weights_only=False)
    fixed_rep_id = compute_fixed_anchor(train_payload, "REPEAT", anchors)
    fixed_br_id = compute_fixed_anchor(train_payload, "BRANCH", anchors)
    bp_actions = compute_benchmark_prior(train_payload, anchors).actions
    del train_payload

    traj_spec = load_trajectory_spec(config)
    layer_ids = traj_spec.layer_ids()
    hidden_size = config["model"]["hidden_size"]
    num_traj_layers = len(layer_ids)
    probe_depth = max(layer_ids) + 1
    rep_depths = {int(a.layer_span[1]) for a in anchors if a.type == "REPEAT"}
    prefix_depths = tuple(sorted(rep_depths | {probe_depth}))

    ckpt_path = args.selector_checkpoint
    if not Path(ckpt_path).is_absolute():
        ckpt_path = str(PACKAGE_ROOT / ckpt_path)
    selector = build_selector(config["selector"], anchors, hidden_size, num_traj_layers).to(args.device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    selector.load_state_dict(state["model_state"])
    selector.eval()
    print(f"[shard {args.shard_id}] loaded selector {ckpt_path} "
          f"(noop={base_id}, fixed_rep={fixed_rep_id}, fixed_br={fixed_br_id}, "
          f"bp_actions={ {k: v for k, v in bp_actions.items() if k in ('arc_challenge', 'bbh', 'openbookqa')} })",
          flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard{args.shard_id}of{args.num_shards}.jsonl"

    n_contexts, n_generate_calls = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ds, sample in my_rows:
            unit = benchmark_unit_for(ds)
            prior_id = bp_actions.get(unit)
            prompt_text = render_prompt(tokenizer, sample.messages, append_think_seed=append_think_seed)

            enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            probe = probe_with_prefix(model, enc, layer_ids=layer_ids, prefix_depths=prefix_depths)
            traj = stack_trajectory(probe["hiddens"], layer_ids).float()
            selfmorph_id = _batched_selector_argmax(selector, anchors, traj.unsqueeze(0), [0], args.device)[0]

            policy_ids = {
                "base_lm": base_id, "fixed_rep": fixed_rep_id, "fixed_br": fixed_br_id,
                "benchmark_prior": prior_id, "selfmorph": selfmorph_id,
            }
            unique_ids = sorted({a for a in policy_ids.values() if a is not None})

            results_by_id: dict[int, dict] = {}
            for aid in unique_ids:
                anchor = anchors.by_id[aid]
                res = execute_anchor(model, tokenizer, enc, anchor, num_layers=num_layers,
                                     action_cfg=action_cfg, max_new_tokens=args.max_new,
                                     prefix=probe["prefix"])
                n_generate_calls += 1
                text = res["text"]
                try:
                    score = score_sample_fulltext(text, sample, ds)
                except Exception as exc:  # noqa: BLE001
                    score = 0.0
                    print(f"[shard {args.shard_id}] SCORE ERROR {ds}::{sample.qid}: {exc!r}", flush=True)
                cost = anchor.analytic_cost
                reward = score - lambda_cost * (cost - 1.0)
                results_by_id[aid] = {"text": text, "latency_s": res["latency_s"], "score": score,
                                      "cost": cost, "reward": reward, "anchor_type": anchor.type,
                                      "layer_span": list(anchor.layer_span) if anchor.layer_span else None,
                                      "r": anchor.r, "K": anchor.K}

            for policy, aid in policy_ids.items():
                r = results_by_id[aid]
                rec = {"context_id": f"{ds}::{sample.qid}", "dataset": ds, "unit": unit, "policy": policy,
                       "anchor_id": aid, "anchor_type": r["anchor_type"], "layer_span": r["layer_span"],
                       "r": r["r"], "K": r["K"], "score": r["score"], "cost": r["cost"], "reward": r["reward"],
                       "text": r["text"], "latency_s": r["latency_s"], "max_new": args.max_new}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_contexts += 1

    print(f"[shard {args.shard_id}/{args.num_shards}] done: {n_contexts} contexts, "
          f"{n_generate_calls} generate() calls -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
