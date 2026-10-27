#!/usr/bin/env python3
"""Base, fixed REPEAT, fixed BRANCH, and SelfMorph on unseen benchmarks.

Each sample is probed once. Distinct selected actions are executed once and
shared across policies. Benchmark Prior is not defined on these benchmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from selfmorph.action_space import build_action_space  # noqa: E402
from selfmorph.benchmarks import (  # noqa: E402
    eval_max_new_tokens, load_benchmark, model_wants_qwen_think_seed,
    render_prompt, score_sample,
)
from selfmorph.executor import execute_anchor  # noqa: E402
from selfmorph.final_eval import compute_fixed_anchor  # noqa: E402
from selfmorph.model import ModelConfig, load_frozen_model  # noqa: E402
from selfmorph.selector import build_selector  # noqa: E402
from selfmorph.selection import _batched_selector_argmax  # noqa: E402
from selfmorph.trajectory import load_trajectory_spec, probe_with_prefix, stack_trajectory  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

def _flatten_manifest(manifest: dict) -> list[dict]:
    rows = []
    for group in sorted(manifest["splits"]):
        for row in manifest["splits"][group]:
            rows.append({"group": group, **row})
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True, help="Overrides config's model_path_env resolution")
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--train-dev-pt", required=True,
                        help="Oracle tensor used to choose Fixed-REPEAT and Fixed-BR on the training split")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(args.manifest, "r", encoding="utf-8") as f:
        ood_manifest = json.load(f)
    all_rows = _flatten_manifest(ood_manifest)
    my_rows = [r for i, r in enumerate(all_rows) if i % args.num_shards == args.shard_id]
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(my_rows)}/{len(all_rows)} rows assigned", flush=True)

    model_cfg = ModelConfig(
        model_path=args.model_path, num_layers=config["model"]["num_layers"],
        hidden_size=config["model"]["hidden_size"], torch_dtype=config["model"].get("torch_dtype", "bfloat16"),
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
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
          f"(fixed_rep_id={fixed_rep_id}, fixed_br_id={fixed_br_id}, base_id={base_id})", flush=True)

    needed_datasets = sorted({r["dataset"] for r in my_rows})
    qid_maps: dict[str, dict[str, object]] = {}
    for ds in needed_datasets:
        samples = load_benchmark(ds)
        qid_maps[ds] = {s.qid: s for s in samples}
        print(f"[shard {args.shard_id}] loaded {ds}: {len(samples)} rows", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard{args.shard_id}of{args.num_shards}.jsonl"

    n_contexts, n_generate_calls = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in my_rows:
            ds, qid, group = row["dataset"], row["loader_qid"], row["group"]
            sample = qid_maps[ds].get(qid)
            if sample is None:
                print(f"[shard {args.shard_id}] MISSING qid={qid} in dataset={ds}, skipping", flush=True)
                continue

            prompt_text = render_prompt(tokenizer, sample.messages, append_think_seed=append_think_seed)
            enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            max_new = eval_max_new_tokens(ds, config.get("manifest_default_max_new", 32))

            probe = probe_with_prefix(model, enc, layer_ids=layer_ids, prefix_depths=prefix_depths)
            traj = stack_trajectory(probe["hiddens"], layer_ids).float()
            chosen = _batched_selector_argmax(selector, anchors, traj.unsqueeze(0), [0], args.device)
            selfmorph_id = chosen[0]

            policy_anchor_ids = {
                "base_lm": base_id, "fixed_rep": fixed_rep_id, "fixed_br": fixed_br_id, "selfmorph": selfmorph_id,
            }
            unique_ids = sorted(set(policy_anchor_ids.values()))

            results_by_id: dict[int, dict] = {}
            for aid in unique_ids:
                anchor = anchors.by_id[aid]
                res = execute_anchor(
                    model, tokenizer, enc, anchor, num_layers=num_layers,
                    action_cfg=action_cfg, max_new_tokens=max_new,
                    prefix=probe["prefix"],
                )
                n_generate_calls += 1
                score = score_sample(res["text"], sample, ds)
                cost = anchor.analytic_cost
                reward = score - lambda_cost * (cost - 1.0)
                results_by_id[aid] = {
                    "text": res["text"], "latency_s": res["latency_s"], "score": score,
                    "cost": cost, "reward": reward, "anchor_type": anchor.type,
                    "layer_span": list(anchor.layer_span) if anchor.layer_span else None,
                    "r": anchor.r, "K": anchor.K,
                }

            context_id = f"{ds}::{qid}"
            for policy, aid in policy_anchor_ids.items():
                r = results_by_id[aid]
                rec = {
                    "context_id": context_id, "group": group, "dataset": ds, "policy": policy,
                    "anchor_id": aid, "anchor_type": r["anchor_type"], "layer_span": r["layer_span"],
                    "r": r["r"], "K": r["K"], "score": r["score"], "cost": r["cost"], "reward": r["reward"],
                    "text": r["text"], "latency_s": r["latency_s"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_contexts += 1

    print(f"[shard {args.shard_id}/{args.num_shards}] done: {n_contexts} contexts, "
          f"{n_generate_calls} real generate() calls (vs {n_contexts * 4} if not deduped) -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
