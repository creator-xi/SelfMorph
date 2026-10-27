#!/usr/bin/env python3
"""Evaluate the cost-coefficient sweep.

For each base model x lambda in {0, 0.01, 0.05, 0.10, 0.20, 0.50}:
  * rebuild reward(lambda) / utility(lambda) from the persisted
    `scores`/`costs` (no new oracle),
  * load the selector trained for that lambda,
  * evaluate on the eval split (840 samples): mean_score / mean_cost /
    mean_reward / domain-macro reward + per-benchmark-unit score table,
  * report the reference points AT THE SAME LAMBDA:
      - Base LM (= BASE),
      - Fixed-REPEAT / Fixed-BR (train mean-reward argmax within each family,
        recomputed for that lambda -- their argmax depends on lambda),
      - the per-sample argmax-utility ORACLE upper bound.

The λ = 0.05 checkpoint must match the released selector.

Usage:
    python scripts/eval_lambda_sweep.py \\
        --out artifacts/results/lambda_sweep_eval.json
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import torch  # noqa: E402

from ablation_common import (  # noqa: E402
    LAMBDA_SWEEP_CKPT_DIR, LAMBDAS, anchor_descriptions, dump_json, selector_actions,
    eval_actions_on_split, fixed_anchor_ids, lam_prefix, lambda_payload, load_config, load_payload,
    load_trial_selector, model_specs, oracle_upper_bound, load_production_selector,
)
from selfmorph.action_space import build_action_space  # noqa: E402

def _state_dict_maxdiff(sel_a, sel_b) -> float:
    sd_a, sd_b = sel_a.state_dict(), sel_b.state_dict()
    return max(float((sd_a[k].float() - sd_b[k].float()).abs().max()) for k in sd_a)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen35_27b,qwen35_4b,gemma4_12b")
    ap.add_argument("--lambdas", default=",".join(f"{l:g}" for l in LAMBDAS))
    ap.add_argument("--ckpt-dir", default=str(LAMBDA_SWEEP_CKPT_DIR))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="artifacts/results/lambda_sweep_eval.json")
    args = ap.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    ckpt_root = Path(args.ckpt_dir)

    out: dict = {
        "experiment": "lambda_cost_sweep",
        "split": args.split,
        "lambdas": lambdas,
        "notes": "reward(lambda)=score-lambda*(cost-1), utility=reward-reward[BASE]; both tensors rebuilt so "
                 "training target and in-loop early stopping share the same lambda. Base T / architecture / "
                 "capacity / trajectory layer set held fixed. Reference points are recomputed per lambda "
                 "(Fixed-REPEAT/Fixed-BR argmax depends on lambda). Each backbone uses its released selector seed.",
        "models": {},
    }

    for spec in model_specs([m.strip() for m in args.models.split(",") if m.strip()]):
        base_config = load_config(spec)
        raw_payload = load_payload(spec)
        anchors = build_action_space(base_config)
        hidden = base_config["model"]["hidden_size"]
        nlayers = raw_payload["trajectory"].shape[1]
        n_eval = sum(1 for s in raw_payload["splits"] if s == args.split)
        prod_lambda = float(base_config["action_space"]["lambda_cost"])
        descs = {d["id"]: d for d in anchor_descriptions(anchors)}
        model_dir = ckpt_root / spec.key

        model_entry: dict = {
            "label": spec.label,
            "config": spec.config,
            "train_dev_pt": spec.train_dev_pt,
            "selector_checkpoint": spec.checkpoint,
            "production_lambda": prod_lambda,
            "n_eval_samples": n_eval,
            "anchors": anchor_descriptions(anchors),
            "points": {},
        }

        prod_sel = load_production_selector(spec, anchors, hidden, nlayers)
        prod_payload = lambda_payload(raw_payload, anchors, prod_lambda, stored_lambda=prod_lambda)
        prod_actions = selector_actions(prod_payload, anchors, prod_sel, args.split, args.device)

        for lam in lambdas:
            payload = lambda_payload(raw_payload, anchors, lam, stored_lambda=prod_lambda)
            sel = load_trial_selector(model_dir, lam_prefix(lam), anchors, hidden, nlayers)
            actions = selector_actions(payload, anchors, sel, args.split, args.device)
            metrics = eval_actions_on_split(payload, args.split, actions)
            oracle = oracle_upper_bound(payload, args.split)
            base_lm = eval_actions_on_split(payload, args.split, [anchors.base_id] * n_eval)

            rep_id, br_id = fixed_anchor_ids(payload, anchors)
            fixed_rep = eval_actions_on_split(payload, args.split, [rep_id] * n_eval)
            fixed_br = eval_actions_on_split(payload, args.split, [br_id] * n_eval)

            counts = Counter(actions)
            entry: dict = {
                "lambda_cost": lam,
                "n_params": sum(p.numel() for p in sel.parameters()),
                "selfmorph": metrics,
                "oracle_upper_bound_argmax_utility": oracle,
                "base_lm_noop": base_lm,
                "fixed_rep": {"anchor_id": rep_id, "anchor": descs[rep_id], **fixed_rep},
                "fixed_br": {"anchor_id": br_id, "anchor": descs[br_id], **fixed_br},
                "selected_action_counts": {
                    str(aid): {"count": c, "anchor": descs[aid]} for aid, c in sorted(counts.items())
                },
            }
            if abs(lam - prod_lambda) < 1e-12:
                diff = _state_dict_maxdiff(sel, prod_sel)
                entry["self_check_vs_released_selector"] = {
                    "state_dict_max_abs_diff": diff,
                    "actions_identical": actions == prod_actions,
                }

            model_entry["points"][f"{lam:g}"] = entry
            print(f"[{spec.key}][lambda={lam:g}] score={metrics['mean_score']:.4f} cost={metrics['mean_cost']:.4f} "
                  f"reward={metrics['mean_reward']:.4f} DMR={metrics['domain_macro_reward']:.4f} | "
                  f"base={base_lm['mean_score']:.4f} frep={fixed_rep['mean_score']:.4f}@{rep_id} "
                  f"fbr={fixed_br['mean_score']:.4f}@{br_id} | "
                  f"oracle(score={oracle['mean_score']:.4f},cost={oracle['mean_cost']:.4f})")

        out["models"][spec.key] = model_entry

    dump_json(Path(args.out), out)

if __name__ == "__main__":
    main()
