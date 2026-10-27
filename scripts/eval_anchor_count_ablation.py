#!/usr/bin/env python3
"""Evaluate the nested 3 / 9 / 25 action pools.

For each base model x tier (3 / 9 / 25 actions):
  * load the selector trained by scripts/run_anchor_count_ablation.py,
  * evaluate it on the eval split (840 samples) and report
    mean_score / mean_cost / mean_reward / domain-macro reward and the
    per-benchmark-unit score table,
  * report the tier's per-sample argmax-utility ORACLE upper bound,
  * report the Base LM (BASE) reference point,
  * report the selected-action distribution and the tier's anchor list.

The 25-action pool is the full action space, so its checkpoint is checked
against the released selector (state_dict max-abs diff and identical actions).

Usage:
    python scripts/eval_anchor_count_ablation.py \\
        --out artifacts/results/anchor_count_ablation_eval.json
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
    ANCHOR_TIERS, EXP1_CKPT_DIR, anchor_descriptions, build_tier_config, dump_json,
    selector_actions, eval_actions_on_split, load_config, load_payload, load_trial_selector,
    mean_train_reward, model_specs, oracle_upper_bound, load_production_selector, slice_payload_anchors,
    tier_selection_report,
)

def _state_dict_maxdiff(sel_a, sel_b) -> float:
    sd_a, sd_b = sel_a.state_dict(), sel_b.state_dict()
    worst = 0.0
    for k in sd_a:
        worst = max(worst, float((sd_a[k].float() - sd_b[k].float()).abs().max()))
    return worst

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen35_27b,qwen35_4b,gemma4_12b")
    ap.add_argument("--tiers", default=",".join(ANCHOR_TIERS))
    ap.add_argument("--ckpt-dir", default=str(EXP1_CKPT_DIR))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="artifacts/results/anchor_count_ablation_eval.json")
    args = ap.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    ckpt_root = Path(args.ckpt_dir)

    out: dict = {
        "experiment": "anchor_count_ablation",
        "split": args.split,
        "tiers": tiers,
        "notes": "3-action pool nested in the 9-action pool nested in the 25-action pool (nested); only the anchor columns of the persisted oracle are "
                 "subsetted, every base keeps its current production selector config. Each backbone is "
                 "evaluated with its released selector seed.",
        "models": {},
    }

    for spec in model_specs([m.strip() for m in args.models.split(",") if m.strip()]):
        base_config = load_config(spec)
        raw_payload = load_payload(spec)
        hidden = base_config["model"]["hidden_size"]
        nlayers = raw_payload["trajectory"].shape[1]
        vec = mean_train_reward(raw_payload)
        sel_report = tier_selection_report(base_config, vec)

        n_eval = sum(1 for s in raw_payload["splits"] if s == args.split)
        model_entry: dict = {
            "label": spec.label,
            "config": spec.config,
            "train_dev_pt": spec.train_dev_pt,
            "selector_checkpoint": spec.checkpoint,
            "hidden_size": hidden,
            "num_trajectory_layers": nlayers,
            "n_eval_samples": n_eval,
            "tier_selection": sel_report,
            "tiers": {},
        }

        prod_sel = load_production_selector(spec, build_tier_config(base_config, "actions25", vec)[1], hidden, nlayers)
        prod_full_anchors = build_tier_config(base_config, "actions25", vec)[1]
        prod_payload = slice_payload_anchors(raw_payload, list(range(len(prod_full_anchors))))
        prod_actions = selector_actions(prod_payload, prod_full_anchors, prod_sel, args.split, args.device)

        for tier in tiers:
            sub_cfg, sub_anchors, col_ids = build_tier_config(base_config, tier, vec)
            payload = slice_payload_anchors(raw_payload, col_ids)
            model_dir = ckpt_root / spec.key
            sel = load_trial_selector(model_dir, tier, sub_anchors, hidden, nlayers)

            actions = selector_actions(payload, sub_anchors, sel, args.split, args.device)
            metrics = eval_actions_on_split(payload, args.split, actions)
            oracle = oracle_upper_bound(payload, args.split)
            base_lm = eval_actions_on_split(payload, args.split, [sub_anchors.base_id] * n_eval)
            counts = Counter(actions)
            descs = {d["id"]: d for d in anchor_descriptions(sub_anchors)}

            entry: dict = {
                "num_anchors": len(sub_anchors),
                "n_params": sum(p.numel() for p in sel.parameters()),
                "anchors": anchor_descriptions(sub_anchors),
                "selfmorph": metrics,
                "oracle_upper_bound_argmax_utility": oracle,
                "base_lm_noop": base_lm,
                "selected_action_counts": {
                    str(aid): {"count": c, "anchor": descs[aid]} for aid, c in sorted(counts.items())
                },
            }

            if tier == "actions25":
                diff = _state_dict_maxdiff(sel, prod_sel)
                entry["self_check_vs_released_selector"] = {
                    "state_dict_max_abs_diff": diff,
                    "actions_identical": actions == prod_actions,
                    "metrics_identical": metrics == eval_actions_on_split(prod_payload, args.split, prod_actions),
                }

            model_entry["tiers"][tier] = entry
            print(f"[{spec.key}][{tier}] n_anchors={len(sub_anchors)} "
                  f"score={metrics['mean_score']:.4f} cost={metrics['mean_cost']:.4f} "
                  f"reward={metrics['mean_reward']:.4f} DMR={metrics['domain_macro_reward']:.4f} "
                  f"oracle(score={oracle['mean_score']:.4f}, cost={oracle['mean_cost']:.4f}) "
                  f"base_score={base_lm['mean_score']:.4f}")

        out["models"][spec.key] = model_entry

    dump_json(Path(args.out), out)

if __name__ == "__main__":
    main()
