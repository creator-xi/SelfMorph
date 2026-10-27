#!/usr/bin/env python3
"""Evaluate selector input representations.

For each base x arm in {traj10, single, embed}:

  * load the selector (`traj10` is the released checkpoint; the broadcast arms
    come from this experiment's own checkpoints);
  * evaluate on the eval split (840 samples): `mean_score`, `mean_reward`,
    `domain_macro_reward`, `mean_cost`, the per-benchmark-unit score tables,
    and the selected-action distribution;
  * report the difference from the `traj10` arm;
  * report the (arm-independent) per-sample argmax-utility oracle upper bound
    and the Base LM / Fixed-REPEAT / Fixed-BR reference points;
  * audit the arm: input tensor shape `[n, 10, H]`, all 10 slots bit-identical
    (the broadcast claim), and parameter count identical across arms;
  * for `traj10`, check the checkpoint against the released selector.

Usage:
    .venv/bin/python scripts/input_rep_ablation/eval_input_rep_ablation.py \\
        --out artifacts/results/input_rep_ablation_eval.json
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1]))

import torch  # noqa: E402

from selfmorph.action_space import build_action_space  # noqa: E402

from ablation_common import (  # noqa: E402
    REPO_ROOT, anchor_descriptions, dump_json, selector_actions, eval_actions_on_split,
    fixed_anchor_ids, load_config, load_payload, load_trial_selector, model_specs, oracle_upper_bound,
    load_production_selector,
)
from input_rep_common import (  # noqa: E402
    ARMS, ARM_INPUT_LABEL, ARM_SLOT_SETS, BROADCAST_ARMS, CKPT_DIR, TRAJ10_CKPT_DIR, arm_slot_map,
    build_arm_payload, broadcast_gain_path, dense_layer_ids, single_layer, verify_broadcast_gain,
)

def _state_dict_maxdiff(sel_a, sel_b) -> float:
    sd_a, sd_b = sel_a.state_dict(), sel_b.state_dict()
    if list(sd_a) != list(sd_b):
        return float("inf")
    return max(float((sd_a[k].float() - sd_b[k].float()).abs().max()) for k in sd_a)

def _input_audit(payload: dict, arm: str) -> dict:
    t = payload["trajectory"]
    ref = t[:, 0:1, :].expand_as(t)
    same = bool(torch.equal(t, ref))
    slots = arm_slot_map(arm)
    audit = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "all_slots_bit_identical": same,
        "slot0_vs_broadcast_max_abs_diff": float((t - ref).abs().max()),
        "expect_all_slots_identical": arm in BROADCAST_ARMS,
    }
    if slots is not None:
        audit["slot_map"] = slots
        audit["n_distinct_slots"] = len({t[:, l, :].numpy().tobytes() for l in range(t.shape[1])})
        audit["slot_map_matches_input"] = all(
            torch.equal(t[:, l, :], t[:, slots[l], :]) for l in range(t.shape[1])
        )
        audit["expect_n_distinct_slots"] = 1 if arm in BROADCAST_ARMS else len(set(slots))
    return audit

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen35_27b,qwen35_4b,gemma4_12b")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    ap.add_argument("--traj10-ckpt-dir", default=str(TRAJ10_CKPT_DIR))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="artifacts/results/input_rep_ablation_eval.json")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ckpt_root = Path(args.ckpt_dir)
    traj10_root = Path(args.traj10_ckpt_dir)

    out: dict = {
        "experiment": "input_rep_ablation",
        "split": args.split,
        "arms": arms,
        "notes": (
            "traj10 = the default 10-layer trajectory; single = ONE layer's last-token hidden state "
            "broadcast to all 10 slots; embed = the prompt embedding (embedding-module output at the last "
            "prompt token) broadcast to all 10 slots. Every arm produces a [n,10,H] input, so encoder/trunk "
            "architecture and parameter count are unchanged; only the input information differs. Everything "
            "else is each base's current production selector config and the default data/splits. Each backbone "
            "uses its released selector seed. Because all 10 "
            "slots are identical, any depth-wise structure is unusable by construction (for a delta encoder "
            "delta_l is exactly 0)."
        ),
        "models": {},
    }

    for spec in model_specs([m.strip() for m in args.models.split(",") if m.strip()]):
        base_config = load_config(spec)
        raw_payload = load_payload(spec)
        hidden = int(base_config["model"]["hidden_size"])
        nlayers = int(raw_payload["trajectory"].shape[1])
        anchors = build_action_space(base_config)
        n_eval = sum(1 for s in raw_payload["splits"] if s == args.split)

        fixed_rep_id, fixed_br_id = fixed_anchor_ids(raw_payload, anchors)
        descs = {d["id"]: d for d in anchor_descriptions(anchors)}
        references = {
            "base_lm_noop": eval_actions_on_split(raw_payload, args.split, [anchors.base_id] * n_eval),
            "fixed_rep": {"anchor_id": fixed_rep_id, "anchor": descs[fixed_rep_id],
                          "metrics": eval_actions_on_split(raw_payload, args.split, [fixed_rep_id] * n_eval)},
            "fixed_br": {"anchor_id": fixed_br_id, "anchor": descs[fixed_br_id],
                         "metrics": eval_actions_on_split(raw_payload, args.split, [fixed_br_id] * n_eval)},
            "oracle_upper_bound_argmax_utility": oracle_upper_bound(raw_payload, args.split),
        }

        prod_sel = load_production_selector(spec, anchors, hidden, nlayers)
        prod_actions = selector_actions(raw_payload, anchors, prod_sel, args.split, args.device)
        prod_metrics = eval_actions_on_split(raw_payload, args.split, prod_actions)

        model_entry: dict = {
            "label": spec.label,
            "config": spec.config,
            "train_dev_pt": spec.train_dev_pt,
            "selector_checkpoint": spec.checkpoint,
            "hidden_size": hidden,
            "num_trajectory_layers": nlayers,
            "n_eval_samples": n_eval,
            "dense_layer_ids": [int(x) for x in base_config["trajectory"]["dense_layer_ids"]],
            "single_arm_slot_index": 4,
            "single_arm_layer": single_layer(base_config),
            "embed_arm_captured_layer": -1,
            "arm_geometry": {
                arm: {
                    "input_label": ARM_INPUT_LABEL[arm],
                    "selected_slots_0based": list(ARM_SLOT_SETS[arm]) if arm in ARM_SLOT_SETS else None,
                    "selected_slots_1based": [i + 1 for i in ARM_SLOT_SETS[arm]] if arm in ARM_SLOT_SETS else None,
                    "slot_map": arm_slot_map(arm),
                    "trajectory_layer_ids": ([dense_layer_ids(base_config)[i] for i in arm_slot_map(arm)]
                                             if arm_slot_map(arm) is not None else None),
                } for arm in arms
            },
            "selector_config": base_config["selector"],
            "references_full_pool": references,
            "released_selector": {
                "n_params": sum(p.numel() for p in prod_sel.parameters()),
                "metrics": prod_metrics,
            },
            "gain_verification": {
                arm: verify_broadcast_gain(broadcast_gain_path(arm, base_config, spec.key), base_config, arm)
                for arm in arms if arm != "traj10"
            },
            "arms": {},
        }

        for arm in arms:
            payload = build_arm_payload(raw_payload, spec, arm)
            ckpt_dir = (REPO_ROOT / spec.checkpoint).parent if arm == "traj10" else (ckpt_root / spec.key)
            prefix = "released" if arm == "traj10" else arm
            sel = (
                prod_sel
                if arm == "traj10"
                else load_trial_selector(ckpt_dir, prefix, anchors, hidden, nlayers)
            )

            actions = selector_actions(payload, anchors, sel, args.split, args.device)
            metrics = eval_actions_on_split(payload, args.split, actions)
            oracle = oracle_upper_bound(payload, args.split)
            counts = Counter(actions)

            entry: dict = {
                "arm": arm,
                "checkpoint_dir": str(ckpt_dir.relative_to(REPO_ROOT)),
                "checkpoint_prefix": prefix,
                "n_params": sum(p.numel() for p in sel.parameters()),
                "input_audit": _input_audit(payload, arm),
                "selfmorph": metrics,
                "oracle_upper_bound_argmax_utility": oracle,
                "oracle_minus_selector_mean_score": oracle["mean_score"] - metrics["mean_score"],
                "selected_action_counts": {
                    str(aid): {"count": c, "anchor": descs[aid]} for aid, c in sorted(counts.items())
                },
            }
            if arm == "traj10":
                diff = _state_dict_maxdiff(sel, prod_sel)
                entry["self_check_vs_released_selector"] = {
                    "state_dict_max_abs_diff": diff,
                    "actions_identical": actions == prod_actions,
                    "metrics_identical": metrics == prod_metrics,
                }
            model_entry["arms"][arm] = entry
            print(f"[{spec.key}][{arm}] score={metrics['mean_score']:.4f} reward={metrics['mean_reward']:.4f} "
                  f"cost={metrics['mean_cost']:.4f} DMR={metrics['domain_macro_reward']:.4f} "
                  f"| base={references['base_lm_noop']['mean_score']:.4f} "
                  f"| slots_identical={entry['input_audit']['all_slots_bit_identical']} "
                  f"| params={entry['n_params']}")

        if "traj10" in model_entry["arms"]:
            ref = model_entry["arms"]["traj10"]
            for arm, entry in model_entry["arms"].items():
                entry["delta_vs_traj10"] = {
                    "mean_score": entry["selfmorph"]["mean_score"] - ref["selfmorph"]["mean_score"],
                    "domain_macro_reward":
                        entry["selfmorph"]["domain_macro_reward"] - ref["selfmorph"]["domain_macro_reward"],
                    "mean_cost": entry["selfmorph"]["mean_cost"] - ref["selfmorph"]["mean_cost"],
                }
        out["models"][spec.key] = model_entry

    dump_json(Path(args.out), out)

if __name__ == "__main__":
    main()
