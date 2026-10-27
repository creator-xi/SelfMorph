#!/usr/bin/env python3
"""Evaluate the 13-action REPEAT pool, the 13-action BRANCH pool, and the full 25-action pool."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1]))

from selfmorph.action_space import build_action_space  # noqa: E402

from ablation_common import (  # noqa: E402
    anchor_descriptions, dump_json, selector_actions, eval_actions_on_split, fixed_anchor_ids,
    load_config, load_payload, load_trial_selector, model_specs, oracle_upper_bound, load_production_selector,
    slice_payload_anchors,
)
from action_type_common import (  # noqa: E402
    EXP_CKPT_DIR, POOLS, POOL_TYPES, assert_payload_columns, build_pool_config, install,
)

def _state_dict_maxdiff(sel_a, sel_b) -> float:
    sd_a, sd_b = sel_a.state_dict(), sel_b.state_dict()
    if list(sd_a) != list(sd_b):
        return float("inf")
    worst = 0.0
    for k in sd_a:
        worst = max(worst, float((sd_a[k].float() - sd_b[k].float()).abs().max()))
    return worst

def _head_audit(sel) -> dict:
    """Head width and parameter names for the active morphology families."""
    head = sel.head
    head_keys = list(head.state_dict().keys())
    names = {0: "BASE", 1: "REPEAT", 2: "BRANCH"}
    audit = {
        "active_types_global_ids": list(head.active_types),
        "active_types_names": [names[t] for t in head.active_types],
        "type_col_global_to_local": {str(k): v for k, v in sorted(head.type_col.items())},
        "head_variant": head.variant,
        "head_type_out_features": int(head.head_type.out_features),
        "has_rep_modules": bool(head.has_type_rep),
        "has_br_modules": bool(head.has_type_br),
        "head_state_dict_keys": head_keys,
    }
    for family, flag, needle in (("REPEAT", head.has_type_rep, "_REP"), ("BR", head.has_type_br, "_BR")):
        audit[f"{family}_keys_present"] = sorted(k for k in head_keys if needle in k)
        if not flag and audit[f"{family}_keys_present"]:
            raise RuntimeError(
                f"head built for a pool without {family} still exposes "
                f"{audit[f'{family}_keys_present']}"
            )
    return audit

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen35_27b,qwen35_4b,gemma4_12b")
    ap.add_argument("--pools", default=",".join(POOLS))
    ap.add_argument("--ckpt-dir", default=str(EXP_CKPT_DIR))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="artifacts/results/action_type_ablation_eval.json")
    args = ap.parse_args()

    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    ckpt_root = Path(args.ckpt_dir)
    install_log = install()

    out: dict = {
        "experiment": "action_type_ablation",
        "split": args.split,
        "pools": pools,
        "pool_definitions": {p: {"types": list(POOL_TYPES[p])} for p in pools},
        "installation": install_log,
        "notes": (
            "repeat13 = BASE+REPEAT only, branch13 = BASE+BRANCH only, full25 = the production 25-anchor pool. "
            "Only the action pool changes (payload anchor columns + the type-conditioned head/loss in "
            "single-family action pools; every backbone keeps its released selector config, and the default data "
            "splits (train 7293 / dev_tune 1216 / eval 840). Each backbone uses its released selector seed. "
            "Base LM / Fixed-REPEAT / Fixed-BR references "
            "are computed on the FULL 25-column payload, because the fixed anchors of a family need not be "
            "inside the reduced pool. The per-pool oracle upper bound IS computed on the cut payload."
        ),
        "models": {},
    }

    for spec in model_specs([m.strip() for m in args.models.split(",") if m.strip()]):
        base_config = load_config(spec)
        raw_payload = load_payload(spec)
        hidden = int(base_config["model"]["hidden_size"])
        nlayers = int(raw_payload["trajectory"].shape[1])
        full_anchors = build_action_space(base_config)
        col_check = assert_payload_columns(raw_payload, base_config)
        n_eval = sum(1 for s in raw_payload["splits"] if s == args.split)

        fixed_rep_id, fixed_br_id = fixed_anchor_ids(raw_payload, full_anchors)
        base_id = full_anchors.base_id
        descs_full = {d["id"]: d for d in anchor_descriptions(full_anchors)}
        references = {
            "base_lm_noop": eval_actions_on_split(raw_payload, args.split, [base_id] * n_eval),
            "fixed_rep": {
                "anchor_id": fixed_rep_id, "anchor": descs_full[fixed_rep_id],
                "metrics": eval_actions_on_split(raw_payload, args.split, [fixed_rep_id] * n_eval),
            },
            "fixed_br": {
                "anchor_id": fixed_br_id, "anchor": descs_full[fixed_br_id],
                "metrics": eval_actions_on_split(raw_payload, args.split, [fixed_br_id] * n_eval),
            },
            "full_pool_oracle_upper_bound": oracle_upper_bound(raw_payload, args.split),

            "fixed_anchor_in_reduced_pool": {},
        }
        for pool in pools:
            _, sub_anchors, _ = build_pool_config(base_config, pool)
            keys = {a.type for a in sub_anchors}
            references["fixed_anchor_in_reduced_pool"][pool] = {
                "fixed_rep_in_pool": descs_full[fixed_rep_id]["type"] in keys,
                "fixed_br_in_pool": descs_full[fixed_br_id]["type"] in keys,
            }

        prod_sel = load_production_selector(spec, full_anchors, hidden, nlayers)
        prod_actions = selector_actions(raw_payload, full_anchors, prod_sel, args.split, args.device)
        prod_metrics = eval_actions_on_split(raw_payload, args.split, prod_actions)

        model_entry: dict = {
            "label": spec.label,
            "config": spec.config,
            "train_dev_pt": spec.train_dev_pt,
            "selector_checkpoint": spec.checkpoint,
            "hidden_size": hidden,
            "num_trajectory_layers": nlayers,
            "n_eval_samples": n_eval,
            "payload_column_check_max_abs_cost_diff": col_check,
            "selector_config": base_config["selector"],
            "references_full_pool": references,
            "released_selector": {
                "n_params": sum(p.numel() for p in prod_sel.parameters()),
                "metrics": prod_metrics,
            },
            "pools": {},
        }

        for pool in pools:
            sub_cfg, sub_anchors, col_ids = build_pool_config(base_config, pool)
            payload = slice_payload_anchors(raw_payload, col_ids)
            model_dir = ckpt_root / spec.key
            sel = load_trial_selector(model_dir, pool, sub_anchors, hidden, nlayers)

            actions = selector_actions(payload, sub_anchors, sel, args.split, args.device)
            metrics = eval_actions_on_split(payload, args.split, actions)
            oracle = oracle_upper_bound(payload, args.split)
            pool_noop = eval_actions_on_split(payload, args.split, [sub_anchors.base_id] * n_eval)
            counts = Counter(actions)
            descs = {d["id"]: d for d in anchor_descriptions(sub_anchors)}

            entry: dict = {
                "pool": pool,
                "types": list(POOL_TYPES[pool]),
                "num_anchors": len(sub_anchors),
                "payload_columns": col_ids,
                "n_params": sum(p.numel() for p in sel.parameters()),
                "head": _head_audit(sel),
                "anchors": anchor_descriptions(sub_anchors),
                "selfmorph": metrics,
                "oracle_upper_bound_argmax_utility_in_pool": oracle,
                "oracle_minus_selector_mean_score": oracle["mean_score"] - metrics["mean_score"],
                "base_lm_noop_in_pool": pool_noop,
                "selected_action_counts": {
                    str(aid): {"count": c, "anchor": descs[aid]} for aid, c in sorted(counts.items())
                },
            }

            if pool == "full25":
                diff = _state_dict_maxdiff(sel, prod_sel)
                entry["self_check_vs_released_selector"] = {
                    "state_dict_max_abs_diff": diff,
                    "actions_identical": actions == prod_actions,
                    "metrics_identical": metrics == prod_metrics,
                }

            model_entry["pools"][pool] = entry
            print(f"[{spec.key}][{pool}] n={len(sub_anchors)} "
                  f"score={metrics['mean_score']:.4f} reward={metrics['mean_reward']:.4f} "
                  f"cost={metrics['mean_cost']:.4f} DMR={metrics['domain_macro_reward']:.4f} "
                  f"| oracle(score={oracle['mean_score']:.4f}) "
                  f"| base={pool_noop['mean_score']:.4f} "
                  f"| head_type_out={entry['head']['head_type_out_features']} "
                  f"| params={entry['n_params']}")

        out["models"][spec.key] = model_entry

    dump_json(Path(args.out), out)

if __name__ == "__main__":
    main()
