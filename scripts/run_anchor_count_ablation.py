#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from ablation_common import (  # noqa: E402
    ANCHOR_TIERS, EXP1_CKPT_DIR, REPO_ROOT, build_tier_config, load_config, load_payload,
    mean_train_reward, model_specs, run_trials, tier_selection_report,
)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen35_27b,qwen35_4b,gemma4_12b")
    ap.add_argument("--tiers", default=",".join(ANCHOR_TIERS))
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--output-dir", default=str(EXP1_CKPT_DIR))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    for spec in model_specs([m.strip() for m in args.models.split(",") if m.strip()]):
        base_config = load_config(spec)
        raw_payload = load_payload(spec)
        vec = mean_train_reward(raw_payload)
        sel_report = tier_selection_report(base_config, vec)
        model_dir = Path(args.output_dir) / spec.key

        trials: list[dict] = []
        for tier in tiers:
            sub_cfg, sub_anchors, col_ids = build_tier_config(base_config, tier, vec)
            trials.append({
                "trial_id": tier,
                "model": spec.key,
                "tier": tier,
                "num_anchors": len(sub_anchors),
                "selector_config": sub_cfg["selector"],
                "action_space": sub_cfg["action_space"],
                "payload_spec": {"kind": "anchor_subset", "col_ids": col_ids},
            })

        plan = {
            "schema_version": "selfmorph-anchor-count-ablation-plan-1",
            "purpose": "nested anchor-count ablation 3-action pool nested in the 9-action pool nested in the 25-action pool on the persisted 25-anchor oracle; "
                       "only the selector is retrained, each base keeps its current production selector config",
            "model": spec.key, "label": spec.label, "config": spec.config,
            "train_dev_pt": spec.train_dev_pt, "selector_checkpoint": spec.checkpoint,
            "tiers": tiers, "num_trials": len(trials),
            "tier_selection": sel_report,
            "trials": [{k: v for k, v in t.items() if k not in ("selector_config", "action_space")} for t in trials],
        }
        print(f"[{spec.key}] {len(trials)} trials -> {model_dir}")
        print(f"  actions3 repeat={sel_report['actions3_repeat_span']} br={sel_report['actions3_branch_span']} "
              f"(rep mean train reward={sel_report['rep_mean_train_reward']}, "
              f"br mean train reward={sel_report['br_mean_train_reward']})")
        if args.dry_run:
            for t in trials:
                print("   ", t["trial_id"], t["num_anchors"])
            continue

        run_trials(trials, base_config, REPO_ROOT / spec.train_dev_pt, model_dir, gpus, plan)

if __name__ == "__main__":
    main()
