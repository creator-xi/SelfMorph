#!/usr/bin/env python3
"""Train selectors for different input representations.

The 10-layer trajectory reuses the released selector. The single-state and
prompt-embedding arms broadcast one vector to the same 10 positions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1]))

from input_rep_common import (  # noqa: E402
    ARMS, ARM_SLOT_SETS, CKPT_DIR, build_plan, build_trials, run_trials, single_layer,
)
from ablation_common import REPO_ROOT, load_config, model_specs  # noqa: E402

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen35_27b,qwen35_4b,gemma4_12b")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="arms to include in the plan; `traj10` is never retrained (its checkpoints are reused)")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--output-dir", default=str(CKPT_DIR))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}, expected a subset of {list(ARMS)}")
    train_arms = [a for a in arms if a != "traj10"]

    for spec in model_specs([m.strip() for m in args.models.split(",") if m.strip()]):
        base_config = load_config(spec)
        all_trials, meta = build_trials(spec, base_config, arms)
        trials = [t for t in all_trials if t["arm"] in train_arms]
        plan = build_plan(spec, base_config, arms, all_trials, meta)
        model_dir = Path(args.output_dir) / spec.key

        print(f"[{spec.key}] {len(trials)} trials to train ({'+'.join(train_arms) or 'none'}) -> {model_dir}")
        print(f"  single arm: slot {meta['single_slot_index']} = physical layer {meta['single_layer']}; "
              f"embed arm: prompt embedding as 'layer -1'")
        for arm in ARM_SLOT_SETS:
            if arm in arms:
                print(f"  {arm} arm: selected slots {list(ARM_SLOT_SETS[arm])} "
                      f"(1-based {[i + 1 for i in ARM_SLOT_SETS[arm]]}) = layers "
                      f"{[meta['dense_layer_ids'][i] for i in ARM_SLOT_SETS[arm]]}; "
                      f"replicated to 10 slots as {meta['arm_slot_maps'][arm]}")
        for arm, path in meta["gain_paths"].items():
            print(f"  gain[{arm}] = {path}")
        if "traj10" in arms:
            print(f"  traj10 reuses {plan['traj10_reused_checkpoints']}/k10__s<seed> (not retrained)")
        if args.dry_run:
            for t in trials:
                print("   ", t["trial_id"], t["broadcast_layer"])
            continue
        if not trials:
            continue
        run_trials(trials, base_config, REPO_ROOT / spec.train_dev_pt, model_dir, gpus, plan)

if __name__ == "__main__":
    main()
