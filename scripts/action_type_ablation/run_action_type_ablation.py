#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1]))

from action_type_common import (  # noqa: E402
    EXP_CKPT_DIR, POOLS, POOL_SIZES, build_plan, build_pool_config, build_trials, install,
    run_trials,
)
from ablation_common import REPO_ROOT, load_config, model_specs  # noqa: E402

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen35_27b,qwen35_4b,gemma4_12b")
    ap.add_argument("--pools", default=",".join(POOLS))
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--output-dir", default=str(EXP_CKPT_DIR))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    install()

    for spec in model_specs([m.strip() for m in args.models.split(",") if m.strip()]):
        base_config = load_config(spec)
        model_dir = Path(args.output_dir) / spec.key
        trials = build_trials(spec, base_config, pools)
        plan = build_plan(spec, base_config, pools, trials)

        print(f"[{spec.key}] {len(trials)} trials -> {model_dir}")
        for p in pools:
            _, sub_anchors, col_ids = build_pool_config(base_config, p)
            print(f"  {p}: {len(sub_anchors)} anchors (expected {POOL_SIZES[p]}), "
                  f"types={sorted({a.type for a in sub_anchors})}, payload columns={col_ids}")
        if args.dry_run:
            for t in trials:
                print("   ", t["trial_id"], t["num_anchors"], t["anchor_types"])
            continue

        run_trials(trials, base_config, REPO_ROOT / spec.train_dev_pt, model_dir, gpus, plan)

if __name__ == "__main__":
    main()
