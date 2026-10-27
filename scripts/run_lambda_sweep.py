#!/usr/bin/env python3
"""Retrain selectors across cost coefficients and record the training-split Pareto curve."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from ablation_common import (  # noqa: E402
    LAMBDA_SWEEP_CKPT_DIR, LAMBDAS, REPO_ROOT, analytic_costs_double, build_action_space, lam_prefix,
    load_config, load_payload, model_specs, run_trials,
)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen35_27b,qwen35_4b,gemma4_12b")
    ap.add_argument("--lambdas", default=",".join(f"{l:g}" for l in LAMBDAS))
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--output-dir", default=str(LAMBDA_SWEEP_CKPT_DIR))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    for spec in model_specs([m.strip() for m in args.models.split(",") if m.strip()]):
        base_config = load_config(spec)
        _ = load_payload(spec)
        anchors = build_action_space(base_config)
        costs64 = analytic_costs_double(anchors).tolist()
        prod_lambda = float(base_config["action_space"]["lambda_cost"])
        model_dir = Path(args.output_dir) / spec.key

        trials: list[dict] = []
        for lam in lambdas:
            ac = copy.deepcopy(base_config["action_space"])
            ac["lambda_cost"] = lam
            trials.append({
                "trial_id": lam_prefix(lam),
                "model": spec.key,
                "lambda_cost": lam,
                "selector_config": base_config["selector"],
                "action_space": ac,
                "payload_spec": {
                    "kind": "lambda", "lambda_cost": lam, "base_id": 0,
                    "costs64": costs64, "stored_lambda": prod_lambda,
                },
            })

        plan = {
            "schema_version": "selfmorph-lambda-sweep-plan-1",
            "purpose": "lambda_cost sensitivity / Pareto sweep on the persisted 25-anchor oracle; "
                       "reward & utility recomputed per lambda, selector retrained per lambda",
            "model": spec.key, "label": spec.label, "config": spec.config,
            "train_dev_pt": spec.train_dev_pt, "selector_checkpoint": spec.checkpoint,
            "lambdas": lambdas, "num_trials": len(trials),
            "production_lambda": prod_lambda,
            "trials": [{k: v for k, v in t.items() if k not in ("selector_config", "action_space")} for t in trials],
        }
        print(f"[{spec.key}] {len(trials)} trials -> {model_dir}")
        if args.dry_run:
            for t in trials:
                print("   ", t["trial_id"])
            continue

        run_trials(trials, base_config, REPO_ROOT / spec.train_dev_pt, model_dir, gpus, plan)

if __name__ == "__main__":
    main()
