#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from selfmorph.action_space import build_action_space  # noqa: E402
from selfmorph.selector import build_selector  # noqa: E402
from selfmorph.final_eval import (  # noqa: E402
    compute_domain_prior, compute_benchmark_prior, compute_fixed_anchor,
    policy_actions, evaluate_policies,
)
from selfmorph.selection import benchmark_unit_for, evaluate_selector_on_split  # noqa: E402

RELEASED_CHECKPOINT = Path("artifacts/checkpoints/qwen35_27b/best_checkpoint.pt")
PRODUCTION_SELECTOR_CONFIG = {
    "encoder": "projection", "proj_dim": 192, "trunk": "attention", "trunk_hidden": 384, "trunk_depth": 0,
    "head_variant": "hybrid", "head_hidden_dim": 48, "layer_set": "dense", "dropout": 0.05,
    "lr": 8e-4, "weight_decay": 0.0, "batch_size": 256, "effective_batch_size": 256,
    "micro_batch_size": 128, "max_epochs": 50, "patience": 8, "checkpoint_selection": "best",
    "target_mode": "softmax", "temperature_mode": "fixed", "temperature_fixed": 0.15,
    "temperature_cosine_start": 0.15, "temperature_cosine_end": 0.15, "schedule_epochs": 50,
    "seed": 42, "prenorm_mode": "rms_gain", "lr_schedule": "cosine_warmup",
    "lr_warmup_frac": 0.05, "lr_min_scale": 0.02, "weight_decay_exempt_norm_bias": True,
    "attention_d_model": 64, "attention_n_heads": 2, "attention_n_blocks": 1,
}

def load_production(hidden, num_traj_layers, anchors):
    sel = build_selector(PRODUCTION_SELECTOR_CONFIG, anchors, hidden, num_traj_layers)
    st = torch.load(RELEASED_CHECKPOINT, map_location="cpu", weights_only=False)
    sel.load_state_dict(st["model_state"])
    return sel

def load_trial(grid_dir: Path, prefix: str, anchors, hidden, num_traj_layers):
    d = grid_dir / prefix
    meta = json.load(open(d / "trial_meta.json"))
    sc = meta["selector_config"]
    sel = build_selector(sc, anchors, hidden, num_traj_layers)
    st = torch.load(d / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    sel.load_state_dict(st["model_state"])
    return sel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/runtime.json")
    parser.add_argument("--train-dev-pt", default="artifacts/training_data/qwen35_27b/train_dev.pt")
    parser.add_argument("--out", default="artifacts/results/architecture_ablation.json")
    args = parser.parse_args()

    config = json.load(open(args.config))
    anchors = build_action_space(config)
    hidden = config["model"]["hidden_size"]
    payload = torch.load(args.train_dev_pt, map_location="cpu", weights_only=False)
    num_traj_layers = payload["trajectory"].shape[1]

    domain_prior = compute_domain_prior(payload, anchors)
    benchmark_prior = compute_benchmark_prior(payload, anchors)
    fixed_rep_id = compute_fixed_anchor(payload, "REPEAT", anchors)
    fixed_br_id = compute_fixed_anchor(payload, "BRANCH", anchors)

    mask = [s == "eval" for s in payload["splits"]]
    idx = [i for i, m in enumerate(mask) if m]
    units = sorted({benchmark_unit_for(payload["datasets"][i]) for i in idx})
    policies = ["base_lm", "fixed_rep", "fixed_br", "domain_prior", "benchmark_prior", "selfmorph"]

    groups = {}
    groups["production"] = load_production(hidden, num_traj_layers, anchors)

    ablation_dir = Path("artifacts/checkpoints/qwen35_27b/ablations")
    groups["no_prenorm"] = load_trial(ablation_dir, "no_prenorm", anchors, hidden, num_traj_layers)
    groups["no_attention"] = load_trial(ablation_dir, "no_attention", anchors, hidden, num_traj_layers)
    groups["d96_dz48"] = load_trial(ablation_dir, "d96_dz48", anchors, hidden, num_traj_layers)
    groups["d384_dz128"] = load_trial(ablation_dir, "d384_dz128", anchors, hidden, num_traj_layers)
    groups["d512_dz192"] = load_trial(ablation_dir, "d512_dz192", anchors, hidden, num_traj_layers)
    groups["d1024_dz256"] = load_trial(ablation_dir, "d1024_dz256", anchors, hidden, num_traj_layers)

    for temperature in [0.01, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        groups[f"temperature_{temperature:g}"] = load_trial(
            ablation_dir, f"temperature_{temperature:g}", anchors, hidden, num_traj_layers,
        )

    results = {}
    for label, sel in groups.items():
        actions = policy_actions(payload, "eval", sel, anchors, domain_prior, fixed_rep_id, fixed_br_id,
                                 device="cpu", benchmark_prior=benchmark_prior)
        metrics = evaluate_policies(payload, "eval", actions)

        official_dp = evaluate_selector_on_split(sel, anchors, payload, "eval", domain_prior.actions, device="cpu")
        official_bp = evaluate_selector_on_split(
            sel, anchors, payload, "eval", benchmark_prior.actions, device="cpu",
            prior_key_fn=lambda p, i: benchmark_unit_for(p["datasets"][i]),
        )

        by_score, by_reward = {u: {} for u in units}, {u: {} for u in units}
        for pol in policies:
            per_u_s, per_u_r = {u: [] for u in units}, {u: [] for u in units}
            for pos, i in enumerate(idx):
                a = actions[pol][pos]
                if a is None:
                    continue
                u = benchmark_unit_for(payload["datasets"][i])
                per_u_s[u].append(float(payload["scores"][i][a]))
                per_u_r[u].append(float(payload["rewards"][i][a]))
            for u in units:
                by_score[u][pol] = sum(per_u_s[u]) / len(per_u_s[u]) if per_u_s[u] else None
                by_reward[u][pol] = sum(per_u_r[u]) / len(per_u_r[u]) if per_u_r[u] else None

        results[label] = {
            "aggregate": {pol: {k: v for k, v in metrics[pol].items() if k in ("domain_macro_reward", "mean_score", "mean_reward", "mean_cost")}
                          for pol in policies},
            "official_vs_domain_prior": official_dp,
            "official_vs_benchmark_prior": official_bp,
            "by_benchmark_score": by_score,
            "by_benchmark_reward": by_reward,
        }
        m = metrics["selfmorph"]
        print(f"[{label}] DMR={m['domain_macro_reward']:.4f} score={m['mean_score']:.4f} "
              f"vsDP_double_wins={official_dp['benchmark_double_wins']}/{official_dp['n_benchmark_units']} "
              f"vsBP_double_wins={official_bp['benchmark_double_wins']}/{official_bp['n_benchmark_units']} "
              f"vsBP_worst_domain={official_bp['worst_domain_reward_margin']:+.4f} "
              f"vsBP_score_margin={official_bp['score_margin_over_prior']:+.4f} "
              f"vsBP_reward_margin={official_bp['reward_margin_over_prior']:+.4f}")

    bp_by_unit_score, bp_by_unit_reward = {}, {}
    for u in units:
        vals_s, vals_r = [], []
        for i in idx:
            if benchmark_unit_for(payload["datasets"][i]) != u:
                continue
            a = benchmark_prior.action_for(u)
            vals_s.append(float(payload["scores"][i][a]))
            vals_r.append(float(payload["rewards"][i][a]))
        bp_by_unit_score[u] = sum(vals_s) / len(vals_s)
        bp_by_unit_reward[u] = sum(vals_r) / len(vals_r)

    out = {
        "units": units,
        "benchmark_prior_actions": benchmark_prior.actions,
        "benchmark_prior_by_unit_score": bp_by_unit_score,
        "benchmark_prior_by_unit_reward": bp_by_unit_reward,
        "domain_prior_actions": domain_prior.actions,
        "fixed_rep_id": fixed_rep_id, "fixed_br_id": fixed_br_id,
        "candidates": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.out}")

if __name__ == "__main__":
    main()
