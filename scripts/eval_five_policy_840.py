#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from selfmorph.action_space import build_action_space  # noqa: E402
from selfmorph.final_eval import (compute_benchmark_prior, compute_domain_prior,  # noqa: E402
                                  compute_fixed_anchor, policy_actions)
from selfmorph.selector import build_selector  # noqa: E402
from selfmorph.selection import benchmark_unit_for  # noqa: E402

MODELS = {
    "Qwen3.5-27B": dict(cfg="config/runtime.json",
                        pt="artifacts/training_data/qwen35_27b/train_dev.pt",
                        ckpt="artifacts/checkpoints/qwen35_27b/best_checkpoint.pt"),
    "Qwen3.5-4B": dict(cfg="config/runtime_qwen35_4b.json",
                       pt="artifacts/training_data/qwen35_4b/train_dev.pt",
                       ckpt="artifacts/checkpoints/qwen35_4b/best_checkpoint.pt"),
    "Gemma4-12B": dict(cfg="config/runtime_gemma4_12b.json",
                       pt="artifacts/training_data/gemma4_12b/train_dev.pt",
                       ckpt="artifacts/checkpoints/gemma4_12b/best_checkpoint.pt"),
}
POLICIES = ["base_lm", "fixed_rep", "fixed_br", "benchmark_prior", "selfmorph"]
BENCH = ["arc_challenge", "bbh", "gsm8k", "math500", "openbookqa"]

def fmt(x) -> str:
    return f"{x:.4f}" if x is not None else "NA"

def gather(payload, idx, actions, pol) -> dict:
    per = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for i, a in zip(idx, actions[pol]):
        if a is None:
            continue
        u = benchmark_unit_for(payload["datasets"][i])
        per[u][0] += float(payload["scores"][i][a])
        per[u][1] += float(payload["rewards"][i][a])
        per[u][2] += 1
    return {u: {"score": acc[0] / acc[2], "reward": acc[1] / acc[2], "n": acc[2]}
            for u, acc in per.items()}

def pooled_of(payload, idx, actions, pol) -> dict:
    ssum = rsum = 0.0
    nn = 0
    for i, a in zip(idx, actions[pol]):
        if a is None:
            continue
        ssum += float(payload["scores"][i][a])
        rsum += float(payload["rewards"][i][a])
        nn += 1
    return {"score": ssum / nn, "reward": rsum / nn, "n": nn}

def show(title, payload, idx, actions, result) -> None:
    print(f"--- {title} (n={len(idx)}) ---")
    print(f"{'benchmark':<16}" + "".join(f"{p.replace('_', '-'):>26}" for p in POLICIES)
          + "  [score / reward]")
    g = {p: gather(payload, idx, actions, p) for p in POLICIES}
    for b in BENCH:
        row = f"{b:<16}"
        for p in POLICIES:
            v = g[p].get(b)
            row += f"  {fmt(v['score'])} / {fmt(v['reward'])}" if v else "  NA / NA"
        print(row)
    pooled = {p: pooled_of(payload, idx, actions, p) for p in POLICIES}
    print(f"{'pooled':<16}" + "".join(f"  {fmt(pooled[p]['score'])} / {fmt(pooled[p]['reward'])}"
                                      for p in POLICIES))
    result["per_unit"] = g
    result["pooled"] = pooled

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/results/eval840/five_policy_840.json")
    args = ap.parse_args()

    out: dict = {
        "protocol": "short-budget evaluation on the 840 held-out test samples",
        "source": "train_dev.pt plus the released selector checkpoint for each backbone",
        "models": {},
    }
    for model, spec in MODELS.items():
        print("=" * 100)
        print("MODEL", model)
        cfg = json.load(open(ROOT / spec["cfg"]))
        payload = torch.load(ROOT / spec["pt"], map_location="cpu", weights_only=False)
        anchors = build_action_space(cfg)
        hidden = cfg["model"]["hidden_size"]
        ntl = payload["trajectory"].shape[1]
        sel = build_selector(cfg["selector"], anchors, hidden, ntl).to("cpu")
        st = torch.load(ROOT / spec["ckpt"], map_location="cpu", weights_only=False)
        sel.load_state_dict(st["model_state"])
        dp = compute_domain_prior(payload, anchors)
        bp = compute_benchmark_prior(payload, anchors)
        fr = compute_fixed_anchor(payload, "REPEAT", anchors)
        fb = compute_fixed_anchor(payload, "BRANCH", anchors)
        actions = policy_actions(payload, "eval", sel, anchors, dp, fr, fb,
                                 device="cpu", benchmark_prior=bp)
        print("BP actions:", {k: v for k, v in bp.actions.items()})
        mask = [s == "eval" for s in payload["splits"]]
        idx = [i for i, m in enumerate(mask) if m]
        entry = {"bp_actions": {k: int(v) for k, v in bp.actions.items()}}
        show("eval", payload, idx, actions, entry.setdefault("eval", {}))
        out["models"][model] = entry

    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {dest}")

if __name__ == "__main__":
    main()
