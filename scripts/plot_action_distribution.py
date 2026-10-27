#!/usr/bin/env python3
"""Plot the oracle-optimal action distribution.

Partitions the training split (7,293 samples on Qwen3.5-27B) by `argmax(U(x, m))`
into:
  - Left panel: `Base-enough` (oracle is `BASE` and `score(x, BASE) == 1`),
    `Base-hopeless` (oracle is `BASE` and `score(x, BASE) == 0`), `Repeat`, `Branch`;
  - Center panel: intensity `r in {1, 2, 3}` breakdown within `Repeat`;
  - Right panel: intensity `K in {2, 4, 8}` breakdown within `Branch`.

Usage:
    .venv/bin/python scripts/action_distribution/plot_action_distribution_split_base.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from selfmorph.action_space import build_action_space  # noqa: E402

MODELS = {
    "qwen35_27b": {
        "display": "Qwen3.5-27B",
        "config": "config/runtime.json",
        "train_dev_pt": "artifacts/training_data/qwen35_27b/train_dev.pt",
    },
    "qwen35_4b": {
        "display": "Qwen3.5-4B",
        "config": "config/runtime_qwen35_4b.json",
        "train_dev_pt": "artifacts/training_data/qwen35_4b/train_dev.pt",
    },
    "gemma4_12b": {
        "display": "Gemma-4-12B",
        "config": "config/runtime_gemma4_12b.json",
        "train_dev_pt": "artifacts/training_data/gemma4_12b/train_dev.pt",
    },
}

COLOR_BASE_ENOUGH = "#8C9BAF"
COLOR_BASE_HOPELESS = "#5A6B85"
REP_R_COLORS = {1: "#F2B8A6", 2: "#D77A61", 3: "#A8492F"}
BR_K_COLORS = {2: "#63E6BE", 4: "#12B886", 8: "#087F5B"}

BASE_ENOUGH_LABEL = "Base-enough"
BASE_HOPELESS_LABEL = "Base-hopeless"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.linewidth": 0.6,
})

def _pie(ax, sizes, labels, colors, title=None, radius=1.0):
    total = sum(sizes)
    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=90, counterclock=False, radius=radius,
        wedgeprops=dict(edgecolor="white", linewidth=1.6),
    )
    for w, s in zip(wedges, sizes):
        pct = 100.0 * s / total
        ang = np.radians((w.theta1 + w.theta2) / 2.0)
        d = 0.77 - 0.25 * min(pct / 25.0, 1.0)
        x, y = np.cos(ang) * d * radius, np.sin(ang) * d * radius
        ax.text(
            x, y, f"{pct:.1f}%", ha="center", va="center", color="white",
            fontsize=15.0, fontweight="bold",
        )
    ax.legend(
        wedges, list(labels),
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
        fontsize=15.0, handlelength=1.3, handletextpad=0.6,
    )
    if title:
        ax.set_title(title, fontsize=18, fontweight="bold", pad=22)
    ax.set_aspect("equal")

def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}_300dpi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / {name}_300dpi.png")

def _is_base_enough(base_scores: torch.Tensor) -> torch.Tensor:
    vals = base_scores.float()
    if torch.all((vals == 0.0) | (vals == 1.0)):
        return vals == 1.0
    return vals >= 0.5

def build_combined_split_base(
    family_sizes, family_labels, family_colors,
    rep_sizes, rep_labels, rep_colors,
    br_sizes, br_labels, br_colors,
    out_dir: Path, name: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21.4, 7.8))
    _pie(axes[0], family_sizes, family_labels, family_colors, "oracle action type", radius=1.35)
    if sum(rep_sizes) > 0:
        _pie(axes[1], rep_sizes, rep_labels, rep_colors, "repeat count $r$", radius=1.35)
    else:
        axes[1].text(0.5, 0.5, "no REPEAT-preferred samples", ha="center", va="center")
        axes[1].set_axis_off()
    if sum(br_sizes) > 0:
        _pie(axes[2], br_sizes, br_labels, br_colors, "branch count $K$", radius=1.35)
    else:
        axes[2].text(0.5, 0.5, "no BR-preferred samples", ha="center", va="center")
        axes[2].set_axis_off()
    fig.subplots_adjust(wspace=1.05, top=0.82, bottom=0.05)
    save(fig, out_dir, name)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35_27b", choices=sorted(MODELS))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "figures" / "action_distribution"))
    ap.add_argument("--name-suffix", default="splitbase")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    model_key = args.model
    cfg = MODELS[model_key]
    config = json.loads((REPO_ROOT / cfg["config"]).read_text(encoding="utf-8"))
    anchors = build_action_space(config)
    rep_ids = {a.id for a in anchors if a.type == "REPEAT"}
    br_ids = {a.id for a in anchors if a.type == "BRANCH"}
    r_of = {a.id: a.r for a in anchors if a.type == "REPEAT"}
    k_of = {a.id: a.K for a in anchors if a.type == "BRANCH"}

    payload = torch.load(REPO_ROOT / cfg["train_dev_pt"], map_location="cpu", weights_only=True)
    U, S, splits = payload["utilities"], payload["scores"], payload["splits"]
    idx = [i for i, s in enumerate(splits) if s == "train"]
    n_train = len(idx)
    idx_t = torch.tensor(idx)

    best = torch.argmax(U[idx_t], dim=1).tolist()
    base_enough_mask = _is_base_enough(S[idx_t, 0])

    counts = {"BASE_ENOUGH": 0, "BASE_HOPELESS": 0, "REPEAT": 0, "BRANCH": 0}
    r_counts = {1: 0, 2: 0, 3: 0}
    k_counts = {2: 0, 4: 0, 8: 0}
    for j, b in enumerate(best):
        if b == 0:
            counts["BASE_ENOUGH" if bool(base_enough_mask[j]) else "BASE_HOPELESS"] += 1
        elif b in rep_ids:
            counts["REPEAT"] += 1
            r_counts[r_of[b]] += 1
        elif b in br_ids:
            counts["BRANCH"] += 1
            k_counts[k_of[b]] += 1

    fam_order = ["BASE_ENOUGH", "BASE_HOPELESS", "REPEAT", "BRANCH"]
    fam_labels = {
        "BASE_ENOUGH": BASE_ENOUGH_LABEL, "BASE_HOPELESS": BASE_HOPELESS_LABEL,
        "REPEAT": "Repeat", "BRANCH": "Branch",
    }
    fam_colors = {
        "BASE_ENOUGH": COLOR_BASE_ENOUGH, "BASE_HOPELESS": COLOR_BASE_HOPELESS,
        "REPEAT": "#D77A61", "BRANCH": "#12B886",
    }

    r_order = [1, 2, 3]
    r_sizes = [r_counts[r] for r in r_order]
    k_order = [2, 4, 8]
    k_sizes = [k_counts[k] for k in k_order]

    name = f"{model_key}_combined_{args.name_suffix}"
    build_combined_split_base(
        [counts[f] for f in fam_order],
        [fam_labels[f] for f in fam_order],
        [fam_colors[f] for f in fam_order],
        r_sizes, [f"r={r}" for r in r_order], [REP_R_COLORS[r] for r in r_order],
        k_sizes, [f"K={k}" for k in k_order], [BR_K_COLORS[k] for k in k_order],
        out_dir, name,
    )

    out = {
        "purpose": "oracle action distribution on the train split",
        "model": model_key, "n_train": n_train,
        "counts": counts,
        "pct": {k: round(100.0 * v / n_train, 2) for k, v in counts.items()},
        "base_total": counts["BASE_ENOUGH"] + counts["BASE_HOPELESS"],
        "rep_r_counts": r_counts,
        "branch_k_counts": k_counts,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
