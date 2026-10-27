
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

COLOR_REP = "#E85A5A"
COLOR_BR = "#2B8A5A"
COLOR_BASE = "#4A6FA5"
COLOR_HOPELESS = "#8C8C8C"
COLOR_GRID = "#DDDDDD"
GROUP_ORDER = ["base_hopeless", "repeat_best", "base_enough", "branch_best"]
GROUP_COLOR = {"base_hopeless": COLOR_HOPELESS, "repeat_best": COLOR_REP,
               "base_enough": COLOR_BASE, "branch_best": COLOR_BR}
GROUP_LABEL = {"base_hopeless": "base-hopeless", "repeat_best": "repeat-best",
               "base_enough": "base-enough", "branch_best": "branch-best"}

NUMBER_FONTSIZE = 36
XTICK_LABEL_FONTSIZE = 34
YLABEL_FONTSIZE = 38
TITLE_FONTSIZE = 44

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 1.8, "font.size": NUMBER_FONTSIZE,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 7, "ytick.major.size": 7, "legend.frameon": False,
    "xtick.major.pad": 20, "ytick.major.pad": 14,
    "axes.titlepad": 36, "axes.labelpad": 26,
})

def _grid(ax):
    ax.grid(True, color=COLOR_GRID, linewidth=0.5, zorder=0, axis="y")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("0.35")

def panel_violin(ax, regime, split, key, ylabel, title):
    feats = regime["by_split"][split]["features_by_group"]
    groups = [g for g in GROUP_ORDER if g in feats and feats[g][key]]
    data = [feats[g][key] for g in groups]
    parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.75)
    for pc, g in zip(parts["bodies"], groups):
        pc.set_facecolor(GROUP_COLOR[g])
        pc.set_alpha(0.55)
        pc.set_edgecolor(GROUP_COLOR[g])
    parts["cmedians"].set_color("#222222")
    parts["cmedians"].set_linewidth(1.3)

    all_vals = [v for d in data for v in d]
    dy = 0.03 * (max(all_vals) - min(all_vals))
    for i, g in enumerate(groups):
        med = float(np.median(data[i]))
        ax.text(i + 1, med + dy, f"{med:.3f}", fontsize=NUMBER_FONTSIZE, color="#222222",
                ha="center", va="bottom", zorder=5)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([GROUP_LABEL[g] for g in groups], fontsize=XTICK_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=YLABEL_FONTSIZE)
    ax.set_title(title, loc="center", fontsize=TITLE_FONTSIZE, fontweight="bold")
    _grid(ax)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "artifacts" / "results" / "trajectory"))
    ap.add_argument("--fig-dir", default=str(REPO_ROOT / "figures" / "trajectory" / "trajectory"))
    ap.add_argument("--split", default="eval")
    args = ap.parse_args()

    res = Path(args.results_dir)
    regime = json.loads((res / "base_regime_large_n.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(1, 2, figsize=(38.0, 11.6))
    panel_violin(ax[0], regime, args.split, "last_relupd",
                 "relative update",
                 "Magnitude of hidden state")
    panel_violin(ax[1], regime, args.split, "min_cos",
                 "cosine similarity",
                 "Smoothness of hidden state")

    fig.tight_layout(w_pad=1.0)

    out_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "trajectory_regimes.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "trajectory_regimes_300dpi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'trajectory_regimes.pdf'}")
    print(f"wrote {out_dir / 'trajectory_regimes_300dpi.png'}")

if __name__ == "__main__":
    main()
