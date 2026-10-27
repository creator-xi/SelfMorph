#!/usr/bin/env python3
"""Plot the cost-coefficient sweep on the training split.

Reads `artifacts/results/lambda_sweep_train.json`, written by
`scripts/eval_lambda_sweep.py --split train`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

from ablation_common import REPO_ROOT  # noqa: E402

MODEL_ORDER = ["qwen35_27b", "qwen35_4b", "gemma4_12b"]
MODEL_TITLE = {"qwen35_27b": "Qwen3.5-27B", "qwen35_4b": "Qwen3.5-4B", "gemma4_12b": "Gemma4-12B"}
MODEL_COLOR = {"qwen35_27b": "#472d7b", "qwen35_4b": "#21918c", "gemma4_12b": "#5ab54a"}
MODEL_MARKER = {"qwen35_27b": "o", "qwen35_4b": "s", "gemma4_12b": "D"}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9.5, "axes.linewidth": 0.9})

def _series(model_entry: dict) -> dict:
    pts = [model_entry["points"][k] for k in sorted(model_entry["points"], key=lambda x: float(x))]
    return {
        "lambda": [p["lambda_cost"] for p in pts],
        "selfmorph_score": [p["selfmorph"]["mean_score"] for p in pts],
        "selfmorph_cost": [p["selfmorph"]["mean_cost"] for p in pts],
    }

def _style(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color("0.35")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.55)

def plot_lambda_response(
    data: dict, models: list[str], out_dir: Path, *,
    only_lambdas: list[float] | None = None, name: str = "lambda_vs_score_cost_train_final",
) -> None:
    """2x3 grid: top row = mean score vs lambda, bottom row = mean analytic cost vs lambda."""
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(3.4 * n + 0.5, 4.9), constrained_layout=True, squeeze=False)
    keep = {round(float(x), 9) for x in only_lambdas} if only_lambdas is not None else None
    for col, k in enumerate(models):
        s = _series(data["models"][k])
        if keep is not None:
            idx = [i for i, lam in enumerate(s["lambda"]) if round(float(lam), 9) in keep]
            s = {f: [v[i] for i in idx] if isinstance(v, list) else v for f, v in s.items()}
        c = MODEL_COLOR[k]
        mk = MODEL_MARKER[k]
        ax_score, ax_cost = axes[0, col], axes[1, col]

        ax_score.plot(s["lambda"], s["selfmorph_score"], marker=mk, color=c, lw=1.7, ms=5.5)
        ax_score.set_title(MODEL_TITLE.get(k, k), fontsize=10, color=c, fontweight="bold")
        ax_score.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax_score.margins(x=0.06, y=0.12)
        _style(ax_score)

        ax_cost.plot(s["lambda"], s["selfmorph_cost"], marker=mk, color=c, lw=1.7, ms=5.5)
        ax_cost.set_xlabel(r"$\lambda_{\mathrm{cost}}$")
        ax_cost.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax_cost.margins(x=0.06, y=0.12)
        _style(ax_cost)

    axes[0, 0].set_ylabel("mean score")
    axes[1, 0].set_ylabel("mean cost")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved {out_dir / name}.{{pdf,png}}")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(REPO_ROOT / "artifacts" / "results" / "lambda_sweep_train.json"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "figures" / "lambda"))
    args = ap.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    models = [m for m in MODEL_ORDER if m in data["models"]]
    plot_lambda_response(data, models, Path(args.out_dir), name="lambda_vs_score_cost")

if __name__ == "__main__":
    main()
