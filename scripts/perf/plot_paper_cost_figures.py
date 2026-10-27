#!/usr/bin/env python3
"""Plot inference time and memory.
comparing Base / Repeat / Branch / SelfMorph across the three evaluated bases.

Read-only w.r.t. every existing artifact: this script only reads the already
frozen `paper_cost_eval840*.json` files (produced by bench_paper_cost.py +
aggregate_paper_cost.py) and writes two new figures under
`artifacts/results/inference_speedup/figures/` (a new directory; nothing
existing is modified or overwritten).

time_cost.{pdf,png}: decode throughput (tokens/s), one panel per
base model, 4 bars per panel (Base/Repeat/Branch/SelfMorph), with the % change
relative to Base annotated under each non-Base bar.

memory_cost.{pdf,png}: total GPU memory during inference, one
column per base model, two rows:
  - top row: absolute average resident memory (GB), stacked as
    "Model weights" (constant within a column) + "Runtime overhead"
    (activations + KV cache; this is NOT KV-cache-only -- it is the full
    avg_mem_gb - resident_after_load_gb delta), so the total bar height is
    exactly avg_mem_gb and weights are shown explicitly as requested.
  - bottom row: the same runtime-overhead delta zoomed in (in MB), because
    weights dominate total memory by >99% and the top row alone would make
    the four bars look identical -- the bottom row is what actually
    distinguishes the four methods.

Usage:
    .venv/bin/python scripts/perf/plot_paper_cost_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "artifacts" / "results" / "inference_speedup"

OUT_DIR = ROOT / "figures" / "paper_cost"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECS = [
    ("Qwen3.5-27B", "recost_eval840.json", "26.9B params"),
    ("Qwen3.5-4B", "recost_qwen35_4b.json", "4.2B params"),
    ("Gemma-4-12B", "recost_gemma4_12b.json", "12.0B params"),
]
POLICIES = ["base_lm", "fixed_rep", "fixed_br", "selfmorph"]
LABELS = ["Base", "Repeat", "Branch", "SelfMorph"]
COLORS = {
    "base_lm": "#8C8C8C",
    "fixed_rep": "#4C72B0",
    "fixed_br": "#DD8452",
    "selfmorph": "#C44E52",
}
WEIGHT_COLOR = "#D9D9D9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.edgecolor": "#333333",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def _load():
    out = {}
    for name, fname, params in SPECS:
        d = json.loads((DATA_DIR / fname).read_text(encoding="utf-8"))
        pp = d["per_policy"]
        fp = d["model_footprint"]
        out[name] = {
            "params": params,
            "resident_after_load_gb": fp["resident_after_load_gb"],
            "weights_gb": fp["weights_gb"],

            "tps": {p: pp[p]["tps_decode"]["mean"] for p in POLICIES},

            "tps_total": {p: pp[p]["tps_total"]["mean"] for p in POLICIES},
            "gen_tokens": {p: pp[p]["gen_tokens"]["mean"] for p in POLICIES},
            "prefill_s": {p: pp[p]["prefill_s"]["mean"] for p in POLICIES},
            "avg_mem_gb": {p: pp[p]["avg_mem_gb"]["mean"] for p in POLICIES},
            "peak_mem_gb": {p: pp[p]["peak_mem_gb"]["mean"] for p in POLICIES},
            "kv_mb": {p: pp[p]["kv_mb"]["mean"] for p in POLICIES},
        }
    return out

def _style_axes(ax, y_grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_grid:
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

def _legend_handles():
    return [mpatches.Patch(facecolor=COLORS[p], edgecolor="#333333", linewidth=0.4, label=lab)
            for p, lab in zip(POLICIES, LABELS)]

def make_time_figure(data: dict) -> None:
    """Two rows, both in tokens/s.

    Top row is the decode-only rate (length-normalised, but it EXCLUDES each
    policy's prompt prefill -- so a policy whose prefill is more expensive, e.g.
    a K-branch transformation, can look marginally *faster* there while being
    slower end-to-end). Bottom row is the end-to-end rate (generated tokens /
    total wall clock), which includes the prefill. Both are shown because
    neither alone is sufficient: see the figure note.
    """
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.5))
    x = list(range(4))
    rows = [("tps", "Decode only\n(tokens/s)"), ("tps_total", "End-to-end\n(tokens/s)")]
    for r, (key, rowlabel) in enumerate(rows):
        for c, (name, _, params) in enumerate(SPECS):
            ax = axes[r][c]
            vals = [data[name][key][p] for p in POLICIES]
            base = vals[0]
            ax.bar(x, vals, width=0.62, color=[COLORS[p] for p in POLICIES],
                   edgecolor="#333333", linewidth=0.5, zorder=3)
            ax.axhline(base, color="#555555", linestyle=":", linewidth=0.7, alpha=0.7, zorder=2)
            top = max(vals) * 1.30
            ax.set_ylim(0, top)
            for xi, v in zip(x, vals):
                ax.text(xi, v + max(vals) * 0.03, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7.4, color="#222222", zorder=4)
                if xi > 0:
                    pct = (v - base) / base * 100.0
                    sign = "+" if pct >= 0 else "\u2212"
                    col = "#2E7D32" if pct >= 0 else "#B71C1C"
                    ax.text(xi, v + max(vals) * 0.14, f"({sign}{abs(pct):.1f}%)",
                            ha="center", va="bottom", fontsize=6.6, color=col,
                            style="italic", zorder=4)
            _style_axes(ax)
            if r == 0:
                ax.set_title(name, fontsize=9, fontweight="bold", pad=5)
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels(LABELS, fontsize=7.6)
            if c == 0:
                ax.set_ylabel(rowlabel, fontsize=8.8)

    handles = _legend_handles()
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.03), fontsize=8.5, handlelength=1.4, columnspacing=1.3)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.12, top=0.86, hspace=0.28, wspace=0.28)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"time_cost.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_memory_figure(data: dict) -> None:
    fig = plt.figure(figsize=(7.6, 4.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.3, 1.0], hspace=0.32, wspace=0.36,
                         left=0.09, right=0.98, top=0.86, bottom=0.13)
    x = list(range(4))

    for col, (name, _, params) in enumerate(SPECS):
        d = data[name]
        base_w = d["resident_after_load_gb"]
        totals = [d["avg_mem_gb"][p] for p in POLICIES]
        overhead_gb = [t - base_w for t in totals]
        overhead_mb = [v * 1024.0 for v in overhead_gb]

        ax_top = fig.add_subplot(gs[0, col])
        ax_top.bar(x, [base_w] * 4, width=0.62, color=WEIGHT_COLOR,
                  edgecolor="#333333", linewidth=0.5, zorder=3, label="Model weights")
        ax_top.bar(x, overhead_gb, width=0.62, bottom=[base_w] * 4,
                  color=[COLORS[p] for p in POLICIES], edgecolor="#333333",
                  linewidth=0.5, zorder=3)
        ymax_top = max(totals) * 1.14
        ax_top.set_ylim(0, ymax_top)
        for xi, t in zip(x, totals):
            ax_top.text(xi, t + ymax_top * 0.018, f"{t:.2f}", ha="center", va="bottom",
                       fontsize=7.4, color="#222222", zorder=4)
        ax_top.set_xticks(x)
        ax_top.set_xticklabels([])
        ax_top.tick_params(axis="x", length=0)
        ax_top.set_title(name, fontsize=9, fontweight="bold", pad=5)
        _style_axes(ax_top)
        if col == 0:
            ax_top.set_ylabel("Total GPU memory (GB)", fontsize=9)

        ax_top.text(3.55, base_w * 0.5, f"weights\n{base_w:.1f} GB", ha="right", va="center",
                   fontsize=6.6, color="#555555", style="italic")

        ax_bot = fig.add_subplot(gs[1, col])
        ax_bot.bar(x, overhead_mb, width=0.62, color=[COLORS[p] for p in POLICIES],
                  edgecolor="#333333", linewidth=0.5, zorder=3)
        ymax_bot = max(overhead_mb) * 1.34
        ax_bot.set_ylim(0, ymax_bot)
        for xi, v in zip(x, overhead_mb):
            ax_bot.text(xi, v + ymax_bot * 0.03, f"{v:.0f}", ha="center", va="bottom",
                       fontsize=7.0, color="#222222", zorder=4)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(LABELS, fontsize=7.6, rotation=12, ha="right", rotation_mode="anchor")
        _style_axes(ax_bot)
        if col == 0:
            ax_bot.set_ylabel("Runtime\noverhead (MB)", fontsize=8.4)

    handles = [mpatches.Patch(facecolor=WEIGHT_COLOR, edgecolor="#333333", linewidth=0.4,
                              label="Model weights")] + _legend_handles()
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
              bbox_to_anchor=(0.5, 1.03), fontsize=8.3, handlelength=1.4, columnspacing=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"memory_cost.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=None,
                    help="comma-separated '<label>:<json>:<subtitle>' triples")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.inputs:
        global SPECS, OUT_DIR
        SPECS = []
        for spec in args.inputs.split(","):
            name, fn, sub = spec.split(":", 2)
            SPECS.append((name, fn, sub))
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    make_time_figure(data)
    make_memory_figure(data)
    print(f"Wrote figures to {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*")):
        print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
