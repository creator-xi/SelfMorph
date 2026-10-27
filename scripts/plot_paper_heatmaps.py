#!/usr/bin/env python3
"""Plot span-screening score-gain heatmaps.

Reads the diagnostic scan shards (`math_16` numerical estimation probe and
`eq_16` emotion-intensity probe) for Qwen3.5-27B, Qwen3.5-4B, and Gemma4-12B,
and renders the triangular score-gain heatmaps `Delta(i, j) = score(i, j) - score_BASE`
into `figures/paper_heatmaps/<model>/<type>_<dataset>_delta_vs_base.{pdf,png}`
along with `*_data.csv`.

Usage:
    .venv/bin/python scripts/plot_paper_heatmaps.py
"""
from __future__ import annotations

import csv
import glob as globmod
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "figures" / "paper_heatmaps"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MODELS = [
    {
        "tag": "qwen35_27b",
        "display": "Qwen3.5-27B",
        "glob": str(REPO_ROOT / "artifacts" / "scan_span_math_eq" / "shard*" / "scan.jsonl"),
        "num_layers": 64,
    },
    {
        "tag": "qwen35_4b",
        "display": "Qwen3.5-4B",
        "glob": str(REPO_ROOT / "artifacts" / "scan_span_math_eq_qwen35_4b" / "shard*" / "scan.jsonl"),
        "num_layers": 32,
    },
    {
        "tag": "gemma4_12b",
        "display": "Gemma-4-12B",
        "glob": str(REPO_ROOT / "artifacts" / "scan_span_math_eq_gemma4_12b" / "shard*" / "scan.jsonl"),
        "num_layers": 48,
    },
]
DATASETS = ["math_16", "eq_16"]
ANCHOR_TYPES = ["REPEAT", "BRANCH"]

PERCENTILE_OVERRIDES = {
    ("qwen35_27b", "REPEAT", "eq_16"): 90.0,
}
VMIN_VMAX_OVERRIDES = {
    ("qwen35_4b", "REPEAT", "eq_16"): {"vmin_pct": 10, "vmax_pct": 100},
    ("gemma4_12b", "BRANCH", "math_16"): {"vmin_pct": 0, "vmax_pct": 100},
    ("gemma4_12b", "REPEAT", "eq_16"): {"vmin_pct": 0, "vmax_pct": 100},
    ("gemma4_12b", "REPEAT", "math_16"): {"vmin_pct": 0, "vmax_pct": 100},
    ("gemma4_12b", "BRANCH", "eq_16"): {"vmin_pct": 0, "vmax_pct": 100},
}
ABSOLUTE_VMIN_VMAX_OVERRIDES: dict[tuple[str, str, str], tuple[float, float | None]] = {
    ("gemma4_12b", "REPEAT", "math_16"): (-0.4, 0.1),
    ("gemma4_12b", "BRANCH", "math_16"): (-0.4, 0.04),
    ("gemma4_12b", "REPEAT", "eq_16"): (-0.10, 0.10),
}
CUSTOM_TICKS_OVERRIDES: dict[tuple[str, str, str], list[float]] = {
    ("gemma4_12b", "REPEAT", "math_16"): [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1],
    ("gemma4_12b", "BRANCH", "math_16"): [-0.4, -0.2, 0.0, 0.02, 0.04],
    ("gemma4_12b", "REPEAT", "eq_16"): [-0.10, -0.05, 0.0, 0.05, 0.10],
}

def load_model_records(glob_pattern: str) -> list[dict]:
    records = []
    for path in sorted(globmod.glob(glob_pattern)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records

def aggregate(records: list[dict]) -> tuple[dict, dict]:
    sums: dict = defaultdict(float)
    counts: dict = defaultdict(int)
    noop_sums = defaultdict(float)
    noop_counts = defaultdict(int)

    for rec in records:
        ds = rec["dataset"]
        if rec["anchor_type"] == "BASE":
            noop_sums[ds] += rec["score"]
            noop_counts[ds] += 1
            continue
        i, j = rec["layer_span"]
        key = (rec["anchor_type"], ds, (i, j))
        sums[key] += rec["score"]
        counts[key] += 1

    base_mean = {ds: noop_sums[ds] / noop_counts[ds] for ds in noop_sums}
    span_mean: dict = {t: defaultdict(dict) for t in ANCHOR_TYPES}
    for (atype, ds, span), total in sums.items():
        n = counts[(atype, ds, span)]
        span_mean[atype][ds][span] = total / n
    return base_mean, span_mean

def build_matrix(span_mean_ds: dict, num_layers: int, subtract: float | None) -> np.ndarray:
    Z = np.full((num_layers, num_layers), np.nan, dtype=float)
    offset = subtract if subtract is not None else 0.0
    for (i, j), mean_score in span_mean_ds.items():
        Z[j - 1, i] = mean_score - offset
    return Z

def plot_delta_heatmap(
    Z: np.ndarray, num_layers: int, model_display: str,
    anchor_type: str, dataset: str, base_mean_ds: float,
    out_path_base: Path, percentile: float = 99.0,
    vmin_override: float | None = None, vmax_override: float | None = None,
    custom_ticks: list[float] | None = None,
) -> dict:
    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        raise RuntimeError(f"no finite values for {model_display} {anchor_type} {dataset}")

    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad(color="#eeeeee")

    if vmin_override is not None or vmax_override is not None:
        vmin = vmin_override if vmin_override is not None else -float(np.nanpercentile(np.abs(finite), percentile))
        vmax = vmax_override if vmax_override is not None else float(np.nanpercentile(np.abs(finite), percentile))
        vmin, vmax = min(vmin, -1e-6), max(vmax, 1e-6)
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        exceeds_min = bool(np.any(finite < vmin - 1e-12))
        exceeds_max = bool(np.any(finite > vmax + 1e-12))
        vlim = max(abs(vmin), abs(vmax))
    else:
        vlim = max(float(np.nanpercentile(np.abs(finite), percentile)), 1e-6)
        norm = mcolors.Normalize(vmin=-vlim, vmax=vlim)
        exceeds_min = exceeds_max = bool(np.any(np.abs(finite) > vlim + 1e-12))

    if exceeds_min and exceeds_max:
        extend = "both"
    elif exceeds_min:
        extend = "min"
    elif exceeds_max:
        extend = "max"
    else:
        extend = "neither"

    fig_w = 4.6 if num_layers <= 40 else 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.86), constrained_layout=True)
    im = ax.imshow(
        Z, origin="lower", cmap=cmap, norm=norm, interpolation="none",
        extent=(-0.5, num_layers - 0.5, 0.5, num_layers + 0.5), aspect="equal",
    )
    morphology = "Repeat" if anchor_type == "REPEAT" else "Branch"
    ax.plot([-0.5, num_layers - 0.5], [0.5, num_layers + 0.5], color="0.35", lw=0.8, ls=(0, (4, 3)), zorder=3, alpha=0.9)
    ax.set_xlabel(r"Span start layer $i$")
    ax.set_ylabel(r"Span end layer $j$", labelpad=10.0)
    ax.set_title(f"{morphology} morphology, {model_display}")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.set_ylim(0.5, num_layers + 0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, extend=extend)
    cbar.set_label("Score gain over Base LM")
    cbar.ax.tick_params(labelsize=9)

    if custom_ticks is not None:
        cbar.set_ticks(sorted(custom_ticks))
    elif isinstance(norm, mcolors.TwoSlopeNorm):
        in_range_ticks = [t for t in cbar.get_ticks() if norm.vmin - 1e-9 <= t <= norm.vmax + 1e-9]
        new_ticks = list(in_range_ticks)
        if not any(t > 1e-9 for t in in_range_ticks) and norm.vmax > 1e-9:
            new_ticks.append(float(norm.vmax))
        if not any(t < -1e-9 for t in in_range_ticks) and norm.vmin < -1e-9:
            new_ticks.append(float(norm.vmin))
        if len(new_ticks) != len(in_range_ticks):
            cbar.set_ticks(sorted(new_ticks))

    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path_base.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    return {
        "vmin_used": float(norm.vmin),
        "vmax_used": float(norm.vmax),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "extend": extend,
    }

def write_data_csv(Z: np.ndarray, num_layers: int, out_path_base: Path, column: str) -> None:
    csv_path = out_path_base.with_name(out_path_base.name + "_data.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", column])
        for r in range(num_layers):
            j = r + 1
            for c in range(num_layers):
                i = c
                val = Z[r, c]
                if np.isfinite(val):
                    w.writerow([i, j, f"{val:.6f}"])

def main() -> None:
    for model in MODELS:
        print(f"=== {model['display']} ({model['tag']}, N={model['num_layers']}) ===")
        records = load_model_records(model["glob"])
        base_mean, span_mean = aggregate(records)

        for atype in ANCHOR_TYPES:
            for ds in DATASETS:
                span_mean_ds = span_mean[atype].get(ds, {})
                if not span_mean_ds:
                    continue
                Z_delta = build_matrix(span_mean_ds, model["num_layers"], subtract=base_mean[ds])
                out_delta = OUT_DIR / model["tag"] / f"{atype.lower()}_{ds}_delta_vs_base"

                key = (model["tag"], atype, ds)
                custom_ticks = CUSTOM_TICKS_OVERRIDES.get(key)
                if key in ABSOLUTE_VMIN_VMAX_OVERRIDES:
                    vmin_ov, vmax_ov = ABSOLUTE_VMIN_VMAX_OVERRIDES[key]
                    if vmax_ov is None:
                        finite = Z_delta[np.isfinite(Z_delta)]
                        vmax_ov = float(np.nanmax(finite))
                    if custom_ticks is not None and vmax_ov not in custom_ticks:
                        custom_ticks = sorted(custom_ticks + [vmax_ov])
                    percentile = 99.0
                elif key in VMIN_VMAX_OVERRIDES:
                    asym = VMIN_VMAX_OVERRIDES[key]
                    finite = Z_delta[np.isfinite(Z_delta)]
                    vmin_ov = float(np.percentile(finite, asym["vmin_pct"]))
                    vmax_ov = float(np.percentile(finite, asym["vmax_pct"]))
                    percentile = 99.0
                else:
                    vmin_ov = vmax_ov = None
                    percentile = PERCENTILE_OVERRIDES.get(key, 99.0)

                dstats = plot_delta_heatmap(
                    Z_delta, model["num_layers"], model["display"], atype, ds, base_mean[ds],
                    out_delta, percentile=percentile, vmin_override=vmin_ov, vmax_override=vmax_ov,
                    custom_ticks=custom_ticks,
                )
                write_data_csv(Z_delta, model["num_layers"], out_delta, "delta_vs_base")
                print(
                    f"  [{atype}/{ds}] range=[{dstats['min']:.3f},{dstats['max']:.3f}] "
                    f"-> {out_delta.relative_to(REPO_ROOT)}.{{pdf,png}}"
                )

if __name__ == "__main__":
    main()
