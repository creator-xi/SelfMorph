#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from selfmorph.action_space import Anchor, build_action_space  # noqa: E402
from selfmorph.benchmarks import paper_domain  # noqa: E402

TRAIN_ROOT = ROOT / "artifacts" / "training_data"
OUTPUT_DIR = ROOT / "figures" / "task_dissociation"
PAPER_OUT_DIR = OUTPUT_DIR / "task_dissociation"

EXPECTED_TRAIN_N = 7293
EXPECTED_DATASET_KEYS = 23
EXPECTED_DOMAINS = {"BBH Reasoning", "Science Multiple Choice", "Mathematics"}

BASES = [
    {
        "key": "qwen35_27b", "label": "Qwen3.5-27B", "short": "27B",
        "color": "#472d7b", "marker": "o",
        "data": "qwen35_27b", "runtime": "config/runtime.json",
    },
    {
        "key": "qwen35_4b", "label": "Qwen3.5-4B", "short": "4B",
        "color": "#21918c", "marker": "s",
        "data": "qwen35_4b", "runtime": "config/runtime_qwen35_4b.json",
    },
    {
        "key": "gemma4_12b", "label": "Gemma4-12B", "short": "Gemma",
        "color": "#5ab54a", "marker": "D",
        "data": "gemma4_12b", "runtime": "config/runtime_gemma4_12b.json",
    },
]
MERGED_KEY = "merged"
MERGED_LABEL = "Merged (3 bases pooled)"
BASE_COLOR = {b["key"]: b["color"] for b in BASES}
BASE_COLOR[MERGED_KEY] = "#333333"
ROW_KEYS = [b["key"] for b in BASES] + [MERGED_KEY]
BASE_KEYS = [b["key"] for b in BASES]

REP_ACTION_IDS = list(range(1, 13))
BR_ACTION_IDS = list(range(13, 25))

DOMAIN_COLOR = {"BBH Reasoning": "#4C72B0", "Science Multiple Choice": "#DD8452", "Mathematics": "#55A868"}
COLOR_REP = "#E85A5A"
COLOR_BR = "#2B8A5A"

DATASET_DISPLAY = {
    "arc_challenge": "ARC-Challenge", "openbookqa": "OpenBookQA",
    "gsm8k": "GSM8K", "math500": "MATH500",
}
BBH_SUBTASK_ORDER = [
    "bbh:date_understanding", "bbh:disambiguation_qa", "bbh:formal_fallacies",
    "bbh:geometric_shapes", "bbh:hyperbaton", "bbh:logical_deduction_five_objects",
    "bbh:logical_deduction_seven_objects", "bbh:logical_deduction_three_objects",
    "bbh:movie_recommendation", "bbh:penguins_in_a_table",
    "bbh:reasoning_about_colored_objects", "bbh:ruin_names",
    "bbh:salient_translation_error_detection", "bbh:snarks",
    "bbh:sports_understanding", "bbh:temporal_sequences",
    "bbh:tracking_shuffled_objects_five_objects",
    "bbh:tracking_shuffled_objects_seven_objects",
    "bbh:tracking_shuffled_objects_three_objects",
]
BENCH_SPLIT_N = 5
NON_BBH_DATASETS = ["arc_challenge", "openbookqa", "gsm8k", "math500"]
BENCH_MARKER = {"bbh": "o", "arc_challenge": "s", "openbookqa": "^", "gsm8k": "D", "math500": "v"}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 3
plt.rcParams["font.size"] = 8

@dataclass
class Panel:
    key: str
    label: str
    scores: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray
    anchors: list[Anchor]
    domain: np.ndarray
    dataset: np.ndarray
    base_of_row: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.scores.shape[0]

def dataset_display_name(dataset: str) -> str:
    if dataset.startswith("bbh:"):
        return "bbh:" + dataset.split(":", 1)[1]
    return DATASET_DISPLAY.get(dataset, dataset)

def dataset_domain(dataset: str) -> str:
    return paper_domain(dataset)

def group_domain(group_key: str) -> str:
    return "BBH Reasoning" if group_key == "bbh" else dataset_domain(group_key)

def load_base(spec: dict) -> tuple[Panel, list[str]]:
    payload = torch.load(TRAIN_ROOT / spec["data"] / "train_dev.pt", map_location="cpu", weights_only=False)
    config = json.loads((ROOT / spec["runtime"]).read_text(encoding="utf-8"))
    aset = build_action_space(config)
    idx = [i for i, s in enumerate(payload["splits"]) if s == "train"]
    if len(idx) != EXPECTED_TRAIN_N:
        raise SystemExit(f"{spec['key']}: expected {EXPECTED_TRAIN_N} train rows, got {len(idx)}")

    costs = payload["costs"][idx].numpy().astype(np.float32)
    scores = payload["scores"][idx].numpy().astype(np.float32)
    domains = np.array([payload["domains"][i] for i in idx])
    datasets = np.array([payload["datasets"][i] for i in idx])

    panel = Panel(
        key=spec["key"], label=spec["label"],
        scores=scores, rewards=payload["rewards"][idx].numpy().astype(np.float32), costs=costs,
        anchors=list(aset.anchors), domain=domains, dataset=datasets,
    )
    context_ids = [payload["context_ids"][i] for i in idx]
    return panel, context_ids

def build_panels() -> dict[str, Panel]:
    panels: dict[str, Panel] = {}
    ctxs: dict[str, list[str]] = {}
    for spec in BASES:
        panels[spec["key"]], ctxs[spec["key"]] = load_base(spec)

    order = sorted(ctxs[BASES[0]["key"]])
    for key, ids in ctxs.items():
        pos = {c: i for i, c in enumerate(ids)}
        perm = np.array([pos[c] for c in order], dtype=np.int64)
        p = panels[key]
        panels[key] = replace(
            p, scores=p.scores[perm], rewards=p.rewards[perm], costs=p.costs[perm],
            domain=p.domain[perm], dataset=p.dataset[perm],
        )

    ordered = [panels[b["key"]] for b in BASES]
    panels[MERGED_KEY] = Panel(
        key=MERGED_KEY, label=MERGED_LABEL,
        scores=np.concatenate([p.scores for p in ordered]),
        rewards=np.concatenate([p.rewards for p in ordered]),
        costs=np.concatenate([p.costs for p in ordered]),
        anchors=ordered[0].anchors,
        domain=np.concatenate([p.domain for p in ordered]),
        dataset=np.concatenate([p.dataset for p in ordered]),
        base_of_row=np.concatenate([np.full(p.n, i, dtype=np.int64) for i, p in enumerate(ordered)]),
    )
    return panels

def net_help_hurt(s0: np.ndarray, sa: np.ndarray) -> tuple[float, float, float]:
    n = len(s0)
    help_ = float((~s0 & sa).sum()) / n * 100
    hurt = float((s0 & ~sa).sum()) / n * 100
    return help_, hurt, help_ - hurt

def resolve_fixed_actions(panel: Panel) -> tuple[int, int]:
    mean_reward = panel.rewards.mean(axis=0)
    fixed_rep = max(REP_ACTION_IDS, key=lambda a: mean_reward[a])
    fixed_br = max(BR_ACTION_IDS, key=lambda a: mean_reward[a])
    return fixed_rep, fixed_br

def _bench_blocks(dataset: str, row_idx: np.ndarray, k: int) -> list[np.ndarray]:
    rng = np.random.default_rng(zlib.crc32(dataset.encode("utf-8")))
    perm = rng.permutation(len(row_idx))
    return [row_idx[blk] for blk in np.array_split(perm, k)]

def _dissociation_records(panel: Panel, *, split_benches: bool = True, merge_bbh: bool = False) -> list[dict]:
    ok = panel.scores > 0.5
    s0 = ok[:, 0]
    fixed_rep, fixed_br = resolve_fixed_actions(panel)
    ds = panel.dataset
    is_bbh = np.array([str(d).startswith("bbh:") for d in ds])
    recs: list[dict] = []

    def add(label: str, domain: str, group_key: str, idx: np.ndarray) -> None:
        if len(idx) == 0:
            return
        _, _, rep_net = net_help_hurt(s0[idx], ok[idx, fixed_rep])
        _, _, br_net = net_help_hurt(s0[idx], ok[idx, fixed_br])
        recs.append({
            "label": label, "domain": domain, "group_key": group_key, "n": int(len(idx)),
            "rep_net_pp": rep_net, "br_net_pp": br_net,
        })

    if merge_bbh:
        add("bbh", "BBH Reasoning", "bbh", np.where(is_bbh)[0])
    else:
        for sub in BBH_SUBTASK_ORDER:
            add(dataset_display_name(sub), "BBH Reasoning", "bbh", np.where(ds == sub)[0])

    for bench in NON_BBH_DATASETS:
        idx = np.where(ds == bench)[0]
        if not split_benches:
            add(dataset_display_name(bench), dataset_domain(bench), bench, idx)
        else:
            for b, blk in enumerate(_bench_blocks(bench, idx, BENCH_SPLIT_N)):
                add(f"{dataset_display_name(bench)} \u00d7{b + 1}", dataset_domain(bench), bench, blk)
    return recs

def axes_grid(
    fig, *, width_in, height_in, left_in, right_in, top_in, bottom_in,
    ncols, nrows, hgap_in, vgap_in,
):
    grid_w = width_in - left_in - right_in
    grid_h = height_in - top_in - bottom_in
    cell_w = (grid_w - (ncols - 1) * hgap_in) / ncols
    cell_h = (grid_h - (nrows - 1) * vgap_in) / nrows
    boxes = []
    for r in range(nrows):
        for c in range(ncols):
            l = left_in + c * (cell_w + hgap_in)
            t = top_in + r * (cell_h + vgap_in)
            boxes.append([l / width_in, 1.0 - (t + cell_h) / height_in, cell_w / width_in, cell_h / height_in])
    return boxes, cell_w, cell_h

def _draw_dissociation_panel(
    ax, recs: list[dict], *, y_lo: float, y_hi: float,
    marker_size: float, alpha: float, marker_of=None,
    x_lim: tuple[float, float] | None = None,
) -> None:
    if marker_of is None:
        marker_of = lambda r: "o"  # noqa: E731
    if x_lim is not None:
        x_lo, x_hi = x_lim
    else:
        xs = np.array([r["rep_net_pp"] for r in recs])
        pad = max(6.0, 0.10 * float(xs.max() - xs.min()))
        x_lo, x_hi = xs.min() - pad, xs.max() + pad
    if x_lo < 0:
        ax.add_patch(Rectangle((x_lo, 0), -x_lo, y_hi, facecolor=COLOR_BR, alpha=0.07, zorder=0))
    if x_hi > 0:
        ax.add_patch(Rectangle((0, y_lo), x_hi, -y_lo, facecolor=COLOR_REP, alpha=0.07, zorder=0))
    diag_lo, diag_hi = max(x_lo, y_lo), min(x_hi, y_hi)
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="#999999", linewidth=0.6, linestyle="--", zorder=0.5)
    ax.axhline(0, color="#CCCCCC", linewidth=0.5, zorder=0.4)
    ax.axvline(0, color="#CCCCCC", linewidth=0.5, zorder=0.4)
    for r in recs:
        ax.scatter(
            r["rep_net_pp"], np.clip(r["br_net_pp"], y_lo, y_hi), s=marker_size,
            marker=marker_of(r), color=DOMAIN_COLOR[r["domain"]], edgecolors="white",
            linewidths=0.4, alpha=alpha, zorder=2,
        )
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Fixed-REPEAT net effect vs Base (pp)", fontsize=8)
    ax.set_ylabel("Fixed-BR net effect vs Base (pp)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.grid(False)

def _double_lines(ax, factor: float = 2.0) -> None:
    for line in ax.lines:
        line.set_linewidth(line.get_linewidth() * factor)
        if line.get_markersize():
            line.set_markersize(line.get_markersize() * factor)
    for coll in ax.collections:
        sizes = coll.get_sizes()
        if sizes is not None and len(sizes):
            coll.set_sizes(sizes * (factor ** 2))
        lw = coll.get_linewidths()
        if lw is not None and len(lw):
            coll.set_linewidths([w * factor for w in lw])

def build_dissociation_granular(panels: dict[str, Panel]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    y_lo, y_hi = -20.0, 40.0

    rows_all = []
    for key in ROW_KEYS:
        p = panels[key]
        for r in _dissociation_records(p, split_benches=True, merge_bbh=False):
            rows_all.append([
                key, p.label, r["label"], r["domain"], r["n"], r["rep_net_pp"],
                r["br_net_pp"], min(max(r["br_net_pp"], y_lo), y_hi),
            ])
    with (OUTPUT_DIR / "task_dissociation_data.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow([
            "base", "base_label", "group", "domain", "n", "rep_net_pp",
            "br_net_pp_raw", "br_net_pp_clipped_for_plot",
        ])
        w.writerows(rows_all)

    data = {key: _dissociation_records(panels[key], split_benches=True, merge_bbh=False) for key in BASE_KEYS}
    width_in, height_in = 31.5, 12.4
    fig = plt.figure(figsize=(width_in, height_in))
    boxes, _, _ = axes_grid(
        fig, width_in=width_in, height_in=height_in, left_in=2.05, right_in=0.35,
        top_in=2.35, bottom_in=2.35, ncols=3, nrows=1, hgap_in=2.60, vgap_in=0.0,
    )
    for bi, key in enumerate(BASE_KEYS):
        ax = fig.add_axes(boxes[bi])
        recs = data[key]
        if key == "gemma4_12b":
            panel_y_lo, panel_y_hi, panel_x_lim = -10.0, 10.0, (-10.0, 10.0)
        else:
            panel_y_lo, panel_y_hi, panel_x_lim = y_lo, y_hi, None
        _draw_dissociation_panel(
            ax, recs, y_lo=panel_y_lo, y_hi=panel_y_hi, marker_size=48, alpha=0.92,
            marker_of=lambda r: BENCH_MARKER[r["group_key"]], x_lim=panel_x_lim,
        )
        _double_lines(ax)
        ax.set_xlabel("Repeat vs Base (pp)", fontsize=32, labelpad=20)
        ax.set_ylabel("Branch vs Base (pp)", fontsize=32, labelpad=10)
        ax.tick_params(labelsize=30, pad=14)
        ax.set_title(panels[key].label, fontsize=40, fontweight="bold", color=BASE_COLOR[key], pad=30)
    fig.text(0.5, 0.97, "Per-task dissociation", fontsize=52, fontweight="bold", ha="center", va="top")

    name = "task_dissociation"
    fig.savefig(PAPER_OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(PAPER_OUT_DIR / f"{name}_300dpi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {PAPER_OUT_DIR / name}.pdf/.png")

def main() -> None:
    panels = build_panels()
    build_dissociation_granular(panels)

if __name__ == "__main__":
    main()
