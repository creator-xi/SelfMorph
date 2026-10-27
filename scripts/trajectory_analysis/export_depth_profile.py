
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

COLOR_REP = "#E85A5A"
COLOR_REP_DARK = "#A32020"
COLOR_BR = "#2B8A5A"
COLOR_BR_DARK = "#12563A"
COLOR_BASE = "#8C8C8C"
COLOR_GRID = "#DDDDDD"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 16, "axes.linewidth": 1.1,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4.5, "ytick.major.size": 4.5, "legend.frameon": False,
})

class Traces:
    def __init__(self, out_dir: Path, context_id: str):
        self.dir = out_dir / "traces" / context_id
        self.context_id = context_id
        self.blobs = {
            int(p.stem.replace("anchor", "")): torch.load(p, map_location="cpu", weights_only=True)
            for p in sorted(self.dir.glob("anchor*.pt"))
        }
        if 0 not in self.blobs:
            raise FileNotFoundError(f"no base (anchor0) trace under {self.dir}")
        self.base = self.blobs[0]["states"].squeeze(1)
        self.N = self.blobs[0]["num_layers"]

    def rep(self, aid: int):
        b = self.blobs[aid]
        return b["states"].squeeze(1), b["layer_span"][0], b["layer_span"][1], b["r"]

    def br(self, aid: int):
        b = self.blobs[aid]
        return b["states"], b["layer_span"][0], b["layer_span"][1], b["K"]

def _style_axes(ax) -> None:
    ax.grid(True, color=COLOR_GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("0.35")

def draw_panel_a(ax, tr: Traces, rep_aids: list[int], br_aid: int) -> None:
    """Base / repeat / branch L2 norm profile across executed depth."""
    nb = tr.base.norm(dim=-1).numpy()
    st, bi, bj, K = tr.br(br_aid)
    _, ri, rj, _ = tr.rep(rep_aids[0])
    ax.axvspan(bi, bj, color=COLOR_BR, alpha=0.10, zorder=1)
    ax.axvspan(ri, rj, color=COLOR_REP, alpha=0.10, zorder=1)
    nk = st.norm(dim=-1).numpy()
    for k in range(nk.shape[1]):
        ax.plot(range(nk.shape[0]), nk[:, k], color=COLOR_BR, lw=0.5, alpha=0.6, zorder=3)
    ax.plot([bj], [st[bj].mean(0).norm().item()], marker="o", ms=5, color=COLOR_BR_DARK,
            mec="white", mew=0.6, zorder=6)
    for aid, c, ls in zip(rep_aids, (COLOR_REP, COLOR_REP_DARK), ("-", "--")):
        s, i, j, r = tr.rep(aid)
        n = s.norm(dim=-1).numpy()
        w = r * (j - i)
        ax.plot(range(len(n)), n, color=c, lw=1.2, ls=ls, zorder=4)
        ax.plot(range(j, j + w + 1), n[j:j + w + 1], color=c, lw=3.2, alpha=0.30, zorder=2)
    ax.plot(range(len(nb)), nb, color=COLOR_BASE, lw=2.0, zorder=3.5)
    ax.set_yscale("log")
    ax.set_xlabel("executed depth")
    ax.set_ylabel("L2 norm of hidden state")

    h = [
        Line2D([], [], color=COLOR_BASE, lw=2.0, label="base"),
        Line2D([], [], color=COLOR_REP, lw=1.2, label=f"repeat [{ri},{rj}) x{tr.blobs[rep_aids[0]]['r']}"),
        Line2D([], [], color=COLOR_REP_DARK, lw=1.2, ls="--",
               label=f"repeat [{ri},{rj}) x{tr.blobs[rep_aids[1]]['r']}"),
        Line2D([], [], color=COLOR_BR, lw=0.7, label=f"branch [{bi},{bj}) K={K}"),
        Line2D([], [], color=COLOR_BR_DARK, lw=0, marker="o", ms=5, mec="white", mew=0.6,
               label="merged state"),
    ]
    ax.legend(handles=h, loc="lower right", fontsize=12.0, handlelength=1.6, labelspacing=0.22)
    _style_axes(ax)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "artifacts" / "results" / "trajectory"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "figures" / "trajectory" / "trajectory"))
    ap.add_argument("--context", default="arc_challenge::MCAS_2006_8_35")
    ap.add_argument("--rep-anchors", default="4,5,6")
    ap.add_argument("--br-anchors", default="13,14,15")
    ap.add_argument("--name", default="depth_profile")
    args = ap.parse_args()

    tr = Traces(Path(args.results_dir), args.context)
    rep_aids = [int(x) for x in args.rep_anchors.split(",")]
    br_aids = [int(x) for x in args.br_anchors.split(",")]

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.4))
    draw_panel_a(ax, tr, rep_aids[:2], br_aids[-1])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{args.name}_300dpi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / (args.name + '.pdf')}")
    print(f"wrote {out_dir / (args.name + '_300dpi.png')}")

if __name__ == "__main__":
    main()
