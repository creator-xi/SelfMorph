
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from selfmorph.action_space import build_action_space  # noqa: E402

def auc(score: torch.Tensor, y: torch.Tensor) -> float:
    y = y.bool()
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = torch.argsort(score)
    ranks = torch.empty_like(score, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.float64)
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "runtime.json"))
    ap.add_argument("--train-dev-pt", default=str(
        REPO_ROOT / "artifacts" / "training_data" / "qwen35_27b" / "train_dev.pt"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "results" / "trajectory"))
    ap.add_argument("--out-name", default="base_regime_large_n.json")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    anchors = build_action_space(config)
    rep_ids = {a.id for a in anchors if a.type == "REPEAT"}
    br_ids = {a.id for a in anchors if a.type == "BRANCH"}

    p = torch.load(args.train_dev_pt, map_location="cpu", weights_only=True)
    H, U, S = p["trajectory"].float(), p["utilities"], p["scores"]
    splits, datasets = p["splits"], p["datasets"]

    best = U.argmax(dim=1)
    n = H.norm(dim=-1)
    d = H[:, 1:] - H[:, :-1]
    last_relupd = (d.norm(dim=-1) / n[:, :-1].clamp_min(1e-9))[:, -1]
    min_cos = torch.nn.functional.cosine_similarity(H[:, 1:], H[:, :-1], dim=-1).min(dim=1).values

    def group_of(i: int) -> str:
        b = int(best[i])
        if b in rep_ids:
            return "repeat_best"
        if b in br_ids:
            return "branch_best"
        return "base_enough" if float(S[i, 0]) == 1.0 else "base_hopeless"

    groups_order = ["base_hopeless", "repeat_best", "base_enough", "branch_best"]
    result: dict[str, dict] = {}
    for split_name in ("eval", "train"):
        idx = [i for i, s in enumerate(splits) if s == split_name]
        gl = [group_of(i) for i in idx]
        by_group: dict[str, list[int]] = {g: [i for i, gg in zip(idx, gl) if gg == g] for g in groups_order}
        feat_by_group = {g: {"last_relupd": last_relupd[torch.tensor(v)].tolist(),
                              "min_cos": min_cos[torch.tensor(v)].tolist()}
                          for g, v in by_group.items() if v}

        summary = {}
        for g, v in by_group.items():
            if not v:
                continue
            t = torch.tensor(v)
            lru, mc = last_relupd[t], min_cos[t]
            summary[g] = {
                "n": len(v),
                "last_relupd_median": round(float(lru.median()), 4),
                "last_relupd_q25_q75": [round(float(lru.quantile(0.25)), 4), round(float(lru.quantile(0.75)), 4)],
                "min_cos_median": round(float(mc.median()), 4),
                "min_cos_q25_q75": [round(float(mc.quantile(0.25)), 4), round(float(mc.quantile(0.75)), 4)],
            }

        pairwise = {}
        present = [g for g in groups_order if by_group.get(g)]
        for i1 in range(len(present)):
            for i2 in range(i1 + 1, len(present)):
                g1, g2 = present[i1], present[i2]
                t1, t2 = torch.tensor(by_group[g1]), torch.tensor(by_group[g2])
                y = torch.cat([torch.ones(len(t1)), torch.zeros(len(t2))])
                pairwise[f"{g1}_vs_{g2}"] = {
                    "last_relupd_auc(g1>g2)": round(auc(torch.cat([last_relupd[t1], last_relupd[t2]]), y), 4),
                    "min_cos_auc(g1>g2)": round(auc(torch.cat([min_cos[t1], min_cos[t2]]), y), 4),
                }

        result[split_name] = {"n_total": len(idx), "summary": summary, "pairwise_auc": pairwise,
                              "features_by_group": feat_by_group}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / args.out_name).write_text(json.dumps({
        "purpose": "trajectory features before any morphology is applied",
        "dense_layer_ids": config["trajectory"]["dense_layer_ids"],
        "groups": {"base_enough": "best anchor is BASE and BASE score == 1.0 (converged & correct)",
                   "base_hopeless": "best anchor is BASE and BASE score == 0.0 (nothing in the pool helps)",
                   "repeat_best": "best anchor is a REPEAT anchor", "branch_best": "best anchor is a BRANCH anchor"},
        "by_split": result,
    }, indent=2), encoding="utf-8")

    for split_name, r in result.items():
        print(f"\n=== {split_name} (n={r['n_total']}) ===")
        for g in groups_order:
            s = r["summary"].get(g)
            if not s:
                continue
            print(f"  {g:14s} n={s['n']:5d}  last_relupd med={s['last_relupd_median']:.4f} "
                  f"iqr={s['last_relupd_q25_q75']}  min_cos med={s['min_cos_median']:.4f} "
                  f"iqr={s['min_cos_q25_q75']}")
        print("  pairwise AUC (P(g1's feature > g2's feature)):")
        for k, v in r["pairwise_auc"].items():
            print(f"    {k:32s} last_relupd={v['last_relupd_auc(g1>g2)']:.4f}  "
                  f"min_cos={v['min_cos_auc(g1>g2)']:.4f}")
    print(f"\n-> {out_dir / args.out_name}")

if __name__ == "__main__":
    main()
