#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

def _safe_z(values: list[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5
    if std < 1e-12:
        std = 1.0
    return [(v - mean) / std for v in values]

def load_records(pattern: str, max_span_width: int | None = None) -> list[dict]:
    records = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if (
                    max_span_width is not None
                    and rec.get("layer_span")
                    and (rec["layer_span"][1] - rec["layer_span"][0]) > max_span_width
                ):
                    continue
                records.append(rec)
    return records

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", required=True, help="Glob of scan.jsonl shards")
    parser.add_argument("--out", required=True, help="Output directory for the two summary CSVs")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument(
        "--max-span-width", type=int, default=None,
        help=(
            "Optional post-hoc filter: drop records whose span_len = j - i > "
            "this value before ranking (default: None = rank everything). "
            "Useful for limiting the candidate pool to a cost-driven width "
            "cap (e.g. 10 -> span_len 1..10), the ranking z-scores are then "
            "recomputed over the filtered subset."
        ),
    )
    args = parser.parse_args()

    records = load_records(args.glob, args.max_span_width)
    if not records:
        raise SystemExit(f"No records matched glob: {args.glob}")

    by_context: dict[str, dict[int, dict]] = defaultdict(dict)
    anchor_meta: dict[int, dict] = {}
    for rec in records:
        by_context[rec["context_id"]][rec["anchor_id"]] = rec
        anchor_meta[rec["anchor_id"]] = {
            "type": rec["anchor_type"], "layer_span": rec["layer_span"],
            "r": rec["r"], "K": rec["K"],
        }
    base_id = next(aid for aid, m in anchor_meta.items() if m["type"] == "BASE")

    all_dataset_names = sorted({rec["dataset"] for rec in records})
    score_columns = [f"score_{ds}" for ds in all_dataset_names]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for anchor_type in ("REPEAT", "BRANCH"):
        candidate_ids = sorted(aid for aid, m in anchor_meta.items() if m["type"] == anchor_type)
        if not candidate_ids:
            continue

        scores: dict[int, dict[str, list[float]]] = {aid: defaultdict(list) for aid in candidate_ids}
        deltas: dict[int, dict[str, list[float]]] = {aid: defaultdict(list) for aid in candidate_ids}

        for context_id, recs in by_context.items():
            if base_id not in recs:
                continue
            dataset = recs[base_id]["dataset"]
            noop_score = float(recs[base_id]["score"])
            for aid in candidate_ids:
                rec = recs.get(aid)
                if rec is None:
                    continue
                score = float(rec["score"])
                scores[aid][dataset].append(score)
                deltas[aid][dataset].append(score - noop_score)

        datasets = sorted({d for aid in candidate_ids for d in scores[aid]})
        z_by_dataset: dict[str, dict[int, float]] = {}
        for ds in datasets:
            mean_delta = {aid: sum(deltas[aid][ds]) / len(deltas[aid][ds]) for aid in candidate_ids if deltas[aid][ds]}
            ids = list(mean_delta.keys())
            zs = _safe_z(list(mean_delta.values()))
            z_by_dataset[ds] = dict(zip(ids, zs))

        rows = []
        for aid in candidate_ids:
            mean_score = {ds: sum(scores[aid][ds]) / len(scores[aid][ds]) for ds in scores[aid] if scores[aid][ds]}
            balanced_z = sum(z_by_dataset[ds].get(aid, 0.0) for ds in datasets)

            present_scores = [mean_score[ds] for ds in datasets if ds in mean_score]
            combined_5050 = sum(present_scores) / len(present_scores) if present_scores else 0.0
            row = {
                "rank": 0,
                "anchor_id": aid,
                "layer_span": anchor_meta[aid]["layer_span"],
                "r": anchor_meta[aid]["r"],
                "K": anchor_meta[aid]["K"],
            }
            for ds in all_dataset_names:
                row[f"score_{ds}"] = mean_score.get(ds)
            row["combined_5050"] = combined_5050
            row["balanced_z_delta"] = balanced_z
            rows.append(row)
        rows.sort(key=lambda r: r["balanced_z_delta"], reverse=True)
        for rank, r in enumerate(rows, start=1):
            r["rank"] = rank

        csv_path = out_dir / f"{anchor_type.lower()}_span_scan.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["rank", "anchor_id", "layer_span", "r", "K",
                               *score_columns, "combined_5050", "balanced_z_delta"],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n=== {anchor_type} span scan ({len(rows)} candidates) -> {csv_path} ===")
        header = f"{'rank':>4} {'layer_span':>12} " + " ".join(f"{ds:>10}" for ds in all_dataset_names) + f" {'5050':>8} {'z_delta':>8}"
        print(header)
        for r in rows[: args.top_k]:
            score_str = " ".join(
                f"{r[f'score_{ds}']:.4f}" if r.get(f"score_{ds}") is not None else f"{'--':>10}"
                for ds in all_dataset_names
            )
            print(
                f"{r['rank']:>4} {str(r['layer_span']):>12} "
                f"{score_str}  {r['combined_5050']:.4f}  {r['balanced_z_delta']:.4f}"
            )

    noop_scores: dict[str, list[float]] = defaultdict(list)
    for context_id, recs in by_context.items():
        if base_id in recs:
            noop_scores[recs[base_id]["dataset"]].append(float(recs[base_id]["score"]))
    print("\n=== BASE baseline ===")
    for ds, vals in sorted(noop_scores.items()):
        print(f"  {ds}: mean={sum(vals) / len(vals):.4f} (n={len(vals)})")

if __name__ == "__main__":
    main()
