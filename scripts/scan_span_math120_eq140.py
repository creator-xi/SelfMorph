#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfmorph.action_space import Anchor, branch_cost, rep_cost
from selfmorph.benchmarks.prompts import render_prompt
from selfmorph.benchmarks.types import EvalSample, RAW_TEXT_ROLE
from selfmorph.executor import execute_anchor
from selfmorph.model import load_frozen_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scan_span_math_eq import (  # noqa: E402
    math_16_prompt,
    _read_num_layers_from_model_path,
    score_eq_16 as score_eq_140,
    score_math_16 as score_math_120,
)

DATASETS_DIR = Path(__file__).resolve().parents[1] / "resources" / "datasets"
MATH_MAX_NEW = 64
EQ_MAX_NEW = 64
REP_R = 1
BR_K = 4

REP_CANDIDATE_SPANS: list[tuple[int, int]] = [
    (2, 12), (3, 12), (13, 19), (14, 17), (15, 21), (16, 21), (20, 28), (21, 31), (22, 23), (22, 29),
    (22, 30), (22, 31), (23, 29), (24, 29), (24, 30), (24, 31), (24, 32), (25, 27), (25, 29), (25, 30),
    (26, 29), (26, 30), (26, 31), (26, 34), (27, 29), (30, 36), (34, 44), (35, 45), (36, 43), (37, 45),
    (38, 48), (39, 45), (39, 48), (41, 42), (41, 44), (41, 45), (41, 46), (41, 47), (41, 48), (41, 51),
    (42, 44), (43, 47), (45, 49), (46, 52), (47, 52),
]
BR_CANDIDATE_SPANS: list[tuple[int, int]] = [
    (1, 2), (1, 3), (1, 6), (1, 7), (1, 10), (1, 11), (2, 9), (3, 6), (3, 7), (3, 8),
    (3, 9), (3, 11), (3, 12), (3, 13), (4, 7), (4, 8), (4, 9), (4, 13), (4, 14), (5, 6),
    (5, 7), (5, 8), (5, 10), (5, 13), (6, 13), (7, 8), (8, 11), (8, 12), (8, 13), (8, 14),
    (8, 15), (9, 13), (9, 17), (10, 11), (10, 12), (11, 20), (13, 18), (14, 19), (18, 19), (29, 36),
    (30, 32),
]
assert len(REP_CANDIDATE_SPANS) == 45
assert len(BR_CANDIDATE_SPANS) == 41

def load_candidate_spans_from_csv(csv_path: str, top_k: int) -> list[tuple[int, int]]:
    """Read the top-k spans (by balanced_z_delta) from a rank_span_math_eq.py
    CSV. Lets the validation scan target e.g. Qwen3.5-4B's span_len<=10
    REPEAT/BR top-30 instead of the hardcoded 27B 45+41 union above.
    """
    import ast as _ast
    import csv as _csv
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            r["layer_span"] = _ast.literal_eval(r["layer_span"])
            rows.append(r)
    rows.sort(key=lambda r: float(r["balanced_z_delta"]), reverse=True)
    spans = [tuple(r["layer_span"]) for r in rows[:top_k]]
    if len(spans) != top_k:
        raise SystemExit(f"{csv_path}: only {len(spans)} rows < requested top_k={top_k}")
    return spans

def load_math_120() -> list[EvalSample]:
    with (DATASETS_DIR / "math_120.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = []
    for qid, row in raw.items():
        prompt = math_16_prompt(row["question"])
        samples.append(EvalSample(
            qid=qid, task="math_120",
            messages=[{"role": RAW_TEXT_ROLE, "content": prompt}],
            meta={"answer": row["answer"]},
        ))
    return samples

def load_eq_140() -> list[EvalSample]:
    with (DATASETS_DIR / "eq_140.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = []
    for qid, row in raw.items():
        reference = row.get("reference_answer", row.get("reference_answer_fullscale", {}))
        samples.append(EvalSample(
            qid=qid, task="eq_140",
            messages=[{"role": RAW_TEXT_ROLE, "content": row["prompt"]}],
            meta={"reference": reference},
        ))
    return samples

def build_validation_anchors(num_layers: int, k_eff: float,
                             rep_spans: list[tuple[int, int]],
                             br_spans: list[tuple[int, int]]) -> list[Anchor]:
    anchors: list[Anchor] = [Anchor(id=0, type="BASE", layer_span=None, analytic_cost=1.0)]
    next_id = 1
    for span in rep_spans:
        cost = rep_cost(span, REP_R, num_layers)
        anchors.append(Anchor(id=next_id, type="REPEAT", layer_span=span, r=REP_R, analytic_cost=cost))
        next_id += 1
    for span in br_spans:
        cost = branch_cost(span, k_eff, num_layers)
        anchors.append(Anchor(id=next_id, type="BRANCH", layer_span=span, K=BR_K, K_eff=k_eff, analytic_cost=cost))
        next_id += 1
    return anchors

@dataclass
class ScanRecord:
    context_id: str
    dataset: str
    anchor_id: int
    anchor_type: str
    layer_span: Optional[list]
    r: Optional[int]
    K: Optional[int]
    score: float
    cost: float
    reward: float
    latency_s: float

def _completed_keys(output_path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not output_path.is_file():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["context_id"], int(rec["anchor_id"])))
            except Exception:
                continue
    return done

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fsync-every", type=int, default=200)
    parser.add_argument("--rep-spans-csv", default=None,
                        help="rank CSV to take REPEAT top-k spans from (default: hardcoded 27B 45-span union)")
    parser.add_argument("--br-spans-csv", default=None,
                        help="rank CSV to take BRANCH top-k spans from (default: hardcoded 27B 41-span union)")
    parser.add_argument("--rep-top-k", type=int, default=30)
    parser.add_argument("--br-top-k", type=int, default=30)
    parser.add_argument("--model-path", default=None,
                        help="Override frozen base-LM dir; num_layers read from its config.json")
    parser.add_argument("--device-map", default=None,
                        help="Optional accelerate device_map for multi-GPU models (e.g. 'auto')")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    lambda_cost = config["action_space"]["lambda_cost"]
    k_eff = config["action_space"]["branch_cost_table"][str(BR_K)]

    if args.model_path:

        num_layers = _read_num_layers_from_model_path(args.model_path)
        os.environ["SELFMORPH_MODEL_PATH"] = args.model_path
        print(
            f"[shard {args.shard_id}/{args.num_shards}] --model-path: scanning "
            f"{args.model_path} (num_layers={num_layers} read from its config.json)",
            flush=True,
        )
    else:
        num_layers = config["model"]["num_layers"]

    if args.rep_spans_csv:
        rep_spans = load_candidate_spans_from_csv(args.rep_spans_csv, args.rep_top_k)
    else:
        rep_spans = REP_CANDIDATE_SPANS
    if args.br_spans_csv:
        br_spans = load_candidate_spans_from_csv(args.br_spans_csv, args.br_top_k)
    else:
        br_spans = BR_CANDIDATE_SPANS

    model, tokenizer = load_frozen_model(device=args.device, device_map=args.device_map)
    anchors = build_validation_anchors(num_layers, k_eff, rep_spans, br_spans)
    print(
        f"[shard {args.shard_id}/{args.num_shards}] validation anchors: BASE + "
        f"{len(rep_spans)} REPEAT + {len(br_spans)} BRANCH = {len(anchors)}",
        flush=True,
    )

    samples: list[tuple[str, EvalSample]] = (
        [("math_120", s) for s in load_math_120()] + [("eq_140", s) for s in load_eq_140()]
    )
    my_samples = samples[args.shard_id::args.num_shards]

    output_dir = Path(args.output_dir)
    shard_dir = output_dir / f"shard{args.shard_id}of{args.num_shards}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    out_path = shard_dir / "scan.jsonl"
    done = _completed_keys(out_path)

    f = open(out_path, "a", buffering=1, encoding="utf-8")
    n_written = 0
    try:
        for ds, sample in my_samples:
            context_id = f"{ds}::{sample.qid}"
            prompt_text = render_prompt(tokenizer, sample.messages)
            enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            max_new = MATH_MAX_NEW if ds == "math_120" else EQ_MAX_NEW

            for anchor in anchors:
                if (context_id, anchor.id) in done:
                    continue
                result = execute_anchor(
                    model, tokenizer, enc, anchor, num_layers=num_layers,
                    action_cfg=config["action_space"], max_new_tokens=max_new,
                )
                if ds == "math_120":
                    score = score_math_120(result["text"], sample.meta["answer"])
                else:
                    score = score_eq_140(result["text"], sample.meta["reference"])
                rec = ScanRecord(
                    context_id=context_id, dataset=ds, anchor_id=anchor.id,
                    anchor_type=anchor.type,
                    layer_span=list(anchor.layer_span) if anchor.layer_span else None,
                    r=anchor.r, K=anchor.K,
                    score=score, cost=anchor.analytic_cost,
                    reward=score - lambda_cost * (anchor.analytic_cost - 1.0),
                    latency_s=result["latency_s"],
                )
                f.write(json.dumps(rec.__dict__) + "\n")
                f.flush()
                n_written += 1
                if n_written % args.fsync_every == 0:
                    os.fsync(f.fileno())
                    print(
                        f"[shard {args.shard_id}/{args.num_shards}] {n_written} records written "
                        f"(latest: {context_id} anchor={anchor.id} {anchor.type} span={anchor.layer_span} score={score:.3f})",
                        flush=True,
                    )
    finally:
        os.fsync(f.fileno())
        f.close()
    print(f"[shard {args.shard_id}/{args.num_shards}] done, {n_written} new records -> {out_path}")

if __name__ == "__main__":
    main()
