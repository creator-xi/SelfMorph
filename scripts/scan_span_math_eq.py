#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
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

DATASETS_DIR = Path(__file__).resolve().parents[1] / "resources" / "datasets"
MATH_MAX_NEW = 64
EQ_MAX_NEW = 64
REP_R = 1
BR_K = 4
EMOTION_KEYS = ["emotion1_score", "emotion2_score", "emotion3_score", "emotion4_score"]

MATH_16_PREFIX = (
    "You are a highly intelligent AI with extraordinary intuition and can "
    "easily make accurate estimations. For the following question, always "
    "provide your best integer guess, even if you cannot compute it exactly. "
    "Do not show any reasoning or working.\n\n"
)

def math_16_prompt(question: str) -> str:
    return f"{MATH_16_PREFIX}Q: {question}\nA: The answer (arabic numerals) is"

def strip_thinking(text: str) -> str:
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    result = re.sub(r"<think>.*$", "", result, flags=re.DOTALL)
    return result.strip()

def extract_integers(text: str) -> list[int]:
    split_parts = re.split(r"\D+", text)
    return [int(part) for part in split_parts if part.isdigit()]

def calculate_score(actual, estimate) -> float:
    try:
        actual_str = str(int(actual))
        estimate_str = str(int(estimate))
    except (ValueError, OverflowError):
        return 0.0
    max_length = max(len(actual_str), len(estimate_str))
    actual_padded = actual_str.ljust(max_length, "0")
    estimate_padded = estimate_str.ljust(max_length, "0")
    padding_size = max_length - min(len(actual_str), len(estimate_str))
    actual_int = int(actual_padded)
    estimate_int = int(estimate_padded)
    if max(actual_int, estimate_int) == 0:
        return 0.0
    relative_diff = abs(actual_int - estimate_int) / max(actual_int, estimate_int)
    correction_factor = 1 - (padding_size / max_length)
    score = (1 - relative_diff) * correction_factor
    return max(0.0, min(score, 1.0))

def score_math_16(text: str, answer) -> float:
    stripped = strip_thinking(text)
    integers = extract_integers(stripped)
    if not integers:
        integers = extract_integers(text)
    if not integers:
        return 0.0
    return max(calculate_score(answer, i) for i in integers)

def _extract_scores_from_section(text: str) -> Optional[list[float]]:
    score_pattern = r"(?:\d\.\s*)?[A-Za-z]+:\s*(\d+(?:\.\d+)?)"
    matches = re.findall(score_pattern, text)
    valid_scores = []
    for m in matches[:4]:
        try:
            val = float(m)
            if 0 <= val <= 10:
                valid_scores.append(val)
        except ValueError:
            continue
    if len(valid_scores) >= 3:
        while len(valid_scores) < 4:
            valid_scores.append(5.0)
        return valid_scores
    return None

def extract_emotion_scores(text: str) -> tuple[Optional[dict], float]:
    default_scores = {k: 5.0 for k in EMOTION_KEYS}
    first_pass_scores = None
    revised_scores = None

    revised_match = re.search(r"Revised scores:", text, re.IGNORECASE)
    if revised_match:
        after_revised = text[revised_match.end():]
        end_match = re.search(r"\[End of answer\]", after_revised, re.IGNORECASE)
        if end_match:
            after_revised = after_revised[: end_match.start()]
        revised_scores = _extract_scores_from_section(after_revised)

    first_pass_match = re.search(r"First pass scores:", text, re.IGNORECASE)
    if first_pass_match:
        after_first = text[first_pass_match.end():]
        critique_match = re.search(r"Critique:", after_first, re.IGNORECASE)
        if critique_match:
            after_first = after_first[: critique_match.start()]
        first_pass_scores = _extract_scores_from_section(after_first)

    if revised_scores and first_pass_scores:
        combined = [0.5 * fp + 0.5 * rv for fp, rv in zip(first_pass_scores, revised_scores)]
        return dict(zip(EMOTION_KEYS, combined)), 1.0
    if revised_scores:
        return dict(zip(EMOTION_KEYS, revised_scores)), 0.9
    if first_pass_scores:
        return dict(zip(EMOTION_KEYS, first_pass_scores)), 0.8

    all_numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    valid_in_range = []
    for n in all_numbers:
        try:
            val = float(n)
            if 0 <= val <= 10:
                valid_in_range.append(val)
        except ValueError:
            continue
    if len(valid_in_range) >= 4:
        return dict(zip(EMOTION_KEYS, valid_in_range[:4])), 0.5
    if len(valid_in_range) >= 1:
        scores = dict(default_scores)
        for i, val in enumerate(valid_in_range[:4]):
            scores[EMOTION_KEYS[i]] = val
        return scores, len(valid_in_range) / 8.0
    return default_scores, 0.0

def calculate_eq_score(predicted: Optional[dict], reference: dict, confidence: float = 1.0) -> float:
    if predicted is None:
        return 0.5
    pred_scores = [predicted.get(k, 5.0) for k in EMOTION_KEYS]
    ref_scores = [reference.get(k, 5.0) for k in EMOTION_KEYS]
    total_diff = sum(abs(p - r) for p, r in zip(pred_scores, ref_scores))
    max_possible_diff = 10 * 4
    raw_score = max(0.0, 1.0 - (total_diff / max_possible_diff))
    return confidence * raw_score + (1 - confidence) * 0.5

def score_eq_16(text: str, reference: dict) -> float:
    stripped = strip_thinking(text)
    predicted, confidence = extract_emotion_scores(stripped)
    return calculate_eq_score(predicted, reference, confidence)

def load_math_16() -> list[EvalSample]:
    with (DATASETS_DIR / "math_16.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = []
    for qid, row in raw.items():
        prompt = math_16_prompt(row["question"])
        samples.append(EvalSample(
            qid=qid, task="math_16",
            messages=[{"role": RAW_TEXT_ROLE, "content": prompt}],
            meta={"answer": row["answer"]},
        ))
    return samples

def load_eq_16() -> list[EvalSample]:
    with (DATASETS_DIR / "eq_16.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = []
    for qid, row in raw.items():
        reference = row.get("reference_answer", row.get("reference_answer_fullscale", {}))
        samples.append(EvalSample(
            qid=qid, task="eq_16",
            messages=[{"role": RAW_TEXT_ROLE, "content": row["prompt"]}],
            meta={"reference": reference},
        ))
    return samples

def _wants_qwen_think_seed(model_path: Optional[str]) -> bool:
    """Append the empty-think seed only for Qwen. Gemma treats that string as text."""
    if not model_path:
        return True
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        return True
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_type = str(cfg.get("model_type", "")).lower()
    archs = " ".join(cfg.get("architectures") or []).lower()
    if "gemma" in model_type or "gemma" in archs:
        return False
    return True

def _read_num_layers_from_model_path(model_path: str) -> int:
    """Read the decoder layer count from a local model dir's config.json,
    preferring the multimodal `text_config.num_hidden_layers` (Qwen3.x
    vision-language style) then top-level `num_hidden_layers`. Used by
    `--model-path` so a different-size base LM (e.g. Qwen3.5-4B, 32 layers)
    can be scanned with the same runtime config that was written for the
    64-layer Qwen3.5-27B.
    """
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"--model-path {model_path}: config.json not found")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    tc = cfg.get("text_config") or {}
    n = tc.get("num_hidden_layers") or cfg.get("num_hidden_layers")
    if n is None:
        raise SystemExit(
            f"--model-path {model_path}: cannot determine num_hidden_layers "
            "(no text_config.num_hidden_layers / num_hidden_layers in config.json)"
        )
    return int(n)

def build_scan_anchors(num_layers: int, k_eff: float) -> tuple[list[Anchor], dict[int, object]]:
    anchors: list[Anchor] = [Anchor(id=0, type="BASE", layer_span=None, analytic_cost=1.0)]
    exec_key: dict[int, object] = {0: "BASE"}
    next_id = 1

    pairs = [(i, j) for j in range(1, num_layers + 1) for i in range(j)]

    for i, j in pairs:
        span = (i, j)
        cost = rep_cost(span, REP_R, num_layers)
        anchors.append(Anchor(id=next_id, type="REPEAT", layer_span=span, r=REP_R, analytic_cost=cost))
        exec_key[next_id] = next_id
        next_id += 1

    for i, j in pairs:
        if j >= num_layers:
            continue
        span = (i, j)
        cost = branch_cost(span, k_eff, num_layers)
        anchors.append(Anchor(id=next_id, type="BRANCH", layer_span=span, K=BR_K, K_eff=k_eff, analytic_cost=cost))
        exec_key[next_id] = next_id
        next_id += 1

    return anchors, exec_key

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
    parser.add_argument("--config", required=True, help="Path to runtime.json (model.num_layers, action_space.branch_cost_table/lambda_cost)")
    parser.add_argument("--output-dir", required=True, help="Base output dir; writes to <output-dir>/shard{i}of{n}/scan.jsonl")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fsync-every", type=int, default=200)
    parser.add_argument(
        "--max-span-width", type=int, default=None,
        help=(
            "Optional temporary cap on span width (j - i) actually executed "
            "this run, to save compute; anchors with span_len > this value "
            "are SKIPPED (not executed, not written) unless already present "
            "in the output file from a prior (uncapped) run. Anchor "
            "id/count/exec_key assignment is unaffected (still built over "
            "the full (i,j) grid), so this is safe to toggle across runs "
            "without corrupting resume-by-(context_id, anchor_id). Does not "
            "affect BASE."
        ),
    )
    parser.add_argument(
        "--model-path", default=None,
        help=(
            "Optional override of the frozen base-LM directory (defaults to "
            "the SELFMORPH_MODEL_PATH env var). When set, num_layers is read "
            "from this model's own config.json (text_config.num_hidden_layers "
            "preferred), NOT from --config's model.num_layers, so a "
            "different-size model (e.g. Qwen3.5-4B with 32 layers) can be "
            "scanned using the same action_space/lambda_cost settings. Anchor "
            "ids/counts are recomputed from the actual num_layers; use a "
            "separate --output-dir per model."
        ),
    )
    parser.add_argument(
        "--device-map", default=None,
        help=(
            "Optional accelerate device_map for loading the frozen base LM "
            "(e.g. 'auto'); required for models too large for one GPU "
            "(Qwen3.5-122B-A10B ~250 GB bf16). Default: None = single-GPU "
            ".to(--device). When used with multiple shards, scope each "
            "process with its own CUDA_VISIBLE_DEVICES."
        ),
    )
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
    append_think_seed = _wants_qwen_think_seed(args.model_path)
    if not append_think_seed:
        print(
            f"[shard {args.shard_id}/{args.num_shards}] model family has no forced-<think> "
            f"mechanism (detected from config.json); NOT appending QWEN_EMPTY_THINK_SEED to "
            f"RAW_TEXT prompts for this run.",
            flush=True,
        )

    model, tokenizer = load_frozen_model(device=args.device, device_map=args.device_map)
    anchors, exec_key = build_scan_anchors(num_layers, k_eff)

    samples: list[tuple[str, EvalSample]] = (
        [("math_16", s) for s in load_math_16()] + [("eq_16", s) for s in load_eq_16()]
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
            prompt_text = render_prompt(tokenizer, sample.messages, append_think_seed=append_think_seed)
            enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            max_new = MATH_MAX_NEW if ds == "math_16" else EQ_MAX_NEW

            result_cache: dict[object, dict] = {}

            for anchor in anchors:
                if (context_id, anchor.id) in done:
                    continue
                if (
                    args.max_span_width is not None
                    and anchor.layer_span is not None
                    and (anchor.layer_span[1] - anchor.layer_span[0]) > args.max_span_width
                ):
                    continue
                key = exec_key[anchor.id]
                result = result_cache.get(key)
                if result is None:
                    result = execute_anchor(
                        model, tokenizer, enc, anchor, num_layers=num_layers,
                        action_cfg=config["action_space"], max_new_tokens=max_new,
                    )
                    result_cache[key] = result
                if ds == "math_16":
                    score = score_math_16(result["text"], sample.meta["answer"])
                else:
                    score = score_eq_16(result["text"], sample.meta["reference"])
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
