"""Sharded offline oracle collection with exact resumption on `(context_id, anchor_id)`."""

from __future__ import annotations

import json
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from selfmorph.action_space import AnchorSet, build_action_space
from selfmorph.benchmarks import (
    EvalSample, load_benchmark, model_wants_qwen_think_seed,
    oracle_max_new_tokens, render_prompt, score_sample,
)
from selfmorph.executor import execute_anchor
from selfmorph.split_manifest import dataset_source_splits, lookup_split
from selfmorph.trajectory import collect_trajectory, load_trajectory_spec, stack_trajectory

def _manifest_samples(ds: str) -> list[EvalSample]:
    src_splits = dataset_source_splits(ds)
    if not src_splits:

        return load_benchmark(ds)
    out: list[EvalSample] = []
    for src_split in sorted(src_splits):
        try:
            rows = load_benchmark(ds, split=src_split)
        except TypeError:

            rows = load_benchmark(ds)
        for s in rows:
            if lookup_split(ds, src_split, s.qid) is not None:
                out.append(s)
    return out

def _read_ppid(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None

def assert_no_duplicate_process(script_name: str, shard_id: int) -> None:
    import subprocess

    try:
        pids = subprocess.run(["pgrep", "-f", script_name], capture_output=True, text=True).stdout.split()
    except FileNotFoundError:
        return

    my_pid = os.getpid()
    ancestors = {my_pid}
    p = my_pid
    while p > 1:
        parent = _read_ppid(p)
        if parent is None or parent in ancestors:
            break
        ancestors.add(parent)
        p = parent

    for pid_s in pids:
        pid_i = int(pid_s)
        if pid_i in ancestors:
            continue
        try:
            cmdline = open(f"/proc/{pid_i}/cmdline").read().replace("\0", " ").strip()
        except FileNotFoundError:
            continue
        tokens = cmdline.split()
        if not tokens:
            continue
        exe = tokens[0].split("/")[-1]
        if "python" not in exe:
            continue
        for i, t in enumerate(tokens):
            if t == "--shard-id" and i + 1 < len(tokens) and tokens[i + 1] == str(shard_id):
                print(f"[ABORT] another {script_name} python process with --shard-id {shard_id} "
                      f"is already running (pid={pid_i}): {cmdline[:200]}", file=sys.stderr)
                raise SystemExit(1)

@dataclass
class OracleRecord:
    context_id: str
    dataset: str
    anchor_id: int
    score: float
    cost: float
    reward: float
    text: str
    latency_s: float

def _completed_keys(output_path: Path) -> set[tuple[str, int]]:
    """Resume: scan an existing shard jsonl and build the set of
    (context_id, anchor_id) pairs already recorded.
    """
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

def _completed_trajectory_contexts(trajectory_index_path: Path) -> set[str]:
    done: set[str] = set()
    if not trajectory_index_path.is_file():
        return done
    with trajectory_index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["context_id"])
            except Exception:
                continue
    return done

def collect_oracle_shard(
    *,
    dataset_names: list[str],
    output_dir: Path,
    shard_id: int,
    num_shards: int,
    config: dict,
    model,
    tokenizer,
    fsync_every: int = 1000,
    collect_mode: str = "full",
    context_ids_file: Optional[Path] = None,
) -> None:
    if collect_mode not in ("full", "br_only", "rep_only"):
        raise ValueError(f"Unknown collect_mode {collect_mode!r}")
    shard_dir = output_dir / f"shard{shard_id}of{num_shards}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = shard_dir / "trajectories"
    if collect_mode == "full":
        trajectory_dir.mkdir(parents=True, exist_ok=True)
    out_path = shard_dir / "oracle.jsonl"
    trajectory_index_path = shard_dir / "trajectory_index.jsonl"
    success_path = shard_dir / "_SUCCESS.json"

    if success_path.is_file():
        print(f"[shard {shard_id}] already complete: {success_path}", flush=True)
        return

    done = _completed_keys(out_path)
    done_trajectories = _completed_trajectory_contexts(trajectory_index_path)

    anchors: AnchorSet = build_action_space(config)
    lambda_cost = config["action_space"]["lambda_cost"]
    action_cfg = config["action_space"]
    num_layers = config["model"]["num_layers"]

    append_think_seed = model_wants_qwen_think_seed(model)

    if collect_mode == "br_only":
        target_anchors = [a for a in anchors if a.type == "BRANCH"]
    elif collect_mode == "rep_only":
        target_anchors = [a for a in anchors if a.type == "REPEAT"]
    else:
        target_anchors = list(anchors)
    expected_anchor_ids = {a.id for a in target_anchors}

    traj_spec = load_trajectory_spec(config) if collect_mode == "full" else None
    layer_ids = traj_spec.layer_ids() if traj_spec is not None else None

    samples: list[tuple[str, EvalSample]] = []
    for ds in dataset_names:
        for s in _manifest_samples(ds):
            samples.append((ds, s))

    my_samples = [
        (ds, s) for ds, s in samples
        if int(hashlib.sha256(f"{ds}::{s.qid}".encode()).hexdigest(), 16) % num_shards == shard_id
    ]

    if context_ids_file is not None:
        whitelist = {
            line.strip() for line in context_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        my_samples = [(ds, s) for ds, s in my_samples if f"{ds}::{s.qid}" in whitelist]

    oracle_f = open(out_path, "a", buffering=1, encoding="utf-8")
    traj_index_f = open(trajectory_index_path, "a", buffering=1, encoding="utf-8") if collect_mode == "full" else None
    n_written = 0
    try:
        for ds, sample in my_samples:
            context_id = f"{ds}::{sample.qid}"
            prompt_text = render_prompt(tokenizer, sample.messages, append_think_seed=append_think_seed)
            enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            max_new = oracle_max_new_tokens(ds, config.get("manifest_default_max_new", 32))

            if collect_mode == "full":
                base_anchor = next(a for a in anchors if a.type == "BASE")

                need_score = (context_id, base_anchor.id) not in done
                need_traj = context_id not in done_trajectories
                if need_score or need_traj:
                    hiddens = collect_trajectory(model, enc, layer_ids=layer_ids)
                    traj = stack_trajectory(hiddens, layer_ids).to(torch.float16)

                    if need_score:
                        base_result = execute_anchor(
                            model, tokenizer, enc, base_anchor, num_layers=num_layers,
                            action_cfg=action_cfg, max_new_tokens=max_new,
                        )
                        base_score = score_sample(base_result["text"], sample, ds)
                        rec = OracleRecord(
                            context_id=context_id, dataset=ds, anchor_id=base_anchor.id,
                            score=base_score, cost=base_anchor.analytic_cost,
                            reward=base_score - lambda_cost * (base_anchor.analytic_cost - 1.0),
                            text=base_result["text"], latency_s=base_result["latency_s"],
                        )
                        oracle_f.write(json.dumps(rec.__dict__) + "\n")
                        oracle_f.flush()
                        n_written += 1

                    if need_traj:

                        traj_path = trajectory_dir / f"{context_id}.pt"
                        traj_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(traj, traj_path)
                        traj_index_f.write(json.dumps({
                            "context_id": context_id, "dataset": ds,
                            "trajectory_path": str(traj_path.relative_to(shard_dir)),
                            "shape": list(traj.shape), "dtype": str(traj.dtype).replace("torch.", ""),
                        }) + "\n")
                        traj_index_f.flush()

            for anchor in target_anchors:
                if anchor.type == "BASE":
                    continue
                if (context_id, anchor.id) in done:
                    continue
                result = execute_anchor(
                    model, tokenizer, enc, anchor, num_layers=num_layers,
                    action_cfg=action_cfg, max_new_tokens=max_new,
                )
                score = score_sample(result["text"], sample, ds)
                rec = OracleRecord(
                    context_id=context_id, dataset=ds, anchor_id=anchor.id,
                    score=score, cost=anchor.analytic_cost,
                    reward=score - lambda_cost * (anchor.analytic_cost - 1.0),
                    text=result["text"], latency_s=result["latency_s"],
                )
                oracle_f.write(json.dumps(rec.__dict__) + "\n")
                oracle_f.flush()
                n_written += 1
                if n_written % fsync_every == 0:
                    os.fsync(oracle_f.fileno())
    finally:
        os.fsync(oracle_f.fileno())
        oracle_f.close()
        if traj_index_f is not None:
            os.fsync(traj_index_f.fileno())
            traj_index_f.close()

    final_done = _completed_keys(out_path)
    per_context_anchor_sets: dict[str, set[int]] = {}
    for (cid, aid) in final_done:
        per_context_anchor_sets.setdefault(cid, set()).add(aid)
    expected_contexts = {f"{ds}::{s.qid}" for ds, s in my_samples}
    if collect_mode == "full":
        final_traj = _completed_trajectory_contexts(trajectory_index_path)
        complete = expected_contexts <= final_traj and all(
            per_context_anchor_sets.get(cid, set()) == expected_anchor_ids
            for cid in expected_contexts
        )
    else:
        complete = all(
            per_context_anchor_sets.get(cid, set()) == expected_anchor_ids
            for cid in expected_contexts
        )
    if complete:
        with success_path.open("w", encoding="utf-8") as f:
            json.dump({
                "shard_id": shard_id, "num_shards": num_shards, "collect_mode": collect_mode,
                "contexts": len(expected_contexts),
                "oracle_rows": len(expected_contexts) * len(expected_anchor_ids),
            }, f, indent=2)
