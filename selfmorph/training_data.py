"""Assemble oracle records into cached trajectory and utility tensors."""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from selfmorph.action_space import AnchorSet, build_action_space
from selfmorph.benchmarks import paper_domain
from selfmorph.settings import load_runtime_config
from selfmorph.split_manifest import dataset_source_splits, lookup_split

SCHEMA_VERSION = "selfmorph-train-dev"

@dataclass
class ContextRecord:
    context_id: str
    dataset: str
    domain: str
    scores: list[float]
    costs: list[float]
    rewards: list[float]
    utilities: list[float]
    trajectory: list[list[float]]
    split: str

def _load_oracle_records(oracle_glob: str) -> tuple[dict[str, dict[int, dict]], dict[str, Path]]:
    by_context: dict[str, dict[int, dict]] = {}
    traj_paths: dict[str, Path] = {}
    for path_str in sorted(glob.glob(oracle_glob, recursive=True)):
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ctx = by_context.setdefault(rec["context_id"], {})
                ctx[int(rec["anchor_id"])] = rec

        traj_index_path = path.parent / "trajectory_index.jsonl"
        if traj_index_path.is_file():
            with traj_index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    idx_rec = json.loads(line)
                    traj_paths[idx_rec["context_id"]] = path.parent / idx_rec["trajectory_path"]
    return by_context, traj_paths

def _assign_split(dataset: str, qid: str) -> str | None:
    src_splits = dataset_source_splits(dataset)
    if not src_splits:
        return None
    for src_split in src_splits:
        split = lookup_split(dataset, src_split, qid)
        if split is not None:
            return split
    return None

def build_training_data(
    *,
    oracle_glob: str,
    config: dict | None = None,
) -> tuple[list[ContextRecord], AnchorSet]:
    config = config or load_runtime_config()
    anchors = build_action_space(config)
    lambda_cost = config["action_space"]["lambda_cost"]
    include_math = config["domains"]["include_math_in_oracle"]
    math_datasets = set(config["domains"]["id_math_datasets"])

    by_context, traj_paths = _load_oracle_records(oracle_glob)
    records: list[ContextRecord] = []
    n_total, n_incomplete, n_unmapped, n_missing_traj = 0, 0, 0, 0

    for context_id, anchor_records in by_context.items():
        n_total += 1
        dataset = next(iter(anchor_records.values()))["dataset"]
        if not include_math and dataset in math_datasets:
            continue
        if len(anchor_records) < len(anchors):
            n_incomplete += 1
            continue

        qid = context_id.split("::", 1)[1]
        split = _assign_split(dataset, qid)
        if split is None:
            n_unmapped += 1
            continue

        traj_path = traj_paths.get(context_id)
        if traj_path is None or not traj_path.is_file():
            n_missing_traj += 1
            continue
        trajectory = torch.load(traj_path, map_location="cpu", weights_only=True).float().tolist()

        scores = [0.0] * len(anchors)
        costs = [1.0] * len(anchors)
        for a in anchors:
            rec = anchor_records[a.id]
            scores[a.id] = float(rec["score"])
            costs[a.id] = float(rec["cost"])

        rewards = [s - lambda_cost * (c - 1.0) for s, c in zip(scores, costs)]
        base_id = anchors.base_id
        r_noop = rewards[base_id]
        utilities = [r - r_noop for r in rewards]

        records.append(ContextRecord(
            context_id=context_id, dataset=dataset, domain=paper_domain(dataset, config),
            scores=scores, costs=costs, rewards=rewards, utilities=utilities,
            trajectory=trajectory, split=split,
        ))

    if n_incomplete:
        raise ValueError(f"{n_incomplete} contexts have incomplete oracle (< {len(anchors)} anchors).")
    if n_missing_traj:
        raise ValueError(f"{n_missing_traj} contexts are missing trajectory .pt files.")
    if n_unmapped:
        print(f"[build_training_data] WARNING: {n_unmapped}/{n_total} contexts not in split manifest; dropped.")

    return records, anchors

def softmax_p_star(utilities: list[float], temperature: float) -> torch.Tensor:
    u = torch.tensor(utilities, dtype=torch.float32)
    return torch.softmax(u / temperature, dim=-1)

def save_training_data(
    records: list[ContextRecord],
    anchors: AnchorSet,
    *,
    out_pt_path: Path,
    out_contexts_path: Path,
    include_math_in_oracle: bool,
) -> None:
    scores = torch.tensor([r.scores for r in records], dtype=torch.float32)
    costs = torch.tensor([r.costs for r in records], dtype=torch.float32)
    rewards = torch.tensor([r.rewards for r in records], dtype=torch.float32)
    utilities = torch.tensor([r.utilities for r in records], dtype=torch.float32)
    trajectory = torch.tensor([r.trajectory for r in records], dtype=torch.float32)
    splits = [r.split for r in records]
    domains = [r.domain for r in records]
    datasets = [r.dataset for r in records]
    context_ids = [r.context_id for r in records]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "scores": scores, "costs": costs, "rewards": rewards, "utilities": utilities,
        "trajectory": trajectory, "splits": splits, "domains": domains, "datasets": datasets,
        "context_ids": context_ids, "num_anchors": len(anchors),
        "include_math_in_oracle": include_math_in_oracle,
    }
    out_pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt_path)

    with out_contexts_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "context_id": r.context_id, "dataset": r.dataset, "domain": r.domain, "split": r.split,
            }) + "\n")

def merge_dev_splits(payload: dict, *, into: str = "dev_tune", merge_from: tuple[str, ...] = ("dev_select",)) -> dict:
    """Merge validation sub-pools (`dev_tune` + `dev_select`) into a single validation split."""
    merged = dict(payload)
    merge_set = set(merge_from)
    merged["splits"] = [into if s in merge_set else s for s in payload["splits"]]
    return merged
