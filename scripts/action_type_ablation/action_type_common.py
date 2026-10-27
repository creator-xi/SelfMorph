#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from selfmorph.action_space import build_action_space  # noqa: E402

from ablation_common import (  # noqa: E402
    anchors_by_key, anchor_key, load_config, load_payload, mean_train_reward, model_specs,
    run_trials as _run_trials, train_worker as _stock_train_worker,
)

POOLS = ["repeat13", "branch13", "full25"]
EXP_CKPT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "action_type_ablation"

POOL_TYPES: dict[str, tuple[str, ...]] = {
    "repeat13": ("BASE", "REPEAT"),
    "branch13": ("BASE", "BRANCH"),
    "full25": ("BASE", "REPEAT", "BRANCH"),
}

POOL_SIZES: dict[str, int] = {"repeat13": 13, "branch13": 13, "full25": 25}

def install() -> dict:
    """Confirm that `selfmorph.selector` provides type-conditioned head/loss."""
    import selfmorph.selector as MS
    import selfmorph.training as TR

    return {
        "selfmorph.selector.build_selector": {"source": MS.build_selector.__module__},
        "selfmorph.training.selector_loss": {"source": TR.selector_loss.__module__},
    }

def build_pool_config(base_config: dict, pool: str) -> tuple[dict, object, list[int]]:
    """Build the sub-config whose `action_space` enumerates exactly `pool`.

    Returns `(sub_config, sub_anchor_set, col_ids)` where `col_ids[i]` is the
    column of pool anchor `i` inside the FULL 25-anchor payload (used to subset
    `scores/costs/rewards/utilities` with
    `ablation_common.slice_payload_anchors`).

    Mirrors `ablation_common.build_tier_config`'s approach (rewrite the action
    space, then map anchor identities) but varies the FAMILY rather than the
    span/intensity subset.
    """
    if pool not in POOL_TYPES:
        raise ValueError(f"unknown pool {pool!r}, expected one of {POOLS}")

    full_anchors = build_action_space(base_config)
    if pool == "full25":

        return copy.deepcopy(base_config), full_anchors, list(range(len(full_anchors)))

    ac = copy.deepcopy(base_config["action_space"])
    if not ac.get("include_base", True):
        raise ValueError("include_base must stay true: BASE is the common base action of every pool")
    if pool == "repeat13":
        ac["branch_spans"] = []
    elif pool == "branch13":
        ac["repeat_spans"] = []

    sub_config = copy.deepcopy(base_config)
    sub_config["action_space"] = ac
    sub_anchors = build_action_space(sub_config)

    got_types = tuple(sorted({a.type for a in sub_anchors}))
    expected_types = tuple(sorted(POOL_TYPES[pool]))
    if got_types != expected_types:
        raise ValueError(f"pool {pool}: enumerated types {got_types} != expected {expected_types}")
    if len(sub_anchors) != POOL_SIZES[pool]:
        raise ValueError(f"pool {pool}: {len(sub_anchors)} anchors != expected {POOL_SIZES[pool]}")
    if sub_anchors.base_id != 0:
        raise ValueError(f"pool {pool}: BASE must keep global id 0, got {sub_anchors.base_id}")

    by_key = anchors_by_key(full_anchors)
    col_ids = []
    for a in sub_anchors:
        key = anchor_key(a)
        if key not in by_key:
            raise ValueError(f"pool {pool}: anchor {key} is not part of the full pool")
        col_ids.append(by_key[key].id)
    if sorted(col_ids) != sorted(set(col_ids)):
        raise ValueError(f"pool {pool}: duplicate columns {col_ids}")
    return sub_config, sub_anchors, col_ids

def pool_report(base_config: dict, pool: str) -> dict:
    """Description of one action pool, stored in plan.json."""
    _, sub_anchors, col_ids = build_pool_config(base_config, pool)
    ac = build_action_space(base_config)
    return {
        "pool": pool,
        "types": list(POOL_TYPES[pool]),
        "num_anchors": len(sub_anchors),
        "base_id": sub_anchors.base_id,
        "payload_columns": col_ids,
        "rep_spans": [list(a.layer_span) for a in sub_anchors if a.type == "REPEAT"][:4],
        "rep_r_values": sorted({a.r for a in sub_anchors if a.type == "REPEAT"}),
        "br_spans": [list(a.layer_span) for a in sub_anchors if a.type == "BRANCH"][:4],
        "br_K_values": sorted({a.K for a in sub_anchors if a.type == "BRANCH"}),
        "full_pool_num_anchors": len(ac),
    }

def assert_payload_columns(payload: dict, base_config: dict, atol: float = 1e-5) -> float:
    """Fail-loud check that the persisted payload's anchor axis matches
    `build_action_space(base_config)` column-for-column (analytic costs are
    deterministic and differ per anchor, so they are a perfect identity probe).
    Returns the max abs difference for the record."""
    anchors = build_action_space(base_config)
    if int(payload["num_anchors"]) != len(anchors):
        raise ValueError(f"payload has {payload['num_anchors']} anchors, config enumerates {len(anchors)}")
    costs_ref = torch.tensor([a.analytic_cost for a in anchors], dtype=torch.float32)
    diff = float((payload["costs"][0] - costs_ref).abs().max())
    if diff > atol:
        raise ValueError(
            f"payload anchor columns do not match build_action_space(config): max analytic-cost diff {diff}"
        )

    if not all(a.id == i for i, a in enumerate(anchors)):
        raise ValueError("build_action_space does not assign sequential ids")
    return diff

def build_trials(spec, base_config: dict, pools: list[str]) -> list[dict]:
    """One trial per pool. The only difference between pools is `action_space`."""
    trials: list[dict] = []
    for pool in pools:
        sub_cfg, sub_anchors, col_ids = build_pool_config(base_config, pool)
        active_types = sorted({a.type for a in sub_anchors})
        trials.append({
            "trial_id": pool,
            "model": spec.key,
            "pool": pool,
            "num_anchors": len(sub_anchors),
            "anchor_types": active_types,
            "selector_config": sub_cfg["selector"],
            "action_space": sub_cfg["action_space"],
            "payload_spec": {"kind": "anchor_subset", "col_ids": col_ids},
        })
    return trials

def build_plan(spec, base_config: dict, pools: list[str], trials: list[dict]) -> dict:
    return {
        "schema_version": "selfmorph-action-type-ablation-plan-1",
        "purpose": "13-action REPEAT pool, 13-action BRANCH pool, and the full 25-action pool",
        "model": spec.key,
        "label": spec.label,
        "config": spec.config,
        "train_dev_pt": spec.train_dev_pt,
        "selector_checkpoint": spec.checkpoint,
        "pools": pools,
        "num_trials": len(trials),
        "payload_column_check_max_abs_cost_diff": assert_payload_columns(load_payload(spec), base_config),
        "head_installation": "action-type-conditioned selector head",
        "pools_detail": {p: pool_report(base_config, p) for p in pools},
        "trials": [{k: v for k, v in t.items() if k not in ("selector_config", "action_space")} for t in trials],
    }

def train_worker_action_type(worker_trials: list[dict], base_config: dict, train_dev_pt: str, output_dir: str,
                        device: str, result_queue) -> None:
    """Child-process entry point: bind the ablation head/loss, then run the
    stock `ablation_common.train_worker` (whose `train_selector` call then picks
    up the patched module attributes).

    This thin wrapper exists because `multiprocessing` spawn re-imports the
    target module fresh in the child, so a patch made in the parent process
    would NOT propagate -- the install must happen inside the child. The
    parent therefore rebinds `ablation_common.train_worker` to this function
    right before calling `ablation_common.run_trials` (which resolves its
    spawn target through a module-global lookup), reusing that dispatcher's
    plan and result files.
    """
    install()
    if _stock_train_worker is train_worker_action_type:  # pragma: no cover - defensive
        raise RuntimeError("train_worker_action_type would recurse into itself")
    _stock_train_worker(worker_trials, base_config, train_dev_pt, output_dir, device, result_queue)

def run_trials(trials: list[dict], base_config: dict, train_dev_pt: Path, output_dir: Path,
               gpus: list[str], plan: dict | None = None) -> list[dict]:
    """`ablation_common.run_trials` with the ablation worker installed."""
    import ablation_common as MEC

    MEC.train_worker = train_worker_action_type
    return _run_trials(trials, base_config, train_dev_pt, output_dir, gpus, plan)

def selftest_pool_construction(models: list[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    for spec in model_specs(models):
        base_config = load_config(spec)
        payload = load_payload(spec)
        diff = assert_payload_columns(payload, base_config)
        for pool in POOLS:
            sub_cfg, sub_anchors, col_ids = build_pool_config(base_config, pool)
            rows.append({
                "model": spec.key, "pool": pool, "num_anchors": len(sub_anchors),
                "types": sorted({a.type for a in sub_anchors}), "base_id": sub_anchors.base_id,
                "columns": col_ids, "payload_cost_check_max_abs_diff": diff,
                "include_base": sub_cfg["action_space"].get("include_base", True),
                "morphable_layer_range": list(sub_cfg["action_space"]["morphable_layer_range"]),
            })
    return rows

if __name__ == "__main__":
    print(json.dumps(selftest_pool_construction(), indent=2))
