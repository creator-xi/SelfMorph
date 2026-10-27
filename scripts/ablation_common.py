#!/usr/bin/env python3
"""Shared helpers for the ablation scripts."""
from __future__ import annotations

import copy
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from selfmorph.action_space import Anchor, AnchorSet, build_action_space  # noqa: E402
from selfmorph.selection import benchmark_unit_for  # noqa: E402

LAMBDAS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]
ANCHOR_TIERS = ["actions3", "actions9", "actions25"]
EXP1_CKPT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "anchor_count_ablation"
LAMBDA_SWEEP_CKPT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "lambda_sweep"
FIG_DIR = REPO_ROOT / "figures" / "lambda_pareto"

@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    config: str
    train_dev_pt: str
    checkpoint: str

MODELS = (
    ModelSpec(
        key="qwen35_27b",
        label="Qwen3.5-27B",
        config="config/runtime.json",
        train_dev_pt="artifacts/training_data/qwen35_27b/train_dev.pt",
        checkpoint="artifacts/checkpoints/qwen35_27b/best_checkpoint.pt",
    ),
    ModelSpec(
        key="qwen35_4b",
        label="Qwen3.5-4B",
        config="config/runtime_qwen35_4b.json",
        train_dev_pt="artifacts/training_data/qwen35_4b/train_dev.pt",
        checkpoint="artifacts/checkpoints/qwen35_4b/best_checkpoint.pt",
    ),
    ModelSpec(
        key="gemma4_12b",
        label="Gemma4-12B (base)",
        config="config/runtime_gemma4_12b.json",
        train_dev_pt="artifacts/training_data/gemma4_12b/train_dev.pt",
        checkpoint="artifacts/checkpoints/gemma4_12b/best_checkpoint.pt",
    ),
)

def model_specs(keys: list[str] | None = None) -> list[ModelSpec]:
    if not keys:
        return list(MODELS)
    by_key = {m.key: m for m in MODELS}
    return [by_key[k] for k in keys]

def load_config(spec: ModelSpec) -> dict:
    with (REPO_ROOT / spec.config).open("r", encoding="utf-8") as f:
        return json.load(f)

def load_payload(spec: ModelSpec) -> dict:
    return torch.load(REPO_ROOT / spec.train_dev_pt, map_location="cpu", weights_only=False)

def anchor_key(a: Anchor) -> tuple:
    """Identity of an anchor independent of its (tier-dependent) id."""
    return (a.type, a.layer_span, a.r, a.K)

def anchors_by_key(anchors: AnchorSet) -> dict[tuple, Anchor]:
    return {anchor_key(a): a for a in anchors}

def mean_train_reward(payload: dict) -> torch.Tensor:
    """[num_anchors] mean reward over the TRAIN split (the criterion used by
    `final_eval.compute_fixed_anchor` and by the 3-action span choice)."""
    idx = [i for i, s in enumerate(payload["splits"]) if s == "train"]
    return payload["rewards"][torch.tensor(idx, dtype=torch.long)].mean(dim=0)

def build_tier_config(base_config: dict, tier: str, train_mean_reward_vec: torch.Tensor) -> tuple[dict, AnchorSet, list[int]]:
    """Build the sub-config whose `action_space` enumerates exactly `tier`.

    Returns `(config_with_subset_action_space, subset_anchor_set, col_ids)`
    where `col_ids[i]` is the column index of subset anchor `i` inside the
    FULL 25-anchor payload (used to subset the score/cost/reward tensors).

    Nesting: A9 lists the 3-action-chosen REPEAT span first and the 3-action-chosen BR span
    first, so the 3-action pool's anchor set is literally the first three anchors of A9.
    """
    full_anchors = build_action_space(base_config)
    if tier == "actions25":
        return copy.deepcopy(base_config), full_anchors, list(range(len(full_anchors)))
    if tier not in ("actions3", "actions9"):
        raise ValueError(f"unknown tier {tier!r}")

    ac = copy.deepcopy(base_config["action_space"])
    rep_spans = [tuple(s) for s in ac["repeat_spans"]]
    br_spans = [tuple(s) for s in ac["branch_spans"]]
    by_key = anchors_by_key(full_anchors)
    rep_r, br_k = 1, 2

    rep_mean = {s: float(train_mean_reward_vec[by_key[("REPEAT", s, rep_r, None)].id]) for s in rep_spans}
    br_mean = {s: float(train_mean_reward_vec[by_key[("BRANCH", s, None, br_k)].id]) for s in br_spans}
    best_rep = max(rep_spans, key=lambda s: rep_mean[s])
    best_br = max(br_spans, key=lambda s: br_mean[s])

    if tier == "actions3":
        rep_order, br_order = [best_rep], [best_br]
    else:
        rep_order = [best_rep] + [s for s in rep_spans if s != best_rep]
        br_order = [best_br] + [s for s in br_spans if s != best_br]

    ac["repeat_spans"] = [list(s) for s in rep_order]
    ac["repeat_counts"] = [rep_r]
    ac["branch_spans"] = [list(s) for s in br_order]
    ac["branch_counts"] = [br_k]

    sub_config = copy.deepcopy(base_config)
    sub_config["action_space"] = ac
    sub_anchors = build_action_space(sub_config)
    col_ids = [by_key[anchor_key(a)].id for a in sub_anchors]
    return sub_config, sub_anchors, col_ids

def tier_selection_report(base_config: dict, train_mean_reward_vec: torch.Tensor) -> dict:
    """Record which spans the nested 3-action and 9-action pools keep."""
    full = build_action_space(base_config)
    by_key = anchors_by_key(full)
    ac = base_config["action_space"]
    out: dict = {"rep_r": 1, "br_K": 2, "rep_mean_train_reward": {}, "br_mean_train_reward": {}}
    for s in [tuple(x) for x in ac["repeat_spans"]]:
        out["rep_mean_train_reward"][f"{s[0]}-{s[1]}"] = float(
            train_mean_reward_vec[by_key[("REPEAT", s, 1, None)].id]
        )
    for s in [tuple(x) for x in ac["branch_spans"]]:
        out["br_mean_train_reward"][f"{s[0]}-{s[1]}"] = float(
            train_mean_reward_vec[by_key[("BRANCH", s, None, 2)].id]
        )
    _, _, _ = build_tier_config(base_config, "actions3", train_mean_reward_vec)
    a3_cfg = build_tier_config(base_config, "actions3", train_mean_reward_vec)[0]
    out["actions3_repeat_span"] = list(a3_cfg["action_space"]["repeat_spans"][0])
    out["actions3_branch_span"] = list(a3_cfg["action_space"]["branch_spans"][0])
    out["actions9_repeat_spans"] = [list(x) for x in build_tier_config(base_config, "actions9", train_mean_reward_vec)[0]["action_space"]["repeat_spans"]]
    out["actions9_branch_spans"] = [list(x) for x in build_tier_config(base_config, "actions9", train_mean_reward_vec)[0]["action_space"]["branch_spans"]]
    return out

def anchor_descriptions(anchors: AnchorSet) -> list[dict]:
    out = []
    for a in anchors:
        if a.type == "BASE":
            out.append({"id": a.id, "type": "BASE", "span": None, "r": None, "K": None, "analytic_cost": a.analytic_cost})
        else:
            out.append({
                "id": a.id, "type": a.type, "span": list(a.layer_span), "r": a.r, "K": a.K,
                "analytic_cost": a.analytic_cost,
            })
    return out

def slice_payload_anchors(payload: dict, col_ids: list[int]) -> dict:
    """Subset the anchor axis of every per-anchor tensor to `col_ids`.

    `trajectory` / `splits` / `domains` / `datasets` / `context_ids` are
    shared by reference (they are anchor-independent), so this is cheap.
    """
    idx = torch.tensor(col_ids, dtype=torch.long)
    out = dict(payload)
    for key in ("scores", "costs", "rewards", "utilities"):
        out[key] = torch.index_select(payload[key], 1, idx)
    out["num_anchors"] = len(col_ids)
    return out

def slice_payload_trajectory(payload: dict, slot_idx: list[int]) -> dict:
    """Keep trajectory positions `slot_idx` (0-based slots, not layer ids)."""
    n_layers = int(payload["trajectory"].shape[1])
    slots = [int(x) for x in slot_idx]
    if not slots or min(slots) < 0 or max(slots) >= n_layers:
        raise ValueError(
            f"slot_idx must be positions in [0, {n_layers}) along the existing trajectory "
            f"layer axis (NOT physical layer ids); got {slots}"
        )
    out = dict(payload)
    out["trajectory"] = torch.index_select(payload["trajectory"], 1, torch.tensor(slots, dtype=torch.long))
    out["trajectory_slot_indices"] = slots
    return out

def analytic_costs_double(anchors: AnchorSet) -> torch.Tensor:
    """Exact float64 analytic costs `[num_anchors]` re-derived from the
    AnchorSet (the same Python-double formula the oracle used).  The
    persisted `costs` tensor is the float32 rounding of these values, and
    re-deriving them is what makes the lambda arithmetic exact."""
    return torch.tensor([a.analytic_cost for a in anchors], dtype=torch.float64)

def recompute_lambda_payload(payload: dict, lam: float, base_id: int = 0, *,
                             costs64: torch.Tensor | None = None,
                             stored_lambda: float | None = None) -> dict:
    """reward = score - lambda*(cost-1); utility = reward - reward[BASE].

    BOTH tensors are rebuilt so the training target (`utilities`, used for
    p_star) and the in-loop early-stopping metric (`rewards`, used by
    `training.domain_macro_metrics`) are consistent with the same lambda.

    Rounding: the arithmetic is done in float64 and only then cast to
    float32 -- exactly what `training_data.build_training_data` does
    (Python floats are doubles, then `torch.tensor(..., dtype=float32)`).
    `costs64` should be the exact costs re-derived from the AnchorSet
    (`analytic_costs_double`); using the persisted float32 `costs` instead
    shifts the reward by ~1e-9, which is enough to miss a bit-exact match
    against the released checkpoints.

    `stored_lambda`: if given and equal to `lam`, the persisted tensors are
    returned VERBATIM -- they *are*, by construction, reward(stored_lambda)
    for the production data, so this makes the lambda=production self-check
    bit-exact instead of only ~1e-8-close.
    """
    if stored_lambda is not None and abs(lam - stored_lambda) < 1e-12:
        return dict(payload)
    out = dict(payload)
    scores64 = payload["scores"].double()
    if costs64 is None:
        costs64 = payload["costs"].double()
    costs64 = costs64.reshape(1, -1)
    rewards64 = scores64 - lam * (costs64 - 1.0)
    utilities64 = rewards64 - rewards64[:, base_id:base_id + 1]
    out["rewards"] = rewards64.float()
    out["utilities"] = utilities64.float()
    return out

def lambda_payload(raw_payload: dict, anchors: AnchorSet, lam: float,
                   stored_lambda: float | None = None, base_id: int = 0) -> dict:
    """`recompute_lambda_payload` with the exact float64 costs of `anchors`."""
    return recompute_lambda_payload(
        raw_payload, lam, base_id, costs64=analytic_costs_double(anchors), stored_lambda=stored_lambda,
    )

def build_derived_payload(raw_payload: dict, spec: dict) -> dict:
    kind = spec["kind"]
    if kind == "anchor_subset":
        return slice_payload_anchors(raw_payload, spec["col_ids"])
    if kind == "traj_subset":
        return slice_payload_trajectory(raw_payload, spec["layer_idx"])
    if kind == "lambda":
        costs64 = spec.get("costs64")
        return recompute_lambda_payload(
            raw_payload, spec["lambda_cost"], spec.get("base_id", 0),
            costs64=None if costs64 is None else torch.tensor(costs64, dtype=torch.float64),
            stored_lambda=spec.get("stored_lambda"),
        )
    raise ValueError(f"unknown payload spec kind {kind!r}")

def _split_idx(payload: dict, split: str) -> list[int]:
    idx = [i for i, s in enumerate(payload["splits"]) if s == split]
    if not idx:
        raise ValueError(f"no rows for split={split!r}")
    return idx

def _gather(payload: dict, idx: list[int], actions, field: str) -> list[float]:
    """Batched gather of `payload[field][row, action]` over `idx` (tensor
    indexing, never a per-sample Python loop)."""
    idx_t = torch.tensor(idx, dtype=torch.long)
    act_t = torch.as_tensor(actions, dtype=torch.long)
    rows = torch.arange(len(idx))
    return payload[field][idx_t][rows, act_t].tolist()

def lam_prefix(lam: float) -> str:
    """Checkpoint-directory prefix used by run_lambda_sweep.py."""
    return f"lam{lam:g}"

def _mean(xs) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None

def _aggregate(scores, costs, rewards, domains, units) -> dict:
    per_domain_reward: dict[str, list[float]] = {}
    per_unit_score: dict[str, list[float]] = {}
    per_unit_reward: dict[str, list[float]] = {}
    for d, u, s, r in zip(domains, units, scores, rewards):
        per_domain_reward.setdefault(d, []).append(r)
        per_unit_score.setdefault(u, []).append(s)
        per_unit_reward.setdefault(u, []).append(r)
    return {
        "n": len(scores),
        "mean_score": _mean(scores),
        "mean_cost": _mean(costs),
        "mean_reward": _mean(rewards),
        "domain_macro_reward": _mean([_mean(v) for v in per_domain_reward.values()]),
        "by_benchmark_unit_score": {u: _mean(v) for u, v in per_unit_score.items()},
        "by_benchmark_unit_reward": {u: _mean(v) for u, v in per_unit_reward.items()},
    }

def eval_actions_on_split(payload: dict, split: str, actions) -> dict:
    """Metrics of an explicit per-row action list (length = #rows of split)."""
    idx = _split_idx(payload, split)
    if len(actions) != len(idx):
        raise ValueError(f"actions length {len(actions)} != split size {len(idx)}")
    scores = _gather(payload, idx, actions, "scores")
    costs = _gather(payload, idx, actions, "costs")
    rewards = _gather(payload, idx, actions, "rewards")
    domains = [payload["domains"][i] for i in idx]
    units = [benchmark_unit_for(payload["datasets"][i]) for i in idx]
    return _aggregate(scores, costs, rewards, domains, units)

def selector_actions(payload: dict, anchors: AnchorSet, selector, split: str = "eval",
                     device: str = "cpu") -> list[int]:
    from selfmorph.selection import _batched_selector_argmax

    idx = _split_idx(payload, split)
    return _batched_selector_argmax(selector, anchors, payload["trajectory"], idx, device)

def fixed_anchor_ids(payload: dict, anchors: AnchorSet) -> tuple[int, int]:
    """(fixed_rep_id, fixed_br_id): train mean-reward argmax within each
    family, exactly `final_eval.compute_fixed_anchor`'s semantics."""
    vec = mean_train_reward(payload)
    rep_ids = [a.id for a in anchors if a.type == "REPEAT"]
    br_ids = [a.id for a in anchors if a.type == "BRANCH"]
    rep = max(rep_ids, key=lambda i: float(vec[i]))
    br = max(br_ids, key=lambda i: float(vec[i]))
    return rep, br

def oracle_upper_bound(payload: dict, split: str = "eval") -> dict:
    """Per-sample argmax-utility oracle (an upper bound on any selector):
    for every eval row pick the anchor with the highest utility, then report
    the mean score/cost/reward of those choices."""
    idx = _split_idx(payload, split)
    idx_t = torch.tensor(idx, dtype=torch.long)
    best = payload["utilities"][idx_t].argmax(dim=-1).tolist()
    return eval_actions_on_split(payload, split, best)

def load_trial_selector(ckpt_dir: Path, prefix: str, anchors: AnchorSet, hidden_size: int,
                        num_traj_layers: int):
    """Load `<ckpt_dir>/<prefix>/best_checkpoint.pt`.

    The trial directory carries `trial_meta.json`, so the selector is rebuilt
    with the selector config that trained it.
    """
    from selfmorph.selector import build_selector

    d = ckpt_dir / prefix
    meta = json.loads((d / "trial_meta.json").read_text())
    sel = build_selector(meta["selector_config"], anchors, hidden_size, num_traj_layers)
    state = torch.load(d / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    sel.load_state_dict(state["model_state"])
    return sel

def load_production_selector(spec: ModelSpec, anchors: AnchorSet, hidden_size: int, num_traj_layers: int):
    from selfmorph.selector import build_selector

    cfg = load_config(spec)
    path = REPO_ROOT / spec.checkpoint
    sel = build_selector(cfg["selector"], anchors, hidden_size, num_traj_layers)
    state = torch.load(path, map_location="cpu", weights_only=False)
    sel.load_state_dict(state["model_state"])
    return sel

def train_worker(worker_trials: list[dict], base_config: dict, train_dev_pt: str, output_dir: str,
                 device: str, result_queue) -> None:
    import torch as _torch

    from selfmorph.training import train_selector
    from selfmorph.training_data import merge_dev_splits

    _torch.backends.cuda.matmul.allow_tf32 = True
    _torch.backends.cudnn.allow_tf32 = True

    raw_payload = _torch.load(train_dev_pt, map_location="cpu", weights_only=False)
    results = []
    for trial in worker_trials:
        config = copy.deepcopy(base_config)
        config["selector"] = trial["selector_config"]
        if trial.get("action_space") is not None:
            config["action_space"] = trial["action_space"]
        payload = build_derived_payload(raw_payload, trial["payload_spec"])
        payload = merge_dev_splits(payload, into="dev_tune", merge_from=("dev_select",))
        trial_dir = Path(output_dir) / trial["trial_id"]
        t0 = time.time()
        try:
            result = train_selector(
                train_dev_pt=Path(train_dev_pt), output_dir=trial_dir, config=config,
                device=device, payload=payload,
            )
            result.update({"trial_id": trial["trial_id"], "status": "completed", "device": device})
        except Exception as exc:  # noqa: BLE001
            result = {
                "trial_id": trial["trial_id"], "status": "failed", "device": device,
                "error": str(exc), "traceback": traceback.format_exc(), "elapsed_sec": time.time() - t0,
            }
        trial_dir.mkdir(parents=True, exist_ok=True)
        with (trial_dir / "trial_meta.json").open("w", encoding="utf-8") as f:
            json.dump({**trial, "result": result}, f, indent=2, default=str)
        print(f"[{trial['trial_id']}] {result['status']} on {device} "
              f"(epoch={result.get('best_epoch')}, dmr={result.get('best_dev_tune_dmr')}, "
              f"elapsed={result.get('elapsed_sec', 0):.1f}s)", flush=True)
        results.append(result)
    result_queue.put(results)

def run_trials(trials: list[dict], base_config: dict, train_dev_pt: str, output_dir: Path,
               gpus: list[str], plan: dict | None = None) -> list[dict]:
    """Dispatch `trials` over `gpus` (one persistent worker process per GPU,
    the project's standard selector-training grid pattern)."""
    import multiprocessing as mp

    output_dir.mkdir(parents=True, exist_ok=True)
    if plan is not None:
        with (output_dir / "plan.json").open("w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, default=str)

    devices = [f"cuda:{g}" for g in gpus] if gpus else ["cpu"]
    n_workers = len(devices)
    worker_trials: list[list[dict]] = [[] for _ in range(n_workers)]
    for i, trial in enumerate(trials):
        worker_trials[i % n_workers].append(trial)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []
    t0 = time.time()
    for w, device in enumerate(devices):
        p = ctx.Process(
            target=train_worker,
            args=(worker_trials[w], base_config, str(train_dev_pt), str(output_dir), device, result_queue),
        )
        p.start()
        procs.append(p)
    all_results: list[dict] = []
    for _ in procs:
        all_results.extend(result_queue.get())
    for p in procs:
        p.join()

    elapsed = time.time() - t0
    n_ok = sum(1 for r in all_results if r["status"] == "completed")
    n_fail = sum(1 for r in all_results if r["status"] == "failed")
    with (output_dir / "grid_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Done: {n_ok} completed, {n_fail} failed, elapsed={elapsed:.1f}s")
    for r in all_results:
        if r["status"] == "failed":
            print(f"  FAILED {r['trial_id']}: {r['error']}")
    return all_results

def dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"saved {path}")
