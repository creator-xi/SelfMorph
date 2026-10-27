"""Benchmark units and selector inference (`_batched_selector_argmax`)."""

from __future__ import annotations

from typing import Callable

import torch

from selfmorph.action_space import AnchorSet
from selfmorph.selector import MorphologySelector

def benchmark_unit_for(dataset: str) -> str:
    """Map a raw dataset name to its benchmark unit (`bbh:<subtask>` -> `"bbh"`)."""
    return "bbh" if dataset.startswith("bbh:") else dataset

def benchmark_units(datasets: list[str]) -> list[str]:
    """Distinct benchmark units present in `datasets`."""
    return sorted({benchmark_unit_for(d) for d in datasets})

def _batched_selector_argmax(
    selector: MorphologySelector, anchors: AnchorSet, trajectory: torch.Tensor,
    idx: list[int], device: str, batch_size: int = 256,
) -> list[int]:
    """Batched selector inference over `idx` rows of `trajectory`, returning argmax anchor id per row."""
    selector.eval()
    chosen: list[int] = []
    idx_t = torch.tensor(idx, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            batch_idx = idx_t[start:start + batch_size]
            traj = trajectory[batch_idx].to(device, non_blocking=True)
            out = selector(traj)
            logp = selector.head.joint_logp_anchors(out, anchors)
            chosen.extend(logp.argmax(dim=-1).cpu().tolist())
    return chosen

def evaluate_selector_on_split(
    selector: MorphologySelector,
    anchors: AnchorSet,
    payload: dict,
    split: str,
    prior_actions: dict[str, int],
    *,
    device: str = "cpu",
    prior_key_fn: Callable[[dict, int], str] | None = None,
) -> dict:
    """Evaluate one selector on `split` relative to `prior_actions`."""
    if prior_key_fn is None:
        prior_key_fn = lambda payload, i: payload["domains"][i]  # noqa: E731

    mask = [s == split for s in payload["splits"]]
    idx = [i for i, m in enumerate(mask) if m]
    if not idx:
        raise ValueError(f"No records for split={split!r}")

    chosen = _batched_selector_argmax(selector, anchors, payload["trajectory"], idx, device)

    per_dataset_selector: dict[str, list[float]] = {}
    per_dataset_prior: dict[str, list[float]] = {}
    per_domain_selector: dict[str, list[float]] = {}
    per_domain_prior: dict[str, list[float]] = {}
    sel_scores: list[float] = []
    prior_scores: list[float] = []
    double_wins = 0

    for i, sel_action in zip(idx, chosen):
        key = prior_key_fn(payload, i)
        if key not in prior_actions:
            continue
        prior_action = prior_actions[key]
        domain = payload["domains"][i]
        rewards = payload["rewards"][i]
        scores = payload["scores"][i]
        dataset = payload["datasets"][i]

        sel_reward, sel_score = float(rewards[sel_action]), float(scores[sel_action])
        prior_reward, prior_score = float(rewards[prior_action]), float(scores[prior_action])
        sel_scores.append(sel_score)
        prior_scores.append(prior_score)

        unit = benchmark_unit_for(dataset)
        per_dataset_selector.setdefault(unit, []).append(sel_reward)
        per_dataset_prior.setdefault(unit, []).append(prior_reward)
        per_domain_selector.setdefault(domain, []).append(sel_reward)
        per_domain_prior.setdefault(domain, []).append(prior_reward)
        if sel_score > prior_score and sel_reward > prior_reward:
            double_wins += 1

    if not sel_scores:
        raise ValueError(f"split={split!r}: no samples had a key present in prior_actions")

    def _mean(d: dict[str, list[float]]) -> dict[str, float]:
        return {k: sum(v) / len(v) for k, v in d.items()}

    ds_sel_mean, ds_prior_mean = _mean(per_dataset_selector), _mean(per_dataset_prior)
    dom_sel_mean, dom_prior_mean = _mean(per_domain_selector), _mean(per_domain_prior)

    benchmark_double_wins = sum(
        1 for ds in ds_sel_mean if ds_sel_mean[ds] > ds_prior_mean.get(ds, -float("inf"))
    )
    worst_domain_reward_margin = min(
        (dom_sel_mean[d] - dom_prior_mean.get(d, 0.0) for d in dom_sel_mean), default=0.0,
    )
    reward_margin = sum(dom_sel_mean.values()) / len(dom_sel_mean) - sum(dom_prior_mean.values()) / len(dom_prior_mean)

    return {
        "score_margin_over_prior": float(sum(sel_scores) / len(sel_scores) - sum(prior_scores) / len(prior_scores)),
        "reward_margin_over_prior": float(reward_margin),
        "benchmark_double_wins": int(benchmark_double_wins),
        "n_benchmark_units": len(ds_sel_mean),
        "worst_domain_reward_margin": float(worst_domain_reward_margin),
        "dmr": sum(dom_sel_mean.values()) / len(dom_sel_mean),
        "sample_level_double_wins": int(double_wins),
    }
