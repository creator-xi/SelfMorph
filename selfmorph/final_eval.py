"""Fixed-action baselines."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from selfmorph.action_space import AnchorSet
from selfmorph.selection import _batched_selector_argmax, benchmark_unit_for

@dataclass
class DomainPrior:
    """Train-only static baseline: one fixed best anchor per domain (by mean train utility)."""

    actions: dict[str, int]

    def action_for(self, domain: str) -> int | None:
        return self.actions.get(domain)

def compute_domain_prior(payload: dict, anchors: AnchorSet) -> DomainPrior:
    mask = [s == "train" for s in payload["splits"]]
    per_domain_reward: dict[str, list[list[float]]] = {}
    for i, is_train in enumerate(mask):
        if not is_train:
            continue
        domain = payload["domains"][i]
        per_domain_reward.setdefault(domain, []).append(payload["rewards"][i].tolist())

    actions: dict[str, int] = {}
    for domain, reward_rows in per_domain_reward.items():
        mean_reward = torch.tensor(reward_rows).mean(dim=0)
        actions[domain] = int(mean_reward.argmax().item())
    return DomainPrior(actions=actions)

@dataclass
class BenchmarkPrior:
    """Train-only static baseline (`Bench Prior`): one fixed best anchor per benchmark unit."""

    actions: dict[str, int]

    def action_for(self, benchmark_unit: str) -> int | None:
        return self.actions.get(benchmark_unit)

def compute_benchmark_prior(payload: dict, anchors: AnchorSet) -> BenchmarkPrior:
    mask = [s == "train" for s in payload["splits"]]
    per_unit_reward: dict[str, list[list[float]]] = {}
    for i, is_train in enumerate(mask):
        if not is_train:
            continue
        unit = benchmark_unit_for(payload["datasets"][i])
        per_unit_reward.setdefault(unit, []).append(payload["rewards"][i].tolist())

    actions: dict[str, int] = {}
    for unit, reward_rows in per_unit_reward.items():
        mean_reward = torch.tensor(reward_rows).mean(dim=0)
        actions[unit] = int(mean_reward.argmax().item())
    return BenchmarkPrior(actions=actions)

def compute_fixed_anchor(payload: dict, anchor_type: str, anchors: AnchorSet) -> int:
    """`Repeat` (`REPEAT`) or `Branch` (`BRANCH`) baseline: single best anchor of `anchor_type` on train."""
    mask = [s == "train" for s in payload["splits"]]
    idx = [i for i, m in enumerate(mask) if m]
    candidate_ids = [a.id for a in anchors if a.type == anchor_type]
    mean_reward = torch.stack([payload["rewards"][i] for i in idx]).mean(dim=0)
    return max(candidate_ids, key=lambda aid: mean_reward[aid].item())

def policy_actions(
    payload: dict,
    split: str,
    selector,
    anchors: AnchorSet,
    domain_prior: DomainPrior,
    fixed_rep_id: int,
    fixed_br_id: int,
    device: str = "cpu",
    id_eval_split: str = "eval",
    benchmark_prior: BenchmarkPrior | None = None,
) -> dict[str, list[int | None]]:
    """Per-sample chosen anchor id for each policy on `split`."""
    mask = [s == split for s in payload["splits"]]
    idx = [i for i, m in enumerate(mask) if m]
    base_id = anchors.base_id
    is_id_split = split == id_eval_split

    chosen = _batched_selector_argmax(selector, anchors, payload["trajectory"], idx, device)

    base_lm = [base_id] * len(idx)
    fixed_rep = [fixed_rep_id] * len(idx)
    fixed_br = [fixed_br_id] * len(idx)
    prior = [domain_prior.action_for(payload["domains"][i]) if is_id_split else None for i in idx]

    result = {
        "base_lm": base_lm, "fixed_rep": fixed_rep, "fixed_br": fixed_br,
        "domain_prior": prior, "selfmorph": chosen,
    }
    if benchmark_prior is not None:
        bp = [
            benchmark_prior.action_for(benchmark_unit_for(payload["datasets"][i])) if is_id_split else None
            for i in idx
        ]
        result["benchmark_prior"] = bp
    return result

def _gather_metric(payload: dict, split: str, actions: list[int | None], field: str) -> list[float | None]:
    mask = [s == split for s in payload["splits"]]
    idx = [i for i, m in enumerate(mask) if m]
    return [None if action is None else float(payload[field][row_idx][action]) for row_idx, action in zip(idx, actions)]

def evaluate_policies(payload: dict, split: str, actions_by_policy: dict[str, list[int | None]]) -> dict:
    """Aggregate per-policy score, cost, utility (`reward`), and domain-macro utility on `split`."""
    mask = [s == split for s in payload["splits"]]
    idx = [i for i, m in enumerate(mask) if m]
    domains = [payload["domains"][i] for i in idx]
    datasets = [payload["datasets"][i] for i in idx]

    results: dict[str, dict] = {}
    for policy, actions in actions_by_policy.items():
        scores = _gather_metric(payload, split, actions, "scores")
        costs = _gather_metric(payload, split, actions, "costs")
        rewards = _gather_metric(payload, split, actions, "rewards")

        by_domain: dict[str, list[float]] = {}
        by_dataset: dict[str, list[float]] = {}
        n_applicable, n_total = 0, len(actions)
        for d, ds, s, r in zip(domains, datasets, scores, rewards):
            if s is None:
                continue
            n_applicable += 1
            by_domain.setdefault(d, []).append(r)
            by_dataset.setdefault(ds, []).append(s)

        results[policy] = {
            "domains_used": len(by_domain), "domains_total": len(set(domains)),
            "samples_applicable": n_applicable, "samples_total": n_total,
            "domain_macro_reward": (
                sum(sum(v) / len(v) for v in by_domain.values()) / len(by_domain)
            ) if by_domain else None,
            "by_dataset_mean_score": {k: sum(v) / len(v) for k, v in by_dataset.items()},
            "mean_score": (sum(s for s in scores if s is not None) / n_applicable) if n_applicable else None,
            "mean_cost": (sum(c for c in costs if c is not None) / n_applicable) if n_applicable else None,
            "mean_reward": (sum(r for r in rewards if r is not None) / n_applicable) if n_applicable else None,
        }
    return results
