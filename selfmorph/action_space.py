"""BASE, REPEAT, and BRANCH actions and their analytic costs.

`r` is the number of additional passes through a REPEAT span, so the span is visited `r + 1` times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from selfmorph.settings import load_runtime_config

ActionType = Literal["BASE", "REPEAT", "BRANCH"]

@dataclass(frozen=True)
class Anchor:
    id: int
    type: ActionType
    layer_span: tuple[int, int] | None
    r: int | None = None
    K: int | None = None
    K_eff: float | None = None
    analytic_cost: float = 1.0

    @property
    def span_len(self) -> int:
        if self.layer_span is None:
            return 0
        i, j = self.layer_span
        return j - i

@dataclass(frozen=True)
class AnchorSet:
    anchors: tuple[Anchor, ...]
    num_layers: int

    def __len__(self) -> int:
        return len(self.anchors)

    def __iter__(self):
        return iter(self.anchors)

    @property
    def by_id(self) -> dict[int, Anchor]:
        return {a.id: a for a in self.anchors}

    @property
    def base_id(self) -> int:
        for a in self.anchors:
            if a.type == "BASE":
                return a.id
        raise ValueError("AnchorSet has no BASE anchor")

def rep_cost(span: tuple[int, int], r: int, num_layers: int) -> float:
    i, j = span
    return 1.0 + r * (j - i) / num_layers

def branch_cost(span: tuple[int, int], k_eff: float, num_layers: int) -> float:
    i, j = span
    return 1.0 + (k_eff - 1.0) * (j - i) / num_layers

def _k_eff(K: int, table: dict[str, float]) -> float:
    key = str(K)
    if key not in table:
        raise KeyError(f"No K_eff entry for K={K} in branch_cost_table={table}. Add one to config/runtime.json.")
    return table[key]

def _validate_span(span: tuple[int, int], morphable_range: tuple[int, int]) -> None:
    i, j = span
    lo, hi = morphable_range
    if not (0 <= i < j):
        raise ValueError(f"Invalid layer_span {span}: need 0 <= i < j")
    if i < lo or j - 1 > hi:
        raise ValueError(
            f"layer_span {span} (layers {i}..{j - 1}) falls outside morphable_layer_range {morphable_range}"
        )

def build_action_space(config: dict | None = None) -> AnchorSet:
    """Build the AnchorSet from config/runtime.json's `action_space` /
    `model` sections (or an equivalent dict, e.g. for scan_span.py to build
    a much larger scan pool without touching the file on disk).
    """
    config = config or load_runtime_config()
    num_layers = config["model"]["num_layers"]
    ac = config["action_space"]
    morphable_range = tuple(ac["morphable_layer_range"])

    anchors: list[Anchor] = []
    next_id = 0

    if ac.get("include_base", True):
        anchors.append(Anchor(id=next_id, type="BASE", layer_span=None, analytic_cost=1.0))
        next_id += 1

    for i, j in ac["repeat_spans"]:
        span = (i, j)
        _validate_span(span, morphable_range)
        for r in ac["repeat_counts"]:
            cost = rep_cost(span, r, num_layers)
            anchors.append(Anchor(id=next_id, type="REPEAT", layer_span=span, r=r, analytic_cost=cost))
            next_id += 1

    k_eff_table = ac["branch_cost_table"]
    for i, j in ac["branch_spans"]:
        span = (i, j)
        _validate_span(span, morphable_range)
        for K in ac["branch_counts"]:
            k_eff = _k_eff(K, k_eff_table)
            cost = branch_cost(span, k_eff, num_layers)
            anchors.append(Anchor(id=next_id, type="BRANCH", layer_span=span, K=K, K_eff=k_eff, analytic_cost=cost))
            next_id += 1

    return AnchorSet(anchors=tuple(anchors), num_layers=num_layers)

def expand_single_block(num_layers: int, span: tuple[int, int]) -> list[int]:
    i, j = span
    return list(range(0, j)) + list(range(i, num_layers))

def repeat_layer_indices(num_layers: int, span: tuple[int, int], r: int) -> list[int]:
    """General REPEAT layer-index expansion. r = number of EXTRA passes
    through layers i..j-1 (r=1 is exactly `expand_single_block`).
    """
    i, j = span
    return list(range(0, j)) + list(range(i, j)) * r + list(range(j, num_layers))
