"""BRANCH execution: split hidden states, run the span on each branch, and merge by averaging."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from selfmorph.core.branch_perturbation import HookHandle, PerturbationConfig, install_perturbation_hooks

def _get_text_stack(base_model: nn.Module) -> tuple[Any, str, nn.ModuleList, int]:
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        owner, attr = base_model.model, "layers"
    elif (
        hasattr(base_model, "model")
        and hasattr(base_model.model, "language_model")
        and hasattr(base_model.model.language_model, "layers")
    ):
        owner, attr = base_model.model.language_model, "layers"
    elif hasattr(base_model, "language_model") and hasattr(base_model.language_model, "layers"):
        owner, attr = base_model.language_model, "layers"
    else:
        raise AttributeError(
            "Could not locate text decoder layers (model.layers / model.language_model.layers / language_model.layers)."
        )
    layers = getattr(owner, attr)
    hidden_size = int(getattr(base_model.config, "hidden_size", 0)) or int(
        getattr(getattr(base_model.config, "text_config", None), "hidden_size", 0)
    )
    return owner, attr, layers, hidden_size

def _replicate_inputs(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], K: int):
    """Repeat the leading batch dim by K: [B,T] -> [K*B,T]. Branch index
    varies outermost: row k*B+i corresponds to (branch=k, sample=i).
    """
    if K <= 1:
        return input_ids, attention_mask
    rep_ids = input_ids.repeat(K, 1)
    rep_mask = attention_mask.repeat(K, 1) if attention_mask is not None else None
    return rep_ids, rep_mask

def _make_split_merge_pre_hook(num_branches: int, batch_size: int, *, strategy: str = "mean"):
    """Pre-hook on layer e: combine the K branches and re-expand so the
    KV cache layout stays valid. strategy="mean" is plain arithmetic mean;
    "norm_weighted" weights by hidden-state L2 norm per branch.
    """
    K = int(num_branches)
    valid = {"mean", "norm_weighted"}
    if strategy not in valid:
        raise ValueError(f"strategy must be in {valid}, got {strategy!r}")

    def _hook(module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        if K <= 1 or not args:
            return args, kwargs
        h = args[0]
        if not isinstance(h, torch.Tensor) or h.shape[0] % K != 0:
            return args, kwargs
        local_B = h.shape[0] // K
        blk = h.reshape(K, local_B, *h.shape[1:])

        if strategy == "mean":
            merged = blk.mean(dim=0, keepdim=False)
        else:
            per_token_norm = blk.detach().float().norm(dim=-1)
            w_logits = per_token_norm.mean(dim=-1)
            w = torch.softmax(w_logits, dim=0).to(blk.dtype)
            for _ in range(blk.dim() - w.dim()):
                w = w.unsqueeze(-1)
            merged = (blk * w).sum(dim=0)

        h_new = merged.unsqueeze(0).expand(K, *merged.shape).reshape(h.shape)
        return (h_new,) + args[1:], kwargs

    _hook.__name__ = f"split_merge_pre_hook[{strategy}]"
    return _hook

@dataclass
class BranchConfig:
    """Top-level config describing a branch-ensemble setup.

    method is always "split_merge" -- the field is kept (rather than
    hardcoding) so BranchConfig.__post_init__ still fails loudly if a
    stale config/runtime.json or caller ever passes anything else.
    """

    method: str
    branch_start: int
    merge_at: int
    num_branches: int = 4
    perturbation: str = "gaussian"
    perturbation_strength: float = 0.1
    perturbation_strengths: Optional[list] = None
    merge_strategy: str = "mean"
    seed: int = 42
    branch_average: str = "prob"
    gaussian_l2_normalize: bool = True
    gaussian_scale_mode: str = "relative_l2"
    dropout_oneshot: bool = False

    def __post_init__(self) -> None:
        if self.method != "split_merge":
            raise ValueError(f"method must be 'split_merge', got {self.method!r}")
        if self.merge_strategy not in {"mean", "norm_weighted"}:
            raise ValueError(f"merge_strategy must be 'mean' or 'norm_weighted', got {self.merge_strategy!r}")
        if self.merge_at is None:
            raise ValueError("split_merge requires merge_at to be set")
        if self.merge_at <= self.branch_start:
            raise ValueError(f"merge_at ({self.merge_at}) must be > branch_start ({self.branch_start})")
        if self.num_branches < 1:
            raise ValueError("num_branches must be >= 1")
        if self.branch_average not in {"prob", "logits"}:
            raise ValueError("branch_average must be 'prob' or 'logits'")
        if self.gaussian_scale_mode not in {"relative_l2", "direct_add"}:
            raise ValueError("gaussian_scale_mode must be 'relative_l2' or 'direct_add'")
        if self.perturbation_strengths is not None:
            if len(self.perturbation_strengths) != self.num_branches:
                raise ValueError("perturbation_strengths length must equal num_branches")
            if abs(float(self.perturbation_strengths[0])) > 1e-12:
                raise ValueError("perturbation_strengths[0] must be 0.0 (anchor)")
            self.perturbation_strengths = [float(x) for x in self.perturbation_strengths]

    def to_perturb_cfg(self, batch_size: int) -> PerturbationConfig:
        return PerturbationConfig(
            kind=self.perturbation,
            strength=self.perturbation_strength,
            strengths_per_branch=self.perturbation_strengths,
            num_branches=self.num_branches,
            batch_size=batch_size,
            seed=self.seed,
            gaussian_l2_normalize=self.gaussian_l2_normalize,
            gaussian_scale_mode=self.gaussian_scale_mode,
            dropout_oneshot=self.dropout_oneshot,
        )

class SplitMergeModel(nn.Module):

    def __init__(self, base_model: nn.Module, cfg: BranchConfig):
        super().__init__()
        if cfg.method != "split_merge":
            raise ValueError(f"SplitMergeModel requires method='split_merge'; got {cfg.method!r}")
        self.base_model = base_model
        self.cfg = cfg
        self.config = base_model.config
        owner, attr, layers, hidden_size = _get_text_stack(base_model)
        self._layers_owner = owner
        self._layers_attr = attr
        self._layers = layers
        self._hidden_size = hidden_size
        self._n_layers = len(layers)

        if not (0 <= cfg.branch_start < self._n_layers):
            raise ValueError(f"branch_start={cfg.branch_start} out of range [0,{self._n_layers})")
        if not (cfg.branch_start < cfg.merge_at <= self._n_layers):
            raise ValueError(f"merge_at={cfg.merge_at} must satisfy s<e<=N (s={cfg.branch_start}, N={self._n_layers})")

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    @contextmanager
    def _activate(self, batch_size: int):
        K = self.cfg.num_branches
        if K <= 1:
            yield
            return
        perturb_cfg = self.cfg.to_perturb_cfg(batch_size=batch_size)
        handle: HookHandle = install_perturbation_hooks(
            self._layers, branch_start=self.cfg.branch_start, branch_end=self.cfg.merge_at,
            cfg=perturb_cfg, hidden_size=self._hidden_size,
        )
        merge_layer = self._layers[self.cfg.merge_at]
        merge_handle = merge_layer.register_forward_pre_hook(
            _make_split_merge_pre_hook(K, batch_size, strategy=self.cfg.merge_strategy), with_kwargs=True,
        )
        try:
            yield
        finally:
            handle.remove()
            try:
                merge_handle.remove()
            except Exception:
                pass

    def generate(self, *args, **kwargs):
        kwargs.pop("past_key_values", None)
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None:
            input_ids = args[0] if args else None
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2-D [B,T]; got shape {tuple(input_ids.shape)}")

        B = int(input_ids.shape[0])
        K = int(self.cfg.num_branches)
        attention_mask = kwargs.get("attention_mask", None)
        rep_ids, rep_mask = _replicate_inputs(input_ids, attention_mask, K)
        kwargs["input_ids"] = rep_ids
        if rep_mask is not None:
            kwargs["attention_mask"] = rep_mask
        if args and args[0] is not None and args[0].shape == input_ids.shape:
            args = (rep_ids,) + tuple(args[1:])

        with self._activate(batch_size=B):
            output = self.base_model.generate(*args, **kwargs)

        if K > 1 and isinstance(output, torch.Tensor) and output.shape[0] == K * B:
            output = output[:B]
        elif K > 1 and hasattr(output, "sequences") and output.sequences.shape[0] == K * B:
            output.sequences = output.sequences[:B]
        return output

def build_branch_model(base_model: nn.Module, cfg: BranchConfig) -> SplitMergeModel:
    if cfg.method == "split_merge":
        return SplitMergeModel(base_model, cfg)
    raise ValueError(f"Unknown method: {cfg.method!r}; only 'split_merge' is supported")
