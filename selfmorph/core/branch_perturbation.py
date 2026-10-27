"""Gaussian perturbation applied at the BRANCH span entrance."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn

GAUSSIAN_CHUNKED_NORM = os.environ.get("SELFMORPH_GAUSSIAN_CHUNKED_NORM", "1") == "1"
_GAUSSIAN_NORM_CHUNK_BYTES = int(os.environ.get("SELFMORPH_GAUSSIAN_NORM_CHUNK_BYTES", str(64 << 20)))

def _row_l2_norm(x: torch.Tensor, *, keepdim: bool = True) -> torch.Tensor:
    """Bit-identical fp32 L2 norm over the last dim, computed in batch chunks to bound peak memory."""
    rows = int(x.shape[0]) if x.dim() >= 1 else 1
    per_row = 1
    for s in x.shape[1:]:
        per_row *= int(s)
    chunk = int(_GAUSSIAN_NORM_CHUNK_BYTES // max(4 * per_row, 1))
    if chunk >= rows or chunk < 1:
        return x.detach().float().norm(dim=-1, keepdim=keepdim)
    out = torch.empty((rows,) + tuple(x.shape[1:-1]) + (1,), dtype=torch.float32, device=x.device)
    for start in range(0, rows, chunk):
        end = min(rows, start + chunk)
        out[start:end] = x[start:end].detach().float().norm(dim=-1, keepdim=True)
    return out if keepdim else out.squeeze(-1)

@dataclass
class PerturbationConfig:
    kind: str = "gaussian"
    strength: float = 0.1
    num_branches: int = 1
    batch_size: int = 1
    seed: int = 42
    strengths_per_branch: Optional[List[float]] = None
    gaussian_l2_normalize: bool = True
    gaussian_scale_mode: str = "relative_l2"
    dropout_oneshot: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"none", "gaussian"}:
            raise ValueError(f"Unknown perturbation kind: {self.kind!r}")
        if self.num_branches < 1:
            raise ValueError("num_branches must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.gaussian_scale_mode not in {"relative_l2", "direct_add"}:
            raise ValueError("gaussian_scale_mode must be 'relative_l2' or 'direct_add'")
        if self.strengths_per_branch is not None:
            if len(self.strengths_per_branch) != self.num_branches:
                raise ValueError("strengths_per_branch length must equal num_branches")
            if abs(float(self.strengths_per_branch[0])) > 1e-12:
                raise ValueError("strengths_per_branch[0] must be 0.0 (anchor branch)")
            self.strengths_per_branch = [float(x) for x in self.strengths_per_branch]

def _make_gaussian_hook(cfg: PerturbationConfig) -> Callable[..., Any]:
    """Pre-forward hook injecting radially scheduled Gaussian noise at the branch-start layer."""
    K = cfg.num_branches
    base_seed = cfg.seed

    if cfg.strengths_per_branch is not None:
        eps_list = list(cfg.strengths_per_branch)
    else:
        eps_list = [0.0] + [float(cfg.strength)] * (K - 1)
    eps_cpu = torch.tensor(eps_list, dtype=torch.float32)

    cache: dict = {"eps_dev": None, "eps_dev_dtype": None}
    state = {"step": 0, "applied": False}

    def _hook(module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        if not args:
            return args, kwargs
        h = args[0]
        if not isinstance(h, torch.Tensor) or state["applied"] or h.shape[0] % K != 0:
            return args, kwargs
        local_B = h.shape[0] // K

        if GAUSSIAN_CHUNKED_NORM:
            norm = _row_l2_norm(h, keepdim=True).to(h.dtype)
        else:
            norm = h.detach().float().norm(dim=-1, keepdim=True).to(h.dtype)

        if cache["eps_dev"] is None or cache["eps_dev"].device != h.device or cache["eps_dev_dtype"] != h.dtype:
            cache["eps_dev"] = eps_cpu.to(device=h.device, dtype=h.dtype)
            cache["eps_dev_dtype"] = h.dtype
        eps_per_branch = cache["eps_dev"]
        scale = eps_per_branch.repeat_interleave(local_B).view(-1, 1, 1)

        gen = torch.Generator(device=h.device)
        gen.manual_seed(base_seed * 100003 + state["step"])
        state["step"] += 1
        noise = torch.randn(h.shape, generator=gen, device=h.device, dtype=h.dtype)
        if cfg.gaussian_scale_mode == "direct_add":
            h_perturbed = h + scale * noise
        else:
            if cfg.gaussian_l2_normalize:
                if GAUSSIAN_CHUNKED_NORM:
                    noise_norm = _row_l2_norm(noise, keepdim=True).clamp_min(1e-12).to(h.dtype)
                else:
                    noise_norm = noise.float().norm(dim=-1, keepdim=True).clamp_min(1e-12).to(h.dtype)
                noise = noise / noise_norm
            h_perturbed = h + scale * norm * noise
        state["applied"] = True

        return (h_perturbed,) + args[1:], kwargs

    _hook.__name__ = "gaussian_perturb_pre_hook"
    return _hook

@dataclass
class HookHandle:
    handles: List[Any] = field(default_factory=list)

    def remove(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()

def install_perturbation_hooks(
    layers: nn.ModuleList,
    branch_start: int,
    branch_end: Optional[int],
    cfg: PerturbationConfig,
    *,
    hidden_size: int,
) -> HookHandle:
    """Install the Gaussian perturbation hook at `layers[branch_start]`."""
    handle = HookHandle()
    if cfg.kind == "none" or cfg.num_branches <= 1:
        return handle

    n_layers = len(layers)
    if not (0 <= branch_start < n_layers):
        raise ValueError(f"branch_start={branch_start} out of range [0,{n_layers})")
    e = branch_end if branch_end is not None else n_layers
    if not (branch_start < e <= n_layers):
        raise ValueError(f"branch_end={branch_end} must satisfy s<e<=N (got s={branch_start}, N={n_layers})")

    if cfg.kind == "gaussian":
        target = layers[branch_start]
        h = target.register_forward_pre_hook(_make_gaussian_hook(cfg), with_kwargs=True)
        handle.handles.append(h)
        return handle

    raise RuntimeError(f"Unhandled perturbation kind: {cfg.kind}")
