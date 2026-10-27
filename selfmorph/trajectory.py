"""Trajectory extraction and truncated prefix probing.

During prompt prefill (`Probe`), SelfMorph extracts the last-token hidden state
at `L` preselected depths (`layer_ids`). By default:
  - `collect_trajectory` runs a truncated direct forward through layers
    `[0, L_probe)` where `L_probe = max(layer_ids) + 1`, skipping layers beyond
    the deepest observed trajectory layer (`truncate=True`).
  - `probe_with_prefix` runs the probe pass with `use_cache=True` and captures
    reusable KV caches and boundary hidden states (`SharedPrefix`) so that a
    subsequently selected `BASE` (`BASE`) or `REPEAT` (`REPEAT`) action can resume
    directly without recomputing the shared prefix (`execute_anchor(..., prefix=...)`).
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch

from selfmorph.core.layer_duplicator import build_model_with_layers
from selfmorph.executor import SharedPrefix
from selfmorph.model import HiddenStateCollector, get_decoder_layers
from selfmorph.settings import load_runtime_config

USE_INFERENCE_MODE = os.environ.get("SELFMORPH_INFERENCE_MODE", "1") == "1"

def _guard():
    return torch.inference_mode() if USE_INFERENCE_MODE else torch.no_grad()

@dataclass(frozen=True)
class TrajectorySpec:
    dense_layer_ids: tuple[int, ...]
    sparse_layer_ids: tuple[int, ...]
    default_layer_set: str = "dense"

    def layer_ids(self, layer_set: str | None = None) -> tuple[int, ...]:
        layer_set = layer_set or self.default_layer_set
        if layer_set == "dense":
            return self.dense_layer_ids
        if layer_set == "sparse":
            return self.sparse_layer_ids
        raise ValueError(f"Unknown layer_set {layer_set!r}, expected 'dense' or 'sparse'")

def load_trajectory_spec(config: dict | None = None) -> TrajectorySpec:
    config = config or load_runtime_config()
    tj = config["trajectory"]
    return TrajectorySpec(
        dense_layer_ids=tuple(tj["dense_layer_ids"]),
        sparse_layer_ids=tuple(tj["sparse_layer_ids"]),
        default_layer_set=tj.get("default_layer_set", "dense"),
    )

_TRUNCATED_PROBE_CACHE: dict[tuple, object] = {}

def _truncated_probe_wrapper(model, depth: int):
    key = (id(model), int(depth))
    w = _TRUNCATED_PROBE_CACHE.get(key)
    if w is None:
        w = build_model_with_layers(model, list(range(int(depth))))
        _TRUNCATED_PROBE_CACHE[key] = w
    return w

def collect_trajectory_forward(
    model,
    enc: dict,
    *,
    layer_ids: tuple[int, ...],
    use_cache: bool = False,
    truncate: bool = True,
) -> dict[int, torch.Tensor]:
    """Capture last-prompt-token hidden states at `layer_ids` via a single direct forward.

    When `truncate=True` (default), execution stops at `L_probe = max(layer_ids) + 1`,
    avoiding unnecessary computation in deeper layers while producing bit-identical
    trajectory features.
    """
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    layers = get_decoder_layers(model)
    depth = (max(layer_ids) + 1) if (truncate and layer_ids) else None
    with HiddenStateCollector(model, list(layer_ids)) as collector:
        with _guard():
            if depth is not None and depth < len(layers):
                _truncated_probe_wrapper(model, depth).forward(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache,
                )
            else:
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)
        return collector.collected()

def collect_trajectory(
    model,
    enc: dict,
    *,
    layer_ids: tuple[int, ...],
    pad_id: int | None = None,
    truncate: bool = True,
) -> dict[int, torch.Tensor]:
    """Extract the depth-wise hidden trajectory `H` at `layer_ids` during prompt prefill."""
    return collect_trajectory_forward(
        model, enc, layer_ids=layer_ids, use_cache=False, truncate=truncate,
    )

def probe_with_prefix(
    model,
    enc: dict,
    *,
    layer_ids: tuple[int, ...],
    prefix_depths: tuple[int, ...],
    truncate: bool = True,
) -> dict:
    """Run the Probe prefill once and return both trajectory `hiddens` and a `SharedPrefix`.

    `prefix_depths` specifies the reusable boundary layer indices (e.g., `L_probe`
    for `BASE` and candidate span exits `j` for `REPEAT`).
    """
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    layers = get_decoder_layers(model)
    traj_depth = (max(layer_ids) + 1) if (truncate and layer_ids) else len(layers)
    wanted = sorted({int(d) for d in prefix_depths if int(d) > 0})
    depth = max([traj_depth] + wanted)
    if depth > len(layers):
        raise ValueError(f"prefix depth {depth} exceeds stack depth {len(layers)}")

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for d in wanted:
        if d > depth:
            continue

        def _mk(dd):
            def hook(module, inputs, output):
                captured[dd] = output[0] if isinstance(output, tuple) else output
            return hook

        if d - 1 < len(layers):
            handles.append(layers[d - 1].register_forward_hook(_mk(d)))
    try:
        with HiddenStateCollector(model, list(layer_ids)) as collector:
            with _guard():
                if depth < len(layers):
                    out = _truncated_probe_wrapper(model, depth).forward(
                        input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                    )
                else:
                    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            hiddens = collector.collected()
        cache = out.past_key_values
        slots = list(cache.layers[:depth])
    finally:
        for h in handles:
            h.remove()
    return {
        "hiddens": hiddens,
        "depth": depth,
        "prefix": SharedPrefix(
            cache_layers=slots, hidden_at=captured,
            prompt_len=int(input_ids.shape[1]), depth=depth,
        ),
    }

class _SteppedHiddenCollector:
    """Forward-hook collector that captures only forward step `capture_step` (default 0 = prefill)."""

    def __init__(self, model, layer_ids, capture_step: int = 0):
        self.layer_ids = sorted(set(layer_ids))
        self._layers = get_decoder_layers(model)
        self._handles: list = []
        self.capture_step = int(capture_step)
        self.n_forwards = 0
        self.captured: dict[int, torch.Tensor] = {}
        self._first_id = self.layer_ids[0]

    def __enter__(self) -> "_SteppedHiddenCollector":
        for lid in self.layer_ids:
            self._handles.append(self._layers[lid].register_forward_hook(self._make_hook(lid)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, layer_id: int):
        def hook(module, inputs, output):
            if layer_id == self._first_id:
                self.n_forwards += 1
            if self.n_forwards - 1 != self.capture_step:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            self.captured[layer_id] = hidden[:, -1, :].detach().to(torch.float32).cpu()

        return hook

def noop_generate_with_trajectory(
    model,
    tokenizer,
    enc: dict,
    *,
    layer_ids: tuple[int, ...],
    pad_id: Optional[int],
    max_new_tokens: int,
    measure_latency: bool = True,
    prefix_layers: Optional[int] = None,
    prefix_boundaries: tuple[int, ...] = (),
) -> dict:
    """Run a single `BASE` generation and capture step-0 prefill trajectory + optional `SharedPrefix`."""
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_len = int(input_ids.shape[1])

    harvest: dict = {}
    boundary_handles: list = []
    orig_forward = model.forward
    if prefix_layers:
        layers = get_decoder_layers(model)

        def _boundary_hook(b):
            def hook(module, inputs, output):
                if collector.n_forwards - 1 != 0:
                    return
                harvest.setdefault("hidden_at", {})[b] = output[0] if isinstance(output, tuple) else output
            return hook

        for b in prefix_boundaries:
            if b > 0:
                boundary_handles.append(layers[b - 1].register_forward_hook(_boundary_hook(int(b))))

        def _patched_forward(*a, **kw):
            out = orig_forward(*a, **kw)
            if "cache" not in harvest:
                pkv = kw.get("past_key_values")
                if pkv is not None:
                    harvest["cache"] = [copy.deepcopy(x) for x in pkv.layers[: int(prefix_layers)]]
            return out

        _patched_forward.__wrapped__ = orig_forward  # type: ignore[attr-defined]
        model.forward = _patched_forward  # type: ignore[method-assign]

    if measure_latency and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        with _SteppedHiddenCollector(model, list(layer_ids)) as collector:
            with _guard():
                out = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id,
                )
    finally:
        model.forward = orig_forward  # type: ignore[method-assign]
        for h in boundary_handles:
            h.remove()
    if measure_latency and torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    seq = out.sequences if hasattr(out, "sequences") else out
    text = tokenizer.decode(seq[0][prompt_len:], skip_special_tokens=True)

    result = {
        "hiddens": collector.captured,
        "text": text,
        "latency_s": latency,
        "n_forwards": collector.n_forwards,
        "prefix": None,
    }
    if prefix_layers:
        result["prefix"] = SharedPrefix(
            cache_layers=harvest["cache"], hidden_at=harvest["hidden_at"],
            prompt_len=prompt_len, depth=int(prefix_layers),
        )
    return result

def stack_trajectory(hiddens: dict[int, torch.Tensor], layer_ids: tuple[int, ...]) -> torch.Tensor:
    """Stack per-layer `[1, H]` hidden states into `[num_layers, H]` in `layer_ids` order."""
    missing = [lid for lid in layer_ids if lid not in hiddens]
    if missing:
        raise KeyError(f"Missing captured layers {missing}; got {sorted(hiddens.keys())}")
    return torch.stack([hiddens[lid].squeeze(0) for lid in layer_ids], dim=0)
