"""Execute a selected morphology (`BASE` / `REPEAT` / `BRANCH`) on an encoded input.

By default, execution uses these lossless optimizations:
  1. `torch.inference_mode()` and cached `LayerDuplicatedModel` / `SplitMergeModel`
     wrappers across calls;
  2. Prefix reuse (`SharedPrefix` / `execute_from_prefix`) when resuming from a
     shared `Probe` pass for `BASE` (`BASE`) and `REPEAT` (`REPEAT`);
  3. Lossless KV cache batch deduplication (`br_kv_dedup`) outside the active
     branch span `[i, j)` for `BRANCH` (`BRANCH`).
"""

from __future__ import annotations

import contextlib
import copy
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch

from selfmorph.action_space import Anchor, repeat_layer_indices
from selfmorph.core.br_kv_dedup import br_kv_dedup
from selfmorph.core.branch_model import BranchConfig, build_branch_model
from selfmorph.core.layer_duplicator import build_model_with_layers

USE_INFERENCE_MODE = os.environ.get("SELFMORPH_INFERENCE_MODE", "1") == "1"
USE_PREFIX_SHARING = os.environ.get("SELFMORPH_PREFIX_SHARING", "1") == "1"

def _guard():
    return torch.inference_mode() if USE_INFERENCE_MODE else torch.no_grad()

_PREFIX_CFG_CACHE: dict = {}
_PREFIX_CFG_CACHE_CAP = 64

def _permuted_config(model, seq: list[int]):
    """Deep-copied model config whose `layer_types` and `num_hidden_layers` follow `seq`."""
    key = (id(model), tuple(seq))
    got = _PREFIX_CFG_CACHE.get(key)
    if got is None:
        cfg = copy.deepcopy(model.config)
        targets = [cfg]
        for attr in ("text_config", "language_config", "llm_config"):
            sub = getattr(cfg, attr, None)
            if sub is not None:
                targets.append(sub)
        for t in targets:
            lt = getattr(t, "layer_types", None)
            if lt is not None and len(lt) > max(seq):
                t.layer_types = [lt[k] for k in seq]
            elif lt is not None:
                t.layer_types = [lt[k % len(lt)] for k in seq]
            if getattr(t, "num_hidden_layers", None) is not None or hasattr(t, "num_hidden_layers"):
                t.num_hidden_layers = len(seq)
        if len(_PREFIX_CFG_CACHE) >= _PREFIX_CFG_CACHE_CAP:
            _PREFIX_CFG_CACHE.clear()
        _PREFIX_CFG_CACHE[key] = cfg
        got = cfg
    return got

@dataclass
class SharedPrefix:
    """Reusable prompt prefill state captured during the Probe stage."""

    cache_layers: list
    hidden_at: dict
    prompt_len: int
    depth: int

_PREFIX_STACK_CACHE: dict = {}

def _truncated_prefix_stack(model, depth: int):
    key = (id(model), int(depth))
    got = _PREFIX_STACK_CACHE.get(key)
    if got is None:
        got = build_model_with_layers(model, list(range(int(depth))))
        _PREFIX_STACK_CACHE[key] = got
    return got

def build_shared_prefix(model, enc, *, needed_layers, pad_id=None) -> SharedPrefix:
    """Run a single base forward over layers `[0, max(needed_layers))` and capture reusable prefix state."""
    from selfmorph.model import get_decoder_layers

    depth = int(max(needed_layers))
    layers = get_decoder_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for j in sorted(set(int(x) for x in needed_layers)):
        if j <= 0:
            continue

        def _mk(jj):
            def hook(module, inputs, output):
                captured[jj] = output[0] if isinstance(output, tuple) else output
            return hook

        handles.append(layers[j - 1].register_forward_hook(_mk(j)))
    try:
        runner = _truncated_prefix_stack(model, depth)
        with _guard():
            out = runner.forward(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=True,
            )
        cache = out.past_key_values
        slots = list(cache.layers[:depth])
    finally:
        for h in handles:
            h.remove()
    return SharedPrefix(
        cache_layers=slots, hidden_at=captured,
        prompt_len=int(enc["input_ids"].shape[1]), depth=depth,
    )

def _merged_cache(model, seq: list[int], layers: list):
    from transformers.cache_utils import DynamicCache

    cfg = _permuted_config(model, seq)
    c = DynamicCache(config=cfg)
    if len(c.layers) != len(layers):
        raise RuntimeError(f"cache typing mismatch: built {len(c.layers)} slots for seq len {len(layers)}")
    for k, layer in enumerate(layers):
        c.layers[k] = layer
    return c

def _prefix_start_depth(anchor: Anchor, prefix: SharedPrefix, num_layers: int) -> Optional[int]:
    """Return the deepest layer position `d` at which `anchor` can resume from `prefix`."""
    if anchor.type == "BASE":
        cands = [int(d) for d in prefix.hidden_at if 0 < int(d) <= num_layers]
        return max(cands) if cands else None
    if anchor.type == "REPEAT":
        j = int(anchor.layer_span[1])
        seq = _cached_repeat_layer_indices(num_layers, anchor.layer_span, anchor.r)
        if j in prefix.hidden_at and seq[:j] == list(range(j)):
            return j
        return None
    return None

def _execute_from_prefix_impl(
    model, tokenizer, enc, anchor: Anchor, *, prefix: SharedPrefix,
    seq: list[int], d: int, max_new_tokens: int, measure_latency: bool = True,
) -> dict:
    """Execute positions `[d, L)` over the prompt from `prefix.hidden_at[d]` and decode."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    L = len(seq)
    h_in = prefix.hidden_at[d]
    attention_mask = enc["attention_mask"]
    if measure_latency and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    suffix = seq[d:]
    suffix_wrapper = _get_rep_wrapper(model, suffix)
    with _guard():
        out_b = suffix_wrapper.forward(inputs_embeds=h_in, attention_mask=attention_mask, use_cache=True)
        first_tok = out_b.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cache_b = out_b.past_key_values

    prefix_slots = [copy.deepcopy(x) for x in prefix.cache_layers[:d]]
    merged = _merged_cache(model, seq, prefix_slots + [cache_b.layers[k] for k in range(L - d)])
    ext_mask = torch.cat(
        [attention_mask, torch.ones(1, 1, dtype=attention_mask.dtype, device=attention_mask.device)],
        dim=1,
    )
    full_wrapper = _get_rep_wrapper(model, seq)
    eos = tokenizer.eos_token_id
    if eos is not None and int(first_tok[0, 0]) == int(eos):
        cont = first_tok
    else:
        with full_wrapper._apply_layer_config():
            with _guard():
                gen = model.generate(
                    input_ids=first_tok, attention_mask=ext_mask, past_key_values=merged,
                    max_new_tokens=max(1, max_new_tokens - 1), do_sample=False, pad_token_id=pad_id,
                )
        cont = gen.sequences if hasattr(gen, "sequences") else gen
    if measure_latency and torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    tail = cont[0]
    text = tokenizer.decode(tail, skip_special_tokens=True)
    return {
        "text": text, "latency_s": latency, "anchor_id": anchor.id,
        "sigma_max": None, "_cont_ids": tail.detach().cpu(),
    }

def execute_from_prefix(
    model, tokenizer, enc, anchor: Anchor, *, prefix: SharedPrefix,
    num_layers: int, max_new_tokens: int, measure_latency: bool = True,
) -> dict:
    """Execute a prefix-compatible anchor (`BASE` or `REPEAT`) reusing `prefix`."""
    if not USE_PREFIX_SHARING:
        raise RuntimeError("prefix sharing is disabled (SELFMORPH_PREFIX_SHARING=0).")
    d = _prefix_start_depth(anchor, prefix, num_layers)
    if d is None:
        raise ValueError(
            f"anchor {anchor.id} ({anchor.type}) is not prefix-compatible with depth {prefix.depth}"
        )
    seq = (
        _cached_repeat_layer_indices(num_layers, anchor.layer_span, anchor.r)
        if anchor.type == "REPEAT"
        else list(range(num_layers))
    )
    return _execute_from_prefix_impl(
        model, tokenizer, enc, anchor, prefix=prefix, seq=seq, d=d,
        max_new_tokens=max_new_tokens, measure_latency=measure_latency,
    )

def execute_rep_with_prefix(
    model, tokenizer, enc, anchor: Anchor, *, prefix: SharedPrefix,
    num_layers: int, action_cfg: dict, max_new_tokens: int, measure_latency: bool = True,
) -> dict:
    """Execute a `REPEAT` anchor reusing `prefix` through span end `j`."""
    if anchor.type != "REPEAT":
        raise ValueError("execute_rep_with_prefix only handles REPEAT anchors")
    return execute_from_prefix(
        model, tokenizer, enc, anchor, prefix=prefix,
        num_layers=num_layers, max_new_tokens=max_new_tokens, measure_latency=measure_latency,
    )

_REP_LAYER_INDICES_CACHE: dict[tuple, list[int]] = {}
_SIGMA_SCHEDULE_CACHE: dict[tuple, list[float]] = {}
_BRANCH_CFG_CACHE: dict[tuple, BranchConfig] = {}
_ACTION_CFG_KEYS = (
    "branch_method", "branch_average", "branch_l2_normalize",
    "branch_scale_mode", "branch_seed", "branch_sigma_max",
)

def _action_cfg_signature(action_cfg: dict) -> tuple:
    return tuple((k, action_cfg.get(k)) for k in _ACTION_CFG_KEYS)

def _cached_repeat_layer_indices(num_layers: int, span: tuple[int, int], r: int) -> list[int]:
    key = (num_layers, span, r)
    got = _REP_LAYER_INDICES_CACHE.get(key)
    if got is None:
        got = repeat_layer_indices(num_layers, span, r)
        _REP_LAYER_INDICES_CACHE[key] = got
    return got

_WRAPPER_CACHE: dict[tuple, object] = {}
_WRAPPER_CACHE_CAP = 128

def clear_wrapper_cache() -> None:
    _WRAPPER_CACHE.clear()

def wrapper_cache_stats() -> dict:
    return {"size": len(_WRAPPER_CACHE), "cap": _WRAPPER_CACHE_CAP}

def _cache_put(key: tuple, value):
    if len(_WRAPPER_CACHE) >= _WRAPPER_CACHE_CAP:
        _WRAPPER_CACHE.clear()
    _WRAPPER_CACHE[key] = value
    return value

def _get_rep_wrapper(base_model, layer_indices: list[int]):
    key = ("rep", id(base_model), len(layer_indices), tuple(layer_indices))
    got = _WRAPPER_CACHE.get(key)
    if got is None:
        got = _cache_put(key, build_model_with_layers(base_model, layer_indices))
    return got

def _get_br_wrapper(base_model, bcfg: BranchConfig):
    key = (
        "br", id(base_model), bcfg.branch_start, bcfg.merge_at, bcfg.num_branches,
        bcfg.perturbation_strength, tuple(bcfg.perturbation_strengths or ()),
        bcfg.merge_strategy, bcfg.seed, bcfg.gaussian_l2_normalize, bcfg.gaussian_scale_mode,
    )
    got = _WRAPPER_CACHE.get(key)
    if got is None:
        got = _cache_put(key, build_branch_model(base_model, bcfg))
    return got

def linear_radial_schedule(sigma_max: float, K: int) -> list[float]:
    """[0, sigma_max/(K-1), 2*sigma_max/(K-1), ..., sigma_max]; branch 0 is unperturbed."""
    if K <= 1:
        return [0.0]
    key = (sigma_max, K)
    got = _SIGMA_SCHEDULE_CACHE.get(key)
    if got is None:
        step = sigma_max / (K - 1)
        got = [i * step for i in range(K)]
        _SIGMA_SCHEDULE_CACHE[key] = got
    return list(got)

def _build_branch_config(anchor: Anchor, action_cfg: dict, sigma_max: Optional[float]) -> BranchConfig:
    """Build a memoized `BranchConfig` for `BRANCH(span=(i, j), K)`."""
    assert anchor.type == "BRANCH"
    i, j = anchor.layer_span
    sm = sigma_max if sigma_max is not None else action_cfg["branch_sigma_max"]
    key = ("brcfg", anchor.id, anchor.layer_span, anchor.K, sm, _action_cfg_signature(action_cfg))
    got = _BRANCH_CFG_CACHE.get(key)
    if got is not None:
        return got
    sigmas = linear_radial_schedule(sm, anchor.K)
    method = action_cfg.get("branch_method", "split_merge")
    cfg = BranchConfig(
        method=method,
        branch_start=i,
        merge_at=j if method == "split_merge" else None,
        merge_strategy="mean",
        num_branches=anchor.K,
        perturbation="gaussian",
        perturbation_strength=max(sigmas) if sigmas else 0.0,
        perturbation_strengths=list(sigmas),
        branch_average=action_cfg.get("branch_average", "prob"),
        gaussian_l2_normalize=action_cfg.get("branch_l2_normalize", True),
        gaussian_scale_mode=action_cfg.get("branch_scale_mode", "relative_l2"),
        seed=action_cfg.get("branch_seed", 42),
    )
    _BRANCH_CFG_CACHE[key] = cfg
    return cfg

def execute_anchor(
    base_model,
    tokenizer,
    enc: dict,
    anchor: Anchor,
    *,
    num_layers: int,
    action_cfg: dict,
    max_new_tokens: int,
    sigma_max: Optional[float] = None,
    measure_latency: bool = True,
    prefix: Optional[SharedPrefix] = None,
    noop_result: Optional[dict] = None,
    use_br_kv_dedup: bool = True,
) -> dict:
    """Execute one morphology action (`BASE` / `REPEAT` / `BRANCH`) on an encoded sample.

    By default:
      - If `prefix` is provided (from `probe_with_prefix`) and `anchor` is
        prefix-compatible (`BASE` or `REPEAT`), shared prompt prefill layers are
        reused automatically without recomputation.
      - For `BRANCH`, lossless KV cache deduplication (`br_kv_dedup`) outside
        the branch span `[i, j)` is enabled automatically (`use_br_kv_dedup=True`).
    """
    if anchor.type == "BASE" and noop_result is not None:
        return dict(noop_result)

    if prefix is not None and USE_PREFIX_SHARING:
        d = _prefix_start_depth(anchor, prefix, num_layers)
        if d is not None:
            return execute_from_prefix(
                base_model, tokenizer, enc, anchor, prefix=prefix,
                num_layers=num_layers, max_new_tokens=max_new_tokens,
                measure_latency=measure_latency,
            )

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_len = int(input_ids.shape[1])

    if measure_latency and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    used_sigma_max = None

    if anchor.type == "BASE":
        with _guard():
            out = base_model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id,
            )

    elif anchor.type == "REPEAT":
        layer_indices = _cached_repeat_layer_indices(num_layers, anchor.layer_span, anchor.r)
        custom = _get_rep_wrapper(base_model, layer_indices)
        with _guard():
            out = custom.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id,
            )

    elif anchor.type == "BRANCH":
        used_sigma_max = sigma_max if sigma_max is not None else action_cfg["branch_sigma_max"]
        bcfg = _build_branch_config(anchor, action_cfg, used_sigma_max)
        wrapper = _get_br_wrapper(base_model, bcfg)
        dedup_ctx = (
            br_kv_dedup(K=int(anchor.K), span=(int(anchor.layer_span[0]), int(anchor.layer_span[1])))
            if use_br_kv_dedup
            else contextlib.nullcontext({})
        )
        with dedup_ctx, _guard():
            out = wrapper.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id,
            )
    else:
        raise ValueError(f"unknown anchor type: {anchor.type}")

    if measure_latency and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    seq = out.sequences if hasattr(out, "sequences") else out
    cont = seq[0][prompt_len:]
    text = tokenizer.decode(cont, skip_special_tokens=True)

    return {
        "text": text,
        "latency_s": (t1 - t0),
        "anchor_id": anchor.id,
        "sigma_max": used_sigma_max,
    }
