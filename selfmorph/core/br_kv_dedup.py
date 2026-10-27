"""Lossless deduplication of identical BRANCH KV-cache rows outside the branch span.

For a `BRANCH` morphology with K branches and half-open span `[i, j)`, the
decoder stack runs at batch size `K * B`. For standard-attention layers outside
the active branch span `[i, j)`, the K replica rows process identical inputs
before branching and identical merged states after branching, so their KV cache
histories are bit-identical across replicas.

This module deduplicates those layers' caches by storing a single physical row
(`[1, ...]`) and exposing `K` logical rows (`[K, ...]`) to attention via a
zero-stride broadcast view (`expand`), while layers inside `[i, j)` and layers
with in-place recurrent state updates retain their full `K` rows. This is
enabled by default during `BRANCH` execution.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch

def enabled() -> bool:
    """Return True if lossless branch KV cache deduplication is enabled (default: True)."""
    return os.environ.get("SELFMORPH_BR_KV_DEDUP", "1") == "1"

_ORIG: Optional[tuple] = None
_ACTIVE = False
_CUR_LAYER = -1
_K = 0
_LO = 0
_HI = 0
STATS: dict = {"fired": 0, "skipped_outside": 0}

def _dedup_body(self, key_states, value_states, sliding, orig, args, kwargs):
    """Store a single physical cache row and return a K-row broadcast view to attention."""
    K = _K
    if not self.is_initialized:
        full_k, full_v = orig(self, key_states, value_states, *args, **kwargs)
        src_k = self.keys if sliding else full_k
        src_v = self.values if sliding else full_v
        ret_k, ret_v = full_k, full_v
        base_k = src_k[0:1].clone().contiguous()
        base_v = src_v[0:1].clone().contiguous()
    else:
        ret_k = torch.cat([self._mk, key_states[0:1]], dim=-2)
        ret_v = torch.cat([self._mv, value_states[0:1]], dim=-2)
        if sliding:
            base_k = ret_k[:, :, -(sliding - 1):, :].contiguous()
            base_v = ret_v[:, :, -(sliding - 1):, :].contiguous()
        else:
            base_k, base_v = ret_k.contiguous(), ret_v.contiguous()
    self._mk, self._mv = base_k, base_v
    self.keys, self.values = base_k, base_v
    return ret_k.expand(K, *ret_k.shape[1:]), ret_v.expand(K, *ret_v.shape[1:])

def _eligible() -> bool:
    """Layers outside `[lo, hi)` hold identical rows across replicas and can be deduplicated."""
    return _ACTIVE and _K > 1 and (_CUR_LAYER < _LO or _CUR_LAYER >= _HI)

@contextmanager
def br_kv_dedup(*, K: int, span: tuple[int, int]) -> Iterator[dict]:
    """Context manager that deduplicates the KV cache batch dimension outside `span = (i, j)`."""
    global _ORIG, _ACTIVE, _CUR_LAYER, _K, _LO, _HI, STATS
    import transformers.cache_utils as cu

    if not enabled():
        yield {"fired": 0, "skipped_outside": 0}
        return

    if _ORIG is None:
        _ORIG = (cu.Cache.update, cu.DynamicLayer.update, cu.DynamicSlidingWindowLayer.update)
    orig_cache, orig_dyn, orig_slide = _ORIG

    _K, _LO, _HI = int(K), int(span[0]), int(span[1])
    _CUR_LAYER = -1
    _ACTIVE = True
    STATS = {"fired": 0, "skipped_outside": 0}

    def cache_update(self, key_states, value_states, layer_idx, *args, **kwargs):
        global _CUR_LAYER
        _CUR_LAYER = layer_idx
        return orig_cache(self, key_states, value_states, layer_idx, *args, **kwargs)

    def dyn_update(self, key_states, value_states, *args, **kwargs):
        if not _eligible():
            STATS["skipped_outside"] += 1
            return orig_dyn(self, key_states, value_states, *args, **kwargs)
        STATS["fired"] += 1
        return _dedup_body(self, key_states, value_states, None, orig_dyn, args, kwargs)

    def slide_update(self, key_states, value_states, *args, **kwargs):
        if not _eligible():
            STATS["skipped_outside"] += 1
            return orig_slide(self, key_states, value_states, *args, **kwargs)
        STATS["fired"] += 1
        if self.is_initialized:
            self.cumulative_length += key_states.shape[-2]
        w = int(getattr(self, "sliding_window", 0) or 0)
        return _dedup_body(self, key_states, value_states, w if w > 0 else None, orig_slide, args, kwargs)

    cu.Cache.update = cache_update
    cu.DynamicLayer.update = dyn_update
    cu.DynamicSlidingWindowLayer.update = slide_update
    try:
        yield STATS
    finally:
        _ACTIVE = False
        cu.Cache.update, cu.DynamicLayer.update, cu.DynamicSlidingWindowLayer.update = _ORIG

def _iter_tensors(obj: Any, seen: set[int], depth: int = 0):
    """Yield every `torch.Tensor` reachable from `obj` (bounded depth)."""
    if depth > 8 or obj is None:
        return
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)
    if isinstance(obj, torch.Tensor):
        yield obj
        return
    if isinstance(obj, (list, tuple, set)):
        for x in obj:
            yield from _iter_tensors(x, seen, depth + 1)
        return
    if isinstance(obj, dict):
        for x in obj.values():
            yield from _iter_tensors(x, seen, depth + 1)
        return
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for x in d.values():
            yield from _iter_tensors(x, seen, depth + 1)
    slots = getattr(type(obj), "__slots__", None)
    if slots:
        for name in slots:
            try:
                yield from _iter_tensors(getattr(obj, name), seen, depth + 1)
            except AttributeError:
                pass

def kv_cache_bytes(cache: Any) -> int:
    """Total bytes of unique physical tensor storages inside a KV cache object."""
    if cache is None:
        return 0
    total = 0
    seen_ptrs: set[int] = set()
    for t in _iter_tensors(cache, set()):
        try:
            ptr = t.untyped_storage().data_ptr()
            nbytes = t.untyped_storage().nbytes()
        except Exception:
            continue
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        total += int(nbytes)
    return total
