"""Layer Duplication for HuggingFace Transformers Models (`REPEAT` morphology)."""

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch.nn as nn

def _get_text_layer_owner(base_model) -> tuple[object, str]:
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        return base_model.model, "layers"
    if (
        hasattr(base_model, "model")
        and hasattr(base_model.model, "language_model")
        and hasattr(base_model.model.language_model, "layers")
    ):
        return base_model.model.language_model, "layers"
    if hasattr(base_model, "language_model") and hasattr(base_model.language_model, "layers"):
        return base_model.language_model, "layers"
    raise AttributeError(
        "Could not locate text decoder layers. "
        "Expected one of: model.layers, model.language_model.layers, language_model.layers."
    )

def _rebind_accelerate_hook(original_module: nn.Module, copied_module: nn.Module) -> None:
    source_hook = getattr(original_module, "_hf_hook", None)
    if source_hook is None:
        return
    try:
        from accelerate.hooks import add_hook_to_module
    except Exception:
        return
    if hasattr(copied_module, "_hf_hook"):
        delattr(copied_module, "_hf_hook")
    if hasattr(copied_module, "_old_forward"):
        delattr(copied_module, "_old_forward")
    copied_module.forward = type(copied_module).forward.__get__(copied_module, type(copied_module))
    add_hook_to_module(copied_module, copy.copy(source_hook), append=False)

def _shallow_copy_layer(layer, new_layer_idx: int):
    """Shallow-copy a decoder layer with an updated `layer_idx` (weights shared, distinct cache slot)."""
    new_layer = copy.copy(layer)
    new_layer._modules = dict(layer._modules)

    if hasattr(layer, "self_attn"):
        new_attn = copy.copy(layer.self_attn)
        new_attn._modules = dict(layer.self_attn._modules)
        new_attn.layer_idx = new_layer_idx
        _rebind_accelerate_hook(layer.self_attn, new_attn)
        new_layer._modules["self_attn"] = new_attn

    if hasattr(layer, "linear_attn"):
        new_linear_attn = copy.copy(layer.linear_attn)
        new_linear_attn._modules = dict(layer.linear_attn._modules)
        new_linear_attn.layer_idx = new_layer_idx
        _rebind_accelerate_hook(layer.linear_attn, new_linear_attn)
        new_layer._modules["linear_attn"] = new_linear_attn

    _rebind_accelerate_hook(layer, new_layer)
    return new_layer

class LayerDuplicatedModel(nn.Module):
    """Wrapper enabling layer repetition/truncation with weight sharing."""

    def __init__(self, base_model, layer_indices: list[int]):
        super().__init__()
        self.base_model = base_model
        self.layer_indices = layer_indices
        self.config = base_model.config

        self._layers_owner, self._layers_attr = _get_text_layer_owner(base_model)
        self._original_layers = list(getattr(self._layers_owner, self._layers_attr))
        self._original_num_layers = len(self._original_layers)

        for idx in layer_indices:
            if idx < 0 or idx >= self._original_num_layers:
                raise ValueError(f"Layer index {idx} out of range [0, {self._original_num_layers})")

        self._new_num_layers = len(layer_indices)
        new_layer_list = [
            _shallow_copy_layer(self._original_layers[orig_idx], new_pos)
            for new_pos, orig_idx in enumerate(layer_indices)
        ]
        self._new_layers = nn.ModuleList(new_layer_list)

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    @contextmanager
    def _apply_layer_config(self):
        orig_layers = getattr(self._layers_owner, self._layers_attr)
        orig_num_layers = getattr(self.base_model.config, "num_hidden_layers", None)
        orig_layer_types = getattr(self.base_model.config, "layer_types", None)
        orig_text_cfg = getattr(self.base_model.config, "text_config", None)
        orig_text_num_layers = (
            getattr(orig_text_cfg, "num_hidden_layers", None) if orig_text_cfg is not None else None
        )
        orig_text_layer_types = (
            getattr(orig_text_cfg, "layer_types", None) if orig_text_cfg is not None else None
        )
        try:
            setattr(self._layers_owner, self._layers_attr, self._new_layers)
            if orig_num_layers is not None:
                self.base_model.config.num_hidden_layers = self._new_num_layers
            if orig_layer_types is not None:
                if len(orig_layer_types) == self._original_num_layers:
                    self.base_model.config.layer_types = [orig_layer_types[i] for i in self.layer_indices]
                else:
                    self.base_model.config.layer_types = list(orig_layer_types)
            if orig_text_cfg is not None and orig_text_num_layers is not None:
                orig_text_cfg.num_hidden_layers = self._new_num_layers
            if orig_text_cfg is not None and orig_text_layer_types is not None:
                if len(orig_text_layer_types) == self._original_num_layers:
                    orig_text_cfg.layer_types = [orig_text_layer_types[i] for i in self.layer_indices]
                else:
                    orig_text_cfg.layer_types = list(orig_text_layer_types)
            yield
        finally:
            setattr(self._layers_owner, self._layers_attr, orig_layers)
            if orig_num_layers is not None:
                self.base_model.config.num_hidden_layers = orig_num_layers
            if orig_layer_types is not None:
                self.base_model.config.layer_types = orig_layer_types
            if orig_text_cfg is not None and orig_text_num_layers is not None:
                orig_text_cfg.num_hidden_layers = orig_text_num_layers
            if orig_text_cfg is not None and orig_text_layer_types is not None:
                orig_text_cfg.layer_types = orig_text_layer_types

    def generate(self, *args, **kwargs):
        kwargs.pop("past_key_values", None)
        with self._apply_layer_config():
            return self.base_model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
            cache = kwargs["past_key_values"]
            if hasattr(cache, "key_cache"):
                cache_layers = len(cache.key_cache)
            elif isinstance(cache, tuple):
                cache_layers = len(cache)
            else:
                cache_layers = 0
            if cache_layers != self._new_num_layers:
                kwargs["past_key_values"] = None
        with self._apply_layer_config():
            return self.base_model(*args, **kwargs)

def build_model_with_layers(model, layer_indices: list[int]) -> LayerDuplicatedModel:
    """Wrap `model` so its forward/generate use `layer_indices` as the decoder layer sequence."""
    return LayerDuplicatedModel(model, layer_indices)
