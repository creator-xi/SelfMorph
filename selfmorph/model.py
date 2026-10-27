"""Frozen base-LM loading and hidden-state capture hooks.

Model config is read from config/runtime.json's `model` section rather
than hardcoded constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfmorph.settings import load_runtime_config, model_path

@dataclass
class ModelConfig:
    model_path: str
    num_layers: int
    hidden_size: int
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = False

def load_model_config() -> ModelConfig:
    cfg = load_runtime_config()["model"]
    return ModelConfig(
        model_path=model_path(),
        num_layers=cfg["num_layers"],
        hidden_size=cfg["hidden_size"],
        torch_dtype=cfg.get("torch_dtype", "bfloat16"),
        trust_remote_code=cfg.get("trust_remote_code", False),
    )

def _resolve_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]

def load_frozen_model(device: str = "cuda", model_config: ModelConfig | None = None,
                      device_map: str | None = None):
    """Load the frozen backbone and tokenizer in eval mode."""
    model_config = model_config or load_model_config()
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_path, trust_remote_code=model_config.trust_remote_code
    )
    load_kwargs = {
        "torch_dtype": _resolve_dtype(model_config.torch_dtype),
        "trust_remote_code": model_config.trust_remote_code,
    }
    if device_map:
        load_kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_config.model_path, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.model_path, **load_kwargs).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer

def get_decoder_layers(model) -> torch.nn.ModuleList:
    """Return the decoder layer list for text-only and multimodal-unified checkpoints."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(f"Cannot locate decoder layer list on {type(model)}")

class HiddenStateCollector:
    """Forward hooks that capture the last-token hidden state at a fixed set
    of decoder layers, in a single forward pass (used by oracle.py so that
    trajectory extraction is a free byproduct of the score/cost forward,
    never a second forward pass).
    """

    def __init__(self, model, layer_ids: list[int]):
        self.layer_ids = sorted(set(layer_ids))
        self._layers = get_decoder_layers(model)
        self._handles: list[Any] = []
        self._captured: dict[int, torch.Tensor] = {}

    def __enter__(self) -> "HiddenStateCollector":
        for lid in self.layer_ids:
            handle = self._layers[lid].register_forward_hook(self._make_hook(lid))
            self._handles.append(handle)
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, layer_id: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured[layer_id] = hidden[:, -1, :].detach().to(torch.float32).cpu()

        return hook

    def collected(self) -> dict[int, torch.Tensor]:
        """Return {layer_id: last-token hidden [batch, hidden]} for the most
        recent forward pass.
        """
        return dict(self._captured)
