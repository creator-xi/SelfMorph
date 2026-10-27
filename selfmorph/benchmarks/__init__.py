"""Benchmark loading + scoring: single entry point for the rest of the
pipeline (oracle.py, final_eval.py, span_scan.py all go through here).
"""

from __future__ import annotations

from selfmorph.benchmarks.datasets import load_benchmark
from selfmorph.benchmarks.max_new import eval_max_new_tokens, oracle_max_new_tokens
from selfmorph.benchmarks.prompts import model_wants_qwen_think_seed, render_prompt
from selfmorph.benchmarks.scorers import score_sample
from selfmorph.benchmarks.types import EvalSample
from selfmorph.settings import load_runtime_config

__all__ = [
    "load_benchmark", "score_sample", "render_prompt", "model_wants_qwen_think_seed",
    "oracle_max_new_tokens", "eval_max_new_tokens", "EvalSample", "paper_domain",
]

def paper_domain(dataset: str, config: dict | None = None) -> str:
    """Map a raw dataset name (e.g. "gsm8k", "bbh:formal_fallacies") to its
    reporting domain (e.g. "Mathematics", "BBH Reasoning"), per
    config/runtime.json's domains.dataset_domain_map.
    """
    config = config or load_runtime_config()
    dataset_map = config["domains"]["dataset_domain_map"]
    key = dataset.split(":", 1)[0] if dataset.startswith("bbh:") else dataset
    if key not in dataset_map:
        raise KeyError(f"No paper_domain mapping for dataset {dataset!r}; add one to config/runtime.json")
    return dataset_map[key]
