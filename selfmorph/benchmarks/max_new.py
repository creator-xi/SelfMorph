"""Dataset-specific generation-length budgets (`max_new_tokens`)."""

from __future__ import annotations

def oracle_max_new_tokens(dataset: str, manifest_default: int) -> int:
    """Budget used during offline oracle collection."""
    return 32

def eval_max_new_tokens(dataset: str, manifest_default: int = 32) -> int:
    """Short-budget evaluation limit shared across policies."""
    fixed = {
        "arc_challenge": 32,
        "openbookqa": 32,
        "truthfulqa_mc1": 1024,
        "hellaswag": 1024,
        "gpqa_diamond": 1024,
        "gsm8k": 1024,
        "math500": 1024,
        "twoWikiMultihop": 1024,
    }
    if dataset in fixed:
        return fixed[dataset]
    if dataset.startswith("bbh:"):
        return 32
    return manifest_default
