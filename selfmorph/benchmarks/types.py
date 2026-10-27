"""Shared benchmark types: EvalSample dataclass + the "raw_text" role
sentinel used to mark prompts that must be sent to the model as a plain
string, bypassing the chat template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

RAW_TEXT_ROLE = "raw_text"

@dataclass
class EvalSample:
    qid: str
    task: str
    messages: list[dict]
    meta: dict = field(default_factory=dict)
    candidates: Optional[list[str]] = None
    gold_index: Optional[int] = None
