from __future__ import annotations

import re
from typing import Optional

from selfmorph.benchmarks.scorers import (  # noqa: F401
    BBH_YESNO_SUBTASKS,
    _score_bbh_yesno,
    _strip_thinking as strip_thinking,
)

_PRODUCTION_PAREN_RE = re.compile(r"\(\s*([A-Za-z])\s*\)")
_PRODUCTION_LINE_HEAD_RE = re.compile(r"^\(?\s*([A-Za-z])\b")

def extract_letter_fulltext(text: str) -> Optional[str]:
    """Production ``score_bbh`` letter rule applied to the whole output.

    Returns the lowercase letter, or None when the rule finds no match.
    """
    t = strip_thinking(text).strip()
    if not t:
        return None
    m = _PRODUCTION_PAREN_RE.search(t) or _PRODUCTION_LINE_HEAD_RE.match(t)
    return m.group(1).lower() if m is not None else None

def score_bbh_fulltext(output: str, gold: str) -> float:
    """Production ``score_bbh`` with 'first line' -> 'whole text'."""
    gold = gold.strip().lower().strip("() ")
    pred = extract_letter_fulltext(output)
    return float(pred is not None and pred == gold)

def score_letter_fulltext(text: str, gold_letter: str) -> float:
    """Whole-output MC/BBH-letter scoring (gold is a single letter)."""
    gold = gold_letter.strip().lower().strip("() ")
    if not gold:
        return 0.0
    return score_bbh_fulltext(text, gold)

def score_bbh_yesno_fulltext(text: str, sample) -> float:
    """Yes/no score over the full generation."""
    return float(_score_bbh_yesno(text, sample))

def score_sample_fulltext(text: str, sample, ds_name: str) -> float:
    """Dispatcher: production scorer + whole-text letter parse."""
    if ds_name.startswith("bbh:"):
        sub = ds_name.split(":", 1)[1]
        if sub in BBH_YESNO_SUBTASKS:
            return score_bbh_yesno_fulltext(text, sample)
        return score_bbh_fulltext(text, str(sample.meta.get("target", "")))
    gold = str(sample.meta.get("gold_letter", "")).strip()
    if not gold:
        return 0.0
    return score_bbh_fulltext(text, gold)

__all__ = [
    "extract_letter_fulltext",
    "score_bbh_fulltext",
    "score_bbh_yesno_fulltext",
    "score_letter_fulltext",
    "score_sample_fulltext",
]
