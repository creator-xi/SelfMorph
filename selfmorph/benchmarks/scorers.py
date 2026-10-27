"""Scoring functions for the evaluated benchmarks.

Covers:
  - Multiple-choice letter accuracy (`arc_challenge`, `openbookqa`, `hellaswag`,
    `truthfulqa_mc1`, `gpqa_diamond`, and letter-style `bbh:*` subtasks);
  - BBH affirmative/negative keyword & letter cascade (`formal_fallacies`,
    `sports_understanding`);
  - GSM8K numeric extraction and comparison (`|pred - gold| < 1e-3`);
  - MATH-500 brace-matched `\\boxed{}` extraction with Hendrycks normalization;
  - 2WikiMultiHop normalized exact match with token-level F1 fallback.
"""

from __future__ import annotations

import re
from typing import Any, Optional

def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def score_bbh(output: str, gold: str) -> float:
    """Score a `(X)` first-line letter answer against a gold letter/target."""
    gold = gold.strip().lower().strip("() ")
    lines = [line.strip() for line in _strip_thinking(output).splitlines() if line.strip()]
    if not lines:
        return 0.0
    first = lines[0]
    match = re.search(r"\(\s*([A-Za-z])\s*\)", first) or re.match(r"^\(?\s*([A-Za-z])\b", first)
    return float(match is not None and match.group(1).lower() == gold)

def score_math(output: str, answer: float) -> float:
    """Last-number extraction + tolerance compare."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", _strip_thinking(output))
    if not matches:
        return 0.0
    try:
        return float(abs(float(matches[-1]) - float(answer)) < 1e-3)
    except ValueError:
        return 0.0

def _score_mc(text: str, sample) -> float:
    gold = sample.meta.get("gold_letter", "").upper().strip()
    if not gold:
        return 0.0
    return float(score_bbh(text, f"({gold})"))

_ANSWER_IS_NUM_RE = re.compile(r"the\s+answer\s+is\s*[:=]?\s*[\*\$]*\s*(-?[0-9][0-9\.,]*)", re.IGNORECASE)
_LAST_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")

def _extract_last_boxed(text: str) -> Optional[str]:
    """Return the content of the last `\\boxed{...}` in `text` (brace-matched)."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        alt = re.search(r"\\boxed\s+([^\s$]+)", text)
        return alt.group(1) if alt else None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start - 1, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None

def _extract_final_number(text: str) -> Optional[float]:
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    t = re.sub(r"</?think>", "", t).strip()
    if not t:
        return None

    def _clean(s: str) -> Optional[float]:
        s = s.strip().rstrip(".").replace(",", "").replace("$", "")
        try:
            return float(s)
        except ValueError:
            return None

    b = _extract_last_boxed(t)
    if b is not None:
        v = _clean(b)
        if v is not None:
            return v

    matches = _ANSWER_IS_NUM_RE.findall(t)
    if matches:
        v = _clean(matches[-1])
        if v is not None:
            return v

    for line in reversed(t.splitlines()):
        line = line.strip()
        if not line:
            continue
        nums = _LAST_NUM_RE.findall(line)
        if nums:
            v = _clean(nums[-1])
            if v is not None:
                return v
        break

    return None

def _score_arithmetic(text: str, sample) -> float:
    try:
        gold = float(sample.meta["answer"])
    except (KeyError, ValueError, TypeError):
        return 0.0
    cut = re.search(r"\n\s*(?:q|question)\s*:", text, flags=re.IGNORECASE)
    scored_text = text[: cut.start()] if cut else text
    pred = _extract_final_number(scored_text)
    if pred is None:
        return 0.0
    return 1.0 if abs(pred - gold) < 1e-3 else 0.0

def _fix_fracs(s: str) -> str:
    substrs = s.split("\\frac")
    new_str = substrs[0]
    for sub in substrs[1:]:
        new_str += "\\frac"
        if sub.startswith("{"):
            new_str += sub
        else:
            if len(sub) < 2:
                return s
            a, b, rest = sub[0], sub[1], sub[2:]
            if b != "{":
                new_str += "{" + a + "}{" + b + "}" + rest
            else:
                new_str += "{" + a + "}" + b + rest
    return new_str

def _fix_a_slash_b(s: str) -> str:
    if len(s.split("/")) != 2:
        return s
    a, b = s.split("/")
    try:
        int(a)
        int(b)
        assert s == f"{int(a)}/{int(b)}"
        return f"\\frac{{{int(a)}}}{{{int(b)}}}"
    except (ValueError, AssertionError):
        return s

def _fix_sqrt(s: str) -> str:
    if "\\sqrt" not in s:
        return s
    splits = s.split("\\sqrt")
    new_str = splits[0]
    for sub in splits[1:]:
        if sub and sub[0] != "{":
            new_str += "\\sqrt{" + sub[0] + "}" + sub[1:]
        else:
            new_str += "\\sqrt" + sub
    return new_str

def _strip_string(s: str) -> str:
    s = s.replace("\n", "")
    s = s.replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac")
    s = s.replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("$", "")
    if "\\text{" in s:
        s = re.sub(r"\\text\{[^{}]*\}", "", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.")
    s = s.replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    if "=" in s and len(s.split("=")) == 2:
        left, right = s.split("=")
        if len(right.strip()) >= 1:
            s = right
    s = _fix_sqrt(s)
    s = _fix_fracs(s)
    s = s.replace(" ", "")
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    return s

def _score_math500(text: str, sample) -> float:
    gold = str(sample.meta.get("answer_str", "")).strip()
    if not gold:
        return 0.0
    cut = re.search(r"\n\s*(?:q|question)\s*:", text, flags=re.IGNORECASE)
    scored_text = text[: cut.start()] if cut else text
    boxed = _extract_last_boxed(scored_text)
    if boxed is None:
        m = re.search(r"the answer is\s*[:=]?\s*([^\n.]+)", scored_text, re.IGNORECASE)
        if m:
            boxed = m.group(1).strip()
    if boxed is None:
        return 0.0
    if _strip_string(boxed) == _strip_string(gold):
        return 1.0
    try:
        return 1.0 if abs(float(boxed) - float(gold)) < 1e-6 else 0.0
    except (ValueError, TypeError):
        return 0.0

_ARTICLES_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")

def _normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = _ARTICLES_RE.sub("", s)
    s = _PUNCT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def _f1(pred: str, gold: str) -> float:
    p_tokens = _normalize_answer(pred).split()
    g_tokens = _normalize_answer(gold).split()
    if not p_tokens or not g_tokens:
        return 1.0 if p_tokens == g_tokens else 0.0
    common = {t: min(p_tokens.count(t), g_tokens.count(t)) for t in p_tokens}
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)

def _extract_short_answer(text: str) -> str:
    for line in text.strip().splitlines():
        s = line.strip()
        if s:
            return s
    return text.strip()

def _score_extractive(text: str, sample) -> float:
    gold = str(sample.meta.get("answer", "")).strip()
    if not gold:
        return 0.0
    pred = _extract_short_answer(text)
    if _normalize_answer(pred) == _normalize_answer(gold):
        return 1.0
    return _f1(pred, gold)

BBH_YESNO_SUBTASKS = {
    "formal_fallacies",
    "sports_understanding",
}

_BBH_YESNO_CANON: dict[str, dict[str, Any]] = {
    "yes": {"letter": "y", "neg_letter": "n",
            "words": [r"\byes\b", r"\bplausible\b"],
            "neg_words": [r"\bno\b", r"\bimplausible\b", r"not\s+plausible"]},
    "no": {"letter": "n", "neg_letter": "y",
           "words": [r"\bno\b", r"\bimplausible\b", r"not\s+plausible"],
           "neg_words": [r"\byes\b", r"\bplausible\b"]},
    "true": {"letter": "t", "neg_letter": "f", "words": [r"\btrue\b"], "neg_words": [r"\bfalse\b"]},
    "false": {"letter": "f", "neg_letter": "t", "words": [r"\bfalse\b"], "neg_words": [r"\btrue\b"]},
    "valid": {"letter": "v", "neg_letter": "i",
              "words": [r"deductively\s+valid", r"\bvalid\b"], "neg_words": [r"\binvalid\b"]},
    "invalid": {"letter": "i", "neg_letter": "v",
                "words": [r"\binvalid\b"], "neg_words": [r"deductively\s+valid", r"\bvalid\b"]},
}

def _score_bbh_yesno(text: str, sample) -> float:
    target = str(sample.meta.get("target", "")).strip().lower()
    if not target:
        return 0.0
    canon = _BBH_YESNO_CANON.get(target)
    if canon is None:
        t0 = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().lower()
        return 1.0 if target in t0 else 0.0

    txt = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).lower()
    m_q = re.search(r"\n\s*(?:q|question)\s*:", txt)
    if m_q:
        txt = txt[: m_q.start()]
    txt = txt.strip()
    if not txt:
        return 0.0

    for m in re.finditer(r"\(\s*([a-z])\s*\)", txt):
        letter = m.group(1)
        if letter == canon["letter"]:
            return 1.0
        if letter == canon["neg_letter"]:
            return 0.0

    m = re.search(r"(?:the\s+answer\s+is|answer\s*[:=])\s*[\*\(\s]*([a-z]+)", txt)
    if m:
        w = m.group(1)
        if w == canon["letter"] or any(re.fullmatch(p.strip(r"\b"), w) for p in canon["words"]):
            return 1.0
        if w == canon["neg_letter"] or any(re.fullmatch(p.strip(r"\b"), w) for p in canon["neg_words"]):
            return 0.0

    ok_pos = None
    for pat in canon["words"]:
        m2 = re.search(pat, txt)
        if m2 is not None and (ok_pos is None or m2.start() < ok_pos):
            ok_pos = m2.start()
    neg_pos = None
    for pat in canon["neg_words"]:
        m2 = re.search(pat, txt)
        if m2 is not None and (neg_pos is None or m2.start() < neg_pos):
            neg_pos = m2.start()
    if ok_pos is not None and neg_pos is not None:
        return 1.0 if ok_pos < neg_pos else 0.0
    if ok_pos is not None:
        return 1.0
    if neg_pos is not None:
        return 0.0
    return 0.0

_SCORERS: dict[str, Any] = {
    "arc_challenge": _score_mc,
    "openbookqa": _score_mc,
    "truthfulqa_mc1": _score_mc,
    "hellaswag": _score_mc,
    "gpqa_diamond": _score_mc,
    "gsm8k": _score_arithmetic,
    "math500": _score_math500,
    "twoWikiMultihop": _score_extractive,
}
for _sub in BBH_YESNO_SUBTASKS:
    _SCORERS[f"bbh:{_sub}"] = _score_bbh_yesno

def score_sample(text: str, sample, ds_name: str) -> float:
    """Score one sample for its benchmark."""
    if ds_name in _SCORERS:
        return float(_SCORERS[ds_name](text, sample))
    if ds_name.startswith("bbh:"):
        return score_bbh(text, str(sample.meta.get("target", "")))
    raise ValueError(f"No scorer registered for dataset {ds_name!r}")

__all__ = ["score_sample", "score_bbh", "score_math"]
