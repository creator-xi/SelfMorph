"""Dataset loaders for the evaluated benchmarks.

In-distribution benchmarks (5):
  - `bbh:<subtask>` (BIG-Bench Hard, 19 subtasks)
  - `arc_challenge` (ARC-Challenge)
  - `openbookqa`    (OpenBookQA)
  - `gsm8k`         (GSM8K)
  - `math500`       (MATH-500)

Unseen benchmarks (4):
  - `twoWikiMultihop` (2WikiMultiHopQA)
  - `hellaswag`       (HellaSwag)
  - `truthfulqa_mc1`  (TruthfulQA-MC1)
  - `gpqa_diamond`    (GPQA-Diamond)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from selfmorph.benchmarks.prompts import (
    ARC_FEW_SHOT, DIRECT_OPTION_INSTRUCTION, OBQA_FEW_SHOT, TQA_FEW_SHOT,
    TQA_MAX_LETTERS, build_bbh_direct_prompt, mc_prompt,
    reinforced_direct_answer_prompt, reinforced_direct_boxed_prompt, tqa_letters,
)
from selfmorph.benchmarks.types import RAW_TEXT_ROLE, EvalSample
from selfmorph.settings import dataset_root, eval_dataset_root

_EXTRACTIVE_QA_SYSTEM = (
    "Answer the question with a short phrase (a single entity, a short noun phrase, "
    "or 'yes'/'no'). Use the provided context."
)
MULTIHOP_CONTEXT_CHAR_BUDGET = 8000
_SHUFFLE_SEED_BASE = 20260810

def _read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path)

def _data_root() -> Path:
    return eval_dataset_root() or dataset_root()

def load_arc_challenge(*, split: str = "test", max_samples: Optional[int] = None) -> list[EvalSample]:
    df = _read_parquet(_data_root() / "arc_challenge" / f"{split}.parquet")
    out: list[EvalSample] = []
    for i, row in df.iterrows():
        if max_samples is not None and i >= max_samples:
            break
        qid = str(row["id"])
        question = str(row["question"])
        choices = row["choices"]
        texts = list(choices["text"])
        labels = [str(L) for L in choices["label"]]
        gold_letter = str(row["answerKey"]).upper()
        if gold_letter not in labels:
            continue
        prompt = mc_prompt(question, texts, labels, ARC_FEW_SHOT)
        out.append(EvalSample(
            qid=qid, task="arc_challenge",
            messages=[{"role": RAW_TEXT_ROLE, "content": prompt}],
            meta={"target": f"({gold_letter})", "question": question,
                  "options": texts, "letters": labels, "gold_letter": gold_letter},
        ))
    return out

def load_openbookqa(*, split: str = "test", max_samples: Optional[int] = None) -> list[EvalSample]:
    df = _read_parquet(_data_root() / "openbookqa" / f"{split}.parquet")
    out: list[EvalSample] = []
    for i, row in df.iterrows():
        if max_samples is not None and i >= max_samples:
            break
        qid = str(row["id"])
        question = str(row["question_stem"])
        choices = row["choices"]
        texts = list(choices["text"])
        labels = [str(L) for L in choices["label"]]
        gold_letter = str(row["answerKey"]).upper()
        if gold_letter not in labels:
            continue
        prompt = mc_prompt(question, texts, labels, OBQA_FEW_SHOT)
        out.append(EvalSample(
            qid=qid, task="openbookqa",
            messages=[{"role": RAW_TEXT_ROLE, "content": prompt}],
            meta={"target": f"({gold_letter})", "question": question,
                  "options": texts, "letters": labels, "gold_letter": gold_letter},
        ))
    return out

def _parse_gsm8k_answer(answer_field: str) -> Optional[float]:
    if "####" not in answer_field:
        return None
    tail = answer_field.split("####")[-1].strip().replace(",", "")
    try:
        return float(tail)
    except ValueError:
        return None

def load_gsm8k(*, split: str = "test", max_samples: Optional[int] = None) -> list[EvalSample]:
    df = _read_parquet(_data_root() / "gsm8k" / f"{split}.parquet")
    out: list[EvalSample] = []
    for i, row in df.iterrows():
        if max_samples is not None and i >= max_samples:
            break
        q = str(row["question"])
        ans = _parse_gsm8k_answer(str(row["answer"]))
        if ans is None:
            continue
        out.append(EvalSample(
            qid=f"gsm8k_{split}_{i}", task="gsm8k",
            messages=[{"role": RAW_TEXT_ROLE, "content": reinforced_direct_answer_prompt(q)}],
            meta={"answer": ans, "question": q, "raw_answer": str(row["answer"])},
        ))
    return out

def load_math500(*, split: str = "test", max_samples: Optional[int] = None) -> list[EvalSample]:
    df = _read_parquet(_data_root() / "math500" / f"{split}.parquet")
    out: list[EvalSample] = []
    for i, row in df.iterrows():
        if max_samples is not None and i >= max_samples:
            break
        q, gold = str(row["problem"]), str(row["answer"])
        out.append(EvalSample(
            qid=str(row["unique_id"]), task="math500",
            messages=[{"role": RAW_TEXT_ROLE, "content": reinforced_direct_boxed_prompt(q)}],
            meta={"answer_str": gold, "question": q,
                  "subject": str(row.get("subject", "")), "level": int(row.get("level", 0))},
        ))
    return out

def _resolve_bbh_path(subtask: str) -> Path:
    return eval_dataset_root() / "bbh_full" / subtask / "test-00000-of-00001.parquet"

def load_bbh_subtask(subtask: str, *, split: Optional[str] = None, max_samples: Optional[int] = None) -> list[EvalSample]:
    df = _read_parquet(_resolve_bbh_path(subtask))
    prefix = build_bbh_direct_prompt(subtask)
    out: list[EvalSample] = []
    for i, row in df.iterrows():
        if max_samples is not None and i >= max_samples:
            break
        question = str(row.get("input", row.get("question", "")))
        target = str(row.get("target", row.get("answer", ""))).strip()
        user_text = f"{prefix}\n\nQ: {question}\nA:"
        out.append(EvalSample(
            qid=f"{subtask}_{i}", task=f"bbh:{subtask}",
            messages=[{"role": RAW_TEXT_ROLE, "content": user_text}],
            meta={"target": target, "question": question},
        ))
    return out

def load_bbh(subtasks: list[str], *, max_samples_per_task: Optional[int] = None) -> list[EvalSample]:
    out: list[EvalSample] = []
    for sub in subtasks:
        out.extend(load_bbh_subtask(sub, max_samples=max_samples_per_task))
    return out

def _coerce_sentence_list(sents) -> list[str]:
    if hasattr(sents, "tolist"):
        sents = sents.tolist()
    if isinstance(sents, str):
        stripped = sents.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (json.JSONDecodeError, ValueError):
                pass
        return [sents]
    if isinstance(sents, list):
        return [str(x) for x in sents]
    return [str(sents)]

def _join_sentences(sents: list[str]) -> str:
    return " ".join(s.strip() for s in sents if s.strip())

def _coerce_title(title) -> str:
    s = str(title)
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            return str(json.loads(s))
        except (json.JSONDecodeError, ValueError):
            return s[1:-1]
    return s

def _format_2wiki_context(context) -> str:
    parts = []
    for item in context:
        item = item.tolist() if hasattr(item, "tolist") else item
        if len(item) >= 2:
            title, sents = _coerce_title(item[0]), item[1]
            body = _join_sentences(_coerce_sentence_list(sents))
            parts.append(f"[{title}] {body}")
    return "\n".join(parts)

def load_two_wiki_multihop(*, split: str = "validation", max_samples: Optional[int] = None) -> list[EvalSample]:
    df = _read_parquet(_data_root() / "twoWikiMultihop" / f"{split}.parquet")
    out: list[EvalSample] = []
    for i, row in df.iterrows():
        if max_samples is not None and i >= max_samples:
            break
        ctx = _format_2wiki_context(row["context"])
        if len(ctx) > MULTIHOP_CONTEXT_CHAR_BUDGET:
            ctx = ctx[:MULTIHOP_CONTEXT_CHAR_BUDGET] + "..."
        q, a = str(row["question"]), str(row["answer"]).strip()
        prompt = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        out.append(EvalSample(
            qid=str(row["_id"]), task="twoWikiMultihop",
            messages=[{"role": "system", "content": _EXTRACTIVE_QA_SYSTEM}, {"role": "user", "content": prompt}],
            meta={"answer": a, "question": q},
        ))
    return out

def load_truthfulqa_mc1(*, max_samples: Optional[int] = None, shuffle_seed_base: int = 20260709) -> list[EvalSample]:
    path = eval_dataset_root() / "truthful_qa" / "multiple_choice" / "validation-00000-of-00001.parquet"
    df = _read_parquet(path)
    out: list[EvalSample] = []
    for i, row in df.iterrows():
        if max_samples is not None and i >= max_samples:
            break
        question = str(row["question"])
        mc1 = row["mc1_targets"]
        choices = list(mc1["choices"])
        labels = [int(x) for x in mc1["labels"]]
        if 1 not in labels:
            continue
        orig_gold = labels.index(1)
        n = len(choices)
        if n > TQA_MAX_LETTERS:
            continue

        rng = random.Random(shuffle_seed_base + int(i))
        perm = list(range(n))
        rng.shuffle(perm)
        shuffled = [choices[perm[k]] for k in range(n)]
        new_gold = perm.index(orig_gold)
        letters = tqa_letters(n)
        gold_letter = letters[new_gold]

        prompt = mc_prompt(question, shuffled, letters, TQA_FEW_SHOT, instruction=DIRECT_OPTION_INSTRUCTION)
        out.append(EvalSample(
            qid=f"tqa_{i}", task="truthfulqa_mc1",
            messages=[{"role": RAW_TEXT_ROLE, "content": prompt}],
            meta={
                "target": f"({gold_letter})", "gold_letter": gold_letter, "question": question,
                "options": shuffled, "letters": letters,
                "shuffle_seed": shuffle_seed_base + int(i), "orig_gold_index": orig_gold, "shuffle_perm": perm,
            },
            candidates=shuffled, gold_index=new_gold,
        ))
    return out

def load_hellaswag(*, max_samples: Optional[int] = None) -> list[EvalSample]:
    df = _read_parquet(eval_dataset_root() / "hellaswag" / "validation.parquet")
    out: list[EvalSample] = []
    for i, row in df.iterrows():
        if max_samples is not None and i >= max_samples:
            break
        ctx = str(row["ctx"])
        endings = [str(e) for e in row["endings"]]
        label = int(row["label"])
        if not (0 <= label < len(endings)):
            continue
        qid = str(row["ind"])
        rng = random.Random(_SHUFFLE_SEED_BASE + i)
        perm = list(range(len(endings)))
        rng.shuffle(perm)
        shuffled = [endings[p] for p in perm]
        new_gold = perm.index(label)
        letters = [chr(ord("A") + k) for k in range(len(shuffled))]
        gold_letter = letters[new_gold]
        prompt = mc_prompt(ctx, shuffled, letters, instruction=DIRECT_OPTION_INSTRUCTION)
        out.append(EvalSample(
            qid=qid, task="hellaswag",
            messages=[{"role": RAW_TEXT_ROLE, "content": prompt}],
            meta={"target": f"({gold_letter})", "question": ctx, "options": shuffled,
                  "letters": letters, "gold_letter": gold_letter,
                  "orig_gold_index": label, "shuffle_perm": perm,
                  "activity_label": str(row["activity_label"])},
        ))
    return out

def load_gpqa_diamond(*, max_samples: Optional[int] = None) -> list[EvalSample]:
    import pandas as pd

    df = pd.read_csv(eval_dataset_root() / "gpqa_diamond" / "gpqa_diamond.csv")
    out: list[EvalSample] = []
    for i, row in df.iterrows():
        if max_samples is not None and i >= max_samples:
            break
        question = str(row["Question"])
        correct = str(row["Correct Answer"])
        distractors = [str(row[f"Incorrect Answer {k}"]) for k in (1, 2, 3)]
        distractors = [d for d in distractors if d and d.lower() != "nan"]
        if len(distractors) != 3:
            continue
        qid = f"gpqa_diamond_{i}"
        rng = random.Random(_SHUFFLE_SEED_BASE + i)
        perm = list(range(4))
        rng.shuffle(perm)
        pool = [correct] + distractors
        shuffled = [pool[p] for p in perm]
        new_gold = perm.index(0)
        letters = ["A", "B", "C", "D"]
        gold_letter = letters[new_gold]
        prompt = mc_prompt(question, shuffled, letters, instruction=DIRECT_OPTION_INSTRUCTION)
        out.append(EvalSample(
            qid=qid, task="gpqa_diamond",
            messages=[{"role": RAW_TEXT_ROLE, "content": prompt}],
            meta={"target": f"({gold_letter})", "question": question, "options": shuffled,
                  "letters": letters, "gold_letter": gold_letter,
                  "subdomain": str(row.get("Subdomain", ""))},
        ))
    return out

_LOADERS = {
    "arc_challenge": load_arc_challenge,
    "openbookqa": load_openbookqa,
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "twoWikiMultihop": load_two_wiki_multihop,
    "truthfulqa_mc1": load_truthfulqa_mc1,
    "hellaswag": load_hellaswag,
    "gpqa_diamond": load_gpqa_diamond,
}

def load_benchmark(name: str, **kwargs) -> list[EvalSample]:
    """Load one benchmark by name, or `bbh:<subtask>`."""
    if name in _LOADERS:
        return _LOADERS[name](**kwargs)
    if name.startswith("bbh:"):
        return load_bbh_subtask(name.split(":", 1)[1], **kwargs)
    raise ValueError(f"Unknown dataset name: {name!r}")
