"""Chat-template rendering and direct-answer prompt templates."""

from __future__ import annotations

import re
from pathlib import Path

from selfmorph.benchmarks.types import RAW_TEXT_ROLE
from selfmorph.settings import PACKAGE_ROOT

QWEN_EMPTY_THINK_SEED = "<think>\n\n</think>\n\n"
DIRECT_OPTION_INSTRUCTION = "Answer with only the option letter. Do not show any reasoning or working."
MMLU_PRO_INSTRUCTION = DIRECT_OPTION_INSTRUCTION

def model_wants_qwen_think_seed(model) -> bool:
    """Return True for Qwen-family models and False for models without `<think>` (e.g. Gemma)."""
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    class_name = type(model).__name__.lower()
    if "gemma" in model_type or "gemma" in class_name:
        return False
    return True

def _strip_forced_think(prompt: str) -> str:
    for suffix in ("<think>\n", "<think>"):
        if prompt.endswith(suffix):
            return prompt[: -len(suffix)]
    return prompt

def render_prompt(
    tokenizer, messages: list[dict], *, enable_thinking: bool = False,
    append_think_seed: bool = True,
) -> str:
    """Render raw-text inputs verbatim; chat messages go through `apply_chat_template`."""
    is_raw = len(messages) == 1 and messages[0].get("role") == RAW_TEXT_ROLE
    if is_raw:
        seed = QWEN_EMPTY_THINK_SEED if append_think_seed else ""
        return messages[0]["content"] + seed
    if getattr(tokenizer, "chat_template", None) is None:
        return "\n\n".join(m["content"] for m in messages) + "\n"
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    return _strip_forced_think(prompt)

def mc_prompt(
    question: str, options: list[str], letters: list[str], few_shot_prefix: str = "",
    instruction: str = "",
) -> str:
    """Multiple-choice raw-text prompt (ARC-Challenge, OpenBookQA, HellaSwag, TruthfulQA-MC1, GPQA-Diamond)."""
    body = f"Q: {question}\nOptions:\n"
    for L, opt in zip(letters, options):
        body += f"({L}) {opt}\n"
    if instruction:
        body += instruction + "\n"
    body += "A:"
    return f"{few_shot_prefix}\n\n{body}" if few_shot_prefix else body

ARC_FEW_SHOT = """\
Q: Which factor will most likely cause a person to develop a fever?
Options:
(A) a leg muscle relaxing after exercise
(B) a bacterial population in the bloodstream
(C) several viral particles on the skin
(D) carbohydrates being digested in the stomach
A: (B)

Q: A scientist wants to study the effect of temperature on chemical reactions. Which is the best independent variable?
Options:
(A) the volume of the container
(B) the type of chemicals used
(C) the temperature of the water bath
(D) the time the reaction takes
A: (C)

Q: Lichens are organisms produced by symbiotic relationships between:
(A) bacteria and viruses
(B) fungi and algae
(C) plants and animals
(D) lichens and animals
A: (B)"""

OBQA_FEW_SHOT = """\
Q: The sun is responsible for
Options:
(A) puppies learning new tricks
(B) children growing up and getting old
(C) flowers wilting in a vase
(D) plants sprouting, blooming and wilting
A: (D)

Q: Producing offspring helps to
Options:
(A) make food
(B) increase a population
(C) decrease a population
(D) help the food chain
A: (B)

Q: Sound is created when
Options:
(A) objects collide
(B) objects move
(C) objects vibrate
(D) objects float
A: (C)"""

TQA_FEW_SHOT = ""
TQA_MAX_LETTERS = 13

def tqa_letters(n: int) -> list[str]:
    assert 1 <= n <= TQA_MAX_LETTERS, f"n_candidates out of range: {n}"
    return [chr(ord("A") + i) for i in range(n)]

REINFORCED_DIRECT_ANSWER_PREFIX = """Answer with only the final number. Do not show any reasoning or working.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: The answer (arabic numerals) is 5.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: The answer (arabic numerals) is 8.

"""

def reinforced_direct_answer_prompt(question: str) -> str:
    return REINFORCED_DIRECT_ANSWER_PREFIX + f"Q: {question}\nA: The answer (arabic numerals) is"

REINFORCED_DIRECT_BOXED_ANSWER_PREFIX = """Answer with only the final answer inside \\boxed{}. Do not show any reasoning or working.

Q: What is $1+1$?
A: \\boxed{2}

Q: Simplify $\\frac{2}{4}$.
A: \\boxed{\\frac{1}{2}}

"""

def reinforced_direct_boxed_prompt(question: str) -> str:
    return REINFORCED_DIRECT_BOXED_ANSWER_PREFIX + f"Q: {question}\nA:"

_BBH_PROMPTS_DIR = PACKAGE_ROOT / "resources" / "eval_protocols" / "bbh_cot_prompts"
_BBH_FINAL_ANSWER_RE = re.compile(r"So the answer is\s*\(?\s*([A-Za-z])\s*\)?", re.IGNORECASE)
_BBH_DIRECT_PROMPTS_CACHE: dict[str, str] = {}

def build_bbh_direct_prompt(subtask: str) -> str:
    """Convert the official BIG-Bench-Hard 3-shot CoT examples into direct-answer form (`A: (X)`)."""
    if subtask in _BBH_DIRECT_PROMPTS_CACHE:
        return _BBH_DIRECT_PROMPTS_CACHE[subtask]

    path = _BBH_PROMPTS_DIR / f"{subtask}.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing BBH official cot-prompt for {subtask!r} at {path}."
        )
    raw = path.read_text(encoding="utf-8")
    body = re.split(r"^-----\s*$", raw, maxsplit=1, flags=re.MULTILINE)[-1].lstrip("\n")
    parts = re.split(r"\n\s*\n(?=Q:)", body)
    if len(parts) < 4:
        raise ValueError(f"Expected 1 description + >=3 examples in {path}, got {len(parts)}")
    description = parts[0].rstrip()
    examples_cot = parts[1:4]

    direct_examples: list[str] = []
    for ex in examples_cot:
        m = re.match(r"(Q:.*?)(\n\s*A:)", ex, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Cannot find 'A:' in example:\n{ex[:200]}")
        q_block = m.group(1).rstrip()
        a_block = ex[m.end():]
        am = _BBH_FINAL_ANSWER_RE.search(a_block)
        if not am:
            raise ValueError(f"Cannot find 'So the answer is (X)' in:\n{a_block[:200]}")
        letter = am.group(1).upper()
        direct_examples.append(f"{q_block}\nA: ({letter})")

    prefix = description + "\n\n" + "\n\n".join(direct_examples)
    _BBH_DIRECT_PROMPTS_CACHE[subtask] = prefix
    return prefix
