from __future__ import annotations

import re

LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)
BOOL_RE = re.compile(r"\b(yes|no|true|false|y|n)\b", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_TAG_RE = re.compile(r"answer\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
# Last number-ish span, including integers, decimals, simple signed numbers, and latex-ish fractions.
NUMBERISH_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?|\\frac\{[^{}]+\}\{[^{}]+\}")


def _clean_text(text: str) -> str:
    text = THINK_BLOCK_RE.sub(" ", text)
    text = text.replace("<|im_start|>", " ").replace("<|im_end|>", " ")
    text = text.replace("**", " ")
    return text.strip()


def _normalize_math_string(text: str) -> str | None:
    text = _clean_text(text)
    if not text:
        return None

    boxed = BOXED_RE.findall(text)
    if boxed:
        candidate = boxed[-1].strip()
        return candidate.replace(" ", "").replace("$", "")

    answer_tag = ANSWER_TAG_RE.findall(text)
    if answer_tag:
        candidate = answer_tag[-1].strip().splitlines()[0]
        return candidate.replace(" ", "").replace("$", "")

    # Fall back to the last non-empty line.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1].replace("$", "")
        nums = NUMBERISH_RE.findall(last)
        if nums:
            return nums[-1].replace(",", "").replace(" ", "")
        return last.replace(" ", "")

    nums = NUMBERISH_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "").replace(" ", "")
    return None


def normalize_prediction(task_name: str, text: str) -> str | None:
    text = _clean_text(text)

    if task_name == "boolq":
        match = BOOL_RE.search(text)
        if not match:
            return None
        token = match.group(1).lower()
        if token in {"yes", "true", "y"}:
            return "yes"
        if token in {"no", "false", "n"}:
            return "no"
        return None

    if task_name in {"commonsenseqa", "arc_challenge"}:
        match = LETTER_RE.search(text.upper())
        return match.group(1).upper() if match else None

    if task_name in {"gsm8k", "math", "aime_2024"}:
        return _normalize_math_string(text)

    raise ValueError(f"Unsupported task: {task_name}")


def exact_match_reward(task_name: str, prediction: str, gold_label: str) -> float:
    pred = normalize_prediction(task_name, prediction)
    if task_name == "boolq":
        gold = gold_label.strip().lower()
    elif task_name in {"commonsenseqa", "arc_challenge"}:
        gold = gold_label.strip().upper()
    else:
        gold = _normalize_math_string(gold_label) or gold_label.strip().replace(" ", "").replace("$", "")
    return 1.0 if pred == gold else 0.0
