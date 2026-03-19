from __future__ import annotations

import re


LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)
BOOL_RE = re.compile(r"\b(yes|no|true|false|y|n)\b", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _clean_text(text: str) -> str:
    text = THINK_BLOCK_RE.sub(" ", text)
    text = text.replace("<|im_start|>", " ").replace("<|im_end|>", " ")
    return text.strip()


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
    raise ValueError(f"Unsupported task: {task_name}")


def exact_match_reward(task_name: str, prediction: str, gold_label: str) -> float:
    pred = normalize_prediction(task_name, prediction)
    gold = gold_label.strip().lower() if task_name == "boolq" else gold_label.strip().upper()
    return 1.0 if pred == gold else 0.0
