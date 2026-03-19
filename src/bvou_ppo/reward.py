from __future__ import annotations

import re


LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)
YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def normalize_prediction(task_name: str, text: str) -> str | None:
    text = text.strip()
    if task_name == "boolq":
        match = YESNO_RE.search(text)
        return match.group(1).lower() if match else None
    if task_name in {"commonsenseqa", "arc_challenge"}:
        match = LETTER_RE.search(text.upper())
        return match.group(1).upper() if match else None
    raise ValueError(f"Unsupported task: {task_name}")


def exact_match_reward(task_name: str, prediction: str, gold_label: str) -> float:
    pred = normalize_prediction(task_name, prediction)
    gold = gold_label.strip().lower() if task_name == "boolq" else gold_label.strip().upper()
    return 1.0 if pred == gold else 0.0
