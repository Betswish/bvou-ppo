from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bvou_ppo.registry import ModelSpec


@dataclass
class PromptExample:
    task_name: str
    prompt_text: str
    gold_label: str


TASK_INSTRUCTIONS = {
    "boolq": "Read the passage and answer the question. Reply with only one word: yes or no.",
    "commonsenseqa": "Choose the correct option. Reply with only the option label: A, B, C, D, or E.",
    "arc_challenge": "Choose the correct option. Reply with only the option label: A, B, C, D, or E.",
}


def default_system_prompt(model: ModelSpec, deepseek_prompt_date: str) -> str | None:
    if model.family == "deepseek_r1_0528":
        return f"该助手为DeepSeek-R1，由深度求索公司创造。\n今天是{deepseek_prompt_date}。"
    return None


def build_task_user_prompt(task_name: str, row: dict[str, Any]) -> tuple[str, str]:
    if task_name == "boolq":
        passage = row["passage"].strip()
        question = row["question"].strip()
        answer = "yes" if bool(row["answer"]) else "no"
        text = (
            f"{TASK_INSTRUCTIONS[task_name]}\n\n"
            f"Passage:\n{passage}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return text, answer

    if task_name == "commonsenseqa":
        stem = row["question"]
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        text = (
            f"{TASK_INSTRUCTIONS[task_name]}\n\n"
            f"Question: {stem}\n"
            f"Options:\n{options}\n"
            "Answer:"
        )
        return text, row["answerKey"].strip()

    if task_name == "arc_challenge":
        stem = row["question"]
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        text = (
            f"{TASK_INSTRUCTIONS[task_name]}\n\n"
            f"Question: {stem}\n"
            f"Options:\n{options}\n"
            "Answer:"
        )
        return text, row["answerKey"].strip()

    raise ValueError(f"Unsupported task: {task_name}")


def render_chat_prompt(
    tokenizer,
    model: ModelSpec,
    user_prompt: str,
    enable_thinking: bool,
    use_official_system_prompt: bool,
    deepseek_prompt_date: str,
) -> str:
    messages: list[dict[str, str]] = []
    system_prompt = default_system_prompt(model, deepseek_prompt_date) if use_official_system_prompt else None
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    apply_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model.supports_enable_thinking:
        apply_kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **apply_kwargs)
