from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from beippo.registry import ModelSpec


@dataclass
class PromptExample:
    task_name: str
    prompt_text: str
    gold_label: str


TASK_INSTRUCTIONS = {
    "boolq": "Read the passage and answer the question. Reply with only one word: yes or no.",
    "commonsenseqa": "Choose the correct option. Reply with only the option label: A, B, C, D, or E.",
    "arc_challenge": "Choose the correct option. Reply with only the option label: A, B, C, D, or E.",
    "gsm8k": (
        "Solve the following grade-school math word problem step by step. "
        "On the last line, write exactly: Answer: \\boxed{final answer}."
    ),
    "math": (
        "Solve the following competition-math problem step by step. "
        "On the last line, write exactly: Answer: \\boxed{final answer}."
    ),
    "aime_2024": (
        "Solve the following AIME problem step by step. "
        "On the last line, write exactly: Answer: \\boxed{final integer answer between 0 and 999}."
    ),
}


def default_system_prompt(model: ModelSpec, deepseek_prompt_date: str) -> str | None:
    if model.family == "deepseek_r1_0528":
        return f"该助手为DeepSeek-R1，由深度求索公司创造。\n今天是{deepseek_prompt_date}。"
    return None


def _extract_gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip().splitlines()[-1].strip()


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

    if task_name == "gsm8k":
        question = row["question"].strip()
        gold = _extract_gsm8k_gold(row["answer"])
        text = (
            f"{TASK_INSTRUCTIONS[task_name]}\n\n"
            f"Question: {question}\n"
        )
        return text, gold

    if task_name == "math":
        problem = row["problem"].strip()

        if "answer" in row and row["answer"] not in (None, ""):
            gold = str(row["answer"]).strip()
        elif "solution" in row and row["solution"] not in (None, ""):
            gold = str(row["solution"]).strip()
        else:
            raise KeyError("Neither 'answer' nor 'solution' found for math example")

        text = (
            f"{TASK_INSTRUCTIONS[task_name]}\n\n"
            f"Problem:\n{problem}\n"
        )
        return text, gold

    if task_name == "aime_2024":
        # Common public mirrors expose `problem` and `answer`.
        problem = row["problem"].strip()
        gold = str(row["answer"]).strip()
        text = (
            f"{TASK_INSTRUCTIONS[task_name]}\n\n"
            f"Problem:\n{problem}\n"
        )
        return text, gold

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
