from __future__ import annotations

from dataclasses import dataclass
from datasets import load_dataset

from beippo.prompts import build_task_user_prompt, render_chat_prompt
from beippo.registry import resolve_model


@dataclass
class TaskExample:
    task_name: str
    query: str
    gold_label: str


def _load_reasoning_dataset(task_name: str, split: str):
    if task_name == "boolq":
        return load_dataset("google/boolq", split=split)
    if task_name == "commonsenseqa":
        return load_dataset("tau/commonsense_qa", split=split)
    if task_name == "arc_challenge":
        return load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    if task_name == "gsm8k":
        # RL/RLVR-common low-complexity math reasoning benchmark.
        return load_dataset("openai/gsm8k", "main", split=split)
    if task_name == "math":
        # RL/RLVR-common medium-complexity mathematical reasoning benchmark.
        # return load_dataset("HuggingFaceH4/MATH", split=split)
        return load_dataset("HuggingFaceH4/MATH-500", split=split)
    if task_name == "aime_2024":
        # RL/RLVR-common high-complexity competition-math benchmark.
        # The HF mirror commonly exposes a single train split.
        return load_dataset("HuggingFaceH4/aime_2024", split=split)
    raise ValueError(f"Unsupported task: {task_name}")


def load_task_examples(
    task_name: str,
    split: str,
    max_samples: int,
    tokenizer,
    model_name_or_alias: str,
    enable_thinking: bool,
    use_official_system_prompt: bool,
    deepseek_prompt_date: str,
) -> list[TaskExample]:
    model_spec = resolve_model(model_name_or_alias)
    ds = _load_reasoning_dataset(task_name, split)

    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    examples: list[TaskExample] = []
    for row in ds:
        user_prompt, gold = build_task_user_prompt(task_name, row)
        query = render_chat_prompt(
            tokenizer=tokenizer,
            model=model_spec,
            user_prompt=user_prompt,
            enable_thinking=enable_thinking,
            use_official_system_prompt=use_official_system_prompt,
            deepseek_prompt_date=deepseek_prompt_date,
        )
        examples.append(TaskExample(task_name=task_name, query=query, gold_label=gold))
    return examples
