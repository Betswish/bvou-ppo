from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

from bvou_ppo.prompts import build_task_user_prompt, render_chat_prompt
from bvou_ppo.registry import resolve_model


@dataclass
class TaskExample:
    task_name: str
    query: str
    gold_label: str


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
    if task_name == "boolq":
        ds = load_dataset("google/boolq", split=split)
    elif task_name == "commonsenseqa":
        ds = load_dataset("tau/commonsense_qa", split=split)
    elif task_name == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    else:
        raise ValueError(f"Unsupported task: {task_name}")

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
