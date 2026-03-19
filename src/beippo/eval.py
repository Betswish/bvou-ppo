from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from beippo.reward import exact_match_reward


@dataclass
class EvalResult:
    task_name: str
    split: str
    samples: int
    exact_match: float


@torch.no_grad()
def run_task_eval(model, tokenizer, task_name: str, split: str, examples, batch_size: int = 4, max_new_tokens: int = 8, collect_rollouts: bool = False):
    device = next(model.parameters()).device
    correct = 0
    total = 0
    rollout_records: list[dict] = []
    for start in tqdm(range(0, len(examples), batch_size), desc=f"eval:{task_name}/{split}", leave=False):
        batch = examples[start : start + batch_size]
        tokens = tokenizer([x.query for x in batch], return_tensors="pt", padding=True, truncation=True).to(device)
        generations = model.generate(
            **tokens,
            do_sample=False,
            min_new_tokens=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(generations[:, tokens.input_ids.shape[1]:], skip_special_tokens=True)
        for pred, ex in zip(decoded, batch):
            is_correct = int(exact_match_reward(task_name, pred, ex.gold_label))
            correct += is_correct
            total += 1
            if collect_rollouts:
                rollout_records.append({
                    'kind': 'eval_rollout',
                    'task': task_name,
                    'split': split,
                    'query': ex.query,
                    'gold_label': ex.gold_label,
                    'response': pred,
                    'correct': bool(is_correct),
                })
    return EvalResult(task_name=task_name, split=split, samples=total, exact_match=correct / max(total, 1)), rollout_records
