from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from beippo.reward import exact_match_reward


@dataclass
class EvalResult:
    task_name: str
    split: str
    samples: int
    exact_match: float


@torch.no_grad()
def run_task_eval(model, tokenizer, task_name: str, split: str, examples, batch_size: int = 4, max_new_tokens: int = 8, collect_rollouts: bool = False, accelerator=None):
    device = next(model.parameters()).device
    rank = accelerator.process_index if accelerator is not None else 0
    world_size = accelerator.num_processes if accelerator is not None else 1
    local_examples = examples[rank::world_size]

    correct = 0
    total = 0
    rollout_records: list[dict] = []
    for start in tqdm(range(0, len(local_examples), batch_size), desc=f"eval:{task_name}/{split}", leave=False, disable=(accelerator is not None and not accelerator.is_local_main_process)):
        batch = local_examples[start : start + batch_size]
        if not batch:
            continue
        tokens = tokenizer([x.query for x in batch], return_tensors="pt", padding=True, truncation=True).to(device)
        gen_kwargs = dict(
            do_sample=False,
            min_new_tokens=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if accelerator is not None and accelerator.num_processes > 1:
            gen_kwargs["synced_gpus"] = True
        generations = model.generate(**tokens, **gen_kwargs)
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

    if accelerator is not None and accelerator.num_processes > 1:
        device = accelerator.device
        counts = torch.tensor([correct, total], device=device, dtype=torch.long)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        correct = int(counts[0].item())
        total = int(counts[1].item())
        if collect_rollouts:
            gathered = [None for _ in range(accelerator.num_processes)]
            dist.all_gather_object(gathered, rollout_records)
            if accelerator.is_main_process:
                rollout_records = [rec for part in gathered for rec in (part or [])]
            else:
                rollout_records = []

    return EvalResult(task_name=task_name, split=split, samples=total, exact_match=correct / max(total, 1)), rollout_records
