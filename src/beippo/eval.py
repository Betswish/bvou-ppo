from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from accelerate.utils import DistributedType
from tqdm.auto import tqdm

from beippo.reward import exact_match_reward


@dataclass
class EvalResult:
    task_name: str
    split: str
    samples: int
    exact_match: float


def _is_zero3(accelerator=None, model=None) -> bool:
    # Prefer model/engine introspection when available; this is more reliable
    # than accelerate plugin metadata across versions.
    for obj in (model, getattr(model, "module", None)):
        if obj is None:
            continue
        try:
            stage_fn = getattr(obj, "zero_optimization_stage", None)
            if callable(stage_fn):
                return int(stage_fn()) == 3
        except Exception:
            pass
        try:
            opt = getattr(obj, "optimizer", None)
            stage = getattr(opt, "stage", None)
            if stage is not None:
                return int(stage) == 3
        except Exception:
            pass

    # Fall back to parameter attribute inspection. ZeRO-3 parameters usually
    # carry ds_* metadata and local numel may differ from ds_numel.
    for obj in (model, getattr(model, "module", None)):
        if obj is None:
            continue
        try:
            for p in obj.parameters():
                ds_numel = getattr(p, "ds_numel", None)
                ds_id = getattr(p, "ds_id", None)
                if ds_numel is not None or ds_id is not None:
                    if ds_numel is None:
                        return True
                    try:
                        if int(ds_numel) != int(p.numel()):
                            return True
                    except Exception:
                        return True
        except Exception:
            pass

    if accelerator is None:
        return False
    try:
        plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        if plugin is None:
            return False
        zero_stage = getattr(plugin, "zero_stage", None)
        if zero_stage is not None:
            return int(zero_stage) == 3
        cfg = getattr(plugin, "deepspeed_config", None)
        if isinstance(cfg, dict):
            stage = cfg.get("zero_optimization", {}).get("stage", 0)
            return int(stage) == 3
        hf_ds = getattr(plugin, "hf_ds_config", None)
        if hf_ds is not None and hasattr(hf_ds, "config"):
            stage = hf_ds.config.get("zero_optimization", {}).get("stage", 0)
            return int(stage) == 3
    except Exception:
        return False
    return False


def _broadcast_eval_outputs(accelerator, correct: int, total: int, rollout_records: list[dict], collect_rollouts: bool):
    if accelerator is None or accelerator.num_processes == 1:
        return correct, total, rollout_records
    device = accelerator.device
    counts = torch.tensor([correct, total], device=device, dtype=torch.long)
    dist.broadcast(counts, src=0)
    correct = int(counts[0].item())
    total = int(counts[1].item())
    if collect_rollouts:
        obj = [rollout_records if accelerator.is_main_process else None]
        dist.broadcast_object_list(obj, src=0)
        rollout_records = obj[0] or []
    else:
        rollout_records = []
    return correct, total, rollout_records


@torch.no_grad()
def _run_task_eval_zero3_rank0_only(model, tokenizer, task_name: str, split: str, examples, batch_size: int = 4, max_new_tokens: int = 8, collect_rollouts: bool = False, accelerator=None):
    import deepspeed

    base_model = accelerator.unwrap_model(model)
    lm = base_model.pretrained_model
    device = accelerator.device
    correct = 0
    total = 0
    rollout_records: list[dict] = []

    params = [p for p in lm.parameters()]
    with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
        if accelerator.is_main_process:
            for start in tqdm(range(0, len(examples), batch_size), desc=f"eval:{task_name}/{split}", leave=False):
                batch = examples[start : start + batch_size]
                if not batch:
                    continue
                tokens = tokenizer([x.query for x in batch], return_tensors="pt", padding=True, truncation=True).to(device)
                generations = lm.generate(
                    **tokens,
                    do_sample=False,
                    min_new_tokens=1,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    synced_gpus=False,
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
        if accelerator is not None:
            accelerator.wait_for_everyone()

    correct, total, rollout_records = _broadcast_eval_outputs(accelerator, correct, total, rollout_records, collect_rollouts)
    return EvalResult(task_name=task_name, split=split, samples=total, exact_match=correct / max(total, 1)), rollout_records


@torch.no_grad()
def run_task_eval(model, tokenizer, task_name: str, split: str, examples, batch_size: int = 4, max_new_tokens: int = 8, collect_rollouts: bool = False, accelerator=None):
    zero3 = _is_zero3(accelerator, model)
    if zero3 and accelerator is not None:
        return _run_task_eval_zero3_rank0_only(
            model, tokenizer, task_name, split, examples, batch_size=batch_size, max_new_tokens=max_new_tokens, collect_rollouts=collect_rollouts, accelerator=accelerator
        )

    eval_model = accelerator.unwrap_model(model) if accelerator is not None else model
    device = accelerator.device if accelerator is not None else next(eval_model.parameters()).device
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
        generations = eval_model.generate(**tokens, **gen_kwargs)
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
