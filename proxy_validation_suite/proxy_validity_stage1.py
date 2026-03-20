from __future__ import annotations
from tqdm import tqdm

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from beippo.config import ExperimentConfig
from beippo.modeling import extract_block_index, freeze_all_parameters, get_decoder_layers
from beippo.modes import apply_mode
from beippo.ppo import (
    RolloutBatch,
    masked_mean,
    ppo_loss,
    shift_logprobs,
)
from beippo.proxy_validity import (
    _gather_grads_by_block,
    _pearson,
    _rankdata,
    collate_examples,
    ExampleDataset,
    generate_rollout_batch,
    load_policy_and_reference,
)
from beippo.reward import exact_match_reward
from beippo.selector import BlockSelector


def spearman_correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def _is_lora_param(name: str) -> bool:
    return 'lora_' in name


def _allowed_block_indices(model, cfg: ExperimentConfig) -> set[int]:
    num_layers = len(get_decoder_layers(model))
    allowed = set(range(num_layers))
    if cfg.selector.candidate_start_layer is not None:
        allowed = {i for i in allowed if i >= cfg.selector.candidate_start_layer}
    if cfg.selector.candidate_end_layer is not None:
        allowed = {i for i in allowed if i <= cfg.selector.candidate_end_layer}
    if cfg.selector.candidate_last_n_layers is not None:
        start = max(0, num_layers - int(cfg.selector.candidate_last_n_layers))
        allowed = {i for i in allowed if i >= start}
    if cfg.selector.search_upper_half_only:
        midpoint = num_layers // 2
        allowed = {i for i in allowed if i >= midpoint}
    return allowed


def _selector_named_params(model, lora_enabled: bool, allowed_blocks: set[int]):
    for name, param in model.named_parameters():
        if name.startswith('value_head'):
            continue
        block_idx = extract_block_index(name)
        if block_idx is None or block_idx not in allowed_blocks:
            continue
        if lora_enabled:
            if _is_lora_param(name):
                yield name, param, block_idx
        else:
            if not _is_lora_param(name):
                yield name, param, block_idx


def _score_objective_adv_pg(model, rollout: RolloutBatch) -> torch.Tensor:
    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    new_logprobs = shift_logprobs(outputs.logits, rollout.input_ids)
    return masked_mean(rollout.advantages * new_logprobs, rollout.response_mask)


def _score_objective_mean_logp(model, rollout: RolloutBatch) -> torch.Tensor:
    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    new_logprobs = shift_logprobs(outputs.logits, rollout.input_ids)
    return masked_mean(new_logprobs, rollout.response_mask)


def _score_objective_ppo(model, rollout: RolloutBatch, cfg: ExperimentConfig) -> torch.Tensor:
    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    loss_out = ppo_loss(outputs.logits, outputs.values, rollout, cfg.ppo.clip_range, cfg.ppo.value_coef, cfg.ppo.kl_coef)
    return loss_out.loss


def _named_params_and_grads_from_objective(model, rollout, cfg, lora_enabled, allowed_blocks, objective_name: str):
    freeze_all_parameters(model)
    named_params = []
    for name, param, block_idx in _selector_named_params(model, lora_enabled=lora_enabled, allowed_blocks=allowed_blocks):
        param.requires_grad_(True)
        named_params.append((name, param, block_idx))
    params = [p for _, p, _ in named_params]
    if objective_name == 'adv_pg':
        objective = _score_objective_adv_pg(model, rollout)
    elif objective_name == 'mean_logp':
        objective = _score_objective_mean_logp(model, rollout)
    elif objective_name == 'ppo':
        objective = _score_objective_ppo(model, rollout, cfg)
    else:
        raise ValueError(objective_name)
    grads = torch.autograd.grad(objective, params, retain_graph=False, create_graph=False, allow_unused=True)
    return named_params, grads


def compute_adv_grad_energy_scores(model, rollout, cfg, lora_enabled, allowed_blocks):
    named_params, grads = _named_params_and_grads_from_objective(model, rollout, cfg, lora_enabled, allowed_blocks, 'adv_pg')
    per_block = _gather_grads_by_block(named_params, grads)
    scores = {}
    for block_idx, tensors in per_block.items():
        score = 0.0
        for grad in tensors:
            score += float(grad.float().pow(2).sum().item())
        scores[block_idx] = score
    freeze_all_parameters(model)
    return scores


def compute_fisher_diag_energy_scores(model, rollout, cfg, lora_enabled, allowed_blocks, damping=1e-8):
    freeze_all_parameters(model)
    named_params = []
    for name, param, block_idx in _selector_named_params(model, lora_enabled=lora_enabled, allowed_blocks=allowed_blocks):
        param.requires_grad_(True)
        named_params.append((name, param, block_idx))
    params = [p for _, p, _ in named_params]
    g_obj = _score_objective_adv_pg(model, rollout)
    f_obj = _score_objective_mean_logp(model, rollout)
    g_grads = torch.autograd.grad(g_obj, params, retain_graph=True, create_graph=False, allow_unused=True)
    f_grads = torch.autograd.grad(f_obj, params, retain_graph=False, create_graph=False, allow_unused=True)
    scores = defaultdict(float)
    for (_name, _param, block_idx), g_grad, f_grad in zip(named_params, g_grads, f_grads):
        if g_grad is None:
            continue
        g2 = g_grad.detach().float().pow(2)
        fdiag = torch.zeros_like(g2) if f_grad is None else f_grad.detach().float().pow(2)
        scores[block_idx] += float((g2 / (fdiag + damping)).sum().item())
    freeze_all_parameters(model)
    return dict(scores)


def compute_grad_norm_scores(model, rollout, cfg, lora_enabled, allowed_blocks):
    named_params, grads = _named_params_and_grads_from_objective(model, rollout, cfg, lora_enabled, allowed_blocks, 'ppo')
    per_block = _gather_grads_by_block(named_params, grads)
    scores = {}
    for block_idx, tensors in per_block.items():
        s = 0.0
        for grad in tensors:
            s += float(grad.float().pow(2).sum().item())
        scores[block_idx] = math.sqrt(s) if s > 0 else 0.0
    freeze_all_parameters(model)
    return scores


def _block_numel(named_params, grads):
    numel = defaultdict(int)
    for (_name, param, block_idx), grad in zip(named_params, grads):
        if grad is not None:
            numel[block_idx] += grad.numel()
    return dict(numel)


def compute_lisa_scores(model, rollout, cfg, lora_enabled, allowed_blocks):
    named_params, grads = _named_params_and_grads_from_objective(model, rollout, cfg, lora_enabled, allowed_blocks, 'ppo')
    per_block = _gather_grads_by_block(named_params, grads)
    numel = _block_numel(named_params, grads)
    scores = {}
    for block_idx, tensors in per_block.items():
        s = 0.0
        for grad in tensors:
            s += float(grad.float().pow(2).sum().item())
        denom = math.sqrt(max(1, numel.get(block_idx, 1)))
        scores[block_idx] = (math.sqrt(s) / denom) if s > 0 else 0.0
    freeze_all_parameters(model)
    return scores


def compute_adagradselect_scores(model, rollout, cfg, lora_enabled, allowed_blocks):
    named_params, grads = _named_params_and_grads_from_objective(model, rollout, cfg, lora_enabled, allowed_blocks, 'ppo')
    per_block = _gather_grads_by_block(named_params, grads)
    numel = _block_numel(named_params, grads)
    scores = {}
    for block_idx, tensors in per_block.items():
        s = 0.0
        for grad in tensors:
            s += float(grad.float().abs().sum().item())
        denom = max(1, numel.get(block_idx, 1))
        scores[block_idx] = s / denom
    freeze_all_parameters(model)
    return scores


def compute_gate_grad_scores(model, rollout, cfg, allowed_blocks):
    freeze_all_parameters(model)
    selector = BlockSelector(model, top_k=len(get_decoder_layers(model)), search_upper_half_only=False)
    selector.controller.gates.requires_grad_(True)
    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    loss_out = ppo_loss(outputs.logits, outputs.values, rollout, cfg.ppo.clip_range, cfg.ppo.value_coef, cfg.ppo.kl_coef)
    gate_grads = torch.autograd.grad(loss_out.loss, selector.controller.gates, retain_graph=False, create_graph=False, allow_unused=True)[0]
    if gate_grads is None:
        gate_grads = torch.zeros_like(selector.controller.gates)
    scores = {
        idx: float(val)
        for idx, val in enumerate(gate_grads.detach().abs().float().cpu().tolist())
        if idx in allowed_blocks
    }
    selector.close()
    freeze_all_parameters(model)
    return scores


def compute_random_scores(model, allowed_blocks, seed: int):
    rng = random.Random(seed)
    return {idx: rng.random() for idx in sorted(allowed_blocks)}


def _block_objective_gain(model, rollout, block_params, block_grads, step_size: float):
    with torch.no_grad():
        originals = [p.detach().clone() for p in block_params]
    base_value = float(_score_objective_adv_pg(model, rollout).detach().item())
    with torch.no_grad():
        for param, grad in zip(block_params, block_grads):
            if grad is None:
                continue
            param.add_(step_size * grad.to(param.dtype))
    new_value = float(_score_objective_adv_pg(model, rollout).detach().item())
    with torch.no_grad():
        for param, original in zip(block_params, originals):
            param.copy_(original)
    return new_value - base_value


def compute_one_step_block_gains(model, rollout, cfg, lora_enabled, candidate_blocks, step_size: float, allowed_blocks):
    freeze_all_parameters(model)
    named_params = []
    candset = set(candidate_blocks) & set(allowed_blocks)
    by_block = defaultdict(list)
    for name, param, block_idx in _selector_named_params(model, lora_enabled=lora_enabled, allowed_blocks=allowed_blocks):
        if block_idx in candset:
            param.requires_grad_(True)
            named_params.append((name, param, block_idx))
            by_block[block_idx].append((name, param))
    objective = _score_objective_adv_pg(model, rollout)
    grads = torch.autograd.grad(objective, [p for _, p, _ in named_params], retain_graph=False, create_graph=False, allow_unused=True)
    grads_by_block = defaultdict(list)
    params_by_block = defaultdict(list)
    for (_name, param, block_idx), grad in zip(named_params, grads):
        grads_by_block[block_idx].append(grad.detach() if grad is not None else None)
        params_by_block[block_idx].append(param)
    gains = {}
    for block_idx in candidate_blocks:
        if block_idx not in params_by_block:
            continue
        gains[block_idx] = _block_objective_gain(model, rollout, params_by_block[block_idx], grads_by_block[block_idx], step_size)
    freeze_all_parameters(model)
    return gains


def _topk_union(score_maps: dict[str, dict[int, float]], top_k: int) -> list[int]:
    chosen = set()
    for scores in score_maps.values():
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        chosen.update(idx for idx, _ in top)
    return sorted(chosen)


def topk_overlap(score_map: dict[int, float], true_gains: dict[int, float], k: int) -> float | None:
    if not score_map or not true_gains:
        return None
    pred_top = [idx for idx, _ in sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:k]]
    true_top = [idx for idx, _ in sorted(true_gains.items(), key=lambda kv: kv[1], reverse=True)[:k]]
    if not pred_top or not true_top:
        return None
    return len(set(pred_top) & set(true_top)) / float(min(k, len(true_top), len(pred_top)))


def top1_hit(score_map: dict[int, float], true_gains: dict[int, float]) -> float | None:
    if not score_map or not true_gains:
        return None
    pred_top = max(score_map.items(), key=lambda kv: kv[1])[0]
    true_top = max(true_gains.items(), key=lambda kv: kv[1])[0]
    return 1.0 if pred_top == true_top else 0.0


def run_proxy_validation_stage1(cfg: ExperimentConfig, checkpoint=None, split='validation', max_samples=64, max_batches=1, mode='bvou', top_k=8, step_size=1e-4, fisher_damping=1e-8, proxies=None, output_dir=None):
    if mode not in {'bvou', 'bvou_lora', 'lora', 'full'}:
        raise ValueError(mode)
    cfg = apply_mode(cfg, mode)
    lora_enabled = mode in {'lora', 'bvou_lora'}

    model, reference_model, tokenizer = load_policy_and_reference(cfg, checkpoint=checkpoint)
    from beippo.data import load_task_examples
    examples = load_task_examples(
        task_name=cfg.train.task_name,
        split=split,
        max_samples=max_samples,
        tokenizer=tokenizer,
        model_name_or_alias=cfg.train.model_name_or_path,
        enable_thinking=cfg.train.enable_thinking,
        use_official_system_prompt=cfg.train.use_official_system_prompt,
        deepseek_prompt_date=cfg.train.deepseek_prompt_date,
    )
    loader = DataLoader(ExampleDataset(examples), batch_size=cfg.train.per_device_batch_size, shuffle=False, collate_fn=collate_examples)

    if output_dir is None:
        output_dir = cfg.train.output_dir
    out_root = Path(output_dir+f"_{max_samples}_{max_batches}_{top_k}") / 'proxy_validation_stage1'
    out_root.mkdir(parents=True, exist_ok=True)

    allowed_blocks = _allowed_block_indices(model, cfg)
    proxy_names = proxies or [
        'adv_grad_energy', 'fisher_diag_energy', 'grad_norm', 'gate_grad', 'lisa_score', 'adagradselect_score', 'random'
    ]
    summary_acc = defaultdict(list)
    batch_summaries = []

    for batch_idx, batch in enumerate(tqdm(loader)):
        if batch_idx >= max_batches:
            break
        rollout, metadata = generate_rollout_batch(model, reference_model, tokenizer, batch, cfg)
        score_maps = {}
        for proxy in proxy_names:
            if proxy == 'adv_grad_energy':
                score_maps[proxy] = compute_adv_grad_energy_scores(model, rollout, cfg, lora_enabled, allowed_blocks)
            elif proxy == 'fisher_diag_energy':
                score_maps[proxy] = compute_fisher_diag_energy_scores(model, rollout, cfg, lora_enabled, allowed_blocks, damping=fisher_damping)
            elif proxy == 'grad_norm':
                score_maps[proxy] = compute_grad_norm_scores(model, rollout, cfg, lora_enabled, allowed_blocks)
            elif proxy == 'gate_grad':
                score_maps[proxy] = compute_gate_grad_scores(model, rollout, cfg, allowed_blocks)
            elif proxy == 'lisa_score':
                score_maps[proxy] = compute_lisa_scores(model, rollout, cfg, lora_enabled, allowed_blocks)
            elif proxy == 'adagradselect_score':
                score_maps[proxy] = compute_adagradselect_scores(model, rollout, cfg, lora_enabled, allowed_blocks)
            elif proxy == 'random':
                score_maps[proxy] = compute_random_scores(model, allowed_blocks, seed=cfg.train.seed + batch_idx)
            else:
                raise ValueError(f'Unknown proxy: {proxy}')

        candidate_blocks = _topk_union(score_maps, top_k=top_k)
        true_gains = compute_one_step_block_gains(model, rollout, cfg, lora_enabled, candidate_blocks, step_size, allowed_blocks)

        metrics = {}
        for proxy, score_map in score_maps.items():
            xs, ys = [], []
            for block_idx in candidate_blocks:
                if block_idx in score_map and block_idx in true_gains:
                    xs.append(float(score_map[block_idx]))
                    ys.append(float(true_gains[block_idx]))
            sp = spearman_correlation(xs, ys)
            pe = _pearson(xs, ys)
            overlap = topk_overlap(score_map, true_gains, top_k)
            hit = top1_hit(score_map, true_gains)
            metrics[proxy] = {
                'spearman': sp,
                'pearson': pe,
                'topk_overlap': overlap,
                'top1_hit_rate': hit,
                'num_blocks': len(xs),
            }
            for key, val in [('spearman', sp), ('pearson', pe), ('topk_overlap', overlap), ('top1_hit_rate', hit)]:
                if val is not None:
                    summary_acc[(proxy, key)].append(val)

        batch_record = {
            'batch_idx': batch_idx,
            'task': cfg.train.task_name,
            'mode': mode,
            'avg_reward': float(rollout.rewards.mean().item()),
            'allowed_blocks': sorted(allowed_blocks),
            'candidate_blocks': candidate_blocks,
            'true_one_step_gains': true_gains,
            'proxy_scores': score_maps,
            'metrics': metrics,
            'queries': metadata['queries'],
            'responses': metadata['responses'],
            'gold_labels': metadata['gold_labels'],
        }
        batch_summaries.append(batch_record)
        with (out_root / f'batch_{batch_idx:04d}.json').open('w', encoding='utf-8') as f:
            json.dump(batch_record, f, ensure_ascii=False, indent=2)

    summary = {
        'task': cfg.train.task_name,
        'mode': mode,
        'split': split,
        'max_samples': max_samples,
        'max_batches': max_batches,
        'top_k': top_k,
        'step_size': step_size,
        'fisher_damping': fisher_damping,
        'proxy_summaries': {},
    }
    for proxy in proxy_names:
        summary['proxy_summaries'][proxy] = {
            'mean_spearman': (sum(summary_acc[(proxy, 'spearman')]) / len(summary_acc[(proxy, 'spearman')])) if summary_acc[(proxy, 'spearman')] else None,
            'mean_pearson': (sum(summary_acc[(proxy, 'pearson')]) / len(summary_acc[(proxy, 'pearson')])) if summary_acc[(proxy, 'pearson')] else None,
            'mean_topk_overlap': (sum(summary_acc[(proxy, 'topk_overlap')]) / len(summary_acc[(proxy, 'topk_overlap')])) if summary_acc[(proxy, 'topk_overlap')] else None,
            'mean_top1_hit_rate': (sum(summary_acc[(proxy, 'top1_hit_rate')]) / len(summary_acc[(proxy, 'top1_hit_rate')])) if summary_acc[(proxy, 'top1_hit_rate')] else None,
            'num_batches': len(batch_summaries),
        }
    with (out_root / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary, batch_summaries
