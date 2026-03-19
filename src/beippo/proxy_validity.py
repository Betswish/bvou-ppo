from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from beippo.config import ExperimentConfig
from beippo.modeling import extract_block_index, freeze_all_parameters, get_decoder_layers
from beippo.ppo import (
    RolloutBatch,
    build_response_mask,
    compute_returns_and_advantages,
    masked_mean,
    ppo_loss,
    shift_logprobs,
)
from beippo.registry import resolve_model
from beippo.reward import exact_match_reward
from beippo.selector import BlockSelector


def _is_lora_param(name: str) -> bool:
    return "lora_" in name


@dataclass
class ProxyValiditySummary:
    proxy: str
    spearman: float | None
    pearson: float | None
    num_blocks: int


class ExampleDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {"query": ex.query, "gold_label": ex.gold_label, "task_name": ex.task_name}


def collate_examples(batch):
    return batch


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    x_centered = xt - xt.mean()
    y_centered = yt - yt.mean()
    denom = torch.sqrt((x_centered.pow(2).sum()) * (y_centered.pow(2).sum()))
    if denom.item() == 0.0:
        return None
    return float((x_centered * y_centered).sum() / denom)


def spearman_correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def _current_lr(step: int, total_steps: int, base_lr: float, warmup_ratio: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def _selector_named_params(model, lora_enabled: bool):
    for name, param in model.named_parameters():
        if name.startswith("value_head"):
            continue
        block_idx = extract_block_index(name)
        if block_idx is None:
            continue
        if lora_enabled:
            if _is_lora_param(name):
                yield name, param, block_idx
        else:
            if not _is_lora_param(name):
                yield name, param, block_idx


def _collect_block_params(model, lora_enabled: bool) -> dict[int, list[tuple[str, torch.nn.Parameter]]]:
    by_block: dict[int, list[tuple[str, torch.nn.Parameter]]] = defaultdict(list)
    for name, param, block_idx in _selector_named_params(model, lora_enabled=lora_enabled):
        by_block[block_idx].append((name, param))
    return dict(by_block)


def _apply_lora_if_needed(model, cfg: ExperimentConfig):
    if cfg.lora.enabled:
        model = apply_lora(model, cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout, cfg.lora.target_modules)
    return model


def load_policy_and_reference(cfg: ExperimentConfig, checkpoint: str | None = None):
    from transformers import AutoModelForCausalLM
    from beippo.models.policy_value_model import PolicyWithValueHead, apply_lora, build_tokenizer

    model_path = checkpoint or cfg.train.model_name_or_path
    tokenizer = build_tokenizer(model_path)
    model = PolicyWithValueHead(model_path)
    model = _apply_lora_if_needed(model, cfg)
    if checkpoint is not None:
        value_head_path = Path(checkpoint) / "value_head.pt"
        if value_head_path.exists():
            model.value_head.load_state_dict(torch.load(value_head_path, map_location="cpu"))
    reference_model = AutoModelForCausalLM.from_pretrained(
        cfg.train.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    reference_model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
        reference_model = reference_model.to("cuda")
    model.eval()
    return model, reference_model, tokenizer


@torch.no_grad()
def generate_rollout_batch(model, reference_model, tokenizer, batch, cfg: ExperimentConfig):
    device = next(model.parameters()).device
    queries = [x["query"] for x in batch]
    gold_labels = [x["gold_label"] for x in batch]
    task_name = batch[0]["task_name"]
    prompt_tokens = tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.train.prompt_max_length,
    ).to(device)
    prompt_lengths = prompt_tokens.attention_mask.sum(dim=1)
    spec = resolve_model(cfg.train.model_name_or_path)
    generations = model.generate(
        **prompt_tokens,
        do_sample=True,
        temperature=spec.recommended_temperature,
        top_p=spec.recommended_top_p,
        min_new_tokens=1,
        max_new_tokens=cfg.train.response_max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    attention_mask = (generations != tokenizer.pad_token_id).long()
    policy_outputs = model(input_ids=generations, attention_mask=attention_mask)
    ref_outputs = reference_model(input_ids=generations, attention_mask=attention_mask, use_cache=False)
    old_logprobs = shift_logprobs(policy_outputs.logits, generations)
    old_values = policy_outputs.values[:, :-1]
    ref_logprobs = shift_logprobs(ref_outputs.logits, generations)
    response_mask = build_response_mask(attention_mask, prompt_lengths)
    responses = tokenizer.batch_decode(generations[:, prompt_tokens.input_ids.shape[1] :], skip_special_tokens=True)
    reward_values = torch.tensor(
        [exact_match_reward(task_name, pred, gold) for pred, gold in zip(responses, gold_labels)],
        dtype=torch.float32,
        device=device,
    )
    returns, advantages = compute_returns_and_advantages(reward_values, old_values, response_mask, cfg.ppo.whiten_advantages)
    rollout = RolloutBatch(
        input_ids=generations,
        attention_mask=attention_mask,
        response_mask=response_mask,
        old_logprobs=old_logprobs.detach(),
        old_values=old_values.detach(),
        ref_logprobs=ref_logprobs.detach(),
        rewards=reward_values.detach(),
        returns=returns.detach(),
        advantages=advantages.detach(),
        prompt_lengths=prompt_lengths.detach(),
    )
    metadata = {
        "task_name": task_name,
        "queries": queries,
        "gold_labels": gold_labels,
        "responses": responses,
    }
    return rollout, metadata


def _score_objective_adv_pg(model, rollout: RolloutBatch) -> torch.Tensor:
    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    new_logprobs = shift_logprobs(outputs.logits, rollout.input_ids)
    return masked_mean(rollout.advantages * new_logprobs, rollout.response_mask)


def _score_objective_mean_logp(model, rollout: RolloutBatch) -> torch.Tensor:
    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    new_logprobs = shift_logprobs(outputs.logits, rollout.input_ids)
    return masked_mean(new_logprobs, rollout.response_mask)


def _gather_grads_by_block(named_params, grads) -> dict[int, list[torch.Tensor]]:
    per_block: dict[int, list[torch.Tensor]] = defaultdict(list)
    for (name, param, block_idx), grad in zip(named_params, grads):
        if grad is None:
            continue
        per_block[block_idx].append(grad.detach())
    return dict(per_block)


def compute_adv_grad_energy_scores(model, rollout: RolloutBatch, lora_enabled: bool) -> dict[int, float]:
    freeze_all_parameters(model)
    named_params = []
    for name, param, block_idx in _selector_named_params(model, lora_enabled=lora_enabled):
        param.requires_grad_(True)
        named_params.append((name, param, block_idx))
    objective = _score_objective_adv_pg(model, rollout)
    grads = torch.autograd.grad(objective, [p for _, p, _ in named_params], retain_graph=False, create_graph=False, allow_unused=True)
    per_block = _gather_grads_by_block(named_params, grads)
    scores: dict[int, float] = {}
    for block_idx, tensors in per_block.items():
        score = 0.0
        for grad in tensors:
            score += float(grad.float().pow(2).sum().item())
        scores[block_idx] = score
    freeze_all_parameters(model)
    return scores


def compute_grad_norm_scores(model, rollout: RolloutBatch, lora_enabled: bool, clip_range: float, value_coef: float, kl_coef: float) -> dict[int, float]:
    freeze_all_parameters(model)
    named_params = []
    for name, param, block_idx in _selector_named_params(model, lora_enabled=lora_enabled):
        param.requires_grad_(True)
        named_params.append((name, param, block_idx))
    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    loss_out = ppo_loss(outputs.logits, outputs.values, rollout, clip_range, value_coef, kl_coef)
    grads = torch.autograd.grad(loss_out.loss, [p for _, p, _ in named_params], retain_graph=False, create_graph=False, allow_unused=True)
    per_block = _gather_grads_by_block(named_params, grads)
    scores: dict[int, float] = {}
    for block_idx, tensors in per_block.items():
        score = 0.0
        for grad in tensors:
            score += float(grad.float().pow(2).sum().item())
        scores[block_idx] = math.sqrt(score) if score > 0 else 0.0
    freeze_all_parameters(model)
    return scores


def compute_fisher_diag_energy_scores(model, rollout: RolloutBatch, lora_enabled: bool, damping: float = 1e-8) -> dict[int, float]:
    """Diagonal empirical-Fisher approximation to g^T F^{-1} g.

    g comes from the advantage-weighted policy gradient objective.
    F_diag is approximated from the masked mean log-prob objective.
    """
    freeze_all_parameters(model)
    named_params = []
    for name, param, block_idx in _selector_named_params(model, lora_enabled=lora_enabled):
        param.requires_grad_(True)
        named_params.append((name, param, block_idx))

    g_obj = _score_objective_adv_pg(model, rollout)
    fisher_obj = _score_objective_mean_logp(model, rollout)
    params = [p for _, p, _ in named_params]
    g_grads = torch.autograd.grad(g_obj, params, retain_graph=True, create_graph=False, allow_unused=True)
    fisher_grads = torch.autograd.grad(fisher_obj, params, retain_graph=False, create_graph=False, allow_unused=True)

    scores: dict[int, float] = defaultdict(float)
    for (name, param, block_idx), g_grad, f_grad in zip(named_params, g_grads, fisher_grads):
        if g_grad is None:
            continue
        g2 = g_grad.detach().float().pow(2)
        if f_grad is None:
            fdiag = torch.zeros_like(g2)
        else:
            fdiag = f_grad.detach().float().pow(2)
        scores[block_idx] += float((g2 / (fdiag + damping)).sum().item())
    freeze_all_parameters(model)
    return dict(scores)


def compute_gate_grad_scores(model, rollout: RolloutBatch, clip_range: float, value_coef: float, kl_coef: float) -> dict[int, float]:
    freeze_all_parameters(model)
    selector = BlockSelector(model, top_k=len(get_decoder_layers(model)))
    selector.controller.gates.requires_grad_(True)
    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    loss_out = ppo_loss(outputs.logits, outputs.values, rollout, clip_range, value_coef, kl_coef)
    gate_grads = torch.autograd.grad(loss_out.loss, selector.controller.gates, retain_graph=False, create_graph=False, allow_unused=True)[0]
    if gate_grads is None:
        gate_grads = torch.zeros_like(selector.controller.gates)
    scores = {idx: float(val) for idx, val in enumerate(gate_grads.detach().abs().float().cpu().tolist())}
    selector.close()
    freeze_all_parameters(model)
    return scores


def _block_objective_gain(
    model,
    rollout: RolloutBatch,
    block_params: list[torch.nn.Parameter],
    block_grads: list[torch.Tensor],
    step_size: float,
) -> float:
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


def compute_one_step_block_gains(model, rollout: RolloutBatch, lora_enabled: bool, candidate_blocks: list[int], step_size: float) -> dict[int, float]:
    freeze_all_parameters(model)
    named_params = []
    by_block: dict[int, list[tuple[str, torch.nn.Parameter]]] = defaultdict(list)
    for name, param, block_idx in _selector_named_params(model, lora_enabled=lora_enabled):
        if block_idx in set(candidate_blocks):
            param.requires_grad_(True)
            named_params.append((name, param, block_idx))
            by_block[block_idx].append((name, param))
    objective = _score_objective_adv_pg(model, rollout)
    grads = torch.autograd.grad(objective, [p for _, p, _ in named_params], retain_graph=False, create_graph=False, allow_unused=True)
    grads_by_block: dict[int, list[torch.Tensor]] = defaultdict(list)
    params_by_block: dict[int, list[torch.nn.Parameter]] = defaultdict(list)
    for (name, param, block_idx), grad in zip(named_params, grads):
        grads_by_block[block_idx].append(grad.detach() if grad is not None else None)
        params_by_block[block_idx].append(param)
    gains: dict[int, float] = {}
    for block_idx in candidate_blocks:
        gains[block_idx] = _block_objective_gain(model, rollout, params_by_block[block_idx], grads_by_block[block_idx], step_size)
    freeze_all_parameters(model)
    return gains


def _topk_union(score_maps: dict[str, dict[int, float]], top_k: int) -> list[int]:
    chosen: set[int] = set()
    for scores in score_maps.values():
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        chosen.update(idx for idx, _ in top)
    return sorted(chosen)


def run_proxy_validity_experiment(
    cfg: ExperimentConfig,
    checkpoint: str | None = None,
    split: str = "validation",
    max_samples: int = 64,
    max_batches: int = 1,
    mode: str = "bvou",
    top_k: int = 8,
    step_size: float = 1e-4,
    fisher_damping: float = 1e-8,
    output_dir: str | None = None,
):
    if mode not in {"full", "lora", "bvou", "bvou_lora"}:
        raise ValueError(f"Unsupported mode for proxy validity: {mode}")
    lora_enabled = mode in {"lora", "bvou_lora"}

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
    out_root = Path(output_dir) / "proxy_validity"
    out_root.mkdir(parents=True, exist_ok=True)

    batch_summaries = []
    summary_acc = defaultdict(list)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        rollout, metadata = generate_rollout_batch(model, reference_model, tokenizer, batch, cfg)
        proxy_scores = {
            "adv_grad_energy": compute_adv_grad_energy_scores(model, rollout, lora_enabled=lora_enabled),
            "fisher_diag_energy": compute_fisher_diag_energy_scores(model, rollout, lora_enabled=lora_enabled, damping=fisher_damping),
            "grad_norm": compute_grad_norm_scores(model, rollout, lora_enabled=lora_enabled, clip_range=cfg.ppo.clip_range, value_coef=cfg.ppo.value_coef, kl_coef=cfg.ppo.kl_coef),
            "gate_grad": compute_gate_grad_scores(model, rollout, clip_range=cfg.ppo.clip_range, value_coef=cfg.ppo.value_coef, kl_coef=cfg.ppo.kl_coef),
        }
        candidate_blocks = _topk_union(proxy_scores, top_k=top_k)
        true_gains = compute_one_step_block_gains(model, rollout, lora_enabled=lora_enabled, candidate_blocks=candidate_blocks, step_size=step_size)

        correlations = {}
        for proxy_name, score_map in proxy_scores.items():
            xs, ys = [], []
            for block_idx in candidate_blocks:
                if block_idx not in score_map or block_idx not in true_gains:
                    continue
                xs.append(float(score_map[block_idx]))
                ys.append(float(true_gains[block_idx]))
            sp = spearman_correlation(xs, ys)
            pe = _pearson(xs, ys)
            correlations[proxy_name] = {
                "spearman": sp,
                "pearson": pe,
                "num_blocks": len(xs),
            }
            if sp is not None:
                summary_acc[(proxy_name, "spearman")].append(sp)
            if pe is not None:
                summary_acc[(proxy_name, "pearson")].append(pe)

        batch_record = {
            "batch_idx": batch_idx,
            "task": cfg.train.task_name,
            "mode": mode,
            "avg_reward": float(rollout.rewards.mean().item()),
            "candidate_blocks": candidate_blocks,
            "true_one_step_gains": true_gains,
            "proxy_scores": proxy_scores,
            "correlations": correlations,
            "queries": metadata["queries"],
            "responses": metadata["responses"],
            "gold_labels": metadata["gold_labels"],
        }
        batch_summaries.append(batch_record)
        with (out_root / f"batch_{batch_idx:04d}.json").open("w", encoding="utf-8") as f:
            json.dump(batch_record, f, ensure_ascii=False, indent=2)

    summary = {
        "task": cfg.train.task_name,
        "mode": mode,
        "split": split,
        "max_samples": max_samples,
        "max_batches": max_batches,
        "top_k": top_k,
        "step_size": step_size,
        "fisher_damping": fisher_damping,
        "proxy_summaries": {},
    }
    for proxy_name in ["adv_grad_energy", "fisher_diag_energy", "grad_norm", "gate_grad"]:
        summary["proxy_summaries"][proxy_name] = {
            "mean_spearman": float(sum(summary_acc[(proxy_name, "spearman")]) / len(summary_acc[(proxy_name, "spearman")])) if summary_acc[(proxy_name, "spearman")] else None,
            "mean_pearson": float(sum(summary_acc[(proxy_name, "pearson")]) / len(summary_acc[(proxy_name, "pearson")])) if summary_acc[(proxy_name, "pearson")] else None,
            "num_batches": len(batch_summaries),
        }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, batch_summaries
