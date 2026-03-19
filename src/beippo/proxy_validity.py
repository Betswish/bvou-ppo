from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from beippo.block_scores import (
    compute_adagradselect_scores,
    compute_adv_grad_energy_scores,
    compute_fisher_diag_energy_scores,
    compute_gate_grad_scores,
    compute_grad_norm_scores,
    compute_lisa_scores,
    compute_random_scores,
    score_tensor_to_dict,
)
from beippo.config import ExperimentConfig
from beippo.modeling import extract_block_index, freeze_all_parameters
from beippo.ppo import RolloutBatch, build_response_mask, compute_returns_and_advantages, masked_mean, shift_logprobs
from beippo.registry import resolve_model
from beippo.reward import exact_match_reward


def _is_lora_param(name: str) -> bool:
    return "lora_" in name


@dataclass
class ProxyValiditySummary:
    proxy: str
    spearman: float | None
    pearson: float | None
    topk_overlap: float | None
    top1_hit: float | None
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


def _topk_indices(score_map: dict[int, float], k: int) -> list[int]:
    return [idx for idx, _ in sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def _topk_overlap_fraction(score_map: dict[int, float], true_gain_map: dict[int, float], k: int) -> float | None:
    if not score_map or not true_gain_map:
        return None
    ks = min(k, len(score_map), len(true_gain_map))
    if ks <= 0:
        return None
    top_pred = set(_topk_indices(score_map, ks))
    top_true = set(_topk_indices(true_gain_map, ks))
    return float(len(top_pred & top_true) / ks)


def _top1_hit(score_map: dict[int, float], true_gain_map: dict[int, float]) -> float | None:
    if not score_map or not true_gain_map:
        return None
    pred_top = _topk_indices(score_map, 1)
    true_top = _topk_indices(true_gain_map, 1)
    if not pred_top or not true_top:
        return None
    return 1.0 if pred_top[0] == true_top[0] else 0.0


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


def _apply_lora_if_needed(model, cfg: ExperimentConfig):
    if cfg.lora.enabled:
        from beippo.models.policy_value_model import apply_lora
        model = apply_lora(model, cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout, cfg.lora.target_modules)
    return model


def load_policy_and_reference(cfg: ExperimentConfig, checkpoint: str | None = None):
    from transformers import AutoModelForCausalLM
    from beippo.models.policy_value_model import PolicyWithValueHead, build_tokenizer

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


def _block_objective_gain(model, rollout: RolloutBatch, block_params: list[torch.nn.Parameter], block_grads: list[torch.Tensor | None], step_size: float) -> float:
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
    target_set = set(candidate_blocks)
    for name, param, block_idx in _selector_named_params(model, lora_enabled=lora_enabled):
        if block_idx in target_set:
            param.requires_grad_(True)
            named_params.append((name, param, block_idx))
    objective = _score_objective_adv_pg(model, rollout)
    grads = torch.autograd.grad(objective, [p for _, p, _ in named_params], retain_graph=False, create_graph=False, allow_unused=True)
    grads_by_block: dict[int, list[torch.Tensor | None]] = defaultdict(list)
    params_by_block: dict[int, list[torch.nn.Parameter]] = defaultdict(list)
    for (_name, param, block_idx), grad in zip(named_params, grads):
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


def _compute_proxy_scores(model, rollout: RolloutBatch, lora_enabled: bool, cfg: ExperimentConfig, proxies: list[str], fisher_damping: float, seed_offset: int = 0) -> dict[str, dict[int, float]]:
    score_maps: dict[str, dict[int, float]] = {}
    for proxy_name in proxies:
        if proxy_name == "adv_grad_energy":
            scores = compute_adv_grad_energy_scores(model, rollout, lora_enabled=lora_enabled)
        elif proxy_name == "fisher_diag_energy":
            scores = compute_fisher_diag_energy_scores(model, rollout, lora_enabled=lora_enabled, damping=fisher_damping)
        elif proxy_name == "grad_norm":
            scores = compute_grad_norm_scores(model, rollout, lora_enabled=lora_enabled, clip_range=cfg.ppo.clip_range, value_coef=cfg.ppo.value_coef, kl_coef=cfg.ppo.kl_coef)
        elif proxy_name == "gate_grad":
            scores = compute_gate_grad_scores(model, rollout, clip_range=cfg.ppo.clip_range, value_coef=cfg.ppo.value_coef, kl_coef=cfg.ppo.kl_coef)
        elif proxy_name == "lisa_score":
            scores = compute_lisa_scores(model, rollout, lora_enabled=lora_enabled, clip_range=cfg.ppo.clip_range, value_coef=cfg.ppo.value_coef, kl_coef=cfg.ppo.kl_coef)
        elif proxy_name == "adagradselect_score":
            scores = compute_adagradselect_scores(model, rollout, lora_enabled=lora_enabled, clip_range=cfg.ppo.clip_range, value_coef=cfg.ppo.value_coef, kl_coef=cfg.ppo.kl_coef)
        elif proxy_name == "random":
            scores = compute_random_scores(model, seed=cfg.train.seed + seed_offset)
        else:
            raise ValueError(f"Unknown proxy: {proxy_name}")
        score_maps[proxy_name] = score_tensor_to_dict(scores)
    return score_maps


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
    proxies: list[str] | None = None,
):
    if mode not in {"full", "lora", "bvou", "bvou_lora"}:
        raise ValueError(f"Unsupported mode for proxy validity: {mode}")
    lora_enabled = mode in {"lora", "bvou_lora"}
    if proxies is None:
        proxies = [
            "adv_grad_energy",
            "fisher_diag_energy",
            "grad_norm",
            "gate_grad",
            "lisa_score",
            "adagradselect_score",
            "random",
        ]

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
        proxy_scores = _compute_proxy_scores(
            model=model,
            rollout=rollout,
            lora_enabled=lora_enabled,
            cfg=cfg,
            proxies=proxies,
            fisher_damping=fisher_damping,
            seed_offset=batch_idx,
        )
        candidate_blocks = _topk_union(proxy_scores, top_k=top_k)
        true_gains = compute_one_step_block_gains(model, rollout, lora_enabled=lora_enabled, candidate_blocks=candidate_blocks, step_size=step_size)

        correlations = {}
        for proxy_name, score_map in proxy_scores.items():
            filtered_score_map = {b: float(score_map[b]) for b in candidate_blocks if b in score_map}
            filtered_true_gains = {b: float(true_gains[b]) for b in candidate_blocks if b in true_gains}
            xs = [filtered_score_map[b] for b in filtered_score_map if b in filtered_true_gains]
            ys = [filtered_true_gains[b] for b in filtered_score_map if b in filtered_true_gains]
            sp = spearman_correlation(xs, ys)
            pe = _pearson(xs, ys)
            overlap = _topk_overlap_fraction(filtered_score_map, filtered_true_gains, top_k)
            top1 = _top1_hit(filtered_score_map, filtered_true_gains)
            correlations[proxy_name] = {
                "spearman": sp,
                "pearson": pe,
                "topk_overlap": overlap,
                "top1_hit": top1,
                "num_blocks": len(xs),
            }
            for metric_name, metric_value in [("spearman", sp), ("pearson", pe), ("topk_overlap", overlap), ("top1_hit", top1)]:
                if metric_value is not None:
                    summary_acc[(proxy_name, metric_name)].append(float(metric_value))

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
        "protocol": "stage1_proxy_validity",
        "task": cfg.train.task_name,
        "mode": mode,
        "split": split,
        "max_samples": max_samples,
        "max_batches": max_batches,
        "top_k": top_k,
        "step_size": step_size,
        "fisher_damping": fisher_damping,
        "proxies": proxies,
        "proxy_summaries": {},
    }
    for proxy_name in proxies:
        summary["proxy_summaries"][proxy_name] = {
            "mean_spearman": float(sum(summary_acc[(proxy_name, "spearman")]) / len(summary_acc[(proxy_name, "spearman")])) if summary_acc[(proxy_name, "spearman")] else None,
            "mean_pearson": float(sum(summary_acc[(proxy_name, "pearson")]) / len(summary_acc[(proxy_name, "pearson")])) if summary_acc[(proxy_name, "pearson")] else None,
            "mean_topk_overlap": float(sum(summary_acc[(proxy_name, "topk_overlap")]) / len(summary_acc[(proxy_name, "topk_overlap")])) if summary_acc[(proxy_name, "topk_overlap")] else None,
            "mean_top1_hit_rate": float(sum(summary_acc[(proxy_name, "top1_hit")]) / len(summary_acc[(proxy_name, "top1_hit")])) if summary_acc[(proxy_name, "top1_hit")] else None,
            "num_batches": len(batch_summaries),
        }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, batch_summaries
