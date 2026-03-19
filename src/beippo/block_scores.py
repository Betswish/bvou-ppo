from __future__ import annotations

from collections import defaultdict

import torch

from beippo.modeling import extract_block_index, freeze_all_parameters, get_decoder_layers
from beippo.ppo import RolloutBatch, masked_mean, ppo_loss, shift_logprobs
from beippo.selector import BlockSelector


def _is_lora_param(name: str) -> bool:
    return "lora_" in name


def num_blocks(model) -> int:
    return len(get_decoder_layers(model))


def empty_scores(model, device: torch.device | None = None) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    return torch.zeros(num_blocks(model), dtype=torch.float32, device=device)


def score_tensor_to_dict(scores: torch.Tensor) -> dict[int, float]:
    scores = scores.detach().float().cpu()
    return {idx: float(val) for idx, val in enumerate(scores.tolist())}


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


def _enable_selector_params(model, lora_enabled: bool):
    freeze_all_parameters(model)
    named_params = []
    for name, param, block_idx in _selector_named_params(model, lora_enabled=lora_enabled):
        param.requires_grad_(True)
        named_params.append((name, param, block_idx))
    return named_params


def _score_objective_adv_pg(model, rollout: RolloutBatch) -> torch.Tensor:
    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    new_logprobs = shift_logprobs(outputs.logits, rollout.input_ids)
    return masked_mean(rollout.advantages * new_logprobs, rollout.response_mask)


def _score_objective_mean_logp(model, rollout: RolloutBatch) -> torch.Tensor:
    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    new_logprobs = shift_logprobs(outputs.logits, rollout.input_ids)
    return masked_mean(new_logprobs, rollout.response_mask)


def _grads_by_block(named_params, grads):
    per_block: dict[int, list[torch.Tensor]] = defaultdict(list)
    for (_name, _param, block_idx), grad in zip(named_params, grads):
        if grad is None:
            continue
        per_block[block_idx].append(grad.detach().float())
    return per_block


def compute_adv_grad_energy_scores(model, rollout: RolloutBatch, lora_enabled: bool) -> torch.Tensor:
    named_params = _enable_selector_params(model, lora_enabled=lora_enabled)
    scores = empty_scores(model)
    if not named_params:
        return scores
    objective = _score_objective_adv_pg(model, rollout)
    grads = torch.autograd.grad(objective, [p for _, p, _ in named_params], retain_graph=False, create_graph=False, allow_unused=True)
    per_block = _grads_by_block(named_params, grads)
    for block_idx, tensors in per_block.items():
        block_score = torch.zeros((), dtype=torch.float32, device=scores.device)
        for grad in tensors:
            block_score = block_score + grad.to(scores.device).pow(2).sum()
        scores[block_idx] = block_score
    freeze_all_parameters(model)
    return scores


def compute_grad_norm_scores(model, rollout: RolloutBatch, lora_enabled: bool, clip_range: float, value_coef: float, kl_coef: float) -> torch.Tensor:
    named_params = _enable_selector_params(model, lora_enabled=lora_enabled)
    scores = empty_scores(model)
    if not named_params:
        return scores
    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    loss_out = ppo_loss(outputs.logits, outputs.values, rollout, clip_range, value_coef, kl_coef)
    grads = torch.autograd.grad(loss_out.loss, [p for _, p, _ in named_params], retain_graph=False, create_graph=False, allow_unused=True)
    per_block = _grads_by_block(named_params, grads)
    for block_idx, tensors in per_block.items():
        block_sq = torch.zeros((), dtype=torch.float32, device=scores.device)
        for grad in tensors:
            block_sq = block_sq + grad.to(scores.device).pow(2).sum()
        scores[block_idx] = block_sq.sqrt() if block_sq.item() > 0 else block_sq
    freeze_all_parameters(model)
    return scores


def compute_lisa_scores(model, rollout: RolloutBatch, lora_enabled: bool, clip_range: float, value_coef: float, kl_coef: float) -> torch.Tensor:
    """A size-normalized layer importance proxy.

    We approximate LISA-style layer importance with the mean absolute PPO gradient
    magnitude per block, rather than raw L2 norm that scales with block size.
    """
    named_params = _enable_selector_params(model, lora_enabled=lora_enabled)
    scores = empty_scores(model)
    if not named_params:
        return scores
    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    loss_out = ppo_loss(outputs.logits, outputs.values, rollout, clip_range, value_coef, kl_coef)
    grads = torch.autograd.grad(loss_out.loss, [p for _, p, _ in named_params], retain_graph=False, create_graph=False, allow_unused=True)
    per_block = _grads_by_block(named_params, grads)
    for block_idx, tensors in per_block.items():
        total_abs = torch.zeros((), dtype=torch.float32, device=scores.device)
        total_numel = 0
        for grad in tensors:
            grad = grad.to(scores.device)
            total_abs = total_abs + grad.abs().sum()
            total_numel += int(grad.numel())
        if total_numel > 0:
            scores[block_idx] = total_abs / float(total_numel)
    freeze_all_parameters(model)
    return scores


def compute_adagradselect_scores(model, rollout: RolloutBatch, lora_enabled: bool, clip_range: float, value_coef: float, kl_coef: float, eps: float = 1e-12) -> torch.Tensor:
    """A size-normalized adaptive-gradient proxy.

    We use block RMS gradient as a simple AdaGrad-style score: sqrt(mean(g^2)).
    This differs from raw grad_norm by removing direct dependence on block size.
    """
    named_params = _enable_selector_params(model, lora_enabled=lora_enabled)
    scores = empty_scores(model)
    if not named_params:
        return scores
    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    loss_out = ppo_loss(outputs.logits, outputs.values, rollout, clip_range, value_coef, kl_coef)
    grads = torch.autograd.grad(loss_out.loss, [p for _, p, _ in named_params], retain_graph=False, create_graph=False, allow_unused=True)
    per_block = _grads_by_block(named_params, grads)
    for block_idx, tensors in per_block.items():
        total_sq = torch.zeros((), dtype=torch.float32, device=scores.device)
        total_numel = 0
        for grad in tensors:
            grad = grad.to(scores.device)
            total_sq = total_sq + grad.pow(2).sum()
            total_numel += int(grad.numel())
        if total_numel > 0:
            scores[block_idx] = torch.sqrt(total_sq / float(total_numel) + eps)
    freeze_all_parameters(model)
    return scores


def compute_fisher_diag_energy_scores(model, rollout: RolloutBatch, lora_enabled: bool, damping: float = 1e-8) -> torch.Tensor:
    named_params = _enable_selector_params(model, lora_enabled=lora_enabled)
    scores = empty_scores(model)
    if not named_params:
        return scores
    g_obj = _score_objective_adv_pg(model, rollout)
    fisher_obj = _score_objective_mean_logp(model, rollout)
    params = [p for _, p, _ in named_params]
    g_grads = torch.autograd.grad(g_obj, params, retain_graph=True, create_graph=False, allow_unused=True)
    fisher_grads = torch.autograd.grad(fisher_obj, params, retain_graph=False, create_graph=False, allow_unused=True)
    for (_name, _param, block_idx), g_grad, f_grad in zip(named_params, g_grads, fisher_grads):
        if g_grad is None:
            continue
        g2 = g_grad.detach().float()
        g2 = g2.to(scores.device).pow(2)
        if f_grad is None:
            fdiag = torch.zeros_like(g2)
        else:
            fdiag = f_grad.detach().float().to(scores.device).pow(2)
        scores[block_idx] = scores[block_idx] + (g2 / (fdiag + damping)).sum()
    freeze_all_parameters(model)
    return scores


def compute_gate_grad_scores(model, rollout: RolloutBatch, clip_range: float, value_coef: float, kl_coef: float) -> torch.Tensor:
    freeze_all_parameters(model)
    selector = BlockSelector(model, top_k=num_blocks(model))
    selector.controller.gates.requires_grad_(True)
    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    loss_out = ppo_loss(outputs.logits, outputs.values, rollout, clip_range, value_coef, kl_coef)
    gate_grads = torch.autograd.grad(loss_out.loss, selector.controller.gates, retain_graph=False, create_graph=False, allow_unused=True)[0]
    if gate_grads is None:
        gate_grads = torch.zeros_like(selector.controller.gates)
    scores = gate_grads.detach().abs().float().to(next(model.parameters()).device)
    selector.close()
    freeze_all_parameters(model)
    return scores


def compute_random_scores(model, seed: int = 0) -> torch.Tensor:
    device = next(model.parameters()).device
    g = torch.Generator(device=device if device.type == "cpu" else "cpu")
    g.manual_seed(int(seed))
    scores = torch.rand(num_blocks(model), generator=g, dtype=torch.float32)
    return scores.to(device)
