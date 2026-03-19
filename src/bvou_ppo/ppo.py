from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor
    old_values: torch.Tensor
    ref_logprobs: torch.Tensor
    rewards: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    prompt_lengths: torch.Tensor


@dataclass
class PPOLossOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    kl: torch.Tensor


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1e-8)
    return (values * mask).sum() / denom


def shift_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = labels[:, 1:].unsqueeze(-1)
    return log_probs.gather(-1, target).squeeze(-1)


def build_response_mask(attention_mask: torch.Tensor, prompt_lengths: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    mask = torch.zeros(batch_size, seq_len - 1, device=attention_mask.device, dtype=torch.float32)
    for i in range(batch_size):
        start = max(int(prompt_lengths[i].item()) - 1, 0)
        end = int(attention_mask[i].sum().item()) - 1
        if end > start:
            mask[i, start:end] = 1.0
    return mask


def compute_returns_and_advantages(rewards, old_values, response_mask, whiten=True):
    returns = rewards.unsqueeze(-1) * response_mask
    advantages = returns - old_values * response_mask
    if whiten:
        valid = response_mask > 0
        adv = advantages[valid]
        if adv.numel() > 1:
            advantages = advantages.clone()
            advantages[valid] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    return returns, advantages


def ppo_loss(logits, values, rollout: RolloutBatch, clip_range, value_coef, kl_coef) -> PPOLossOutput:
    new_logprobs = shift_logprobs(logits, rollout.input_ids)
    new_values = values[:, :-1]
    mask = rollout.response_mask
    ratio = torch.exp(new_logprobs - rollout.old_logprobs)
    unclipped = ratio * rollout.advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * rollout.advantages
    policy_loss = -masked_mean(torch.minimum(unclipped, clipped), mask)
    value_loss = masked_mean((new_values - rollout.returns) ** 2, mask)
    approx_kl = masked_mean(new_logprobs - rollout.ref_logprobs, mask)
    total = policy_loss + value_coef * value_loss + kl_coef * approx_kl
    return PPOLossOutput(loss=total, policy_loss=policy_loss, value_loss=value_loss, kl=approx_kl)
