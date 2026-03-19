from __future__ import annotations

import json
from dataclasses import asdict
from itertools import cycle
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from beippo.config import ExperimentConfig
from beippo.data import TaskExample, load_task_examples
from beippo.eval import run_task_eval
from beippo.rollouts import save_eval_rollouts, save_train_rollouts
from beippo.modeling import count_trainable_parameters, extract_block_index, freeze_all_parameters, maybe_enable_gradient_checkpointing
from beippo.models.policy_value_model import PolicyWithValueHead, apply_lora, build_tokenizer
from beippo.ppo import RolloutBatch, build_response_mask, compute_returns_and_advantages, ppo_loss, shift_logprobs
from beippo.registry import resolve_model
from beippo.reward import exact_match_reward
from beippo.selector import BlockSelector
from beippo.utils import JsonlLogger, peak_memory_gb, reset_peak_memory_stats, set_seed


class ExampleDataset(Dataset):
    def __init__(self, examples: list[TaskExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {"query": ex.query, "gold_label": ex.gold_label, "task_name": ex.task_name}


def collate_examples(batch):
    return batch


def _is_lora_param(name: str) -> bool:
    return "lora_" in name


def configure_trainable_parameters(model, selected_blocks, full_tune, lora_enabled):
    """Toggle requires_grad for the current training mode.

    - full: all params trainable
    - lora: all LoRA params + value head trainable
    - bvou: only selected full-rank blocks + value head trainable
    - bvou_lora: only selected LoRA params inside selected blocks + value head trainable
    """
    freeze_all_parameters(model)
    for p in model.value_head.parameters():
        p.requires_grad_(True)

    using_selector = selected_blocks is not None
    selected = set(selected_blocks or [])

    for name, param in model.named_parameters():
        if name.startswith("value_head"):
            continue

        if full_tune and not using_selector:
            param.requires_grad_(True)
            continue

        if lora_enabled and not using_selector:
            if _is_lora_param(name):
                param.requires_grad_(True)
            continue

        block_idx = extract_block_index(name)
        if block_idx is None or block_idx not in selected:
            continue

        if lora_enabled:
            if _is_lora_param(name):
                param.requires_grad_(True)
        else:
            param.requires_grad_(True)



def safe_zero_grad(obj) -> None:
    zero_grad = getattr(obj, "zero_grad", None)
    if zero_grad is None:
        return
    try:
        zero_grad(set_to_none=True)
    except TypeError:
        zero_grad()


def _iter_monitored_named_params(model, selected_blocks, lora_enabled: bool, include_value_head: bool = False):
    selected = set(selected_blocks or [])
    using_selector = selected_blocks is not None
    for name, param in model.named_parameters():
        if name.startswith("value_head"):
            if include_value_head:
                yield name, param
            continue
        if not param.requires_grad:
            continue
        if not using_selector:
            yield name, param
            continue
        block_idx = extract_block_index(name)
        if block_idx is None or block_idx not in selected:
            continue
        if lora_enabled:
            if _is_lora_param(name):
                yield name, param
        else:
            if not _is_lora_param(name):
                yield name, param


def _grad_l2_and_count(named_params) -> tuple[float, int]:
    total = None
    count = 0
    for _name, param in named_params:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        term = grad.detach().float().pow(2).sum()
        total = term if total is None else total + term.to(total.device)
        count += 1
    if total is None:
        return 0.0, 0
    return float(total.sqrt().item()), count


def _capture_param_probe(named_params, max_per_tensor: int = 16):
    probe = {}
    total_numel = 0
    for name, param in named_params:
        flat = param.detach().view(-1)
        if flat.numel() == 0:
            continue
        k = min(max_per_tensor, flat.numel())
        if k == flat.numel():
            idx = torch.arange(k, device=flat.device)
        else:
            idx = torch.linspace(0, flat.numel() - 1, steps=k, device=flat.device).long()
        vals = flat.index_select(0, idx).float().cpu()
        probe[name] = vals
        total_numel += int(k)
    return probe, total_numel


def _probe_delta_l2(before_probe: dict[str, torch.Tensor], after_probe: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, before in before_probe.items():
        after = after_probe.get(name)
        if after is None:
            continue
        total += float((after - before).pow(2).sum().item())
    return total ** 0.5


def build_optimizer(model, lr: float, weight_decay: float) -> AdamW:
    # Keep all parameters in the optimizer so selector-based modes can toggle
    # requires_grad dynamically without rebuilding the optimizer under DeepSpeed.
    return AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay)


def _named_selector_params(model, lora_enabled: bool):
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


def _select_blocks_from_adv_grad_energy(accelerator, model, rollout: RolloutBatch, selector: BlockSelector, lora_enabled: bool) -> tuple[list[int], list[float]]:
    unwrapped = accelerator.unwrap_model(model)
    freeze_all_parameters(unwrapped)

    named_params = []
    for name, param, block_idx in _named_selector_params(unwrapped, lora_enabled=lora_enabled):
        param.requires_grad_(True)
        named_params.append((name, param, block_idx))

    if not named_params:
        zero_scores = torch.zeros_like(selector.controller.gates.detach(), dtype=torch.float32)
        result = selector.select_from_block_scores(zero_scores)
        return result.selected_blocks, result.scores

    outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
    new_logprobs = shift_logprobs(outputs.logits, rollout.input_ids)
    mask = rollout.response_mask
    denom = mask.sum().clamp_min(1e-8)
    # First-order proxy for block expected improvement:
    #   g_b = E[A_t * grad_theta_b log pi(a_t | s_t)]
    #   U_b ~ eta * ||g_b||^2
    objective = ((new_logprobs * rollout.advantages * mask).sum() / denom)

    grads = torch.autograd.grad(
        objective,
        [param for _, param, _ in named_params],
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    scores = torch.zeros_like(
        selector.controller.gates.detach(),
        dtype=torch.float32,
        device=accelerator.device,
    )
    for grad, (_, _param, block_idx) in zip(grads, named_params):
        if grad is None:
            continue
        block_energy = grad.detach().float().pow(2).sum()
        if block_energy.device != scores.device:
            block_energy = block_energy.to(scores.device)
        scores[block_idx] += block_energy

    scores = accelerator.reduce(scores, reduction="mean")
    result = selector.select_from_block_scores(scores)
    return result.selected_blocks, result.scores


def _current_lr(step: int, total_steps: int, base_lr: float, warmup_ratio: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def save_checkpoint(accelerator, model, tokenizer, output_dir: Path, step: int | None = None, tag: str | None = None, update_latest: bool = True) -> Path | None:
    if not accelerator.is_main_process:
        return None
    import shutil

    if tag is not None:
        ckpt_dir = output_dir / tag
    elif step is not None:
        ckpt_dir = output_dir / f"step-{step}"
    else:
        raise ValueError("Either step or tag must be provided.")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    if update_latest:
        latest = output_dir / "latest"
        if latest.exists():
            if latest.is_symlink() or latest.is_file():
                latest.unlink()
            else:
                shutil.rmtree(latest)
        shutil.copytree(ckpt_dir, latest)
    return ckpt_dir


@torch.no_grad()
def generate_rollout_batch(accelerator, model, reference_model, tokenizer, batch, cfg: ExperimentConfig):
    device = accelerator.device
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
    generations = accelerator.unwrap_model(model).generate(
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

    responses = tokenizer.batch_decode(generations[:, prompt_tokens.input_ids.shape[1]:], skip_special_tokens=True)
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
    return rollout, responses


def run_evals(accelerator, model, tokenizer, cfg: ExperimentConfig, logger: JsonlLogger, step: int, phase: str, write_summary_to: Path | None = None) -> list[dict]:
    results_payloads: list[dict] = []
    if not accelerator.is_main_process:
        return results_payloads

    for split in cfg.train.eval_splits:
        eval_examples = load_task_examples(
            task_name=cfg.train.task_name,
            split=split,
            max_samples=cfg.train.max_eval_samples,
            tokenizer=tokenizer,
            model_name_or_alias=cfg.train.model_name_or_path,
            enable_thinking=cfg.train.enable_thinking,
            use_official_system_prompt=cfg.train.use_official_system_prompt,
            deepseek_prompt_date=cfg.train.deepseek_prompt_date,
        )
        result, rollout_records = run_task_eval(
            accelerator.unwrap_model(model),
            tokenizer,
            cfg.train.task_name,
            split,
            eval_examples,
            batch_size=cfg.train.per_device_batch_size,
            max_new_tokens=cfg.train.response_max_new_tokens,
            collect_rollouts=cfg.rollouts.save_eval_rollouts,
        )
        if cfg.rollouts.max_eval_rollouts_per_save > 0:
            rollout_records = rollout_records[: cfg.rollouts.max_eval_rollouts_per_save]
        if cfg.rollouts.save_eval_rollouts:
            save_eval_rollouts(
                output_dir=cfg.train.output_dir,
                step=step,
                phase=phase,
                task_name=result.task_name,
                split=result.split,
                records=rollout_records,
            )
        payload = {
            "step": step,
            "phase": phase,
            "is_baseline": phase == "baseline_eval",
            "task": result.task_name,
            "split": result.split,
            "exact_match": result.exact_match,
            "samples": result.samples,
            "saved_eval_rollouts": bool(cfg.rollouts.save_eval_rollouts),
        }
        logger.log(payload)
        results_payloads.append(payload)

    if write_summary_to is not None:
        write_summary_to.parent.mkdir(parents=True, exist_ok=True)
        with write_summary_to.open("w", encoding="utf-8") as f:
            json.dump(results_payloads, f, ensure_ascii=False, indent=2)

    return results_payloads


def run_training(cfg: ExperimentConfig) -> None:
    accelerator = Accelerator(gradient_accumulation_steps=cfg.train.gradient_accumulation_steps)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output_dir)
    set_seed(cfg.train.seed + accelerator.process_index)

    tokenizer = build_tokenizer(cfg.train.model_name_or_path)
    model = PolicyWithValueHead(cfg.train.model_name_or_path)
    if cfg.lora.enabled:
        model = apply_lora(model, cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout, cfg.lora.target_modules)
    reference_model = AutoModelForCausalLM.from_pretrained(
        cfg.train.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    reference_model.eval()
    if cfg.train.gradient_checkpointing:
        maybe_enable_gradient_checkpointing(model.pretrained_model)

    train_examples = load_task_examples(
        task_name=cfg.train.task_name,
        split=cfg.train.train_split,
        max_samples=cfg.train.max_train_samples,
        tokenizer=tokenizer,
        model_name_or_alias=cfg.train.model_name_or_path,
        enable_thinking=cfg.train.enable_thinking,
        use_official_system_prompt=cfg.train.use_official_system_prompt,
        deepseek_prompt_date=cfg.train.deepseek_prompt_date,
    )
    train_loader = DataLoader(ExampleDataset(train_examples), batch_size=cfg.train.per_device_batch_size, shuffle=True, collate_fn=collate_examples)

    configure_trainable_parameters(model, None, cfg.train.full_tune and not cfg.selector.enabled, cfg.lora.enabled and not cfg.selector.enabled)
    optimizer = build_optimizer(model, cfg.train.learning_rate, cfg.train.weight_decay)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    reference_model.to(accelerator.device)

    selector = None
    if cfg.selector.enabled:
        selector = BlockSelector(accelerator.unwrap_model(model), top_k=cfg.selector.top_k, search_upper_half_only=cfg.selector.search_upper_half_only)

    if accelerator.is_main_process:
        with (output_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    accelerator.wait_for_everyone()
    save_checkpoint(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        tag="init",
        update_latest=False,
    )
    accelerator.wait_for_everyone()
    run_evals(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        logger=logger,
        step=0,
        phase="baseline_eval",
        write_summary_to=output_dir / "baseline_metrics.json",
    )
    accelerator.wait_for_everyone()

    progress = tqdm(range(cfg.train.num_train_steps), disable=not accelerator.is_local_main_process)
    step_iter = cycle(train_loader)
    for step in progress:
        reset_peak_memory_stats()
        batch = next(step_iter)
        lr = _current_lr(step, cfg.train.num_train_steps, cfg.train.learning_rate, cfg.train.warmup_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr

        rollout, responses = generate_rollout_batch(accelerator, model, reference_model, tokenizer, batch, cfg)

        selected_blocks: list[int] = []
        selector_scores: list[float] = []
        if selector:
            if cfg.selector.scorer == "adv_grad_energy":
                selected_blocks, selector_scores = _select_blocks_from_adv_grad_energy(
                    accelerator=accelerator,
                    model=model,
                    rollout=rollout,
                    selector=selector,
                    lora_enabled=cfg.lora.enabled,
                )
            elif cfg.selector.scorer == "gate_grad":
                freeze_all_parameters(accelerator.unwrap_model(model))
                selector.controller.gates.requires_grad_(True)
                outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
                scout_loss = ppo_loss(outputs.logits, outputs.values, rollout, cfg.ppo.clip_range, cfg.ppo.value_coef, cfg.ppo.kl_coef)
                gate_grads = torch.autograd.grad(
                    scout_loss.loss,
                    selector.controller.gates,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if gate_grads is None:
                    gate_grads = torch.zeros_like(selector.controller.gates)
                gate_grads = accelerator.reduce(gate_grads.detach(), reduction="mean")
                selector.controller.gates.grad = gate_grads
                result = selector.select_from_gate_grads()
                selected_blocks = result.selected_blocks
                selector_scores = result.scores
                selector.controller.gates.grad = None
                selector.controller.gates.requires_grad_(False)
            else:
                raise ValueError(f"Unsupported selector scorer: {cfg.selector.scorer}")

        configure_trainable_parameters(
            accelerator.unwrap_model(model),
            selected_blocks if selector else None,
            cfg.train.full_tune and not selector,
            cfg.lora.enabled,
        )

        outputs = model(input_ids=rollout.input_ids, attention_mask=rollout.attention_mask)
        loss_out = ppo_loss(outputs.logits, outputs.values, rollout, cfg.ppo.clip_range, cfg.ppo.value_coef, cfg.ppo.kl_coef)
        safe_zero_grad(optimizer)
        accelerator.backward(loss_out.loss)

        unwrapped = accelerator.unwrap_model(model)
        monitored_named_params = list(_iter_monitored_named_params(
            unwrapped,
            selected_blocks if selector else None,
            lora_enabled=cfg.lora.enabled,
            include_value_head=False,
        ))
        value_head_named_params = list(_iter_monitored_named_params(
            unwrapped,
            selected_blocks if selector else None,
            lora_enabled=cfg.lora.enabled,
            include_value_head=True,
        ))
        monitored_grad_l2, monitored_grad_tensors = _grad_l2_and_count(monitored_named_params)
        value_head_grad_l2, value_head_grad_tensors = _grad_l2_and_count(
            [(n, p) for n, p in value_head_named_params if n.startswith("value_head")]
        )
        monitored_probe_before, monitored_probe_numel = _capture_param_probe(monitored_named_params)
        value_head_probe_before, value_head_probe_numel = _capture_param_probe(
            [(n, p) for n, p in value_head_named_params if n.startswith("value_head")]
        )

        accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.ppo.max_grad_norm)
        optimizer.step()

        monitored_probe_after, _ = _capture_param_probe(monitored_named_params)
        value_head_probe_after, _ = _capture_param_probe(
            [(n, p) for n, p in value_head_named_params if n.startswith("value_head")]
        )
        monitored_probe_delta_l2 = _probe_delta_l2(monitored_probe_before, monitored_probe_after)
        value_head_probe_delta_l2 = _probe_delta_l2(value_head_probe_before, value_head_probe_after)

        safe_zero_grad(model)

        trainable, total = count_trainable_parameters(unwrapped)
        metrics = {
            "step": step + 1,
            "task": cfg.train.task_name,
            "reward_mean": float(rollout.rewards.mean().item()),
            "policy_loss": float(loss_out.policy_loss.item()),
            "value_loss": float(loss_out.value_loss.item()),
            "approx_kl": float(loss_out.kl.item()),
            "peak_memory_gb": peak_memory_gb(),
            "trainable_params": trainable,
            "total_params": total,
            "selected_blocks": selected_blocks,
            "selector_scorer": cfg.selector.scorer if selector else None,
            "selector_scores": selector_scores,
            "monitored_param_scope": "selected_blocks" if selector else "trainable",
            "monitored_param_grad_l2": monitored_grad_l2,
            "monitored_param_grad_tensors": monitored_grad_tensors,
            "monitored_param_probe_numel": monitored_probe_numel,
            "monitored_param_probe_delta_l2": monitored_probe_delta_l2,
            "value_head_grad_l2": value_head_grad_l2,
            "value_head_grad_tensors": value_head_grad_tensors,
            "value_head_probe_numel": value_head_probe_numel,
            "value_head_probe_delta_l2": value_head_probe_delta_l2,
        }
        if accelerator.is_main_process and cfg.rollouts.save_train_rollouts:
            limited_batch = batch
            limited_responses = responses
            limited_rewards = rollout.rewards.detach().float().cpu().tolist()
            if cfg.rollouts.max_train_rollouts_per_save > 0:
                k = cfg.rollouts.max_train_rollouts_per_save
                limited_batch = batch[:k]
                limited_responses = responses[:k]
                limited_rewards = limited_rewards[:k]
            train_rollout_path = save_train_rollouts(
                output_dir=cfg.train.output_dir,
                step=step + 1,
                batch=limited_batch,
                responses=limited_responses,
                rewards=limited_rewards,
                selected_blocks=selected_blocks,
                model_name_or_path=cfg.train.model_name_or_path,
                task_name=cfg.train.task_name,
            )
            metrics['train_rollout_file'] = str(train_rollout_path)
        if accelerator.is_main_process:
            logger.log(metrics)
        progress.set_postfix(reward=f"{metrics['reward_mean']:.3f}", mem=f"{metrics['peak_memory_gb']:.1f}G")

        if (step + 1) % cfg.train.eval_every == 0:
            accelerator.wait_for_everyone()
            run_evals(
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                logger=logger,
                step=step + 1,
                phase="train_eval",
            )
            accelerator.wait_for_everyone()

        if (step + 1) % cfg.train.save_every == 0:
            accelerator.wait_for_everyone()
            save_checkpoint(accelerator, model, tokenizer, output_dir, step=step + 1)
            accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    save_checkpoint(accelerator, model, tokenizer, output_dir, step=cfg.train.num_train_steps)
    if selector is not None:
        selector.close()
