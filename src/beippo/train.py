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

def build_optimizer(model, lr: float, weight_decay: float) -> AdamW:
    # Keep all parameters in the optimizer so selector-based modes can toggle
    # requires_grad dynamically without rebuilding the optimizer under DeepSpeed.
    return AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay)


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
        if selector:
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
            selected_blocks = selector.select_from_gate_grads().selected_blocks
            selector.controller.gates.grad = None
            selector.controller.gates.requires_grad_(False)

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
        accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.ppo.max_grad_norm)
        optimizer.step()
        safe_zero_grad(model)

        trainable, total = count_trainable_parameters(accelerator.unwrap_model(model))
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
