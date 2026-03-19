from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


@dataclass
class SelectorConfig:
    enabled: bool = False
    scorer: str = "adv_grad_energy"
    top_k: int = 6
    select_every: int = 1
    search_upper_half_only: bool = False
    active_only_optimizer_state: bool = False


@dataclass
class PPOConfig:
    clip_range: float = 0.2
    value_coef: float = 0.5
    kl_coef: float = 0.02
    whiten_advantages: bool = True
    max_grad_norm: float = 1.0


@dataclass
class TrainConfig:
    seed: int = 42
    output_dir: str = "outputs/default"
    run_name: str = "default"
    model_name_or_path: str = "Qwen/Qwen3.5-4B"
    model_family: str = "qwen35"
    task_name: str = "boolq"
    train_split: str = "train"
    eval_splits: list[str] = field(default_factory=lambda: ["validation"])
    max_train_samples: int = 5000
    max_eval_samples: int = 500
    prompt_max_length: int = 768
    response_max_new_tokens: int = 8
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_steps: int = 500
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    bf16: bool = True
    gradient_checkpointing: bool = True
    save_every: int = 100
    eval_every: int = 100
    full_tune: bool = False
    report_to: str = "none"
    enable_thinking: bool = False
    use_official_system_prompt: bool = True
    deepseek_prompt_date: str = "2026年3月19日，星期四"


@dataclass
class RolloutSaveConfig:
    save_train_rollouts: bool = False
    save_eval_rollouts: bool = False
    max_train_rollouts_per_save: int = 0
    max_eval_rollouts_per_save: int = 0


@dataclass
class ExperimentConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    rollouts: RolloutSaveConfig = field(default_factory=RolloutSaveConfig)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = {
        "train": TrainConfig().__dict__,
        "ppo": PPOConfig().__dict__,
        "lora": LoRAConfig().__dict__,
        "selector": SelectorConfig().__dict__,
        "rollouts": RolloutSaveConfig().__dict__,
    }
    merged = _deep_update(base, raw)
    return ExperimentConfig(
        train=TrainConfig(**merged["train"]),
        ppo=PPOConfig(**merged["ppo"]),
        lora=LoRAConfig(**merged["lora"]),
        selector=SelectorConfig(**merged["selector"]),
        rollouts=RolloutSaveConfig(**merged["rollouts"]),
    )
