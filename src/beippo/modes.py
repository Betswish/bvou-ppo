from __future__ import annotations

from copy import deepcopy

from beippo.config import ExperimentConfig


FOUR_MODES = ("full", "lora", "bvou", "bvou_lora")


class UnknownModeError(ValueError):
    pass


def apply_mode(cfg: ExperimentConfig, mode: str) -> ExperimentConfig:
    mode = mode.lower()
    cfg = deepcopy(cfg)

    if mode == "full":
        cfg.train.full_tune = True
        cfg.lora.enabled = False
        cfg.selector.enabled = False
    elif mode == "lora":
        cfg.train.full_tune = False
        cfg.lora.enabled = True
        cfg.selector.enabled = False
    elif mode == "bvou":
        cfg.train.full_tune = False
        cfg.lora.enabled = False
        cfg.selector.enabled = True
    elif mode == "bvou_lora":
        cfg.train.full_tune = False
        cfg.lora.enabled = True
        cfg.selector.enabled = True
    else:
        raise UnknownModeError(f"Unknown mode: {mode}")

    return cfg


def mode_suffix(mode: str) -> str:
    if mode not in FOUR_MODES:
        raise UnknownModeError(f"Unknown mode: {mode}")
    return mode
