from __future__ import annotations

import re
from collections.abc import Iterable

import torch.nn as nn

_LAYER_PATTERNS = [
    re.compile(r"(?:^|\.)(?:model\.layers|layers|transformer\.h)\.(\d+)(?:\.|$)"),
]


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidate_paths = [
        "pretrained_model.model.layers",
        "pretrained_model.layers",
        "pretrained_model.transformer.h",
        "model.layers",
        "transformer.h",
    ]
    for path in candidate_paths:
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError("Could not find decoder layers.")


def extract_block_index(parameter_name: str) -> int | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(parameter_name)
        if match:
            return int(match.group(1))
    return None


def freeze_all_parameters(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


def maybe_enable_gradient_checkpointing(model: nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
