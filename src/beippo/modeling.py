from __future__ import annotations

import re
from collections.abc import Iterable

import torch.nn as nn

_LAYER_PATTERNS = [
    re.compile(r"(?:^|\.)(?:model\.layers|layers|transformer\.h)\.(\d+)(?:\.|$)"),
]


def _walk_attr_paths(root: nn.Module, attr_paths: list[str]):
    for path in attr_paths:
        obj = root
        ok = True
        for attr in path.split('.'):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            yield path, obj


def _iter_wrapper_roots(model: nn.Module):
    stack = [model]
    seen: set[int] = set()
    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        yield obj
        for attr in ('pretrained_model', 'base_model', 'model'):
            if hasattr(obj, attr):
                child = getattr(obj, attr)
                if isinstance(child, nn.Module):
                    stack.append(child)


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    candidate_paths = [
        'model.layers',
        'layers',
        'transformer.h',
        'base_model.model.model.layers',
        'base_model.model.layers',
        'base_model.layers',
        'pretrained_model.base_model.model.model.layers',
        'pretrained_model.base_model.model.layers',
        'pretrained_model.model.model.layers',
        'pretrained_model.model.layers',
        'pretrained_model.layers',
        'pretrained_model.transformer.h',
    ]
    for root in _iter_wrapper_roots(model):
        for _path, obj in _walk_attr_paths(root, candidate_paths):
            if isinstance(obj, nn.ModuleList):
                return obj

    # Fallback: search named modules for a plausible decoder stack.
    module_lists: list[tuple[str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            lname = name.lower()
            if any(tok in lname for tok in ('layers', 'h')):
                module_lists.append((name, module))
    if module_lists:
        # Prefer the deepest / longest path to avoid wrapper-level lists.
        module_lists.sort(key=lambda x: (x[0].count('.'), len(x[1])), reverse=True)
        return module_lists[0][1]

    raise ValueError('Could not find decoder layers.')


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
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    if hasattr(model, 'config') and hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
