from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from bvou_ppo.modeling import get_decoder_layers


@dataclass
class SelectionResult:
    selected_blocks: list[int]
    scores: list[float]


class BlockGateController(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.layers = get_decoder_layers(model)
        self.gates = nn.Parameter(torch.ones(len(self.layers)), requires_grad=False)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def install(self) -> None:
        if self._handles:
            return
        for idx, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(idx)))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            gate = self.gates[idx]
            if isinstance(output, tuple):
                return (output[0] * gate, *output[1:])
            return output * gate
        return hook

    def saliency(self) -> torch.Tensor:
        if self.gates.grad is None:
            return torch.zeros_like(self.gates)
        return self.gates.grad.detach().abs().float().cpu()


class BlockSelector:
    def __init__(self, model: nn.Module, top_k: int = 6, search_upper_half_only: bool = False) -> None:
        self.controller = BlockGateController(model)
        self.controller.install()
        self.top_k = top_k
        self.search_upper_half_only = search_upper_half_only

    def close(self) -> None:
        self.controller.remove()

    def select_from_gate_grads(self) -> SelectionResult:
        scores = self.controller.saliency()
        if self.search_upper_half_only:
            midpoint = scores.numel() // 2
            scores[:midpoint] = -1
        k = min(self.top_k, scores.numel())
        values, indices = torch.topk(scores, k=k)
        return SelectionResult(selected_blocks=indices.tolist(), scores=values.tolist())
