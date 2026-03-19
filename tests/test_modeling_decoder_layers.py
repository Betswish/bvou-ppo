from __future__ import annotations

import torch.nn as nn

from beippo.modeling import get_decoder_layers


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock() for _ in range(3)])


class PeftLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = Backbone()


def test_get_decoder_layers_through_peft_like_wrapper():
    wrapped = PeftLike()
    layers = get_decoder_layers(wrapped)
    assert isinstance(layers, nn.ModuleList)
    assert len(layers) == 3
