from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PolicyValueOutput:
    logits: torch.Tensor
    values: torch.Tensor


class PolicyWithValueHead(nn.Module):
    def __init__(self, model_name_or_path: str, value_head_init_std: float = 0.02) -> None:
        super().__init__()
        self.pretrained_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )
        hidden_size = self.pretrained_model.config.hidden_size
        backbone_param = next(self.pretrained_model.parameters())
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        self.value_head.to(device=backbone_param.device, dtype=backbone_param.dtype)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=value_head_init_std)

    def forward(self, *args, **kwargs) -> PolicyValueOutput:
        kwargs.setdefault("output_hidden_states", True)
        kwargs.setdefault("return_dict", True)
        kwargs.setdefault("use_cache", False)
        outputs = self.pretrained_model(*args, **kwargs)
        last_hidden = outputs.hidden_states[-1]
        if last_hidden.dtype != self.value_head.weight.dtype:
            last_hidden = last_hidden.to(self.value_head.weight.dtype)
        values = self.value_head(last_hidden).squeeze(-1)
        return PolicyValueOutput(logits=outputs.logits, values=values)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.pretrained_model.generate(*args, **kwargs)

    def save_pretrained(self, output_dir: str) -> None:
        self.pretrained_model.save_pretrained(output_dir)
        torch.save(self.value_head.state_dict(), f"{output_dir}/value_head.pt")


def build_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def apply_lora(model: PolicyWithValueHead, r: int, alpha: int, dropout: float, target_modules: list[str]):
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model.pretrained_model = get_peft_model(model.pretrained_model, peft_config)
    return model
