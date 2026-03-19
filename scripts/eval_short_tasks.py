#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse

import torch
from transformers import AutoTokenizer

from beippo.data import load_task_examples
from beippo.eval import run_task_eval
from beippo.models.policy_value_model import PolicyWithValueHead
from beippo.rollouts import save_eval_rollouts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["boolq", "commonsenseqa", "arc_challenge"])
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--no-official-system-prompt", action="store_true")
    parser.add_argument("--deepseek-prompt-date", type=str, default="2026年3月19日，星期四")
    parser.add_argument("--model-id", type=str, default=None, help="Original Hub model id or registry alias for prompt-family inference.")
    parser.add_argument("--save-rollouts", action="store_true")
    parser.add_argument("--rollout-output", type=str, default=None)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = PolicyWithValueHead(str(ckpt))
    value_head_path = ckpt / "value_head.pt"
    if value_head_path.exists():
        model.value_head.load_state_dict(torch.load(value_head_path, map_location="cpu"))
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    examples = load_task_examples(
        task_name=args.task,
        split=args.split,
        max_samples=args.max_samples,
        tokenizer=tokenizer,
        model_name_or_alias=args.model_id or str(ckpt),
        enable_thinking=args.enable_thinking,
        use_official_system_prompt=not args.no_official_system_prompt,
        deepseek_prompt_date=args.deepseek_prompt_date,
    )
    result, rollout_records = run_task_eval(
        model,
        tokenizer,
        args.task,
        args.split,
        examples,
        collect_rollouts=args.save_rollouts,
    )
    if args.save_rollouts:
        out_root = args.rollout_output or str(ckpt)
        path = save_eval_rollouts(out_root, step=0, phase="manual_eval", task_name=args.task, split=args.split, records=rollout_records)
        print({"saved_rollouts": str(path)})
    print({"task": result.task_name, "split": result.split, "samples": result.samples, "exact_match": result.exact_match})
