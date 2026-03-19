#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from beippo.data import load_task_examples
from beippo.eval import run_task_eval
from beippo.modes import FOUR_MODES
from beippo.models.policy_value_model import PolicyWithValueHead


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def evaluate_run(run_dir: Path, checkpoint_tag: str | None, task_override: str | None, split_override: str | None, max_samples_override: int | None) -> dict:
    cfg = load_run_config(run_dir)
    train_cfg = cfg["train"]

    checkpoint_dir = run_dir / checkpoint_tag if checkpoint_tag else run_dir / "latest"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PolicyWithValueHead(str(checkpoint_dir))
    value_head_path = checkpoint_dir / "value_head.pt"
    if value_head_path.exists():
        model.value_head.load_state_dict(torch.load(value_head_path, map_location="cpu"))
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    task_name = task_override or train_cfg["task_name"]
    split = split_override or train_cfg["eval_splits"][0]
    max_samples = max_samples_override if max_samples_override is not None else train_cfg["max_eval_samples"]

    examples = load_task_examples(
        task_name=task_name,
        split=split,
        max_samples=max_samples,
        tokenizer=tokenizer,
        model_name_or_alias=train_cfg["model_name_or_path"],
        enable_thinking=train_cfg.get("enable_thinking", False),
        use_official_system_prompt=train_cfg.get("use_official_system_prompt", False),
        deepseek_prompt_date=train_cfg.get("deepseek_prompt_date", "2026年3月19日，星期四"),
    )
    result = run_task_eval(
        model,
        tokenizer,
        task_name,
        split,
        examples,
        batch_size=train_cfg["per_device_batch_size"],
        max_new_tokens=train_cfg["response_max_new_tokens"],
    )
    return {
        "run": run_dir.name,
        "task": result.task_name,
        "split": result.split,
        "samples": result.samples,
        "exact_match": result.exact_match,
        "checkpoint": str(checkpoint_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full / lora / bvou / bvou_lora run directories.")
    parser.add_argument("--run-root", type=str, required=True, help="Directory that contains full/lora/bvou/bvou_lora subdirs")
    parser.add_argument("--checkpoint-tag", type=str, default=None, help="Defaults to latest")
    parser.add_argument("--task", type=str, default=None, choices=["boolq", "commonsenseqa", "arc_challenge"])
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--modes", nargs="+", default=list(FOUR_MODES), choices=list(FOUR_MODES))
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows = []
    for mode in args.modes:
        run_dir = run_root / mode
        payload = evaluate_run(run_dir, args.checkpoint_tag, args.task, args.split, args.max_samples)
        rows.append(payload)

    headers = ["run", "task", "split", "samples", "exact_match", "checkpoint"]
    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(row[h]) for h in headers))


if __name__ == "__main__":
    main()