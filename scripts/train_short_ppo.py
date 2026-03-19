#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse

from beippo.config import load_config
from beippo.modes import apply_mode, FOUR_MODES
from beippo.train import run_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=FOUR_MODES, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--task", type=str, default=None, choices=["boolq", "commonsenseqa", "arc_challenge"])
    parser.add_argument("--model-id", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode is not None:
        cfg = apply_mode(cfg, args.mode)
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    if args.run_name is not None:
        cfg.train.run_name = args.run_name
    if args.task is not None:
        cfg.train.task_name = args.task
    if args.model_id is not None:
        cfg.train.model_name_or_path = args.model_id

    run_training(cfg)