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

from beippo.config import load_config
from beippo.modes import FOUR_MODES, apply_mode
from beippo.proxy_validity import run_proxy_validity_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate block-utility proxies against actual one-step gains.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=FOUR_MODES, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional local checkpoint to analyze instead of the base model.")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=8, help="Take the union of top-k blocks from each proxy before measuring true gains.")
    parser.add_argument("--step-size", type=float, default=1e-4, help="One-step gain validation step size.")
    parser.add_argument("--fisher-damping", type=float, default=1e-8)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode
    if mode is None:
        if cfg.selector.enabled and cfg.lora.enabled:
            mode = "bvou_lora"
        elif cfg.selector.enabled:
            mode = "bvou"
        elif cfg.lora.enabled:
            mode = "lora"
        else:
            mode = "full"
    cfg = apply_mode(cfg, mode)

    summary, _ = run_proxy_validity_experiment(
        cfg=cfg,
        checkpoint=args.checkpoint,
        split=args.split,
        max_samples=args.max_samples,
        max_batches=args.max_batches,
        mode=mode,
        top_k=args.top_k,
        step_size=args.step_size,
        fisher_damping=args.fisher_damping,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
