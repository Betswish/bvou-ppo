#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import subprocess
import tempfile
from copy import deepcopy

import yaml

from beippo.config import load_config
from beippo.modes import apply_mode


FULLRANK_BASELINES = [
    ("full", None),
    ("bvou", "adv_grad_energy"),
    ("bvou", "fisher_diag_energy"),
    ("bvou", "grad_norm"),
    ("bvou", "lisa_score"),
    ("bvou", "adagradselect_score"),
]

LORA_BASELINES = [
    ("lora", None),
    ("bvou_lora", "adv_grad_energy"),
    ("bvou_lora", "fisher_diag_energy"),
    ("bvou_lora", "grad_norm"),
    ("bvou_lora", "lisa_score"),
    ("bvou_lora", "adagradselect_score"),
]


def _variant_name(mode: str, scorer: str | None) -> str:
    if scorer is None:
        return mode
    return f"{mode}_{scorer}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stage-2 end-to-end baselines for the paper protocol.")
    parser.add_argument("--config", type=str, required=True, help="Base experiment YAML.")
    parser.add_argument("--route", choices=["fullrank", "lora"], default="fullrank")
    parser.add_argument("--launcher", choices=["python", "accelerate"], default="python")
    parser.add_argument("--deepspeed-config", type=str, default="deepspeed/zero2.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    baselines = FULLRANK_BASELINES if args.route == "fullrank" else LORA_BASELINES

    out_root = Path(cfg.train.output_dir).with_name(Path(cfg.train.output_dir).name + f"_stage2_{args.route}")
    cfg_dir = out_root / "generated_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    for mode, scorer in baselines:
        run_cfg = apply_mode(deepcopy(cfg), mode)
        if scorer is not None:
            run_cfg.selector.enabled = True
            run_cfg.selector.scorer = scorer
        run_name = _variant_name(mode, scorer)
        run_cfg.train.run_name = run_name
        run_cfg.train.output_dir = str(out_root / run_name)
        yaml_path = cfg_dir / f"{run_name}.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump({
                "train": run_cfg.train.__dict__,
                "ppo": run_cfg.ppo.__dict__,
                "lora": run_cfg.lora.__dict__,
                "selector": run_cfg.selector.__dict__,
                "rollouts": run_cfg.rollouts.__dict__,
            }, f, sort_keys=False, allow_unicode=True)

        if args.launcher == "python":
            cmd = [sys.executable, "scripts/train_short_ppo.py", "--config", str(yaml_path)]
        else:
            cmd = [
                "accelerate", "launch", "--use_deepspeed", "--deepspeed_config_file", args.deepspeed_config,
                "scripts/train_short_ppo.py", "--config", str(yaml_path)
            ]
        commands.append(cmd)

    for cmd in commands:
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
