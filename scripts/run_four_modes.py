#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from bvou_ppo.config import load_config
from bvou_ppo.modes import FOUR_MODES, apply_mode


def build_command(mode_config_path: Path, launcher: str, accelerate_config: str | None) -> list[str]:
    if launcher == "python":
        return [sys.executable, "scripts/train_short_ppo.py", "--config", str(mode_config_path)]
    if launcher == "accelerate":
        if not accelerate_config:
            raise ValueError("--accelerate-config is required when launcher=accelerate")
        return [
            "accelerate",
            "launch",
            "--config_file",
            accelerate_config,
            "scripts/train_short_ppo.py",
            "--config",
            str(mode_config_path),
        ]
    raise ValueError(f"Unsupported launcher: {launcher}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full / lora / bvou / bvou_lora sequentially.")
    parser.add_argument("--config", type=str, required=True, help="Base config YAML")
    parser.add_argument("--launcher", choices=["python", "accelerate"], default="python")
    parser.add_argument("--accelerate-config", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None, help="Root directory for the four runs")
    parser.add_argument("--generated-config-dir", type=str, default=None, help="Where to write derived YAML files")
    parser.add_argument("--modes", nargs="+", default=list(FOUR_MODES), choices=list(FOUR_MODES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    base_config_path = Path(args.config)
    base_stem = base_config_path.stem

    output_root = Path(args.output_root) if args.output_root else Path(base_cfg.train.output_dir).parent / f"{base_stem}_four_modes"
    generated_config_dir = Path(args.generated_config_dir) if args.generated_config_dir else output_root / "generated_configs"
    output_root.mkdir(parents=True, exist_ok=True)
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    run_dirs: list[str] = []
    for mode in args.modes:
        cfg = apply_mode(base_cfg, mode)
        cfg.train.run_name = f"{base_cfg.train.run_name}_{mode}"
        cfg.train.output_dir = str(output_root / mode)
        run_dirs.append(cfg.train.output_dir)

        mode_config_path = generated_config_dir / f"{base_stem}__{mode}.yaml"
        with mode_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(cfg), f, allow_unicode=True, sort_keys=False)

        cmd = build_command(mode_config_path, args.launcher, args.accelerate_config)
        print("$", " ".join(cmd), flush=True)
        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1])
        if result.returncode != 0:
            message = f"Mode {mode} failed with exit code {result.returncode}."
            if args.continue_on_error:
                print(message, file=sys.stderr)
                continue
            raise SystemExit(message)

    if not args.dry_run:
        summary_cmd = [sys.executable, "scripts/make_results_table.py", *run_dirs]
        print("$", " ".join(summary_cmd), flush=True)
        subprocess.run(summary_cmd, cwd=Path(__file__).resolve().parents[1], check=False)


if __name__ == "__main__":
    main()
