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
from pathlib import Path

from beippo.modes import FOUR_MODES


def split_prefix_and_mode(config_path: Path) -> tuple[str, str]:
    stem = config_path.stem
    for mode in sorted(FOUR_MODES, key=len, reverse=True):
        suffix = f"_{mode}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], mode
    raise ValueError(
        f"Config filename must end with one of: {', '.join(FOUR_MODES)}. Got: {config_path.name}"
    )


def build_command(mode_config_path: Path, launcher: str, deepspeed_config: str | None) -> list[str]:
    if launcher == "python":
        return [sys.executable, "scripts/train_short_ppo.py", "--config", str(mode_config_path)]
    if launcher == "accelerate":
        if not deepspeed_config:
            raise ValueError("--deepspeed-config is required when launcher=accelerate")
        return [
            "accelerate",
            "launch",
            "--use_deepspeed",
            "--deepspeed_config_file",
            deepspeed_config,
            "scripts/train_short_ppo.py",
            "--config",
            str(mode_config_path),
        ]
    raise ValueError(f"Unsupported launcher: {launcher}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full / lora / bvou / bvou_lora for one model-task pair.")
    parser.add_argument("--config", type=str, required=True, help="Any one explicit mode config YAML")
    parser.add_argument("--launcher", choices=["python", "accelerate"], default="python")
    parser.add_argument("--deepspeed-config", type=str, default="deepspeed/zero2.json")
    parser.add_argument("--modes", nargs="+", default=list(FOUR_MODES), choices=list(FOUR_MODES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    anchor_config = Path(args.config)
    config_dir = anchor_config.parent
    prefix, _anchor_mode = split_prefix_and_mode(anchor_config)

    run_dirs: list[str] = []
    for mode in args.modes:
        mode_config_path = config_dir / f"{prefix}_{mode}.yaml"
        if not mode_config_path.exists():
            raise FileNotFoundError(f"Missing config for mode {mode}: {mode_config_path}")
        run_dirs.append(f"outputs/{prefix}_{mode}")
        cmd = build_command(mode_config_path, args.launcher, args.deepspeed_config)
        print("$", " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            message = f"Mode {mode} failed with exit code {result.returncode}."
            if args.continue_on_error:
                print(message, file=sys.stderr)
                continue
            raise SystemExit(message)

    if not args.dry_run:
        summary_cmd = [sys.executable, "scripts/make_results_table.py", *run_dirs]
        print("$", " ".join(summary_cmd), flush=True)
        subprocess.run(summary_cmd, cwd=REPO_ROOT, check=False)


if __name__ == "__main__":
    main()
