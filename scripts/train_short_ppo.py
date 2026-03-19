#!/usr/bin/env python
from __future__ import annotations

import argparse

from bvou_ppo.config import load_config
from bvou_ppo.train import run_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_training(load_config(args.config))
