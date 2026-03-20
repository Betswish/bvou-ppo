#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Allow running either inside the main repo or from this extracted folder.
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / 'src'
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from beippo.config import load_config
from proxy_validity_stage1 import run_proxy_validation_stage1


def main():
    parser = argparse.ArgumentParser(description='Stage-1 proxy validation for block utility proxies.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['full', 'lora', 'bvou', 'bvou_lora'], default='bvou')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--max-samples', type=int, default=64)
    parser.add_argument('--max-batches', type=int, default=2)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--step-size', type=float, default=1e-4)
    parser.add_argument('--fisher-damping', type=float, default=1e-8)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--proxies', nargs='*', default=None,
                        help='Optional subset of proxies, e.g. adv_grad_energy fisher_diag_energy grad_norm lisa_score adagradselect_score random')
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary, _ = run_proxy_validation_stage1(
        cfg=cfg,
        checkpoint=args.checkpoint,
        split=args.split,
        max_samples=args.max_samples,
        max_batches=args.max_batches,
        mode=args.mode,
        top_k=args.top_k,
        step_size=args.step_size,
        fisher_damping=args.fisher_damping,
        proxies=args.proxies,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
