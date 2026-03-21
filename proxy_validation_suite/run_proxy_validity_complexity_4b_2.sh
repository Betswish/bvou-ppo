#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_proxy_validity_complexity_4b.sh
#
# This script runs three RL/RLVR-common reasoning datasets intended as
# low/medium/high complexity buckets for proxy validation:
#   low    -> gsm8k
#   medium -> math
#   long   -> aime_2024
#
# Notes:
# - GSM8K and MATH use the HF `test` split for validation-style analysis.
# - AIME 2024 commonly exposes a single `train` split on public mirrors.
# - The task names map to:
#     gsm8k, math, aime_2024

# CUDA_VISIBLE_DEVICES=0 python run_proxy_validation_stage1.py   \
#   --config configs/qwen35_4b_gsm8k_bvou.yaml \
#   --mode bvou \
#   --split test \
#   --max-samples 256 \
#   --max-batches 32 \
#   --top-k 4

# CUDA_VISIBLE_DEVICES=0 python run_proxy_validation_stage1.py   \
#   --config configs/qwen35_4b_math_bvou.yaml \
#   --mode bvou \
#   --split test \
#   --max-samples 256 \
#   --max-batches 32 \
#   --top-k 4

# CUDA_VISIBLE_DEVICES=0 python run_proxy_validation_stage1.py   \
#   --config configs/qwen35_4b_aime_2024_bvou.yaml \
#   --mode bvou \
#   --split train \
#   --max-samples 30 \
#   --max-batches 15 \
#   --top-k 4

CUDA_VISIBLE_DEVICES=1 python run_proxy_validation_stage1.py   \
  --config configs/qwen35_4b_gsm8k_bvou_lora.yaml \
  --mode bvou_lora \
  --split test \
  --max-samples 256 \
  --max-batches 32 \
  --top-k 4

CUDA_VISIBLE_DEVICES=1 python run_proxy_validation_stage1.py   \
  --config configs/qwen35_4b_math_bvou_lora.yaml \
  --mode bvou_lora \
  --split test \
  --max-samples 256 \
  --max-batches 32 \
  --top-k 4

CUDA_VISIBLE_DEVICES=1 python run_proxy_validation_stage1.py   \
  --config configs/qwen35_4b_aime_2024_bvou_lora.yaml \
  --mode bvou_lora \
  --split train \
  --max-samples 30 \
  --max-batches 15 \
  --top-k 4
