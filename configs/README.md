# Single-GPU Qwen3.5-4B configs

This folder contains 12 single-GPU experiment configs for:

- Model: `Qwen/Qwen3.5-4B`
- Tasks:
  - `boolq`
  - `commonsenseqa`
  - `arc_challenge`
- Training modes:
  - `full`
  - `lora`
  - `bvou`
  - `bvou_lora`

## Important update

All BVoU-style configs now use **all transformer layers as candidate layers**.

Concretely, every BVoU / BVoU+LoRA YAML now sets:

```yaml
selector:
  candidate_last_n_layers: null
  candidate_start_layer: null
  candidate_end_layer: null
```

This means block selection is performed over the full model depth rather than only the last few layers.

## Install

From the repo root:

```bash
python -m pip uninstall -y beippo bvou-ppo
python -m pip install -e .
```

## Run examples

Full tuning on BoolQ:

```bash
python scripts/train_short_ppo.py   --config configs/qwen35_4b_boolq_full.yaml
```

LoRA on CommonsenseQA:

```bash
python scripts/train_short_ppo.py   --config configs/qwen35_4b_commonsenseqa_lora.yaml
```

Pure BVoU on ARC-Challenge:

```bash
python scripts/train_short_ppo.py   --config configs/qwen35_4b_arc_challenge_bvou.yaml
```

BVoU+LoRA on BoolQ:

```bash
python scripts/train_short_ppo.py   --config configs/qwen35_4b_boolq_bvou_lora.yaml
```

## Notes

- `full` and `lora` do not use selector candidate-layer settings.
- `bvou` and `bvou_lora` now search over **all layers**.
- `active_only_optimizer_state: true` is intended for single-process use.
- For single-GPU runs, do not use DeepSpeed launchers.
