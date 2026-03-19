# bvou-ppo

A research repo for testing **block-selective PPO updates** with **short-output tasks** that are better suited to mechanism validation than long chain-of-thought math rollouts.

This revision adds first-class support for three newer model families:

- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`

and three short-output evaluation tasks:

- `boolq`
- `commonsenseqa`
- `arc_challenge`

The main idea is unchanged:

> use PPO-style value supervision to derive block saliency, then update only a subset of transformer blocks.

What changed is the benchmark design.

Instead of GSM8K-style long reasoning rollouts, this repo now focuses on **short, exact-match outputs** (`yes/no` or `A/B/C/D/E`). That makes it much easier to tell whether block selection itself is useful, without confounding the experiment with long generation windows.

---

## What this repo now supports

### Models

- **Qwen3.5-4B**
- **Qwen3.5-9B**
- **DeepSeek-R1-0528-Qwen3-8B**

All prompts are built with the **tokenizer's official chat template** via `tokenizer.apply_chat_template(...)`.

### Prompt behavior by model family

- **Qwen3.5**
  - uses the tokenizer chat template
  - supports `enable_thinking=False`
  - default in this repo: **thinking disabled** for mechanism validation
- **DeepSeek-R1-0528-Qwen3-8B**
  - uses the tokenizer chat template
  - can optionally inject the official DeepSeek system prompt
  - default in this repo: **official system prompt enabled**, thinking not forced

### Tasks

- **BoolQ**: output only `yes` or `no`
- **CommonsenseQA**: output only `A/B/C/D/E`
- **ARC-Challenge**: output only `A/B/C/D/E`

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

---

## Quick start

### Train with Qwen3.5-4B on BoolQ

```bash
python scripts/train_short_ppo.py --config configs/qwen35_4b_boolq.yaml
```

### Train with Qwen3.5-9B on CommonsenseQA

```bash
python scripts/train_short_ppo.py --config configs/qwen35_9b_commonsenseqa.yaml
```

### Train with DeepSeek-R1-0528-Qwen3-8B on ARC-Challenge

```bash
python scripts/train_short_ppo.py --config configs/deepseek_r1_0528_qwen3_8b_arc.yaml
```

### Evaluate a checkpoint

```bash
python scripts/eval_short_tasks.py \
  --checkpoint outputs/qwen35_4b_boolq/latest \
  --task boolq \
  --split validation
```

---

## Core files

```text
bvou-ppo/
├── configs/
│   ├── qwen35_4b_boolq.yaml
│   ├── qwen35_9b_commonsenseqa.yaml
│   └── deepseek_r1_0528_qwen3_8b_arc.yaml
├── scripts/
│   ├── train_short_ppo.py
│   ├── eval_short_tasks.py
│   └── make_results_table.py
└── src/bvou_ppo/
    ├── config.py
    ├── data.py
    ├── eval.py
    ├── modeling.py
    ├── ppo.py
    ├── prompts.py
    ├── registry.py
    ├── reward.py
    ├── selector.py
    ├── train.py
    ├── utils.py
    └── models/
        └── policy_value_model.py
```

---

## Notes on benchmark design

This repo intentionally uses short-output tasks for the first phase of validation.

That gives you cleaner answers to questions like:

- does block selection preserve accuracy?
- does it reduce trainable-state memory?
- does `bvou + lora` beat plain LoRA on the same task?

Only after that should you move back to long-reasoning settings.

---

## Recommended experiment matrix

For each task, compare:

1. full tuning
2. LoRA tuning
3. BVoU selective tuning
4. BVoU + LoRA

and report:

- validation exact match
- peak GPU memory
- trainable parameter count
- reward mean during training

---

## Limitations

- The `gate_grad` selector is still a proxy for block utility, not an exact counterfactual update-value estimator.
- The default training loop is intentionally minimal and should be treated as a research scaffold, not a production trainer.
- DeepSpeed/FSDP integration is not included in this refreshed version; the focus here is model/task support and prompt correctness.

---

## License

MIT
