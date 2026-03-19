# bvou-ppo

A research scaffold for testing **block-selective PPO updates** on **short-output classification-style tasks**.

This repo is built around one question:

> Can PPO-style value supervision be turned into a useful **block selection signal**, so we update only part of a decoder-only LLM while preserving task performance?

Instead of long chain-of-thought math rollouts, this version focuses on **short, exact-match outputs** like `yes/no` or `A/B/C/D/E`. That keeps the experiment centered on the **update-allocation mechanism** rather than long generation windows.

---

## What is in this repo

### Supported model families

This repo currently has first-class configs and prompt handling for:

- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`

### Supported tasks

- `boolq`
- `commonsenseqa`
- `arc_challenge`

All three are evaluated with **exact-match reward / exact-match accuracy**.

### Supported training modes

The repo supports four comparison modes:

1. `full` — full-parameter PPO tuning
2. `lora` — PPO tuning with LoRA adapters
3. `bvou` — block-selective PPO tuning
4. `bvou_lora` — block-selective PPO with LoRA adapters

These same four modes can be run either one by one or through the batch runner.

---

## Core idea

The motivating object is a **block update utility**:

\[
U_b(\theta, \Delta\theta_b) = J(\theta + \Delta\theta_b) - J(\theta)
\]

where only block `b` is updated.

In practice, this repo does **not** assume PPO's standard value head is already a block-value estimator. Instead, it uses the PPO objective to derive a **block saliency proxy**, then updates only the top-k blocks.

The default selector is a **gate-gradient scout pass**:

- add one scalar gate per transformer block
- backpropagate the PPO loss to those gates
- rank blocks by gate-gradient magnitude
- update only the selected blocks

This is a **research proxy**, not a theorem-proof exact counterfactual estimator.

---

## Prompting behavior

All prompts are rendered with the model tokenizer's **official chat template** through:

```python
 tokenizer.apply_chat_template(...)
```

### Qwen3.5

- uses the official chat template
- supports `enable_thinking=False`
- default in the provided configs: **thinking off**
- default in the provided configs: **no extra official system prompt**

### DeepSeek-R1-0528-Qwen3-8B

- uses the official chat template
- default config currently sets:
  - `enable_thinking: false`
  - `use_official_system_prompt: true`
- the official system prompt can be disabled in config with:

```yaml
train:
  use_official_system_prompt: false
```

The DeepSeek prompt date string is configurable through:

```yaml
train:
  deepseek_prompt_date: 2026年3月19日，星期四
```

---

## Repo layout

```text
bvou-ppo/
├── configs/
│   ├── qwen35_4b_boolq.yaml
│   ├── qwen35_9b_commonsenseqa.yaml
│   └── deepseek_r1_0528_qwen3_8b_arc.yaml
├── scripts/
│   ├── train_short_ppo.py
│   ├── eval_short_tasks.py
│   ├── run_four_modes.py
│   ├── eval_four_modes.py
│   └── make_results_table.py
├── src/bvou_ppo/
│   ├── config.py
│   ├── data.py
│   ├── eval.py
│   ├── modeling.py
│   ├── modes.py
│   ├── ppo.py
│   ├── prompts.py
│   ├── registry.py
│   ├── reward.py
│   ├── selector.py
│   ├── train.py
│   ├── utils.py
│   └── models/
│       └── policy_value_model.py
└── tests/
    ├── test_modes.py
    ├── test_prompts.py
    ├── test_registry.py
    └── test_reward.py
```

---

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

---

## Quick start

### 1) Train a single config directly

```bash
python scripts/train_short_ppo.py --config configs/qwen35_4b_boolq.yaml
```

### 2) Override the mode from the command line

```bash
python scripts/train_short_ppo.py \
  --config configs/qwen35_4b_boolq.yaml \
  --mode bvou_lora
```

Supported values for `--mode`:

- `full`
- `lora`
- `bvou`
- `bvou_lora`

### 3) Evaluate a single checkpoint

```bash
python scripts/eval_short_tasks.py \
  --checkpoint outputs/qwen35_4b_boolq/latest \
  --model-id Qwen/Qwen3.5-4B \
  --task boolq \
  --split validation
```

---

## Run all four modes automatically

The easiest way to compare methods is:

```bash
python scripts/run_four_modes.py \
  --config configs/qwen35_4b_boolq.yaml
```

This generates derived configs and runs:

- `full`
- `lora`
- `bvou`
- `bvou_lora`

By default it writes results under a directory like:

```text
outputs/<base_config_name>_four_modes/
```

### Dry run

```bash
python scripts/run_four_modes.py \
  --config configs/qwen35_4b_boolq.yaml \
  --dry-run
```

### Use Accelerate launcher

```bash
python scripts/run_four_modes.py \
  --config configs/qwen35_9b_commonsenseqa.yaml \
  --launcher accelerate \
  --accelerate-config path/to/accelerate_config.yaml
```

---

## Evaluate all four modes automatically

```bash
python scripts/eval_four_modes.py \
  --run-root outputs/qwen35_4b_boolq_four_modes
```

You can also override the task and split:

```bash
python scripts/eval_four_modes.py \
  --run-root outputs/qwen35_9b_commonsenseqa_four_modes \
  --task commonsenseqa \
  --split validation
```

---

## Baseline saving and evaluation

Before training starts, the trainer now does two things automatically:

1. saves an **untrained checkpoint** to:

```text
<output_dir>/init/
```

2. runs **baseline evaluation** before any updates and writes:

```text
<output_dir>/baseline_metrics.json
```

The same baseline rows are also written into `metrics.jsonl` with:

- `phase = "baseline_eval"`
- `is_baseline = true`
- `step = -1`

This means every run directory contains both:

- a true pre-training checkpoint
- pre-training accuracy numbers

---

## Output files per run

A typical run directory contains:

```text
outputs/<run_name>/
├── config.json
├── metrics.jsonl
├── baseline_metrics.json
├── init/
├── latest/
└── step-*/
```

### What is in `metrics.jsonl`

It includes both training-time logs and eval logs, including:

- `reward_mean`
- `policy_loss`
- `value_loss`
- `approx_kl`
- `peak_memory_gb`
- `trainable_params`
- `selected_blocks`
- baseline eval rows
- training eval rows

---

## Summarize results

You can summarize one or more run directories with:

```bash
python scripts/make_results_table.py outputs/qwen35_4b_boolq
```

or multiple runs:

```bash
python scripts/make_results_table.py \
  outputs/qwen35_4b_boolq_four_modes/full \
  outputs/qwen35_4b_boolq_four_modes/lora \
  outputs/qwen35_4b_boolq_four_modes/bvou \
  outputs/qwen35_4b_boolq_four_modes/bvou_lora
```

The summary table includes:

- baseline accuracy columns like `baseline_boolq_validation`
- latest eval accuracy columns like `boolq_validation`
- `peak_memory_gb`
- `reward_mean_last`

---

## Recommended experiment order

A practical sequence is:

1. `Qwen3.5-4B` on `boolq`
2. `Qwen3.5-9B` on `commonsenseqa`
3. `DeepSeek-R1-0528-Qwen3-8B` on `arc_challenge`

For each, compare:

- `full`
- `lora`
- `bvou`
- `bvou_lora`

Main things to report:

- baseline exact match
- final exact match
- mean reward during training
- peak GPU memory
- trainable parameter count

---

## Important limitations

- `gate_grad` is still a **proxy** for block utility, not an exact update-value oracle.
- The training loop is intentionally lightweight and should be treated as a **research scaffold**.
- The default implementation rebuilds the optimizer in the update step, which is simple and practical for comparison, but not the final word on training efficiency.
- This repo is designed for decoder-only Hugging Face chat models with recognizable block structure; other architectures may need small adjustments.

---

## License

MIT
