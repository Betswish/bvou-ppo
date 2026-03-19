# beippo

A research scaffold for testing **block-selective PPO updates** on **short-output classification-style tasks**.

This repo is built around one question:

> Can PPO-style value supervision be turned into a useful **block selection signal**, so we update only part of a decoder-only LLM while preserving task performance?

This version focuses on **short, exact-match outputs** like `yes/no` or `A/B/C/D/E`. That keeps the experiment centered on the **update-allocation mechanism** rather than long generation windows.

---

## What is in this repo

### Supported models

- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`

### Supported tasks

- `boolq`
- `commonsenseqa`
- `arc_challenge`

### Supported training modes

- `full`
- `lora`
- `bvou`
- `bvou_lora`

All tasks are evaluated with exact-match accuracy.

---

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

---

## Launchers: why there are both Accelerate and DeepSpeed configs

The repo uses:

- **Accelerate** as the launcher / distributed orchestration layer
- **DeepSpeed** as the ZeRO backend

So a command like this:

```bash
accelerate launch --config_file accelerate/zero2.yaml ...
```

means:

- `accelerate/zero2.yaml` controls how Accelerate launches the job
- that YAML points to `deepspeed/zero2.json`
- `deepspeed/zero2.json` controls the actual ZeRO partitioning behavior

Available launcher configs:

- `accelerate/zero1.yaml`
- `accelerate/zero2.yaml`
- `accelerate/zero3.yaml`

Available DeepSpeed backends:

- `deepspeed/zero1.json`
- `deepspeed/zero2.json`
- `deepspeed/zero3.json`

For most runs, start with **ZeRO-2**.

---

## Config layout

The experiment matrix lives under:

```text
configs/matrix/
```

Each config is fully explicit and already fixes:

- model
- task
- mode
- output directory
- run name
- thinking/system-prompt behavior
- train split / eval splits
- training sample count / evaluation sample count
- train steps / eval cadence / save cadence
- batch size / grad accumulation
- prompt length / response length
- learning rate / weight decay / warmup ratio
- bf16 / gradient checkpointing
- selector settings
- rollout-saving settings

Config naming pattern:

```text
configs/matrix/<model>_<task>_<mode>.yaml
```

Examples:

- `configs/matrix/qwen35_4b_boolq_full.yaml`
- `configs/matrix/qwen35_4b_boolq_lora.yaml`
- `configs/matrix/qwen35_4b_boolq_bvou.yaml`
- `configs/matrix/qwen35_4b_boolq_bvou_lora.yaml`


## What is now explicit in every matrix YAML

All 36 files under `configs/matrix/` now explicitly include the core experiment-budget fields instead of relying on hidden defaults.

At minimum, every matrix YAML now spells out:

```yaml
train:
  train_split: train
  eval_splits: [validation]
  max_train_samples: ...
  max_eval_samples: ...
  num_train_steps: ...
  per_device_batch_size: ...
  gradient_accumulation_steps: ...
  prompt_max_length: 768
  response_max_new_tokens: 8
  learning_rate: ...
  weight_decay: 0.01
  warmup_ratio: 0.03
  bf16: true
  gradient_checkpointing: true
  eval_every: 100
  save_every: 100
```

That means you can now inspect a single YAML and immediately know:

- how many training samples were used
- how many evaluation samples were used
- how many optimizer steps were run
- what effective batch/accumulation setup was used
- what sequence-length budget was allowed

There is no need to look up these values in code defaults for the official matrix runs.

## Default budget choices used in the matrix configs

The matrix configs currently use these task/model budgets:

### Task sample counts

- `boolq`: `max_train_samples = 5000`, `max_eval_samples = 500`
- `commonsenseqa`: `max_train_samples = 8000`, `max_eval_samples = 1000`
- `arc_challenge`: `max_train_samples = 2250`, `max_eval_samples = 570`

### Shared training/eval cadence

- `num_train_steps = 500`
- `eval_every = 100`
- `save_every = 100`
- `prompt_max_length = 768`
- `response_max_new_tokens = 8`
- `weight_decay = 0.01`
- `warmup_ratio = 0.03`
- `bf16 = true`

### Batch sizing by model family

- `Qwen/Qwen3.5-4B`: `per_device_batch_size = 2`, `gradient_accumulation_steps = 8`
- `Qwen/Qwen3.5-9B`: `per_device_batch_size = 1`, `gradient_accumulation_steps = 16`
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`: `per_device_batch_size = 1`, `gradient_accumulation_steps = 16`

If you want to change the experiment budget, edit the YAML directly; for the official matrix, the YAML is now the source of truth.

---

## Full experiment matrix

### Models

- `qwen35_4b`
- `qwen35_9b`
- `deepseek_r1_0528_qwen3_8b`

### Tasks

- `boolq`
- `commonsenseqa`
- `arc_challenge`

### Modes

- `full`
- `lora`
- `bvou`
- `bvou_lora`

That gives **36 ready-to-run YAML files**.

### All config file names

#### Qwen3.5-4B

- `configs/matrix/qwen35_4b_boolq_full.yaml`
- `configs/matrix/qwen35_4b_boolq_lora.yaml`
- `configs/matrix/qwen35_4b_boolq_bvou.yaml`
- `configs/matrix/qwen35_4b_boolq_bvou_lora.yaml`
- `configs/matrix/qwen35_4b_commonsenseqa_full.yaml`
- `configs/matrix/qwen35_4b_commonsenseqa_lora.yaml`
- `configs/matrix/qwen35_4b_commonsenseqa_bvou.yaml`
- `configs/matrix/qwen35_4b_commonsenseqa_bvou_lora.yaml`
- `configs/matrix/qwen35_4b_arc_challenge_full.yaml`
- `configs/matrix/qwen35_4b_arc_challenge_lora.yaml`
- `configs/matrix/qwen35_4b_arc_challenge_bvou.yaml`
- `configs/matrix/qwen35_4b_arc_challenge_bvou_lora.yaml`

#### Qwen3.5-9B

- `configs/matrix/qwen35_9b_boolq_full.yaml`
- `configs/matrix/qwen35_9b_boolq_lora.yaml`
- `configs/matrix/qwen35_9b_boolq_bvou.yaml`
- `configs/matrix/qwen35_9b_boolq_bvou_lora.yaml`
- `configs/matrix/qwen35_9b_commonsenseqa_full.yaml`
- `configs/matrix/qwen35_9b_commonsenseqa_lora.yaml`
- `configs/matrix/qwen35_9b_commonsenseqa_bvou.yaml`
- `configs/matrix/qwen35_9b_commonsenseqa_bvou_lora.yaml`
- `configs/matrix/qwen35_9b_arc_challenge_full.yaml`
- `configs/matrix/qwen35_9b_arc_challenge_lora.yaml`
- `configs/matrix/qwen35_9b_arc_challenge_bvou.yaml`
- `configs/matrix/qwen35_9b_arc_challenge_bvou_lora.yaml`

#### DeepSeek-R1-0528-Qwen3-8B

- `configs/matrix/deepseek_r1_0528_qwen3_8b_boolq_full.yaml`
- `configs/matrix/deepseek_r1_0528_qwen3_8b_boolq_lora.yaml`
- `configs/matrix/deepseek_r1_0528_qwen3_8b_boolq_bvou.yaml`
- `configs/matrix/deepseek_r1_0528_qwen3_8b_boolq_bvou_lora.yaml`
- `configs/matrix/deepseek_r1_0528_qwen3_8b_commonsenseqa_full.yaml`
- `configs/matrix/deepseek_r1_0528_qwen3_8b_commonsenseqa_lora.yaml`
- `configs/matrix/deepseek_r1_0528_qwen3_8b_commonsenseqa_bvou.yaml`
- `configs/matrix/deepseek_r1_0528_qwen3_8b_commonsenseqa_bvou_lora.yaml`
- `configs/matrix/deepseek_r1_0528_qwen3_8b_arc_challenge_full.yaml`
- `configs/matrix/deepseek_r1_0528_qwen3_8b_arc_challenge_lora.yaml`
- `configs/matrix/deepseek_r1_0528_qwen3_8b_arc_challenge_bvou.yaml`
- `configs/matrix/deepseek_r1_0528_qwen3_8b_arc_challenge_bvou_lora.yaml`

---

## How to run a single config

### Python launcher

Example: Qwen3.5-4B + BoolQ + full tuning

```bash
python scripts/train_short_ppo.py \
  --config configs/matrix/qwen35_4b_boolq_full.yaml
```

Example: Qwen3.5-9B + CommonsenseQA + BVoU+LoRA

```bash
python scripts/train_short_ppo.py \
  --config configs/matrix/qwen35_9b_commonsenseqa_bvou_lora.yaml
```

Example: DeepSeek-R1-0528-Qwen3-8B + ARC-Challenge + LoRA

```bash
python scripts/train_short_ppo.py \
  --config configs/matrix/deepseek_r1_0528_qwen3_8b_arc_challenge_lora.yaml
```

### Accelerate + DeepSpeed launcher

Example with ZeRO-2:

```bash
accelerate launch --config_file accelerate/zero2.yaml \
  scripts/train_short_ppo.py \
  --config configs/matrix/qwen35_4b_boolq_full.yaml
```

Example with ZeRO-3:

```bash
accelerate launch --config_file accelerate/zero3.yaml \
  scripts/train_short_ppo.py \
  --config configs/matrix/qwen35_9b_commonsenseqa_bvou.yaml
```

---

## Recommended launch choices

- `Qwen/Qwen3.5-4B`: start with `accelerate/zero2.yaml`
- `Qwen/Qwen3.5-9B`: prefer `accelerate/zero2.yaml`, move to `zero3` if needed
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`: start with `accelerate/zero2.yaml`

If you are debugging or checking that the code path works, start with **ZeRO-1**.

---

## Baseline and step=0 evaluation

Before any training updates, the trainer automatically:

1. saves an untrained checkpoint at:

```text
<output_dir>/init/
```

2. runs baseline evaluation and records it as:

- `phase = baseline_eval`
- `step = 0`

Files written include:

- `<output_dir>/baseline_metrics.json`
- `<output_dir>/metrics.jsonl`

---

## Rollout saving

The repo can save both training and evaluation rollouts.

Config section:

```yaml
rollouts:
  save_train_rollouts: true
  save_eval_rollouts: true
  max_train_rollouts_per_save: 0
  max_eval_rollouts_per_save: 0
```

Recommended for short-output tasks:

- save all eval rollouts
- save all train rollouts if disk is acceptable

Train rollouts are written to:

```text
<output_dir>/rollouts/train/
```

Eval rollouts are written to:

```text
<output_dir>/rollouts/eval/<task>/<split>/
```

---

## How to evaluate a trained checkpoint

Example:

```bash
python scripts/eval_short_tasks.py \
  --checkpoint outputs/qwen35_4b_boolq_full/latest \
  --model-id Qwen/Qwen3.5-4B \
  --task boolq \
  --split validation
```

Save eval rollouts too:

```bash
python scripts/eval_short_tasks.py \
  --checkpoint outputs/qwen35_4b_boolq_full/latest \
  --model-id Qwen/Qwen3.5-4B \
  --task boolq \
  --split validation \
  --save-rollouts
```

---

## How to run four modes automatically for one model/task pair

If you want one base config to generate and run all four modes automatically, use:

```bash
python scripts/run_four_modes.py \
  --config configs/qwen35_4b_boolq.yaml
```

With Accelerate:

```bash
python scripts/run_four_modes.py \
  --config configs/qwen35_9b_commonsenseqa.yaml \
  --launcher accelerate \
  --accelerate-config accelerate/zero2.yaml
```

But if you want **fully explicit experiment bookkeeping**, prefer the `configs/matrix/*.yaml` files instead.

---

## How to evaluate four modes automatically

```bash
python scripts/eval_four_modes.py \
  --run-root outputs/qwen35_4b_boolq_four_modes
```

---

## Output structure

A single explicit config run writes to its own output directory, for example:

```text
outputs/qwen35_4b_boolq_full/
```

Inside you will typically see:

```text
init/
latest/
step-100/
step-200/
...
baseline_metrics.json
config.json
metrics.jsonl
rollouts/
```

---

## Repo layout

```text
beippo/
├── accelerate/
│   ├── zero1.yaml
│   ├── zero2.yaml
│   └── zero3.yaml
├── configs/
│   ├── qwen35_4b_boolq.yaml
│   ├── qwen35_9b_commonsenseqa.yaml
│   ├── deepseek_r1_0528_qwen3_8b_arc.yaml
│   └── matrix/
│       └── 36 explicit model-task-mode YAMLs
├── deepspeed/
│   ├── zero1.json
│   ├── zero2.json
│   └── zero3.json
├── scripts/
│   ├── train_short_ppo.py
│   ├── eval_short_tasks.py
│   ├── run_four_modes.py
│   ├── eval_four_modes.py
│   └── make_results_table.py
├── src/
│   └── beippo/
└── tests/
```

---

## Practical recommendation

For the cleanest comparison, run the explicit matrix files directly. Each matrix YAML is now self-contained with explicit sample counts, train steps, batch sizing, and sequence-length budget.

Good first three commands:

```bash
accelerate launch --config_file accelerate/zero2.yaml \
  scripts/train_short_ppo.py \
  --config configs/matrix/qwen35_4b_boolq_full.yaml

accelerate launch --config_file accelerate/zero2.yaml \
  scripts/train_short_ppo.py \
  --config configs/matrix/qwen35_4b_boolq_lora.yaml

accelerate launch --config_file accelerate/zero2.yaml \
  scripts/train_short_ppo.py \
  --config configs/matrix/qwen35_4b_boolq_bvou_lora.yaml
```

Then expand to the other 33 configs.
