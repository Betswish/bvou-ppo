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

## Launching: what uses Accelerate vs DeepSpeed

The training code uses `Accelerator()` inside Python, so the natural launcher is still:

```bash
accelerate launch ...
```

But for this repo, the **primary recommended invocation** is now:

```bash
accelerate launch   --use_deepspeed   --deepspeed_config_file deepspeed/zero2.json   scripts/train_short_ppo.py   --config configs/qwen35_4b_boolq_full.yaml
```

That means:

- **Accelerate** is the launcher / process orchestration layer
- **DeepSpeed JSON** is the ZeRO backend configuration

You do **not** need a separate `accelerate/*.yaml` to run this repo.
The optional `accelerate/zero1.yaml`, `accelerate/zero2.yaml`, `accelerate/zero3.yaml` files are kept only as examples for users who prefer file-based launcher configs.

### Available DeepSpeed ZeRO backends

- `deepspeed/zero1.json`
- `deepspeed/zero2.json`
- `deepspeed/zero3.json`

Recommended starting point:

- use `deepspeed/zero2.json` first
- drop to `zero1` for debugging
- move to `zero3` only if memory is still too tight

---

## Config layout

All experiment configs now live directly under:

```text
configs/
```

There are **36 explicit YAML files** covering:

- 3 models
- 3 tasks
- 4 modes

Config naming pattern:

```text
configs/<model>_<task>_<mode>.yaml
```

Examples:

- `configs/qwen35_4b_boolq_full.yaml`
- `configs/qwen35_4b_boolq_lora.yaml`
- `configs/qwen35_4b_boolq_bvou.yaml`
- `configs/qwen35_4b_boolq_bvou_lora.yaml`

The old three base configs have been removed. There is no longer a separate `configs/matrix/` directory.

---

## What is explicit in every experiment YAML

All 36 YAMLs explicitly contain the core experiment-budget and generation-behavior fields.

At minimum, every file includes:

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
  enable_thinking: false
  use_official_system_prompt: false
rollouts:
  save_train_rollouts: false
  save_eval_rollouts: false
  max_train_rollouts_per_save: 0
  max_eval_rollouts_per_save: 0
```

So you can inspect a single YAML and immediately know:

- the train/eval sample counts
- the train-step budget
- the batch/accumulation setup
- the prompt/response length budget
- whether thinking mode is off
- whether rollouts are saved

---

## Default budget choices used in the 36 YAMLs

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
- `rollouts.save_train_rollouts = false`
- `rollouts.save_eval_rollouts = false`

### Batch sizing by model family

- `Qwen/Qwen3.5-4B`: `per_device_batch_size = 2`, `gradient_accumulation_steps = 8`
- `Qwen/Qwen3.5-9B`: `per_device_batch_size = 1`, `gradient_accumulation_steps = 16`
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`: `per_device_batch_size = 1`, `gradient_accumulation_steps = 16`

---

## Full experiment matrix

### Qwen3.5-4B

- `configs/qwen35_4b_boolq_full.yaml`
- `configs/qwen35_4b_boolq_lora.yaml`
- `configs/qwen35_4b_boolq_bvou.yaml`
- `configs/qwen35_4b_boolq_bvou_lora.yaml`
- `configs/qwen35_4b_commonsenseqa_full.yaml`
- `configs/qwen35_4b_commonsenseqa_lora.yaml`
- `configs/qwen35_4b_commonsenseqa_bvou.yaml`
- `configs/qwen35_4b_commonsenseqa_bvou_lora.yaml`
- `configs/qwen35_4b_arc_challenge_full.yaml`
- `configs/qwen35_4b_arc_challenge_lora.yaml`
- `configs/qwen35_4b_arc_challenge_bvou.yaml`
- `configs/qwen35_4b_arc_challenge_bvou_lora.yaml`

### Qwen3.5-9B

- `configs/qwen35_9b_boolq_full.yaml`
- `configs/qwen35_9b_boolq_lora.yaml`
- `configs/qwen35_9b_boolq_bvou.yaml`
- `configs/qwen35_9b_boolq_bvou_lora.yaml`
- `configs/qwen35_9b_commonsenseqa_full.yaml`
- `configs/qwen35_9b_commonsenseqa_lora.yaml`
- `configs/qwen35_9b_commonsenseqa_bvou.yaml`
- `configs/qwen35_9b_commonsenseqa_bvou_lora.yaml`
- `configs/qwen35_9b_arc_challenge_full.yaml`
- `configs/qwen35_9b_arc_challenge_lora.yaml`
- `configs/qwen35_9b_arc_challenge_bvou.yaml`
- `configs/qwen35_9b_arc_challenge_bvou_lora.yaml`

### DeepSeek-R1-0528-Qwen3-8B

- `configs/deepseek_r1_0528_qwen3_8b_boolq_full.yaml`
- `configs/deepseek_r1_0528_qwen3_8b_boolq_lora.yaml`
- `configs/deepseek_r1_0528_qwen3_8b_boolq_bvou.yaml`
- `configs/deepseek_r1_0528_qwen3_8b_boolq_bvou_lora.yaml`
- `configs/deepseek_r1_0528_qwen3_8b_commonsenseqa_full.yaml`
- `configs/deepseek_r1_0528_qwen3_8b_commonsenseqa_lora.yaml`
- `configs/deepseek_r1_0528_qwen3_8b_commonsenseqa_bvou.yaml`
- `configs/deepseek_r1_0528_qwen3_8b_commonsenseqa_bvou_lora.yaml`
- `configs/deepseek_r1_0528_qwen3_8b_arc_challenge_full.yaml`
- `configs/deepseek_r1_0528_qwen3_8b_arc_challenge_lora.yaml`
- `configs/deepseek_r1_0528_qwen3_8b_arc_challenge_bvou.yaml`
- `configs/deepseek_r1_0528_qwen3_8b_arc_challenge_bvou_lora.yaml`

---

## How to run a single experiment

### Plain Python

```bash
python scripts/train_short_ppo.py   --config configs/qwen35_4b_boolq_full.yaml
```

### Recommended: Accelerate + DeepSpeed JSON

```bash
accelerate launch   --use_deepspeed   --deepspeed_config_file deepspeed/zero2.json   scripts/train_short_ppo.py   --config configs/qwen35_4b_boolq_full.yaml
```

More examples:

```bash
accelerate launch   --use_deepspeed   --deepspeed_config_file deepspeed/zero2.json   scripts/train_short_ppo.py   --config configs/qwen35_9b_commonsenseqa_bvou_lora.yaml
```

```bash
accelerate launch   --use_deepspeed   --deepspeed_config_file deepspeed/zero2.json   scripts/train_short_ppo.py   --config configs/deepseek_r1_0528_qwen3_8b_arc_challenge_lora.yaml
```

---

## Baseline and step=0 evaluation

Before any training updates, the trainer automatically:

1. saves an untrained checkpoint at

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

Every explicit experiment YAML now sets rollout saving to off by default:

```yaml
rollouts:
  save_train_rollouts: false
  save_eval_rollouts: false
  max_train_rollouts_per_save: 0
  max_eval_rollouts_per_save: 0
```

If you want to turn it on for a specific run, edit that YAML directly.

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

```bash
python scripts/eval_short_tasks.py   --checkpoint outputs/qwen35_4b_boolq_full/latest   --model-id Qwen/Qwen3.5-4B   --task boolq   --split validation
```

To save evaluation rollouts for a manual analysis run:

```bash
python scripts/eval_short_tasks.py   --checkpoint outputs/qwen35_4b_boolq_full/latest   --model-id Qwen/Qwen3.5-4B   --task boolq   --split validation   --save-rollouts
```

---

## How to run all four modes for one model/task pair

`run_four_modes.py` now takes **any one explicit config** and discovers the sibling `full`, `lora`, `bvou`, `bvou_lora` YAMLs automatically.

Example:

```bash
python scripts/run_four_modes.py   --config configs/qwen35_4b_boolq_full.yaml   --launcher accelerate   --deepspeed-config deepspeed/zero2.json
```

It will automatically run:

- `configs/qwen35_4b_boolq_full.yaml`
- `configs/qwen35_4b_boolq_lora.yaml`
- `configs/qwen35_4b_boolq_bvou.yaml`
- `configs/qwen35_4b_boolq_bvou_lora.yaml`

Dry run:

```bash
python scripts/run_four_modes.py   --config configs/qwen35_4b_boolq_full.yaml   --launcher accelerate   --deepspeed-config deepspeed/zero2.json   --dry-run
```

---

## How to evaluate four modes automatically

```bash
python scripts/eval_four_modes.py   --run-root outputs/qwen35_4b_boolq_four_modes
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
│   └── 36 explicit model-task-mode YAMLs
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

For the cleanest comparison, run the explicit config files directly. Each YAML is now self-contained with explicit sample counts, train steps, batch sizing, sequence-length budget, thinking-mode settings, and rollout-saving settings.

Good first three commands:

```bash
accelerate launch   --use_deepspeed   --deepspeed_config_file deepspeed/zero2.json   scripts/train_short_ppo.py   --config configs/qwen35_4b_boolq_full.yaml

accelerate launch   --use_deepspeed   --deepspeed_config_file deepspeed/zero2.json   scripts/train_short_ppo.py   --config configs/qwen35_4b_boolq_lora.yaml

accelerate launch   --use_deepspeed   --deepspeed_config_file deepspeed/zero2.json   scripts/train_short_ppo.py   --config configs/qwen35_4b_boolq_bvou_lora.yaml
```

Then expand to the remaining configs.
