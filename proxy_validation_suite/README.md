# Proxy Validation Suite (Stage 1)

This folder isolates the **stage-1 proxy validation** experiment from the larger `beippo` training repo.
It is intended to answer the question:

> Which block-utility proxy is closest to the ideal one-step block gain in RL post-training?

## What this suite measures

For each batch, the script computes a set of block-level proxy scores, then computes `true_one_step_gain` for **all allowed candidate layers** by taking a single local update step on one block at a time. It then reports two evaluation views: (i) `all_layers`, which computes correlations over all allowed layers, and (ii) `topk_union`, which computes correlations only over the union of the per-proxy top-$k$ blocks.

The suite reports, for each proxy:

- **Spearman** correlation with true gains
- **Pearson** correlation with true gains
- **top-k overlap** with the top-k blocks under true gains
- **top-1 hit rate** against the top-1 block under true gains


## Two correlation scopes

The suite now saves **all-layer true gains** and reports two sets of metrics for every proxy:

- `all_layers`
  - Correlations and ranking metrics computed over **all allowed candidate layers**.
- `topk_union`
  - Correlations and ranking metrics computed over the union of each proxy's top-$k$ blocks.

This makes it possible to distinguish:

1. how well a proxy matches the true-gain ranking globally, and
2. how well it behaves on the smaller subset of blocks that proxy-based selection would actually consider.

Batch JSON files now contain:

- `true_one_step_gains_all_layers`
- `true_one_step_gains_topk_union`
- `candidate_blocks_topk_union`
- `metrics[proxy]['all_layers']`
- `metrics[proxy]['topk_union']`

## Included proxies

- `adv_grad_energy`
  - The main first-order proxy from the project:
    \[
    g_b = \mathbb{E}[A_t 
abla_{	heta_b} \log \pi_	heta(a_t \mid s_t)],
    \quad U_b pprox \eta \|g_b\|^2
    \]
- `no_adv_grad_energy`
  - Ablation of `adv_grad_energy` that removes the advantage weighting and uses the masked mean log-probability objective.
  - This isolates the contribution of the critic-induced advantage signal.
- `fisher_diag_energy`
  - Diagonal empirical-Fisher approximation to the second-order form.
- `grad_norm`
  - Generic PPO-gradient L2 norm baseline.
- `gate_grad`
  - Legacy gate-saliency proxy.
- `lisa_score`
  - **Operational baseline** approximating layer-importance style selection via size-normalized PPO gradient norm.
- `adagradselect_score`
  - **Operational baseline** approximating gradient-guided layer/block ranking via mean absolute PPO gradient.
- `random`
  - Deterministic random baseline.

## Important note on LISA / AdaGradSelect baselines

The implementations here are **block-level operational baselines**, not exact reproductions of the original papers.
They are included so that proxy validation can compare against strong, familiar importance-scoring heuristics.
If you need paper-faithful reproductions, treat these as placeholders for Codex to refine.

## Recommended first experiment

- Model: `Qwen/Qwen3.5-4B`
- Task: `boolq`
- Modes:
  - `bvou`
  - `bvou_lora`
- Proxies:
  - all eight above
- Metrics:
  - Spearman
  - Pearson
  - top-k overlap
  - top-1 hit rate

## Expected workflow

1. Install `beippo` from the main repo.
2. Copy this folder into the repo root, or call the script from this folder while `beippo` is importable.
3. Run the provided command.
4. Inspect `summary.json` and per-batch JSON files.

## Example

```bash
python run_proxy_validation_stage1.py   --config config_templates/qwen35_4b_boolq_bvou_stage1.yaml   --mode bvou   --split validation   --max-samples 64   --max-batches 2   --top-k 8
```

For `bvou_lora`:

```bash
python run_proxy_validation_stage1.py   --config config_templates/qwen35_4b_boolq_bvou_lora_stage1.yaml   --mode bvou_lora   --split validation   --max-samples 64   --max-batches 2   --top-k 8
```

## Outputs

By default the script writes to:

```text
outputs/<run_name>/proxy_validation_stage1/
```

Files:

- `summary.json`
- `batch_0000.json`
- `batch_0001.json`
- ...

## Integration guidance for Codex

If you want to fold this back into the main repo, the safest path is:

1. Merge `proxy_validity_stage1.py` into `src/beippo/proxy_validity.py`.
2. Merge `run_proxy_validation_stage1.py` into `scripts/run_proxy_validity.py`.
3. Keep the stage-1 metrics protocol stable:
   - Spearman
   - Pearson
   - top-k overlap
   - top-1 hit rate
4. Treat `lisa_score` and `adagradselect_score` as replaceable operational baselines.

## Do YAML or shell launchers need to change?

In the normal case, **no new YAML file is needed**.
The new proxy is integrated as an internal proxy option inside `proxy_validity_stage1.py`, and it is included in the default proxy list when `--proxies` is omitted.

You also usually do **not** need a new shell script.
The only case where a launcher must change is if it explicitly hard-codes a proxy whitelist such as:

```bash
--proxies adv_grad_energy fisher_diag_energy grad_norm lisa_score adagradselect_score random
```

In that case, simply add `no_adv_grad_energy` to the existing list.
No additional config file is required.
