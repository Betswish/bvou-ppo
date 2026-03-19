from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def train_rollout_path(output_dir: str | Path, step: int) -> Path:
    return Path(output_dir) / 'rollouts' / 'train' / f'step_{step:07d}.jsonl'


def eval_rollout_path(output_dir: str | Path, phase: str, task_name: str, split: str, step: int) -> Path:
    safe_phase = phase.replace('/', '_')
    return Path(output_dir) / 'rollouts' / 'eval' / task_name / split / f'{safe_phase}_step_{step:07d}.jsonl'


def save_train_rollouts(
    output_dir: str | Path,
    step: int,
    batch: list[dict[str, Any]],
    responses: list[str],
    rewards: list[float],
    selected_blocks: list[int] | None,
    model_name_or_path: str,
    task_name: str,
) -> Path:
    records: list[dict[str, Any]] = []
    for idx, (example, response, reward) in enumerate(zip(batch, responses, rewards)):
        records.append(
            {
                'kind': 'train_rollout',
                'step': step,
                'sample_index': idx,
                'task': task_name,
                'model_name_or_path': model_name_or_path,
                'query': example['query'],
                'gold_label': example['gold_label'],
                'response': response,
                'reward': float(reward),
                'selected_blocks': list(selected_blocks or []),
            }
        )
    path = train_rollout_path(output_dir, step)
    _write_jsonl(path, records)
    return path


def save_eval_rollouts(
    output_dir: str | Path,
    step: int,
    phase: str,
    task_name: str,
    split: str,
    records: list[dict[str, Any]],
) -> Path:
    path = eval_rollout_path(output_dir, phase, task_name, split, step)
    _write_jsonl(path, records)
    return path
