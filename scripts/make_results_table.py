#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        metrics_path = Path(run) / "metrics.jsonl"
        logs = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        train_logs = [x for x in logs if "reward_mean" in x]
        eval_logs = [x for x in logs if "exact_match" in x]
        baseline_logs = [x for x in eval_logs if x.get("is_baseline") or x.get("phase") == "baseline_eval"]
        train_eval_logs = [x for x in eval_logs if not (x.get("is_baseline") or x.get("phase") == "baseline_eval")]
        peak_memory = max((x.get("peak_memory_gb", 0.0) for x in train_logs), default=0.0)
        grouped_baseline = defaultdict(list)
        grouped_train = defaultdict(list)
        for item in baseline_logs:
            grouped_baseline[(item["task"], item["split"])].append(item)
        for item in train_eval_logs:
            grouped_train[(item["task"], item["split"])].append(item)
        row = {"run": Path(run).name, "peak_memory_gb": peak_memory}
        for key, values in grouped_baseline.items():
            row[f"baseline_{key[0]}_{key[1]}"] = values[-1]["exact_match"]
        for key, values in grouped_train.items():
            row[f"{key[0]}_{key[1]}"] = values[-1]["exact_match"]
        if train_logs:
            row["reward_mean_last"] = train_logs[-1]["reward_mean"]
        rows.append(row)
    headers = sorted({k for row in rows for k in row})
    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(row.get(h, "")) for h in headers))
