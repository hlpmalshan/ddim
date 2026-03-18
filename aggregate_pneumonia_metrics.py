#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np


METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
]


def mean_std(values: List[float]):
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def load_condition_metrics(condition_dir: str) -> Dict[str, List[float]]:
    out = {k: [] for k in METRICS}
    for name in sorted(os.listdir(condition_dir)):
        metrics_path = os.path.join(condition_dir, name, "metrics.json")
        if not os.path.isfile(metrics_path):
            continue
        with open(metrics_path, "r") as f:
            m = json.load(f)
        for key in METRICS:
            if key in m:
                out[key].append(float(m[key]))
    return out


def format_pm(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "nan"
    return f"{mean:.4f} +- {std:.4f}"


def main(args):
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Missing root: {root}")

    conditions = [
        d for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]
    if args.conditions:
        wanted = set(args.conditions.split(","))
        conditions = [c for c in conditions if c in wanted]

    if not conditions:
        raise RuntimeError(f"No condition directories found under: {root}")

    rows = []
    for cond in conditions:
        cond_dir = os.path.join(root, cond)
        grouped = load_condition_metrics(cond_dir)
        n_runs = len(grouped["accuracy"])
        row = {"condition": cond, "n_runs": n_runs}
        for key in METRICS:
            mean, std = mean_std(grouped[key])
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
            row[f"{key}_pm"] = format_pm(mean, std)
        rows.append(row)

    print("\nFinal Results (mean +- std)")
    print("-" * 125)
    print(
        f"{'condition':30s} {'n':>3s} {'AUROC':>15s} {'AUPRC':>15s} "
        f"{'BalAcc':>15s} {'Sensitivity':>15s} {'Specificity':>15s}"
    )
    print("-" * 125)
    for r in rows:
        print(
            f"{r['condition']:30s} {r['n_runs']:>3d} "
            f"{r['auroc_pm']:>15s} {r['auprc_pm']:>15s} "
            f"{r['balanced_accuracy_pm']:>15s} {r['sensitivity_pm']:>15s} "
            f"{r['specificity_pm']:>15s}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    json_out = os.path.join(args.output_dir, "pneumonia_experiment_summary.json")
    csv_out = os.path.join(args.output_dir, "pneumonia_experiment_summary.csv")
    with open(json_out, "w") as f:
        json.dump(rows, f, indent=2)

    csv_fields = ["condition", "n_runs"] + [f"{k}_{s}" for k in METRICS for s in ("mean", "std", "pm")]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in csv_fields})

    print(f"\nSaved summary JSON: {json_out}")
    print(f"Saved summary CSV : {csv_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root dir containing condition/seed_x/metrics.json")
    parser.add_argument("--output_dir", default=".", help="Where to write summary json/csv")
    parser.add_argument("--conditions", default="", help="Optional comma-separated condition filter")
    args = parser.parse_args()
    main(args)
