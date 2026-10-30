#!/usr/bin/env python3
"""Run the pre-specified paired ico4 versus ico4 patch-mean comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE = "ico4_md"
PATCH = "ico4_patch_mean_md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model-root", type=Path, required=True)
    parser.add_argument("--patch-model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261028)
    return parser.parse_args()


def load_prediction(root: Path, representation: str) -> pd.DataFrame:
    return pd.read_csv(root / representation / f"{representation}_oof_predictions.csv").sort_values(
        "row_index"
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_prediction(args.baseline_model_root, BASELINE)
    patch = load_prediction(args.patch_model_root, PATCH)
    if len(baseline) != len(patch):
        raise RuntimeError("Baseline and patch predictions have different subject counts")
    for column in ("row_index", "true_age", "outer_fold"):
        if not np.allclose(baseline[column], patch[column]):
            raise RuntimeError(f"Baseline and patch predictions differ in paired column {column}")
    baseline_metrics = json.loads(
        (args.baseline_model_root / BASELINE / "metrics.json").read_text(encoding="utf-8")
    )
    patch_metrics = json.loads(
        (args.patch_model_root / PATCH / "metrics.json").read_text(encoding="utf-8")
    )
    paired = baseline[["row_index", "true_age", "outer_fold", "absolute_error"]].rename(
        columns={"absolute_error": "ico4_absolute_error"}
    )
    paired["ico4_patch_mean_absolute_error"] = patch["absolute_error"]
    paired["paired_error_difference_patch_minus_ico4"] = (
        paired["ico4_patch_mean_absolute_error"] - paired["ico4_absolute_error"]
    )
    paired.to_csv(args.output_dir / "ico4_patch_mean_paired_absolute_errors.csv", index=False)
    difference = paired["paired_error_difference_patch_minus_ico4"].to_numpy()
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(difference), size=(args.bootstrap_replicates, len(difference)))
    bootstrap_delta = difference[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_delta, [2.5, 97.5])
    p_lower = (np.sum(bootstrap_delta <= 0) + 1) / (args.bootstrap_replicates + 1)
    p_upper = (np.sum(bootstrap_delta >= 0) + 1) / (args.bootstrap_replicates + 1)
    result = {
        "status": "complete",
        "comparison": "ico4_patch_mean_md minus ico4_md",
        "interpretation": "negative delta MAE favors patch mean",
        "n": len(difference),
        "ico4_oof_mae": baseline_metrics["overall_oof_mae"],
        "ico4_patch_mean_oof_mae": patch_metrics["overall_oof_mae"],
        "delta_mae_patch_minus_ico4": float(difference.mean()),
        "delta_mae_ci95_lower": float(lower),
        "delta_mae_ci95_upper": float(upper),
        "bootstrap_two_sided_p": float(min(1.0, 2 * min(p_lower, p_upper))),
        "ico4_oof_pearson_r": baseline_metrics["overall_oof_pearson_r"],
        "ico4_patch_mean_oof_pearson_r": patch_metrics["overall_oof_pearson_r"],
        "delta_pearson_r_patch_minus_ico4": (
            patch_metrics["overall_oof_pearson_r"] - baseline_metrics["overall_oof_pearson_r"]
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "ridge_solver": patch_metrics["ridge_solver"],
        "alpha_grid": patch_metrics["alpha_grid"],
    }
    pd.DataFrame([result]).to_csv(args.output_dir / "ico4_patch_mean_comparison.csv", index=False)
    (args.output_dir / "ico4_patch_mean_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
