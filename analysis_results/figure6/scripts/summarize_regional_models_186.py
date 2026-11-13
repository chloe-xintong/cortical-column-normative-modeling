#!/usr/bin/env python3
"""Compare all regional models on the identical frozen 186-person cohort.

Description
-----------
Compare all regional models on the identical frozen 186-person cohort.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--result-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/summarize_regional_models_186.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_final import (TARGETS, boot_metrics, grouped_metrics, metrics, paired_bootstrap,
                             subject_arrays)  # noqa: E402


ANCHORS = [0, 4, 8, 12, 16, 20]


def distribution_bootstrap(gaussian: pd.DataFrame, warped: pd.DataFrame, points: pd.DataFrame,
                           n_boot: int, seed: int) -> pd.DataFrame:
    subjects = sorted(gaussian.subject_id.unique())
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)),
                             size=n_boot).astype(np.int16)
    g = boot_metrics(subject_arrays(gaussian, subjects), counts)
    w = boot_metrics(subject_arrays(warped, subjects), counts)
    point = points.set_index("distribution_model")
    rows = []
    for metric in ("mean_log_loss", "mean_z", "sd_z", "coverage_50", "coverage_80", "coverage_90",
                   "mean_absolute_coverage_error"):
        if metric == "mean_z":
            diff = np.abs(w[metric]) - np.abs(g[metric])
            estimate = abs(point.loc["warped_M2", metric]) - abs(point.loc["Gaussian_M2", metric])
        elif metric == "sd_z":
            diff = np.abs(w[metric] - 1) - np.abs(g[metric] - 1)
            estimate = abs(point.loc["warped_M2", metric] - 1) - abs(point.loc["Gaussian_M2", metric] - 1)
        elif metric in TARGETS:
            target = TARGETS[metric]
            diff = np.abs(w[metric] - target) - np.abs(g[metric] - target)
            estimate = abs(point.loc["warped_M2", metric] - target) - abs(point.loc["Gaussian_M2", metric] - target)
        else:
            diff = w[metric] - g[metric]
            estimate = point.loc["warped_M2", metric] - point.loc["Gaussian_M2", metric]
        low, high = np.quantile(diff, [.025, .975])
        rows.append({"contrast": "warped_M2-Gaussian_M2", "metric": metric,
                     "difference_warped_minus_gaussian": float(estimate),
                     "ci_low": float(low), "ci_high": float(high),
                     "probability_warped_better": float(np.mean(diff < 0)),
                     "n_bootstrap": n_boot, "seed": seed,
                     "interpretation": "negative favors warped M2"})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20261108)
    a = p.parse_args()
    root = a.result_root
    gaussian_path = root / "gaussian_models" / "shards" / "predictions_all.parquet"
    warped_path = root / "oof_predictions_warped_M2_long.parquet"
    gaussian = pd.read_parquet(gaussian_path)
    warped = pd.read_parquet(warped_path)
    if "nlpd" in gaussian and "mean_log_loss" not in gaussian:
        gaussian = gaussian.rename(columns={"nlpd": "mean_log_loss"})
    gkeys = ["subject_id", "model", "region_index", "depth_index"]
    wkeys = ["subject_id", "region_index", "depth_index"]
    if len(gaussian) != 186 * 68 * 21 * 4 or gaussian.duplicated(gkeys).any():
        raise RuntimeError("Gaussian M0-M3 integrity failure")
    if len(warped) != 186 * 68 * 21 or warped.duplicated(wkeys).any():
        raise RuntimeError("Warped M2 integrity failure")
    gm2 = gaussian.query("model == 'M2'").copy()
    if set(map(tuple, gm2[wkeys].to_numpy())) != set(map(tuple, warped[wkeys].to_numpy())):
        raise RuntimeError("Gaussian and warped M2 held-out keys differ")
    finite_g = ["z", "logp", "mean_log_loss", "pred_mean", "pred_var"]
    finite_w = ["z", "logp", "mean_log_loss", "pred_median"]
    if not np.isfinite(gaussian[finite_g].to_numpy(float)).all() or not np.isfinite(warped[finite_w].to_numpy(float)).all():
        raise RuntimeError("Nonfinite prediction in unified regional results")

    gaussian.to_parquet(root / "oof_predictions_gaussian_M0_M3_long.parquet", index=False)
    model_21 = grouped_metrics(gaussian, ["model"], "pred_mean")
    model_21.insert(0, "scope", "all_21_depths")
    model_6 = grouped_metrics(gaussian[gaussian.depth_index.isin(ANCHORS)], ["model"], "pred_mean")
    model_6.insert(0, "scope", "six_anchor_depths")
    pd.concat([model_21, model_6], ignore_index=True).to_csv(root / "calibration_gaussian_M0_M3.csv", index=False)
    paired_bootstrap(gaussian, model_21, a.n_bootstrap, a.seed).to_csv(
        root / "model_comparison_subject_paired_bootstrap.csv", index=False)
    paired_bootstrap(gaussian[gaussian.depth_index.isin(ANCHORS)], model_6, a.n_bootstrap, a.seed + 1).to_csv(
        root / "model_comparison_anchor6_subject_paired_bootstrap.csv", index=False)

    dist_21 = pd.DataFrame([
        {"scope": "all_21_depths", "distribution_model": "Gaussian_M2", **metrics(gm2, "pred_mean")},
        {"scope": "all_21_depths", "distribution_model": "warped_M2", **metrics(warped, "pred_median")},
    ])
    g6, w6 = gm2[gm2.depth_index.isin(ANCHORS)], warped[warped.depth_index.isin(ANCHORS)]
    dist_6 = pd.DataFrame([
        {"scope": "six_anchor_depths", "distribution_model": "Gaussian_M2", **metrics(g6, "pred_mean")},
        {"scope": "six_anchor_depths", "distribution_model": "warped_M2", **metrics(w6, "pred_median")},
    ])
    pd.concat([dist_21, dist_6], ignore_index=True).to_csv(root / "gaussian_M2_vs_warped_M2.csv", index=False)
    distribution_bootstrap(gm2, warped, dist_21, a.n_bootstrap, a.seed + 2).to_csv(
        root / "gaussian_vs_warped_subject_paired_bootstrap.csv", index=False)
    distribution_bootstrap(g6, w6, dist_6, a.n_bootstrap, a.seed + 3).to_csv(
        root / "gaussian_vs_warped_anchor6_subject_paired_bootstrap.csv", index=False)
    grouped_metrics(gaussian, ["model", "age_bin"], "pred_mean").to_csv(
        root / "gaussian_calibration_by_model_age_group.csv", index=False)
    grouped_metrics(warped, ["age_bin"], "pred_median").to_csv(
        root / "warped_M2_calibration_by_age_group.csv", index=False)
    report = {
        "status": "complete", "cohort_n": 186, "folds": 5, "regions": 68, "depths": 21,
        "models": ["M0", "M1", "M2", "M3", "warped_M2"],
        "gaussian_rows": len(gaussian), "warped_rows": len(warped),
        "duplicate_keys": 0, "nonfinite_predictions": 0, "failed_fits": 0,
        "anchor_depth_indices": ANCHORS,
        "comparison_policy": "All regional models use identical 186 subjects and original retained folds.",
    }
    (root / "unified_model_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
