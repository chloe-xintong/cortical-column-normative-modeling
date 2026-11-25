#!/usr/bin/env python3
"""Validate and summarize the frozen 186-person regional Warped-M2 run.

Description
-----------
Validate and summarize the frozen 186-person regional Warped-M2 run.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--result-root`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/summarize_regional_warped_m2_186.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ANCHORS = [0, 4, 8, 12, 16, 20]


def metrics(d: pd.DataFrame) -> dict[str, float | int]:
    z = d.z.to_numpy(float)
    ordered = np.sort(z)
    q = stats.norm.ppf((np.arange(len(z)) + 0.5) / len(z))
    coverage = {}
    for level in (50, 80, 90):
        coverage[level] = float(((d.observed_md >= d[f"lower_{level}"]) &
                                 (d.observed_md <= d[f"upper_{level}"])).mean())
    return {
        "n_observations": len(d), "n_subjects": d.subject_id.nunique(),
        "mean_z": float(z.mean()), "sd_z": float(z.std(ddof=1)),
        "skewness_z": float(stats.skew(z, bias=False)),
        "excess_kurtosis_z": float(stats.kurtosis(z, fisher=True, bias=False)),
        "max_abs_z": float(np.abs(z).max()),
        "proportion_abs_z_gt_2": float((np.abs(z) > 2).mean()),
        "proportion_abs_z_gt_3": float((np.abs(z) > 3).mean()),
        "proportion_abs_z_gt_4": float((np.abs(z) > 4).mean()),
        "qq_rmse": float(np.sqrt(np.mean((ordered - q) ** 2))),
        "mean_log_likelihood": float(d.logp.mean()),
        "mean_log_loss_mll": float(d.mean_log_loss.mean()),
        "coverage_50": coverage[50], "coverage_80": coverage[80], "coverage_90": coverage[90],
        "mean_absolute_coverage_error": float(np.mean(np.abs(
            np.array([coverage[50], coverage[80], coverage[90]]) - np.array([.5, .8, .9])
        ))),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-root", type=Path, required=True)
    a = p.parse_args()
    root = a.result_root
    shard = root / "warped_m2" / "shards" / "predictions_all.parquet"
    diag_path = root / "warped_m2" / "shards" / "model_diagnostics_all.csv"
    failure_path = root / "warped_m2" / "shards" / "failures_all.json"
    report_path = root / "warped_m2" / "shards" / "run_report_all.json"
    for path in (shard, diag_path, failure_path, report_path):
        if not path.is_file():
            raise RuntimeError(f"Missing run artifact: {path}")
    pred = pd.read_parquet(shard)
    diagnostics = pd.read_csv(diag_path)
    failures = json.loads(failure_path.read_text(encoding="utf-8"))
    expected = 186 * 68 * 21
    keys = ["subject_id", "region_index", "depth_index"]
    finite = ["observed_md", "pred_median", "z", "logp", "mean_log_loss",
              "lower_90", "lower_80", "lower_50", "upper_50", "upper_80", "upper_90"]
    if len(pred) != expected or pred.subject_id.nunique() != 186 or pred.duplicated(keys).any():
        raise RuntimeError("OOF row, subject, or duplicate-key validation failed")
    if not np.isfinite(pred[finite].to_numpy(float)).all():
        raise RuntimeError("Nonfinite OOF value")
    if len(diagnostics) != 68 * 21 * 5 or failures:
        raise RuntimeError("Fit diagnostic count mismatch or failed task present")
    if not np.isfinite(diagnostics[["negative_log_evidence", "warp_epsilon", "warp_log_b", "warp_b"]].to_numpy(float)).all():
        raise RuntimeError("Nonfinite final model parameter")
    if not np.allclose(pred.logp.to_numpy(float), -pred.mean_log_loss.to_numpy(float)):
        raise RuntimeError("log-likelihood/MLL sign audit failed")

    final_pred = root / "oof_predictions_warped_M2_long.parquet"
    shutil.copy2(shard, final_pred)
    overall = pd.DataFrame([
        {"scope": "all_21_depths", **metrics(pred)},
        {"scope": "six_anchor_depths", **metrics(pred[pred.depth_index.isin(ANCHORS)])},
    ])
    overall.to_csv(root / "calibration_overall_and_anchor6.csv", index=False)
    pd.DataFrame([
        {"depth_index": int(depth), "depth_fraction": float(depth / 20), **metrics(group)}
        for depth, group in pred.groupby("depth_index", sort=True)
    ]).to_csv(root / "calibration_by_depth.csv", index=False)
    pd.DataFrame([
        {"age_bin": age_bin, **metrics(group)} for age_bin, group in pred.groupby("age_bin", sort=True)
    ]).to_csv(root / "calibration_by_age_group.csv", index=False)
    pd.DataFrame([
        {"hemisphere": hemi, **metrics(group)} for hemi, group in pred.groupby("hemisphere", sort=True)
    ]).to_csv(root / "calibration_by_hemisphere.csv", index=False)
    pred[pred.depth_index.isin(ANCHORS)].to_parquet(root / "oof_predictions_anchor6.parquet", index=False)

    validation = {
        "status": "complete", "cohort_n": 186, "regions": 68, "depths": 21,
        "fits_expected": 68 * 21 * 5, "fits_successful": len(diagnostics), "fits_failed": 0,
        "oof_rows_expected": expected, "oof_rows_observed": len(pred), "duplicate_keys": 0,
        "nonfinite_values": 0, "anchor_depth_indices": ANCHORS,
        "density_metric": {
            "logp": "elementwise predictive log likelihood in original MD units; higher is better",
            "MLL": "mean_log_loss = -logp; lower is better",
        },
    }
    (root / "integrity_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    rows = []
    for path in sorted(x for x in root.rglob("*") if x.is_file() and x.name != "artifact_manifest.csv"):
        rows.append({"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                     "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(root / "artifact_manifest.csv", index=False)
    print(json.dumps(validation))


if __name__ == "__main__":
    main()
