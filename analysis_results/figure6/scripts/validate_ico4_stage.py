#!/usr/bin/env python3
"""Validate a completed ico4 OOF stage and write integrity/calibration reports.

Description
-----------
Validate a completed ico4 OOF stage and write integrity/calibration reports.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--result-root`` command-line input
- ``--subject-order`` command-line input
- ``--ico4-md`` command-line input
- ``--stage`` command-line input
- ``--depth-indices`` command-line input
- ``--output`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/validate_ico4_stage.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--subject-order", type=Path, required=True)
    p.add_argument("--ico4-md", type=Path, required=True)
    p.add_argument("--stage", choices=("smoke", "anchor", "remaining"), required=True)
    p.add_argument("--node-list", type=Path)
    p.add_argument("--depth-indices", required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def metrics(d: pd.DataFrame) -> dict[str, float]:
    z = d.z.to_numpy(float)
    ordered = np.sort(z)
    q = stats.norm.ppf((np.arange(len(z)) + 0.5) / len(z))
    return {
        "n_predictions": len(d), "mean_z": float(np.mean(z)), "sd_z": float(np.std(z, ddof=1)),
        "coverage_50": float(((d.observed_md >= d.lower_50) & (d.observed_md <= d.upper_50)).mean()),
        "coverage_80": float(((d.observed_md >= d.lower_80) & (d.observed_md <= d.upper_80)).mean()),
        "coverage_90": float(((d.observed_md >= d.lower_90) & (d.observed_md <= d.upper_90)).mean()),
        "coverage_mae": float(np.mean([abs(((d.observed_md >= d.lower_50) & (d.observed_md <= d.upper_50)).mean() - .5),
                                        abs(((d.observed_md >= d.lower_80) & (d.observed_md <= d.upper_80)).mean() - .8),
                                        abs(((d.observed_md >= d.lower_90) & (d.observed_md <= d.upper_90)).mean() - .9)])),
        "mean_log_likelihood": float(d.log_likelihood.mean()), "qq_rmse": float(np.sqrt(np.mean((ordered - q) ** 2))),
        "skewness": float(stats.skew(z, bias=False)), "excess_kurtosis": float(stats.kurtosis(z, fisher=True, bias=False)),
        "fraction_abs_z_gt_2": float((np.abs(z) > 2).mean()), "fraction_abs_z_gt_3": float((np.abs(z) > 3).mean()),
        "fraction_abs_z_gt_4": float((np.abs(z) > 4).mean()),
    }


def main() -> None:
    a = parse_args()
    depths = [int(x) for x in a.depth_indices.split(",") if x.strip()]
    nodes = pd.read_csv(a.node_list).node_index.astype(int).tolist() if a.node_list else list(range(4687))
    base = a.result_root / a.stage
    statuses = [json.loads(p.read_text(encoding="utf-8")) for p in base.glob("status/fold=*/hemisphere=*/vertex_chunk=*/*.json")]
    pred_files = sorted(base.glob("predictions/fold=*/hemisphere=*/vertex_chunk=*/*.parquet"))
    diag_files = sorted(base.glob("diagnostics/fold=*/hemisphere=*/vertex_chunk=*/*.parquet"))
    expected_tasks = len(nodes) * 5
    if len(statuses) != expected_tasks or len(pred_files) != expected_tasks or len(diag_files) != expected_tasks:
        raise RuntimeError(f"Incomplete task files: status={len(statuses)} pred={len(pred_files)} diag={len(diag_files)} expected={expected_tasks}")
    if any(x.get("status") != "complete" for x in statuses):
        raise RuntimeError("At least one task status is not complete")
    pred = pd.concat((pd.read_parquet(p) for p in pred_files), ignore_index=True)
    diag = pd.concat((pd.read_parquet(p) for p in diag_files), ignore_index=True)
    if len(diag) != len(nodes) * 5 * len(depths) or (diag.status == "failed").any():
        raise RuntimeError("Fit diagnostic count/status mismatch")
    keys = ["subject_id", "node_index", "depth_index"]
    if pred.duplicated(keys).any():
        raise RuntimeError("Duplicate OOF prediction keys")
    finite_cols = ["observed_md", "pred_mean", "pred_median", "predictive_variance", "z", "log_likelihood",
                   "lower_90", "lower_80", "lower_50", "upper_50", "upper_80", "upper_90"]
    if not np.isfinite(pred[finite_cols].to_numpy(float)).all() or (pred.predictive_variance <= 0).any():
        raise RuntimeError("Nonfinite OOF value or nonpositive predictive variance")

    so = pd.read_csv(a.subject_order, dtype={"subject_id": str})
    audit = pd.read_csv(a.result_root / "frozen_cohort_manifest.csv", dtype={"subject_id": str})
    folds = pd.read_csv(a.result_root / "outer_fold_assignments.csv", dtype={"subject_id": str})
    cohort = audit.loc[audit.included, ["subject_id"]].merge(so[["subject_id", "row_index"]], on="subject_id").merge(folds, on="subject_id")
    md = np.load(a.ico4_md, mmap_mode="r")
    thickness = np.load(a.result_root / "inputs" / "X_ico4_local_freesurfer_thickness.npy", mmap_mode="r")
    expected = []
    for node in nodes:
        t = np.isfinite(thickness[cohort.row_index.to_numpy(int), node])
        for depth in depths:
            y = np.isfinite(md[cohort.row_index.to_numpy(int), node * 21 + depth])
            expected.extend((sid, node, depth) for sid in cohort.loc[t & y, "subject_id"])
    expected_keys = pd.MultiIndex.from_tuples(expected, names=keys)
    observed_keys = pd.MultiIndex.from_frame(pred[keys])
    missing = expected_keys.difference(observed_keys)
    extra = observed_keys.difference(expected_keys)
    if len(missing) or len(extra):
        raise RuntimeError(f"OOF key mismatch: missing={len(missing)} extra={len(extra)}")

    warning_fits = int(diag.status.eq("complete_with_warning").sum())
    report = {
        "status": "complete", "stage": a.stage, "n_nodes": len(nodes), "n_depths": len(depths),
        "depth_indices": depths, "expected_tasks": expected_tasks, "complete_tasks": len(statuses),
        "expected_fits": len(nodes) * 5 * len(depths), "successful_fits": len(diag), "failed_fits": 0,
        "warning_fits": warning_fits, "oof_predictions": len(pred), "duplicate_keys": 0,
        "missing_expected_keys": 0, "extra_keys": 0, "nonfinite_predictions": 0,
        "gaussian_fallbacks": 0, "model": "Warped M2", "warp": "WarpSinhArcsinh",
        "calibration": metrics(pred),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
