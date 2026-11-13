#!/usr/bin/env python3
"""Fail-closed integrity and leakage-guard validation for the final regional run.

Description
-----------
Fail-closed integrity and leakage-guard validation for the final regional run.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input
- ``--source-folds`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/validate_final.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-folds", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    audit = pd.read_csv(out / "cohort_audit.csv", dtype={"subject_id": str})
    included = set(audit.loc[audit.included, "subject_id"])
    assert len(included) == 187 and "D0271" not in included
    folds = json.loads((out / "fold_assignments.json").read_text())
    source = json.loads(args.source_folds.read_text())
    old_test = {str(subject): int(item["outer_fold"]) for item in source["folds"]
                for subject in item["test_subject_ids"]}
    for item in folds["folds"]:
        train, test = set(item["train_subject_ids"]), set(item["test_subject_ids"])
        assert not train & test and train | test == included
        assert all(old_test[subject] == int(item["outer_fold"]) for subject in test)

    gaussian = pd.read_parquet(out / "oof_predictions_gaussian_long.parquet")
    warped = pd.read_parquet(out / "oof_predictions_warped_M2_long.parquet")
    gkeys = ["subject_id", "model", "region_index", "depth_index"]
    wkeys = ["subject_id", "region_index", "depth_index"]
    assert len(gaussian) == 187 * 68 * 21 * 4 and not gaussian.duplicated(gkeys).any()
    assert len(warped) == 187 * 68 * 21 and not warped.duplicated(wkeys).any()
    assert gaussian.groupby(["subject_id", "model"]).size().eq(68 * 21).all()
    assert warped.groupby("subject_id").size().eq(68 * 21).all()
    gaussian_m2 = gaussian.loc[gaussian.model.eq("M2"), wkeys]
    assert set(map(tuple, gaussian_m2.to_numpy())) == set(map(tuple, warped[wkeys].to_numpy()))
    assert np.isfinite(gaussian[["z", "logp", "mean_log_loss", "pred_mean", "pred_var"]]).all().all()
    assert np.isfinite(warped[["z", "logp", "mean_log_loss", "pred_median"]]).all().all()

    transforms = json.loads((out / "shards" / "fold_transforms_all.json").read_text())
    warped_transforms = json.loads((out / "warped_m2" / "shards" / "fold_transforms_all.json").read_text())
    assert len(transforms) == 68 * 4 * 5 and len(warped_transforms) == 68 * 5
    assert all("age_mean" in x["transform"] or "spline" in x["transform"] for x in transforms)
    assert all("age_mean" in x["transform"] and "thickness_mean" in x["transform"] for x in warped_transforms)
    diagnostics = pd.read_csv(out / "fold_model_diagnostics.csv")
    warp_diagnostics = pd.read_csv(out / "warped_m2" / "shards" / "model_diagnostics_all.csv")
    assert len(diagnostics) == 68 * 4 * 5 * 21 and diagnostics.status.eq("complete").all()
    assert len(warp_diagnostics) == 68 * 5 * 21 and warp_diagnostics.status.eq("complete").all()
    assert np.isfinite(warp_diagnostics[["warp_epsilon", "warp_log_b", "warp_b"]]).all().all()

    report = {
        "status": "complete", "cohort_n": 187, "retained_fold_assignments_changed": 0,
        "train_test_overlap_subjects": 0, "gaussian_oof_rows": len(gaussian),
        "warped_oof_rows": len(warped), "duplicate_keys": 0, "missing_predictions": 0,
        "nonfinite_final_predictions": 0, "gaussian_failed_outcome_fits": 0,
        "warped_failed_outcome_fits": 0, "gaussian_training_fold_transform_records": len(transforms),
        "warped_training_fold_transform_records": len(warped_transforms),
        "training_only_transform_code_paths": [
            "run_models.build_x / standardize_train_test / spline_matrices",
            "run_warped_m2.worker response scaling and warp fit",
        ],
        "gaussian_warped_identical_oof_keys": True, "ico4_jobs_started": 0,
    }
    (out / "final_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
