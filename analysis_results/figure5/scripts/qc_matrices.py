#!/usr/bin/env python3
"""Run integrity and missingness QC on all fixed-order Figure 5 matrices."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


REPRESENTATIONS = {
    "regional_mean_md": "features_regional_mean_md.csv",
    "regional_thickness": "features_regional_thickness.csv",
    "regional_profile": "features_regional_profile.csv",
    "ico3_md": "features_ico3_md.csv",
    "ico4_md": "features_ico4_md.csv",
    "ico5_md": "features_ico5_md.csv",
    "ico3_column_length": "features_ico3_column_length.csv",
    "ico3_curvature": "features_ico3_curvature.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = pd.read_csv(args.feature_dir / "subject_order.csv", dtype={"subject_id": str})
    if subjects["subject_id"].duplicated().any() or not np.array_equal(
        subjects["row_index"].to_numpy(), np.arange(len(subjects))
    ):
        raise RuntimeError("Subject order is duplicated or non-contiguous")
    coverage = pd.read_csv(args.feature_dir / "subject_coverage.csv")
    if not np.array_equal(coverage["row_index"].to_numpy(), subjects["row_index"].to_numpy()):
        raise RuntimeError("Coverage and matrix subject orders differ")

    subject_missing = pd.DataFrame({"row_index": subjects["row_index"]})
    integrity: dict[str, object] = {}
    failures: list[str] = []
    for name, feature_file in REPRESENTATIONS.items():
        matrix_path = args.feature_dir / "matrices" / f"X_{name}.npy"
        metadata_path = args.feature_dir / "feature_metadata" / feature_file
        matrix = np.load(matrix_path, mmap_mode="r")
        metadata = pd.read_csv(metadata_path)
        hard_checks = {
            "two_dimensional": matrix.ndim == 2,
            "subject_count_matches": matrix.shape[0] == len(subjects),
            "feature_count_matches": matrix.shape[1] == len(metadata),
            "feature_index_contiguous": np.array_equal(
                metadata["feature_index"].to_numpy(), np.arange(len(metadata))
            ),
            "feature_definitions_unique": not bool(
                metadata.drop(columns=["feature_index"]).duplicated().any()
            ),
            "contains_no_infinity": not bool(np.isinf(matrix).any()),
        }
        if name == "regional_profile":
            hard_checks["regional_profile_has_21_depths_per_region"] = bool(
                metadata.groupby(["hemisphere", "dk_label_id"], sort=False).size().eq(21).all()
            )
        if name.startswith("ico") and name.endswith("_md"):
            hard_checks["ico_profile_has_21_depths_per_node"] = bool(
                metadata.groupby(["hemisphere", "ico_node_id"], sort=False).size().eq(21).all()
            )
        if name.startswith("ico"):
            level = int(name[3])
            vertex_order = pd.read_csv(
                args.feature_dir.parent / "template" / f"ico{level}_vertex_order.csv"
            )
            vertex_order = vertex_order.loc[vertex_order["in_analysis_mask"].astype(bool)]
            observed_nodes = metadata.drop_duplicates(["hemisphere", "ico_node_id"])[
                ["hemisphere", "ico_node_id", "fsaverage_vertex_id"]
            ].reset_index(drop=True)
            expected_nodes = vertex_order[
                ["hemisphere", "ico_node_id", "fsaverage_vertex_id"]
            ].reset_index(drop=True)
            hard_checks["node_order_matches_fixed_vertex_table"] = observed_nodes.equals(expected_nodes)
        if "depth_direction" in metadata.columns:
            hard_checks["single_depth_direction"] = metadata["depth_direction"].nunique() == 1
        finite_count = np.sum(np.isfinite(matrix), axis=0, dtype=np.int64)
        missing_fraction = 1.0 - finite_count / matrix.shape[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            feature_min = np.nanmin(matrix, axis=0)
            feature_max = np.nanmax(matrix, axis=0)
        all_missing = finite_count == 0
        zero_variance = (~all_missing) & np.isclose(feature_min, feature_max, equal_nan=False)
        feature_qc = metadata[["feature_index"]].copy()
        feature_qc["n_finite"] = finite_count
        feature_qc["missing_fraction"] = missing_fraction
        feature_qc["all_missing"] = all_missing
        feature_qc["zero_variance"] = zero_variance
        feature_qc.to_csv(args.output_dir / f"{name}_feature_qc.csv", index=False)
        subject_missing[f"{name}_missing_fraction"] = np.mean(~np.isfinite(matrix), axis=1)
        content_warnings = {
            "contains_all_missing_features": bool(all_missing.any()),
            "contains_zero_variance_features": bool(zero_variance.any()),
        }
        passed = all(hard_checks.values())
        if not passed:
            failures.append(name)
        integrity[name] = {
            "shape": list(matrix.shape),
            "hard_checks": hard_checks,
            "warnings": content_warnings,
            "n_all_missing_features": int(all_missing.sum()),
            "n_zero_variance_features": int(zero_variance.sum()),
            "median_feature_missing_fraction": float(np.median(missing_fraction)),
            "maximum_feature_missing_fraction": float(np.max(missing_fraction)),
            "passed": passed,
        }
    subject_missing = subject_missing.merge(
        coverage[["row_index", "coverage_fraction", "coverage_ge_0_95"]], on="row_index", how="left", validate="one_to_one"
    )
    subject_missing.to_csv(args.output_dir / "subject_missingness_qc.csv", index=False)
    report = {
        "status": "passed" if not failures else "failed",
        "n_subjects": len(subjects),
        "coverage_threshold": 0.95,
        "n_subjects_below_coverage_threshold": int((~subject_missing["coverage_ge_0_95"].astype(bool)).sum()),
        "representations": integrity,
        "failed_representations": failures,
    }
    (args.output_dir / "matrix_qc_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "n_subjects": len(subjects), "failed": failures}))
    if failures:
        raise RuntimeError(f"Matrix QC failed for: {failures}")


if __name__ == "__main__":
    main()
