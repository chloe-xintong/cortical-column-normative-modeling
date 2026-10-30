#!/usr/bin/env python3
"""Validate ico4 geometry matrices and fixed feature ordering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--vertex-order", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = pd.read_csv(args.feature_dir / "subject_order.csv")
    expected = pd.read_csv(args.vertex_order)
    expected = expected.loc[expected["in_analysis_mask"].astype(bool), [
        "hemisphere", "ico_node_id", "fsaverage_vertex_id"
    ]].reset_index(drop=True)
    report: dict[str, object] = {"status": "passed", "n_subjects": len(subjects), "matrices": {}}
    for name in ("ico4_column_length", "ico4_curvature"):
        matrix = np.load(args.feature_dir / "matrices" / f"X_{name}.npy", mmap_mode="r")
        metadata = pd.read_csv(args.feature_dir / "feature_metadata" / f"features_{name}.csv")
        observed = metadata[["hemisphere", "ico_node_id", "fsaverage_vertex_id"]]
        checks = {
            "shape": matrix.shape == (len(subjects), 4687),
            "metadata_length": len(metadata) == 4687,
            "feature_index_contiguous": np.array_equal(
                metadata["feature_index"], np.arange(4687)
            ),
            "fixed_node_order": observed.reset_index(drop=True).equals(expected),
            "no_infinity": not bool(np.isinf(matrix).any()),
        }
        if not all(checks.values()):
            report["status"] = "failed"
        finite = np.isfinite(matrix)
        feature_qc = pd.DataFrame(
            {
                "feature_index": np.arange(matrix.shape[1]),
                "n_finite": finite.sum(axis=0),
                "missing_fraction": 1.0 - finite.mean(axis=0),
                "all_missing": ~finite.any(axis=0),
            }
        )
        feature_qc.to_csv(args.output_dir / f"{name}_feature_qc.csv", index=False)
        report["matrices"][name] = {
            "shape": list(matrix.shape),
            "checks": checks,
            "n_all_missing_features": int((~finite.any(axis=0)).sum()),
            "maximum_feature_missing_fraction": float((1.0 - finite.mean(axis=0)).max()),
        }
    (args.output_dir / "ico4_geometry_qc_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    if report["status"] != "passed":
        raise RuntimeError("ico4 geometry QC failed")


if __name__ == "__main__":
    main()
