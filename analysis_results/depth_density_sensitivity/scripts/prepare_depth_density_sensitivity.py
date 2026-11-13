#!/usr/bin/env python3
"""Prepare 6/11/21-depth profile-fidelity data and reduced Figure 5 features.

Description
-----------
Prepare 6/11/21-depth profile-fidelity data and reduced Figure 5 features.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--figure3-root`` command-line input
- ``--figure5-root`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/depth_density_sensitivity/scripts/prepare_depth_density_sensitivity.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


FULL_DEPTHS = np.linspace(0.0, 1.0, 21)
DEPTH_INDICES = {
    6: np.asarray([0, 4, 8, 12, 16, 20], dtype=int),
    11: np.asarray([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=int),
    21: np.arange(21, dtype=int),
}
REPRESENTATIONS = ("regional_profile", "ico4_md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure3-root", type=Path, required=True)
    parser.add_argument("--figure5-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def interpolate_profiles(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Linearly interpolate retained points to the fixed 21-point grid."""
    sampled_depths = FULL_DEPTHS[indices]
    output = np.empty_like(values, dtype=np.float64)
    for row_index, row in enumerate(values):
        output[row_index] = np.interp(FULL_DEPTHS, sampled_depths, row[indices])
    return output


def profile_metrics(reference: np.ndarray, reconstructed: np.ndarray) -> dict[str, np.ndarray]:
    difference = reconstructed - reference
    reference_centered = reference - reference.mean(axis=1, keepdims=True)
    reconstructed_centered = reconstructed - reconstructed.mean(axis=1, keepdims=True)
    centered_difference = reconstructed_centered - reference_centered
    numerator = np.sum(reference_centered * reconstructed_centered, axis=1)
    denominator = np.sqrt(
        np.sum(reference_centered**2, axis=1)
        * np.sum(reconstructed_centered**2, axis=1)
    )
    correlation = np.divide(
        numerator,
        denominator,
        out=np.full(len(reference), np.nan, dtype=float),
        where=denominator > 0,
    )
    return {
        "raw_profile_rmse": np.sqrt(np.mean(difference**2, axis=1)),
        "mean_centered_profile_rmse": np.sqrt(np.mean(centered_difference**2, axis=1)),
        "mean_centered_pearson_r": correlation,
        "maximum_absolute_difference": np.max(np.abs(difference), axis=1),
    }


def prepare_profile_fidelity(figure3_root: Path, output_dir: Path) -> dict[str, object]:
    source = figure3_root / "tables" / "figure3A_subject_regional_profiles.csv"
    bilateral_source = figure3_root / "tables" / "figure3A_bilateral_subject_profiles.csv"
    selection_source = figure3_root / "qc" / "figure3A_region_selection.json"
    table = pd.read_csv(source, dtype={"subject_id": str})
    required = {
        "subject_id", "hemisphere", "region", "depth_index", "normalized_depth", "md"
    }
    if not required.issubset(table.columns):
        raise RuntimeError(f"Figure 3 profile table lacks {sorted(required - set(table.columns))}")
    keys = ["subject_id", "hemisphere", "region"]
    if table.duplicated(keys + ["depth_index"]).any():
        raise RuntimeError("Duplicate subject x hemisphere x region x depth rows")
    counts = table.groupby(keys, sort=False).size()
    if not counts.eq(21).all():
        raise RuntimeError("Every observed subject x hemisphere x region profile must have 21 depths")
    if set(table["depth_index"].unique()) != set(range(21)):
        raise RuntimeError("Figure 3 profiles do not contain exactly depth indices 0..20")
    expected_depth = table["depth_index"].to_numpy(dtype=int) / 20.0
    if not np.allclose(table["normalized_depth"], expected_depth, atol=1e-12, rtol=0):
        raise RuntimeError("Figure 3 normalized depths differ from the canonical 21-point grid")
    ordered = table.sort_values(keys + ["depth_index"], kind="stable")
    identity = ordered[keys].drop_duplicates().reset_index(drop=True)
    values = ordered["md"].to_numpy(dtype=np.float64).reshape(-1, 21)
    if values.shape[-1] != 21:
        raise RuntimeError("Profile last dimension is not 21")

    region_outputs: list[pd.DataFrame] = []
    exact_shared_depth_checks: dict[str, bool] = {}
    endpoint_checks: dict[str, bool] = {}
    for n_depths in (6, 11):
        indices = DEPTH_INDICES[n_depths]
        endpoint_checks[str(n_depths)] = bool(indices[0] == 0 and indices[-1] == 20)
        sampled = values[:, indices]
        exact_shared_depth_checks[str(n_depths)] = bool(
            np.array_equal(sampled, np.take(values, indices, axis=1), equal_nan=True)
        )
        if not endpoint_checks[str(n_depths)] or not exact_shared_depth_checks[str(n_depths)]:
            raise RuntimeError(f"Depth-subset invariant failed for {n_depths} points")
        reconstructed = interpolate_profiles(values, indices)
        result = identity.copy()
        result.insert(3, "n_depths", n_depths)
        result.insert(4, "retained_depth_indices", ",".join(map(str, indices.tolist())))
        for name, metric_values in profile_metrics(values, reconstructed).items():
            result[name] = metric_values
        region_outputs.append(result)
    subject_region = pd.concat(region_outputs, ignore_index=True)
    subject_region.to_csv(output_dir / "profile_fidelity_subject_region.csv", index=False)

    metric_columns = [
        "raw_profile_rmse",
        "mean_centered_profile_rmse",
        "mean_centered_pearson_r",
        "maximum_absolute_difference",
    ]
    subject_summary = (
        subject_region.groupby(["subject_id", "n_depths"], sort=True)
        .agg(
            n_region_hemisphere_profiles=("region", "size"),
            **{f"{column}_median": (column, "median") for column in metric_columns},
        )
        .reset_index()
    )
    subject_summary.to_csv(output_dir / "profile_fidelity_subject_summary.csv", index=False)
    group_rows: list[dict[str, object]] = []
    for n_depths, group in subject_summary.groupby("n_depths", sort=True):
        for column in [f"{name}_median" for name in metric_columns]:
            values_for_metric = group[column].to_numpy(dtype=float)
            group_rows.append(
                {
                    "n_depths": int(n_depths),
                    "metric": column,
                    "n_subjects": int(np.isfinite(values_for_metric).sum()),
                    "median": float(np.nanmedian(values_for_metric)),
                    "q25": float(np.nanpercentile(values_for_metric, 25)),
                    "q75": float(np.nanpercentile(values_for_metric, 75)),
                }
            )
    pd.DataFrame(group_rows).to_csv(output_dir / "profile_fidelity_group_summary.csv", index=False)

    selection = json.loads(selection_source.read_text(encoding="utf-8"))
    selected_names = [row["region_name"] for row in selection["regions"]]
    bilateral = pd.read_csv(bilateral_source, dtype={"subject_id": str})
    bilateral = bilateral.loc[bilateral["region"].isin(selected_names)].copy()
    selected_counts = bilateral.groupby(["subject_id", "region"], sort=False).size()
    if not selected_counts.eq(21).all():
        raise RuntimeError("A selected Figure 3A bilateral profile lacks the complete 21-depth grid")
    group_profiles = (
        bilateral.groupby(["region", "depth_index", "normalized_depth"], sort=False)["md"]
        .agg(group_mean_md="mean", group_sd_md="std", n_subjects="count")
        .reset_index()
    )
    selection_order = {name: index for index, name in enumerate(selected_names)}
    group_profiles.insert(0, "selection_order", group_profiles["region"].map(selection_order))
    if not group_profiles.groupby("region")["n_subjects"].apply(lambda x: x.eq(320).all()).all():
        raise RuntimeError("Figure 3A selected-region group profiles do not all use 320 subjects")
    profile_rows: list[pd.DataFrame] = []
    for n_depths in (6, 11, 21):
        part = group_profiles.loc[group_profiles["depth_index"].isin(DEPTH_INDICES[n_depths])].copy()
        part.insert(0, "n_depths", n_depths)
        profile_rows.append(part)
    pd.concat(profile_rows, ignore_index=True).to_csv(
        output_dir / "selected_region_group_profiles.csv", index=False
    )
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "bilateral_source": str(bilateral_source),
        "selection_source": str(selection_source),
        "n_subjects": int(table["subject_id"].nunique()),
        "n_observed_region_hemisphere_profiles": int(len(identity)),
        "profile_last_dimension_is_21": True,
        "exact_shared_depth_checks": exact_shared_depth_checks,
        "endpoint_checks": endpoint_checks,
        "selected_regions": selected_names,
    }


def prepare_prediction_features(figure5_root: Path, output_dir: Path) -> dict[str, object]:
    source_features = figure5_root / "features"
    subject_order = source_features / "subject_order.csv"
    subject_table = pd.read_csv(subject_order, dtype={"subject_id": str})
    if len(subject_table) != 196 or subject_table["subject_id"].duplicated().any():
        raise RuntimeError("Expected the fixed 196-subject Figure 5 cohort")
    report: dict[str, object] = {
        "source_feature_root": str(source_features),
        "subject_order": str(subject_order),
        "subject_order_sha256": sha256(subject_order),
        "n_subjects": len(subject_table),
        "representations": {},
    }
    for n_depths in (6, 11):
        destination = output_dir / "features" / f"d{n_depths}"
        (destination / "matrices").mkdir(parents=True, exist_ok=True)
        (destination / "feature_metadata").mkdir(parents=True, exist_ok=True)
        shutil.copy2(subject_order, destination / "subject_order.csv")
    for representation in REPRESENTATIONS:
        matrix_path = source_features / "matrices" / f"X_{representation}.npy"
        metadata_path = source_features / "feature_metadata" / f"features_{representation}.csv"
        matrix = np.load(matrix_path, mmap_mode="r")
        metadata = pd.read_csv(metadata_path)
        if matrix.ndim != 2 or matrix.shape[0] != 196 or matrix.shape[1] != len(metadata):
            raise RuntimeError(f"{representation}: source matrix/metadata dimensions disagree")
        if not np.array_equal(metadata["feature_index"].to_numpy(), np.arange(len(metadata))):
            raise RuntimeError(f"{representation}: unstable feature_index order")
        group_keys = ["hemisphere", "dk_label_id"]
        if representation == "ico4_md":
            group_keys = ["hemisphere", "ico_node_id"]
        if not metadata.groupby(group_keys, sort=False).size().eq(21).all():
            raise RuntimeError(f"{representation}: profile last dimension is not 21")
        if not np.array_equal(metadata["depth_index"].to_numpy(), np.tile(np.arange(21), matrix.shape[1] // 21)):
            raise RuntimeError(f"{representation}: feature order is not profile-major, depth-minor")
        representation_report: dict[str, object] = {
            "source_matrix": str(matrix_path),
            "source_metadata": str(metadata_path),
            "source_shape": list(matrix.shape),
            "source_matrix_sha256": sha256(matrix_path),
            "densities": {},
        }
        for n_depths in (6, 11):
            indices = DEPTH_INDICES[n_depths]
            mask = metadata["depth_index"].isin(indices).to_numpy()
            reduced = np.asarray(matrix[:, mask], dtype=np.float32)
            destination = output_dir / "features" / f"d{n_depths}"
            destination_matrix = destination / "matrices" / f"X_{representation}.npy"
            np.save(destination_matrix, reduced, allow_pickle=False)
            reduced_metadata = metadata.loc[mask].copy().reset_index(drop=True)
            reduced_metadata["feature_index"] = np.arange(len(reduced_metadata), dtype=np.int64)
            destination_metadata = (
                destination / "feature_metadata" / f"features_{representation}.csv"
            )
            reduced_metadata.to_csv(destination_metadata, index=False)
            saved = np.load(destination_matrix, mmap_mode="r")
            exact = bool(np.array_equal(saved, matrix[:, mask], equal_nan=True))
            endpoints = set(reduced_metadata["depth_index"].unique()) >= {0, 20}
            if not exact or not endpoints:
                raise RuntimeError(f"{representation} d{n_depths}: exact-subset check failed")
            representation_report["densities"][str(n_depths)] = {
                "depth_indices": indices.tolist(),
                "shape": list(saved.shape),
                "exact_subset_of_21": exact,
                "pial_and_white_endpoints_retained": bool(endpoints),
                "matrix": str(destination_matrix),
                "metadata": str(destination_metadata),
            }
        report["representations"][representation] = representation_report
    return report


def main() -> None:
    args = parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_report = prepare_profile_fidelity(args.figure3_root, args.output_dir)
    feature_report = prepare_prediction_features(args.figure5_root, args.output_dir)
    payload = {
        "status": "complete",
        "started_unix": start,
        "elapsed_seconds": time.time() - start,
        "depth_indices": {str(key): value.tolist() for key, value in DEPTH_INDICES.items()},
        "profile_fidelity": profile_report,
        "age_prediction_features": feature_report,
    }
    (args.output_dir / "preparation_checks.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
