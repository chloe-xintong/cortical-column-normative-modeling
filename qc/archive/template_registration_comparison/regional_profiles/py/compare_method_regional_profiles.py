#!/usr/bin/env python3
"""Compare four fsaverage regional MD profile methods.

Description
-----------
Compare four fsaverage regional MD profile methods.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--native-profile-root`` command-line input
- ``--o2-root`` command-line input
- ``--o2-column-root`` command-line input
- ``--o3-root`` command-line input
- ``--freesurfer-subjects-dir`` command-line input
- ``--output-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python qc/archive/template_registration_comparison/regional_profiles/py/compare_method_regional_profiles.py --help
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


DEFAULT_DEPTHS = np.linspace(0.0, 1.0, 21)
EXPECTED_REGIONS_PER_HEMISPHERE = 34
EXPECTED_VERTICES = 163842
ZERO_LENGTH_TOLERANCE_MM = 1e-5
METHOD_NATIVE = "Native"
METHOD_FS_DEPTHMAP = "FS DepthMap"
METHOD_PROFILE_INTERP = "Profile Interp."
METHOD_COORDINATE_INTERP = "Coordinate Interp."
PROFILE_PATTERN = re.compile(
    r"^(?P<subject>D\d+)_(?P<hemisphere>lh|rh)_(?P<region>.+)_md\.csv$"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-profile-root", type=Path, required=True)
    parser.add_argument("--o2-root", type=Path, required=True)
    parser.add_argument("--o2-column-root", type=Path, required=True)
    parser.add_argument("--o3-root", type=Path, required=True)
    parser.add_argument("--freesurfer-subjects-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject-id", action="append")
    parser.add_argument("--profile-dir-name", default="profiles")
    parser.add_argument("--min-valid-depths", type=int, default=1)
    return parser.parse_args()


def decode_names(names: list[bytes | str]) -> list[str]:
    """Decode FreeSurfer annotation names."""

    return [name.decode() if isinstance(name, bytes) else str(name) for name in names]


def load_manifest(path: Path, selected_subjects: list[str] | None) -> pd.DataFrame:
    """Load and validate the subject manifest."""

    manifest = pd.read_csv(path, dtype=str)
    if "subject_id" not in manifest.columns:
        raise ValueError(f"Manifest lacks subject_id: {path}")
    manifest = manifest.drop_duplicates("subject_id").copy()
    if selected_subjects:
        selected = set(selected_subjects)
        manifest = manifest[manifest.subject_id.isin(selected)].copy()
    if manifest.empty:
        raise ValueError("No subjects remain after applying subject selection")
    return manifest.sort_values("subject_id").reset_index(drop=True)


def load_fsaverage_labels(
    subjects_dir: Path,
    hemisphere: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load fsaverage DK labels and return masks for 34 cortical regions."""

    annot_path = subjects_dir / "fsaverage" / "label" / f"{hemisphere}.aparc.annot"
    labels, _, names = nib.freesurfer.io.read_annot(str(annot_path))
    decoded = decode_names(list(names))
    excluded = {"unknown", "corpuscallosum", "medial_wall"}
    region_masks = {
        name: labels == code
        for code, name in enumerate(decoded)
        if name not in excluded
    }
    if len(region_masks) != EXPECTED_REGIONS_PER_HEMISPHERE:
        raise ValueError(
            f"Expected {EXPECTED_REGIONS_PER_HEMISPHERE} DK regions for {hemisphere}, "
            f"found {len(region_masks)} in {annot_path}"
        )
    return labels.astype(np.int64), region_masks


def load_csv_profile(path: Path) -> np.ndarray:
    """Load one native profile matrix as rows-by-depths."""

    values = np.loadtxt(path, delimiter=",", ndmin=2).astype(float)
    if values.ndim != 2 or values.shape[1] != len(DEFAULT_DEPTHS):
        raise ValueError(f"Expected n_columns x 21 values in {path}, got {values.shape}")
    return values


def summarize_profile_matrix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return depth-wise means, valid counts, and all-depth valid rows."""

    valid = np.isfinite(values) & (values > 0)
    valid_counts = valid.sum(axis=0).astype(int)
    means = np.full(values.shape[1], np.nan, dtype=float)
    for depth_index in range(values.shape[1]):
        if valid_counts[depth_index]:
            means[depth_index] = values[valid[:, depth_index], depth_index].mean()
    valid_rows = int(valid.all(axis=1).sum())
    return means, valid_counts, valid_rows


def native_profiles(
    subject: str,
    hemisphere: str,
    native_root: Path,
    profile_dir_name: str,
    expected_regions: list[str],
) -> tuple[list[dict], list[dict]]:
    """Aggregate original native column profiles by DK region."""

    profile_dir = native_root / subject / profile_dir_name
    rows: list[dict] = []
    qc_rows: list[dict] = []
    for region in expected_regions:
        path = profile_dir / f"{subject}_{hemisphere}_{region}_md.csv"
        if not path.is_file():
            qc_rows.append(
                {
                    "subject_id": subject,
                    "hemisphere": hemisphere,
                    "method": METHOD_NATIVE,
                    "region": region,
                    "status": "missing_profile",
                    "n_units": 0,
                    "n_valid_depths": 0,
                }
            )
            continue
        values = load_csv_profile(path)
        means, valid_counts, valid_rows = summarize_profile_matrix(values)
        for depth_index, depth in enumerate(DEFAULT_DEPTHS):
            rows.append(
                {
                    "subject_id": subject,
                    "hemisphere": hemisphere,
                    "region": region,
                    "depth_index": depth_index,
                    "depth_fraction": depth,
                    "method": METHOD_NATIVE,
                    "profile_value": means[depth_index],
                    "n_units": values.shape[0],
                    "n_valid_units": int(valid_counts[depth_index]),
                }
            )
        qc_rows.append(
            {
                "subject_id": subject,
                "hemisphere": hemisphere,
                "method": METHOD_NATIVE,
                "region": region,
                "status": "complete" if np.isfinite(means).all() else "partial",
                "n_units": values.shape[0],
                "n_valid_depths": int(np.isfinite(means).sum()),
                "n_valid_units_all_depths": valid_rows,
            }
        )
    return rows, qc_rows


def load_o2_profiles(
    subject: str,
    hemisphere: str,
    o2_root: Path,
    region_masks: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    """Aggregate 21 standard FreeSurfer volume-to-surface maps."""

    map_dir = o2_root / subject / "fsaverage" / hemisphere
    rows: list[dict] = []
    qc_rows: list[dict] = []
    loaded = []
    for depth_index, depth in enumerate(DEFAULT_DEPTHS):
        path = map_dir / f"{subject}_{hemisphere}_md_depth-{depth:.2f}.mgh"
        if not path.is_file():
            raise FileNotFoundError(f"Missing FS DepthMap: {path}")
        values = np.asarray(nib.load(str(path)).dataobj).reshape(-1).astype(float)
        if values.size != EXPECTED_VERTICES:
            raise ValueError(
                f"Unexpected FS DepthMap vertex count in {path}: {values.size}"
            )
        loaded.append(values)
    matrix = np.column_stack(loaded)
    for region, mask in region_masks.items():
        valid = np.isfinite(matrix) & (matrix > 0) & mask[:, None]
        counts = valid.sum(axis=0).astype(int)
        means = np.full(len(DEFAULT_DEPTHS), np.nan, dtype=float)
        for depth_index in range(len(DEFAULT_DEPTHS)):
            if counts[depth_index]:
                means[depth_index] = matrix[valid[:, depth_index], depth_index].mean()
            rows.append(
                {
                    "subject_id": subject,
                    "hemisphere": hemisphere,
                    "region": region,
                    "depth_index": depth_index,
                    "depth_fraction": DEFAULT_DEPTHS[depth_index],
                    "method": METHOD_FS_DEPTHMAP,
                    "profile_value": means[depth_index],
                    "n_units": int(mask.sum()),
                    "n_valid_units": int(counts[depth_index]),
                }
            )
        qc_rows.append(
            {
                "subject_id": subject,
                "hemisphere": hemisphere,
                "method": METHOD_FS_DEPTHMAP,
                "region": region,
                "status": "complete" if np.isfinite(means).all() else "partial",
                "n_units": int(mask.sum()),
                "n_valid_depths": int(np.isfinite(means).sum()),
                "n_valid_units_all_depths": int(valid.all(axis=1).sum()),
            }
        )
    return rows, qc_rows


def load_o3_profiles(
    subject: str,
    hemisphere: str,
    o3_root: Path,
    region_masks: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    """Aggregate Coordinate Interp. fsaverage profiles."""

    path = (
        o3_root
        / subject
        / "fsaverage"
        / "o3_coordinate"
        / f"{subject}_{hemisphere}_profiles.npz"
    )
    with np.load(path, allow_pickle=True) as data:
        depths = np.asarray(data["depths"], dtype=float)
        matrix = np.asarray(data["md_profiles"], dtype=float)
        mapping_valid = np.asarray(data["mapping_valid"], dtype=bool)
        white_anchor = np.asarray(data["white_anchor"], dtype=float)
        pial_anchor = np.asarray(data["pial_anchor"], dtype=float)
        if "coordinate_in_bounds_mask" in data:
            in_bounds = np.asarray(data["coordinate_in_bounds_mask"], dtype=bool)
        else:
            in_bounds = np.ones(matrix.shape, dtype=bool)
    if not np.allclose(depths, DEFAULT_DEPTHS):
        raise ValueError(f"Unexpected O3 depth fractions in {path}: {depths}")
    if matrix.shape != (EXPECTED_VERTICES, len(DEFAULT_DEPTHS)):
        raise ValueError(f"Unexpected O3 profile shape in {path}: {matrix.shape}")
    line_length = np.linalg.norm(pial_anchor - white_anchor, axis=1)
    geometry_valid = mapping_valid & (line_length > ZERO_LENGTH_TOLERANCE_MM)
    rows: list[dict] = []
    qc_rows: list[dict] = []
    for region, mask in region_masks.items():
        valid = (
            np.isfinite(matrix)
            & (matrix > 0)
            & in_bounds
            & geometry_valid[:, None]
            & mask[:, None]
        )
        counts = valid.sum(axis=0).astype(int)
        means = np.full(len(DEFAULT_DEPTHS), np.nan, dtype=float)
        for depth_index in range(len(DEFAULT_DEPTHS)):
            if counts[depth_index]:
                means[depth_index] = matrix[valid[:, depth_index], depth_index].mean()
            rows.append(
                {
                    "subject_id": subject,
                    "hemisphere": hemisphere,
                    "region": region,
                    "depth_index": depth_index,
                    "depth_fraction": DEFAULT_DEPTHS[depth_index],
                    "method": METHOD_COORDINATE_INTERP,
                    "profile_value": means[depth_index],
                    "n_units": int(mask.sum()),
                    "n_valid_units": int(counts[depth_index]),
                }
            )
        qc_rows.append(
            {
                "subject_id": subject,
                "hemisphere": hemisphere,
                "method": METHOD_COORDINATE_INTERP,
                "region": region,
                "status": "complete" if np.isfinite(means).all() else "partial",
                "n_units": int(mask.sum()),
                "n_valid_depths": int(np.isfinite(means).sum()),
                "n_valid_units_all_depths": int(valid.all(axis=1).sum()),
                "n_zero_length_units": int((mask & ~geometry_valid).sum()),
            }
        )
    return rows, qc_rows


def load_o2_column_profiles(
    subject: str,
    hemisphere: str,
    o2_column_root: Path,
    region_masks: dict[str, np.ndarray],
) -> tuple[list[dict], list[dict]]:
    """Aggregate strict Profile Interp. barycentric value profiles."""

    path = (
        o2_column_root
        / subject
        / "fsaverage"
        / "o2_column"
        / f"{subject}_{hemisphere}_profiles.npz"
    )
    with np.load(path, allow_pickle=True) as data:
        matrix = np.asarray(data["md_profiles"], dtype=float)
        valid_mask = np.asarray(data["valid_mask"], dtype=bool)
        depths = np.asarray(data["depths"], dtype=float)
    if not np.allclose(depths, DEFAULT_DEPTHS):
        raise ValueError(f"Unexpected Profile Interp. depths in {path}: {depths}")
    if matrix.shape != (EXPECTED_VERTICES, len(DEFAULT_DEPTHS)):
        raise ValueError(
            f"Unexpected Profile Interp. shape in {path}: {matrix.shape}"
        )
    geometry_valid = valid_mask & np.isfinite(matrix).all(axis=1)
    rows: list[dict] = []
    qc_rows: list[dict] = []
    for region, mask in region_masks.items():
        valid = (
            np.isfinite(matrix)
            & (matrix > 0)
            & geometry_valid[:, None]
            & mask[:, None]
        )
        counts = valid.sum(axis=0).astype(int)
        means = np.full(len(DEFAULT_DEPTHS), np.nan, dtype=float)
        for depth_index in range(len(DEFAULT_DEPTHS)):
            if counts[depth_index]:
                means[depth_index] = matrix[valid[:, depth_index], depth_index].mean()
            rows.append(
                {
                    "subject_id": subject,
                    "hemisphere": hemisphere,
                    "region": region,
                    "depth_index": depth_index,
                    "depth_fraction": DEFAULT_DEPTHS[depth_index],
                    "method": METHOD_PROFILE_INTERP,
                    "profile_value": means[depth_index],
                    "n_units": int(mask.sum()),
                    "n_valid_units": int(counts[depth_index]),
                }
            )
        qc_rows.append(
            {
                "subject_id": subject,
                "hemisphere": hemisphere,
                "method": METHOD_PROFILE_INTERP,
                "region": region,
                "status": "complete" if np.isfinite(means).all() else "partial",
                "n_units": int(mask.sum()),
                "n_valid_depths": int(np.isfinite(means).sum()),
                "n_valid_units_all_depths": int(valid.all(axis=1).sum()),
            }
        )
    return rows, qc_rows


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Calculate Pearson correlation, returning NaN for constant vectors."""

    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def pair_metrics(
    profiles: pd.DataFrame,
    method_a: str,
    method_b: str,
    min_valid_depths: int,
) -> pd.DataFrame:
    """Compare two regional profiles for each subject and hemisphere."""

    key = ["subject_id", "hemisphere", "region", "depth_index"]
    subset = profiles[profiles.method.isin([method_a, method_b])]
    wide = subset.pivot_table(
        index=key,
        columns="method",
        values="profile_value",
        aggfunc="first",
    ).reset_index()
    rows = []
    group_key = ["subject_id", "hemisphere", "region"]
    for identifiers, group in wide.groupby(group_key, sort=True):
        left = group[method_a].to_numpy(dtype=float)
        right = group[method_b].to_numpy(dtype=float)
        valid = np.isfinite(left) & np.isfinite(right)
        left = left[valid]
        right = right[valid]
        if left.size < min_valid_depths:
            metrics = {
                "n_valid_depths": int(left.size),
                "pearson": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "signed_bias_a_minus_b": np.nan,
                "sNMAE_pct": np.nan,
            }
        else:
            difference = left - right
            denominator = np.median(np.abs(left)) + np.median(np.abs(right))
            metrics = {
                "n_valid_depths": int(left.size),
                "pearson": safe_pearson(left, right),
                "mae": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(difference**2))),
                "signed_bias_a_minus_b": float(np.mean(difference)),
                "sNMAE_pct": (
                    float(200.0 * np.mean(np.abs(difference)) / denominator)
                    if denominator > 0
                    else np.nan
                ),
            }
        rows.append(
            {
                "subject_id": identifiers[0],
                "hemisphere": identifiers[1],
                "region": identifiers[2],
                "method_a": method_a,
                "method_b": method_b,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize pairwise metrics across subject and hemisphere units."""

    metric_columns = [
        "pearson",
        "mae",
        "rmse",
        "signed_bias_a_minus_b",
        "sNMAE_pct",
    ]
    group_columns = ["method_a", "method_b", "region"]
    rows = []
    for identifiers, group in metrics.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, identifiers))
        row["n_subject_hemisphere_units"] = len(group)
        row["n_subjects"] = group.subject_id.nunique()
        for column in metric_columns:
            values = group[column].dropna()
            row[f"{column}_median"] = values.median() if not values.empty else np.nan
            row[f"{column}_q25"] = values.quantile(0.25) if not values.empty else np.nan
            row[f"{column}_q75"] = values.quantile(0.75) if not values.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    """Run the three-method regional comparison."""

    manifest = load_manifest(args.manifest, args.subject_id)
    labels_by_hemi = {
        hemisphere: load_fsaverage_labels(args.freesurfer_subjects_dir, hemisphere)
        for hemisphere in ("lh", "rh")
    }
    profile_rows: list[dict] = []
    qc_rows: list[dict] = []
    total = len(manifest)
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        subject = row.subject_id
        logging.info("Progress %d/%d subject=%s", index, total, subject)
        for hemisphere in ("lh", "rh"):
            _, region_masks = labels_by_hemi[hemisphere]
            regions = sorted(region_masks)
            native, native_qc = native_profiles(
                subject,
                hemisphere,
                args.native_profile_root,
                args.profile_dir_name,
                regions,
            )
            o2, o2_qc = load_o2_profiles(
                subject,
                hemisphere,
                args.o2_root,
                region_masks,
            )
            o3, o3_qc = load_o3_profiles(
                subject,
                hemisphere,
                args.o3_root,
                region_masks,
            )
            o2_column, o2_column_qc = load_o2_column_profiles(
                subject,
                hemisphere,
                args.o2_column_root,
                region_masks,
            )
            profile_rows.extend(native + o2 + o2_column + o3)
            qc_rows.extend(native_qc + o2_qc + o2_column_qc + o3_qc)
            logging.info(
                "Completed subject=%s hemisphere=%s regions=%d",
                subject,
                hemisphere,
                len(regions),
            )

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    profiles = pd.DataFrame(profile_rows)
    qc = pd.DataFrame(qc_rows)
    profiles.to_csv(output_root / "regional_profiles_long.csv", index=False)
    qc.to_csv(output_root / "regional_profile_qc.csv", index=False)

    pair_tables = []
    pairs = [
        (METHOD_FS_DEPTHMAP, METHOD_NATIVE),
        (METHOD_PROFILE_INTERP, METHOD_NATIVE),
        (METHOD_COORDINATE_INTERP, METHOD_NATIVE),
        (METHOD_COORDINATE_INTERP, METHOD_PROFILE_INTERP),
    ]
    for method_a, method_b in pairs:
        logging.info("Calculating pair metrics: %s vs %s", method_a, method_b)
        pair_tables.append(
            pair_metrics(profiles, method_a, method_b, args.min_valid_depths)
        )
    metrics = pd.concat(pair_tables, ignore_index=True)
    metrics.to_csv(output_root / "regional_method_metrics.csv", index=False)
    summary = summarize_metrics(metrics)
    summary.to_csv(output_root / "regional_method_summary.csv", index=False)

    depth_metrics = []
    for method_a, method_b in pairs:
        subset = profiles[profiles.method.isin([method_a, method_b])]
        wide = subset.pivot_table(
            index=["subject_id", "hemisphere", "region", "depth_index", "depth_fraction"],
            columns="method",
            values="profile_value",
            aggfunc="first",
        ).reset_index()
        if method_a not in wide or method_b not in wide:
            continue
        wide["valid"] = np.isfinite(wide[method_a]) & np.isfinite(wide[method_b])
        wide["difference_a_minus_b"] = wide[method_a] - wide[method_b]
        wide["abs_difference"] = wide["difference_a_minus_b"].abs()
        wide["method_a"] = method_a
        wide["method_b"] = method_b
        depth_metrics.append(wide)
    depth_long = pd.concat(depth_metrics, ignore_index=True)
    depth_summary = (
        depth_long[depth_long.valid]
        .groupby(["method_a", "method_b", "region", "depth_index", "depth_fraction"])
        .agg(
            n_units=("difference_a_minus_b", "size"),
            mae=("abs_difference", "mean"),
            signed_bias_a_minus_b=("difference_a_minus_b", "mean"),
        )
        .reset_index()
    )
    depth_summary.to_csv(output_root / "regional_depthwise_method_summary.csv", index=False)

    metadata = {
        "manifest": str(args.manifest),
        "n_subjects": int(manifest.subject_id.nunique()),
        "subjects": manifest.subject_id.tolist(),
        "methods": [
            METHOD_NATIVE,
            METHOD_FS_DEPTHMAP,
            METHOD_PROFILE_INTERP,
            METHOD_COORDINATE_INTERP,
        ],
        "template": "fsaverage",
        "hemispheres": ["lh", "rh"],
        "depth_fractions": DEFAULT_DEPTHS.tolist(),
        "native_profile_root": str(args.native_profile_root),
        "o2_root": str(args.o2_root),
        "o2_column_root": str(args.o2_column_root),
        "o3_root": str(args.o3_root),
        "validity_rule": "finite and > 0; O3 additionally excludes zero-length columns and out-of-bounds coordinates",
    }
    pd.Series(metadata, dtype=object).to_json(
        output_root / "run_metadata.json",
        indent=2,
    )
    logging.info("Finished. Outputs written to %s", output_root)


def main() -> None:
    """Run the command-line program."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(parse_args())


if __name__ == "__main__":
    main()
