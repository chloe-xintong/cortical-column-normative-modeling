#!/usr/bin/env python
"""Compute the interpolation-order comparison outside the notebook.

Description
-----------
The script compares O2-Column and O3-Coordinate in template space. It owns
the expensive array, vertex, group, and cross-subject calculations. The
companion notebook only reads the tables and figures produced here.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--o2-root`` command-line input
- ``--o3-root`` command-line input
- ``--template`` command-line input
- ``--output-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python qc/archive/template_registration_comparison/interpolation_order/py/compare_o2_column_o3_coordinate.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


HEMISPHERES = ("lh", "rh")
DEPTH_COUNT = 21
DEFAULT_GROUP_SIZES = (1, 50, 100, 200, 400)
LOGGER = logging.getLogger(__name__)


def read_manifest(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    subjects = [row["subject_id"] for row in rows]
    if len(subjects) != len(set(subjects)):
        raise ValueError("The comparison manifest contains duplicate subjects")
    return subjects


def profile_path(
    root: Path,
    subject: str,
    template: str,
    hemi: str,
    method: str,
) -> Path:
    subdir = "o2_column" if method == "O2-Column" else "o3_coordinate"
    return root / subject / template / subdir / f"{subject}_{hemi}_profiles.npz"


def load_profile(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        profiles = np.asarray(data["md_profiles"], dtype=np.float32)
        target_ids = np.asarray(data["target_vertex_id"], dtype=np.int64)
        valid_mask = np.asarray(
            data["valid_mask"]
            if "valid_mask" in data.files
            else np.isfinite(profiles).all(axis=1),
            dtype=bool,
        )
        raw_in_bounds = np.asarray(
            data["raw_in_bounds_mask"]
            if "raw_in_bounds_mask" in data.files
            else np.ones_like(profiles, dtype=bool),
            dtype=bool,
        )
        sampled_finite = np.asarray(
            data["sampled_finite_mask"]
            if "sampled_finite_mask" in data.files
            else np.isfinite(profiles),
            dtype=bool,
        )
        mapping_valid = np.asarray(
            data["mapping_valid"] if "mapping_valid" in data.files else valid_mask,
            dtype=bool,
        )

    if profiles.ndim != 2 or profiles.shape[1] != DEPTH_COUNT:
        raise ValueError(f"Expected vertex x 21 profiles in {path}")
    if target_ids.ndim != 1 or target_ids.size != profiles.shape[0]:
        raise ValueError(f"Invalid target vertex IDs in {path}")
    expected_shape = profiles.shape
    for name, value in {
        "raw_in_bounds_mask": raw_in_bounds,
        "sampled_finite_mask": sampled_finite,
    }.items():
        if value.shape != expected_shape:
            raise ValueError(f"Invalid {name} in {path}")
    if valid_mask.shape != (profiles.shape[0],):
        raise ValueError(f"Invalid valid_mask in {path}")
    if mapping_valid.shape != (profiles.shape[0],):
        raise ValueError(f"Invalid mapping_valid in {path}")
    return {
        "profiles": profiles,
        "target_ids": target_ids,
        "valid_mask": valid_mask,
        "raw_in_bounds": raw_in_bounds,
        "sampled_finite": sampled_finite,
        "mapping_valid": mapping_valid,
    }


def load_qc(path: Path) -> dict:
    qc_path = path.with_suffix(".qc.json")
    return json.loads(qc_path.read_text()) if qc_path.exists() else {}


def align_profile(
    profile: dict[str, np.ndarray],
    target_ids: np.ndarray,
    vertex_count: int,
) -> dict[str, np.ndarray]:
    aligned = {
        "profiles": np.full((vertex_count, DEPTH_COUNT), np.nan, dtype=np.float32),
        "valid_mask": np.zeros(vertex_count, dtype=bool),
        "mapping_valid": np.zeros(vertex_count, dtype=bool),
        "raw_in_bounds": np.zeros((vertex_count, DEPTH_COUNT), dtype=bool),
        "sampled_finite": np.zeros((vertex_count, DEPTH_COUNT), dtype=bool),
    }
    order = np.argsort(profile["target_ids"])
    source_ids = profile["target_ids"][order]
    indices = np.searchsorted(target_ids, source_ids)
    keep = indices < vertex_count
    in_range = np.flatnonzero(keep)
    keep[in_range] &= target_ids[indices[in_range]] == source_ids[in_range]
    destination = indices[keep]
    source = order[keep]
    for key in aligned:
        aligned[key][destination] = profile[key][source]
    return aligned


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denominator == 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denominator)


def calculate_pair_metrics(
    o2: np.ndarray,
    o3: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    x = np.asarray(o2[valid], dtype=float)
    y = np.asarray(o3[valid], dtype=float)
    difference = y - x
    if x.size == 0:
        return {
            "n_vertices": 0,
            "pearson_r": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "mean_bias_o3_minus_o2": np.nan,
            "nmae_pct": np.nan,
            "snmae_pct": np.nan,
        }
    median_reference = float(np.median(x))
    nmae = (
        100.0 * float(np.mean(np.abs(difference))) / abs(median_reference)
        if median_reference != 0 and np.isfinite(median_reference)
        else np.nan
    )
    symmetric_denominator = np.median(np.abs(x)) + np.median(np.abs(y))
    snmae = (
        200.0 * float(np.mean(np.abs(difference))) / symmetric_denominator
        if symmetric_denominator != 0 and np.isfinite(symmetric_denominator)
        else np.nan
    )
    return {
        "n_vertices": int(x.size),
        "pearson_r": safe_correlation(x, y),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mean_bias_o3_minus_o2": float(np.mean(difference)),
        "nmae_pct": nmae,
        "snmae_pct": snmae,
    }


def summarize_by_depth(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for depth, group in metrics.groupby("depth"):
        row = {"depth": depth, "n_subject_hemisphere": len(group)}
        for metric in (
            "pearson_r",
            "mae",
            "rmse",
            "mean_bias_o3_minus_o2",
            "nmae_pct",
            "snmae_pct",
        ):
            values = group[metric].dropna().to_numpy()
            row[f"{metric}_median"] = np.nanmedian(values) if values.size else np.nan
            row[f"{metric}_q25"] = (
                np.nanpercentile(values, 25) if values.size else np.nan
            )
            row[f"{metric}_q75"] = (
                np.nanpercentile(values, 75) if values.size else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_within_subject(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject, depth), group in metrics.groupby(["subject", "depth"]):
        row = {
            "subject": subject,
            "depth": depth,
            "n_hemispheres": int(group["hemisphere"].nunique()),
        }
        for metric in (
            "pearson_r",
            "mae",
            "rmse",
            "mean_bias_o3_minus_o2",
            "nmae_pct",
            "snmae_pct",
        ):
            values = group[metric].dropna().to_numpy()
            row[metric] = np.nanmedian(values) if values.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def rowwise_profile_metrics(
    o2: np.ndarray,
    o3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    difference = o3 - o2
    o2_centered = o2 - np.nanmean(o2, axis=1, keepdims=True)
    o3_centered = o3 - np.nanmean(o3, axis=1, keepdims=True)
    denominator = np.sqrt(
        np.nansum(o2_centered**2, axis=1) * np.nansum(o3_centered**2, axis=1)
    )
    correlation = np.divide(
        np.nansum(o2_centered * o3_centered, axis=1),
        denominator,
        out=np.full(o2.shape[0], np.nan),
        where=denominator > 0,
    )
    mae = np.nanmean(np.abs(difference), axis=1)
    denominator_snmae = np.nanmedian(np.abs(o2), axis=1) + np.nanmedian(
        np.abs(o3), axis=1
    )
    snmae = np.divide(
        200.0 * mae,
        denominator_snmae,
        out=np.full(o2.shape[0], np.nan),
        where=np.isfinite(denominator_snmae) & (denominator_snmae > 0),
    )
    return correlation, mae, snmae


def profile_summary(
    paired: dict[tuple[str, str], dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    for (subject, hemi), values in paired.items():
        valid = values["common"]
        correlations, maes, snmaes = rowwise_profile_metrics(
            values["o2"][valid],
            values["o3"][valid],
        )
        finite_corr = np.isfinite(correlations)
        rows.append(
            {
                "subject": subject,
                "hemisphere": hemi,
                "n_common_vertices": int(valid.sum()),
                "n_profile_correlations": int(finite_corr.sum()),
                "profile_correlation_median": np.nanmedian(correlations),
                "profile_correlation_q25": np.nanpercentile(
                    correlations[finite_corr], 25
                ),
                "profile_correlation_q75": np.nanpercentile(
                    correlations[finite_corr], 75
                ),
                "profile_mae_median": np.nanmedian(maes),
                "profile_mae_q25": np.nanpercentile(maes, 25),
                "profile_mae_q75": np.nanpercentile(maes, 75),
                "profile_snmae_median": np.nanmedian(snmaes),
                "profile_snmae_q25": np.nanpercentile(snmaes, 25),
                "profile_snmae_q75": np.nanpercentile(snmaes, 75),
            }
        )
    return pd.DataFrame(rows)


def cross_subject_summary(
    o2_stack: np.ndarray,
    o3_stack: np.ndarray,
    common_stack: np.ndarray,
    depth_index: int,
    depth: float,
    min_subjects: int,
) -> dict[str, float]:
    x = o2_stack[:, :, depth_index]
    y = o3_stack[:, :, depth_index]
    valid = common_stack & np.isfinite(x) & np.isfinite(y)
    count = valid.sum(axis=0)
    x_mean = np.divide(
        np.where(valid, x, 0).sum(axis=0),
        count,
        out=np.full(x.shape[1], np.nan),
        where=count > 0,
    )
    y_mean = np.divide(
        np.where(valid, y, 0).sum(axis=0),
        count,
        out=np.full(y.shape[1], np.nan),
        where=count > 0,
    )
    x_centered = np.where(valid, x - x_mean, 0)
    y_centered = np.where(valid, y - y_mean, 0)
    denominator = np.sqrt(
        np.sum(x_centered**2, axis=0) * np.sum(y_centered**2, axis=0)
    )
    correlations = np.divide(
        np.sum(x_centered * y_centered, axis=0),
        denominator,
        out=np.full(x.shape[1], np.nan),
        where=(count >= min_subjects) & (denominator > 0),
    )
    difference = np.where(valid, np.abs(y - x), np.nan)
    method_difference = np.nanmedian(difference, axis=0)
    method_mean = np.where(valid, (x + y) / 2.0, np.nan)
    variation = np.nanstd(method_mean, axis=0)
    ratio = np.divide(
        method_difference,
        variation,
        out=np.full(method_difference.shape, np.nan),
        where=variation > 0,
    )
    valid_corr = np.isfinite(correlations)
    valid_ratio = np.isfinite(ratio) & (count >= min_subjects)
    return {
        "depth": depth,
        "n_vertices": int(valid_corr.sum()),
        "correlation_median": np.nanmedian(correlations[valid_corr]),
        "correlation_q25": np.nanpercentile(correlations[valid_corr], 25),
        "correlation_q75": np.nanpercentile(correlations[valid_corr], 75),
        "pct_below_0_8": 100.0 * float(np.mean(correlations[valid_corr] < 0.8)),
        "pct_below_0_9": 100.0 * float(np.mean(correlations[valid_corr] < 0.9)),
        "ratio_n_vertices": int(valid_ratio.sum()),
        "ratio_median": np.nanmedian(ratio[valid_ratio]),
        "ratio_q25": np.nanpercentile(ratio[valid_ratio], 25),
        "ratio_q75": np.nanpercentile(ratio[valid_ratio], 75),
    }


def read_template_mesh(template: str, hemi: str) -> tuple[np.ndarray, np.ndarray]:
    surface = Path(f"/home/apps/freesurfer/subjects/{template}/surf/{hemi}.white")
    return nib.freesurfer.read_geometry(surface)


def read_cortex_labels(template: str, hemi: str) -> np.ndarray:
    annot = Path(f"/home/apps/freesurfer/subjects/{template}/label/{hemi}.aparc.annot")
    labels, _, names = nib.freesurfer.read_annot(annot)
    decoded = [name.decode().lower() for name in names]
    excluded = {"unknown", "medial_wall", "corpuscallosum"}
    excluded_ids = {index for index, name in enumerate(decoded) if name in excluded}
    result = labels.astype(np.int32)
    for label in excluded_ids:
        result[result == label] = -1
    return result


def build_region_groups(labels: np.ndarray) -> np.ndarray:
    group_ids = np.full(labels.shape, -1, dtype=np.int32)
    next_group = 0
    for label in sorted(set(labels.tolist()) - {-1}):
        group_ids[labels == label] = next_group
        next_group += 1
    return group_ids


def build_spatial_groups(
    faces: np.ndarray,
    labels: np.ndarray,
    target_size: int,
) -> np.ndarray:
    adjacency = [[] for _ in range(labels.size)]
    for triangle in faces:
        a, b, c = map(int, triangle)
        adjacency[a].extend((b, c))
        adjacency[b].extend((a, c))
        adjacency[c].extend((a, b))
    adjacency = [np.unique(neighbors) for neighbors in adjacency]
    group_ids = np.full(labels.shape, -1, dtype=np.int32)
    next_group = 0
    for label in sorted(set(labels.tolist()) - {-1}):
        remaining = set(np.flatnonzero(labels == label).tolist())
        while remaining:
            seed = min(remaining)
            queue = [seed]
            group = []
            while queue and len(group) < target_size:
                vertex = queue.pop(0)
                if vertex not in remaining:
                    continue
                remaining.remove(vertex)
                group.append(vertex)
                queue.extend(
                    int(neighbor)
                    for neighbor in adjacency[vertex]
                    if int(neighbor) in remaining
                )
            group_ids[group] = next_group
            next_group += 1
    return group_ids


def get_group_memberships(
    path: Path,
    legacy_path: Path,
    template: str,
    group_sizes: tuple[int, ...],
) -> dict[str, np.ndarray]:
    expected = [
        f"{hemi}_size_{size}" for hemi in HEMISPHERES for size in group_sizes
    ] + [f"{hemi}_regional_mean" for hemi in HEMISPHERES]
    loaded = {}
    for candidate in (path, legacy_path):
        if candidate.exists():
            with np.load(candidate, allow_pickle=False) as data:
                loaded.update({key: data[key] for key in data.files})
            LOGGER.info("PROGRESS found existing group membership: %s", candidate)
    if set(expected).issubset(loaded):
        return {key: loaded[key] for key in expected}
    memberships = {}
    for hemi in HEMISPHERES:
        missing_sizes = [
            size for size in group_sizes if f"{hemi}_size_{size}" not in loaded
        ]
        labels = None
        if f"{hemi}_regional_mean" not in loaded or missing_sizes:
            coordinates, faces = read_template_mesh(template, hemi)
            del coordinates
            labels = read_cortex_labels(template, hemi)
        if f"{hemi}_regional_mean" in loaded:
            memberships[f"{hemi}_regional_mean"] = loaded[f"{hemi}_regional_mean"]
        else:
            memberships[f"{hemi}_regional_mean"] = build_region_groups(labels)
        for size in group_sizes:
            key = f"{hemi}_size_{size}"
            if key in loaded:
                memberships[key] = loaded[key]
            else:
                memberships[key] = build_spatial_groups(faces, labels, size)
        LOGGER.info(
            "PROGRESS prepared fixed groups for %s: %d vertices",
            hemi,
            memberships[f"{hemi}_size_{group_sizes[0]}"].size,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{**loaded, **memberships})
    return memberships


def aggregate_groups(
    profile: np.ndarray,
    valid: np.ndarray,
    group_ids: np.ndarray,
) -> np.ndarray:
    keep = valid & (group_ids >= 0)
    group_count = int(group_ids.max()) + 1
    output = np.full((group_count, DEPTH_COUNT), np.nan, dtype=np.float32)
    if not keep.any():
        return output
    ids = group_ids[keep]
    counts = np.bincount(ids, minlength=group_count).astype(float)
    for depth in range(DEPTH_COUNT):
        sums = np.bincount(
            ids,
            weights=profile[keep, depth],
            minlength=group_count,
        )
        output[:, depth] = np.divide(
            sums,
            counts,
            out=np.full(group_count, np.nan),
            where=counts > 0,
        )
    return output


def calculate_group_metrics(
    paired: dict[tuple[str, str], dict[str, np.ndarray]],
    subjects: list[str],
    memberships: dict[str, np.ndarray],
    group_sizes: tuple[int, ...],
    min_subjects: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_rows = []
    cross_rows = []
    configurations = [(size, str(size)) for size in group_sizes] + [
        (None, "regional_mean")
    ]
    for requested_size, label in configurations:
        for hemi in HEMISPHERES:
            key = (
                f"{hemi}_regional_mean"
                if requested_size is None
                else f"{hemi}_size_{requested_size}"
            )
            group_ids = memberships[key]
            subject_group_o2 = []
            subject_group_o3 = []
            subject_group_valid = []
            for subject in subjects:
                values = paired[(subject, hemi)]
                group_o2 = aggregate_groups(values["o2"], values["common"], group_ids)
                group_o3 = aggregate_groups(values["o3"], values["common"], group_ids)
                group_valid = np.isfinite(group_o2).all(axis=1) & np.isfinite(
                    group_o3
                ).all(axis=1)
                subject_group_o2.append(group_o2)
                subject_group_o3.append(group_o3)
                subject_group_valid.append(group_valid)
                correlations, maes, snmaes = rowwise_profile_metrics(
                    group_o2[group_valid],
                    group_o3[group_valid],
                )
                subject_rows.append(
                    {
                        "subject": subject,
                        "hemisphere": hemi,
                        "group_size": label,
                        "requested_group_size": requested_size,
                        "n_groups": int(group_valid.sum()),
                        "profile_correlation_median": np.nanmedian(correlations),
                        "profile_mae_median": np.nanmedian(maes),
                        "profile_snmae_median": np.nanmedian(snmaes),
                    }
                )
            o2_stack = np.stack(subject_group_o2)
            o3_stack = np.stack(subject_group_o3)
            valid_stack = np.stack(subject_group_valid)
            for depth_index, depth in enumerate(np.linspace(0, 1, DEPTH_COUNT)):
                x = o2_stack[:, :, depth_index]
                y = o3_stack[:, :, depth_index]
                valid = valid_stack & np.isfinite(x) & np.isfinite(y)
                count = valid.sum(axis=0)
                x_mean = np.divide(
                    np.where(valid, x, 0).sum(axis=0),
                    count,
                    out=np.full(x.shape[1], np.nan),
                    where=count > 0,
                )
                y_mean = np.divide(
                    np.where(valid, y, 0).sum(axis=0),
                    count,
                    out=np.full(y.shape[1], np.nan),
                    where=count > 0,
                )
                x_centered = np.where(valid, x - x_mean, 0)
                y_centered = np.where(valid, y - y_mean, 0)
                denominator = np.sqrt(
                    np.sum(x_centered**2, axis=0) * np.sum(y_centered**2, axis=0)
                )
                correlation = np.divide(
                    np.sum(x_centered * y_centered, axis=0),
                    denominator,
                    out=np.full(x.shape[1], np.nan),
                    where=(count >= min_subjects) & (denominator > 0),
                )
                finite = np.isfinite(correlation)
                cross_rows.append(
                    {
                        "hemisphere": hemi,
                        "group_size": label,
                        "requested_group_size": requested_size,
                        "depth": depth,
                        "n_groups": int(finite.sum()),
                        "cross_subject_correlation_median": np.nanmedian(
                            correlation[finite]
                        ),
                        "cross_subject_correlation_q25": np.nanpercentile(
                            correlation[finite], 25
                        ),
                        "cross_subject_correlation_q75": np.nanpercentile(
                            correlation[finite], 75
                        ),
                    }
                )
            LOGGER.info("PROGRESS group metrics: group=%s hemisphere=%s", label, hemi)
    return pd.DataFrame(subject_rows), pd.DataFrame(cross_rows)


def save_difference_maps(
    paired: dict[tuple[str, str], dict[str, np.ndarray]],
    subjects: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for hemi in HEMISPHERES:
        differences = []
        for subject in subjects:
            values = paired[(subject, hemi)]
            difference = values["o3"][:, 10] - values["o2"][:, 10]
            differences.append(np.where(values["common"], difference, np.nan))
        median_difference = np.nanmedian(np.stack(differences), axis=0).astype(
            np.float32
        )
        path = output_dir / f"fsaverage_{hemi}_depth50_o3_minus_o2.mgh"
        nib.freesurfer.write_morph_data(path, median_difference)
        LOGGER.info("PROGRESS wrote 50%% difference map: %s", path)


def run(args: argparse.Namespace) -> None:
    subjects = read_manifest(args.manifest)
    output_root = args.output_root
    qc_root = output_root / "qc"
    metrics_root = output_root / "metrics"
    groups_root = output_root / "groups"
    figures_root = output_root / "figures"
    logs_root = output_root / "logs"
    for directory in (qc_root, metrics_root, groups_root, figures_root, logs_root):
        directory.mkdir(parents=True, exist_ok=True)

    metadata = {
        "template": args.template,
        "manifest": str(args.manifest),
        "subjects": subjects,
        "group_sizes": list(args.group_sizes),
        "min_subjects": args.min_subjects,
        "depths": np.linspace(0.0, 1.0, DEPTH_COUNT).tolist(),
        "methods": {
            "O2-FS": "mri_vol2surf standard volume-to-surface reference",
            "O2-Column": "MD-value interpolation after native-column sampling",
            "O3-Coordinate": (
                "native-coordinate interpolation before MD-volume sampling"
            ),
        },
    }
    (logs_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    qc_rows = []
    subject_rows = []
    paired = {}
    template_ids = None
    vertex_count = None
    o2_stack = None
    o3_stack = None
    common_stack = None

    for subject_index, subject in enumerate(subjects):
        LOGGER.info(
            "PROGRESS subject %d/%d: %s", subject_index + 1, len(subjects), subject
        )
        for hemi_index, hemi in enumerate(HEMISPHERES):
            o2_path = profile_path(
                args.o2_root, subject, args.template, hemi, "O2-Column"
            )
            o3_path = profile_path(
                args.o3_root, subject, args.template, hemi, "O3-Coordinate"
            )
            if not o2_path.exists() or not o3_path.exists():
                raise FileNotFoundError(f"Missing paired outputs for {subject} {hemi}")
            o2 = load_profile(o2_path)
            o3 = load_profile(o3_path)
            if template_ids is None:
                template_ids = np.sort(o2["target_ids"])
                vertex_count = template_ids.size
                shape = (len(subjects), len(HEMISPHERES), vertex_count, DEPTH_COUNT)
                o2_stack = np.full(shape, np.nan, dtype=np.float32)
                o3_stack = np.full(shape, np.nan, dtype=np.float32)
                common_stack = np.zeros(
                    (len(subjects), len(HEMISPHERES), vertex_count),
                    dtype=bool,
                )
            if not np.array_equal(np.sort(o3["target_ids"]), template_ids):
                raise ValueError(f"Target vertex IDs differ for {subject} {hemi}")
            o2_aligned = align_profile(o2, template_ids, vertex_count)
            o3_aligned = align_profile(o3, template_ids, vertex_count)
            common = (
                o2_aligned["valid_mask"]
                & o3_aligned["valid_mask"]
                & o2_aligned["mapping_valid"]
                & o3_aligned["mapping_valid"]
                & o3_aligned["raw_in_bounds"].all(axis=1)
                & o3_aligned["sampled_finite"].all(axis=1)
                & np.isfinite(o2_aligned["profiles"]).all(axis=1)
                & np.isfinite(o3_aligned["profiles"]).all(axis=1)
            )
            o2_stack[subject_index, hemi_index] = o2_aligned["profiles"]
            o3_stack[subject_index, hemi_index] = o3_aligned["profiles"]
            common_stack[subject_index, hemi_index] = common
            paired[(subject, hemi)] = {
                "o2": o2_aligned["profiles"],
                "o3": o3_aligned["profiles"],
                "common": common,
            }
            qc2 = load_qc(o2_path)
            qc3 = load_qc(o3_path)
            qc_rows.append(
                {
                    "subject": subject,
                    "hemisphere": hemi,
                    "o2_valid_vertices": int(o2_aligned["valid_mask"].sum()),
                    "o3_valid_vertices": int(o3_aligned["valid_mask"].sum()),
                    "common_valid_vertices": int(common.sum()),
                    "common_valid_percent": 100.0 * float(common.mean()),
                    "o3_raw_in_bounds_percent": 100.0
                    * float(o3_aligned["raw_in_bounds"].all(axis=1).mean()),
                    "o3_sampled_finite_percent": 100.0
                    * float(o3_aligned["sampled_finite"].all(axis=1).mean()),
                    "o2_method": qc2.get("method", "O2-Column"),
                    "o3_method": qc3.get("method", "O3-Coordinate"),
                }
            )
            for depth_index, depth in enumerate(np.linspace(0.0, 1.0, DEPTH_COUNT)):
                row = {"subject": subject, "hemisphere": hemi, "depth": depth}
                row.update(
                    calculate_pair_metrics(
                        o2_aligned["profiles"][:, depth_index],
                        o3_aligned["profiles"][:, depth_index],
                        common,
                    )
                )
                subject_rows.append(row)

    subject_metrics = pd.DataFrame(subject_rows)
    subject_level_metrics = summarize_within_subject(subject_metrics)
    cohort_summary = summarize_by_depth(subject_level_metrics)
    validation = pd.DataFrame(qc_rows)
    validation.to_csv(qc_root / "input_validation_summary.csv", index=False)
    subject_metrics.to_csv(
        metrics_root / "subject_hemisphere_depth_metrics.csv", index=False
    )
    subject_level_metrics.to_csv(
        metrics_root / "subject_depth_metrics_by_subject.csv", index=False
    )
    cohort_summary.to_csv(metrics_root / "cohort_depth_summary.csv", index=False)
    LOGGER.info("PROGRESS saved depth-wise metrics")

    profile_table = profile_summary(paired)
    profile_table.to_csv(metrics_root / "subject_profile_metrics.csv", index=False)

    cross_rows = []
    ratio_rows = []
    for hemi_index, hemi in enumerate(HEMISPHERES):
        for depth_index, depth in enumerate(np.linspace(0.0, 1.0, DEPTH_COUNT)):
            row = cross_subject_summary(
                o2_stack[:, hemi_index],
                o3_stack[:, hemi_index],
                common_stack[:, hemi_index],
                depth_index,
                float(depth),
                args.min_subjects,
            )
            row["hemisphere"] = hemi
            cross_rows.append(row)
            ratio_rows.append(
                {
                    "hemisphere": hemi,
                    "depth": depth,
                    "ratio_n_vertices": row["ratio_n_vertices"],
                    "ratio_median": row["ratio_median"],
                    "ratio_q25": row["ratio_q25"],
                    "ratio_q75": row["ratio_q75"],
                }
            )
        LOGGER.info("PROGRESS cross-subject metrics: %s", hemi)
    pd.DataFrame(cross_rows).to_csv(
        metrics_root / "cross_subject_metrics.csv", index=False
    )
    pd.DataFrame(ratio_rows).to_csv(
        metrics_root / "method_difference_to_subject_variation.csv",
        index=False,
    )

    memberships = get_group_memberships(
        groups_root / "fixed_group_membership_python.npz",
        groups_root / "fixed_group_membership.npz",
        args.template,
        args.group_sizes,
    )
    group_subject, group_cross = calculate_group_metrics(
        paired,
        subjects,
        memberships,
        args.group_sizes,
        args.min_subjects,
    )
    group_subject.to_csv(
        metrics_root / "spatial_group_subject_metrics.csv", index=False
    )
    group_cross.to_csv(
        metrics_root / "spatial_group_cross_subject_metrics.csv", index=False
    )

    save_difference_maps(paired, subjects, figures_root)
    (qc_root / "analysis_complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "n_subjects": len(subjects),
                "n_vertices": int(vertex_count),
                "n_qc_rows": len(validation),
                "n_depth_rows": len(subject_metrics),
                "n_group_subject_rows": len(group_subject),
            },
            indent=2,
        )
    )
    LOGGER.info(
        "COMPLETED Python interpolation-order analysis: %d subjects", len(subjects)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--o2-root", type=Path, required=True)
    parser.add_argument("--o3-root", type=Path, required=True)
    parser.add_argument(
        "--template", choices=["fsaverage", "fsaverage6"], required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-subjects", type=int, default=20)
    parser.add_argument(
        "--group-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_GROUP_SIZES),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(parse_args())
