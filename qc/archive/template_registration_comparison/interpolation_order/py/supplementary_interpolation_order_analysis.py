#!/usr/bin/env python
"""Compute provenance QC and supplementary interpolation-order analyses.

Description
-----------
This script reads the existing fsaverage O2-Column, O3-Coordinate, and O2-FS
profiles. It does not rerun volume sampling or surface mapping. Outputs are
written beside the primary Python comparison results for the summary notebook.

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
- ``--o2fs-root`` command-line input
- ``--template`` command-line input
- ``--output-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python qc/archive/template_registration_comparison/interpolation_order/py/supplementary_interpolation_order_analysis.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import nibabel as nib
import numpy as np
import pandas as pd

from compare_o2_column_o3_coordinate import (
    DEPTH_COUNT,
    HEMISPHERES,
    aggregate_groups,
    align_profile,
    calculate_pair_metrics,
    get_group_memberships,
    load_profile,
    profile_path,
    read_cortex_labels,
    read_template_mesh,
    rowwise_profile_metrics,
    safe_correlation,
)


LOGGER = logging.getLogger(__name__)
DEPTHS = np.linspace(0.0, 1.0, DEPTH_COUNT)


def read_manifest(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    subjects = [row["subject_id"] for row in rows]
    if len(subjects) != len(set(subjects)):
        raise ValueError("The comparison manifest contains duplicate subjects")
    return subjects


def load_raw_provenance(path: Path) -> dict[str, np.ndarray | str | float]:
    with np.load(path, allow_pickle=True) as data:
        result = {}
        for key in (
            "target_vertex_id",
            "source_vertex_ids",
            "barycentric_weights",
            "mapping_valid",
            "weights_valid",
            "source_profile_available",
            "depths",
            "raw_in_bounds_mask",
            "sampled_finite_mask",
            "depth_fractions",
            "sampling_order",
            "sampling_mode",
            "coordinate_indexing",
            "metric_image",
            "registration",
        ):
            if key in data.files:
                value = data[key]
                result[key] = value.item() if value.ndim == 0 else value
    return result


def load_fs_profile(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        profiles = np.asarray(data["profiles"], dtype=np.float32)
        target_ids = np.asarray(data["vertex_ids"], dtype=np.int64)
        valid_mask = np.asarray(data["valid_mask"], dtype=bool)
    if profiles.shape[1] != DEPTH_COUNT:
        raise ValueError(f"Expected 21 depths in {path}")
    return {
        "profiles": profiles,
        "target_ids": target_ids,
        "valid_mask": valid_mask,
        "mapping_valid": valid_mask,
        "raw_in_bounds": np.ones_like(profiles, dtype=bool),
        "sampled_finite": np.isfinite(profiles),
    }


def finite_percent(value: np.ndarray) -> float:
    return 100.0 * float(np.isfinite(value).mean())


def provenance_qc_row(
    subject: str,
    hemi: str,
    o2_path: Path,
    o3_path: Path,
    o2: dict[str, np.ndarray],
    o3: dict[str, np.ndarray],
    raw_o2: dict,
    raw_o3: dict,
    common: np.ndarray,
) -> dict:
    o2_sources = np.asarray(raw_o2.get("source_vertex_ids", []))
    o3_sources = np.asarray(raw_o3.get("source_vertex_ids", []))
    o2_weights = np.asarray(raw_o2.get("barycentric_weights", []))
    o3_weights = np.asarray(raw_o3.get("barycentric_weights", []))
    weights_sum = o3_weights.sum(axis=1) if o3_weights.size else np.array([])
    weights_valid = (
        bool(weights_sum.size)
        and np.allclose(weights_sum, 1.0, atol=1e-6)
        and bool(np.all(o3_weights >= -1e-6))
        and bool(np.all(o3_weights <= 1.0 + 1e-6))
    )
    o3_depths = np.asarray(raw_o3.get("depth_fractions", []), dtype=float)
    return {
        "subject": subject,
        "hemisphere": hemi,
        "o2_available": o2_path.exists(),
        "o3_available": o3_path.exists(),
        "target_vertex_ids_shape": str(
            tuple(np.asarray(raw_o3.get("target_vertex_id", [])).shape)
        ),
        "o2_source_vertex_ids_shape": str(tuple(o2_sources.shape)),
        "o3_source_vertex_ids_shape": str(tuple(o3_sources.shape)),
        "o2_weights_shape": str(tuple(o2_weights.shape)),
        "o3_weights_shape": str(tuple(o3_weights.shape)),
        "same_source_vertex_ids": bool(
            o2_sources.shape == o3_sources.shape
            and np.array_equal(o2_sources, o3_sources)
        ),
        "same_barycentric_weights": bool(
            o2_weights.shape == o3_weights.shape
            and np.allclose(o2_weights, o3_weights, atol=1e-6, rtol=0.0)
        ),
        "weights_sum_error_median": (
            float(np.median(np.abs(weights_sum - 1.0)))
            if weights_sum.size
            else np.nan
        ),
        "weights_sum_error_max": (
            float(np.max(np.abs(weights_sum - 1.0)))
            if weights_sum.size
            else np.nan
        ),
        "weights_within_tolerance": weights_valid,
        "source_profile_available_pct": finite_percent(
            np.asarray(raw_o2.get("source_profile_available", []), dtype=float)
        )
        if "source_profile_available" in raw_o2
        else np.nan,
        "o2_complete_depths_pct": 100.0
        if np.array_equal(np.asarray(raw_o2.get("depths", DEPTHS)), DEPTHS)
        else np.nan,
        "o3_depths_complete": bool(np.array_equal(o3_depths, DEPTHS)),
        "o3_sampling_order": raw_o3.get("sampling_order", np.nan),
        "o3_sampling_mode": raw_o3.get("sampling_mode", ""),
        "o3_coordinate_indexing": raw_o3.get("coordinate_indexing", ""),
        "o3_raw_in_bounds_pct": 100.0
        * float(o3["raw_in_bounds"].all(axis=1).mean()),
        "o3_sampled_finite_pct": 100.0
        * float(o3["sampled_finite"].all(axis=1).mean()),
        "o2_valid_pct": 100.0 * float(o2["valid_mask"].mean()),
        "o3_valid_pct": 100.0 * float(o3["valid_mask"].mean()),
        "common_valid_pct": 100.0 * float(common.mean()),
        "included": bool(
            np.array_equal(
                raw_o2.get("target_vertex_id", []),
                raw_o3.get("target_vertex_id", []),
            )
            and o2_sources.shape == o3_sources.shape
            and np.array_equal(o2_sources, o3_sources)
            and o2_weights.shape == o3_weights.shape
            and np.allclose(o2_weights, o3_weights, atol=1e-6, rtol=0.0)
        ),
    }


def load_paired_profiles(
    subjects: list[str],
    o2_root: Path,
    o3_root: Path,
    template: str,
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    paired = {}
    qc_rows = []
    template_ids_by_hemi = {}
    for subject in subjects:
        for hemi in HEMISPHERES:
            o2_path = profile_path(o2_root, subject, template, hemi, "O2-Column")
            o3_path = profile_path(o3_root, subject, template, hemi, "O3-Coordinate")
            o2 = load_profile(o2_path)
            o3 = load_profile(o3_path)
            template_ids = np.sort(o2["target_ids"])
            template_ids_by_hemi[hemi] = template_ids
            o2_aligned = align_profile(o2, template_ids, template_ids.size)
            o3_aligned = align_profile(o3, template_ids, template_ids.size)
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
            raw_o2 = load_raw_provenance(o2_path)
            raw_o3 = load_raw_provenance(o3_path)
            qc_rows.append(
                provenance_qc_row(
                    subject,
                    hemi,
                    o2_path,
                    o3_path,
                    o2_aligned,
                    o3_aligned,
                    raw_o2,
                    raw_o3,
                    common,
                )
            )
            paired[(subject, hemi)] = {
                "o2": o2_aligned["profiles"],
                "o3": o3_aligned["profiles"],
                "common": common,
                "target_ids": template_ids,
            }
    return paired, pd.DataFrame(qc_rows), template_ids_by_hemi


def save_vertex_profile_metrics(paired: dict, metrics_root: Path) -> pd.DataFrame:
    output_dir = metrics_root / "vertex_profile_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (subject, hemi), values in paired.items():
        valid = values["common"]
        correlation, mae, snmae = rowwise_profile_metrics(
            values["o2"][valid],
            values["o3"][valid],
        )
        target_ids = values["target_ids"][valid]
        output_path = output_dir / f"{subject}_{hemi}.npz"
        np.savez_compressed(
            output_path,
            target_vertex_id=target_ids,
            valid_mask=np.isfinite(correlation),
            profile_pearson=correlation.astype(np.float32),
            profile_mae=mae.astype(np.float32),
            profile_snmae=snmae.astype(np.float32),
        )
        finite = np.isfinite(correlation)
        rows.append(
            {
                "subject": subject,
                "hemisphere": hemi,
                "n_common_vertices": int(valid.sum()),
                "n_profile_correlations": int(finite.sum()),
                "profile_correlation_median": np.nanmedian(correlation),
                "profile_mae_median": np.nanmedian(mae),
                "profile_snmae_median": np.nanmedian(snmae),
                "vertex_metrics_file": str(output_path),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(metrics_root / "vertex_profile_metrics.csv", index=False)
    return table


def edge_array(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]],
        axis=0,
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def profile_correlations_for_pairs(
    profiles: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    left = profiles[pairs[:, 0]]
    right = profiles[pairs[:, 1]]
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1)
    )
    return np.divide(
        np.sum(left_centered * right_centered, axis=1),
        denominator,
        out=np.full(pairs.shape[0], np.nan),
        where=denominator > 0,
    )


def supplementary_spatial_metrics(
    paired: dict,
    template: str,
    output_path: Path,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for hemi in HEMISPHERES:
        coordinates, faces = read_template_mesh(template, hemi)
        edges = edge_array(faces)
        if edges.shape[0] > 50000:
            edges = edges[rng.choice(edges.shape[0], 50000, replace=False)]
        n_vertices = coordinates.shape[0]
        candidate_count = min(100000, n_vertices * 4)
        candidate = rng.integers(0, n_vertices, size=(candidate_count, 2))
        candidate = candidate[candidate[:, 0] != candidate[:, 1]]
        distances = np.linalg.norm(
            coordinates[candidate[:, 0]] - coordinates[candidate[:, 1]],
            axis=1,
        )
        threshold = np.percentile(distances, 75)
        distant = candidate[distances >= threshold]
        for subject in sorted({key[0] for key in paired if key[1] == hemi}):
            values = paired[(subject, hemi)]
            valid = values["common"]
            edge_valid = valid[edges[:, 0]] & valid[edges[:, 1]]
            near = edges[edge_valid]
            distant_valid = valid[distant[:, 0]] & valid[distant[:, 1]]
            far = distant[distant_valid]
            o2_near_corr = profile_correlations_for_pairs(values["o2"], near)
            o3_near_corr = profile_correlations_for_pairs(values["o3"], near)
            o2_far_corr = profile_correlations_for_pairs(values["o2"], far)
            o3_far_corr = profile_correlations_for_pairs(values["o3"], far)
            for depth_index, depth in enumerate(DEPTHS):
                o2_edges = values["o2"][edges[edge_valid], depth_index]
                o3_edges = values["o3"][edges[edge_valid], depth_index]
                o2_edge_difference = np.abs(o2_edges[:, 0] - o2_edges[:, 1])
                o3_edge_difference = np.abs(o3_edges[:, 0] - o3_edges[:, 1])
                rows.append(
                    {
                        "subject": subject,
                        "hemisphere": hemi,
                        "depth": depth,
                        "n_adjacent_edges": int(edge_valid.sum()),
                        "o2_adjacent_difference_median": np.nanmedian(
                            o2_edge_difference
                        ),
                        "o3_adjacent_difference_median": np.nanmedian(
                            o3_edge_difference
                        ),
                        "nearby_profile_corr_o2": np.nanmedian(o2_near_corr),
                        "nearby_profile_corr_o3": np.nanmedian(o3_near_corr),
                        "distant_profile_corr_o2": np.nanmedian(o2_far_corr),
                        "distant_profile_corr_o3": np.nanmedian(o3_far_corr),
                        "nearby_minus_distant_o2": np.nanmedian(o2_near_corr)
                        - np.nanmedian(o2_far_corr),
                        "nearby_minus_distant_o3": np.nanmedian(o3_near_corr)
                        - np.nanmedian(o3_far_corr),
                        "nearby_pair_count": int(np.isfinite(o2_near_corr).sum()),
                        "distant_pair_count": int(np.isfinite(o2_far_corr).sum()),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def o2fs_comparison(
    paired: dict,
    subjects: list[str],
    o2fs_root: Path,
    template: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    depth_rows = []
    regional_rows = []
    for subject in subjects:
        for hemi in HEMISPHERES:
            values = paired[(subject, hemi)]
            path = (
                o2fs_root
                / subject
                / template
                / hemi
                / f"{subject}_{hemi}_md_profiles.npz"
            )
            if not path.exists():
                continue
            fs = load_fs_profile(path)
            fs_aligned = align_profile(fs, values["target_ids"], values["target_ids"].size)
            valid = values["common"] & fs_aligned["valid_mask"] & np.isfinite(
                fs_aligned["profiles"]
            ).all(axis=1)
            for depth_index, depth in enumerate(DEPTHS):
                row = {
                    "subject": subject,
                    "hemisphere": hemi,
                    "depth": depth,
                }
                row.update(
                    calculate_pair_metrics(
                        values["o2"][:, depth_index],
                        fs_aligned["profiles"][:, depth_index],
                        valid,
                    )
                )
                depth_rows.append(row)
            labels = read_cortex_labels(template, hemi)
            group_ids = np.full(labels.shape, -1, dtype=np.int32)
            for label in sorted(set(labels.tolist()) - {-1}):
                group_ids[labels == label] = label
            o2_group = aggregate_groups(values["o2"], valid, group_ids)
            fs_group = aggregate_groups(fs_aligned["profiles"], valid, group_ids)
            group_valid = np.isfinite(o2_group).all(axis=1) & np.isfinite(
                fs_group
            ).all(axis=1)
            correlations, maes, snmaes = rowwise_profile_metrics(
                o2_group[group_valid],
                fs_group[group_valid],
            )
            regional_rows.append(
                {
                    "subject": subject,
                    "hemisphere": hemi,
                    "n_regions": int(group_valid.sum()),
                    "regional_profile_correlation_median": np.nanmedian(correlations),
                    "regional_profile_mae_median": np.nanmedian(maes),
                    "regional_profile_snmae_median": np.nanmedian(snmaes),
                }
            )
    depth_table = pd.DataFrame(depth_rows)
    regional_table = pd.DataFrame(regional_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_table.to_csv(output_dir / "o2fs_vs_o2column_depth_metrics.csv", index=False)
    regional_table.to_csv(
        output_dir / "o2fs_vs_o2column_regional_metrics.csv", index=False
    )
    summary = depth_table.groupby("depth", as_index=False).agg(
        pearson_r_median=("pearson_r", "median"),
        pearson_r_q25=("pearson_r", lambda value: value.quantile(0.25)),
        pearson_r_q75=("pearson_r", lambda value: value.quantile(0.75)),
        snmae_pct_median=("snmae_pct", "median"),
        snmae_pct_q25=("snmae_pct", lambda value: value.quantile(0.25)),
        snmae_pct_q75=("snmae_pct", lambda value: value.quantile(0.75)),
    )
    summary.to_csv(output_dir / "o2fs_vs_o2column_metrics.csv", index=False)
    return depth_table, regional_table


def save_difference_surface_plot(
    paired: dict,
    subjects: list[str],
    template: str,
    output_dir: Path,
) -> None:
    differences = {}
    limits = []
    for hemi in HEMISPHERES:
        values = np.stack(
            [
                np.where(
                    paired[(subject, hemi)]["common"],
                    paired[(subject, hemi)]["o3"][:, 10]
                    - paired[(subject, hemi)]["o2"][:, 10],
                    np.nan,
                )
                for subject in subjects
            ]
        )
        difference = np.nanmedian(values, axis=0)
        differences[hemi] = difference
        limits.extend(np.abs(difference[np.isfinite(difference)]).tolist())
    limit = float(np.percentile(limits, 98)) if limits else 1.0
    limit = max(limit, np.finfo(float).eps)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, hemi in zip(axes, HEMISPHERES):
        coordinates, faces = read_template_mesh(template, hemi)
        triangulation = mtri.Triangulation(
            coordinates[:, 0],
            coordinates[:, 1],
            triangles=faces,
        )
        image = ax.tripcolor(
            triangulation,
            differences[hemi],
            shading="flat",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_title(hemi)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.colorbar(image, ax=axes, label="O3-Coordinate - O2-Column")
    fig.suptitle("Spatial Distribution of Interpolation-Order Differences at 50% Cortical Depth")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "mid_depth_difference_map.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "mid_depth_difference_map.pdf", bbox_inches="tight")
    plt.close(fig)


def write_group_manifest(
    memberships: dict[str, np.ndarray],
    template: str,
    output_path: Path,
) -> pd.DataFrame:
    rows = []
    for key, group_ids in memberships.items():
        hemi, group_label = key.split("_", 1)
        for group_id in sorted(set(group_ids.tolist()) - {-1}):
            rows.append(
                {
                    "hemisphere": hemi,
                    "group_size": group_label.replace("size_", "")
                    .replace("regional_mean", "regional_mean"),
                    "group_id": group_id,
                    "n_vertices": int(np.sum(group_ids == group_id)),
                    "template": template,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def write_summary(
    output_root: Path,
    validation: pd.DataFrame,
    vertex_metrics: pd.DataFrame,
    group_metrics: pd.DataFrame,
    spatial_metrics: pd.DataFrame,
) -> None:
    depth = pd.read_csv(output_root / "metrics/cohort_depth_summary.csv")
    cross = pd.read_csv(output_root / "metrics/cross_subject_metrics.csv")
    ratio = pd.read_csv(output_root / "metrics/method_difference_to_subject_variation.csv")
    mid = depth.iloc[(depth.depth - 0.5).abs().argmin()]
    cross_mid = cross.iloc[(cross.depth - 0.5).abs().argmin()]
    ratio_mid = ratio.iloc[(ratio.depth - 0.5).abs().argmin()]
    rows = [
        {
            "analysis_level": "QC",
            "main_metric": "Common valid percentage",
            "cohort_result": validation.common_valid_pct.mean(),
            "interpretation": "Percentage of template vertices eligible for both methods",
        },
        {
            "analysis_level": "Individual column",
            "main_metric": "Profile correlation",
            "cohort_result": vertex_metrics.profile_correlation_median.median(),
            "interpretation": "Agreement across the 21-depth profile",
        },
        {
            "analysis_level": "Individual column",
            "main_metric": "Symmetric NMAE at 50% depth",
            "cohort_result": mid.snmae_pct_median,
            "interpretation": "Method difference relative to typical MD magnitude",
        },
        {
            "analysis_level": "Cross-subject",
            "main_metric": "Cross-subject correlation at 50% depth",
            "cohort_result": cross_mid.correlation_median,
            "interpretation": "Preservation of subject ordering at matched vertices",
        },
        {
            "analysis_level": "Cross-subject",
            "main_metric": "Difference / subject variation at 50% depth",
            "cohort_result": ratio_mid.ratio_median,
            "interpretation": "Method difference relative to between-subject variation",
        },
        {
            "analysis_level": "Spatial groups",
            "main_metric": "Symmetric NMAE by group size",
            "cohort_result": "See spatial_group_subject_metrics.csv",
            "interpretation": "Whether method differences decrease with averaging scale",
        },
        {
            "analysis_level": "Spatial structure",
            "main_metric": "Nearby minus distant profile correlation",
            "cohort_result": spatial_metrics.nearby_minus_distant_o2.median(),
            "interpretation": "Whether O2-Column retains local spatial relationships",
        },
    ]
    pd.DataFrame(rows).to_csv(
        output_root / "summary/interpolation_order_summary.csv", index=False
    )


def run(args: argparse.Namespace) -> None:
    output_root = args.output_root
    metrics_root = output_root / "metrics"
    qc_root = output_root / "qc"
    groups_root = output_root / "groups"
    figures_root = output_root / "figures"
    supplementary_root = output_root / "supplementary"
    summary_root = output_root / "summary"
    for directory in (
        metrics_root,
        qc_root,
        groups_root,
        figures_root,
        supplementary_root,
        summary_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    subjects = read_manifest(args.manifest)
    paired, validation, _ = load_paired_profiles(
        subjects,
        args.o2_root,
        args.o3_root,
        args.template,
    )
    validation.to_csv(qc_root / "paired_subject_manifest.csv", index=False)
    validation.to_csv(qc_root / "input_validation_summary.csv", index=False)
    vertex_metrics = save_vertex_profile_metrics(paired, metrics_root)
    memberships = get_group_memberships(
        groups_root / "fixed_group_membership_python.npz",
        groups_root / "fixed_group_membership.npz",
        args.template,
        tuple(args.group_sizes),
    )
    np.savez_compressed(groups_root / "fixed_group_membership.npz", **memberships)
    write_group_manifest(
        memberships,
        args.template,
        groups_root / "group_manifest.csv",
    )
    spatial_metrics = supplementary_spatial_metrics(
        paired,
        args.template,
        metrics_root / "spatial_structure_metrics.csv",
    )
    o2fs_comparison(
        paired,
        subjects,
        args.o2fs_root,
        args.template,
        supplementary_root,
    )
    save_difference_surface_plot(paired, subjects, args.template, figures_root)
    group_metrics = pd.read_csv(metrics_root / "spatial_group_subject_metrics.csv")
    group_cross_metrics = pd.read_csv(
        metrics_root / "spatial_group_cross_subject_metrics.csv"
    )
    depth_metrics = pd.read_csv(
        metrics_root / "subject_hemisphere_depth_metrics.csv"
    )
    depth_metrics.to_csv(metrics_root / "within_subject_depth_metrics.csv", index=False)
    group_metrics.to_csv(metrics_root / "group_size_metrics.csv", index=False)
    group_cross_metrics.to_csv(
        metrics_root / "group_size_cross_subject_metrics.csv",
        index=False,
    )
    write_summary(output_root, validation, vertex_metrics, group_metrics, spatial_metrics)
    LOGGER.info("COMPLETED supplementary interpolation-order analyses")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--o2-root", type=Path, required=True)
    parser.add_argument("--o3-root", type=Path, required=True)
    parser.add_argument("--o2fs-root", type=Path, required=True)
    parser.add_argument("--template", choices=["fsaverage", "fsaverage6"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[1, 50, 100, 200, 400])
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(parse_args())
