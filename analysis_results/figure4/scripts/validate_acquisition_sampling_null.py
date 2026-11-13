#!/usr/bin/env python3
"""Validate and plot the predefined Figure 4B acquisition-sampling PSF family.

Description
-----------
Validate and plot the predefined Figure 4B acquisition-sampling PSF family.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--condition`` command-line input
- ``--manifest`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/validate_acquisition_sampling_null.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261020


def parse_condition(value: str) -> tuple[float, Path]:
    fwhm, root = value.split("=", 1)
    return float(fwhm), Path(root)


def summarize_at_n(curves: pd.DataFrame, number: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    part = curves.loc[curves.replicate < number]
    subject = (
        part.groupby(
            ["psf_fwhm_mm", "subject_id", "distance_bin_index", "distance_bin"],
            observed=True,
        )
        .agg(
            real_dissimilarity=("real_dissimilarity", "first"),
            null_dissimilarity=("null_dissimilarity", "median"),
            null_minus_real=("null_minus_real", "median"),
            real_local_excess_similarity=("real_local_excess_similarity", "first"),
            null_local_excess_similarity=("null_local_excess_similarity", "median"),
            real_minus_null_local_excess=("real_minus_null_local_excess", "median"),
            anchor_count=("anchor_count", "first"),
            pair_count=("pair_count", "first"),
        )
        .reset_index()
    )
    rows = []
    metrics = (
        "real_dissimilarity", "null_dissimilarity", "null_minus_real",
        "real_local_excess_similarity", "null_local_excess_similarity",
        "real_minus_null_local_excess",
    )
    for keys, table in subject.groupby(
        ["psf_fwhm_mm", "distance_bin_index", "distance_bin"], observed=True
    ):
        fwhm, bin_index, bin_label = keys
        row = {
            "n_realizations": number, "psf_fwhm_mm": fwhm,
            "distance_bin_index": bin_index, "distance_bin": bin_label,
            "n_subjects": table.subject_id.nunique(),
            "anchor_count_total": int(table.anchor_count.sum()),
            "pair_count_total": int(table.pair_count.sum()),
        }
        for metric in metrics:
            values = table[metric]
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_q25"] = values.quantile(.25)
            row[f"{metric}_q75"] = values.quantile(.75)
        rows.append(row)
    return subject, pd.DataFrame(rows)


def bootstrap_median_ci(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap CI with subjects as the resampling unit."""
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    medians = np.median(values[indices], axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]))


def summarize_local_excess_auc(subject_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the real-minus-null local-excess residual over 0--10 mm."""
    near = subject_summary.loc[subject_summary.distance_bin_index.between(0, 4)].copy()
    counts = near.groupby(["psf_fwhm_mm", "subject_id"]).size()
    if not counts.eq(5).all():
        raise ValueError("Every subject/PSF must contribute all five 0--10-mm bins")
    per_subject = (
        near.groupby(["psf_fwhm_mm", "subject_id"], observed=True)
        .real_minus_null_local_excess
        .agg(mean_residual_local_excess_0_10="mean", sum_residual="sum")
        .reset_index()
    )
    per_subject["auc_residual_local_excess_0_10_mm"] = (
        2.0 * per_subject.pop("sum_residual")
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for fwhm, table in per_subject.groupby("psf_fwhm_mm", observed=True):
        row = {
            "psf_fwhm_mm": fwhm,
            "n_subjects": table.subject_id.nunique(),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        }
        for metric in (
            "auc_residual_local_excess_0_10_mm",
            "mean_residual_local_excess_0_10",
        ):
            values = table[metric].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_median_ci(values, rng)
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            row[f"{metric}_q75"] = float(np.quantile(values, 0.75))
            row[f"{metric}_bootstrap_ci_low"] = ci_low
            row[f"{metric}_bootstrap_ci_high"] = ci_high
            row[f"{metric}_positive_fraction"] = float(np.mean(values > 0))
        rows.append(row)
    return per_subject, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected = pd.read_csv(args.manifest, dtype={"subject_id": str}).subject_id.tolist()
    curves = []
    validation_rows = []
    distribution_rows = []
    covariance_rows = []
    for condition in args.condition:
        fwhm, root = parse_condition(condition)
        for subject_id in expected:
            subject_dir = root / "subjects" / subject_id
            completion_path = subject_dir / "completion.json"
            if not completion_path.exists():
                continue
            completion = json.loads(completion_path.read_text())
            curve = pd.read_csv(subject_dir / "figure4_null_subject_bins.csv")
            if not np.allclose(curve["psf_fwhm_mm"], fwhm):
                raise ValueError(
                    f"PSF label mismatch for {subject_id}: expected {fwhm:g} mm"
                )
            distribution = pd.read_csv(
                subject_dir / "figure4_null_depth_distribution.csv",
                keep_default_na=False,
            )
            covariance = pd.read_csv(
                subject_dir / "figure4_null_depth_covariance.csv",
                keep_default_na=False,
            )
            mask = pd.read_csv(subject_dir / "figure4_null_mask_qc.csv")
            curves.append(curve)
            distribution_rows.append(distribution.assign(psf_fwhm_mm=fwhm))
            covariance_rows.append(covariance.assign(psf_fwhm_mm=fwhm))
            real_distribution = distribution.loc[
                distribution.data_type.eq("real")
            ].set_index("depth_index")
            null_distribution = (
                distribution.loc[distribution.data_type.eq("null")]
                .groupby("depth_index")[["mean", "median", "sd", "q05", "q95"]]
                .mean()
            )
            distribution_difference = null_distribution - real_distribution[
                ["mean", "median", "sd", "q05", "q95"]
            ]
            real_covariance = covariance.loc[
                covariance.data_type.eq("real")
            ].set_index(["depth_i", "depth_j"]).covariance
            null_covariance = (
                covariance.loc[covariance.data_type.eq("null")]
                .groupby(["depth_i", "depth_j"]).covariance.mean()
            )
            covariance_difference = null_covariance - real_covariance
            validation_rows.append({
                "subject_id": subject_id, "psf_fwhm_mm": fwhm,
                "n_realizations": curve.replicate.nunique(),
                "null_method": completion["null_method"],
                "laminar_recolored": bool(completion["laminar_recolored"]),
                "volume_shape": json.dumps(completion["volume_shape"]),
                "volume_affine": json.dumps(completion["volume_affine"]),
                "voxel_sizes_mm": json.dumps(completion["voxel_sizes_mm"]),
                "random_field_support": completion["random_field_support"],
                "sampling_coordinates": completion["sampling_coordinates"],
                "interpolation_order": completion["interpolation_order"],
                # Early pilot jobs predate explicit seed fields but used this
                # fixed command-line seed and the documented subject offset.
                "base_seed": completion.get("base_seed", 20261020),
                "subject_seed": completion.get(
                    "subject_seed",
                    20261020 + sum(ord(character) for character in subject_id),
                ),
                "mask_identical": bool(mask.real_null_mask_identical.all()),
                "count_mismatch_rows": int((
                    (curve.anchor_count != curve.real_anchor_count)
                    | (curve.pair_count != curve.real_pair_count)
                ).sum()),
                "all_curve_values_finite": bool(np.isfinite(curve.select_dtypes("number")).all().all()),
                "depth_mean_mae": float(distribution_difference["mean"].abs().mean()),
                "depth_median_mae": float(distribution_difference["median"].abs().mean()),
                "depth_sd_mae": float(distribution_difference["sd"].abs().mean()),
                "depth_q05_mae": float(distribution_difference["q05"].abs().mean()),
                "depth_q95_mae": float(distribution_difference["q95"].abs().mean()),
                "covariance_mae": float(covariance_difference.abs().mean()),
                "covariance_max_abs_error": float(covariance_difference.abs().max()),
            })
    if not curves:
        raise RuntimeError("No completed acquisition-sampling null results")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.concat(curves, ignore_index=True)
    validation = pd.DataFrame(validation_rows)
    distributions = pd.concat(distribution_rows, ignore_index=True)
    covariances = pd.concat(covariance_rows, ignore_index=True)
    if not validation.null_method.eq("acquisition_sampling").all():
        raise ValueError("A non-acquisition-sampling result was supplied")
    if not validation.laminar_recolored.all():
        raise ValueError("Laminar recoloring is required for the primary null")
    if validation.groupby("subject_id").subject_seed.nunique().ne(1).any():
        raise ValueError("A subject has inconsistent subject seeds")
    subject_seeds = validation[["subject_id", "subject_seed"]].drop_duplicates()
    subject_seed_collision_count = int(subject_seeds.subject_seed.duplicated().sum())
    if subject_seed_collision_count:
        raise ValueError(
            f"Detected {subject_seed_collision_count} subject-seed collisions"
        )
    if "replicate_seed" not in curves.columns:
        raise ValueError("Formal null output must record replicate_seed")
    replicate_seeds = curves[[
        "subject_id", "replicate", "replicate_seed"
    ]].drop_duplicates()
    expected_seed_rows = curves[["subject_id", "replicate"]].drop_duplicates()
    if len(replicate_seeds) != len(expected_seed_rows):
        raise ValueError("A subject/realization has inconsistent replicate seeds")
    replicate_seed_collision_count = int(
        replicate_seeds.replicate_seed.duplicated().sum()
    )
    if replicate_seed_collision_count:
        raise ValueError(
            f"Detected {replicate_seed_collision_count} subject-realization seed collisions"
        )
    for column in (
        "volume_shape", "volume_affine", "voxel_sizes_mm",
        "random_field_support", "sampling_coordinates", "interpolation_order",
        "base_seed", "subject_seed",
    ):
        counts = validation.groupby("subject_id")[column].nunique()
        if not counts.eq(1).all():
            raise ValueError(f"DWI geometry/sampling mismatch across PSF conditions: {column}")
    real_key = ["subject_id", "distance_bin_index", "distance_bin"]
    real_curve_consistent = bool(
        curves.groupby(real_key).real_dissimilarity.nunique().eq(1).all()
    )
    count_columns = [
        "real_anchor_count", "real_pair_count", "anchor_count", "pair_count"
    ]
    pair_tables_consistent = all(
        curves.groupby(real_key)[column].nunique().eq(1).all()
        for column in count_columns
    )
    if not real_curve_consistent or not pair_tables_consistent:
        raise ValueError("Real curve or anchor/pair table differs across PSF conditions")
    curves.to_csv(args.output_dir / "figure4B_acquisition_subject_realization_bins.csv", index=False)
    validation.to_csv(args.output_dir / "figure4B_acquisition_validation_by_subject.csv", index=False)
    distributions.to_csv(args.output_dir / "figure4B_acquisition_depth_distribution.csv", index=False)
    covariances.to_csv(args.output_dir / "figure4B_acquisition_depth_covariance.csv", index=False)

    available = int(curves.groupby(["psf_fwhm_mm", "subject_id"]).replicate.nunique().min())
    checkpoints = sorted(set(
        [value for value in (5, 10, 20, 50, 100) if value <= available] + [available]
    ))
    subject_tables = []
    group_tables = []
    for number in checkpoints:
        subject, group = summarize_at_n(curves, number)
        subject.insert(0, "n_realizations", number)
        subject_tables.append(subject)
        group_tables.append(group)
    subject_summary = pd.concat(subject_tables, ignore_index=True)
    group_summary = pd.concat(group_tables, ignore_index=True)
    subject_summary.to_csv(args.output_dir / "figure4B_acquisition_subject_summary.csv", index=False)
    group_summary.to_csv(args.output_dir / "figure4B_acquisition_group_summary.csv", index=False)
    final_subject = subject_summary.loc[
        subject_summary.n_realizations.eq(available)
    ].copy()
    auc_subject, auc_group = summarize_local_excess_auc(final_subject)
    auc_subject.to_csv(
        args.output_dir / "figure4B_local_excess_auc_subject.csv", index=False
    )
    auc_group.to_csv(
        args.output_dir / "figure4B_local_excess_auc_group_summary.csv", index=False
    )

    # Sensitivity of plateau adjustment to prespecified far-distance windows.
    far_windows = {
        "10-20_mm": (5, 9),
        "16-30_mm": (8, 14),
        "20-30_mm_primary": (10, 14),
        "24-30_mm": (12, 14),
    }
    far_subject_rows = []
    for (fwhm, subject_id), table in final_subject.groupby(
        ["psf_fwhm_mm", "subject_id"], observed=True
    ):
        table = table.sort_values("distance_bin_index").copy()
        for window, (first_bin, last_bin) in far_windows.items():
            far = table.distance_bin_index.between(first_bin, last_bin)
            real_far = float(table.loc[far, "real_dissimilarity"].median())
            null_far = float(table.loc[far, "null_dissimilarity"].median())
            residual = (
                real_far - table["real_dissimilarity"]
                - (null_far - table["null_dissimilarity"])
            )
            for (_, current), value in zip(table.iterrows(), residual):
                far_subject_rows.append({
                    "psf_fwhm_mm": fwhm, "subject_id": subject_id,
                    "far_window": window,
                    "distance_bin_index": int(current.distance_bin_index),
                    "distance_bin": current.distance_bin,
                    "real_far_dissimilarity": real_far,
                    "null_far_dissimilarity": null_far,
                    "real_minus_null_local_excess": float(value),
                })
    far_subject = pd.DataFrame(far_subject_rows)
    far_subject.to_csv(
        args.output_dir / "figure4B_far_window_sensitivity_subject_bins.csv",
        index=False,
    )
    far_auc = (
        far_subject.loc[far_subject.distance_bin_index.between(0, 4)]
        .groupby(["psf_fwhm_mm", "subject_id", "far_window"], observed=True)
        .real_minus_null_local_excess.sum().mul(2.0)
        .rename("auc_residual_local_excess_0_10_mm").reset_index()
    )
    far_auc.to_csv(
        args.output_dir / "figure4B_far_window_sensitivity_auc_subject.csv",
        index=False,
    )
    far_auc_group = (
        far_auc.groupby(["psf_fwhm_mm", "far_window"], observed=True)
        .auc_residual_local_excess_0_10_mm.agg(
            n_subjects="count", median="median",
            q25=lambda values: values.quantile(.25),
            q75=lambda values: values.quantile(.75),
            positive_fraction=lambda values: (values > 0).mean(),
        ).reset_index()
    )
    far_auc_group.to_csv(
        args.output_dir / "figure4B_far_window_sensitivity_auc_group.csv",
        index=False,
    )

    convergence_rows = []
    for first, second in zip(checkpoints[:-1], checkpoints[1:]):
        for fwhm in sorted(group_summary.psf_fwhm_mm.unique()):
            left = group_summary.loc[
                group_summary.n_realizations.eq(first)
                & group_summary.psf_fwhm_mm.eq(fwhm)
            ].set_index("distance_bin_index")
            right = group_summary.loc[
                group_summary.n_realizations.eq(second)
                & group_summary.psf_fwhm_mm.eq(fwhm)
            ].set_index("distance_bin_index")
            for metric in ("null_dissimilarity_median", "null_minus_real_median"):
                difference = (right[metric] - left[metric]).abs()
                convergence_rows.append({
                    "psf_fwhm_mm": fwhm, "from_realizations": first,
                    "to_realizations": second, "metric": metric,
                    "median_absolute_change": float(difference.median()),
                    "maximum_absolute_change": float(difference.max()),
                })
    convergence = pd.DataFrame(convergence_rows, columns=[
        "psf_fwhm_mm", "from_realizations", "to_realizations", "metric",
        "median_absolute_change", "maximum_absolute_change",
    ])
    convergence.to_csv(
        args.output_dir / "figure4B_acquisition_convergence.csv", index=False
    )

    replicate_group = (
        curves.groupby(
            ["psf_fwhm_mm", "replicate", "distance_bin_index", "distance_bin"],
            observed=True,
        ).null_dissimilarity.median().reset_index()
    )
    envelope = (
        replicate_group.groupby(
            ["psf_fwhm_mm", "distance_bin_index", "distance_bin"], observed=True
        ).null_dissimilarity.agg(
            realization_q05=lambda values: values.quantile(.05),
            realization_q95=lambda values: values.quantile(.95),
        ).reset_index()
    )
    envelope.to_csv(args.output_dir / "figure4B_acquisition_realization_envelope.csv", index=False)

    final = group_summary.loc[group_summary.n_realizations.eq(available)].copy()
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from qc.publication_style import (
        FIGURE_WIDTH_IN, PALETTE, save_publication_figure,
        set_publication_style, style_axes,
    )
    set_publication_style()
    colors = plt.get_cmap("cividis")(np.linspace(.15, .85, final.psf_fwhm_mm.nunique()))
    color_by_fwhm = dict(zip(sorted(final.psf_fwhm_mm.unique()), colors))
    line_styles = ("-", "--", "-.", ":")
    style_by_fwhm = dict(zip(sorted(final.psf_fwhm_mm.unique()), line_styles))
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(FIGURE_WIDTH_IN, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": .12},
    )
    real = final.loc[final.psf_fwhm_mm.eq(final.psf_fwhm_mm.min())].sort_values("distance_bin_index")
    x = real.distance_bin_index.to_numpy(dtype=float) * 2 + 1
    top.plot(x, real.real_dissimilarity_median, color=PALETTE["blue"], linewidth=1.5, label="Real")
    top.fill_between(x, real.real_dissimilarity_q25, real.real_dissimilarity_q75,
                     color=PALETTE["blue"], alpha=.2, linewidth=0)
    for fwhm, table in final.groupby("psf_fwhm_mm"):
        table = table.sort_values("distance_bin_index")
        current_envelope = envelope.loc[envelope.psf_fwhm_mm.eq(fwhm)].sort_values("distance_bin_index")
        color = color_by_fwhm[fwhm]
        line_style = style_by_fwhm[fwhm]
        top.plot(x, table.null_dissimilarity_median, color=color,
                 linestyle=line_style, linewidth=1.3,
                 label=f"Null, FWHM {fwhm:g} mm")
        top.fill_between(x, current_envelope.realization_q05,
                         current_envelope.realization_q95, color=color,
                         alpha=.12, linewidth=0)
        bottom.plot(x, table.null_minus_real_median, color=color,
                    linestyle=line_style, linewidth=1.3,
                    label=f"{fwhm:g} mm")
        bottom.fill_between(x, table.null_minus_real_q25, table.null_minus_real_q75,
                            color=color, alpha=.12, linewidth=0)
    top.set_ylabel("Profile dissimilarity (1 − r)")
    top.legend(frameon=False, ncol=2, loc="best")
    style_axes(top)
    bottom.axhline(0, color=PALETTE["reference"], linewidth=.8, linestyle="--")
    bottom.set(xlabel="White-surface geodesic distance (mm)",
               ylabel="Null − real\ndissimilarity")
    style_axes(bottom)
    save_publication_figure(
        fig, args.output_dir / "Figure4B_absolute_dissimilarity_supplement"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.3))
    for fwhm, table in final.groupby("psf_fwhm_mm"):
        table = table.sort_values("distance_bin_index")
        color = color_by_fwhm[fwhm]
        ax.plot(x, table.real_minus_null_local_excess_median, color=color,
                linestyle=style_by_fwhm[fwhm], linewidth=1.3,
                label=f"FWHM {fwhm:g} mm")
        ax.fill_between(x, table.real_minus_null_local_excess_q25,
                        table.real_minus_null_local_excess_q75,
                        color=color, alpha=.12, linewidth=0)
    ax.axhline(0, color=PALETTE["reference"], linewidth=.8, linestyle="--")
    ax.set(xlabel="White-surface geodesic distance (mm)",
           ylabel="Real − null local excess similarity (Δr)")
    ax.legend(frameon=False, ncol=2)
    style_axes(ax)
    save_publication_figure(fig, args.output_dir / "Figure4B_local_excess_similarity_pilot")
    plt.close(fig)

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.35))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.15, 1.0], wspace=.34)
    curve_ax = fig.add_subplot(grid[0, 0])
    summary_ax = fig.add_subplot(grid[0, 1])
    for fwhm, table in final.groupby("psf_fwhm_mm"):
        table = table.sort_values("distance_bin_index")
        color = color_by_fwhm[fwhm]
        curve_ax.plot(
            x, table.real_minus_null_local_excess_median,
            color=color, linestyle=style_by_fwhm[fwhm], linewidth=1.3,
            label=f"FWHM {fwhm:g} mm",
        )
        curve_ax.fill_between(
            x, table.real_minus_null_local_excess_q25,
            table.real_minus_null_local_excess_q75,
            color=color, alpha=.12, linewidth=0,
        )
    curve_ax.axhline(0, color=PALETTE["reference"], linewidth=.8, linestyle="--")
    curve_ax.set(
        xlabel="White-surface geodesic distance (mm)",
        ylabel="Real − null local excess similarity (Δr)",
    )
    curve_ax.legend(frameon=False, ncol=2, loc="upper right")
    style_axes(curve_ax)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    positions = np.arange(len(color_by_fwhm))
    for position, fwhm in zip(positions, sorted(color_by_fwhm)):
        values = auc_subject.loc[
            auc_subject.psf_fwhm_mm.eq(fwhm),
            "auc_residual_local_excess_0_10_mm",
        ].to_numpy(dtype=float)
        jitter = rng.uniform(-.13, .13, size=len(values))
        summary_ax.scatter(
            np.full(len(values), position) + jitter, values,
            s=8, color=color_by_fwhm[fwhm], alpha=.55, linewidth=0,
        )
        q25, median, q75 = np.quantile(values, [.25, .5, .75])
        summary_ax.vlines(position, q25, q75, color=PALETTE["reference"], linewidth=2.2)
        summary_ax.hlines(
            median, position - .18, position + .18,
            color=PALETTE["reference"], linewidth=1.2,
        )
    summary_ax.axhline(0, color=PALETTE["reference"], linewidth=.8, linestyle="--")
    summary_ax.set(
        xlabel="PSF FWHM (mm)",
        ylabel="Residual local-excess AUC\n(Δr × mm; 0–10 mm)",
        xticks=positions,
        xticklabels=[f"{value:g}" for value in sorted(color_by_fwhm)],
    )
    style_axes(summary_ax)
    save_publication_figure(fig, args.output_dir / "Figure4B_acquisition_sampling_main")
    plt.close(fig)

    subjects_per_condition = validation.groupby("psf_fwhm_mm").subject_id.nunique()
    all_expected_complete = bool(subjects_per_condition.eq(len(expected)).all())
    if all_expected_complete and len(expected) >= 300 and available >= 100:
        status = "full_cohort_complete"
    elif all_expected_complete and len(expected) >= 20 and available >= 100:
        status = "preliminary_pilot_2_convergence_complete"
    elif all_expected_complete and len(expected) >= 20 and available >= 20:
        status = "preliminary_pilot_2_complete"
    elif all_expected_complete:
        status = "preliminary_pilot_1_complete"
    else:
        status = "preliminary_incomplete"
    target_convergence = convergence.loc[
        convergence.from_realizations.eq(50)
        & convergence.to_realizations.eq(100)
    ]
    convergence_maximum = (
        None if target_convergence.empty
        else float(target_convergence.maximum_absolute_change.max())
    )
    convergence_threshold = 1.0e-3
    convergence_pass = (
        None if convergence_maximum is None
        else bool(convergence_maximum <= convergence_threshold)
    )
    report = {
        "subjects_expected": len(expected),
        "subjects_completed_per_condition": subjects_per_condition.to_dict(),
        "all_expected_subjects_complete": all_expected_complete,
        "realizations_per_subject": available,
        "psf_fwhm_mm": sorted(validation.psf_fwhm_mm.unique().tolist()),
        "count_mismatch_rows": int(validation.count_mismatch_rows.sum()),
        "all_masks_identical": bool(validation.mask_identical.all()),
        "all_curve_values_finite": bool(validation.all_curve_values_finite.all()),
        "maximum_depth_mean_mae": float(validation.depth_mean_mae.max()),
        "maximum_depth_sd_mae": float(validation.depth_sd_mae.max()),
        "maximum_covariance_mae": float(validation.covariance_mae.max()),
        "maximum_50_to_100_curve_change": convergence_maximum,
        "maximum_50_to_100_curve_change_threshold": convergence_threshold,
        "convergence_50_to_100_pass": convergence_pass,
        "native_dwi_geometry_consistent_across_psf": True,
        "paired_subject_and_realization_seeds_across_psf": True,
        "seed_scheme": "blake2b64:F4AcqNullV2(base_seed,subject_id,replicate)",
        "unique_subject_seeds": int(subject_seeds.subject_seed.nunique()),
        "subject_seed_collision_count": subject_seed_collision_count,
        "unique_subject_realization_seeds": int(
            replicate_seeds.replicate_seed.nunique()
        ),
        "subject_realization_seed_collision_count": replicate_seed_collision_count,
        "far_window_sensitivity_windows": far_windows,
        "real_curve_consistent_across_psf": real_curve_consistent,
        "anchor_and_pair_counts_consistent_across_psf": pair_tables_consistent,
        "random_field_support": sorted(validation.random_field_support.unique().tolist()),
        "sampling_coordinates": sorted(validation.sampling_coordinates.unique().tolist()),
        "status": status,
    }
    (args.output_dir / "figure4B_acquisition_validation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
