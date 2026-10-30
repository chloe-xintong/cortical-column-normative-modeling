#!/usr/bin/env python3
"""Aggregate and validate a surface-PCA null pilot, then render Figure 4B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def group_at_n_realizations(curves: pd.DataFrame, number: int) -> pd.DataFrame:
    part = curves.loc[curves.replicate < number]
    subject = (
        part.groupby(["subject_id", "distance_bin_index", "distance_bin"], observed=True)
        .agg(
            real_dissimilarity=("real_dissimilarity", "first"),
            null_dissimilarity=("null_dissimilarity", "median"),
            null_minus_real=("null_minus_real", "median"),
            anchor_count=("anchor_count", "first"),
            pair_count=("pair_count", "first"),
        ).reset_index()
    )
    rows = []
    for (bin_index, bin_label), table in subject.groupby(["distance_bin_index", "distance_bin"]):
        row = {"n_realizations": number, "distance_bin_index": bin_index, "distance_bin": bin_label,
            "n_subjects": table.subject_id.nunique(), "anchor_count_total": int(table.anchor_count.sum()),
            "pair_count_total": int(table.pair_count.sum())}
        for source, prefix in (("real_dissimilarity", "real"), ("null_dissimilarity", "null"), ("null_minus_real", "delta")):
            values = table[source]
            row[f"{prefix}_median"] = values.median(); row[f"{prefix}_q25"] = values.quantile(.25); row[f"{prefix}_q75"] = values.quantile(.75)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = pd.read_csv(args.manifest, dtype={"subject_id": str}).subject_id
    curves = []; distributions = []; covariances = []; acfs = []; spins = []; validation_rows = []
    for subject_id in subjects:
        root = args.results_root / "subjects" / subject_id
        required = [root / "completion.json", root / "figure4_surface_pca_null_subject_bins.csv"]
        if not all(path.is_file() for path in required):
            continue
        curve = pd.read_csv(required[1]); distribution = pd.read_csv(root / "figure4_surface_pca_depth_distribution.csv")
        covariance = pd.read_csv(root / "figure4_surface_pca_depth_covariance.csv")
        acf = pd.read_csv(root / "figure4_surface_pca_spatial_acf.csv"); spin = pd.read_csv(root / "figure4_surface_pca_spin_qc.csv")
        curves.append(curve); distributions.append(distribution); covariances.append(covariance); acfs.append(acf); spins.append(spin)
        real_cov = covariance.loc[covariance.data_type.eq("real")].set_index(["depth_i", "depth_j"]).correlation
        null_cov = covariance.loc[covariance.data_type.eq("surrogate")].groupby(["depth_i", "depth_j"]).correlation.mean()
        real_acf = acf.loc[acf.data_type.eq("real")].set_index(["hemisphere", "depth_index", "distance_bin_index"]).spatial_acf
        null_acf = acf.loc[acf.data_type.eq("surrogate")].groupby(["hemisphere", "depth_index", "distance_bin_index"]).spatial_acf.median()
        cov_difference = null_cov - real_cov; acf_difference = null_acf - real_acf
        real_distribution = distribution.loc[
            distribution.data_type.eq("real")
        ].set_index("depth_index")
        null_distribution = (
            distribution.loc[distribution.data_type.eq("surrogate")]
            .groupby("depth_index")[["median", "sd", "q05", "q95"]].mean()
        )
        distribution_difference = null_distribution - real_distribution[
            ["median", "sd", "q05", "q95"]
        ]
        targets = json.loads((root / "figure4_surface_pca_targets.json").read_text())
        validation_rows.append({"subject_id": subject_id, "n_realizations": curve.replicate.nunique(),
            "n_components": targets["n_components"], "count_mismatch_rows": int(((curve.anchor_count != curve.real_anchor_count) | (curve.pair_count != curve.real_pair_count)).sum()),
            "all_curve_values_finite": bool(np.isfinite(curve.select_dtypes("number")).all().all()),
            "rotation_mode": targets.get("rotation_mode", "shared"),
            "diffusion_strength": float(targets.get("diffusion_strength", 0.0)),
            "distribution_median_mae": float(distribution_difference["median"].abs().mean()),
            "distribution_sd_mae": float(distribution_difference["sd"].abs().mean()),
            "distribution_q05_mae": float(distribution_difference["q05"].abs().mean()),
            "distribution_q95_mae": float(distribution_difference["q95"].abs().mean()),
            "covariance_mae": float(cov_difference.abs().mean()), "covariance_max_abs_error": float(cov_difference.abs().max()),
            "acf_mae": float(acf_difference.abs().mean()), "acf_bias": float(acf_difference.mean()),
            "acf_q95_abs_error": float(acf_difference.abs().quantile(.95)),
            "unique_source_fraction_min": float(spin.unique_source_fraction.min()),
            "mask_repair_fraction_median": float(spin.mask_repair_fraction.median())})
    if not curves:
        raise RuntimeError("No completed surface PCA null subjects")
    curves = pd.concat(curves, ignore_index=True); distributions = pd.concat(distributions, ignore_index=True)
    covariances = pd.concat(covariances, ignore_index=True); acfs = pd.concat(acfs, ignore_index=True); spins = pd.concat(spins, ignore_index=True)
    curves.to_csv(args.output_dir / "figure4B_subject_realization_bins.csv", index=False)
    distributions.to_csv(args.output_dir / "figure4B_depth_distribution_validation.csv", index=False)
    covariances.to_csv(args.output_dir / "figure4B_depth_covariance_validation.csv", index=False)
    acfs.to_csv(args.output_dir / "figure4B_spatial_acf_validation.csv", index=False)
    spins.to_csv(args.output_dir / "figure4B_spin_mapping_validation.csv", index=False)
    validation = pd.DataFrame(validation_rows); validation.to_csv(args.output_dir / "figure4B_validation_by_subject.csv", index=False)
    available = int(curves.groupby("subject_id").replicate.nunique().min())
    checkpoints = sorted(set([value for value in (5, 10, 20, 50, 100) if value <= available] + [available]))
    group = pd.concat([group_at_n_realizations(curves, number) for number in checkpoints], ignore_index=True)
    group.to_csv(args.output_dir / "figure4B_group_summary_by_realization_count.csv", index=False)
    convergence_rows = []
    for first, second in zip(checkpoints[:-1], checkpoints[1:]):
        left = group.loc[group.n_realizations.eq(first)].set_index("distance_bin_index")
        right = group.loc[group.n_realizations.eq(second)].set_index("distance_bin_index")
        for metric in ("null_median", "delta_median", "null_q25", "null_q75"):
            difference = (right[metric] - left[metric]).abs()
            convergence_rows.append({
                "from_realizations": first, "to_realizations": second,
                "metric": metric, "median_absolute_change": float(difference.median()),
                "maximum_absolute_change": float(difference.max()),
            })
    convergence = pd.DataFrame(convergence_rows, columns=[
        "from_realizations", "to_realizations", "metric",
        "median_absolute_change", "maximum_absolute_change",
    ])
    convergence.to_csv(args.output_dir / "figure4B_realization_convergence.csv", index=False)

    final = group.loc[group.n_realizations.eq(available)].sort_values("distance_bin_index")
    x = final.distance_bin_index.to_numpy(dtype=float) * 2 + 1
    repo_root = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(repo_root))
    from qc.publication_style import FIGURE_WIDTH_IN, PALETTE, save_publication_figure, set_publication_style, style_axes
    set_publication_style(); fig, (top, bottom) = plt.subplots(2, 1, figsize=(FIGURE_WIDTH_IN, 5.3), sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": .12})
    top.plot(x, final.real_median, color=PALETTE["blue"], label="Real", linewidth=1.4)
    top.fill_between(x, final.real_q25, final.real_q75, color=PALETTE["blue"], alpha=.2, linewidth=0)
    top.plot(x, final.null_median, color=PALETTE["orange"], label="Smoothness-matched surrogate", linewidth=1.4)
    top.fill_between(x, final.null_q25, final.null_q75, color=PALETTE["orange"], alpha=.2, linewidth=0)
    top.set_ylabel("Profile dissimilarity (1 − r)"); top.legend(frameon=False); style_axes(top)
    bottom.plot(x, final.delta_median, color=PALETTE["blue"], linewidth=1.4)
    bottom.fill_between(x, final.delta_q25, final.delta_q75, color=PALETTE["blue"], alpha=.2, linewidth=0)
    bottom.axhline(0, color=PALETTE["reference"], linewidth=.8, linestyle="--")
    bottom.set(xlabel="White-surface geodesic distance (mm)", ylabel="Null − real\ndissimilarity"); style_axes(bottom)
    save_publication_figure(fig, args.output_dir / "Figure4B_surface_pca_null_pilot")
    plt.close(fig)
    rotation_modes = sorted(validation.rotation_mode.unique().tolist())
    validation_status = (
        "preliminary_spectral_surrogate_pending"
        if rotation_modes == ["independent_components"]
        else "invalid_shared_rotation_diagnostic_only"
    )
    report = {
        "subjects_expected": int(subjects.nunique()),
        "subjects_completed": int(validation.subject_id.nunique()),
        "realizations_per_subject": available,
        "diffusion_strength": float(validation.diffusion_strength.median()),
        "rotation_modes": rotation_modes,
        "count_mismatch_rows": int(validation.count_mismatch_rows.sum()),
        "all_curve_values_finite": bool(validation.all_curve_values_finite.all()),
        "unique_source_fraction_min": float(validation.unique_source_fraction_min.min()),
        "covariance_mae_mean": float(validation.covariance_mae.mean()),
        "acf_mae_mean": float(validation.acf_mae.mean()),
        "acf_bias_mean": float(validation.acf_bias.mean()),
        "acf_q95_abs_error_mean": float(validation.acf_q95_abs_error.mean()),
        "status": validation_status,
    }
    (args.output_dir / "figure4B_validation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"Validated {validation.subject_id.nunique()} subjects with {available} realizations")


if __name__ == "__main__":
    main()
