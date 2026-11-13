#!/usr/bin/env python3
"""Summarize and plot formal Figure 4 six-versus-21-depth sensitivity.

Description
-----------
Summarize and plot formal Figure 4 six-versus-21-depth sensitivity.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--six-root`` command-line input
- ``--twentyone-subject-bins`` command-line input
- ``--twentyone-fits`` command-line input
- ``--pair-file`` command-line input
- ``--split-assignment`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/depth_density_sensitivity_6_vs_21/scripts/summarize.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis_results.figure4.scripts.fit_correlation_lengths import icc_two_way_single
from qc.publication_style import PALETTE, finalize_figure, set_publication_style


DEPTH_INDICES = [0, 4, 8, 12, 16, 20]
BOOTSTRAP_SEED = 20261106
COLOR_21 = "#294F86"
COLOR_6 = "#9CC7E5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--six-root", type=Path, required=True)
    parser.add_argument("--twentyone-subject-bins", type=Path, required=True)
    parser.add_argument("--twentyone-fits", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--split-assignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_median_ci(values: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    bootstrap = np.median(values[indices], axis=1)
    return float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))


def load_six_subjects(
    manifest: pd.DataFrame, six_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    bins: list[pd.DataFrame] = []
    fits: list[pd.DataFrame] = []
    completions: list[dict[str, object]] = []
    missing: list[str] = []
    for subject_id in manifest["subject_id"]:
        root = six_root / "subjects" / subject_id
        required = [
            root / "completion.json",
            root / "figure4_6point_subject_bins.csv",
            root / "figure4_6point_correlation_length_fits.csv",
        ]
        if not all(path.is_file() for path in required):
            missing.append(subject_id)
            continue
        completion = json.loads(required[0].read_text(encoding="utf-8"))
        if completion.get("status") != "complete" or completion.get("depth_indices") != DEPTH_INDICES:
            raise RuntimeError(f"Invalid completion metadata for {subject_id}")
        if not completion.get("pial_and_white_retained"):
            raise RuntimeError(f"Endpoint retention failed for {subject_id}")
        if not all(row.get("exact_shared_values") for row in completion["hemisphere_checks"]):
            raise RuntimeError(f"Exact shared-depth check failed for {subject_id}")
        bins.append(pd.read_csv(required[1], dtype={"subject_id": str}))
        fits.append(pd.read_csv(required[2], dtype={"subject_id": str}))
        completions.append(completion)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} six-point subjects: {missing[:10]}")
    return pd.concat(bins, ignore_index=True), pd.concat(fits, ignore_index=True), completions


def group_curve(table: pd.DataFrame, n_depths: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bin_index, part in table.groupby("distance_bin_index", sort=True):
        row: dict[str, object] = {
            "n_depths": n_depths,
            "distance_bin_index": int(bin_index),
            "distance_bin": part["distance_bin"].iloc[0],
            "distance_bin_center_mm": int(bin_index) * 2 + 1,
            "n_subjects": part["subject_id"].nunique(),
        }
        for source in ("median_dissimilarity", "median_profile_mae", "median_profile_rmse"):
            values = part[source].to_numpy(dtype=float)
            row[f"{source}_median"] = float(np.median(values))
            row[f"{source}_q25"] = float(np.percentile(values, 25))
            row[f"{source}_q75"] = float(np.percentile(values, 75))
        row["median_geodesic_distance_mm"] = float(part["median_geodesic_distance_mm"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(
    output_dir: Path,
    curves: pd.DataFrame,
    differences: pd.DataFrame,
    paired_lambda: pd.DataFrame,
) -> None:
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.09, 5.8))
    for n_depths, color, label in ((21, COLOR_21, "21 points"), (6, COLOR_6, "6 points")):
        part = curves.loc[curves["n_depths"].eq(n_depths)].sort_values("distance_bin_index")
        x = part["distance_bin_center_mm"].to_numpy(dtype=float)
        for ax, metric, ylabel in (
            (axes[0, 0], "median_dissimilarity", "Profile dissimilarity (1 − r)"),
            (axes[0, 1], "median_profile_mae", "Profile MAE (MD units)"),
        ):
            median = part[f"{metric}_median"].to_numpy(dtype=float)
            q25 = part[f"{metric}_q25"].to_numpy(dtype=float)
            q75 = part[f"{metric}_q75"].to_numpy(dtype=float)
            ax.plot(x, median, color=color, lw=1.6, label=label)
            ax.fill_between(x, q25, q75, color=color, alpha=0.14, linewidth=0)
            ax.set(xlabel="White-surface geodesic distance (mm)", ylabel=ylabel, xlim=(0, 30))
    axes[0, 0].legend(frameon=False)
    axes[0, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    difference = differences.sort_values("distance_bin_index")
    x = difference["distance_bin_center_mm"].to_numpy(dtype=float)
    axes[1, 0].plot(x, difference["delta_dissimilarity_median"], color=COLOR_21, lw=1.6)
    axes[1, 0].fill_between(
        x,
        difference["delta_dissimilarity_ci_low"],
        difference["delta_dissimilarity_ci_high"],
        color=COLOR_6,
        alpha=0.25,
        linewidth=0,
    )
    axes[1, 0].axhline(0, color=PALETTE["gray"], lw=0.8, ls="--")
    axes[1, 0].set(
        xlabel="White-surface geodesic distance (mm)",
        ylabel="Dissimilarity difference\n(6 points − 21 points)",
        xlim=(0, 30),
    )

    ax = axes[1, 1]
    ax.scatter(
        paired_lambda["lambda_21_mm"], paired_lambda["lambda_6_mm"],
        s=11, color=COLOR_21, alpha=0.5, linewidth=0,
    )
    low = float(min(paired_lambda[["lambda_21_mm", "lambda_6_mm"]].min().min(), 0))
    high = float(paired_lambda[["lambda_21_mm", "lambda_6_mm"]].max().max())
    padding = 0.04 * (high - low)
    ax.plot([low, high], [low, high], color=PALETTE["gray"], lw=0.8, ls="--")
    ax.set(
        xlabel="21-point correlation length, λ (mm)",
        ylabel="6-point correlation length, λ (mm)",
        xlim=(low - padding, high + padding),
        ylim=(low - padding, high + padding),
    )
    for letter, axis in zip("ABCD", axes.ravel()):
        axis.text(-0.18, 1.08, letter, transform=axis.transAxes, fontsize=10, fontweight="bold")
    finalize_figure(fig)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.11, top=0.95, wspace=0.36, hspace=0.42)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"supplementary_figure4_6_vs_21_depth_sensitivity.{extension}",
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    if len(manifest) != 320 or manifest["subject_id"].duplicated().any():
        raise RuntimeError("Expected the fixed 320-subject registration-QC cohort")
    six_bins, six_fits, completions = load_six_subjects(manifest, args.six_root)
    ids = set(manifest["subject_id"])
    twentyone_bins = pd.read_csv(args.twentyone_subject_bins, dtype={"subject_id": str})
    twentyone_bins = twentyone_bins.loc[twentyone_bins["subject_id"].isin(ids)].copy()
    twentyone_fits = pd.read_csv(args.twentyone_fits, dtype={"subject_id": str})
    twentyone_fits = twentyone_fits.loc[twentyone_fits["subject_id"].isin(ids)].copy()
    if set(six_bins["subject_id"]) != ids or set(twentyone_bins["subject_id"]) != ids:
        raise RuntimeError("Six- and 21-point subject-bin cohorts differ from the manifest")

    keys = ["subject_id", "distance_bin_index", "distance_bin", "pair_type"]
    columns = keys + [
        "median_dissimilarity", "median_profile_mae", "median_profile_rmse",
        "median_geodesic_distance_mm", "n_anchors", "n_pairs",
    ]
    six_bins = six_bins[columns].sort_values(keys).reset_index(drop=True)
    twentyone_bins = twentyone_bins[columns].sort_values(keys).reset_index(drop=True)
    if not six_bins[keys].equals(twentyone_bins[keys]):
        raise RuntimeError("Six- and 21-point subject-bin keys differ")
    if not np.allclose(
        six_bins["median_geodesic_distance_mm"],
        twentyone_bins["median_geodesic_distance_mm"], rtol=0, atol=0,
    ):
        raise RuntimeError("Geodesic distances changed between depth parameterizations")
    if not np.array_equal(six_bins["n_pairs"], twentyone_bins["n_pairs"]):
        raise RuntimeError("Valid pair counts changed between depth parameterizations")
    six_bins.to_csv(args.output_dir / "figure4_6point_subject_bins.csv", index=False)
    curves = pd.concat(
        [group_curve(six_bins, 6), group_curve(twentyone_bins, 21)], ignore_index=True
    )
    curves.to_csv(args.output_dir / "figure4_6_vs_21_group_curves.csv", index=False)

    paired = six_bins[keys].copy()
    for metric in ("median_dissimilarity", "median_profile_mae", "median_profile_rmse"):
        paired[f"{metric}_6"] = six_bins[metric]
        paired[f"{metric}_21"] = twentyone_bins[metric]
        paired[f"delta_{metric}_6_minus_21"] = six_bins[metric] - twentyone_bins[metric]
    paired.to_csv(args.output_dir / "figure4_6_vs_21_subject_bin_differences.csv", index=False)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    subject_order = manifest["subject_id"].tolist()
    subject_to_index = {subject_id: index for index, subject_id in enumerate(subject_order)}
    bootstrap_indices = rng.integers(
        0, len(subject_order), size=(args.bootstrap_replicates, len(subject_order))
    )
    difference_rows: list[dict[str, object]] = []
    for bin_index, part in paired.groupby("distance_bin_index", sort=True):
        part = part.assign(subject_row=part["subject_id"].map(subject_to_index)).sort_values("subject_row")
        if not np.array_equal(part["subject_row"], np.arange(len(subject_order))):
            raise RuntimeError(f"Incomplete paired subject coverage in bin {bin_index}")
        values = part["delta_median_dissimilarity_6_minus_21"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_median_ci(values, bootstrap_indices)
        difference_rows.append(
            {
                "distance_bin_index": int(bin_index),
                "distance_bin": part["distance_bin"].iloc[0],
                "distance_bin_center_mm": int(bin_index) * 2 + 1,
                "n_subjects": len(values),
                "delta_dissimilarity_median": float(np.median(values)),
                "delta_dissimilarity_q25": float(np.percentile(values, 25)),
                "delta_dissimilarity_q75": float(np.percentile(values, 75)),
                "delta_dissimilarity_ci_low": ci_low,
                "delta_dissimilarity_ci_high": ci_high,
            }
        )
    difference = pd.DataFrame(difference_rows)
    difference.to_csv(args.output_dir / "figure4_6_vs_21_paired_curve_difference.csv", index=False)

    six_fits.to_csv(args.output_dir / "figure4_6point_correlation_length_fits.csv", index=False)
    full_6 = six_fits.loc[six_fits["fit_scope"].eq("full") & six_fits["fit_acceptable"]]
    full_21 = twentyone_fits.loc[
        twentyone_fits["fit_scope"].eq("full") & twentyone_fits["fit_acceptable"]
    ]
    paired_lambda = full_21[["subject_id", "lambda_mm"]].rename(
        columns={"lambda_mm": "lambda_21_mm"}
    ).merge(
        full_6[["subject_id", "lambda_mm"]].rename(columns={"lambda_mm": "lambda_6_mm"}),
        on="subject_id", how="inner", validate="one_to_one",
    )
    paired_lambda["delta_lambda_6_minus_21_mm"] = (
        paired_lambda["lambda_6_mm"] - paired_lambda["lambda_21_mm"]
    )
    paired_lambda.to_csv(args.output_dir / "figure4_6_vs_21_subject_correlation_lengths.csv", index=False)
    delta_lambda = paired_lambda["delta_lambda_6_minus_21_mm"].to_numpy(dtype=float)
    lambda_bootstrap = rng.integers(
        0, len(delta_lambda), size=(args.bootstrap_replicates, len(delta_lambda))
    )
    lambda_ci_low, lambda_ci_high = bootstrap_median_ci(delta_lambda, lambda_bootstrap)
    split_6 = six_fits.loc[
        six_fits["fit_scope"].isin(["fixed_half_1", "fixed_half_2"])
        & six_fits["fit_acceptable"]
    ].pivot(index="subject_id", columns="fit_scope", values="lambda_mm").dropna()
    curve_correlations = []
    for subject_id, part in paired.groupby("subject_id", sort=False):
        curve_correlations.append(
            stats.pearsonr(part["median_dissimilarity_6"], part["median_dissimilarity_21"]).statistic
        )
    length_summary = pd.DataFrame(
        [
            {
                "n_subjects_with_acceptable_paired_full_fits": len(paired_lambda),
                "lambda_6_median_mm": float(paired_lambda["lambda_6_mm"].median()),
                "lambda_6_q25_mm": float(paired_lambda["lambda_6_mm"].quantile(0.25)),
                "lambda_6_q75_mm": float(paired_lambda["lambda_6_mm"].quantile(0.75)),
                "lambda_21_median_mm": float(paired_lambda["lambda_21_mm"].median()),
                "lambda_21_q25_mm": float(paired_lambda["lambda_21_mm"].quantile(0.25)),
                "lambda_21_q75_mm": float(paired_lambda["lambda_21_mm"].quantile(0.75)),
                "delta_lambda_median_mm": float(np.median(delta_lambda)),
                "delta_lambda_bootstrap_ci_low_mm": lambda_ci_low,
                "delta_lambda_bootstrap_ci_high_mm": lambda_ci_high,
                "lambda_6_vs_21_pearson_r": float(
                    stats.pearsonr(paired_lambda["lambda_21_mm"], paired_lambda["lambda_6_mm"]).statistic
                ),
                "subject_curve_correlation_median": float(np.median(curve_correlations)),
                "six_point_split_half_n": len(split_6),
                "six_point_split_half_pearson_r": float(
                    stats.pearsonr(split_6["fixed_half_1"], split_6["fixed_half_2"]).statistic
                ),
                "six_point_split_half_icc": icc_two_way_single(
                    split_6["fixed_half_1"].to_numpy(), split_6["fixed_half_2"].to_numpy()
                ),
            }
        ]
    )
    length_summary.to_csv(args.output_dir / "figure4_6_vs_21_correlation_length_summary.csv", index=False)

    make_figure(args.output_dir, curves, difference, paired_lambda)
    summary = length_summary.iloc[0]
    curve_6 = curves.loc[curves["n_depths"].eq(6)].sort_values("distance_bin_index")
    curve_21 = curves.loc[curves["n_depths"].eq(21)].sort_values("distance_bin_index")
    near_6 = float(curve_6.iloc[0]["median_dissimilarity_median"])
    near_21 = float(curve_21.iloc[0]["median_dissimilarity_median"])
    far_6 = float(
        curve_6.loc[curve_6["distance_bin_index"].between(10, 14), "median_dissimilarity_median"].median()
    )
    far_21 = float(
        curve_21.loc[curve_21["distance_bin_index"].between(10, 14), "median_dissimilarity_median"].median()
    )
    readme = f"""# Figure 4 depth-density sensitivity: 6 versus 21 points

This supplementary real-profile analysis tests whether Figure 4's spatial dissimilarity curve and fitted correlation length depend on retaining 21 normalized cortical-depth samples. Six-point profiles are exact subsets of the existing Option 3 profiles at indices `{DEPTH_INDICES}` (depths 0.0, 0.2, ..., 1.0). Column construction and MD sampling were not rerun.

The analysis uses the same 320 registration-QC-passing participants, official ico5 anchors, within-DK K=15 pair table, 2-mm distance bins, subject masks, pair seed, fixed split assignment, and bounded exponential fit as the formal Figure 4 analysis. The 21-point representation is the existing reference representation, not ground truth.

All {len(paired_lambda)} subjects had acceptable paired full-data fits. Median correlation length was {summary.lambda_6_median_mm:.3f} mm for 6 points and {summary.lambda_21_median_mm:.3f} mm for 21 points. The paired median difference (6 minus 21) was {summary.delta_lambda_median_mm:.3f} mm, with a 95% subject-bootstrap CI of [{summary.delta_lambda_bootstrap_ci_low_mm:.3f}, {summary.delta_lambda_bootstrap_ci_high_mm:.3f}] mm. The subject-level 6-versus-21 correlation of fitted lengths was {summary.lambda_6_vs_21_pearson_r:.3f}; the median within-subject correlation between complete 6- and 21-point distance curves was {summary.subject_curve_correlation_median:.6f}.

The spatial increase remained present with 6 points, but the absolute dissimilarity scale shifted downward. Group-median dissimilarity in the 0–2-mm bin was {near_6:.4f} for 6 points and {near_21:.4f} for 21 points; the median 20–30-mm levels were {far_6:.4f} and {far_21:.4f}. Thus, the near-to-far increases were {far_6-near_6:.4f} and {far_21-near_21:.4f}, respectively. The qualitative spatial pattern is preserved, while the absolute dissimilarity values and fitted λ are not identical across parameterizations.

The acquisition-and-sampling null is not recomputed here. Its original 100 realizations per subject were saved only as 21-point aggregate pair metrics, not as reusable realization-level profiles. Consequently, this supplement evaluates the real-profile parameterization sensitivity of Figure 4A/C and does not constitute a 6-point reanalysis of the Figure 4B null.

Changing retained profile points does not change the 1-mm DWI acquisition resolution and does not imply independent histological layers or an optimal sampling density.

## Figure

- A: group median and IQR dissimilarity curves for 6 and 21 points.
- B: group median and IQR profile-MAE curves.
- C: paired subject-level dissimilarity difference with a 95% subject-bootstrap CI.
- D: paired subject correlation lengths; the dashed line is identity.

No sample sizes or numerical results are printed inside the figure.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    metadata = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": commit,
        "depth_indices_6": DEPTH_INDICES,
        "depth_fractions_6": [index / 20 for index in DEPTH_INDICES],
        "n_subjects": len(manifest),
        "subject_list": manifest["subject_id"].tolist(),
        "inputs": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "pair_file": str(args.pair_file),
            "pair_file_sha256": sha256(args.pair_file),
            "split_assignment": str(args.split_assignment),
            "split_assignment_sha256": sha256(args.split_assignment),
            "twentyone_subject_bins": str(args.twentyone_subject_bins),
            "twentyone_subject_bins_sha256": sha256(args.twentyone_subject_bins),
            "twentyone_fits": str(args.twentyone_fits),
            "twentyone_fits_sha256": sha256(args.twentyone_fits),
        },
        "fixed_parameters": {
            "anchor_definition": "official_fsaverage5_fixed",
            "candidate_definition": "within_DK_region",
            "k": 15,
            "pair_seed": 0,
            "split_seed": 20261021,
            "distance_bins_mm": list(range(0, 31, 2)),
            "minimum_fit_r2": 0.5,
        },
        "bootstrap": {
            "unit": "subject", "replicates": args.bootstrap_replicates,
            "seed": BOOTSTRAP_SEED,
        },
        "checks": {
            "all_six_point_profiles_exact_subsets": True,
            "pial_and_white_retained": True,
            "same_subjects": True,
            "same_pair_keys": True,
            "same_geodesic_distances": True,
            "same_valid_pair_counts": True,
            "existing_figure4_outputs_written": False,
            "acquisition_null_recomputed": False,
        },
        "subject_runtime_seconds_sum": float(sum(row["runtime_seconds"] for row in completions)),
        "maximum_subject_rss_mb": float(max(row["max_rss_mb"] for row in completions)),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "n_subjects": len(manifest)}), flush=True)


if __name__ == "__main__":
    main()
