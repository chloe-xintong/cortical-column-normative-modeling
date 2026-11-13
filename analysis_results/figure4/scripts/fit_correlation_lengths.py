#!/usr/bin/env python3
"""Aggregate Figure 4 real curves and fit subject-level correlation lengths.

Description
-----------
Aggregate Figure 4 real curves and fit subject-level correlation lengths.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--results-root`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/fit_correlation_lengths.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def model(distance: np.ndarray, asymptote: float, amplitude: float, scale: float):
    return asymptote - amplitude * np.exp(-distance / scale)


def fit_curve(table: pd.DataFrame, min_fit_r2: float = 0.5) -> dict[str, object]:
    valid = table.loc[
        np.isfinite(table["median_geodesic_distance_mm"])
        & np.isfinite(table["median_dissimilarity"])
    ].sort_values("median_geodesic_distance_mm")
    result: dict[str, object] = {
        "lambda_mm": np.nan,
        "d_infinity": np.nan,
        "amplitude": np.nan,
        "fit_r2": np.nan,
        "residual_ss": np.nan,
        "residual_rmse": np.nan,
        "hit_parameter_boundary": False,
        "fit_acceptable": False,
        "number_of_valid_bins": len(valid),
        "fit_success": False,
        "failure_reason": "",
    }
    if len(valid) < 6:
        result["failure_reason"] = "fewer_than_six_valid_bins"
        return result
    x = valid["median_geodesic_distance_mm"].to_numpy(dtype=float)
    y = valid["median_dissimilarity"].to_numpy(dtype=float)
    initial_asymptote = max(float(np.max(y)), 1e-6)
    initial_amplitude = max(initial_asymptote - float(np.min(y)), 1e-6)
    try:
        parameters, _ = curve_fit(
            model,
            x,
            y,
            p0=(initial_asymptote, initial_amplitude, 5.0),
            bounds=((0.0, 0.0, 0.1), (2.0, 2.0, 100.0)),
            maxfev=20_000,
        )
        predicted = model(x, *parameters)
        residual = np.sum((y - predicted) ** 2)
        total = np.sum((y - np.mean(y)) ** 2)
        result.update(
            {
                "lambda_mm": float(parameters[2]),
                "d_infinity": float(parameters[0]),
                "amplitude": float(parameters[1]),
                "fit_r2": float(1.0 - residual / total) if total > 0 else np.nan,
                "residual_ss": float(residual),
                "residual_rmse": float(np.sqrt(residual / len(y))),
                "hit_parameter_boundary": bool(
                    np.isclose(parameters[0], 0.0)
                    or np.isclose(parameters[0], 2.0)
                    or np.isclose(parameters[1], 0.0)
                    or np.isclose(parameters[1], 2.0)
                    or np.isclose(parameters[2], 0.1)
                    or np.isclose(parameters[2], 100.0)
                ),
                "fit_success": True,
            }
        )
        result["fit_acceptable"] = bool(
            result["fit_r2"] >= min_fit_r2
            and not result["hit_parameter_boundary"]
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        result["failure_reason"] = f"{type(error).__name__}: {error}"
    return result


def icc_two_way_single(first: np.ndarray, second: np.ndarray) -> float:
    values = np.column_stack((first, second))
    complete = np.isfinite(values).all(axis=1)
    values = values[complete]
    n_subjects, n_raters = values.shape
    if n_subjects < 3:
        return np.nan
    grand_mean = values.mean()
    subject_means = values.mean(axis=1)
    rater_means = values.mean(axis=0)
    ms_subject = n_raters * np.sum((subject_means - grand_mean) ** 2) / (n_subjects - 1)
    ms_error = np.sum(
        (values - subject_means[:, None] - rater_means[None, :] + grand_mean) ** 2
    ) / ((n_subjects - 1) * (n_raters - 1))
    denominator = ms_subject + (n_raters - 1) * ms_error
    return float((ms_subject - ms_error) / denominator) if denominator > 0 else np.nan


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_tables = []
    split_tables = []
    statuses = []
    for subject_id in manifest["subject_id"]:
        subject_dir = args.results_root / "subjects" / subject_id
        completion = subject_dir / "completion.json"
        failure = subject_dir / "failure.json"
        if completion.is_file():
            subject_tables.append(
                pd.read_csv(subject_dir / "figure4_real_subject_bins.csv")
            )
            split_tables.append(
                pd.read_csv(subject_dir / "figure4_real_split_half_subject_bins.csv")
            )
            status = "complete"
            reason = ""
        elif failure.is_file():
            failure_data = json.loads(failure.read_text(encoding="utf-8"))
            status = "failed"
            reason = str(failure_data.get("error", ""))
        else:
            status = "incomplete"
            reason = "No completion JSON"
        statuses.append({"subject_id": subject_id, "status": status, "reason": reason})

    status_table = pd.DataFrame(statuses)
    status_table.to_csv(args.output_dir / "figure4_real_status.csv", index=False)
    status_table.loc[~status_table["status"].eq("complete")].to_csv(
        args.output_dir / "figure4_real_failures.csv",
        index=False,
    )
    if not subject_tables:
        raise RuntimeError("No completed Figure 4 real results were found")

    subject_bins = pd.concat(subject_tables, ignore_index=True)
    split_bins = pd.concat(split_tables, ignore_index=True)
    subject_bins.to_csv(args.output_dir / "figure4_real_subject_bins.csv", index=False)
    group_summary = (
        subject_bins.groupby(
            ["distance_bin_index", "distance_bin", "pair_type"],
            observed=True,
        )["median_dissimilarity"]
        .agg(
            n_subjects="count",
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    group_summary.to_csv(
        args.output_dir / "figure4_real_group_summary.csv",
        index=False,
    )

    fits = []
    for subject_id, table in subject_bins.loc[
        subject_bins["pair_type"].eq("all")
    ].groupby("subject_id"):
        fits.append({"subject_id": subject_id, "fit_scope": "full", **fit_curve(table)})
    for (subject_id, split_half), table in split_bins.groupby(
        ["subject_id", "split_half"]
    ):
        fits.append(
            {
                "subject_id": subject_id,
                "fit_scope": f"half_{int(split_half)}",
                **fit_curve(table),
            }
        )
    fit_table = pd.DataFrame(fits)
    fit_table.to_csv(args.output_dir / "figure4_correlation_lengths.csv", index=False)

    halves = fit_table.loc[fit_table["fit_scope"].str.startswith("half_")].pivot(
        index="subject_id",
        columns="fit_scope",
        values="lambda_mm",
    )
    paired = halves.dropna(subset=["half_1", "half_2"])
    difference = paired["half_1"] - paired["half_2"]
    mean_difference = difference.mean()
    sd_difference = difference.std(ddof=1)
    reliability = pd.DataFrame(
        [
            {
                "n_subjects": len(paired),
                "pearson_r": stats.pearsonr(
                    paired["half_1"], paired["half_2"]
                ).statistic
                if len(paired) >= 3
                else np.nan,
                "spearman_rho": stats.spearmanr(
                    paired["half_1"], paired["half_2"]
                ).statistic
                if len(paired) >= 3
                else np.nan,
                "icc_two_way_single": icc_two_way_single(
                    paired["half_1"].to_numpy(),
                    paired["half_2"].to_numpy(),
                ),
                "median_absolute_difference_mm": difference.abs().median(),
                "bland_altman_mean_difference_mm": mean_difference,
                "bland_altman_lower_limit_mm": mean_difference - 1.96 * sd_difference,
                "bland_altman_upper_limit_mm": mean_difference + 1.96 * sd_difference,
            }
        ]
    )
    reliability.to_csv(
        args.output_dir / "figure4_split_half_reliability.csv",
        index=False,
    )

    boundary = subject_bins.loc[
        subject_bins["pair_type"].isin(["within_region", "across_boundary"])
    ]
    boundary_wide = boundary.pivot(
        index=["subject_id", "distance_bin_index", "distance_bin"],
        columns="pair_type",
        values=["median_dissimilarity", "median_geodesic_distance_mm", "n_pairs"],
    )
    boundary_wide.columns = ["_".join(column) for column in boundary_wide.columns]
    boundary_wide = boundary_wide.reset_index()
    boundary_wide["across_minus_within_dissimilarity"] = (
        boundary_wide["median_dissimilarity_across_boundary"]
        - boundary_wide["median_dissimilarity_within_region"]
    )
    boundary_wide["absolute_median_distance_difference_mm"] = np.abs(
        boundary_wide["median_geodesic_distance_mm_across_boundary"]
        - boundary_wide["median_geodesic_distance_mm_within_region"]
    )
    boundary_wide.to_csv(
        args.output_dir / "figure4_within_across_subject_bins.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
