#!/usr/bin/env python3
"""Plot real and FWHM=0 null profile MAE versus geodesic distance.

Description
-----------
Plot real and FWHM=0 null profile MAE versus geodesic distance.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--real-results-root`` command-line input
- ``--null-results-root`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/plot_profile_mae_curves.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BIN_INDICES = np.arange(15)
BIN_CENTERS_MM = 2 * BIN_INDICES + 1
MAE_DISPLAY_SCALE = 1.0e4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--real-results-root", type=Path, required=True)
    parser.add_argument("--null-results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_subject_tables(
    subject_ids: list[str], results_root: Path, filename: str
) -> tuple[pd.DataFrame, list[str]]:
    tables = []
    missing = []
    for subject_id in subject_ids:
        path = results_root / "subjects" / subject_id / filename
        if path.is_file():
            table = pd.read_csv(path)
            table["subject_id"] = subject_id
            tables.append(table)
        else:
            missing.append(subject_id)
    if not tables:
        raise RuntimeError(f"No {filename} files found under {results_root}")
    return pd.concat(tables, ignore_index=True), missing


def select_mae_rows(table: pd.DataFrame, *, is_null: bool) -> pd.DataFrame:
    required = {"subject_id", "distance_bin_index", "median_profile_mae"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    selected = table.copy()
    if "pair_type" in selected.columns:
        selected = selected.loc[selected["pair_type"].eq("all")]
    selected = selected.loc[selected["distance_bin_index"].isin(BIN_INDICES)]
    if is_null:
        if "psf_fwhm_mm" not in selected.columns:
            raise ValueError("Null tables do not contain psf_fwhm_mm")
        selected = selected.loc[np.isclose(selected["psf_fwhm_mm"], 0.0)]
        if "replicate" not in selected.columns:
            raise ValueError("Null tables do not contain replicate")
        selected = (
            selected.groupby(
                ["subject_id", "distance_bin_index"], observed=True
            )["median_profile_mae"]
            .median()
            .rename("subject_median_profile_mae")
            .reset_index()
        )
    else:
        selected = selected.rename(
            columns={"median_profile_mae": "subject_median_profile_mae"}
        )[["subject_id", "distance_bin_index", "subject_median_profile_mae"]]
    duplicated = selected.duplicated(["subject_id", "distance_bin_index"])
    if duplicated.any():
        raise ValueError("More than one MAE value remains per subject and distance bin")
    return selected


def validate_bins(table: pd.DataFrame, label: str) -> None:
    counts = table.groupby("subject_id").distance_bin_index.nunique()
    invalid = counts.loc[~counts.eq(len(BIN_INDICES))]
    if not invalid.empty:
        preview = ", ".join(f"{key} ({value})" for key, value in invalid.head().items())
        raise ValueError(f"{label}: subjects without all 15 distance bins: {preview}")
    if not np.isfinite(table["subject_median_profile_mae"]).all():
        raise ValueError(f"{label}: non-finite profile MAE values found")


def group_summary(table: pd.DataFrame, curve: str) -> pd.DataFrame:
    summary = (
        table.groupby("distance_bin_index", observed=True)["subject_median_profile_mae"]
        .agg(
            n_subjects="count",
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reindex(BIN_INDICES)
        .reset_index()
    )
    summary.insert(0, "curve", curve)
    summary.insert(2, "distance_bin_center_mm", BIN_CENTERS_MM)
    return summary


def plot_curve(ax: plt.Axes, table: pd.DataFrame, color: str, label: str) -> None:
    x = table["distance_bin_center_mm"].to_numpy(dtype=float)
    median = table["median"].to_numpy(dtype=float) * MAE_DISPLAY_SCALE
    q25 = table["q25"].to_numpy(dtype=float) * MAE_DISPLAY_SCALE
    q75 = table["q75"].to_numpy(dtype=float) * MAE_DISPLAY_SCALE
    ax.plot(x, median, color=color, linewidth=1.5, label=label)
    ax.fill_between(x, q25, q75, color=color, alpha=0.2, linewidth=0)


def main() -> None:
    args = parse_args()
    subject_ids = (
        pd.read_csv(args.manifest, dtype={"subject_id": str})["subject_id"]
        .drop_duplicates()
        .tolist()
    )
    real_raw, real_missing = read_subject_tables(
        subject_ids, args.real_results_root, "figure4_real_subject_bins.csv"
    )
    null_raw, null_missing = read_subject_tables(
        subject_ids, args.null_results_root, "figure4_null_subject_bins.csv"
    )
    real = select_mae_rows(real_raw, is_null=False)
    null = select_mae_rows(null_raw, is_null=True)
    validate_bins(real, "Real")
    validate_bins(null, "FWHM=0 null")

    real_subjects = set(real["subject_id"])
    null_subjects = set(null["subject_id"])
    paired_subjects = sorted(real_subjects & null_subjects)
    if not paired_subjects:
        raise RuntimeError("Real and null results have no subjects in common")
    paired_real = real.loc[real["subject_id"].isin(paired_subjects)].copy()
    paired_null = null.loc[null["subject_id"].isin(paired_subjects)].copy()

    real_summary = group_summary(real, "real")
    comparison_summary = pd.concat(
        [
            group_summary(paired_real, "real"),
            group_summary(paired_null, "fwhm0_null"),
        ],
        ignore_index=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    real.to_csv(args.output_dir / "figure4_real_profile_mae_subject_bins.csv", index=False)
    pd.concat(
        [paired_real.assign(curve="real"), paired_null.assign(curve="fwhm0_null")],
        ignore_index=True,
    ).to_csv(
        args.output_dir / "figure4_real_vs_fwhm0_profile_mae_subject_bins.csv",
        index=False,
    )
    real_summary.to_csv(
        args.output_dir / "figure4_real_profile_mae_group_summary.csv", index=False
    )
    comparison_summary.to_csv(
        args.output_dir / "figure4_real_vs_fwhm0_profile_mae_group_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [{
            "manifest_subjects": len(subject_ids),
            "real_subjects": len(real_subjects),
            "fwhm0_null_subjects": len(null_subjects),
            "paired_subjects": len(paired_subjects),
            "real_missing_subjects": len(real_missing),
            "null_missing_subjects": len(null_missing),
        }]
    ).to_csv(args.output_dir / "figure4_profile_mae_status.csv", index=False)

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from qc.publication_style import (
        FIGURE_WIDTH_IN,
        PALETTE,
        save_publication_figure,
        set_publication_style,
        style_axes,
    )

    set_publication_style()
    y_label = r"Profile MAE ($\times 10^{-4}$ mm$^2$/s)"
    x_label = "White-surface geodesic distance (mm)"

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.3))
    plot_curve(ax, real_summary, PALETTE["blue"], "Real")
    ax.set(xlabel=x_label, ylabel=y_label, xticks=np.arange(0, 31, 5), xlim=(0, 30))
    style_axes(ax)
    save_publication_figure(
        fig, args.output_dir / "Figure4_real_profile_mae_0_30mm"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.3))
    for curve, color, label in (
        ("real", PALETTE["blue"], "Real"),
        ("fwhm0_null", PALETTE["orange"], "FWHM=0 null"),
    ):
        plot_curve(
            ax,
            comparison_summary.loc[comparison_summary["curve"].eq(curve)],
            color,
            label,
        )
    ax.set(xlabel=x_label, ylabel=y_label, xticks=np.arange(0, 31, 5), xlim=(0, 30))
    ax.legend(frameon=False)
    style_axes(ax)
    save_publication_figure(
        fig, args.output_dir / "Figure4_real_vs_fwhm0_profile_mae"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
