#!/usr/bin/env python3
"""Evaluate the real-profile dissimilarity plateau from 20 to 100 mm.

Description
-----------
Evaluate the real-profile dissimilarity plateau from 20 to 100 mm.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--pair-table`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/compute_long_range_platform.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_spatial_null import load_sampling_geometry, subject_curve
from surface_pca_surrogate import load_real_profiles


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261020
MATERIAL_INCREASE_MARGIN = 0.01


def bootstrap_median_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.asarray(values, dtype=float)
    indices = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
    )
    return tuple(np.quantile(np.median(values[indices], axis=1), [.025, .975]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_all = pd.read_csv(args.manifest, dtype={"subject_id": str})
    if "input_complete" in manifest_all:
        excluded_incomplete = manifest_all.loc[
            ~manifest_all.input_complete.astype(bool), "subject_id"
        ].tolist()
        manifest = manifest_all.loc[
            manifest_all.input_complete.astype(bool)
        ].copy()
    else:
        excluded_incomplete = []
        manifest = manifest_all
    pairs = pd.read_parquet(args.pair_table)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_rows = []
    failures = []
    for subject_index, (_, row) in enumerate(manifest.iterrows(), start=1):
        subject_id = str(row.subject_id)
        print(
            f"[{subject_index}/{len(manifest)}] {subject_id}",
            flush=True,
        )
        try:
            profiles_by_hemisphere = {}
            valid_by_hemisphere = {}
            for hemisphere in ("lh", "rh"):
                _, geometry_valid, _ = load_sampling_geometry(
                    Path(row[f"clean_profile_{hemisphere}"])
                )
                profiles, profile_valid = load_real_profiles(
                    Path(row[f"clean_profile_{hemisphere}"])
                )
                profiles_by_hemisphere[hemisphere] = profiles
                valid_by_hemisphere[hemisphere] = geometry_valid & profile_valid
            curve = subject_curve(
                profiles_by_hemisphere, valid_by_hemisphere, pairs
            )
            curve.insert(0, "subject_id", subject_id)
            subject_rows.append(curve)
        except Exception as error:
            failures.append({
                "subject_id": subject_id,
                "error_type": type(error).__name__,
                "error": str(error),
            })
    if not subject_rows:
        raise RuntimeError("No long-range subject curves were completed")
    subject = pd.concat(subject_rows, ignore_index=True)
    subject.to_csv(
        args.output_dir / "figure4_long_range_subject_bins.csv", index=False
    )
    pd.DataFrame(failures, columns=["subject_id", "error_type", "error"]).to_csv(
        args.output_dir / "figure4_long_range_failures.csv", index=False
    )
    group_rows = []
    for keys, table in subject.groupby(
        ["distance_bin_index", "distance_bin"], observed=True
    ):
        bin_index, bin_label = keys
        values = table.median_dissimilarity
        group_rows.append({
            "distance_bin_index": bin_index,
            "distance_bin": bin_label,
            "median_distance_mm": table.median_geodesic_distance_mm.median(),
            "dissimilarity_median": values.median(),
            "dissimilarity_q25": values.quantile(.25),
            "dissimilarity_q75": values.quantile(.75),
            "n_subjects": table.subject_id.nunique(),
            "anchor_count_total": int(table.n_anchors.sum()),
            "pair_count_total": int(table.n_pairs.sum()),
        })
    group = pd.DataFrame(group_rows).sort_values("distance_bin_index")
    group.to_csv(
        args.output_dir / "figure4_long_range_group_summary.csv", index=False
    )

    wide = subject.pivot(
        index="subject_id", columns="distance_bin", values="median_dissimilarity"
    )
    wide["30-60"] = wide[["30-45", "45-60"]].median(axis=1)
    wide["60-100"] = wide[["60-80", "80-100"]].median(axis=1)
    comparison_rows = []
    for interval in ("30-60", "60-100"):
        difference = (wide[interval] - wide["20-30"]).dropna().to_numpy()
        ci_low, ci_high = bootstrap_median_ci(difference)
        comparison_rows.append({
            "comparison": f"{interval} minus 20-30 mm",
            "n_subjects": len(difference),
            "median_change": float(np.median(difference)),
            "q25": float(np.quantile(difference, .25)),
            "q75": float(np.quantile(difference, .75)),
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "material_increase_margin": MATERIAL_INCREASE_MARGIN,
            "upper_ci_below_margin": bool(ci_high < MATERIAL_INCREASE_MARGIN),
        })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(
        args.output_dir / "figure4_long_range_platform_comparisons.csv",
        index=False,
    )

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from qc.publication_style import (
        FIGURE_WIDTH_IN, PALETTE, save_publication_figure,
        set_publication_style, style_axes,
    )
    set_publication_style()
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.1))
    x = group.median_distance_mm.to_numpy(dtype=float)
    ax.plot(
        x, group.dissimilarity_median,
        color=PALETTE["blue"], marker="o", markersize=3.2, linewidth=1.4,
    )
    ax.fill_between(
        x, group.dissimilarity_q25, group.dissimilarity_q75,
        color=PALETTE["blue"], alpha=.2, linewidth=0,
    )
    ax.axvspan(20, 30, color=PALETTE["masked"], alpha=.45, linewidth=0)
    ax.set(
        xlabel="White-surface geodesic distance (mm)",
        ylabel="Profile dissimilarity (1 − r)",
        xlim=(18, 102),
    )
    style_axes(ax)
    save_publication_figure(
        fig, args.output_dir / "Figure4_long_range_platform_validation"
    )
    plt.close(fig)
    report = {
        "status": "complete_pilot",
        "subjects_requested": len(manifest_all),
        "subjects_expected_complete_inputs": len(manifest),
        "subjects_completed": subject.subject_id.nunique(),
        "subjects_failed": len(failures),
        "subjects_excluded_incomplete_inputs": excluded_incomplete,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "material_increase_margin": MATERIAL_INCREASE_MARGIN,
        "all_upper_ci_below_margin": bool(
            comparisons.upper_ci_below_margin.all()
        ),
    }
    (args.output_dir / "figure4_long_range_validation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
