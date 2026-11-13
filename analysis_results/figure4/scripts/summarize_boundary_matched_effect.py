#!/usr/bin/env python3
"""Summarize frozen Figure 4D boundary-matched effects and QC.

Description
-----------
Summarize frozen Figure 4D boundary-matched effects and QC.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--pair-root`` command-line input
- ``--effect-root`` command-line input
- ``--output-dir`` command-line input
- ``--repo-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/summarize_boundary_matched_effect.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 20261027
N_BOOTSTRAP = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--effect-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--limit-subjects", type=int, default=None,
        help="Optional leading manifest subset for pilot validation only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_ids = pd.read_csv(args.manifest, dtype={"subject_id": str})[
        "subject_id"
    ].drop_duplicates().tolist()
    if args.limit_subjects is not None:
        if args.limit_subjects < 1:
            raise ValueError("--limit-subjects must be positive")
        subject_ids = subject_ids[: args.limit_subjects]
    subject_tables = []
    subject_pair_qc_parts = []
    audit_rows = []
    status_rows = []
    for subject_id in subject_ids:
        effect_dir = args.effect_root / "subjects" / subject_id
        pair_dir = args.pair_root / "subjects" / subject_id
        subject_path = effect_dir / "figure4D_subject_bin_effects.csv"
        pair_path = pair_dir / "figure4D_boundary_matched_pairs.parquet"
        audit_path = pair_dir / "figure4D_boundary_matching_audit.json"
        pair_qc_path = pair_dir / "figure4D_boundary_matching_qc_by_bin.csv"
        if all(
            path.is_file()
            for path in (subject_path, pair_path, audit_path, pair_qc_path)
        ):
            subject_tables.append(pd.read_csv(subject_path, dtype={"subject_id": str}))
            subject_pair_qc_parts.append(
                pd.read_csv(pair_qc_path, dtype={"subject_id": str})
            )
            audit_rows.append(json.loads(audit_path.read_text()))
            status_rows.append({"subject_id": subject_id, "status": "complete"})
        else:
            status_rows.append({"subject_id": subject_id, "status": "incomplete"})
    status = pd.DataFrame(status_rows)
    status.to_csv(args.output_dir / "figure4D_subject_status.csv", index=False)
    if not subject_tables:
        raise RuntimeError("No completed Figure 4D subject effects")
    if not status["status"].eq("complete").all():
        missing = status.loc[~status["status"].eq("complete"), "subject_id"].tolist()
        raise RuntimeError(f"Incomplete Figure 4D subjects: {missing[:10]}")

    subject = pd.concat(subject_tables, ignore_index=True)
    subject_pair_qc = pd.concat(subject_pair_qc_parts, ignore_index=True)
    audits = pd.DataFrame(audit_rows)
    subject.to_csv(args.output_dir / "figure4D_subject_bin_effects.csv", index=False)
    subject_pair_qc.to_csv(
        args.output_dir / "figure4D_subject_pair_qc_by_bin.csv", index=False
    )

    group = (
        subject.groupby(
            ["distance_bin_index", "distance_bin"], as_index=False, observed=True
        )
        .agg(
            n_subjects=("subject_id", "nunique"),
            delta_d_median=("delta_d", "median"),
            delta_d_q25=("delta_d", lambda values: values.quantile(0.25)),
            delta_d_q75=("delta_d", lambda values: values.quantile(0.75)),
            d_within_median=("d_within", "median"),
            d_across_median=("d_across", "median"),
            delta_profile_mae_median=("delta_profile_mae", "median"),
            region_count_median_subject=("region_count", "median"),
            anchor_count_median_subject=("anchor_count", "median"),
            match_count_median_subject=("match_count", "median"),
        )
    )
    group.to_csv(args.output_dir / "figure4D_group_summary.csv", index=False)

    subject_0_10 = (
        subject.groupby("subject_id", as_index=False)
        .agg(
            mean_delta_d_0_10=("delta_d", "mean"),
            distance_bins=("distance_bin_index", "nunique"),
            mean_delta_profile_mae_0_10=("delta_profile_mae", "mean"),
        )
    )
    if not subject_0_10["distance_bins"].eq(5).all():
        raise RuntimeError("At least one subject lacks a 0--10-mm distance bin")
    values = subject_0_10["mean_delta_d_0_10"].to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    bootstrap = np.median(values[indices], axis=1)
    scalar = pd.DataFrame(
        [
            {
                "n_subjects": len(values),
                "median_mean_delta_d_0_10": np.median(values),
                "q25_mean_delta_d_0_10": np.quantile(values, 0.25),
                "q75_mean_delta_d_0_10": np.quantile(values, 0.75),
                "bootstrap_ci95_low": np.quantile(bootstrap, 0.025),
                "bootstrap_ci95_high": np.quantile(bootstrap, 0.975),
                "positive_subject_count": int(np.sum(values > 0)),
                "positive_subject_fraction": float(np.mean(values > 0)),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_iterations": N_BOOTSTRAP,
            }
        ]
    )
    subject_0_10.to_csv(
        args.output_dir / "figure4D_subject_mean_delta_0_10.csv", index=False
    )
    scalar.to_csv(args.output_dir / "figure4D_0_10_group_summary.csv", index=False)

    qc_rows = []
    for keys, qc_frame in subject_pair_qc.groupby(
        ["distance_bin_index", "distance_bin"], observed=True, sort=True
    ):
        bin_index, bin_label = keys
        anchor_counts = qc_frame["matched_anchor_count"].to_numpy(float)
        full_k_counts = (
            qc_frame["full_k15_anchor_coverage"].to_numpy(float) * anchor_counts
        )
        qc_rows.append(
            {
                "distance_bin_index": bin_index,
                "distance_bin": bin_label,
                "n_subjects": qc_frame["subject_id"].nunique(),
                "matched_pair_rows_total": int(qc_frame["matched_pair_rows"].sum()),
                "matched_anchor_count_total": int(anchor_counts.sum()),
                "matched_anchor_count_median_subject": np.median(anchor_counts),
                "valid_anchor_dk_regions_median_subject": qc_frame[
                    "matched_region_count"
                ].median(),
                "n_matched_median_of_subject_medians": qc_frame[
                    "n_matched_median"
                ].median(),
                "n_matched_q25_median_subject": qc_frame["n_matched_q25"].median(),
                "n_matched_q75_median_subject": qc_frame["n_matched_q75"].median(),
                "full_k15_anchor_coverage": full_k_counts.sum() / anchor_counts.sum(),
                "distance_mismatch_median_mm_median_subject": qc_frame[
                    "distance_mismatch_median_mm"
                ].median(),
                "distance_mismatch_q95_mm_max_subject": qc_frame[
                    "distance_mismatch_q95_mm"
                ].max(),
                "distance_mismatch_maximum_mm": qc_frame[
                    "distance_mismatch_maximum_mm"
                ].max(),
                "within_distance_median_mm_median_subject": qc_frame[
                    "within_distance_median_mm"
                ].median(),
                "within_distance_q25_mm_median_subject": qc_frame[
                    "within_distance_q25_mm"
                ].median(),
                "within_distance_q75_mm_median_subject": qc_frame[
                    "within_distance_q75_mm"
                ].median(),
                "across_distance_median_mm_median_subject": qc_frame[
                    "across_distance_median_mm"
                ].median(),
                "across_distance_q25_mm_median_subject": qc_frame[
                    "across_distance_q25_mm"
                ].median(),
                "across_distance_q75_mm_median_subject": qc_frame[
                    "across_distance_q75_mm"
                ].median(),
            }
        )
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(args.output_dir / "figure4D_group_pair_qc_by_bin.csv", index=False)

    rule_columns = [
        "self_pairs", "duplicate_rows",
        "within_candidate_duplicates_within_anchor_bin",
        "across_candidate_duplicates_within_anchor_bin",
        "duplicate_directed_pairs", "cross_bin_duplicates",
        "within_label_violations", "across_label_violations",
    ]
    rule_totals = {column: int(audits[column].sum()) for column in rule_columns}
    if any(rule_totals.values()):
        raise AssertionError(f"Figure 4D pair QC violations: {rule_totals}")
    maximum_mismatch = float(audits["maximum_distance_difference_mm"].max())
    caliper = float(audits["caliper_mm"].max())
    if maximum_mismatch > caliper + 1e-8:
        raise AssertionError("Figure 4D caliper violation")
    qc_status = {
        "status": "pass",
        "subjects": len(subject_ids),
        **rule_totals,
        "maximum_distance_difference_mm": maximum_mismatch,
        "caliper_mm": caliper,
        "reciprocal_pair_count_total": int(audits["reciprocal_pair_count"].sum()),
    }
    (args.output_dir / "figure4D_pair_qc_status.json").write_text(
        json.dumps(qc_status, indent=2) + "\n"
    )

    plot_cached(args.output_dir, args.repo_root)

    report = {
        "status": "complete",
        "subjects": len(subject_ids),
        "distance_range_mm": [0, 10],
        "null_used": False,
        "aggregation": (
            "matched-pair mean -> anchor-bin mean -> bilateral subject-region-bin "
            "mean -> equal-region subject-bin mean -> group median/IQR"
        ),
        "pair_qc": qc_status,
        "subject_scalar_summary": scalar.iloc[0].to_dict(),
    }
    (args.output_dir / "figure4D_status.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


def plot_cached(output_dir, repo_root):
    output_dir = Path(output_dir)
    group = pd.read_csv(output_dir / "figure4D_group_summary.csv")
    sys.path.insert(0, str(repo_root))
    from qc.publication_style import (  # pylint: disable=import-outside-toplevel
        FIGURE_WIDTH_IN, PALETTE, save_publication_figure,
        set_publication_style, style_axes,
    )
    set_publication_style()
    x = group["distance_bin_index"].to_numpy(float) * 2 + 1
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.1))
    ax.axhline(0, color=PALETTE["reference"], linewidth=0.9, linestyle="--")
    ax.plot(
        x, group["delta_d_median"], color=PALETTE["blue"],
        marker="o", markersize=3, linewidth=1.6,
    )
    ax.fill_between(
        x, group["delta_d_q25"], group["delta_d_q75"],
        color=PALETTE["blue"], alpha=.12, linewidth=0,
    )
    ax.set(
        xlabel="White-surface geodesic distance (mm)",
        ylabel="Matched Δ profile dissimilarity\n(across − within)",
        xlim=(0, 10), xticks=np.arange(0, 11, 2),
    )
    style_axes(ax, grid=True)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.20, top=0.98)
    save_publication_figure(fig, output_dir / "Figure4D_boundary_matched_0_10")
    plt.close(fig)


if __name__ == "__main__":
    main()
