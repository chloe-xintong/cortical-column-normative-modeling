#!/usr/bin/env python3
"""Recompute Figure 4C from cached anchor summaries with fixed stratified halves.

Description
-----------
Recompute Figure 4C from cached anchor summaries with fixed stratified halves.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--results-root`` command-line input
- ``--pair-dir`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/recompute_fixed_split_half.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fit_correlation_lengths import fit_curve, icc_two_way_single
from spatial_metrics import aggregate_subject_bins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--min-fit-r2", type=float, default=0.5)
    return parser.parse_args()


def make_assignment(pair_dir: Path, seed: int) -> pd.DataFrame:
    pairs = pd.read_parquet(
        pair_dir / "figure4_geodesic_pairs_all.parquet",
        columns=["hemisphere", "anchor_vertex", "anchor_dk_label"],
    ).drop_duplicates()
    rng = np.random.default_rng(seed)
    rows = []
    for (hemisphere, label), table in pairs.groupby(
        ["hemisphere", "anchor_dk_label"], sort=True
    ):
        anchors = np.sort(table["anchor_vertex"].unique())
        shuffled = rng.permutation(anchors)
        halves = np.ones(len(shuffled), dtype=int)
        halves[len(shuffled) // 2 :] = 2
        rows.extend(
            {
                "hemisphere": hemisphere,
                "anchor_vertex": int(anchor),
                "anchor_dk_label": int(label),
                "split_half": int(split),
            }
            for anchor, split in zip(shuffled, halves)
        )
    return pd.DataFrame(rows)


def reliability_table(fits: pd.DataFrame) -> pd.DataFrame:
    halves = fits.loc[fits["fit_scope"].str.startswith("fixed_half_")].pivot(
        index="subject_id", columns="fit_scope", values="lambda_mm"
    )
    paired = halves.dropna(subset=["fixed_half_1", "fixed_half_2"])
    difference = paired["fixed_half_1"] - paired["fixed_half_2"]
    return pd.DataFrame(
        [
            {
                "n_subjects": len(paired),
                "pearson_r": stats.pearsonr(
                    paired["fixed_half_1"], paired["fixed_half_2"]
                ).statistic if len(paired) >= 3 else np.nan,
                "spearman_rho": stats.spearmanr(
                    paired["fixed_half_1"], paired["fixed_half_2"]
                ).statistic if len(paired) >= 3 else np.nan,
                "icc_two_way_single": icc_two_way_single(
                    paired["fixed_half_1"].to_numpy(), paired["fixed_half_2"].to_numpy()
                ),
                "median_absolute_difference_mm": difference.abs().median(),
                "n_failed_fits": int(2 * len(halves) - len(paired) * 2),
            }
        ]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignment = make_assignment(args.pair_dir, args.seed)
    assignment.to_csv(args.output_dir / "figure4_fixed_anchor_split_assignment.csv", index=False)

    split_tables = []
    fit_rows = []
    paths = sorted(args.results_root.glob("subjects/*/figure4_real_anchor_bins.parquet"))
    for path in paths:
        table = pd.read_parquet(path)
        all_anchor = table.loc[table["pair_type"].eq("all")].drop(
            columns=["split_half"]
        ).merge(
            assignment[["hemisphere", "anchor_vertex", "split_half"]],
            on=["hemisphere", "anchor_vertex"],
            how="inner",
            validate="many_to_one",
        )
        subject_id = str(all_anchor["subject_id"].iloc[0])
        for half, part in all_anchor.groupby("split_half"):
            summary = aggregate_subject_bins(part)
            summary.insert(0, "split_half", int(half))
            summary.insert(0, "subject_id", subject_id)
            split_tables.append(summary)
            fit_rows.append(
                {
                    "subject_id": subject_id,
                    "fit_scope": f"fixed_half_{int(half)}",
                    **fit_curve(summary, min_fit_r2=args.min_fit_r2),
                }
            )
    split_bins = pd.concat(split_tables, ignore_index=True)
    fits = pd.DataFrame(fit_rows)
    split_bins.to_csv(args.output_dir / "figure4_fixed_split_half_subject_bins.csv", index=False)
    fits.to_csv(args.output_dir / "figure4_fixed_split_half_fits.csv", index=False)
    reliability_table(fits).to_csv(
        args.output_dir / "figure4_fixed_split_half_reliability.csv", index=False
    )
    print(f"Saved fixed split-half summaries for {len(paths)} subjects")


if __name__ == "__main__":
    main()
