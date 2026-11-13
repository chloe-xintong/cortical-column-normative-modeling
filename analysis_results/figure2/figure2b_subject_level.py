#!/usr/bin/env python3
"""Build Figure 2B subject-level summaries from cached vertex metrics.

Description
-----------
This intentionally does not reload profile inputs or recompute per-vertex MAE.
It reads the existing Figure 2 vertex-metrics Parquet files, pools valid LH/RH
vertices within each subject, and writes subject-weighted curvature-bin and
Spearman summaries.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--input-dir`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure2/figure2b_subject_level.py --help
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr, wilcoxon


COMPARISONS = ("O2Column_vs_O3", "O2FS_vs_O3")
METRIC_COLUMNS = (
    "subject_id",
    "valid",
    "profile_mae",
    "absolute_mean_curvature",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    return parser.parse_args()


def curvature_bin_indices(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign values to subject-specific equal-frequency curvature bins."""
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        raise ValueError("Curvature values do not support at least two bins")
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[values == edges[-1]] = edges.size - 2
    return indices


def iter_subject_metrics(path: Path) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield pooled valid MAE and curvature arrays for one subject at a time."""
    parquet_file = pq.ParquetFile(path)
    current_subject: str | None = None
    maes: list[np.ndarray] = []
    curvatures: list[np.ndarray] = []

    def finish() -> tuple[str, np.ndarray, np.ndarray] | None:
        if current_subject is None or not maes:
            return None
        return current_subject, np.concatenate(maes), np.concatenate(curvatures)

    for row_group in range(parquet_file.num_row_groups):
        frame = parquet_file.read_row_group(
            row_group, columns=METRIC_COLUMNS
        ).to_pandas()
        subject_ids = frame["subject_id"].unique()
        if subject_ids.size != 1:
            raise ValueError(
                f"Expected one subject per row group in {path}, found {subject_ids}"
            )
        subject_id = str(subject_ids[0])
        if current_subject is not None and subject_id != current_subject:
            completed = finish()
            if completed is not None:
                yield completed
            maes = []
            curvatures = []
        current_subject = subject_id
        valid = frame["valid"].to_numpy(dtype=bool)
        mae = frame["profile_mae"].to_numpy(dtype=float)
        curvature = frame["absolute_mean_curvature"].to_numpy(dtype=float)
        valid &= np.isfinite(mae) & np.isfinite(curvature)
        if valid.any():
            maes.append(mae[valid])
            curvatures.append(curvature[valid])
    completed = finish()
    if completed is not None:
        yield completed


def calculate_comparison(
    comparison: str,
    input_dir: Path,
    n_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = input_dir / f"figure2_{comparison}_vertex_metrics.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing vertex metrics: {path}")

    bin_rows = []
    spearman_rows = []
    for subject_id, maes, curvatures in iter_subject_metrics(path):
        indices = curvature_bin_indices(curvatures, n_bins)
        for curvature_bin in np.unique(indices):
            in_bin = indices == curvature_bin
            bin_rows.append(
                {
                    "comparison": comparison,
                    "subject_id": subject_id,
                    "curvature_bin": int(curvature_bin),
                    "median_profile_mae": float(np.median(maes[in_bin])),
                    "median_absolute_mean_curvature": float(
                        np.median(curvatures[in_bin])
                    ),
                    "n_vertices": int(in_bin.sum()),
                    "n_curvature_bins_subject": int(indices.max() + 1),
                }
            )
        spearman_rows.append(
            {
                "comparison": comparison,
                "subject_id": subject_id,
                "spearman_rho": float(spearmanr(curvatures, maes).statistic),
                "n_valid_vertices": int(maes.size),
            }
        )
    return pd.DataFrame(bin_rows), pd.DataFrame(spearman_rows)


def summarize_bins(subject_bins: pd.DataFrame) -> pd.DataFrame:
    return (
        subject_bins.groupby(["comparison", "curvature_bin"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            median_mae_across_subjects=("median_profile_mae", "median"),
            q25_mae_across_subjects=(
                "median_profile_mae", lambda values: values.quantile(0.25)
            ),
            q75_mae_across_subjects=(
                "median_profile_mae", lambda values: values.quantile(0.75)
            ),
            median_curvature_across_subjects=(
                "median_absolute_mean_curvature", "median"
            ),
            q25_curvature_across_subjects=(
                "median_absolute_mean_curvature", lambda values: values.quantile(0.25)
            ),
            q75_curvature_across_subjects=(
                "median_absolute_mean_curvature", lambda values: values.quantile(0.75)
            ),
        )
    )


def summarize_spearman(
    subject_spearman: pd.DataFrame,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20261021)
    for comparison, table in subject_spearman.groupby("comparison"):
        values = table["spearman_rho"].dropna().to_numpy()
        if values.size == 0:
            continue
        bootstrap_medians = np.median(
            rng.choice(values, size=(n_bootstrap, values.size), replace=True), axis=1
        )
        if np.any(values != 0.0):
            test = wilcoxon(values, alternative="two-sided", method="auto")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        else:
            statistic, p_value = np.nan, np.nan
        rows.append(
            {
                "comparison": comparison,
                "n_subjects": int(values.size),
                "median_spearman_rho": float(np.median(values)),
                "q25_spearman_rho": float(np.quantile(values, 0.25)),
                "q75_spearman_rho": float(np.quantile(values, 0.75)),
                "bootstrap_ci95_low": float(np.quantile(bootstrap_medians, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(bootstrap_medians, 0.975)),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.n_bins < 2:
        raise ValueError("--n-bins must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bin_tables = []
    spearman_tables = []
    for comparison in COMPARISONS:
        bins, spearman = calculate_comparison(
            comparison, args.input_dir, args.n_bins
        )
        bins.to_csv(
            args.output_dir / f"figure2_{comparison}_curvature_subject_bins.csv",
            index=False,
        )
        spearman.to_csv(
            args.output_dir / f"figure2_{comparison}_subject_spearman.csv",
            index=False,
        )
        bin_tables.append(bins)
        spearman_tables.append(spearman)

    subject_bins = pd.concat(bin_tables, ignore_index=True)
    subject_spearman = pd.concat(spearman_tables, ignore_index=True)
    summarize_bins(subject_bins).to_csv(
        args.output_dir / "figure2_curvature_cross_subject_summary.csv",
        index=False,
    )
    subject_spearman.to_csv(
        args.output_dir / "figure2_subject_spearman.csv", index=False
    )
    summarize_spearman(subject_spearman, args.n_bootstrap).to_csv(
        args.output_dir / "figure2_subject_spearman_summary.csv", index=False
    )
    print(f"Saved subject-level Figure 2B summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
