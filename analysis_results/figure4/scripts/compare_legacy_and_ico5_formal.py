#!/usr/bin/env python3
"""Create the requested legacy-versus-formal Figure 4 robustness table.

Description
-----------
Create the requested legacy-versus-formal Figure 4 robustness table.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--legacy-null-dir`` command-line input
- ``--legacy-fit-file`` command-line input
- ``--formal-null-dir`` command-line input
- ``--formal-fit-file`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/compare_legacy_and_ico5_formal.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def extract(null_dir: Path, fit_path: Path) -> dict[str, float]:
    group = pd.read_csv(null_dir / "figure4B_acquisition_group_summary.csv")
    final = group.loc[group.n_realizations.eq(group.n_realizations.max())]
    nearest = final.loc[final.distance_bin_index.eq(0)].iloc[0]
    auc = pd.read_csv(null_dir / "figure4B_local_excess_auc_group_summary.csv").iloc[0]
    fits = pd.read_csv(fit_path)
    full = fits.loc[fits.fit_scope.eq("full") & fits.fit_success]
    if "fit_acceptable" in full:
        full = full.loc[full.fit_acceptable]
    return {
        "delta_L_0_2_mm": nearest.real_minus_null_local_excess_median,
        "auc_0_10_mm_median": auc.auc_residual_local_excess_0_10_mm_median,
        "auc_0_10_mm_q25": auc.auc_residual_local_excess_0_10_mm_q25,
        "auc_0_10_mm_q75": auc.auc_residual_local_excess_0_10_mm_q75,
        "lambda_median_mm": full.lambda_mm.median(),
        "lambda_q25_mm": full.lambda_mm.quantile(.25),
        "lambda_q75_mm": full.lambda_mm.quantile(.75),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-null-dir", type=Path, required=True)
    parser.add_argument("--legacy-fit-file", type=Path, required=True)
    parser.add_argument("--formal-null-dir", type=Path, required=True)
    parser.add_argument("--formal-fit-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    legacy = extract(args.legacy_null_dir, args.legacy_fit_file)
    formal = extract(args.formal_null_dir, args.formal_fit_file)
    rows = [
        {
            "version": "Legacy analysis",
            "anchor": "2,048 sampled anchors per hemisphere",
            "pair_scope": "all valid cortical candidates",
            "k": 20,
            **legacy,
        },
        {
            "version": "Formal frozen analysis",
            "anchor": "18,742 bilateral official ico5 anchors",
            "pair_scope": "within-DK-region candidates",
            "k": 15,
            **formal,
        },
    ]
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "figure4_legacy_vs_formal_comparison.csv", index=False)
    formatted = table.copy()
    numeric = formatted.select_dtypes(include="number").columns
    formatted[numeric] = formatted[numeric].map(
        lambda value: f"{value:.4f}" if pd.notna(value) else ""
    )
    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows_md = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in formatted.itertuples(index=False, name=None)
    ]
    (args.output_dir / "figure4_legacy_vs_formal_comparison.md").write_text(
        "\n".join([header, separator, *rows_md]) + "\n"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
