#!/usr/bin/env python3
"""Run the two-subject automatic gate before full-cohort submission.

Description
-----------
Run the two-subject automatic gate before full-cohort submission.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--subjects-dir`` command-line input
- ``--output-root`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python qc/full_cohort/scripts/run_full_cohort_preflight.py --help
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    complete_manifest = manifest.loc[manifest["input_complete"]].reset_index(drop=True)
    if len(complete_manifest) < 2:
        raise ValueError("Preflight requires at least two input-complete subjects")
    preflight_manifest = args.output_root / "preflight_two_subjects.csv"
    args.output_root.mkdir(parents=True, exist_ok=True)
    complete_manifest.iloc[:2].to_csv(preflight_manifest, index=False)

    option3_root = args.output_root / "option3_qc"
    pair_dir = args.output_root / "pairs"
    real_root = args.output_root / "figure4_real"
    null_root = args.output_root / "figure4_null"

    run(
        [
            sys.executable,
            str(project / "analysis_results/figure4/scripts/build_geodesic_pair_tables.py"),
            "--subjects-dir",
            str(args.subjects_dir),
            "--output-dir",
            str(pair_dir),
            "--anchors-per-hemisphere",
            "64",
            "--max-neighbors-per-anchor-bin",
            "5",
            "--batch-size",
            "16",
            "--seed",
            "20261020",
        ]
    )
    for index in range(2):
        run(
            [
                sys.executable,
                str(project / "qc/option3/scripts/compute_option3_qc.py"),
                "--manifest",
                str(preflight_manifest),
                "--index",
                str(index),
                "--output-root",
                str(option3_root),
                "--subjects-dir",
                str(args.subjects_dir),
            ]
        )
        run(
            [
                sys.executable,
                str(
                    project
                    / "analysis_results/figure4/scripts/compute_real_spatial_curves.py"
                ),
                "--manifest",
                str(preflight_manifest),
                "--index",
                str(index),
                "--pair-dir",
                str(pair_dir),
                "--output-root",
                str(real_root),
            ]
        )
        run(
            [
                sys.executable,
                str(project / "analysis_results/figure4/scripts/compute_spatial_null.py"),
                "--manifest",
                str(preflight_manifest),
                "--index",
                str(index),
                "--pair-dir",
                str(pair_dir),
                "--output-root",
                str(null_root),
                "--null-method",
                "interpolation_only",
                "--replicates",
                "1",
                "--psf-fwhm-mm",
                "0",
            ]
        )

    pair_counts = pd.read_csv(pair_dir / "figure4_pair_counts.csv")
    expected_bins = {f"{start}-{start + 2}" for start in range(0, 30, 2)}
    observed_bins = set(
        pd.read_parquet(pair_dir / "figure4_geodesic_pairs_all.parquet")[
            "distance_bin"
        ].unique()
    )
    if not expected_bins.issubset(observed_bins):
        missing = sorted(expected_bins - observed_bins)
        raise RuntimeError(f"Preflight pair table is missing distance bins: {missing}")
    if pair_counts.empty:
        raise RuntimeError("Preflight boundary pair table is empty")

    run(
        [
            sys.executable,
            str(project / "analysis_results/figure4/scripts/fit_correlation_lengths.py"),
            "--manifest",
            str(preflight_manifest),
            "--results-root",
            str(real_root),
            "--output-dir",
            str(real_root / "cohort"),
        ]
    )
    run(
        [
            sys.executable,
            str(project / "analysis_results/figure4/scripts/summarize_spatial_null.py"),
            "--manifest",
            str(preflight_manifest),
            "--results-root",
            str(null_root),
            "--output-dir",
            str(null_root / "cohort"),
        ]
    )
    (args.output_root / "preflight_completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "subjects": complete_manifest.iloc[:2]["subject_id"].tolist(),
                "notebooks_executed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
