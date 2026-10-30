#!/usr/bin/env python3
"""Write a final machine-readable status report without running notebooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--option3-summary-dir", type=Path, required=True)
    parser.add_argument("--figure4-real-summary-dir", type=Path, required=True)
    parser.add_argument("--figure4-null-summary-dir", type=Path, required=True)
    parser.add_argument("--method-comparison-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def status_counts(path: Path) -> dict[str, int]:
    table = pd.read_csv(path)
    counts = table["status"].value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    method_manifest = pd.read_csv(args.method_comparison_manifest)
    report = {
        "n_requested": len(manifest),
        "n_input_complete": int(manifest["input_complete"].sum()),
        "n_method_comparison_available": int(
            method_manifest["method_comparison_available"].sum()
        ),
        "option3_qc": status_counts(
            args.option3_summary_dir / "full_cohort_status.csv"
        ),
        "figure4_real": status_counts(
            args.figure4_real_summary_dir / "figure4_real_status.csv"
        ),
        "figure4_null": status_counts(
            args.figure4_null_summary_dir / "figure4_null_status.csv"
        ),
        "notebooks_executed": False,
        "notebook_instruction": (
            "Summary notebooks are intentionally not executed by SLURM. "
            "The investigator will run them manually after reviewing status tables."
        ),
    }
    (args.output_dir / "full_cohort_execution_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"analysis": analysis, "status": status, "n_subjects": count}
            for analysis in ("option3_qc", "figure4_real", "figure4_null")
            for status, count in report[analysis].items()
        ]
    ).to_csv(args.output_dir / "full_cohort_execution_report.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
