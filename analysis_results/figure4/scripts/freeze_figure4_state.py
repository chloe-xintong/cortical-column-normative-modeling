#!/usr/bin/env python3
"""Record immutable identifiers for the Figure 4 real/pair inputs used by nulls.

Description
-----------
Record immutable identifiers for the Figure 4 real/pair inputs used by nulls.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--pair-dir`` command-line input
- ``--real-cohort-dir`` command-line input
- ``--output`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/freeze_figure4_state.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--real-cohort-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pair_files = [
        args.pair_dir / "figure4_geodesic_pairs_all.parquet",
        args.pair_dir / "figure4_geodesic_pairs_boundary.parquet",
        args.pair_dir / "pair_table_completion.json",
    ]
    output_files = [
        args.real_cohort_dir / "figure4_real_group_summary.csv",
        args.real_cohort_dir / "figure4_correlation_lengths.csv",
        args.real_cohort_dir / "figure4_within_across_matched_group_summary.csv",
        args.real_cohort_dir / "figure4_fixed_split_half_fits.csv",
    ]
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    all_pairs = pd.read_parquet(pair_files[0], columns=["hemisphere", "anchor_vertex", "distance_bin_index"])
    record = {
        "description": "Frozen Figure 4 A/C/D and pair-table inputs for null comparison",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
        "git_worktree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip()),
        "n_subjects": int(len(manifest)), "subjects": manifest.subject_id.tolist(),
        "anchor_set_id": "current_2048_per_hemisphere",
        "anchors_per_hemisphere": all_pairs.groupby("hemisphere").anchor_vertex.nunique().astype(int).to_dict(),
        "anchor_selection_seed": 20261020,
        "distance_bins_mm": list(range(0, 32, 2)),
        "files": {str(path): sha256(path) for path in pair_files + output_files if path.is_file()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
