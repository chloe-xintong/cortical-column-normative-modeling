#!/usr/bin/env python3
"""Inventory subjects with complete O2-Column, O2-FS, and O3 fsaverage data.

Description
-----------
Inventory subjects with complete O2-Column, O2-FS, and O3 fsaverage data.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--subject-list`` command-line input
- ``--o3-root`` command-line input
- ``--o2-column-root`` command-line input
- ``--o2-fs-root`` command-line input
- ``--output`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python qc/build_method_comparison_manifest.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-list", type=Path, required=True)
    parser.add_argument("--o3-root", type=Path, required=True)
    parser.add_argument("--o2-column-root", type=Path, required=True)
    parser.add_argument("--o2-fs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects = [
        line.strip()
        for line in args.subject_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    rows = []
    for subject_id in subjects:
        required = []
        for hemisphere in ("lh", "rh"):
            required.extend(
                [
                    args.o3_root
                    / subject_id
                    / "profiles_before_clean"
                    / f"{subject_id}_{hemisphere}_md_profiles.npz",
                    args.o2_column_root
                    / subject_id
                    / f"{subject_id}_{hemisphere}_profiles.npz",
                    args.o2_fs_root
                    / subject_id
                    / "fsaverage"
                    / hemisphere
                    / "vol2surf_manifest.json",
                ]
            )
            required.extend(
                args.o2_fs_root
                / subject_id
                / "fsaverage"
                / hemisphere
                / f"{subject_id}_{hemisphere}_md_depth-{depth:.2f}.mgh"
                for depth in [index / 20 for index in range(21)]
            )
        missing = [str(path) for path in required if not path.is_file()]
        rows.append(
            {
                "subject_id": subject_id,
                "method_comparison_available": not missing,
                "missing_file_count": len(missing),
                "missing_files": ";".join(missing),
            }
        )
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table["method_comparison_available"].value_counts().to_string())


if __name__ == "__main__":
    main()
