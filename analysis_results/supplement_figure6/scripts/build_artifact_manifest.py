#!/usr/bin/env python3
"""Write a complete relative-path and SHA-256 manifest for pilot artifacts.

Description
-----------
Write a complete relative-path and SHA-256 manifest for pilot artifacts.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/build_artifact_manifest.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pilot_utils import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.output_dir / "artifact_manifest.csv"
    rows: list[dict[str, object]] = []
    for path in sorted(args.output_dir.rglob("*")):
        relative = path.relative_to(args.output_dir)
        if "__pycache__" in relative.parts:
            continue
        if path.is_dir():
            if not any(path.iterdir()):
                rows.append({"relative_path": str(relative) + "/", "type": "directory", "bytes": 0, "sha256": ""})
            continue
        if path == manifest:
            continue
        rows.append({"relative_path": str(relative), "type": "file", "bytes": path.stat().st_size,
                     "sha256": sha256(path)})
    rows.append({"relative_path": "artifact_manifest.csv", "type": "file", "bytes": "self",
                 "sha256": "self_not_hashed_to_avoid_recursion"})
    pd.DataFrame(rows).sort_values("relative_path").to_csv(manifest, index=False)


if __name__ == "__main__":
    main()
