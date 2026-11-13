#!/usr/bin/env python3
"""Validate surface-resampling profiles and summarize each subject output.

Description
-----------
Validate surface-resampling profiles and summarize each subject output.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--profiles`` command-line input
- ``--template`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python qc/archive/surface_resampling/validate_outputs.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(path: Path, template: str, subject: str, hemisphere: str) -> dict[str, object]:
    started = time.time()
    data = np.load(path)
    profiles = np.asarray(data["profiles"], dtype=float)
    cortex = np.asarray(data["cortex_mask"], dtype=bool)
    valid = np.asarray(data["valid_mask"], dtype=bool)
    values = profiles[cortex]
    row: dict[str, object] = {
        "subject_id": subject,
        "template": template,
        "hemisphere": hemisphere,
        "total_vertices": profiles.shape[0],
        "cortex_vertices": int(cortex.sum()),
        "finite_value_pct": float(np.isfinite(values).mean() * 100),
        "nan_pct": float(np.isnan(values).mean() * 100),
        "zero_pct": float((values == 0).mean() * 100),
        "all_depths_valid_pct": float(valid[cortex].mean() * 100),
        "file_size_bytes": path.stat().st_size,
        "elapsed_s": time.time() - started,
    }
    for index, depth_values in enumerate(values.T):
        finite = depth_values[np.isfinite(depth_values)]
        prefix = f"depth_{index:02d}"
        row[f"{prefix}_min"] = float(np.min(finite)) if finite.size else np.nan
        row[f"{prefix}_p01"] = float(np.percentile(finite, 1)) if finite.size else np.nan
        row[f"{prefix}_median"] = float(np.median(finite)) if finite.size else np.nan
        row[f"{prefix}_p99"] = float(np.percentile(finite, 99)) if finite.size else np.nan
        row[f"{prefix}_max"] = float(np.max(finite)) if finite.size else np.nan
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = []
    for index, path in enumerate(args.profiles, start=1):
        name = path.stem
        subject, hemisphere = name.split("_")[:2]
        rows.append(summarize(path, args.template, subject, hemisphere))
        logging.info("[%d/%d] Validated %s", index, len(args.profiles), path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "subject_qc.csv", index=False)
    (args.output_dir / "subject_qc.json").write_text(table.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
