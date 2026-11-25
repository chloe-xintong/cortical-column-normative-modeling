#!/usr/bin/env python3
"""Run the frozen Gaussian M0-M3 workflow and canonicalize density loss as MLL.

Description
-----------
Run the frozen Gaussian M0-M3 workflow and canonicalize density loss as MLL.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- No required command-line inputs; optional inputs are defined by the script.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/run_gaussian_models.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ORIGINAL_SCRIPTS = Path(__file__).resolve().parents[2] / "figure6" / "scripts"
sys.path.insert(0, str(ORIGINAL_SCRIPTS))
from run_models import main as run_models  # noqa: E402


def output_dir_from_argv() -> Path:
    position = sys.argv.index("--output-dir")
    return Path(sys.argv[position + 1])


def canonicalize_mll(output_dir: Path) -> None:
    for path in sorted((output_dir / "shards").glob("predictions_*.parquet")):
        frame = pd.read_parquet(path)
        if "nlpd" in frame and "mean_log_loss" not in frame:
            frame = frame.rename(columns={"nlpd": "mean_log_loss"})
            frame.to_parquet(path, index=False)


def main() -> None:
    output_dir = output_dir_from_argv()
    if "--canonicalize-only" in sys.argv:
        canonicalize_mll(output_dir)
        return
    run_models()
    canonicalize_mll(output_dir)


if __name__ == "__main__":
    main()
