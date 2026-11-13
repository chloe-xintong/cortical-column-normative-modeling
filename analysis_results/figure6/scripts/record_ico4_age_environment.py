#!/usr/bin/env python3
"""Record the lightweight runtime lock for the completed ico4 age-conditioned analysis.

Description
-----------
Record the lightweight runtime lock for the completed ico4 age-conditioned analysis.

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
python analysis_results/figure6/scripts/record_ico4_age_environment.py --help
"""

import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main() -> None:
    output = Path(sys.argv[1])
    distributions = sorted(
        (dist.metadata["Name"], dist.version)
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    )
    (output / "environment.lock.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in distributions) + "\n",
        encoding="utf-8",
    )
    payload = {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "pcntoolkit_version": importlib.metadata.version("pcntoolkit"),
        "shared_environment": str(Path(sys.prefix).resolve()),
        "slurm_full_cohort_arrays": [357612, 357633, 357634, 357635, 357636],
        "slurm_full_cohort_validation": 358654,
        "compute_node_access": "The shared repository environment was used directly by all full-cohort Slurm jobs."
    }
    (output / "environment_runtime.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
