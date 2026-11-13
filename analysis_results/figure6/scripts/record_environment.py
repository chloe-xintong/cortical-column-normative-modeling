#!/usr/bin/env python3
"""Record the exact runtime and installed distribution versions.

Description
-----------
Record the exact runtime and installed distribution versions.

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
python analysis_results/figure6/scripts/record_environment.py --help
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    output = Path(sys.argv[1])
    wheel = Path(sys.argv[2])
    distributions = sorted(
        (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    )
    (output / "environment.lock.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in distributions) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    payload = {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "pcntoolkit_version": importlib.metadata.version("pcntoolkit"),
        "pcntoolkit_official_wheel_cache_source": str(wheel),
        "pcntoolkit_wheel_sha256": digest,
        "shared_environment": str(Path(sys.prefix).resolve()),
        "compute_node_access": "Environment resides under the shared repository filesystem and was used by Slurm jobs 345854 and 345855.",
    }
    (output / "environment_runtime.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    rows = []
    for path in sorted(x for x in output.rglob("*") if x.is_file() and x.name != "artifact_manifest.csv"):
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"relative_path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                     "sha256": file_digest})
    pd.DataFrame(rows).to_csv(output / "artifact_manifest.csv", index=False)


if __name__ == "__main__":
    main()
