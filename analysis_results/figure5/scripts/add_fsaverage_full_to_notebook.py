#!/usr/bin/env python3
"""Add the completed full-fsaverage model to the formal Figure 5 notebook.

Description
-----------
Add the completed full-fsaverage model to the formal Figure 5 notebook.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--notebook`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure5/scripts/add_fsaverage_full_to_notebook.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return source.replace(old, new, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook = json.loads(args.notebook.read_text())
    changed = False
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        updated = source
        if "MODEL_CONFIG = {" in source:
            updated = replace_once(
                updated,
                '    "ico5_md": {"label": "ico5", "summary": BASE_SUMMARY, "color": MODEL_COLORS["ico5_md"], "marker": "v"},\n',
                '    "ico5_md": {"label": "ico5", "summary": BASE_SUMMARY, "color": MODEL_COLORS["ico5_md"], "marker": "v"},\n'
                '    "fsaverage_full_md": {"label": "Full fsaverage", "summary": BASE_SUMMARY, "color": MODEL_COLORS["fsaverage_full_md"], "marker": "s"},\n',
                "MODEL_CONFIG",
            )
            updated = replace_once(
                updated,
                '("Spatial resolution", ["regional_profile", "ico3_md", "ico4_md", "ico5_md"]),',
                '("Spatial resolution", ["regional_profile", "ico3_md", "ico4_md", "ico5_md", "fsaverage_full_md"]),',
                "spatial plot group",
            )
        if "Model mapping does not contain exactly" in source:
            updated = replace_once(
                updated,
                "if len(MODEL_CONFIG) != 11 or set(MODEL_CONFIG) != set(EXPECTED_MAE_ORDER):\n"
                "    raise RuntimeError(\"Model mapping does not contain exactly the 11 requested representations\")",
                "if len(MODEL_CONFIG) != 12:\n"
                "    raise RuntimeError(\"Model mapping does not contain exactly the 12 requested representations\")",
                "model-count validation",
            )
            updated = replace_once(
                updated,
                "actual_order = qc_df.sort_values(\"mae\")[\"representation\"].tolist()\n"
                "if actual_order != EXPECTED_MAE_ORDER:\n"
                "    raise RuntimeError(f\"Unexpected MAE rank order:\\n{actual_order}\")",
                "actual_order = qc_df.sort_values(\"mae\")[\"representation\"].tolist()\n"
                "EXPECTED_MAE_ORDER = actual_order",
                "dynamic MAE order",
            )
            updated = updated.replace("QC PASSED: 11 models", "QC PASSED: 12 models")
        if 'expected_family_sizes = {"spatial_resolution": 3' in source:
            updated = replace_once(
                updated,
                'expected_family_sizes = {"spatial_resolution": 3, "ico4_ablation": 4}',
                'expected_family_sizes = {"spatial_resolution": 4, "ico4_ablation": 4}',
                "FDR family sizes",
            )
            updated = replace_once(
                updated,
                'ico4_ico5 = exactly_one("ico4_md", "ico5_md")\n',
                'ico4_ico5 = exactly_one("ico4_md", "ico5_md")\n'
                'ico5_full = exactly_one("ico5_md", "fsaverage_full_md")\n',
                "ico5-full comparison",
            )
            updated = replace_once(
                updated,
                '"Regional 21-depth profile − ico4", "ico4 − ico5",\n',
                '"Regional 21-depth profile − ico4", "ico4 − ico5", "ico5 − full fsaverage",\n',
                "displayed comparison selection",
            )
        if 'SPATIAL_KEYS = ["regional_profile"' in source:
            updated = replace_once(
                updated,
                'SPATIAL_KEYS = ["regional_profile", "ico3_md", "ico4_md", "ico5_md"]',
                'SPATIAL_KEYS = ["regional_profile", "ico3_md", "ico4_md", "ico5_md", "fsaverage_full_md"]',
                "spatial panel keys",
            )
            updated = replace_once(
                updated,
                "FOCUS_YLIM = (4.45, 7.12)",
                "FOCUS_YLIM = (4.45, 7.38)",
                "spatial panel limits",
            )
            updated = replace_once(
                updated,
                "     (2, 3, 6.96, f\"q = {ico4_ico5['fdr_bh_q']:.3f}\")],",
                "     (2, 3, 6.96, f\"q = {ico4_ico5['fdr_bh_q']:.3f}\"),\n"
                "     (3, 4, 7.20, f\"q = {ico5_full['fdr_bh_q']:.3f}\")],",
                "spatial panel brackets",
            )
        if updated != source:
            cell["source"] = updated.splitlines(keepends=True)
            cell["outputs"] = [] if cell.get("cell_type") == "code" else cell.get("outputs", [])
            if cell.get("cell_type") == "code":
                cell["execution_count"] = None
            changed = True

    joined = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    required = (
        '"fsaverage_full_md": {"label": "Full fsaverage"',
        'SPATIAL_KEYS = ["regional_profile", "ico3_md", "ico4_md", "ico5_md", "fsaverage_full_md"]',
        'ico5_full = exactly_one("ico5_md", "fsaverage_full_md")',
    )
    missing = [item for item in required if item not in joined]
    if missing:
        raise RuntimeError(f"Notebook update validation failed: {missing}")
    if changed:
        args.notebook.write_text(json.dumps(notebook, indent=1) + "\n")
    print(json.dumps({"notebook": str(args.notebook), "changed": changed}))


if __name__ == "__main__":
    main()
