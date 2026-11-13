#!/usr/bin/env python3
"""Append D0128 to the frozen Figure 7 inputs for a separate sensitivity run.

Description
-----------
Append D0128 to the frozen Figure 7 inputs for a separate sensitivity run.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--primary-input-dir`` command-line input
- ``--cohort-manifest`` command-line input
- ``--clinical-metadata`` command-line input
- ``--coverage-audit`` command-line input
- ``--node-metadata`` command-line input
- ``--profile-root`` command-line input
- ``--freesurfer-root`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure7/scripts/prepare_d0128_sensitivity.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd


SUBJECT = "D0128"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-input-dir", type=Path, required=True)
    p.add_argument("--cohort-manifest", type=Path, required=True)
    p.add_argument("--clinical-metadata", type=Path, required=True)
    p.add_argument("--coverage-audit", type=Path, required=True)
    p.add_argument("--node-metadata", type=Path, required=True)
    p.add_argument("--profile-root", type=Path, required=True)
    p.add_argument("--freesurfer-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    roster = pd.read_csv(a.primary_input_dir / "final_analysis_roster.csv", dtype={"subject_id": str})
    if len(roster) != 111 or SUBJECT in set(roster.subject_id):
        raise RuntimeError("Expected the frozen N=111 primary roster without D0128")

    manifest = pd.read_csv(a.cohort_manifest, dtype={"subject_id": str, "PTID_clean": str, "visit_id": str})
    source = manifest.loc[manifest.subject_id.eq(SUBJECT)].copy()
    if len(source) != 1:
        raise RuntimeError("D0128 is not unique in the cohort manifest")
    clinical = pd.read_csv(a.clinical_metadata, low_memory=False, dtype={"PTID_clean": str, "VISITNUM": str})
    clinical = clinical.loc[
        clinical.PTID_clean.eq(source.PTID_clean.iloc[0])
        & clinical.VISITNUM.eq(source.visit_id.iloc[0])
    ].copy()
    if len(clinical) != 1:
        raise RuntimeError("D0128 matched clinical visit is not unique")
    coverage = pd.read_csv(a.coverage_audit, dtype={"subject_id": str})
    coverage = coverage.loc[coverage.subject_id.eq(SUBJECT)].copy()
    if len(coverage) != 1:
        raise RuntimeError("D0128 coverage record is not unique")
    if "coverage_fraction" not in coverage and "recomputed_bilateral_coverage_fraction" in coverage:
        coverage["coverage_fraction"] = coverage["recomputed_bilateral_coverage_fraction"]

    new = {column: np.nan for column in roster.columns}
    for column in source.columns:
        if column in new:
            new[column] = source.iloc[0][column]
    for column in clinical.columns:
        if column in new:
            new[column] = clinical.iloc[0][column]
    for column in coverage.columns:
        if column in new:
            new[column] = coverage.iloc[0][column]
    new.update({
        "analysis_row_index": len(roster),
        "subject_id": SUBJECT,
        "age_eligible_41_78": True,
        "model_role": "non_normal_target",
        "figure7_included": True,
        "inclusion_exclusion_reason": "sensitivity_include_registration_failure",
        "outer_fold": np.nan,
        "model_id": "figure6_ico4_warped_m2_outer_fold_1_zero_based",
        "global_cognition": float(clinical.Global_Cognition_Composite.iloc[0]),
        "moca": float(clinical.MOCATOTS.iloc[0]),
        "moca_88_treated_missing": False,
        "coverage_ge_0_98": bool(float(coverage.coverage_fraction.iloc[0]) >= 0.98),
    })
    if new["diagnosis_at_scan"] != "AD" or not 41 <= float(new["age_at_scan"]) <= 78:
        raise RuntimeError("D0128 is not an age-eligible AD participant")
    sensitivity_roster = pd.concat([roster, pd.DataFrame([new])], ignore_index=True)
    sensitivity_roster.to_csv(a.output_dir / "final_analysis_roster.csv", index=False)

    old_md = np.load(a.primary_input_dir / "X_figure7_ico4_md.npy", mmap_mode="r")
    old_thickness = np.load(
        a.primary_input_dir / "X_figure7_local_freesurfer_thickness.npy", mmap_mode="r"
    )
    node_meta = pd.read_csv(a.node_metadata)
    md_parts, thickness_parts = [], []
    mapping_qc = []
    for hemi in ("lh", "rh"):
        nodes = node_meta.loc[node_meta.hemisphere.eq(hemi)]
        ids = nodes.fsaverage_vertex_id.to_numpy(int)
        profile_path = a.profile_root / SUBJECT / "profiles" / f"{SUBJECT}_{hemi}_md_profiles.npz"
        with np.load(profile_path, allow_pickle=False) as profile:
            profiles = profile["profiles"][ids].astype(float)
            column_valid = profile["column_valid"][ids].astype(bool)
            triangles = profile["native_triangle_vertex_ids"][ids].astype(int)
            weights = profile["barycentric_weights"][ids].astype(float)
            mapping_valid = profile["mapping_valid"][ids].astype(bool)
        profiles[~column_valid] = np.nan
        md_parts.append(profiles)
        native = fsio.read_morph_data(a.freesurfer_root / SUBJECT / "surf" / f"{hemi}.thickness").astype(float)
        valid = mapping_valid & np.all(triangles >= 0, axis=1) & np.all(triangles < len(native), axis=1)
        valid &= np.all(np.isfinite(weights), axis=1) & np.isclose(weights.sum(1), 1.0, atol=1e-5)
        values = np.full(len(ids), np.nan)
        values[valid] = np.sum(native[triangles[valid]] * weights[valid], axis=1)
        values[~np.isfinite(values) | (values <= 0)] = np.nan
        thickness_parts.append(values)
        mapping_qc.append({"subject_id": SUBJECT, "hemisphere": hemi,
                           "n_valid_md_nodes": int(column_valid.sum()),
                           "n_valid_thickness": int(np.isfinite(values).sum())})

    new_md = np.concatenate(md_parts, axis=0).reshape(-1).astype(np.float32)
    new_thickness = np.concatenate(thickness_parts).astype(np.float32)
    np.save(a.output_dir / "X_figure7_ico4_md.npy", np.vstack([old_md, new_md]))
    np.save(a.output_dir / "X_figure7_local_freesurfer_thickness.npy",
            np.vstack([old_thickness, new_thickness]))
    pd.DataFrame(mapping_qc).to_csv(a.output_dir / "d0128_input_mapping_qc.csv", index=False)
    (a.output_dir / "sensitivity_design.json").write_text(json.dumps({
        "status": "prepared", "primary_n": 111, "sensitivity_n": 112,
        "added_subject": SUBJECT, "added_group": "AD",
        "age_range_inclusive": [41, 78],
        "registration_qc": "failed_visual_T1_DWI_registration",
        "purpose": "Figure 7A sensitivity only; does not replace the primary analysis",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
