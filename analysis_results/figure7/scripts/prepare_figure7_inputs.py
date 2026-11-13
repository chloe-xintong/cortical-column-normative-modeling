#!/usr/bin/env python3
"""Freeze the Figure 7 cohort and build ico4 MD/local-thickness inputs.

Description
-----------
Freeze the Figure 7 cohort and build ico4 MD/local-thickness inputs.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--coverage-audit`` command-line input
- ``--cohort-manifest`` command-line input
- ``--clinical-metadata`` command-line input
- ``--folds`` command-line input
- ``--node-metadata`` command-line input
- ``--profile-root`` command-line input
- ``--freesurfer-root`` command-line input
- ``--figure6-subject-order`` command-line input
- ``--figure6-md`` command-line input
- ``--figure6-thickness`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure7/scripts/prepare_figure7_inputs.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd


AGE_MIN, AGE_MAX = 41.0, 78.0
NON_NORMAL = {"MCI", "AD", "Other/Impaired non-demented", "Unknown"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coverage-audit", type=Path, required=True)
    p.add_argument("--cohort-manifest", type=Path, required=True)
    p.add_argument("--clinical-metadata", type=Path, required=True)
    p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--node-metadata", type=Path, required=True)
    p.add_argument("--profile-root", type=Path, required=True)
    p.add_argument("--freesurfer-root", type=Path, required=True)
    p.add_argument("--figure6-subject-order", type=Path, required=True)
    p.add_argument("--figure6-md", type=Path, required=True)
    p.add_argument("--figure6-thickness", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--outer-fold", type=int, default=1)
    return p.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    coverage = pd.read_csv(a.coverage_audit, dtype={"subject_id": str})
    coverage = coverage.loc[coverage.passes_minimum_coverage].copy()
    if len(coverage) != 313 or coverage.subject_id.nunique() != 313:
        raise RuntimeError("Formal coverage>=95% source pool is not N=313")
    cohort = pd.read_csv(a.cohort_manifest, dtype={"subject_id": str, "PTID_clean": str,
                                                   "visit_id": str})
    clinical = pd.read_csv(a.clinical_metadata, low_memory=False,
                           dtype={"PTID_clean": str, "VISITNUM": str})
    clinical_columns = ["PTID_clean", "VISITNUM", "VISIT_DATE", "MOCATOTS",
                        "Global_Cognition_Composite", "DX_Label_harmonized"]
    if clinical.duplicated(["PTID_clean", "VISITNUM"]).any():
        raise RuntimeError("Clinical PTID/visit key is not unique")
    cohort = cohort.merge(
        clinical[clinical_columns], left_on=["PTID_clean", "visit_id"],
        right_on=["PTID_clean", "VISITNUM"], how="left", validate="one_to_one",
    )
    pool = coverage.merge(cohort, on="subject_id", how="left", validate="one_to_one")
    if pool[["age_at_scan", "diagnosis_at_scan", "SUBJECT_SEX"]].isna().any().any():
        raise RuntimeError("Formal N=313 pool has missing age, diagnosis, or sex")

    folds = json.loads(a.folds.read_text(encoding="utf-8"))
    if folds.get("n_splits") != 5 or sorted(int(x["outer_fold"]) for x in folds["folds"]) != list(range(5)):
        raise RuntimeError("Expected explicit zero-based outer folds 0..4")
    fold = next(x for x in folds["folds"] if int(x["outer_fold"]) == a.outer_fold)
    train_cn, test_cn = set(fold["train_subject_ids"]), set(fold["test_subject_ids"])
    if len(train_cn) != 148 or len(test_cn) != 38:
        raise RuntimeError("Selected Fold 1 does not have expected 148/38 CN split")
    if not test_cn.issubset(set(pool.subject_id)):
        raise RuntimeError("At least one Fold 1 held-out CN is absent from formal N=313 pool")

    pool["age_eligible_41_78"] = pool.age_at_scan.between(AGE_MIN, AGE_MAX, inclusive="both")
    pool["model_role"] = "non_normal_target"
    pool.loc[pool.subject_id.isin(train_cn), "model_role"] = "fold1_training_cn"
    pool.loc[pool.subject_id.isin(test_cn), "model_role"] = "fold1_heldout_cn"
    other_normal = pool.diagnosis_at_scan.eq("Normal") & ~pool.subject_id.isin(train_cn | test_cn)
    pool.loc[other_normal, "model_role"] = "other_normal"
    pool["figure7_included"] = False
    pool["inclusion_exclusion_reason"] = ""
    outside = ~pool.age_eligible_41_78
    pool.loc[outside, "inclusion_exclusion_reason"] = "outside_age_41_78"
    pool.loc[pool.age_eligible_41_78 & pool.model_role.eq("fold1_training_cn"),
             "inclusion_exclusion_reason"] = "fold1_training_cn_excluded"
    pool.loc[pool.age_eligible_41_78 & pool.model_role.eq("other_normal"),
             "inclusion_exclusion_reason"] = "other_normal_excluded"
    included = pool.age_eligible_41_78 & (
        pool.model_role.eq("fold1_heldout_cn") |
        (pool.model_role.eq("non_normal_target") & pool.diagnosis_at_scan.isin(NON_NORMAL))
    )
    pool.loc[included, "figure7_included"] = True
    pool.loc[included, "inclusion_exclusion_reason"] = "included"
    unresolved = pool.inclusion_exclusion_reason.eq("")
    pool.loc[unresolved, "inclusion_exclusion_reason"] = "not_in_prespecified_diagnostic_scope"

    pool["outer_fold"] = np.where(pool.subject_id.isin(test_cn), a.outer_fold, np.nan)
    pool["model_id"] = "figure6_ico4_warped_m2_outer_fold_1_zero_based"
    pool["global_cognition"] = pd.to_numeric(pool.Global_Cognition_Composite, errors="coerce")
    raw_moca = pd.to_numeric(pool.MOCATOTS, errors="coerce")
    pool["moca"] = raw_moca.mask(raw_moca.eq(88))
    pool["moca_88_treated_missing"] = raw_moca.eq(88)
    pool["coverage_ge_0_98"] = pool.coverage_fraction.ge(0.98)
    pool.to_csv(a.output_dir / "cohort_flow_coverage95_n313.csv", index=False)

    analysis = pool.loc[pool.figure7_included].copy().sort_values("subject_id").reset_index(drop=True)
    analysis.insert(0, "analysis_row_index", np.arange(len(analysis), dtype=int))
    counts = analysis.diagnosis_at_scan.value_counts().to_dict()
    expected = {"Normal": 38, "MCI": 19, "AD": 6,
                "Other/Impaired non-demented": 22, "Unknown": 26}
    if counts != expected or len(analysis) != 111:
        raise RuntimeError(f"Unexpected final Figure 7 cohort: {counts}")
    if analysis.loc[analysis.diagnosis_at_scan.eq("Normal"), "subject_id"].pipe(set) != test_cn:
        raise RuntimeError("Figure 7 Normal group is not exactly Fold 1 held-out CN")
    if set(analysis.subject_id) & train_cn:
        raise RuntimeError("Fold 1 training CN leaked into Figure 7")
    analysis.to_csv(a.output_dir / "final_analysis_roster.csv", index=False)

    node_meta = pd.read_csv(a.node_metadata)
    if len(node_meta) != 4687 or not np.array_equal(node_meta.node_index, np.arange(4687)):
        raise RuntimeError("Unexpected fixed ico4 node mask/order")
    md_path = a.output_dir / "X_figure7_ico4_md.npy"
    thickness_path = a.output_dir / "X_figure7_local_freesurfer_thickness.npy"
    md = np.lib.format.open_memmap(md_path, mode="w+", dtype=np.float32,
                                  shape=(len(analysis), 4687 * 21))
    thickness = np.lib.format.open_memmap(thickness_path, mode="w+", dtype=np.float32,
                                         shape=(len(analysis), 4687))
    md[:] = np.nan
    thickness[:] = np.nan
    qc_rows = []
    for row in analysis.itertuples(index=False):
        md_parts, thickness_parts = [], []
        for hemi in ("lh", "rh"):
            nodes = node_meta.loc[node_meta.hemisphere.eq(hemi)]
            ids = nodes.fsaverage_vertex_id.to_numpy(int)
            profile_path = a.profile_root / row.subject_id / "profiles" / f"{row.subject_id}_{hemi}_md_profiles.npz"
            with np.load(profile_path, allow_pickle=False) as profile:
                vertex_ids = profile["fsaverage_vertex_id"].astype(int)
                profiles = profile["profiles"][ids].astype(float)
                column_valid = profile["column_valid"][ids].astype(bool)
                triangles = profile["native_triangle_vertex_ids"][ids].astype(int)
                weights = profile["barycentric_weights"][ids].astype(float)
                mapping_valid = profile["mapping_valid"][ids].astype(bool)
            if not np.array_equal(vertex_ids, np.arange(163842)) or profiles.shape != (len(ids), 21):
                raise RuntimeError(f"Unstable profile order/shape for {row.subject_id} {hemi}")
            profiles[~column_valid] = np.nan
            md_parts.append(profiles)
            native = fsio.read_morph_data(
                a.freesurfer_root / row.subject_id / "surf" / f"{hemi}.thickness"
            ).astype(float)
            valid = mapping_valid & np.all(triangles >= 0, axis=1) & np.all(triangles < len(native), axis=1)
            valid &= np.all(np.isfinite(weights), axis=1) & np.isclose(weights.sum(1), 1.0, atol=1e-5)
            values = np.full(len(ids), np.nan)
            values[valid] = np.sum(native[triangles[valid]] * weights[valid], axis=1)
            values[~np.isfinite(values) | (values <= 0)] = np.nan
            thickness_parts.append(values)
            qc_rows.append({"subject_id": row.subject_id, "analysis_row_index": row.analysis_row_index,
                            "hemisphere": hemi, "n_nodes": len(ids),
                            "n_valid_md_nodes": int(column_valid.sum()),
                            "n_valid_thickness": int(np.isfinite(values).sum())})
        md[row.analysis_row_index] = np.concatenate(md_parts, axis=0).reshape(-1).astype(np.float32)
        thickness[row.analysis_row_index] = np.concatenate(thickness_parts).astype(np.float32)
        if (row.analysis_row_index + 1) % 10 == 0 or row.analysis_row_index + 1 == len(analysis):
            md.flush(); thickness.flush()
            print(json.dumps({"prepared_subjects": row.analysis_row_index + 1,
                              "total": len(analysis)}), flush=True)
    md.flush(); thickness.flush()
    pd.DataFrame(qc_rows).to_csv(a.output_dir / "input_mapping_qc.csv", index=False)

    source_order = pd.read_csv(a.figure6_subject_order, dtype={"subject_id": str})
    source_md = np.load(a.figure6_md, mmap_mode="r")
    source_thickness = np.load(a.figure6_thickness, mmap_mode="r")
    checks = []
    for row in analysis.loc[analysis.model_role.eq("fold1_heldout_cn")].itertuples(index=False):
        source_row = int(source_order.loc[source_order.subject_id.eq(row.subject_id), "row_index"].iloc[0])
        md_error = float(np.nanmax(np.abs(np.asarray(md[row.analysis_row_index], float) -
                                          np.asarray(source_md[source_row], float))))
        thickness_error = float(np.nanmax(np.abs(np.asarray(thickness[row.analysis_row_index], float) -
                                                 np.asarray(source_thickness[source_row], float))))
        nan_md_equal = bool(np.array_equal(np.isnan(md[row.analysis_row_index]), np.isnan(source_md[source_row])))
        nan_t_equal = bool(np.array_equal(np.isnan(thickness[row.analysis_row_index]),
                                          np.isnan(source_thickness[source_row])))
        checks.append({"subject_id": row.subject_id, "max_abs_md_error": md_error,
                       "max_abs_thickness_error": thickness_error,
                       "md_nan_pattern_equal": nan_md_equal,
                       "thickness_nan_pattern_equal": nan_t_equal})
    check = pd.DataFrame(checks)
    check.to_csv(a.output_dir / "heldout_cn_input_reproduction.csv", index=False)
    if len(check) != 38 or not check[["md_nan_pattern_equal", "thickness_nan_pattern_equal"]].all().all() or \
            check[["max_abs_md_error", "max_abs_thickness_error"]].to_numpy(float).max() > 1e-12:
        raise RuntimeError("Figure 7 held-out CN inputs do not reproduce Figure 6 inputs")

    write_json(a.output_dir / "fold1_cohort_audit.json", {
        "status": "complete", "fold_numbering": "zero-based outer_fold values 0..4",
        "selected_outer_fold": a.outer_fold, "selected_label": "Fold 1",
        "fold1_training_cn": len(train_cn), "fold1_heldout_cn": len(test_cn),
        "coverage95_source_n": len(pool), "age_range_inclusive": [AGE_MIN, AGE_MAX],
        "final_counts": counts, "final_n": len(analysis),
        "global_cognition_available": int(analysis.global_cognition.notna().sum()),
        "moca_available_after_88_missing_rule": int(analysis.moca.notna().sum()),
        "d0076_d0127_88_rule_applied": True,
        "d0271_d0285_in_final_analysis": sorted(set(analysis.subject_id) & {"D0271", "D0285"}),
        "model": "Figure 6 ico4 Warped M2, outer_fold=1: MD ~ linear age + categorical sex code + local FreeSurfer thickness",
        "retraining_permitted": False,
    })
    print(json.dumps({"status": "complete", "final_n": len(analysis), "counts": counts}))


if __name__ == "__main__":
    main()
