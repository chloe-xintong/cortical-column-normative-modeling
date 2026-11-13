#!/usr/bin/env python3
"""Audit inputs and create leakage-safe folds for the Figure 6 regional pilot.

Description
-----------
Audit inputs and create leakage-safe folds for the Figure 6 regional pilot.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--subject-order`` command-line input
- ``--cohort-manifest`` command-line input
- ``--regional-profile`` command-line input
- ``--regional-thickness`` command-line input
- ``--profile-metadata`` command-line input
- ``--thickness-metadata`` command-line input
- ``--figure5-folds`` command-line input
- ``--figure3-ranking`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/prepare_pilot.py --help
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pilot_utils import (AGE_BIN_LABELS, BOOTSTRAP_SEED, DEPTHS, FOLD_SEED, MODEL_SPECS,
                         N_BOOTSTRAP, age_bin, depth_band, sha256, write_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-order", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--regional-profile", type=Path, required=True)
    parser.add_argument("--regional-thickness", type=Path, required=True)
    parser.add_argument("--profile-metadata", type=Path, required=True)
    parser.add_argument("--thickness-metadata", type=Path, required=True)
    parser.add_argument("--figure5-folds", type=Path, required=True)
    parser.add_argument("--figure3-ranking", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-seed", type=int, default=FOLD_SEED)
    return parser.parse_args()


def exclusion_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if not 40 <= row.age < 80:
        reasons.append("age_outside_40_to_79")
    if row.sex_code not in (1, 2):
        reasons.append("sex_code_not_1_or_2_unverified")
    if str(row.diagnosis_at_scan) != "Normal":
        reasons.append("not_CN_at_matched_visit")
    if not bool(row.option3_qc_pass):
        reasons.append("Option3_QC_exclusion")
    if not bool(row.figure5_primary_eligible):
        reasons.append("not_Figure5_primary_eligible")
    if not bool(row.regional_profile_available):
        reasons.append("regional_profile_unavailable")
    if not bool(row.thickness_available):
        reasons.append("regional_thickness_unavailable")
    return ";".join(reasons)


def main() -> None:
    args = parse_args()
    out = args.output_dir
    for directory in (out, out / "logs", out / "shards", out / "models"):
        directory.mkdir(parents=True, exist_ok=True)

    subject_order = pd.read_csv(args.subject_order, dtype={"subject_id": str})
    cohort = pd.read_csv(args.cohort_manifest, dtype={"subject_id": str})
    profile_meta = pd.read_csv(args.profile_metadata)
    thickness_meta = pd.read_csv(args.thickness_metadata)
    profiles = np.load(args.regional_profile, mmap_mode="r")
    thickness = np.load(args.regional_thickness, mmap_mode="r")
    if profiles.shape != (len(subject_order), 68 * 21) or thickness.shape != (len(subject_order), 68):
        raise RuntimeError(f"Unexpected regional matrix shapes: {profiles.shape}, {thickness.shape}")
    if len(profile_meta) != 68 * 21 or len(thickness_meta) != 68:
        raise RuntimeError("Regional metadata dimensions do not match 68 regions x 21 depths")
    if not np.allclose(profile_meta.depth_fraction.unique(), DEPTHS):
        raise RuntimeError("Regional profile depths are not the canonical 21-point grid")
    repeated = profile_meta.iloc[::21].reset_index(drop=True)
    for column in ("hemisphere", "dk_label_id", "dk_label_name"):
        if not np.array_equal(repeated[column].to_numpy(), thickness_meta[column].to_numpy()):
            raise RuntimeError(f"Profile/thickness regional metadata mismatch in {column}")

    joined = subject_order.merge(
        cohort[["subject_id", "age_at_scan", "SUBJECT_SEX", "diagnosis_at_scan", "option3_qc_pass",
                "figure5_primary_eligible", "manual_registration_qc_excluded", "metadata_exclusion_reason"]],
        on="subject_id", how="left", suffixes=("_subject_order", "_metadata"), validate="one_to_one",
    )
    if joined["age_at_scan_metadata"].isna().any() or not np.allclose(
        joined.age_at_scan_subject_order, joined.age_at_scan_metadata
    ):
        raise RuntimeError("subject_order and final metadata ages do not agree")
    joined["age"] = joined.age_at_scan_subject_order.astype(float)
    joined["age_bin"] = age_bin(joined.age).astype(object).fillna("outside_40_to_79")
    joined["sex_code"] = joined.SUBJECT_SEX.astype(int)
    joined["regional_profile_available"] = np.isfinite(profiles).all(axis=1)
    joined["thickness_available"] = np.isfinite(thickness).all(axis=1)
    joined["exclusion_reason"] = joined.apply(exclusion_reasons, axis=1)
    joined["included"] = joined.exclusion_reason.eq("")
    audit_columns = ["subject_id", "age", "age_bin", "sex_code", "included", "exclusion_reason",
                     "regional_profile_available", "thickness_available", "diagnosis_at_scan",
                     "option3_qc_pass", "figure5_primary_eligible", "manual_registration_qc_excluded"]
    joined[audit_columns].to_csv(out / "cohort_audit.csv", index=False)
    included = joined.loc[joined.included].copy().reset_index(drop=True)
    if len(included) != 188:
        raise RuntimeError(f"Expected 188 included subjects after audit; found {len(included)}")
    if set(included.sex_code) != {1, 2}:
        raise RuntimeError("Included sex codes must be exactly {1, 2}")

    summary_rows = [
        {"summary": "Figure5 final cohort", "level": "all", "n": len(joined)},
        {"summary": "regional pilot included", "level": "all", "n": len(included)},
    ]
    for label, count in included.age_bin.value_counts().reindex(AGE_BIN_LABELS).items():
        summary_rows.append({"summary": "age_bin", "level": label, "n": int(count)})
    for code, count in included.sex_code.value_counts().sort_index().items():
        summary_rows.append({"summary": "sex_code", "level": f"sex_code_{code}", "n": int(count)})
    pd.DataFrame(summary_rows).to_csv(out / "cohort_summary.csv", index=False)

    strata = included.age_bin.astype(str) + "__sex_code_" + included.sex_code.astype(str)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.fold_seed)
    folds: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    for fold, (train, test) in enumerate(splitter.split(included, strata)):
        train_ids = included.iloc[train].subject_id.tolist()
        test_ids = included.iloc[test].subject_id.tolist()
        folds.append({"outer_fold": fold, "train_subject_ids": train_ids, "test_subject_ids": test_ids})
        assignments.extend({"subject_id": subject, "outer_fold": fold} for subject in test_ids)
    assignment = pd.DataFrame(assignments)
    if assignment.subject_id.nunique() != len(included) or len(assignment) != len(included):
        raise RuntimeError("New outer folds do not partition subjects exactly once")
    fold_payload = {
        "seed": args.fold_seed,
        "n_splits": 5,
        "source": "new Figure 6 folds",
        "stratification": "fixed decade age_bin x categorical sex_code",
        "reason": "Filtered Figure 5 folds had visibly uneven joint cells, including a one-subject cell; new joint stratification improves balance.",
        "folds": folds,
    }
    write_json(out / "fold_assignments.json", fold_payload)
    balance = assignment.merge(included[["subject_id", "age", "age_bin", "sex_code"]], on="subject_id")
    fold_balance = balance.groupby(["outer_fold", "age_bin", "sex_code"], observed=False).size().rename("n").reset_index()
    fold_balance.to_csv(out / "fold_balance.csv", index=False)

    missing_rows: list[dict[str, object]] = []
    matrix_rows = included.row_index.to_numpy(int)
    for feature_index, meta in profile_meta.iterrows():
        valid = np.isfinite(profiles[matrix_rows, feature_index])
        missing_rows.append({"audit_level": "region_depth", "subject_id": "", "hemisphere": meta.hemisphere,
                             "dk_label_id": int(meta.dk_label_id), "dk_label_name": meta.dk_label_name,
                             "depth": float(meta.depth_fraction), "age_bin": "all", "n_total": len(valid),
                             "n_valid_md": int(valid.sum()), "n_missing_md": int((~valid).sum()),
                             "n_valid_thickness": "", "n_missing_thickness": ""})
    for _, row in included.iterrows():
        pvalid = np.isfinite(profiles[int(row.row_index)])
        tvalid = np.isfinite(thickness[int(row.row_index)])
        missing_rows.append({"audit_level": "subject", "subject_id": row.subject_id, "hemisphere": "all",
                             "dk_label_id": "", "dk_label_name": "all", "depth": "all", "age_bin": row.age_bin,
                             "n_total": len(pvalid), "n_valid_md": int(pvalid.sum()),
                             "n_missing_md": int((~pvalid).sum()), "n_valid_thickness": int(tvalid.sum()),
                             "n_missing_thickness": int((~tvalid).sum())})
    for region_index, meta in thickness_meta.iterrows():
        valid = np.isfinite(thickness[matrix_rows, region_index])
        missing_rows.append({"audit_level": "thickness_region", "subject_id": "", "hemisphere": meta.hemisphere,
                             "dk_label_id": int(meta.dk_label_id), "dk_label_name": meta.dk_label_name,
                             "depth": "all", "age_bin": "all", "n_total": len(valid), "n_valid_md": "",
                             "n_missing_md": "", "n_valid_thickness": int(valid.sum()),
                             "n_missing_thickness": int((~valid).sum())})
    for label in AGE_BIN_LABELS:
        mask = included.age_bin.eq(label).to_numpy()
        missing_rows.append({"audit_level": "age_bin", "subject_id": "", "hemisphere": "all",
                             "dk_label_id": "", "dk_label_name": "all", "depth": "all", "age_bin": label,
                             "n_total": int(mask.sum()), "n_valid_md": int(np.isfinite(profiles[matrix_rows[mask]]).sum()),
                             "n_missing_md": int(np.isnan(profiles[matrix_rows[mask]]).sum()),
                             "n_valid_thickness": int(np.isfinite(thickness[matrix_rows[mask]]).sum()),
                             "n_missing_thickness": int(np.isnan(thickness[matrix_rows[mask]]).sum())})
    pd.DataFrame(missing_rows).to_csv(out / "missingness_audit.csv", index=False)

    selected = pd.read_csv(args.figure3_ranking).query("selected_for_figure3 == True")
    selected_regions = selected[["region_name", "selected_level", "color"]].to_dict("records")
    input_paths = [args.subject_order, args.cohort_manifest, args.regional_profile, args.regional_thickness,
                   args.profile_metadata, args.thickness_metadata, args.figure5_folds, args.figure3_ranking]
    write_json(out / "regional_input_manifest.json", {
        "region_definition": "68 separate hemisphere x Desikan-Killiany regions, exactly as Figure 5/Figure 3 inputs",
        "n_regions": 68, "n_depths": 21, "depths": DEPTHS.tolist(), "depth_direction": "pial_to_white",
        "thickness_definition": "within-region mean of native FreeSurfer cortical thickness mapped using stored Option 3 barycentric mapping; never column length",
        "selected_figure3_region_names": selected_regions,
        "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in input_paths],
    })
    write_json(out / "model_specifications.json", {
        "models": MODEL_SPECS,
        "residual_distribution": "Gaussian", "residual_scale": "constant",
        "sex_encoding": "categorical code 1 reference plus sex_code_2 indicator; no male/female labels",
        "spline": "patsy natural cubic regression spline cr(age, df=3, constraints='center'); fitted on each training fold and applied unchanged to held-out data",
        "standardization": "age/spline columns, local thickness, and response are standardized with training-fold parameters only; sex indicator remains 0/1",
        "blr": "official PCNToolkit BLR; no ARD, homoskedastic Gaussian, no warp, L-BFGS-B",
        "predictive_variance": "PCNToolkit BLR ys_s2: 1/beta + diag(Phi @ A^-1 @ Phi.T), transformed back to MD units",
    })
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unavailable"
    write_json(out / "run_metadata.json", {
        "status": "prepared", "git_commit": git_commit, "python": sys.version,
        "platform": platform.platform(), "subject_count": len(included), "age_range": [float(included.age.min()), float(included.age.max())],
        "sex_code_counts": {str(k): int(v) for k, v in included.sex_code.value_counts().sort_index().items()},
        "sex_code_9_handling": "Excluded by age before modeling: D0343 is age 30; mapping remains unverified.",
        "fold_seed": args.fold_seed, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_repetitions": N_BOOTSTRAP,
        "software_note": "PCNToolkit version is added by run_models.py from the isolated environment.",
        "failed_outcomes": [],
    })
    print(json.dumps({"status": "prepared", "n_included": len(included), "fold_balance": str(out / 'fold_balance.csv')}))


if __name__ == "__main__":
    main()
