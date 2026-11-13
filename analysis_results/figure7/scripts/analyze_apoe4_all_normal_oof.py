#!/usr/bin/env python3
"""APOE e4 association with five-fold OOF burden in all Figure 6 Normal subjects.

Description
-----------
APOE e4 association with five-fold OOF burden in all Figure 6 Normal subjects.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--burden`` command-line input
- ``--cohort-manifest`` command-line input
- ``--clinical-metadata`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure7/scripts/analyze_apoe4_all_normal_oof.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--burden", type=Path, required=True)
    p.add_argument("--cohort-manifest", type=Path, required=True)
    p.add_argument("--clinical-metadata", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    burden = pd.read_csv(a.burden, dtype={"subject_id": str})
    if len(burden) != 186 or burden.subject_id.nunique() != 186:
        raise RuntimeError("Expected all 186 frozen Figure 6 Normal participants")
    if set(burden.diagnosis_at_scan) != {"Normal"}:
        raise RuntimeError("All-Normal APOE source contains a non-Normal diagnosis")
    if burden.outer_fold.value_counts().sort_index().tolist() != [38, 38, 37, 36, 37]:
        raise RuntimeError("Unexpected five-fold OOF composition")

    manifest = pd.read_csv(
        a.cohort_manifest,
        dtype={"subject_id": str, "PTID_clean": str, "visit_id": str},
    )
    manifest = manifest.loc[manifest.subject_id.isin(burden.subject_id),
                            ["subject_id", "PTID_clean", "visit_id"]]
    if len(manifest) != 186:
        raise RuntimeError("Not all Figure 6 Normal subjects map to the cohort manifest")
    clinical = pd.read_csv(
        a.clinical_metadata, low_memory=False,
        dtype={"PTID_clean": str, "VISITNUM": str},
    )
    columns = ["PTID_clean", "VISITNUM", "APOE_genotype_harmonized",
               "APOE4_carriage", "APOE4_dosage"]
    data = burden.merge(manifest, on="subject_id", validate="one_to_one").merge(
        clinical[columns], left_on=["PTID_clean", "visit_id"],
        right_on=["PTID_clean", "VISITNUM"], how="left", validate="one_to_one",
    )
    data["apoe4_carrier"] = pd.to_numeric(data.APOE4_carriage, errors="coerce")
    data["apoe4_status"] = data.apoe4_carrier.map(
        {0.0: "Non-carrier", 1.0: "Carrier"}
    ).fillna("Missing")
    data["burden_percent"] = 100 * data.total_deviation_burden
    data.to_csv(a.output_dir / "all_normal_oof_apoe4_participant_table.csv", index=False)

    complete = data.dropna(subset=["burden_percent", "apoe4_carrier", "age_at_scan",
                                   "SUBJECT_SEX", "outer_fold"]).copy()
    if len(complete) != 128 or complete.apoe4_carrier.value_counts().to_dict() != {0.0: 64, 1.0: 64}:
        raise RuntimeError("Expected 64 APOE e4 carriers and 64 non-carriers")
    formula = (
        "burden_percent ~ apoe4_carrier + age_at_scan + "
        "C(SUBJECT_SEX) + C(outer_fold)"
    )
    model = smf.ols(formula, data=complete).fit(cov_type="HC3")
    term = "apoe4_carrier"
    ci = model.conf_int().loc[term]
    result = pd.DataFrame([{
        "analysis": "supplement_all_Normal_five_fold_OOF_APOE4",
        "model": "age_sex_fold_adjusted",
        "formula": formula,
        "n_participants": int(model.nobs),
        "estimate_percentage_points": float(model.params[term]),
        "ci_low_percentage_points": float(ci.iloc[0]),
        "ci_high_percentage_points": float(ci.iloc[1]),
        "standard_error_hc3": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
        "r_squared": float(model.rsquared),
        "covariance": "HC3",
        "statistical_unit": "participant",
    }])
    result.to_csv(a.output_dir / "all_normal_oof_apoe4_regression_results.csv", index=False)

    fold_status = pd.crosstab(complete.outer_fold, complete.apoe4_status).reindex(
        index=range(5), columns=["Non-carrier", "Carrier"], fill_value=0,
    )
    fold_status["Total"] = fold_status.sum(axis=1)
    fold_status.to_csv(a.output_dir / "all_normal_oof_apoe4_by_fold.csv")
    report = {
        "status": "complete", "supplementary": True, "normal_only": True,
        "source_normal_n": 186, "complete_case_n": 128, "apoe_missing_n": 58,
        "apoe4_carriers_n": 64, "apoe4_noncarriers_n": 64,
        "fold_adjustment": "categorical", "model_refit": False,
        "burden_source": "each participant's own held-out Figure 6 fold z-scores",
        "z_threshold": 2.0,
    }
    (a.output_dir / "all_normal_oof_apoe4_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({**report, **result.iloc[0].to_dict()}))


if __name__ == "__main__":
    main()
