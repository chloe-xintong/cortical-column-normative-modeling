#!/usr/bin/env python3
"""Exploratory APOE e4 association with Figure 7A deviation burden.

Description
-----------
Exploratory APOE e4 association with Figure 7A deviation burden.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--burden`` command-line input
- ``--clinical-metadata`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure7/scripts/analyze_apoe4_burden.py --help
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
    p.add_argument("--clinical-metadata", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def coefficient(model, model_name: str, formula: str) -> dict[str, object]:
    term = "apoe4_carrier"
    ci = model.conf_int().loc[term]
    return {
        "analysis": "exploratory_APOE4_Figure7A",
        "model": model_name,
        "formula": formula,
        "n_participants": int(model.nobs),
        "estimate_percentage_points": float(model.params[term]),
        "ci_low_percentage_points": float(ci.iloc[0]),
        "ci_high_percentage_points": float(ci.iloc[1]),
        "standard_error_hc3": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
        "covariance": "HC3",
        "statistical_unit": "participant",
    }


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    burden = pd.read_csv(
        a.burden, dtype={"subject_id": str, "PTID_clean": str, "visit_id": str}
    )
    burden = burden.loc[burden.diagnosis_at_scan.isin(["Normal", "MCI", "AD"])].copy()
    expected = {"Normal": 38, "MCI": 19, "AD": 6}
    if len(burden) != 63 or burden.diagnosis_at_scan.value_counts().to_dict() != expected:
        raise RuntimeError("APOE analysis source is not the formal Figure 7A N=63 cohort")
    if "D0128" in set(burden.subject_id):
        raise RuntimeError("D0128 must not enter the formal APOE analysis")

    clinical = pd.read_csv(
        a.clinical_metadata, low_memory=False,
        dtype={"PTID_clean": str, "VISITNUM": str},
    )
    apoe_columns = ["PTID_clean", "VISITNUM", "APOE_genotype_harmonized",
                    "APOE4_carriage", "APOE4_dosage"]
    if clinical.duplicated(["PTID_clean", "VISITNUM"]).any():
        raise RuntimeError("Clinical PTID/visit key is not unique")
    data = burden.merge(
        clinical[apoe_columns], left_on=["PTID_clean", "visit_id"],
        right_on=["PTID_clean", "VISITNUM"], how="left", validate="one_to_one",
    )
    data["apoe4_carrier"] = pd.to_numeric(data.APOE4_carriage, errors="coerce")
    data["burden_percent"] = 100 * data.total_deviation_burden
    data["apoe4_status"] = data.apoe4_carrier.map({0.0: "Non-carrier", 1.0: "Carrier"}).fillna("Missing")
    data.to_csv(a.output_dir / "figure7a_apoe4_participant_table.csv", index=False)

    complete = data.dropna(subset=["burden_percent", "apoe4_carrier", "age_at_scan",
                                   "SUBJECT_SEX", "diagnosis_at_scan"]).copy()
    if len(complete) != 43 or complete.apoe4_carrier.value_counts().to_dict() != {0.0: 22, 1.0: 21}:
        raise RuntimeError("Unexpected APOE complete-case count or carrier composition")

    formulas = {
        "base": "burden_percent ~ apoe4_carrier + age_at_scan + C(SUBJECT_SEX)",
        "diagnosis_adjusted": (
            "burden_percent ~ apoe4_carrier + age_at_scan + C(SUBJECT_SEX) + "
            "C(diagnosis_at_scan)"
        ),
    }
    results = []
    for name, formula in formulas.items():
        model = smf.ols(formula, data=complete).fit(cov_type="HC3")
        results.append(coefficient(model, name, formula))
    pd.DataFrame(results).to_csv(a.output_dir / "figure7a_apoe4_regression_results.csv", index=False)

    genotype = pd.crosstab(
        data.APOE_genotype_harmonized.fillna("Missing"), data.diagnosis_at_scan,
    ).reindex(columns=["Normal", "MCI", "AD"], fill_value=0)
    genotype["Total"] = genotype.sum(axis=1)
    genotype.to_csv(a.output_dir / "figure7a_apoe_genotype_by_diagnosis.csv")
    carrier = pd.crosstab(data.apoe4_status, data.diagnosis_at_scan).reindex(
        index=["Non-carrier", "Carrier", "Missing"],
        columns=["Normal", "MCI", "AD"], fill_value=0,
    )
    carrier["Total"] = carrier.sum(axis=1)
    carrier.to_csv(a.output_dir / "figure7a_apoe4_status_by_diagnosis.csv")

    report = {
        "status": "complete", "exploratory": True,
        "candidate_n": 63, "complete_case_n": len(complete), "apoe_missing_n": 20,
        "apoe4_carriers_n": int(complete.apoe4_carrier.eq(1).sum()),
        "apoe4_noncarriers_n": int(complete.apoe4_carrier.eq(0).sum()),
        "complete_case_counts_by_diagnosis": complete.diagnosis_at_scan.value_counts().to_dict(),
        "D0128_included": False, "interaction_fitted": False,
        "full_genotype_model_fitted": False,
        "outcome": "total deviation burden in percentage points",
        "model_source": "fixed Figure 6 Fold 1 normative model used by formal Figure 7A",
    }
    (a.output_dir / "figure7a_apoe4_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
