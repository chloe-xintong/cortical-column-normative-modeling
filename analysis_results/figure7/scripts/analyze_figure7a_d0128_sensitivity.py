#!/usr/bin/env python3
"""Analyze Figure 7A after adding D0128, without changing primary outputs.

Description
-----------
Analyze Figure 7A after adding D0128, without changing primary outputs.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--burden`` command-line input
- ``--primary-results`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure7/scripts/analyze_figure7a_d0128_sensitivity.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--burden", type=Path, required=True)
    p.add_argument("--primary-results", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(a.burden, dtype={"subject_id": str})
    counts = data.diagnosis_at_scan.value_counts().to_dict()
    expected = {"Normal": 38, "MCI": 19, "AD": 7,
                "Other/Impaired non-demented": 22, "Unknown": 26}
    if len(data) != 112 or counts != expected or set(data.loc[data.subject_id.eq("D0128"), "diagnosis_at_scan"]) != {"AD"}:
        raise RuntimeError(f"Unexpected D0128 sensitivity cohort: N={len(data)}, counts={counts}")
    subset = data.loc[data.diagnosis_at_scan.isin(["Normal", "MCI", "AD"])].copy()
    formula = ("total_deviation_burden ~ "
               "C(diagnosis_at_scan, Treatment(reference='Normal')) + "
               "age_at_scan + C(SUBJECT_SEX) + coverage_fraction")
    model = smf.ols(formula, data=subset).fit(cov_type="HC3")
    rows = []
    for group in ("MCI", "AD"):
        term = f"C(diagnosis_at_scan, Treatment(reference='Normal'))[T.{group}]"
        ci = model.conf_int().loc[term]
        rows.append({
            "analysis": "Figure 7A", "sensitivity": "include_D0128_registration_failure",
            "predictor": f"{group}-CN", "n_participants": int(model.nobs),
            "estimate": float(model.params[term]), "ci_low": float(ci.iloc[0]),
            "ci_high": float(ci.iloc[1]), "p_value": float(model.pvalues[term]),
            "standard_error_hc3": float(model.bse[term]),
            "estimate_percentage_points": 100 * float(model.params[term]),
            "ci_low_percentage_points": 100 * float(ci.iloc[0]),
            "ci_high_percentage_points": 100 * float(ci.iloc[1]),
            "covariance": "HC3", "formula": formula,
        })
    adjusted = multipletests([row["p_value"] for row in rows], method="holm")[1]
    for row, p_holm in zip(rows, adjusted):
        row["p_value_holm"] = float(p_holm)
        row["multiplicity_family"] = "two contrasts: MCI-CN and AD-CN"
    result = pd.DataFrame(rows)
    result.to_csv(a.output_dir / "figure7a_include_D0128_results.csv", index=False)

    primary = pd.read_csv(a.primary_results)
    comparison = primary.merge(result, on="predictor", suffixes=("_primary", "_include_D0128"))
    comparison["estimate_change_percentage_points"] = (
        comparison.estimate_percentage_points_include_D0128
        - comparison.estimate_percentage_points_primary
    )
    comparison.to_csv(a.output_dir / "figure7a_primary_vs_include_D0128.csv", index=False)
    d0128 = data.loc[data.subject_id.eq("D0128")].iloc[0]
    report = {
        "status": "complete", "sensitivity_only": True,
        "registration_qc_warning": "D0128 failed visual T1-DWI registration QC",
        "analysis_n": int(model.nobs), "analysis_counts": {"Normal": 38, "MCI": 19, "AD": 7},
        "D0128": {"age": float(d0128.age_at_scan),
                   "coverage_fraction": float(d0128.coverage_fraction),
                   "total_deviation_burden": float(d0128.total_deviation_burden),
                   "positive_deviation_burden": float(d0128.positive_deviation_burden),
                   "negative_deviation_burden": float(d0128.negative_deviation_burden)},
    }
    (a.output_dir / "analysis_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
