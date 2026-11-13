#!/usr/bin/env python3
"""Run participant-level Figure 7 regressions and prespecified sensitivities.

Description
-----------
Run participant-level Figure 7 regressions and prespecified sensitivities.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--burden`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure7/scripts/analyze_figure7.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


BURDENS = ["total_deviation_burden", "positive_deviation_burden", "negative_deviation_burden"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--burden", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def coefficient_row(model, term: str, analysis: str, outcome: str, predictor: str,
                    sensitivity: str, n: int) -> dict[str, object]:
    ci = model.conf_int().loc[term]
    return {"analysis": analysis, "outcome": outcome, "predictor": predictor,
            "sensitivity": sensitivity, "n_participants": n,
            "estimate": float(model.params[term]), "ci_low": float(ci.iloc[0]),
            "ci_high": float(ci.iloc[1]), "p_value": float(model.pvalues[term]),
            "standard_error_hc3": float(model.bse[term]), "covariance": "HC3",
            "statistical_unit": "participant"}


def fit_group(frame: pd.DataFrame, burden: str, sensitivity: str) -> list[dict[str, object]]:
    data = frame[frame.diagnosis_at_scan.isin(["Normal", "MCI", "AD"])].copy()
    formula = (f"{burden} ~ C(diagnosis_at_scan, Treatment(reference='Normal')) + "
               "age_at_scan + C(SUBJECT_SEX) + coverage_fraction")
    model = smf.ols(formula, data=data).fit(cov_type="HC3")
    rows = []
    for group in ("MCI", "AD"):
        term = f"C(diagnosis_at_scan, Treatment(reference='Normal'))[T.{group}]"
        row = coefficient_row(model, term, "Figure 7A", burden, f"{group}-CN",
                              sensitivity, int(model.nobs))
        row["estimate_percentage_points"] = 100 * row["estimate"]
        row["ci_low_percentage_points"] = 100 * row["ci_low"]
        row["ci_high_percentage_points"] = 100 * row["ci_high"]
        rows.append(row)
    return rows


def fit_cognition(frame: pd.DataFrame, burden: str, sensitivity: str,
                  adjust_diagnosis: bool) -> dict[str, object]:
    data = frame.loc[frame.global_cognition.notna()].copy()
    sd = float(data[burden].std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"Cannot standardize {burden}")
    data["burden_z"] = (data[burden] - data[burden].mean()) / sd
    formula = "global_cognition ~ burden_z + age_at_scan + C(SUBJECT_SEX) + coverage_fraction"
    if adjust_diagnosis:
        formula += " + C(diagnosis_at_scan)"
    model = smf.ols(formula, data=data).fit(cov_type="HC3")
    label = sensitivity + ("_diagnosis_adjusted" if adjust_diagnosis else "")
    row = coefficient_row(model, "burden_z", "Figure 7B", "global_cognition",
                          burden + " (per 1 SD)", label, int(model.nobs))
    row.update({"burden_mean": float(data[burden].mean()), "burden_sd": sd,
                "diagnosis_adjusted": adjust_diagnosis,
                "covariate_adjustment": "age + sex + coverage" +
                (" + diagnosis" if adjust_diagnosis else ""),
                "r_squared": float(model.rsquared), "formula": formula,
                "global_cognition_missing": int(frame.global_cognition.isna().sum())})
    return row


def fit_cognition_raw(frame: pd.DataFrame, burden: str) -> dict[str, object]:
    data = frame.loc[frame.global_cognition.notna()].copy()
    sd = float(data[burden].std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"Cannot standardize {burden}")
    data["burden_z"] = (data[burden] - data[burden].mean()) / sd
    formula = "global_cognition ~ burden_z"
    model = smf.ols(formula, data=data).fit(cov_type="HC3")
    row = coefficient_row(model, "burden_z", "Figure 7B", "global_cognition",
                          burden + " (per 1 SD)", "raw_unadjusted", int(model.nobs))
    row.update({"burden_mean": float(data[burden].mean()), "burden_sd": sd,
                "diagnosis_adjusted": False, "covariate_adjustment": "none",
                "r_squared": float(model.rsquared), "formula": formula,
                "global_cognition_missing": int(frame.global_cognition.isna().sum())})
    return row


def group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, data in frame.groupby("diagnosis_at_scan", sort=False):
        rows.append({"diagnosis": group, "n": len(data),
                     "age_mean": data.age_at_scan.mean(), "age_sd": data.age_at_scan.std(ddof=1),
                     "age_min": data.age_at_scan.min(), "age_max": data.age_at_scan.max(),
                     "sex_code_1_n": int(data.SUBJECT_SEX.eq(1).sum()),
                     "sex_code_2_n": int(data.SUBJECT_SEX.eq(2).sum()),
                     "coverage_mean": data.coverage_fraction.mean(),
                     "coverage_sd": data.coverage_fraction.std(ddof=1),
                     "coverage_min": data.coverage_fraction.min(),
                     "coverage_max": data.coverage_fraction.max(),
                     "global_cognition_n": int(data.global_cognition.notna().sum()),
                     **{f"{b}_mean": data[b].mean() for b in BURDENS},
                     **{f"{b}_sd": data[b].std(ddof=1) for b in BURDENS}})
    result = pd.DataFrame(rows)
    cn = result.set_index("diagnosis").loc["Normal"]
    result["age_range_overlaps_cn"] = result.apply(
        lambda x: max(x.age_min, cn.age_min) <= min(x.age_max, cn.age_max), axis=1)
    result["coverage_range_overlaps_cn"] = result.apply(
        lambda x: max(x.coverage_min, cn.coverage_min) <= min(x.coverage_max, cn.coverage_max), axis=1)
    pooled_age_sd = frame.age_at_scan.std(ddof=1)
    pooled_cov_sd = frame.coverage_fraction.std(ddof=1)
    result["age_standardized_mean_difference_vs_cn"] = (result.age_mean - cn.age_mean) / pooled_age_sd
    result["coverage_standardized_mean_difference_vs_cn"] = (
        result.coverage_mean - cn.coverage_mean) / pooled_cov_sd
    return result


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(a.burden, dtype={"subject_id": str})
    if len(data) != 111 or data.subject_id.nunique() != 111:
        raise RuntimeError("Figure 7 burden table is not the frozen N=111 cohort")
    if set(data.loc[data.diagnosis_at_scan.eq("Normal"), "model_role"]) != {"fold1_heldout_cn"}:
        raise RuntimeError("Normal analysis group contains non-held-out participants")

    group_rows = fit_group(data, "total_deviation_burden", "primary")
    raw_p = [row["p_value"] for row in group_rows]
    adjusted = multipletests(raw_p, method="holm")[1]
    for row, p_holm in zip(group_rows, adjusted):
        row["p_value_holm"] = float(p_holm)
        row["multiplicity_family"] = "two primary contrasts: MCI-CN and AD-CN"
    pd.DataFrame(group_rows).to_csv(a.output_dir / "figure7a_primary_results.csv", index=False)

    cognition_raw = fit_cognition_raw(data, "total_deviation_burden")
    cognition_primary = fit_cognition(data, "total_deviation_burden", "primary", False)
    cognition_diagnosis = fit_cognition(data, "total_deviation_burden", "primary", True)
    pd.DataFrame([cognition_raw, cognition_primary, cognition_diagnosis]).to_csv(
        a.output_dir / "figure7b_primary_and_diagnosis_sensitivity.csv", index=False)

    robustness = []
    coverage98 = data.loc[data.coverage_ge_0_98].copy()
    for row in fit_group(coverage98, "total_deviation_burden", "coverage_ge_0_98"):
        robustness.append(row)
    robustness.append(fit_cognition(coverage98, "total_deviation_burden", "coverage_ge_0_98", False))
    robustness.append(fit_cognition(coverage98, "total_deviation_burden", "coverage_ge_0_98", True))
    for burden in ("positive_deviation_burden", "negative_deviation_burden"):
        robustness.extend(fit_group(data, burden, "directional_burden"))
        robustness.append(fit_cognition(data, burden, "directional_burden", False))
        robustness.append(fit_cognition(data, burden, "directional_burden", True))
    pd.DataFrame(robustness).to_csv(a.output_dir / "robustness_results.csv", index=False)
    group_summary(data).to_csv(a.output_dir / "covariate_balance_by_group.csv", index=False)

    excluded_special = sorted({"D0271", "D0285"} & set(data.subject_id))
    report = {"status": "complete", "analysis_cohort_n": len(data),
              "figure7a_n": int(data.diagnosis_at_scan.isin(["Normal", "MCI", "AD"]).sum()),
              "figure7a_counts": data[data.diagnosis_at_scan.isin(["Normal", "MCI", "AD"])].diagnosis_at_scan.value_counts().to_dict(),
              "figure7b_global_cognition_n": int(data.global_cognition.notna().sum()),
              "figure7b_global_cognition_missing": data.loc[data.global_cognition.isna(), "subject_id"].tolist(),
              "coverage98_n": len(coverage98),
              "coverage98_counts": coverage98.diagnosis_at_scan.value_counts().to_dict(),
              "d0271_d0285_in_analysis": excluded_special,
              "d0271_d0285_exclusion_sensitivity": "not applicable; neither subject is in the prespecified analysis cohort",
              "moca_primary_figure": False, "statistical_unit": "participant",
              "primary_burden": "total_deviation_burden",
              "primary_group_comparisons": ["MCI-CN", "AD-CN"],
              "primary_group_multiplicity": "Holm correction across two contrasts",
              "covariance": "HC3 robust standard errors"}
    (a.output_dir / "analysis_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
