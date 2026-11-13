#!/usr/bin/env python3
"""Explore vascular, metabolic, imaging, and biomarker correlates of Normal OOF burden.

Description
-----------
Explore vascular, metabolic, imaging, and biomarker correlates of Normal OOF burden.

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
- ``--raw-imaging-root`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure7/scripts/analyze_normal_risk_factors.py --help
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--burden", type=Path, required=True)
    p.add_argument("--cohort-manifest", type=Path, required=True)
    p.add_argument("--clinical-metadata", type=Path, required=True)
    p.add_argument("--raw-imaging-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--burden-column", default="total_deviation_burden")
    p.add_argument("--output-stem", default="all_normal_risk_factor")
    return p.parse_args()


def dwi_date(root: Path, subject: str) -> pd.Timestamp:
    dates = []
    for name in glob.glob(str(root / subject / "*.json")):
        if Path(name).name == "T1.json" or "revphase" in Path(name).name.lower():
            continue
        try:
            value = json.loads(Path(name).read_text()).get("AcquisitionDateTime")
            if value:
                dates.append(pd.to_datetime(value).normalize())
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    dates = sorted(set(dates))
    if len(dates) != 1:
        return pd.NaT
    return dates[0]


def fit_exposure(data: pd.DataFrame, exposure: str, label: str, unit: str,
                 kind: str, events: int | None = None) -> tuple[dict, dict | None]:
    needed = ["burden_percent", exposure, "age_at_scan", "SUBJECT_SEX", "outer_fold"]
    complete = data.dropna(subset=needed).copy()
    row = {
        "exposure": exposure, "label": label, "unit": unit, "kind": kind,
        "candidate_n": len(data), "complete_case_n": len(complete),
        "missing_n": int(data[exposure].isna().sum()),
        "events_or_positive_n": events if events is not None else np.nan,
    }
    if len(complete) < 30 or complete[exposure].nunique() < 2:
        row.update({"status": "not_fitted_insufficient_data_or_variation"})
        return row, None
    if kind == "binary":
        event_n = int(complete[exposure].eq(1).sum())
        nonevent_n = int(complete[exposure].eq(0).sum())
        row["events_or_positive_n"] = event_n
        if min(event_n, nonevent_n) < 5:
            row.update({"status": "not_fitted_fewer_than_5_in_one_binary_cell"})
            return row, None
    formula = f"burden_percent ~ {exposure} + age_at_scan + C(SUBJECT_SEX) + C(outer_fold)"
    model = smf.ols(formula, data=complete).fit(cov_type="HC3")
    ci = model.conf_int().loc[exposure]
    row.update({
        "status": "fitted", "formula": formula,
        "estimate_percentage_points": float(model.params[exposure]),
        "ci_low_percentage_points": float(ci.iloc[0]),
        "ci_high_percentage_points": float(ci.iloc[1]),
        "standard_error_hc3": float(model.bse[exposure]),
        "p_value": float(model.pvalues[exposure]), "r_squared": float(model.rsquared),
    })

    complete["age_centered"] = complete.age_at_scan - complete.age_at_scan.mean()
    interaction_formula = (
        f"burden_percent ~ age_centered * {exposure} + "
        "C(SUBJECT_SEX) + C(outer_fold)"
    )
    interaction_model = smf.ols(interaction_formula, data=complete).fit(cov_type="HC3")
    term = f"age_centered:{exposure}"
    ici = interaction_model.conf_int().loc[term]
    interaction = {
        "exposure": exposure, "label": label, "unit": f"per year × {unit}",
        "complete_case_n": len(complete), "formula": interaction_formula,
        "estimate_percentage_points": float(interaction_model.params[term]),
        "ci_low_percentage_points": float(ici.iloc[0]),
        "ci_high_percentage_points": float(ici.iloc[1]),
        "standard_error_hc3": float(interaction_model.bse[term]),
        "p_value": float(interaction_model.pvalues[term]),
        "interpretation_scope": "age heterogeneity of burden association; not a raw-MD age-slope test",
    }
    return row, interaction


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    burden = pd.read_csv(a.burden, dtype={"subject_id": str})
    if len(burden) != 186 or set(burden.diagnosis_at_scan) != {"Normal"}:
        raise RuntimeError("Expected all 186 frozen Figure 6 Normal participants")
    manifest = pd.read_csv(
        a.cohort_manifest,
        dtype={"subject_id": str, "PTID_clean": str, "visit_id": str},
    )
    manifest = manifest.loc[manifest.subject_id.isin(burden.subject_id)].copy()
    clinical = pd.read_csv(
        a.clinical_metadata, low_memory=False,
        dtype={"PTID_clean": str, "VISITNUM": str},
    )
    variables = [
        "PTID_clean", "VISITNUM", "VISIT_DATE", "SUBJECT_AGE_SCREEN", "DECAGE", "VISIT_AGE",
        "BPSYS_AVG", "BPDIA_AVG", "DIABETES", "BMI_corrected",
        "IMAGMWMH", "IMAGEWMH", "IMAGLINF", "IMAGLAC", "IMAGMICH",
        "NFL", "GFAP", "PTAU181", "AB42", "AB40",
    ]
    data = burden.merge(
        manifest[["subject_id", "PTID_clean", "visit_id", "matched_visit_date",
                  "QIAL_match_confidence"]], on="subject_id", validate="one_to_one",
    ).merge(
        clinical[variables], left_on=["PTID_clean", "visit_id"],
        right_on=["PTID_clean", "VISITNUM"], how="left", validate="one_to_one",
    )
    if a.burden_column not in data:
        raise RuntimeError(f"Missing requested burden column: {a.burden_column}")
    data["burden_percent"] = 100 * pd.to_numeric(data[a.burden_column], errors="coerce")
    data["dwi_date"] = [dwi_date(a.raw_imaging_root, sid) for sid in data.subject_id]
    data["clinical_visit_date"] = pd.to_datetime(data.VISIT_DATE)
    data["scan_visit_gap_days"] = (data.dwi_date - data.clinical_visit_date).dt.days
    data["abs_scan_visit_gap_days"] = data.scan_visit_gap_days.abs()

    # Explicit cleaning/scaling. Special codes are never treated as measurements.
    data["systolic_per_10"] = pd.to_numeric(data.BPSYS_AVG, errors="coerce").where(
        lambda x: x.between(70, 230)
    ) / 10
    data["diastolic_per_10"] = pd.to_numeric(data.BPDIA_AVG, errors="coerce").where(
        lambda x: x.between(30, 140)
    ) / 10
    diabetes = pd.to_numeric(data.DIABETES, errors="coerce")
    data["diabetes_history"] = diabetes.map({0.0: 0.0, 1.0: 1.0, 2.0: 1.0})
    data["bmi_per_5"] = pd.to_numeric(data.BMI_corrected, errors="coerce").where(
        lambda x: x.between(10, 100)
    ) / 5
    for source in ["IMAGMWMH", "IMAGEWMH", "IMAGLINF", "IMAGLAC", "IMAGMICH"]:
        data[source.lower() + "_binary"] = pd.to_numeric(data[source], errors="coerce").map(
            {0.0: 0.0, 1.0: 1.0}
        )
    biomarker_sources = ["NFL", "GFAP", "PTAU181", "AB42", "AB40"]
    for source in biomarker_sources:
        values = pd.to_numeric(data[source], errors="coerce").where(lambda x: x > 0)
        transformed = np.log2(values)
        data[source.lower() + "_log2_z"] = (transformed - transformed.mean()) / transformed.std(ddof=0)
    ratio = (pd.to_numeric(data.AB42, errors="coerce") /
             pd.to_numeric(data.AB40, errors="coerce")).where(lambda x: x > 0)
    data["ab42_ab40_ratio_z"] = (ratio - ratio.mean()) / ratio.std(ddof=0)
    data.to_csv(a.output_dir / f"{a.output_stem}_participant_table.csv", index=False)

    specs = [
        ("systolic_per_10", "Systolic blood pressure", "10 mmHg", "continuous"),
        ("diastolic_per_10", "Diastolic blood pressure", "10 mmHg", "continuous"),
        ("diabetes_history", "Diabetes history", "history vs absent", "binary"),
        ("bmi_per_5", "Corrected BMI", "5 kg/m²", "continuous"),
        ("imagmwmh_binary", "Moderate WMH", "yes vs no", "binary"),
        ("imagewmh_binary", "Extensive WMH", "yes vs no", "binary"),
        ("imaglinf_binary", "Large infarct", "yes vs no", "binary"),
        ("imaglac_binary", "Lacunar infarct", "yes vs no", "binary"),
        ("imagmich_binary", "Microhemorrhage", "yes vs no", "binary"),
        ("nfl_log2_z", "NfL", "1 SD log2", "continuous"),
        ("gfap_log2_z", "GFAP", "1 SD log2", "continuous"),
        ("ptau181_log2_z", "p-tau181", "1 SD log2", "continuous"),
        ("ab42_log2_z", "Aβ42", "1 SD log2", "continuous"),
        ("ab40_log2_z", "Aβ40", "1 SD log2", "continuous"),
        ("ab42_ab40_ratio_z", "Aβ42/Aβ40 ratio", "1 SD", "continuous"),
    ]
    rows, interactions = [], []
    for exposure, label, unit, kind in specs:
        row, interaction = fit_exposure(data, exposure, label, unit, kind)
        rows.append(row)
        if interaction is not None:
            interactions.append(interaction)
    results = pd.DataFrame(rows)
    fitted = results.status.eq("fitted")
    results["p_value_fdr_bh"] = np.nan
    if fitted.any():
        results.loc[fitted, "p_value_fdr_bh"] = multipletests(
            results.loc[fitted, "p_value"], method="fdr_bh"
        )[1]
    results.to_csv(a.output_dir / f"{a.output_stem}_burden_results.csv", index=False)
    interaction_results = pd.DataFrame(interactions)
    if len(interaction_results):
        interaction_results["p_value_fdr_bh"] = multipletests(
            interaction_results.p_value, method="fdr_bh"
        )[1]
    interaction_results.to_csv(
        a.output_dir / f"{a.output_stem}_age_interactions.csv", index=False
    )

    availability = []
    for exposure, label, unit, kind in specs:
        values = data[exposure]
        availability.append({
            "exposure": exposure, "label": label, "available_n": int(values.notna().sum()),
            "missing_n": int(values.isna().sum()), "unique_nonmissing": int(values.nunique()),
            "positive_n_if_binary": int(values.eq(1).sum()) if kind == "binary" else np.nan,
        })
    pd.DataFrame(availability).to_csv(
        a.output_dir / f"{a.output_stem}_availability.csv", index=False
    )
    gap = data.abs_scan_visit_gap_days.dropna()
    audit = {
        "status": "complete", "source_n": 186,
        "burden_column": a.burden_column,
        "exact_dwi_date_available_n": int(data.dwi_date.notna().sum()),
        "clinical_visit_date_available_n": int(data.clinical_visit_date.notna().sum()),
        "median_absolute_scan_visit_gap_days": float(gap.median()),
        "minimum_absolute_scan_visit_gap_days": int(gap.min()),
        "maximum_absolute_scan_visit_gap_days": int(gap.max()),
        "within_30_days_n": int(gap.le(30).sum()),
        "within_90_days_n": int(gap.le(90).sum()),
        "age_source": "VISIT_AGE at QIAL-matched clinical visit",
        "blood_pressure_special_codes_removed": [777, 888, -4],
        "diabetes_definition": "codes 1 recent/active or 2 remote/inactive vs code 0 absent",
        "imaging_code_8_treated_missing": True,
        "biomarker_date_caveat": "values are on the matched clinical row; specimen collection dates/platforms are not available here",
        "multiplicity": "Benjamini-Hochberg FDR separately for main associations and exploratory age interactions",
        "interaction_scope": "burden association heterogeneity by age; not a raw MD age-slope interaction",
    }
    (a.output_dir / f"{a.output_stem}_validation.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
