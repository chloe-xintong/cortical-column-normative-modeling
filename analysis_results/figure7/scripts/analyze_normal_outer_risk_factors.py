#!/usr/bin/env python3
"""Repeat the Normal risk-factor screen using outer-band OOF burden.

Description
-----------
Repeat the Normal risk-factor screen using outer-band OOF burden.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--prepared-total-table`` command-line input
- ``--all-normal-burden`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure7/scripts/analyze_normal_outer_risk_factors.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from analyze_normal_risk_factors import fit_exposure


SPECS = [
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-total-table", type=Path, required=True)
    parser.add_argument("--all-normal-burden", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    a = parser.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    prepared = pd.read_csv(a.prepared_total_table, dtype={"subject_id": str})
    burden = pd.read_csv(a.all_normal_burden, dtype={"subject_id": str})
    if len(prepared) != 186 or len(burden) != 186:
        raise RuntimeError("Expected the same 186 Normal participants")
    data = prepared.drop(columns=["burden_percent"], errors="ignore").merge(
        burden[["subject_id", "outer_deviation_burden"]],
        on="subject_id", how="left", validate="one_to_one",
    )
    data["burden_percent"] = 100 * data.outer_deviation_burden
    if data.burden_percent.isna().any():
        raise RuntimeError("Outer burden is missing")
    data.to_csv(a.output_dir / "all_normal_outer_risk_factor_participant_table.csv", index=False)

    rows, interactions = [], []
    for exposure, label, unit, kind in SPECS:
        row, interaction = fit_exposure(data, exposure, label, unit, kind)
        rows.append(row)
        if interaction is not None:
            interactions.append(interaction)
    results = pd.DataFrame(rows)
    fitted = results.status.eq("fitted")
    results["p_value_fdr_bh"] = np.nan
    results.loc[fitted, "p_value_fdr_bh"] = multipletests(
        results.loc[fitted, "p_value"], method="fdr_bh"
    )[1]
    results.to_csv(a.output_dir / "all_normal_outer_risk_factor_burden_results.csv", index=False)
    interaction_results = pd.DataFrame(interactions)
    interaction_results["p_value_fdr_bh"] = multipletests(
        interaction_results.p_value, method="fdr_bh"
    )[1]
    interaction_results.to_csv(
        a.output_dir / "all_normal_outer_risk_factor_age_interactions.csv", index=False
    )
    report = {
        "status": "complete", "source_n": 186,
        "burden": "outer OOF depth-band burden",
        "depth_direction": "pial_to_white", "depth_indices": [0, 6],
        "covariates": ["age_at_scan", "categorical sex", "categorical outer_fold"],
        "multiplicity": "Benjamini-Hochberg FDR across fitted outer-burden models",
        "exposure_cleaning_source": str(a.prepared_total_table),
    }
    (a.output_dir / "all_normal_outer_risk_factor_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
