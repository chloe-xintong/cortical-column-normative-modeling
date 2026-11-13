#!/usr/bin/env python3
"""Compare total and cortical-depth-band burdens in formal Figure 7B.

Description
-----------
Compare total and cortical-depth-band burdens in formal Figure 7B.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--total-burden`` command-line input
- ``--band-burden`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure7/scripts/analyze_figure7b_depth_bands.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


BANDS = ["outer", "middle", "inner"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-burden", type=Path, required=True)
    p.add_argument("--band-burden", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def fit(data: pd.DataFrame, predictor: str, label: str, diagnosis_adjusted: bool) -> dict:
    values = data[predictor].to_numpy(float)
    sd = float(np.std(values, ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"Cannot standardize {predictor}")
    frame = data.copy()
    frame["burden_z"] = (frame[predictor] - frame[predictor].mean()) / sd
    formula = "global_cognition ~ burden_z + age_at_scan + C(SUBJECT_SEX) + coverage_fraction"
    if diagnosis_adjusted:
        formula += " + C(diagnosis_at_scan)"
    model = smf.ols(formula, data=frame).fit(cov_type="HC3")
    ci = model.conf_int().loc["burden_z"]
    return {
        "burden": label,
        "diagnosis_adjusted": diagnosis_adjusted,
        "n_participants": int(model.nobs),
        "estimate_per_1sd": float(model.params["burden_z"]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "standard_error_hc3": float(model.bse["burden_z"]),
        "p_value": float(model.pvalues["burden_z"]),
        "burden_mean": float(frame[predictor].mean()),
        "burden_sd": sd,
        "r_squared": float(model.rsquared),
        "formula": formula,
        "covariance": "HC3",
    }


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    total = pd.read_csv(a.total_burden, dtype={"subject_id": str})
    bands = pd.read_csv(a.band_burden, dtype={"subject_id": str})
    if len(total) != 111 or bands.shape[0] != 333:
        raise RuntimeError("Expected formal N=111 total burden and 3×111 band rows")
    if set(bands.depth_band) != set(BANDS):
        raise RuntimeError("Unexpected depth-band labels")
    wide = bands.pivot(index="subject_id", columns="depth_band",
                       values="total_deviation_burden").reset_index()
    wide = wide.rename(columns={band: f"{band}_burden" for band in BANDS})
    data = total.merge(wide, on="subject_id", validate="one_to_one")
    data = data.loc[data.global_cognition.notna()].copy()
    if len(data) != 109 or data.subject_id.nunique() != 109:
        raise RuntimeError("Formal Figure 7B cognition-complete cohort is not N=109")
    data.to_csv(a.output_dir / "figure7b_depth_band_participant_table.csv", index=False)

    rank_rows = []
    for band in BANDS:
        x = data.total_deviation_burden.to_numpy(float)
        y = data[f"{band}_burden"].to_numpy(float)
        spearman = stats.spearmanr(x, y)
        kendall = stats.kendalltau(x, y)
        total_rank = pd.Series(x).rank(method="average")
        band_rank = pd.Series(y).rank(method="average")
        cutoff = int(np.ceil(.25 * len(data)))
        total_top = set(np.argsort(x)[-cutoff:])
        band_top = set(np.argsort(y)[-cutoff:])
        rank_rows.append({
            "comparison": f"{band}_vs_total",
            "n_participants": len(data),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "kendall_tau": float(kendall.statistic),
            "kendall_p": float(kendall.pvalue),
            "median_absolute_rank_difference": float(np.median(np.abs(total_rank-band_rank))),
            "top_quartile_n": cutoff,
            "top_quartile_overlap_n": len(total_top & band_top),
            "top_quartile_jaccard": len(total_top & band_top) / len(total_top | band_top),
        })
    rank = pd.DataFrame(rank_rows)
    rank.to_csv(a.output_dir / "figure7b_depth_band_rank_agreement.csv", index=False)

    predictors = [("total_deviation_burden", "total"), *
                  [(f"{band}_burden", band) for band in BANDS]]
    results = []
    for predictor, label in predictors:
        results.append(fit(data, predictor, label, False))
        results.append(fit(data, predictor, label, True))
    pd.DataFrame(results).to_csv(
        a.output_dir / "figure7b_depth_band_cognition_results.csv", index=False
    )
    report = {
        "status": "complete", "analysis_n": len(data),
        "missing_global_cognition": sorted(set(total.subject_id) - set(data.subject_id)),
        "depth_direction": "pial_to_white",
        "depth_bands": {"outer": [0, 6], "middle": [7, 13], "inner": [14, 20]},
        "primary_comparison": "outer versus total burden",
        "standardization": "each burden standardized within the same N=109 cohort using population SD",
        "primary_covariates": ["age_at_scan", "categorical sex", "coverage_fraction"],
        "diagnosis_adjusted_sensitivity": True,
        "model_refit": False,
    }
    (a.output_dir / "figure7b_depth_band_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
