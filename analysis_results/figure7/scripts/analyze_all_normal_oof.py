#!/usr/bin/env python3
"""Analyze cognition against total deviation burden in all 186 OOF CN.

Description
-----------
Analyze cognition against total deviation burden in all 186 OOF CN.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--figure6-root`` command-line input
- ``--cohort-flow`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure7/scripts/analyze_all_normal_oof.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure6-root", type=Path, required=True)
    parser.add_argument("--cohort-flow", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def result_row(model, sensitivity: str, burden_mean: float, burden_sd: float) -> dict[str, object]:
    ci = model.conf_int().loc["burden_z"]
    return {
        "analysis": "all_normal_five_fold_oof",
        "sensitivity": sensitivity,
        "n_participants": int(model.nobs),
        "estimate": float(model.params["burden_z"]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "p_value": float(model.pvalues["burden_z"]),
        "standard_error_hc3": float(model.bse["burden_z"]),
        "r_squared": float(model.rsquared),
        "burden_mean": burden_mean,
        "burden_sd": burden_sd,
        "formula": model.model.formula,
        "covariance": "HC3",
        "statistical_unit": "participant",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(
        args.figure6_root / "frozen_cohort_manifest.csv", dtype={"subject_id": str}
    )
    cn = manifest.loc[manifest.included].copy()
    if len(cn) != 186 or cn.subject_id.nunique() != 186:
        raise RuntimeError("Expected the frozen 186-person Figure 6 CN cohort")
    if set(cn.diagnosis_at_scan) != {"Normal"}:
        raise RuntimeError("Frozen included cohort contains a non-Normal diagnosis")

    assignments = json.loads((args.figure6_root / "fold_assignments.json").read_text())
    fold_rows = []
    for fold in assignments["folds"]:
        fold_rows.extend(
            {"subject_id": subject_id, "outer_fold": int(fold["outer_fold"])}
            for subject_id in fold["test_subject_ids"]
        )
    fold_map = pd.DataFrame(fold_rows)
    if len(fold_map) != 186 or fold_map.subject_id.nunique() != 186:
        raise RuntimeError("Every frozen CN must occur in exactly one held-out fold")

    aggregates = pd.read_parquet(
        args.figure6_root / "oof" / "subject_stratum_aggregates.parquet"
    )
    aggregates = aggregates.loc[
        aggregates.scope.eq("overall") & aggregates.level.eq("all"),
        ["subject_id", "n", "gt2"],
    ].copy()
    aggregates = aggregates.rename(
        columns={"n": "n_valid_positions", "gt2": "n_total_abs_z_gt_2"}
    )
    if len(aggregates) != 186 or aggregates.subject_id.nunique() != 186:
        raise RuntimeError("Expected one overall OOF aggregate per frozen CN")
    aggregates["n_valid_positions"] = aggregates.n_valid_positions.astype(int)
    aggregates["n_total_abs_z_gt_2"] = aggregates.n_total_abs_z_gt_2.astype(int)
    aggregates["total_deviation_burden"] = (
        aggregates.n_total_abs_z_gt_2 / aggregates.n_valid_positions
    )
    depth = pd.read_parquet(
        args.figure6_root / "oof" / "subject_stratum_aggregates.parquet"
    )
    depth = depth.loc[depth.scope.eq("depth"), ["subject_id", "level", "n", "gt2"]].copy()
    depth["depth_index"] = pd.to_numeric(depth.level, errors="raise").astype(int)
    depth["depth_band"] = pd.cut(
        depth.depth_index, bins=[-1, 6, 13, 20], labels=["outer", "middle", "inner"]
    )
    depth_bands = depth.groupby(["subject_id", "depth_band"], observed=True)[["n", "gt2"]].sum()
    depth_bands["burden"] = depth_bands.gt2 / depth_bands.n
    depth_bands = depth_bands.burden.unstack("depth_band").reset_index().rename(columns={
        "outer": "outer_deviation_burden",
        "middle": "middle_deviation_burden",
        "inner": "inner_deviation_burden",
    })
    if len(depth_bands) != 186 or depth_bands.isna().any().any():
        raise RuntimeError("Expected complete outer/middle/inner OOF burden for all 186 CN")

    cohort = pd.read_csv(args.cohort_flow, dtype={"subject_id": str})
    keep = [
        "subject_id", "age_at_scan", "SUBJECT_SEX", "coverage_fraction",
        "diagnosis_at_scan", "global_cognition", "coverage_ge_0_98",
    ]
    data = (
        cn[["subject_id", "age", "sex_code"]]
        .merge(fold_map, on="subject_id", how="left", validate="one_to_one")
        .merge(aggregates, on="subject_id", how="left", validate="one_to_one")
        .merge(depth_bands, on="subject_id", how="left", validate="one_to_one")
        .merge(cohort[keep], on="subject_id", how="left", validate="one_to_one")
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    required = [
        "outer_fold", "n_valid_positions", "total_deviation_burden",
        "age_at_scan", "SUBJECT_SEX", "coverage_fraction", "global_cognition",
    ]
    if data[required].isna().any().any():
        missing = data.loc[data[required].isna().any(axis=1), "subject_id"].tolist()
        raise RuntimeError(f"Missing required all-CN data: {missing}")
    if not data.age_at_scan.between(41, 78).all():
        raise RuntimeError("Frozen CN fall outside the prespecified age range")
    if set(data.diagnosis_at_scan) != {"Normal"}:
        raise RuntimeError("Merged analysis data are not all Normal")
    if not np.allclose(data.age, data.age_at_scan) or not np.array_equal(
        data.sex_code.to_numpy(), data.SUBJECT_SEX.to_numpy()
    ):
        raise RuntimeError("Figure 6 and Figure 7 covariates disagree")

    burden_mean = float(data.total_deviation_burden.mean())
    burden_sd = float(data.total_deviation_burden.std(ddof=0))
    data["burden_z"] = (data.total_deviation_burden - burden_mean) / burden_sd
    primary_formula = (
        "global_cognition ~ burden_z + age_at_scan + C(SUBJECT_SEX) + coverage_fraction"
    )
    fold_formula = primary_formula + " + C(outer_fold)"
    primary = smf.ols(primary_formula, data=data).fit(cov_type="HC3")
    fold_adjusted = smf.ols(fold_formula, data=data).fit(cov_type="HC3")

    data.to_csv(args.output_dir / "all_normal_oof_burden_cognition.csv", index=False)
    results = pd.DataFrame([
        result_row(primary, "primary_all_normal_oof", burden_mean, burden_sd),
        result_row(fold_adjusted, "fold_fixed_effect_sensitivity", burden_mean, burden_sd),
    ])
    results.to_csv(args.output_dir / "all_normal_oof_cognition_results.csv", index=False)

    report = {
        "status": "complete",
        "analysis": "all Normal frozen Figure 6 reference participants",
        "n_participants": len(data),
        "global_cognition_n": int(data.global_cognition.notna().sum()),
        "age_range": [float(data.age_at_scan.min()), float(data.age_at_scan.max())],
        "fold_test_sizes": {
            str(int(k)): int(v) for k, v in data.outer_fold.value_counts().sort_index().items()
        },
        "z_source": "saved Figure 6 five-fold OOF predictions",
        "per_subject_rule": "exactly one held-out fold per participant; no cross-fold averaging",
        "model_refit": False,
        "primary_burden": "fraction of valid OOF positions with absolute z greater than 2",
        "depth_bands": {"outer": [0, 6], "middle": [7, 13], "inner": [14, 20]},
        "primary_formula": primary_formula,
        "fold_sensitivity_formula": fold_formula,
    }
    (args.output_dir / "all_normal_oof_validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
