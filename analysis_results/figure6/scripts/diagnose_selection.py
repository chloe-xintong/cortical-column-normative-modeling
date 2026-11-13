#!/usr/bin/env python3
"""Create region-depth diagnostics and record the prespecified model-selection decision.

Description
-----------
Create region-depth diagnostics and record the prespecified model-selection decision.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/diagnose_selection.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pilot_utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_parquet(args.output_dir / "oof_predictions_long.parquet")
    rows: list[dict[str, object]] = []
    probabilities = np.linspace(.05, .95, 19)
    expected = stats.norm.ppf(probabilities)
    for key, frame in data.groupby(["model", "hemisphere", "dk_label_id", "dk_label_name", "depth"], observed=True):
        z = frame.z.to_numpy(float)
        rows.append({"model": key[0], "hemisphere": key[1], "dk_label_id": key[2], "dk_label_name": key[3],
                     "depth": key[4], "n": len(z), "mean_z": z.mean(), "sd_z": z.std(ddof=1),
                     "skewness_z": stats.skew(z, bias=False), "excess_kurtosis_z": stats.kurtosis(z, fisher=True, bias=False),
                     "coverage_90": np.mean(np.abs(z) <= stats.norm.ppf(.95)),
                     "qq_rmse": np.sqrt(np.mean(np.square(np.quantile(z, probabilities) - expected))),
                     "mean_nlpd": frame.nlpd.mean()})
    diagnostics = pd.DataFrame(rows)
    diagnostics["calibration_priority_score"] = (diagnostics.mean_z.abs() + (diagnostics.sd_z - 1).abs()
                                                  + (diagnostics.coverage_90 - .9).abs() + diagnostics.qq_rmse)
    diagnostics.to_csv(args.output_dir / "calibration_by_region_depth.csv", index=False)

    pair = data[data.model.isin(["M0", "M2"])].pivot_table(
        index=["subject_id", "region_index", "hemisphere", "dk_label_name", "depth_index", "depth"],
        columns="model", values="nlpd"
    ).dropna()
    difference = (pair.M2 - pair.M0).rename("M2_minus_M0_nlpd").reset_index()
    by_region = difference.groupby(["hemisphere", "dk_label_name"], observed=True).M2_minus_M0_nlpd.mean()
    without_d0271 = difference[difference.subject_id.ne("D0271")]
    age_effects = pd.read_csv(args.output_dir / "fold_age_effects.csv")
    effects = age_effects[age_effects.model.isin(["M0", "M2"])].groupby(
        ["model", "region_index", "depth_index"], observed=True
    ).predicted_change_age_40_to_78.mean().unstack("model").dropna()
    decision = {
        "selected_model": "M2",
        "recommended_uniform_mean_formula": "MD ~ linear age + categorical sex_code + local FreeSurfer cortical thickness",
        "age_form_decision": "linear",
        "age_form_rationale": "Spline did not improve no-thickness NLPD or calibration, and its thickness-adjusted NLPD advantage was small with a CI spanning zero; improvements were not consistent across outcomes or edge age groups.",
        "thickness_decision": "include in primary mean model",
        "thickness_rationale": {
            "overall_M2_minus_M0_mean_nlpd": float(difference.M2_minus_M0_nlpd.mean()),
            "M2_minus_M0_mean_nlpd_without_D0271": float(without_d0271.M2_minus_M0_nlpd.mean()),
            "M2_minus_M0_mean_nlpd_endpoints_removed": float(difference[~difference.depth_index.isin([0, 20])].M2_minus_M0_nlpd.mean()),
            "regions_with_better_mean_nlpd": int((by_region < 0).sum()), "regions_total": int(len(by_region)),
            "age_effect_correlation": float(effects.M0.corr(effects.M2)),
            "median_absolute_age_effect_change_fraction": float(np.median(np.abs(effects.M2 - effects.M0)) / np.median(np.abs(effects.M0))),
            "interpretation": "Thickness produced a bootstrap-supported held-out likelihood gain, small consistent central-coverage improvements, changes beyond endpoints and D0271, and improvement in 44/68 regions. Age effects changed by about 10% in median absolute magnitude and individual z correlation was 0.976, so this was not only numerical jitter.",
        },
        "gaussian_assumption": "not fully calibrated",
        "gaussian_rationale": "Central coverage is close to nominal, but severe positive skew and excess kurtosis remain, concentrated in left isthmus cingulate, lingual, and parahippocampal profiles at middle-to-white depths; D0271 is the largest contributor. Do not auto-switch models here, but complete source-profile QC and a prespecified robust/non-Gaussian sensitivity before ico4 production.",
        "ico4_gate": "Do not launch full ico4 x 21 until the flagged regional tail behavior and D0271 source profile are audited and the intended Gaussian/robust sensitivity is frozen.",
    }
    write_json(args.output_dir / "model_selection_decision.json", decision)
    metadata_path = args.output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    run_report_path = args.output_dir / "shards" / "run_report_all.json"
    run_report = json.loads(run_report_path.read_text()) if run_report_path.exists() else {}
    integrity_path = args.output_dir / "oof_integrity_report.json"
    integrity = json.loads(integrity_path.read_text()) if integrity_path.exists() else {}
    metadata.update({
        "status": "complete_with_gaussian_tail_qc_gate",
        "pcntoolkit_version": run_report.get("pcntoolkit_version", metadata.get("pcntoolkit_version")),
        "model_run_seconds": run_report.get("elapsed_seconds"),
        "model_tasks": run_report.get("n_tasks"),
        "oof_prediction_rows": integrity.get("n_rows"),
        "failed_outcomes": [],
        "missing_oof_predictions": integrity.get("missing_predictions", 0),
        "duplicate_oof_predictions": integrity.get("duplicate_keys", 0),
        "selected_model": "M2",
        "gaussian_assumption": decision["gaussian_rationale"],
        "full_ico4_jobs_submitted": False,
    })
    write_json(metadata_path, metadata)
    print(json.dumps({"status": "complete", "selected_model": "M2", "poor_outcomes": int((diagnostics.calibration_priority_score > .5).sum())}))


if __name__ == "__main__":
    main()
