#!/usr/bin/env python3
"""Merge OOF shards, compute calibration, and run 10,000 subject bootstraps.

Description
-----------
Merge OOF shards, compute calibration, and run 10,000 subject bootstraps.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/summarize_pilot.py --help
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from pilot_utils import (BOOTSTRAP_SEED, MODEL_SPECS, N_BOOTSTRAP, bootstrap_additive_metrics,
                         metric_point, percentile_interval, write_json)

CI_METRICS = ["mean_z", "sd_z", "coverage_50", "coverage_80", "coverage_90", "mean_nlpd", "mse_md", "mae_md"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def bootstrap_counts(subjects: list[str], n_bootstrap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(subjects)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_bootstrap).astype(np.int16)


def summarize_group(frame: pd.DataFrame, label: str, n_bootstrap: int, base_seed: int) -> dict[str, object]:
    point = metric_point(frame)
    subjects = sorted(frame.subject_id.unique())
    seed = base_seed + zlib.crc32(label.encode("utf-8")) % 1_000_000
    counts = bootstrap_counts(subjects, n_bootstrap, seed)
    boot = bootstrap_additive_metrics(frame, counts)
    for metric in CI_METRICS:
        low, high = percentile_interval(boot[metric])
        point[f"{metric}_ci_low"] = low
        point[f"{metric}_ci_high"] = high
    for metric, expected in (("coverage_50", 0.50), ("coverage_80", 0.80), ("coverage_90", 0.90)):
        error = np.abs(boot[metric] - expected)
        low, high = percentile_interval(error)
        point[f"{metric}_absolute_error_ci_low"] = low
        point[f"{metric}_absolute_error_ci_high"] = high
    return point


def grouped_summary(data: pd.DataFrame, columns: list[str], n_bootstrap: int, seed: int,
                    bootstrap: bool = True) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["model", *columns]
    for key, frame in data.groupby(group_columns, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        identity = "|".join(map(str, key))
        metrics = summarize_group(frame, identity, n_bootstrap, seed) if bootstrap else metric_point(frame)
        rows.append({**dict(zip(group_columns, key)), **metrics})
    return pd.DataFrame(rows)


def calibration_score(metrics: dict[str, np.ndarray] | pd.Series) -> np.ndarray | float:
    return (np.abs(metrics["mean_z"]) + np.abs(metrics["sd_z"] - 1.0)
            + (np.abs(metrics["coverage_50"] - 0.50) + np.abs(metrics["coverage_80"] - 0.80)
               + np.abs(metrics["coverage_90"] - 0.90)) / 3.0)


def comparison_rows(data: pd.DataFrame, n_bootstrap: int, seed: int) -> list[dict[str, object]]:
    subjects = sorted(data.subject_id.unique())
    counts = bootstrap_counts(subjects, n_bootstrap, seed)
    model_boot: dict[str, dict[str, np.ndarray]] = {}
    model_point: dict[str, dict[str, float | int]] = {}
    outcome = data.groupby(["model", "region_index", "depth_index"], observed=True).agg(
        mean_nlpd=("nlpd", "mean"), mean_z=("z", "mean"), sd_z=("z", "std"),
        coverage_50=("z", lambda x: np.mean(np.abs(x) <= 0.6744897501960817)),
        coverage_80=("z", lambda x: np.mean(np.abs(x) <= 1.2815515655446004)),
        coverage_90=("z", lambda x: np.mean(np.abs(x) <= 1.6448536269514722)),
    ).reset_index()
    outcome["calibration_score"] = calibration_score(outcome)
    for model in MODEL_SPECS:
        frame = data[data.model.eq(model)]
        model_boot[model] = bootstrap_additive_metrics(frame, counts)
        model_point[model] = metric_point(frame)
    pairs = [("M1", "M0", "spline_vs_linear_without_thickness"),
             ("M3", "M2", "spline_vs_linear_with_thickness"),
             ("M2", "M0", "thickness_vs_none_linear"),
             ("M3", "M1", "thickness_vs_none_spline")]
    rows: list[dict[str, object]] = []
    for candidate, reference, comparison in pairs:
        candidate_outcome = outcome[outcome.model.eq(candidate)].set_index(["region_index", "depth_index"])
        reference_outcome = outcome[outcome.model.eq(reference)].set_index(["region_index", "depth_index"])
        for metric in ("mean_nlpd", "mean_z", "sd_z", "coverage_50", "coverage_80", "coverage_90"):
            candidate_values = model_boot[candidate][metric]
            reference_values = model_boot[reference][metric]
            if metric == "mean_z":
                difference = np.abs(candidate_values) - np.abs(reference_values)
                point = abs(float(model_point[candidate][metric])) - abs(float(model_point[reference][metric]))
                interpretation = "negative favors candidate (smaller absolute mean z)"
            elif metric == "sd_z":
                difference = np.abs(candidate_values - 1) - np.abs(reference_values - 1)
                point = abs(float(model_point[candidate][metric]) - 1) - abs(float(model_point[reference][metric]) - 1)
                interpretation = "negative favors candidate (SD closer to one)"
            elif metric.startswith("coverage"):
                expected = {"coverage_50": 0.5, "coverage_80": 0.8, "coverage_90": 0.9}[metric]
                difference = np.abs(candidate_values - expected) - np.abs(reference_values - expected)
                point = abs(float(model_point[candidate][metric]) - expected) - abs(float(model_point[reference][metric]) - expected)
                interpretation = "negative favors candidate (coverage closer to target)"
            else:
                difference = candidate_values - reference_values
                point = float(model_point[candidate][metric]) - float(model_point[reference][metric])
                interpretation = "negative favors candidate"
            low, high = percentile_interval(difference)
            outcome_difference = candidate_outcome[metric] - reference_outcome[metric]
            if metric == "mean_z":
                outcome_difference = candidate_outcome[metric].abs() - reference_outcome[metric].abs()
            elif metric == "sd_z":
                outcome_difference = (candidate_outcome[metric] - 1).abs() - (reference_outcome[metric] - 1).abs()
            elif metric.startswith("coverage"):
                expected = {"coverage_50": 0.5, "coverage_80": 0.8, "coverage_90": 0.9}[metric]
                outcome_difference = (candidate_outcome[metric] - expected).abs() - (reference_outcome[metric] - expected).abs()
            rows.append({"comparison": comparison, "candidate": candidate, "reference": reference, "metric": metric,
                         "difference_candidate_minus_reference": point, "ci_low": low, "ci_high": high,
                         "bootstrap_probability_candidate_better": float(np.mean(difference < 0)),
                         "fraction_region_depth_outcomes_candidate_better": float(np.mean(outcome_difference < 0)),
                         "interpretation": interpretation, "n_bootstrap": n_bootstrap, "seed": seed})
        candidate_cal = calibration_score(model_boot[candidate])
        reference_cal = calibration_score(model_boot[reference])
        difference = candidate_cal - reference_cal
        point = float(calibration_score(model_point[candidate]) - calibration_score(model_point[reference]))
        low, high = percentile_interval(difference)
        outcome_difference = candidate_outcome.calibration_score - reference_outcome.calibration_score
        rows.append({"comparison": comparison, "candidate": candidate, "reference": reference,
                     "metric": "calibration_score", "difference_candidate_minus_reference": point,
                     "ci_low": low, "ci_high": high,
                     "bootstrap_probability_candidate_better": float(np.mean(difference < 0)),
                     "fraction_region_depth_outcomes_candidate_better": float(np.mean(outcome_difference < 0)),
                     "interpretation": "negative favors candidate", "n_bootstrap": n_bootstrap, "seed": seed})
    return rows


def thickness_summary(data: pd.DataFrame, age_effects: pd.DataFrame, diagnostics: pd.DataFrame,
                      comparisons: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for without, with_thickness, age_form in (("M0", "M2", "linear"), ("M1", "M3", "spline_df3")):
        pair = data[data.model.isin([without, with_thickness])].pivot_table(
            index=["subject_id", "region_index", "depth_index"], columns="model", values="z"
        ).dropna()
        endpoint_mask = ~pair.index.get_level_values("depth_index").isin([0, 20])
        rows.extend([
            {"comparison": f"{with_thickness}_vs_{without}", "measure": "individual_z_pearson_r_all",
             "value": float(pair[without].corr(pair[with_thickness]))},
            {"comparison": f"{with_thickness}_vs_{without}", "measure": "individual_z_pearson_r_endpoints_removed",
             "value": float(pair.loc[endpoint_mask, without].corr(pair.loc[endpoint_mask, with_thickness]))},
        ])
        effects = age_effects[age_effects.model.isin([without, with_thickness])].groupby(
            ["model", "region_index", "depth_index"], observed=True
        ).predicted_change_age_40_to_78.mean().unstack("model").dropna()
        rows.append({"comparison": f"{with_thickness}_vs_{without}", "measure": "age_effect_pearson_r",
                     "value": float(effects[without].corr(effects[with_thickness]))})
        for metric in ("mean_nlpd", "calibration_score", "mean_z", "sd_z", "coverage_50", "coverage_80", "coverage_90"):
            match = comparisons[(comparisons.candidate.eq(with_thickness)) & (comparisons.reference.eq(without))
                                & (comparisons.metric.eq(metric))].iloc[0]
            rows.append({"comparison": f"{with_thickness}_vs_{without}", "measure": f"bootstrap_difference_{metric}",
                         "value": match.difference_candidate_minus_reference, "ci_low": match.ci_low,
                         "ci_high": match.ci_high})
        beta = pd.to_numeric(diagnostics.loc[diagnostics.model.eq(with_thickness), "raw_thickness_beta"], errors="coerce").dropna()
        for name, value in (("thickness_beta_median", beta.median()), ("thickness_beta_q25", beta.quantile(.25)),
                            ("thickness_beta_q75", beta.quantile(.75)), ("thickness_beta_min", beta.min()),
                            ("thickness_beta_max", beta.max())):
            rows.append({"comparison": f"{with_thickness}_vs_{without}", "measure": name, "value": float(value)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    prediction_files = sorted((args.output_dir / "shards").glob("predictions_*.parquet"))
    if not prediction_files:
        raise FileNotFoundError("No OOF prediction shards found")
    data = pd.concat([pd.read_parquet(path) for path in prediction_files], ignore_index=True)
    age_effects = pd.concat([pd.read_csv(path) for path in sorted((args.output_dir / "shards").glob("age_effects_*.csv"))], ignore_index=True)
    diagnostics = pd.concat([pd.read_csv(path) for path in sorted((args.output_dir / "shards").glob("model_diagnostics_*.csv"))], ignore_index=True)
    expected_rows = 188 * 68 * 21 * 4
    if len(data) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows:,} OOF rows; found {len(data):,}")
    keys = ["subject_id", "model", "region_index", "depth_index"]
    duplicates = data.duplicated(keys, keep=False)
    if duplicates.any() or data[keys].isna().any().any():
        raise RuntimeError("Duplicate or missing OOF prediction keys")
    counts = data.groupby(["subject_id", "model"], observed=True).size()
    if not counts.eq(68 * 21).all():
        raise RuntimeError("At least one subject/model lacks exactly one prediction per outcome")
    numeric = ["observed_md", "pred_mean", "pred_var", "noise_var", "parameter_var", "z", "logp", "nlpd"]
    if not np.isfinite(data[numeric]).all().all() or (data.pred_var <= 0).any():
        raise RuntimeError("OOF prediction table contains invalid numeric values")
    data["squared_error"] = np.square(data.observed_md - data.pred_mean)
    data["absolute_error"] = np.abs(data.observed_md - data.pred_mean)
    data.to_parquet(args.output_dir / "oof_predictions_long.parquet", index=False)
    z_columns = ["subject_id", "age", "age_bin", "sex_code", "outer_fold", "model", "region_index", "hemisphere",
                 "dk_label_id", "dk_label_name", "depth_index", "depth", "depth_band", "z", "pred_var", "logp"]
    data[z_columns].to_parquet(args.output_dir / "oof_zscores_long.parquet", index=False)

    by_model = grouped_summary(data, [], args.n_bootstrap, args.bootstrap_seed, bootstrap=True)
    by_model.to_csv(args.output_dir / "calibration_by_model.csv", index=False)
    grouped_summary(data, ["age_bin"], args.n_bootstrap, args.bootstrap_seed, bootstrap=True).to_csv(
        args.output_dir / "calibration_by_age_group.csv", index=False)
    grouped_summary(data, ["sex_code"], args.n_bootstrap, args.bootstrap_seed, bootstrap=True).to_csv(
        args.output_dir / "calibration_by_sex_code.csv", index=False)
    grouped_summary(data, ["depth_band"], args.n_bootstrap, args.bootstrap_seed, bootstrap=True).to_csv(
        args.output_dir / "calibration_by_depth_band.csv", index=False)
    grouped_summary(data, ["hemisphere", "dk_label_id", "dk_label_name"], args.n_bootstrap,
                    args.bootstrap_seed, bootstrap=False).to_csv(args.output_dir / "calibration_by_region.csv", index=False)

    comparisons = pd.DataFrame(comparison_rows(data, args.n_bootstrap, args.bootstrap_seed))
    comparisons.to_csv(args.output_dir / "model_comparison_bootstrap.csv", index=False)
    sensitivity = thickness_summary(data, age_effects, diagnostics, comparisons)
    sensitivity.to_csv(args.output_dir / "thickness_sensitivity_summary.csv", index=False)
    age_effects.to_csv(args.output_dir / "fold_age_effects.csv", index=False)
    diagnostics.to_csv(args.output_dir / "fold_model_diagnostics.csv", index=False)

    age_group_nlpd = data.groupby(["model", "age_bin"], observed=True).nlpd.mean().unstack("model")
    evidence = {
        "age_form": {
            "comparisons": comparisons[comparisons.comparison.str.startswith("spline_vs_linear")].to_dict("records"),
            "mean_nlpd_by_age_group": age_group_nlpd.to_dict(orient="index"),
        },
        "thickness": {"comparisons": comparisons[comparisons.comparison.str.startswith("thickness_vs_none")].to_dict("records")},
        "selection_policy": "Prefer the simpler model unless held-out NLPD and calibration improve consistently under subject bootstrap and across outcomes; inspect edge age groups and endpoints before final selection.",
    }
    write_json(args.output_dir / "model_selection_evidence.json", evidence)
    write_json(args.output_dir / "oof_integrity_report.json", {
        "status": "complete", "n_rows": len(data), "n_subjects": data.subject_id.nunique(),
        "n_models": data.model.nunique(), "n_outcomes_per_subject_model": 68 * 21,
        "duplicate_keys": int(duplicates.sum()), "missing_predictions": 0, "invalid_predictions": 0,
        "model_failures": int((diagnostics.status != "complete").sum()), "elapsed_summary_seconds": time.monotonic() - started,
        "bootstrap_repetitions": args.n_bootstrap, "bootstrap_seed": args.bootstrap_seed,
        "regional_CI_note": "Region-level point estimates are saved; CIs are supplied for overall, age, sex, and depth-band summaries and every predefined model contrast.",
    })
    print(json.dumps({"status": "complete", "rows": len(data), "elapsed_seconds": time.monotonic() - started}))


if __name__ == "__main__":
    main()
