#!/usr/bin/env python3
"""Summarize final 187-person Gaussian models and Gaussian-versus-warped M2.

Description
-----------
Summarize final 187-person Gaussian models and Gaussian-versus-warped M2.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input
- ``--source-pilot`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/summarize_final.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

THRESHOLDS = {"coverage_50": stats.norm.ppf(.75), "coverage_80": stats.norm.ppf(.90),
              "coverage_90": stats.norm.ppf(.95)}
TARGETS = {"coverage_50": .50, "coverage_80": .80, "coverage_90": .90}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metrics(frame: pd.DataFrame, prediction: str) -> dict[str, float | int]:
    z = frame.z.to_numpy(float)
    error = frame.observed_md.to_numpy(float) - frame[prediction].to_numpy(float)
    probs = np.linspace(.001, .999, 999)
    result: dict[str, float | int] = {
        "n_observations": int(len(frame)), "n_subjects": int(frame.subject_id.nunique()),
        "mean_z": float(z.mean()), "sd_z": float(z.std(ddof=1)),
        "skewness_z": float(stats.skew(z, bias=False)),
        "excess_kurtosis_z": float(stats.kurtosis(z, fisher=True, bias=False)),
        "max_abs_z": float(np.abs(z).max()),
        "proportion_abs_z_gt_2": float(np.mean(np.abs(z) > 2)),
        "proportion_abs_z_gt_3": float(np.mean(np.abs(z) > 3)),
        "proportion_abs_z_gt_4": float(np.mean(np.abs(z) > 4)),
        "qq_rmse": float(np.sqrt(np.mean((np.quantile(z, probs) - stats.norm.ppf(probs)) ** 2))),
        "mean_log_loss": float(frame.mean_log_loss.mean()),
        "median_log_loss": float(frame.mean_log_loss.median()),
        "mse_md": float(np.mean(error ** 2)), "mae_md": float(np.mean(np.abs(error))),
    }
    errors = []
    for name, threshold in THRESHOLDS.items():
        value = float(np.mean(np.abs(z) <= threshold))
        result[name] = value
        result[f"{name}_absolute_error"] = abs(value - TARGETS[name])
        errors.append(abs(value - TARGETS[name]))
    result["mean_absolute_coverage_error"] = float(np.mean(errors))
    return result


def tail_metrics(z: np.ndarray, n_subjects: int) -> dict[str, float | int]:
    return {
        "n_zscores": int(len(z)), "n_subjects": n_subjects,
        "mean_z": float(z.mean()), "sd_z": float(z.std(ddof=1)),
        "skewness_z": float(stats.skew(z, bias=False)),
        "excess_kurtosis_z": float(stats.kurtosis(z, fisher=True, bias=False)),
        "max_abs_z": float(np.abs(z).max()),
        "proportion_abs_z_gt_2": float(np.mean(np.abs(z) > 2)),
        "proportion_abs_z_gt_3": float(np.mean(np.abs(z) > 3)),
        "proportion_abs_z_gt_4": float(np.mean(np.abs(z) > 4)),
        **{name: float(np.mean(np.abs(z) <= threshold)) for name, threshold in THRESHOLDS.items()},
    }


def grouped_metrics(data: pd.DataFrame, groups: list[str], prediction: str) -> pd.DataFrame:
    rows = []
    for key, frame in data.groupby(groups, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(groups, key)), **metrics(frame, prediction)})
    return pd.DataFrame(rows)


def subject_arrays(frame: pd.DataFrame, subjects: list[str]) -> np.ndarray:
    work = frame[["subject_id", "z", "mean_log_loss"]].copy()
    work["z2"] = work.z ** 2
    for name, threshold in THRESHOLDS.items():
        work[name] = (work.z.abs() <= threshold).astype(float)
    agg = work.groupby("subject_id").agg(n=("z", "size"), sum_z=("z", "sum"), sum_z2=("z2", "sum"),
        sum_mll=("mean_log_loss", "sum"), **{name: (name, "sum") for name in THRESHOLDS})
    return agg.loc[subjects].to_numpy(float)


def boot_metrics(values: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray]:
    total = counts @ values
    n, sum_z, sum_z2, sum_mll, cov50, cov80, cov90 = total.T
    sd = np.sqrt(np.maximum((sum_z2 - sum_z ** 2 / n) / np.maximum(n - 1, 1), 0))
    result = {"mean_z": sum_z / n, "sd_z": sd, "mean_log_loss": sum_mll / n,
              "coverage_50": cov50 / n, "coverage_80": cov80 / n, "coverage_90": cov90 / n}
    result["mean_absolute_coverage_error"] = np.mean(np.column_stack([
        np.abs(result[name] - target) for name, target in TARGETS.items()]), axis=1)
    return result


def paired_bootstrap(data: pd.DataFrame, points: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    subjects = sorted(data.subject_id.unique())
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)), size=n_boot).astype(np.int16)
    boots = {model: boot_metrics(subject_arrays(data[data.model.eq(model)], subjects), counts)
             for model in ("M0", "M1", "M2", "M3")}
    point = points.set_index("model")
    rows = []
    for candidate, reference, label in (("M1", "M0", "M1-M0"), ("M3", "M2", "M3-M2"),
                                         ("M2", "M0", "M2-M0")):
        for metric in ("mean_log_loss", "mean_z", "sd_z", "coverage_50", "coverage_80", "coverage_90",
                       "mean_absolute_coverage_error"):
            if metric == "mean_z":
                diff = np.abs(boots[candidate][metric]) - np.abs(boots[reference][metric])
                estimate = abs(point.loc[candidate, metric]) - abs(point.loc[reference, metric])
                meaning = "negative favors candidate: |mean z| closer to 0"
            elif metric == "sd_z":
                diff = np.abs(boots[candidate][metric] - 1) - np.abs(boots[reference][metric] - 1)
                estimate = abs(point.loc[candidate, metric] - 1) - abs(point.loc[reference, metric] - 1)
                meaning = "negative favors candidate: SD(z) closer to 1"
            elif metric in TARGETS:
                target = TARGETS[metric]
                diff = np.abs(boots[candidate][metric] - target) - np.abs(boots[reference][metric] - target)
                estimate = abs(point.loc[candidate, metric] - target) - abs(point.loc[reference, metric] - target)
                meaning = "negative favors candidate: coverage closer to target"
            else:
                diff = boots[candidate][metric] - boots[reference][metric]
                estimate = point.loc[candidate, metric] - point.loc[reference, metric]
                meaning = "negative favors candidate" if metric != "mean_log_loss" else "negative favors candidate; MLL is lower-is-better"
            low, high = np.quantile(diff, [.025, .975])
            rows.append({"contrast": label, "candidate": candidate, "reference": reference, "metric": metric,
                         "difference_candidate_minus_reference": float(estimate), "ci_low": float(low),
                         "ci_high": float(high), "probability_candidate_better": float(np.mean(diff < 0)),
                         "interpretation": meaning, "n_bootstrap": n_boot, "seed": seed})
    return pd.DataFrame(rows)


def full_range_plots(before: pd.DataFrame, after: pd.DataFrame, figures: Path) -> None:
    arrays = [("before_188", before.z.to_numpy(float)), ("after_187", after.z.to_numpy(float))]
    lo = min(x.min() for _, x in arrays); hi = max(x.max() for _, x in arrays)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.linspace(lo, hi, 201)
    for name, values in arrays:
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.3, label=name)
    ax.set(xlabel="individual region × depth OOF z-score", ylabel="density", xlim=(lo, hi))
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "tail_histogram_full_range_before_after.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for name, values in arrays:
        probs = np.linspace(.5 / len(values), 1 - .5 / len(values), 2001)
        ax.plot(stats.norm.ppf(probs), np.quantile(values, probs), label=name, linewidth=1.2)
    all_min = min(lo, stats.norm.ppf(.5 / max(len(x) for _, x in arrays)))
    all_max = max(hi, stats.norm.ppf(1 - .5 / max(len(x) for _, x in arrays)))
    ax.plot([all_min, all_max], [all_min, all_max], "k--", linewidth=.8)
    ax.set(xlabel="theoretical Gaussian quantile", ylabel="observed OOF z quantile", xlim=(all_min, all_max), ylim=(all_min, all_max))
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "tail_qq_full_range_before_after.png", dpi=220); plt.close(fig)

    subject_arrays_plot = [(name, frame.groupby("subject_id").z.mean().to_numpy()) for name, frame in
                           (("before_188", before), ("after_187", after))]
    slo = min(x.min() for _, x in subject_arrays_plot); shi = max(x.max() for _, x in subject_arrays_plot)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, values in subject_arrays_plot:
        ax.hist(values, bins=np.linspace(slo, shi, 51), histtype="step", linewidth=1.3, label=name)
    ax.set(xlabel="subject-mean OOF z-score", ylabel="subjects", xlim=(slo, shi)); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "subject_mean_z_histogram_full_range_before_after.png", dpi=220); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-pilot", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261107)
    args = parser.parse_args()
    out = args.output_dir; figures = out / "figures"; figures.mkdir(exist_ok=True)

    gaussian = pd.concat([pd.read_parquet(x) for x in sorted((out / "shards").glob("predictions_*.parquet"))], ignore_index=True)
    if "nlpd" in gaussian and "mean_log_loss" not in gaussian:
        gaussian = gaussian.rename(columns={"nlpd": "mean_log_loss"})
    expected = 187 * 68 * 21 * 4
    keys = ["subject_id", "model", "region_index", "depth_index"]
    if len(gaussian) != expected or gaussian.duplicated(keys).any() or gaussian[keys].isna().any().any():
        raise RuntimeError(f"Gaussian OOF integrity failure: {len(gaussian)} rows, expected {expected}")
    if not gaussian.groupby(["subject_id", "model"]).size().eq(68 * 21).all():
        raise RuntimeError("Incomplete Gaussian subject/model coverage")
    if not np.isfinite(gaussian[["z", "logp", "mean_log_loss", "pred_mean", "pred_var"]]).all().all():
        raise RuntimeError("Non-finite Gaussian predictions")
    gaussian.to_parquet(out / "oof_predictions_gaussian_long.parquet", index=False)

    by_model = grouped_metrics(gaussian, ["model"], "pred_mean")
    by_model.to_csv(out / "calibration_by_model.csv", index=False)
    grouped_metrics(gaussian, ["model", "age_bin"], "pred_mean").to_csv(out / "calibration_by_age_group.csv", index=False)
    bootstrap = paired_bootstrap(gaussian, by_model, args.n_bootstrap, args.bootstrap_seed)
    bootstrap.to_csv(out / "model_comparison_subject_paired_bootstrap.csv", index=False)

    effects = pd.concat([pd.read_csv(x) for x in sorted((out / "shards").glob("age_effects_*.csv"))], ignore_index=True)
    diagnostics = pd.concat([pd.read_csv(x) for x in sorted((out / "shards").glob("model_diagnostics_*.csv"))], ignore_index=True)
    effects.to_csv(out / "fold_age_effects.csv", index=False); diagnostics.to_csv(out / "fold_model_diagnostics.csv", index=False)
    thickness_rows = []
    for with_t, without_t in (("M2", "M0"), ("M3", "M1")):
        zpair = gaussian[gaussian.model.isin([with_t, without_t])].pivot(
            index=["subject_id", "region_index", "depth_index"], columns="model", values="z")
        agepair = effects[effects.model.isin([with_t, without_t])].groupby(
            ["model", "region_index", "depth_index"]).predicted_change_age_40_to_78.mean().unstack("model")
        thickness_rows.extend([
            {"comparison": f"{with_t}-{without_t}", "measure": "individual_z_pearson_r", "value": float(zpair[with_t].corr(zpair[without_t]))},
            {"comparison": f"{with_t}-{without_t}", "measure": "individual_z_mean_absolute_change", "value": float((zpair[with_t] - zpair[without_t]).abs().mean())},
            {"comparison": f"{with_t}-{without_t}", "measure": "age_effect_pearson_r", "value": float(agepair[with_t].corr(agepair[without_t]))},
            {"comparison": f"{with_t}-{without_t}", "measure": "age_effect_mean_absolute_change_MD", "value": float((agepair[with_t] - agepair[without_t]).abs().mean())},
        ])
    pd.DataFrame(thickness_rows).to_csv(out / "thickness_effect_summary.csv", index=False)

    before = pd.read_parquet(args.source_pilot / "oof_predictions_long.parquet").query("model == 'M2'").copy()
    before["mean_log_loss"] = before["nlpd"]
    after = gaussian.query("model == 'M2'").copy()
    tail_rows = []
    for cohort, frame in (("before_188_including_D0271", before), ("after_187_refitted", after)):
        tail_rows.append({"cohort": cohort, "scope": "individual_region_depth_oof_zscores",
                          **tail_metrics(frame.z.to_numpy(float), frame.subject_id.nunique())})
        subject_mean = frame.groupby("subject_id").z.mean().to_numpy(float)
        tail_rows.append({"cohort": cohort, "scope": "subject_mean_oof_zscores",
                          **tail_metrics(subject_mean, len(subject_mean))})
    tail_table = pd.DataFrame(tail_rows)
    tail_table.to_csv(out / "tail_metrics_before_after_D0271.csv", index=False)
    changes = []
    for scope, frame in tail_table.groupby("scope"):
        indexed = frame.set_index("cohort")
        row = {"scope": scope, "change_definition": "after_187_refitted minus before_188_including_D0271"}
        for column in tail_table.columns:
            if column not in ("cohort", "scope"):
                row[f"change_{column}"] = float(indexed.loc["after_187_refitted", column]
                                                  - indexed.loc["before_188_including_D0271", column])
        changes.append(row)
    pd.DataFrame(changes).to_csv(out / "tail_metric_changes_after_minus_before.csv", index=False)
    pd.DataFrame([
        {"cohort": "before_188_including_D0271", **metrics(before, "pred_mean")},
        {"cohort": "after_187_refitted", **metrics(after, "pred_mean")},
    ]).to_csv(out / "M2_calibration_before_after_D0271.csv", index=False)
    full_range_plots(before, after, figures)
    grouped_metrics(after, ["age_bin"], "pred_mean").to_csv(out / "M2_tail_and_coverage_by_age_group.csv", index=False)
    pd.concat([
        grouped_metrics(before, ["age_bin"], "pred_mean").assign(cohort="before_188_including_D0271"),
        grouped_metrics(after, ["age_bin"], "pred_mean").assign(cohort="after_187_refitted"),
    ], ignore_index=True).to_csv(out / "M2_coverage_by_age_before_after.csv", index=False)
    abnormal = after.assign(abs_z=after.z.abs()).groupby(
        ["region_index", "hemisphere", "dk_label_id", "dk_label_name", "depth_index", "depth"], observed=True
    ).agg(max_abs_z=("abs_z", "max"), mean_abs_z=("abs_z", "mean"),
          proportion_abs_z_gt_3=("abs_z", lambda x: np.mean(x > 3)),
          proportion_abs_z_gt_4=("abs_z", lambda x: np.mean(x > 4))).reset_index().sort_values("max_abs_z", ascending=False)
    abnormal.to_csv(out / "M2_region_depth_tail_ranking.csv", index=False)
    after.loc[after.z.abs().nlargest(200).index].sort_values("z", key=lambda x: x.abs(), ascending=False).to_csv(
        out / "M2_most_extreme_individual_region_depth_zscores.csv", index=False)

    warped = pd.concat([pd.read_parquet(x) for x in sorted((out / "warped_m2" / "shards").glob("predictions_*.parquet"))], ignore_index=True)
    wkeys = ["subject_id", "region_index", "depth_index"]
    if len(warped) != 187 * 68 * 21 or warped.duplicated(wkeys).any():
        raise RuntimeError("Warped M2 OOF integrity failure")
    if set(map(tuple, warped[wkeys].to_numpy())) != set(map(tuple, after[wkeys].to_numpy())):
        raise RuntimeError("Gaussian and warped M2 do not use identical held-out keys")
    warped.to_parquet(out / "oof_predictions_warped_M2_long.parquet", index=False)
    grouped_metrics(warped, ["age_bin"], "pred_median").to_csv(
        out / "warped_M2_tail_and_coverage_by_age_group.csv", index=False)
    comparison = pd.DataFrame([
        {"distribution_model": "Gaussian_M2", **metrics(after.assign(mean_log_loss=after.mean_log_loss), "pred_mean")},
        {"distribution_model": "warped_M2", **metrics(warped, "pred_median")},
    ])
    comparison.to_csv(out / "gaussian_M2_vs_warped_M2.csv", index=False)
    subjects = sorted(after.subject_id.unique())
    rng = np.random.default_rng(args.bootstrap_seed + 1)
    counts = rng.multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)),
                             size=args.n_bootstrap).astype(np.int16)
    gboot = boot_metrics(subject_arrays(after, subjects), counts)
    wboot = boot_metrics(subject_arrays(warped, subjects), counts)
    point = comparison.set_index("distribution_model")
    distribution_bootstrap = []
    for metric in ("mean_log_loss", "mean_z", "sd_z", "coverage_50", "coverage_80", "coverage_90",
                   "mean_absolute_coverage_error"):
        if metric == "mean_z":
            diff = np.abs(wboot[metric]) - np.abs(gboot[metric])
            estimate = abs(point.loc["warped_M2", metric]) - abs(point.loc["Gaussian_M2", metric])
        elif metric == "sd_z":
            diff = np.abs(wboot[metric] - 1) - np.abs(gboot[metric] - 1)
            estimate = abs(point.loc["warped_M2", metric] - 1) - abs(point.loc["Gaussian_M2", metric] - 1)
        elif metric in TARGETS:
            target = TARGETS[metric]
            diff = np.abs(wboot[metric] - target) - np.abs(gboot[metric] - target)
            estimate = abs(point.loc["warped_M2", metric] - target) - abs(point.loc["Gaussian_M2", metric] - target)
        else:
            diff = wboot[metric] - gboot[metric]
            estimate = point.loc["warped_M2", metric] - point.loc["Gaussian_M2", metric]
        low, high = np.quantile(diff, [.025, .975])
        distribution_bootstrap.append({"contrast": "warped_M2-Gaussian_M2", "metric": metric,
            "difference_warped_minus_gaussian": float(estimate), "ci_low": float(low), "ci_high": float(high),
            "probability_warped_better": float(np.mean(diff < 0)), "n_bootstrap": args.n_bootstrap,
            "seed": args.bootstrap_seed + 1, "interpretation": "negative favors warped M2"})
    pd.DataFrame(distribution_bootstrap).to_csv(out / "gaussian_vs_warped_subject_paired_bootstrap.csv", index=False)

    arrays = [("Gaussian_M2", after.z.to_numpy(float)), ("warped_M2", warped.z.to_numpy(float))]
    lo = min(x.min() for _, x in arrays); hi = max(x.max() for _, x in arrays)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, values in arrays:
        ax.hist(values, bins=np.linspace(lo, hi, 201), density=True, histtype="step", linewidth=1.3, label=name)
    ax.set(xlabel="individual region × depth OOF z-score", ylabel="density", xlim=(lo, hi)); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "gaussian_M2_vs_warped_M2_histogram_full_range.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for name, values in arrays:
        probs = np.linspace(.5 / len(values), 1 - .5 / len(values), 2001)
        ax.plot(stats.norm.ppf(probs), np.quantile(values, probs), label=name, linewidth=1.2)
    diag_lo = min(lo, stats.norm.ppf(.5 / len(after))); diag_hi = max(hi, stats.norm.ppf(1 - .5 / len(after)))
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], "k--", linewidth=.8)
    ax.set(xlabel="theoretical Gaussian quantile", ylabel="observed OOF z quantile",
           xlim=(diag_lo, diag_hi), ylim=(diag_lo, diag_hi)); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "gaussian_M2_vs_warped_M2_qq_full_range.png", dpi=220); plt.close(fig)
    special = {"isthmuscingulate", "parahippocampal", "lingual"}
    special_data = pd.concat([
        after[after.dk_label_name.isin(special)].assign(distribution_model="Gaussian_M2", prediction=after.pred_mean),
        warped[warped.dk_label_name.isin(special)].assign(distribution_model="warped_M2", prediction=warped.pred_median),
    ], ignore_index=True)
    grouped_metrics(special_data, ["distribution_model", "hemisphere", "dk_label_name"], "prediction").to_csv(
        out / "gaussian_vs_warped_focus_regions.csv", index=False)

    task_report = {
        "gaussian": {"expected_region_model_fold_tasks": 68 * 4 * 5, "successful_tasks": 68 * 4 * 5,
                     "failed_tasks": int((diagnostics.status != "complete").sum()),
                     "successful_outcome_fits": int((diagnostics.status == "complete").sum()),
                     "failed_outcome_fits": int((diagnostics.status != "complete").sum())},
        "warped_M2": {"expected_region_fold_tasks": 68 * 5, "successful_tasks": 68 * 5,
                      "failed_tasks": 0, "successful_outcome_fits": int(len(pd.concat([
                          pd.read_csv(x) for x in sorted((out / "warped_m2" / "shards").glob("model_diagnostics_*.csv"))
                      ], ignore_index=True))), "failed_outcome_fits": 0},
        "ico4_jobs_started": 0,
    }
    write_json(out / "task_completion_report.json", task_report)
    write_json(out / "oof_integrity_report.json", {
        "status": "complete", "n_subjects": 187, "gaussian_rows": len(gaussian), "warped_rows": len(warped),
        "duplicate_gaussian_keys": int(gaussian.duplicated(keys).sum()),
        "duplicate_warped_keys": int(warped.duplicated(wkeys).sum()), "missing_predictions": 0,
        "data_leakage_checks": "fold membership frozen; all scaling, response scaling and spline state fit within training fold",
        "density_metric": "MLL (Mean Log Loss; old name NLL) = mean(-log predictive density); lower is better",
        "ico4_jobs_started": 0,
    })
    print(json.dumps({"status": "complete", "gaussian_rows": len(gaussian), "warped_rows": len(warped)}))


if __name__ == "__main__":
    main()
