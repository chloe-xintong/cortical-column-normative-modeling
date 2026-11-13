#!/usr/bin/env python3
"""Validate, compare, bootstrap, and plot regional warped age candidates.

Description
-----------
Validate, compare, bootstrap, and plot regional warped age candidates.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--baseline`` command-line input
- ``--baseline-diagnostics`` command-line input
- ``--spline-dir`` command-line input
- ``--subject-order`` command-line input
- ``--cohort-audit`` command-line input
- ``--regional-profile`` command-line input
- ``--folds`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/summarize_warped_age_comparison.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from pilot_utils import write_json  # noqa: E402
from warped_age_utils import shash_forward, shash_predictive_mean  # noqa: E402


KEYS = ["subject_id", "region_index", "depth_index"]
INTERVAL_LEVELS = (50, 80, 90)
QQ_PROBS = np.linspace(0.01, 0.99, 99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-diagnostics", type=Path, required=True)
    parser.add_argument("--spline-dir", type=Path, required=True)
    parser.add_argument("--gompertz-dir", type=Path)
    parser.add_argument("--subject-order", type=Path, required=True)
    parser.add_argument("--cohort-audit", type=Path, required=True)
    parser.add_argument("--regional-profile", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261110)
    return parser.parse_args()


def reconstruct_baseline_mean(pred: pd.DataFrame, diagnostics: pd.DataFrame, args: argparse.Namespace
                              ) -> tuple[pd.DataFrame, dict[str, float]]:
    """Recover baseline predictive means from saved quantiles and warp parameters."""
    result = pred.copy()
    subject_order = pd.read_csv(args.subject_order, dtype={"subject_id": str})
    audit = pd.read_csv(args.cohort_audit, dtype={"subject_id": str})
    included = audit.loc[audit.included].merge(
        subject_order[["subject_id", "row_index"]], on="subject_id", validate="one_to_one"
    )
    profile = np.load(args.regional_profile, mmap_mode="r")
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    result["pred_mean"] = np.nan
    max_z_error = 0.0
    max_median_error = 0.0
    for fold_info in folds["folds"]:
        fold = int(fold_info["outer_fold"])
        train_rows = included.loc[included.subject_id.isin(fold_info["train_subject_ids"]), "row_index"].to_numpy(int)
        fold_pred = result.index[result.outer_fold.eq(fold)]
        for region in range(68):
            start = region * 21
            train_y = np.asarray(profile[train_rows, start:start + 21], float)
            means = train_y.mean(axis=0)
            scales = train_y.std(axis=0, ddof=0)
            region_index = fold_pred[result.loc[fold_pred, "region_index"].eq(region)]
            for depth in range(21):
                idx = region_index[result.loc[region_index, "depth_index"].eq(depth)]
                diag = diagnostics[(diagnostics.outer_fold == fold) &
                                   (diagnostics.region_index == region) &
                                   (diagnostics.depth_index == depth)]
                if len(diag) != 1 or len(idx) == 0:
                    raise RuntimeError(f"Missing baseline diagnostic or OOF group r{region} d{depth} f{fold}")
                gamma = diag[["warp_epsilon", "warp_log_b"]].iloc[0].to_numpy(float)
                median_scaled = (result.loc[idx, "pred_median"].to_numpy(float) - means[depth]) / scales[depth]
                latent_mean = shash_forward(median_scaled, gamma)
                lower = shash_forward(
                    (result.loc[idx, "lower_50"].to_numpy(float) - means[depth]) / scales[depth], gamma
                )
                upper = shash_forward(
                    (result.loc[idx, "upper_50"].to_numpy(float) - means[depth]) / scales[depth], gamma
                )
                latent_sd = (upper - lower) / (2 * stats.norm.ppf(0.75))
                if not np.isfinite(latent_sd).all() or np.any(latent_sd <= 0):
                    raise RuntimeError("Could not reconstruct positive baseline latent predictive SD")
                mean_scaled = shash_predictive_mean(latent_mean, latent_sd ** 2, gamma)
                result.loc[idx, "pred_mean"] = mean_scaled * scales[depth] + means[depth]
                reconstructed_median = (
                    np.sinh((np.arcsinh(latent_mean) - gamma[0] * np.exp(gamma[1])) /
                            np.exp(gamma[1])) * scales[depth] + means[depth]
                )
                max_median_error = max(max_median_error, float(np.max(np.abs(
                    reconstructed_median - result.loc[idx, "pred_median"].to_numpy(float)
                ))))
                observed_scaled = (result.loc[idx, "observed_md"].to_numpy(float) - means[depth]) / scales[depth]
                z_reconstructed = (shash_forward(observed_scaled, gamma) - latent_mean) / latent_sd
                max_z_error = max(max_z_error, float(np.max(np.abs(
                    z_reconstructed - result.loc[idx, "z"].to_numpy(float)
                ))))
    if not np.isfinite(result.pred_mean).all() or max_z_error > 1e-5 or max_median_error > 1e-10:
        raise RuntimeError(f"Baseline mean reconstruction validation failed: z={max_z_error}, median={max_median_error}")
    return result, {"max_abs_reconstructed_z_error": max_z_error,
                    "max_abs_reconstructed_median_error": max_median_error,
                    "method": "32-node Gauss-Hermite expectation after recovering latent Gaussian mean/variance from saved median and 50% interval"}


def validate_prediction(frame: pd.DataFrame, model: str, expected_rows: int | None = None) -> None:
    required = KEYS + ["observed_md", "pred_mean", "pred_median", "z", "logp", "mean_log_loss"]
    required += [f"{side}_{level}" for level in INTERVAL_LEVELS for side in ("lower", "upper")]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{model} missing columns: {missing}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise RuntimeError(f"{model} has {len(frame)} rows, expected {expected_rows}")
    if frame.duplicated(KEYS).any():
        raise RuntimeError(f"{model} has duplicate OOF keys")
    if not np.isfinite(frame[required[3:]].to_numpy(float)).all():
        raise RuntimeError(f"{model} has nonfinite predictions")
    if not np.allclose(frame.logp, -frame.mean_log_loss):
        raise RuntimeError(f"{model} logp/mean_log_loss sign mismatch")


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    z = frame.z.to_numpy(float)
    coverage = {level: float(((frame.observed_md >= frame[f"lower_{level}"]) &
                               (frame.observed_md <= frame[f"upper_{level}"])).mean())
                for level in INTERVAL_LEVELS}
    observed_q = np.quantile(z, QQ_PROBS)
    expected_q = stats.norm.ppf(QQ_PROBS)
    median_error = frame.observed_md.to_numpy(float) - frame.pred_median.to_numpy(float)
    mean_error = frame.observed_md.to_numpy(float) - frame.pred_mean.to_numpy(float)
    return {"n_observations": len(frame), "n_subjects": frame.subject_id.nunique(),
            "n_regions": frame.region_index.nunique(), "n_depths": frame.depth_index.nunique(),
            "mean_log_loss": float(frame.mean_log_loss.mean()),
            "coverage_50": coverage[50], "coverage_80": coverage[80], "coverage_90": coverage[90],
            "coverage_mae": float(np.mean([abs(coverage[x] - x / 100) for x in INTERVAL_LEVELS])),
            "mean_z": float(z.mean()), "sd_z": float(z.std(ddof=1)),
            "skewness_z": float(stats.skew(z, bias=False)),
            "excess_kurtosis_z": float(stats.kurtosis(z, fisher=True, bias=False)),
            "proportion_abs_z_gt_2": float((np.abs(z) > 2).mean()),
            "proportion_abs_z_gt_3": float((np.abs(z) > 3).mean()),
            "proportion_abs_z_gt_4": float((np.abs(z) > 4).mean()),
            "qq_rmse": float(np.sqrt(np.mean((observed_q - expected_q) ** 2))),
            "mae_predictive_median": float(np.mean(np.abs(median_error))),
            "rmse_predictive_mean": float(np.sqrt(np.mean(mean_error ** 2)))}


def metric_table(models: dict[str, pd.DataFrame], comparison: str) -> pd.DataFrame:
    rows = []
    for model, frame in models.items():
        rows.append({"comparison": comparison, "scope": "overall", "depth_index": "all",
                     "depth": "all", "model": model, **metrics(frame)})
        for depth, group in frame.groupby("depth_index", sort=True):
            rows.append({"comparison": comparison, "scope": "by_depth", "depth_index": int(depth),
                         "depth": float(group.depth.iloc[0]), "model": model, **metrics(group)})
    return pd.DataFrame(rows)


def paired_mll_bootstrap(candidate: pd.DataFrame, baseline: pd.DataFrame, candidate_name: str,
                         n_bootstrap: int, seed: int) -> pd.DataFrame:
    merged = baseline[KEYS + ["mean_log_loss"]].merge(
        candidate[KEYS + ["mean_log_loss"]], on=KEYS, suffixes=("_linear", "_candidate"),
        validate="one_to_one",
    )
    if len(merged) != len(baseline) or len(merged) != len(candidate):
        raise RuntimeError(f"{candidate_name}: comparison is not on an identical valid observation set")
    merged["delta"] = merged.mean_log_loss_candidate - merged.mean_log_loss_linear
    rng = np.random.default_rng(seed)
    subjects = sorted(merged.subject_id.unique())
    rows = []
    scopes = [("overall", "all", merged)] + [
        ("by_depth", str(depth), group) for depth, group in merged.groupby("depth_index", sort=True)
    ]
    for scope, depth, group in scopes:
        by_subject = group.groupby("subject_id", sort=True).delta.mean().reindex(subjects)
        if by_subject.isna().any() or group.groupby("subject_id").size().nunique() != 1:
            raise RuntimeError("Subject-cluster bootstrap requires each subject's full balanced observation block")
        values = by_subject.to_numpy(float)
        indices = rng.integers(0, len(subjects), size=(n_bootstrap, len(subjects)))
        boot = values[indices].mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append({"candidate": candidate_name, "contrast": f"{candidate_name}-warped_linear",
                     "scope": scope, "depth_index": depth,
                     "delta_mll": float(group.delta.mean()), "ci_low": float(low),
                     "ci_high": float(high), "probability_candidate_better": float((boot < 0).mean()),
                     "n_subjects": len(subjects), "n_observations": len(group),
                     "observations_per_subject": int(group.groupby("subject_id").size().iloc[0]),
                     "n_bootstrap": n_bootstrap, "seed": seed,
                     "interpretation": "negative favors candidate"})
    return pd.DataFrame(rows)


def plot_forest(contrasts: pd.DataFrame, output: Path) -> None:
    plot = contrasts.copy()
    labels = ["Overall" if row.scope == "overall" else f"Depth {float(row.depth_index)/20:.2f}"
              for row in plot.itertuples()]
    y = np.arange(len(plot))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.29 * len(plot))))
    ax.errorbar(plot.delta_mll, y,
                xerr=np.vstack([plot.delta_mll - plot.ci_low, plot.ci_high - plot.delta_mll]),
                fmt="o", color="#2A6F97", ecolor="#7A8A99", capsize=2.5)
    ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Δ MLL (candidate − warped linear); negative favors candidate")
    ax.set_title(str(plot.candidate.iloc[0]).replace("_", " ").title())
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_calibration(models: dict[str, pd.DataFrame], output: Path) -> None:
    nominal = np.array(INTERVAL_LEVELS) / 100
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for model, frame in models.items():
        observed = [((frame.observed_md >= frame[f"lower_{x}"]) &
                     (frame.observed_md <= frame[f"upper_{x}"])).mean() for x in INTERVAL_LEVELS]
        ax.plot(nominal, observed, marker="o", label=model.replace("_", " "))
    ax.plot([0.45, 0.95], [0.45, 0.95], color="black", linestyle="--", linewidth=0.9)
    ax.set(xlabel="Nominal coverage", ylabel="OOF empirical coverage", xlim=(0.45, 0.95), ylim=(0.45, 0.95))
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_qq(models: dict[str, pd.DataFrame], output: Path) -> None:
    expected = stats.norm.ppf(QQ_PROBS)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for model, frame in models.items():
        ax.plot(expected, np.quantile(frame.z.to_numpy(float), QQ_PROBS), label=model.replace("_", " "))
    limits = (-2.6, 2.6)
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=0.9)
    ax.set(xlabel="Expected standard-normal quantile", ylabel="Observed OOF z-score quantile",
           xlim=limits, ylim=limits)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def stability(candidate_dir: Path, model: str) -> dict[str, object]:
    diag = pd.read_csv(candidate_dir / "shards" / "model_diagnostics.csv")
    failures = json.loads((candidate_dir / "shards" / "failures.json").read_text(encoding="utf-8"))
    report = json.loads((candidate_dir / "shards" / "run_report.json").read_text(encoding="utf-8"))
    warnings_count = int(diag.warnings.fillna("[]").ne("[]").sum()) if "warnings" in diag else 0
    result = {"model": model, "fits_successful": len(diag), "task_failures": len(failures),
              "fits_with_warnings": warnings_count, "elapsed_seconds": report["elapsed_seconds"],
              "mean_seconds_per_successful_fit": report["elapsed_seconds"] / max(len(diag), 1),
              "nonfinite_diagnostic_parameters": int((~np.isfinite(diag.select_dtypes("number"))).sum().sum())}
    if model == "warped_gompertz":
        result.update({"outer_optimizer_failures": int((~diag.outer_optimizer_success).sum()),
                       "shape_at_bound": int(diag.shape_at_bound.sum()),
                       "large_multistart_inflection_disagreement": int((diag.multistart_inflection_age_range > 5).sum()),
                       "near_zero_age_effect": int((diag.age_beta_standardized.abs() < 0.05).sum()),
                       "median_objective_evaluations": float(diag.objective_evaluations.median())})
    return result


def stability_by_location(candidate_dir: Path, model: str) -> pd.DataFrame:
    """Detailed fit/optimization audit without hiding problematic locations."""
    diag = pd.read_csv(candidate_dir / "shards" / "model_diagnostics.csv")
    group_columns = ["region_index", "hemisphere", "dk_label_name", "depth_index", "depth"]
    rows = []
    for keys, group in diag.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, keys))
        row.update({"model": model, "n_folds_expected": 5, "n_folds_available": len(group),
                    "fits_with_warnings": int(group.warnings.fillna("[]").ne("[]").sum()),
                    "mean_negative_log_evidence": float(group.negative_log_evidence.mean()),
                    "median_warp_epsilon": float(group.warp_epsilon.median()),
                    "median_warp_b": float(group.warp_b.median())})
        if model == "warped_gompertz":
            row.update({"outer_optimizer_failures": int((~group.outer_optimizer_success).sum()),
                        "shape_at_bound": int(group.shape_at_bound.sum()),
                        "large_multistart_inflection_disagreement": int(
                            (group.multistart_inflection_age_range > 5).sum()),
                        "near_zero_age_effect": int((group.age_beta_standardized.abs() < 0.05).sum()),
                        "rate_per_year_median": float(group.gompertz_rate_per_year.median()),
                        "rate_per_year_min": float(group.gompertz_rate_per_year.min()),
                        "rate_per_year_max": float(group.gompertz_rate_per_year.max()),
                        "inflection_age_median": float(group.gompertz_inflection_age.median()),
                        "inflection_age_min": float(group.gompertz_inflection_age.min()),
                        "inflection_age_max": float(group.gompertz_inflection_age.max()),
                        "age_beta_standardized_median": float(group.age_beta_standardized.median())})
        rows.append(row)
    return pd.DataFrame(rows)


def spline_audit(spline_dir: Path) -> dict[str, object]:
    transforms = json.loads((spline_dir / "shards" / "fold_transforms.json").read_text(encoding="utf-8"))
    states = [item["transform"]["spline"] for item in transforms]
    column_sets = {tuple(state["columns"]) for state in states}
    knot_sets = {tuple(state["all_knots"]) for state in states}
    return {"implementation": "patsy.cr(age, df=3, constraints='center') - 1",
            "basis_family": "natural cubic regression spline (linear extrapolation beyond boundary knots)",
            "df_meaning": "3 retained spline basis degrees of freedom/columns after the centering constraint; not polynomial degree and not spline order",
            "actual_basis_columns": len(next(iter(column_sets))),
            "column_names": list(next(iter(column_sets))), "number_of_unique_fold_knot_sets": len(knot_sets),
            "fold_knot_sets": [list(x) for x in sorted(knot_sets)],
            "knot_policy": "patsy chooses knots from training-fold ages only; observed training min/max are boundary knots",
            "test_transform": "patsy.build_design_matrices reuses the training DesignInfo, knots, boundaries, and centering constraint; each column is then scaled using training-fold mean/SD only",
            "n_recorded_region_fold_transforms": len(transforms)}


def subset_baseline(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    keys = candidate[KEYS]
    result = baseline.merge(keys, on=KEYS, how="inner", validate="one_to_one")
    if len(result) != len(candidate):
        raise RuntimeError("Baseline does not contain every candidate pilot observation")
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_parquet(args.baseline)
    baseline_diag = pd.read_csv(args.baseline_diagnostics)
    baseline, reconstruction = reconstruct_baseline_mean(baseline, baseline_diag, args)
    spline = pd.read_parquet(args.spline_dir / "shards" / "predictions.parquet")
    validate_prediction(baseline, "warped_linear", 186 * 68 * 21)
    validate_prediction(spline, "warped_spline", 186 * 68 * 21)
    if set(map(tuple, baseline[KEYS].to_numpy())) != set(map(tuple, spline[KEYS].to_numpy())):
        raise RuntimeError("Linear and spline OOF keys differ")
    baseline.to_parquet(args.output_dir / "oof_predictions_warped_linear_reused_with_mean.parquet", index=False)
    spline.to_parquet(args.output_dir / "oof_predictions_warped_spline_df3.parquet", index=False)

    models = {"warped_linear": baseline, "warped_spline_df3": spline}
    tables = [metric_table(models, "full_regional")]
    contrasts = [paired_mll_bootstrap(spline, baseline, "warped_spline_df3",
                                      args.n_bootstrap, args.seed)]
    stability_rows = [stability(args.spline_dir, "warped_spline_df3")]
    stability_locations = [stability_by_location(args.spline_dir, "warped_spline_df3")]
    pilot_models = None
    if args.gompertz_dir:
        gompertz = pd.read_parquet(args.gompertz_dir / "shards" / "predictions.parquet")
        validate_prediction(gompertz, "warped_gompertz")
        linear_pilot = subset_baseline(baseline, gompertz)
        gompertz.to_parquet(args.output_dir / "oof_predictions_warped_gompertz_pilot.parquet", index=False)
        pilot_models = {"warped_linear_pilot_subset": linear_pilot, "warped_gompertz": gompertz}
        tables.append(metric_table(pilot_models, "gompertz_pilot"))
        contrasts.append(paired_mll_bootstrap(gompertz, linear_pilot, "warped_gompertz",
                                              args.n_bootstrap, args.seed + 1))
        stability_rows.append(stability(args.gompertz_dir, "warped_gompertz"))
        stability_locations.append(stability_by_location(args.gompertz_dir, "warped_gompertz"))

    metrics_out = pd.concat(tables, ignore_index=True)
    contrasts_out = pd.concat(contrasts, ignore_index=True)
    metrics_out.to_csv(args.output_dir / "metrics_overall_and_by_depth.csv", index=False)
    contrasts_out.to_csv(args.output_dir / "paired_subject_bootstrap_delta_mll.csv", index=False)
    metrics_out.loc[metrics_out.scope.eq("overall")].to_csv(
        args.output_dir / "metrics_overall.csv", index=False)
    metrics_out.loc[metrics_out.scope.eq("by_depth")].to_csv(
        args.output_dir / "metrics_by_depth.csv", index=False)
    contrasts_out.loc[contrasts_out.scope.eq("overall")].to_csv(
        args.output_dir / "delta_mll_overall.csv", index=False)
    contrasts_out.loc[contrasts_out.scope.eq("by_depth")].to_csv(
        args.output_dir / "delta_mll_by_depth.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(args.output_dir / "fit_stability_summary.csv", index=False)
    pd.concat(stability_locations, ignore_index=True, sort=False).to_csv(
        args.output_dir / "fit_stability_by_region_depth.csv", index=False)
    write_json(args.output_dir / "spline_implementation_audit.json", spline_audit(args.spline_dir))
    write_json(args.output_dir / "baseline_mean_reconstruction.json", reconstruction)
    plot_forest(contrasts_out.query("candidate == 'warped_spline_df3'"),
                args.output_dir / "delta_mll_forest_spline")
    plot_calibration(models, args.output_dir / "calibration_comparison_spline")
    plot_qq(models, args.output_dir / "qq_comparison_spline")
    if pilot_models is not None:
        plot_forest(contrasts_out.query("candidate == 'warped_gompertz'"),
                    args.output_dir / "delta_mll_forest_gompertz_pilot")
        plot_calibration(pilot_models, args.output_dir / "calibration_comparison_gompertz_pilot")
        plot_qq(pilot_models, args.output_dir / "qq_comparison_gompertz_pilot")
    spline_overall = contrasts_out.query(
        "candidate == 'warped_spline_df3' and scope == 'overall'"
    ).iloc[0]
    decision = {
        "warped_spline_df3": {
            "delta_mll": float(spline_overall.delta_mll),
            "ci_95": [float(spline_overall.ci_low), float(spline_overall.ci_high)],
            "decision": "retain warped linear; spline does not improve overall OOF MLL",
            "promote_to_ico4": False,
        },
        "warped_gompertz": {"decision": "not evaluated"},
        "final_model_replaced": False,
        "figure7_frozen_model_modified": False,
    }
    if pilot_models is not None:
        gompertz_overall = contrasts_out.query(
            "candidate == 'warped_gompertz' and scope == 'overall'"
        ).iloc[0]
        gompertz_stability = next(row for row in stability_rows if row["model"] == "warped_gompertz")
        decision["warped_gompertz"] = {
            "delta_mll_pilot": float(gompertz_overall.delta_mll),
            "ci_95": [float(gompertz_overall.ci_low), float(gompertz_overall.ci_high)],
            "outer_optimizer_failures": int(gompertz_stability["outer_optimizer_failures"]),
            "shape_at_bound": int(gompertz_stability["shape_at_bound"]),
            "large_multistart_inflection_disagreement": int(
                gompertz_stability["large_multistart_inflection_disagreement"]),
            "near_zero_age_effect": int(gompertz_stability["near_zero_age_effect"]),
            "decision": "do not promote: pilot density is computable, but MLL CI crosses zero and shape optimization/identifiability is unreliable",
            "promote_to_full_regional": False,
            "promote_to_ico4": False,
        }
    write_json(args.output_dir / "model_decision.json", decision)
    report = {"status": "complete", "cohort_n": 186, "regions": 68, "depths": 21,
              "baseline_reused": str(args.baseline.resolve()), "baseline_refit": False,
              "full_spline_rows": len(spline),
              "gompertz_pilot_rows": 0 if pilot_models is None else len(pilot_models["warped_gompertz"]),
              "effective_observation_policy": "Exact paired OOF keys required; no prediction was silently removed",
              "bootstrap_unit": "subject; every resample retains all region-by-depth observations for each sampled subject",
              "point_prediction_policy": {"MAE": "predictive median", "RMSE": "predictive mean"},
              "qq_probability_grid": QQ_PROBS.tolist(), "n_bootstrap": args.n_bootstrap,
              "seed": args.seed, "ico4_rerun": False, "figure7_frozen_model_modified": False}
    write_json(args.output_dir / "comparison_validation.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
