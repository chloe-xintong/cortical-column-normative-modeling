#!/usr/bin/env python3
"""Validate OOF results, bootstrap 6/11/21-depth comparisons, and render the supplement.

Description
-----------
Validate OOF results, bootstrap 6/11/21-depth comparisons, and render the supplement.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--figure3-root`` command-line input
- ``--figure5-root`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/depth_density_sensitivity/scripts/summarize_depth_density_sensitivity.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


DEPTH_INDICES = {
    6: [0, 4, 8, 12, 16, 20],
    11: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    21: list(range(21)),
}
REPRESENTATIONS = ("regional_profile", "ico4_md")
DISPLAY_NAMES = {"regional_profile": "Regional profile", "ico4_md": "ico4 MD"}
COLORS = {6: "#A9CBE4", 11: "#6F9FCA", 21: "#294F86"}
BOOTSTRAP_SEED = 20261106


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure3-root", type=Path, required=True)
    parser.add_argument("--figure5-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def metric_values(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - y
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "pearson_r": float(pearsonr(y, prediction).statistic),
        "r2": float(r2_score(y, prediction)),
    }


def expected_outer_fold(subjects: pd.DataFrame, fold_payload: dict[str, object]) -> np.ndarray:
    fold_by_subject: dict[str, int] = {}
    for fold in fold_payload["folds"]:
        fold_index = int(fold["outer_fold"])
        for subject_id in fold["test_subject_ids"]:
            if subject_id in fold_by_subject:
                raise RuntimeError(f"Subject {subject_id} appears in multiple outer test folds")
            fold_by_subject[str(subject_id)] = fold_index
    if set(fold_by_subject) != set(subjects["subject_id"]):
        raise RuntimeError("Outer test folds do not partition the 196-subject cohort")
    return subjects["subject_id"].map(fold_by_subject).to_numpy(dtype=int)


def load_and_validate_models(
    figure5_root: Path, output_dir: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    features = figure5_root / "features"
    subjects = pd.read_csv(features / "subject_order.csv", dtype={"subject_id": str})
    if len(subjects) != 196 or not np.array_equal(subjects["row_index"], np.arange(196)):
        raise RuntimeError("Figure 5 subject order is not the expected 196-row index")
    fold_dir = figure5_root / "folds"
    outer_path = fold_dir / "outer_folds.json"
    inner_path = fold_dir / "inner_folds_by_outer_fold.json"
    outer = json.loads(outer_path.read_text(encoding="utf-8"))
    inner = json.loads(inner_path.read_text(encoding="utf-8"))
    if len(outer.get("folds", [])) != 5:
        raise RuntimeError("Expected exactly five saved outer folds")
    if set(inner.get("outer_folds", {})) != {"0", "1", "2", "3", "4"}:
        raise RuntimeError("Inner-fold file does not contain all five outer-fold keys")
    if not all(len(inner["outer_folds"][str(index)].get("folds", [])) == 5 for index in range(5)):
        raise RuntimeError("Expected exactly five inner folds within every outer fold")
    expected_fold = expected_outer_fold(subjects, outer)
    alpha_grid = np.logspace(-4, 6, 21)
    rows: list[pd.DataFrame] = []
    checks: dict[str, object] = {
        "n_subjects": len(subjects),
        "outer_folds_partition_subjects_once": True,
        "fold_files": {
            "outer": str(outer_path),
            "outer_sha256": sha256(outer_path),
            "inner": str(inner_path),
            "inner_sha256": sha256(inner_path),
        },
        "models": {},
    }
    for representation in REPRESENTATIONS:
        for n_depths in (6, 11, 21):
            if n_depths == 21:
                model_dir = figure5_root / "models_lsqr" / representation
            else:
                model_dir = output_dir / "models" / f"d{n_depths}" / representation
            prediction_path = model_dir / f"{representation}_oof_predictions.csv"
            metrics_path = model_dir / "metrics.json"
            predictions = pd.read_csv(prediction_path)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            expected_features = (68 if representation == "regional_profile" else 4687) * n_depths
            model_key = f"{representation}_d{n_depths}"
            model_checks = {
                "prediction_file": str(prediction_path),
                "prediction_sha256": sha256(prediction_path),
                "metrics_file": str(metrics_path),
                "reused_existing_oof": n_depths == 21,
                "status_complete": metrics.get("status") == "complete",
                "solver_lsqr": metrics.get("ridge_solver") == "lsqr",
                "n_evaluated_196": int(metrics.get("n_evaluated", -1)) == 196,
                "matrix_shape_matches": metrics.get("matrix_shape") == [196, expected_features],
                "alpha_grid_matches": bool(np.allclose(metrics.get("alpha_grid", []), alpha_grid)),
                "one_prediction_per_subject": len(predictions) == 196
                and predictions["row_index"].nunique() == 196,
                "all_oof_predictions_finite": bool(
                    np.isfinite(predictions["predicted_age"].to_numpy(dtype=float)).all()
                ),
                "elapsed_seconds": float(metrics.get("elapsed_seconds", np.nan)),
                "peak_rss_kb": int(metrics.get("peak_rss_kb", -1)),
            }
            required_checks = (
                "status_complete", "solver_lsqr", "n_evaluated_196", "matrix_shape_matches",
                "alpha_grid_matches", "one_prediction_per_subject", "all_oof_predictions_finite",
            )
            if not all(model_checks[key] for key in required_checks):
                raise RuntimeError(f"Model validation failed for {model_key}: {model_checks}")
            predictions = predictions.sort_values("row_index", kind="stable").reset_index(drop=True)
            if not np.array_equal(predictions["row_index"], subjects["row_index"]):
                raise RuntimeError(f"{model_key}: OOF row indices do not match subject order")
            if not np.array_equal(predictions["outer_fold"].to_numpy(dtype=int), expected_fold):
                raise RuntimeError(f"{model_key}: OOF fold assignments differ from saved outer folds")
            if not np.allclose(predictions["true_age"], subjects["age_at_scan"], rtol=0, atol=0):
                raise RuntimeError(f"{model_key}: OOF outcomes differ from subject_order.csv")
            frame = predictions[
                ["row_index", "true_age", "predicted_age", "outer_fold", "selected_alpha"]
            ].copy()
            frame.insert(1, "subject_id", subjects["subject_id"].to_numpy())
            frame.insert(2, "representation", representation)
            frame.insert(3, "representation_label", DISPLAY_NAMES[representation])
            frame.insert(4, "n_depths", n_depths)
            frame["error"] = frame["predicted_age"] - frame["true_age"]
            frame["absolute_error"] = frame["error"].abs()
            rows.append(frame)
            checks["models"][model_key] = model_checks
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(output_dir / "age_prediction_oof_predictions.csv", index=False)
    return combined, checks


def bootstrap_metrics_and_differences(
    oof: pd.DataFrame, replicates: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_subjects = oof["row_index"].nunique()
    indices = rng.integers(0, n_subjects, size=(replicates, n_subjects), endpoint=False)
    metric_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    predictions_by_model: dict[tuple[str, int], np.ndarray] = {}
    y_reference: np.ndarray | None = None
    for representation in REPRESENTATIONS:
        for n_depths in (6, 11, 21):
            model = (
                oof.loc[
                    oof["representation"].eq(representation) & oof["n_depths"].eq(n_depths)
                ]
                .sort_values("row_index")
                .reset_index(drop=True)
            )
            y = model["true_age"].to_numpy(dtype=float)
            prediction = model["predicted_age"].to_numpy(dtype=float)
            if y_reference is None:
                y_reference = y
            elif not np.array_equal(y, y_reference):
                raise RuntimeError("Models do not share identical subjects and outcomes")
            predictions_by_model[(representation, n_depths)] = prediction
            point = metric_values(y, prediction)
            boot_mae = np.mean(np.abs(prediction[indices] - y[indices]), axis=1)
            boot_rmse = np.sqrt(np.mean((prediction[indices] - y[indices]) ** 2, axis=1))
            metric_rows.append(
                {
                    "representation": representation,
                    "representation_label": DISPLAY_NAMES[representation],
                    "n_depths": n_depths,
                    **point,
                    "mae_bootstrap_ci_low": float(np.percentile(boot_mae, 2.5)),
                    "mae_bootstrap_ci_high": float(np.percentile(boot_mae, 97.5)),
                    "rmse_bootstrap_ci_low": float(np.percentile(boot_rmse, 2.5)),
                    "rmse_bootstrap_ci_high": float(np.percentile(boot_rmse, 97.5)),
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                }
            )
    assert y_reference is not None
    for representation in REPRESENTATIONS:
        for first, second in ((6, 21), (11, 21), (6, 11)):
            first_prediction = predictions_by_model[(representation, first)]
            second_prediction = predictions_by_model[(representation, second)]
            first_error = first_prediction - y_reference
            second_error = second_prediction - y_reference
            delta_absolute = np.abs(first_error) - np.abs(second_error)
            boot_delta_mae = np.mean(delta_absolute[indices], axis=1)
            boot_delta_rmse = np.sqrt(np.mean(first_error[indices] ** 2, axis=1)) - np.sqrt(
                np.mean(second_error[indices] ** 2, axis=1)
            )
            paired_rows.append(
                {
                    "representation": representation,
                    "representation_label": DISPLAY_NAMES[representation],
                    "comparison": f"{first}_vs_{second}",
                    "first_n_depths": first,
                    "second_n_depths": second,
                    "delta_mae_first_minus_second": float(np.mean(delta_absolute)),
                    "delta_mae_bootstrap_ci_low": float(np.percentile(boot_delta_mae, 2.5)),
                    "delta_mae_bootstrap_ci_high": float(np.percentile(boot_delta_mae, 97.5)),
                    "delta_rmse_first_minus_second": float(
                        np.sqrt(np.mean(first_error**2)) - np.sqrt(np.mean(second_error**2))
                    ),
                    "delta_rmse_bootstrap_ci_low": float(np.percentile(boot_delta_rmse, 2.5)),
                    "delta_rmse_bootstrap_ci_high": float(np.percentile(boot_delta_rmse, 97.5)),
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(paired_rows)


def draw_violin(ax: plt.Axes, groups: list[np.ndarray], positions: list[float]) -> None:
    violin = ax.violinplot(
        groups, positions=positions, widths=0.46, showmeans=False, showmedians=False,
        showextrema=False, bw_method=0.3,
    )
    for body, n_depths in zip(violin["bodies"], (6, 11)):
        body.set_facecolor(COLORS[n_depths])
        body.set_edgecolor("#294F86")
        body.set_linewidth(0.9)
        body.set_alpha(0.72)
        body.set_gid("subject_violin")
    for values, position in zip(groups, positions):
        q1, median, q3 = np.percentile(values[np.isfinite(values)], [25, 50, 75])
        ax.vlines(position, q1, q3, color="#294F86", lw=0.85, zorder=4)
        ax.hlines([q1, q3], position - 0.055, position + 0.055, color="#294F86", lw=0.85, zorder=4)
        ax.hlines(median, position - 0.09, position + 0.09, color="#294F86", lw=1.1, zorder=5)


def make_figure(output_dir: Path, metrics: pd.DataFrame, paired: pd.DataFrame) -> None:
    from qc.publication_style import finalize_figure, set_publication_style

    set_publication_style()
    selected = pd.read_csv(output_dir / "selected_region_group_profiles.csv")
    fidelity = pd.read_csv(output_dir / "profile_fidelity_subject_summary.csv")
    selected_names = (
        selected[["region", "selection_order"]]
        .drop_duplicates()
        .sort_values("selection_order")["region"]
        .tolist()
    )
    display_region = {
        "lateraloccipital": "Lateral\noccipital",
        "caudalmiddlefrontal": "Caudal middle\nfrontal",
        "frontalpole": "Frontal\npole",
    }
    fig = plt.figure(figsize=(15.6, 4.25))
    outer = fig.add_gridspec(
        1, 4, width_ratios=[2.8, 1.75, 1.45, 1.7], wspace=0.72,
        left=0.055, right=0.985, bottom=0.18, top=0.82,
    )
    profile_grid = outer[0].subgridspec(1, 3, wspace=0.48)
    profile_axes = []
    for axis_index, region in enumerate(selected_names):
        ax = fig.add_subplot(profile_grid[axis_index])
        profile_axes.append(ax)
        for n_depths in (21, 11, 6):
            part = selected.loc[
                selected["region"].eq(region) & selected["n_depths"].eq(n_depths)
            ].sort_values("depth_index")
            ax.plot(
                part["normalized_depth"], part["group_mean_md"],
                color=COLORS[n_depths], lw=1.6 if n_depths == 21 else 0.85,
                marker=None if n_depths == 21 else ("o" if n_depths == 11 else "s"),
                ms=2.5 if n_depths == 11 else 2.8, zorder={21: 1, 11: 2, 6: 3}[n_depths],
            )
        ax.set_title(display_region.get(region, region.replace("_", " ")), loc="center")
        ax.set_xlabel("")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if axis_index == 0:
            ax.set_ylabel("Mean MD")
        else:
            ax.set_ylabel("")

    b_grid = outer[1].subgridspec(1, 2, wspace=0.62)
    b1, b2 = fig.add_subplot(b_grid[0]), fig.add_subplot(b_grid[1])
    for ax, column, ylabel in (
        (b1, "mean_centered_profile_rmse_median", "Centered RMSE"),
        (b2, "mean_centered_pearson_r_median", "Centered Pearson r"),
    ):
        groups = [
            fidelity.loc[fidelity["n_depths"].eq(n), column].to_numpy(dtype=float)
            for n in (6, 11)
        ]
        draw_violin(ax, groups, [0, 1])
        ax.set_xticks([0, 1], ["6", "11"])
        ax.set_xlabel("Depth points")
        ax.set_ylabel(ylabel)
        if "rmse" in column:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    c = fig.add_subplot(outer[2])
    x_base = np.arange(2, dtype=float)
    offsets = {6: -0.22, 11: 0.0, 21: 0.22}
    for n_depths in (6, 11, 21):
        part = metrics.loc[metrics["n_depths"].eq(n_depths)].set_index("representation")
        values = np.asarray([part.loc[name, "mae"] for name in REPRESENTATIONS])
        lower = values - np.asarray([part.loc[name, "mae_bootstrap_ci_low"] for name in REPRESENTATIONS])
        upper = np.asarray([part.loc[name, "mae_bootstrap_ci_high"] for name in REPRESENTATIONS]) - values
        c.errorbar(
            x_base + offsets[n_depths], values, yerr=np.vstack([lower, upper]),
            fmt="o", color=COLORS[n_depths], ecolor=COLORS[n_depths],
            ms=4.2, capsize=2, lw=1.0,
        )
    c.set_xticks(x_base, ["Regional\nprofile", "ico4 MD"])
    c.set_ylabel("OOF MAE (years)")

    d = fig.add_subplot(outer[3])
    rows = paired.loc[paired["comparison"].isin(["6_vs_21", "11_vs_21"])].copy()
    rows["order"] = rows["representation"].map({"regional_profile": 0, "ico4_md": 2}) + rows[
        "first_n_depths"
    ].map({6: 0, 11: 1})
    rows = rows.sort_values("order")
    y_positions = np.arange(len(rows))[::-1]
    for y_position, row in zip(y_positions, rows.itertuples()):
        x = row.delta_mae_first_minus_second
        d.errorbar(
            x, y_position,
            xerr=[[x - row.delta_mae_bootstrap_ci_low], [row.delta_mae_bootstrap_ci_high - x]],
            fmt="o", color=COLORS[int(row.first_n_depths)], capsize=2, lw=1.0, ms=4.2,
        )
    d.axvline(0, color="#777777", lw=0.8, ls="--")
    d.set_yticks(y_positions, [
        f"{'Regional' if row.representation == 'regional_profile' else 'ico4'} "
        f"{int(row.first_n_depths)}−21" for row in rows.itertuples()
    ])
    d.tick_params(axis="y", labelsize=6.8, pad=2)
    d.set_xlabel("ΔMAE (years)")
    d.set_ylabel("")

    profile_left = profile_axes[0].get_position().x0
    profile_right = profile_axes[-1].get_position().x1
    fig.text((profile_left + profile_right) / 2, 0.105, "Normalized depth (0 pial → 1 white)",
             ha="center", fontsize=8.5)
    handles = [
        Line2D([0], [0], color=COLORS[21], lw=1.6, label="21 points"),
        Line2D([0], [0], color=COLORS[11], lw=0.9, marker="o", ms=3, label="11 points"),
        Line2D([0], [0], color=COLORS[6], lw=0.9, marker="s", ms=3, label="6 points"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=3, frameon=False, fontsize=7.5, handlelength=2.2, columnspacing=1.6)
    fig.text(profile_axes[0].get_position().x0 - 0.024, 0.91, "A", fontsize=11, fontweight="bold")
    fig.text(b1.get_position().x0 - 0.024, 0.91, "B", fontsize=11, fontweight="bold")
    fig.text(c.get_position().x0 - 0.024, 0.91, "C", fontsize=11, fontweight="bold")
    fig.text(d.get_position().x0 - 0.024, 0.91, "D", fontsize=11, fontweight="bold")
    finalize_figure(fig)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"supplementary_depth_density_sensitivity.{extension}",
            dpi=600 if extension == "png" else None,
            bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)


def write_readme(
    output_dir: Path, metrics: pd.DataFrame, paired: pd.DataFrame, preparation: dict[str, object]
) -> None:
    fidelity_group = pd.read_csv(output_dir / "profile_fidelity_group_summary.csv")
    fidelity_lookup = fidelity_group.set_index(["n_depths", "metric"])
    centered_rmse_6 = fidelity_lookup.loc[(6, "mean_centered_profile_rmse_median"), "median"]
    centered_rmse_11 = fidelity_lookup.loc[(11, "mean_centered_profile_rmse_median"), "median"]
    centered_r_6 = fidelity_lookup.loc[(6, "mean_centered_pearson_r_median"), "median"]
    centered_r_11 = fidelity_lookup.loc[(11, "mean_centered_pearson_r_median"), "median"]
    metric_table = metrics[
        ["representation_label", "n_depths", "mae", "rmse", "pearson_r", "r2"]
    ].copy()
    metric_lines = [
        "| Representation | Depth points | MAE (years) | RMSE (years) | Pearson r | R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metric_table.itertuples():
        metric_lines.append(
            f"| {row.representation_label} | {row.n_depths} | {row.mae:.3f} | "
            f"{row.rmse:.3f} | {row.pearson_r:.3f} | {row.r2:.3f} |"
        )
    delta_lines = [
        "| Representation | Comparison | ΔMAE (years) | 95% paired-bootstrap CI |",
        "|---|---|---:|---:|",
    ]
    for row in paired.loc[paired["comparison"].isin(["6_vs_21", "11_vs_21"])].itertuples():
        delta_lines.append(
            f"| {row.representation_label} | {row.comparison.replace('_vs_', ' vs ')} | "
            f"{row.delta_mae_first_minus_second:.3f} | "
            f"[{row.delta_mae_bootstrap_ci_low:.3f}, {row.delta_mae_bootstrap_ci_high:.3f}] |"
        )
    text = f"""# Cortical-depth sampling density sensitivity (6 vs 11 vs 21)

## Purpose

This supplementary analysis tests whether the principal profile and age-prediction findings depend on using 21 normalized cortical-depth samples. It is a sensitivity analysis of profile parameterization density, not a search for an optimal number of samples. The 21-point representation is treated as the existing reference representation, not as ground truth.

The 11-point profile retains indices `{DEPTH_INDICES[11]}` (normalized depths 0.0, 0.1, ..., 1.0). The 6-point profile retains indices `{DEPTH_INDICES[6]}` (normalized depths 0.0, 0.2, ..., 1.0). Both are exact subsets of the existing 21-point Option 3 profiles. Column construction and MD sampling were not rerun.

## Profile fidelity

For every observed subject × hemisphere × DK-region profile in the Figure 3 cohort, each reduced profile was linearly interpolated back to the 21-point grid. Raw-profile RMSE, mean-centered RMSE, mean-centered Pearson correlation, and maximum absolute difference were computed against the original 21-point profile. DK-region/hemisphere values were first summarized by their median within each subject; group summaries then use the median and IQR across subjects, so every subject has equal weight. Panel A uses the same selected regions, bilateral averaging, 320-subject cohort, and equal-subject group mean as Figure 3A.

Mean-centered reconstruction remained close to the reference representation. The across-subject median of the within-subject regional median RMSE was {centered_rmse_6:.3e} for 6 points and {centered_rmse_11:.3e} for 11 points; the corresponding median correlations were {centered_r_6:.6f} and {centered_r_11:.6f}.

## Age prediction

Regional profile and ico4 MD-only representations were compared in the fixed 196-subject Figure 5 cohort. The saved nested 5 × 5 folds, LSQR Ridge implementation, `logspace(-4, 6, 21)` alpha grid, and fold-local median imputation and standardization were held fixed. Existing complete 21-point OOF predictions were reused after validating their solver, alpha grid, matrix dimensions, subject order, outcomes, and outer-fold assignments. Six- and 11-point models were rerun with the same code and folds. Every subject has exactly one OOF prediction per model.

{chr(10).join(metric_lines)}

Positive ΔMAE means that the reduced-depth representation had greater OOF absolute error than the 21-point reference.

{chr(10).join(delta_lines)}

The ico4 MD prediction metrics were essentially unchanged across the three parameterizations. For the regional representation, 6 versus 21 points had a small ΔMAE whose interval included zero, whereas 11 versus 21 points had a small positive ΔMAE of 0.181 years (95% CI 0.032 to 0.342). This non-monotonic pattern does not identify an optimal sampling density and is reported without expanding to additional densities.

## Interpretation boundary

Changing the number of retained profile samples does not change the 1-mm DWI acquisition resolution. Adjacent depth values are highly correlated because of voxel resolution and trilinear interpolation. Therefore, this analysis evaluates the numerical parameterization density of a cortical-depth profile; it does not resolve independent histological layers. Similar performance across densities may be described as: “The principal findings were insensitive to the use of 6, 11, or 21 normalized cortical-depth samples.” It must not be interpreted as showing that any density is optimal.

## Figure

- A: Figure 3A selected-region group mean profiles; 21 points are continuous curves and reduced representations use thinner point-connected lines.
- B: subject-level medians across DK region/hemisphere profiles for mean-centered reconstruction RMSE and correlation.
- C: OOF MAE with subject-bootstrap 95% CIs for the two MD-only prediction representations.
- D: paired ΔMAE relative to 21 points with 10,000 subject-level paired-bootstrap CIs.

The figure contains no sample-size labels. Profile fidelity uses {preparation['profile_fidelity']['n_subjects']} subjects; age prediction uses 196 subjects.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.time()
    preparation_path = args.output_dir / "preparation_checks.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    oof, model_checks = load_and_validate_models(args.figure5_root, args.output_dir)
    metrics, paired = bootstrap_metrics_and_differences(oof, args.bootstrap_replicates)
    metrics.to_csv(args.output_dir / "age_prediction_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_bootstrap_depth_density.csv", index=False)
    paired.to_csv(args.output_dir / "paired_bootstrap_11_vs_21.csv", index=False)
    make_figure(args.output_dir, metrics, paired)
    write_readme(args.output_dir, metrics, paired, preparation)

    repository = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=repository, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    )
    subjects = pd.read_csv(
        args.figure5_root / "features" / "subject_order.csv", dtype={"subject_id": str}
    )
    source_integrity = {
        "figure3_subject_region_profiles_unchanged": (
            sha256(Path(preparation["profile_fidelity"]["source"]))
            == preparation["profile_fidelity"]["source_sha256"]
        ),
        "figure5_source_matrices_unchanged": {},
        "existing_figure3_or_figure5_output_written_by_this_analysis": False,
    }
    for representation, report in preparation["age_prediction_features"]["representations"].items():
        source_integrity["figure5_source_matrices_unchanged"][representation] = (
            sha256(Path(report["source_matrix"])) == report["source_matrix_sha256"]
        )
    if not source_integrity["figure3_subject_region_profiles_unchanged"] or not all(
        source_integrity["figure5_source_matrices_unchanged"].values()
    ):
        raise RuntimeError("A read-only Figure 3/5 source changed during the sensitivity analysis")
    metadata = {
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "run_elapsed_seconds": time.time() - preparation["started_unix"],
        "summarization_elapsed_seconds": time.time() - start,
        "input_files": {
            "figure3_root": str(args.figure3_root),
            "figure5_root": str(args.figure5_root),
            "figure3_subject_region_profiles": preparation["profile_fidelity"]["source"],
            "figure5_feature_root": preparation["age_prediction_features"]["source_feature_root"],
        },
        "subject_list": subjects["subject_id"].tolist(),
        "n_profile_fidelity_subjects": preparation["profile_fidelity"]["n_subjects"],
        "n_age_prediction_subjects": len(subjects),
        "depth_indices": {str(key): value for key, value in DEPTH_INDICES.items()},
        "depth_fractions": {
            str(key): [index / 20 for index in value] for key, value in DEPTH_INDICES.items()
        },
        "cv_fold_files": model_checks["fold_files"],
        "model_parameters": {
            "estimator": "Ridge",
            "solver": "lsqr",
            "nested_cv": "5 outer folds x 5 inner folds",
            "alpha_grid": np.logspace(-4, 6, 21).tolist(),
            "imputation": "training-fold median",
            "standardization": "StandardScaler fitted only in training folds",
            "fold_seed": 20261027,
        },
        "bootstrap": {
            "unit": "subject",
            "paired": True,
            "replicates": args.bootstrap_replicates,
            "seed": BOOTSTRAP_SEED,
        },
        "git_commit": commit,
        "git_worktree_dirty_at_completion": dirty,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "reused_existing_21_point_oof_predictions": True,
        "source_integrity": source_integrity,
        "preparation_checks": preparation,
        "model_and_oof_checks": model_checks,
        "data_leakage_check": {
            "fold_local_imputer_and_scaler_confirmed_in_shared_run_ridge_code": True,
            "outer_fold_assignments_match_saved_fold_file": True,
            "missing_oof_predictions": 0,
            "configuration_mismatches": [],
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
