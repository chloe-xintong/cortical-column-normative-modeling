#!/usr/bin/env python3
"""Compare auto vs LSQR and run pre-specified paired Figure 5 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


REPRESENTATIONS = [
    "regional_mean_md",
    "regional_thickness",
    "regional_profile",
    "ico3_md",
    "ico4_md",
    "ico5_md",
    "ico3_md_length",
    "ico3_md_curvature",
    "ico3_md_length_curvature",
]

PREDEFINED_COMPARISONS = [
    ("regional_mean_md", "regional_profile"),
    ("regional_profile", "ico3_md"),
    ("regional_profile", "ico4_md"),
    ("ico4_md", "ico5_md"),
    ("ico3_md", "ico3_md_length"),
    ("ico3_md", "ico3_md_curvature"),
    ("ico3_md", "ico3_md_length_curvature"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto-model-root", type=Path, required=True)
    parser.add_argument("--lsqr-model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261028)
    return parser.parse_args()


def prediction_path(root: Path, representation: str) -> Path:
    return root / representation / f"{representation}_oof_predictions.csv"


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stability_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    boundary_rows: list[pd.DataFrame] = []
    cbag_summary_rows: list[dict[str, object]] = []
    lsqr_predictions: dict[str, pd.DataFrame] = {}
    combined_curves: list[pd.DataFrame] = []
    cbag_wide: pd.DataFrame | None = None

    for representation in REPRESENTATIONS:
        auto_metrics = json.loads(
            (args.auto_model_root / representation / "metrics.json").read_text(encoding="utf-8")
        )
        lsqr_metrics = json.loads(
            (args.lsqr_model_root / representation / "metrics.json").read_text(encoding="utf-8")
        )
        auto_prediction = pd.read_csv(prediction_path(args.auto_model_root, representation)).sort_values(
            "row_index"
        )
        lsqr_prediction = pd.read_csv(prediction_path(args.lsqr_model_root, representation)).sort_values(
            "row_index"
        )
        if not np.array_equal(auto_prediction["row_index"], lsqr_prediction["row_index"]):
            raise RuntimeError(f"{representation}: auto and LSQR subject orders differ")
        if not np.allclose(auto_prediction["true_age"], lsqr_prediction["true_age"]):
            raise RuntimeError(f"{representation}: auto and LSQR ages differ")
        difference = lsqr_prediction["predicted_age"].to_numpy() - auto_prediction[
            "predicted_age"
        ].to_numpy()
        stability_rows.append(
            {
                "representation": representation,
                "n": len(difference),
                "auto_mae": auto_metrics["overall_oof_mae"],
                "lsqr_mae": lsqr_metrics["overall_oof_mae"],
                "mae_difference_lsqr_minus_auto": (
                    lsqr_metrics["overall_oof_mae"] - auto_metrics["overall_oof_mae"]
                ),
                "auto_pearson_r": auto_metrics["overall_oof_pearson_r"],
                "lsqr_pearson_r": lsqr_metrics["overall_oof_pearson_r"],
                "pearson_r_difference_lsqr_minus_auto": (
                    lsqr_metrics["overall_oof_pearson_r"]
                    - auto_metrics["overall_oof_pearson_r"]
                ),
                "oof_prediction_pearson_r": float(
                    pearsonr(auto_prediction["predicted_age"], lsqr_prediction["predicted_age"]).statistic
                ),
                "maximum_absolute_prediction_difference": float(np.max(np.abs(difference))),
                "median_absolute_prediction_difference": float(np.median(np.abs(difference))),
            }
        )
        auto_fold = pd.read_csv(args.auto_model_root / representation / "fold_metrics.csv")
        lsqr_fold = pd.read_csv(args.lsqr_model_root / representation / "fold_metrics.csv")
        fold_compare = auto_fold[["outer_fold", "selected_alpha"]].merge(
            lsqr_fold[["outer_fold", "selected_alpha"]],
            on="outer_fold",
            validate="one_to_one",
            suffixes=("_auto", "_lsqr"),
        )
        fold_compare.insert(0, "representation", representation)
        fold_compare["selected_alpha_changed"] = ~np.isclose(
            fold_compare["selected_alpha_auto"], fold_compare["selected_alpha_lsqr"]
        )
        alpha_rows.extend(fold_compare.to_dict(orient="records"))

        curves = pd.read_csv(args.lsqr_model_root / representation / "inner_alpha_mae_curves.csv")
        curves.insert(0, "representation", representation)
        combined_curves.append(curves)
        selected = curves.loc[curves["selected"]].copy()
        selected["minimum_alpha_in_grid"] = curves["alpha"].min()
        selected["maximum_alpha_in_grid"] = curves["alpha"].max()
        boundary_rows.append(
            selected[
                [
                    "representation",
                    "outer_fold",
                    "alpha",
                    "mean_inner_validation_mae",
                    "selected_at_minimum_boundary",
                    "selected_at_maximum_boundary",
                    "minimum_alpha_in_grid",
                    "maximum_alpha_in_grid",
                ]
            ]
        )
        cbag_summary_rows.append(
            {
                "representation": representation,
                "n": lsqr_metrics["n_evaluated"],
                "raw_oof_mae": lsqr_metrics["overall_oof_mae"],
                "raw_oof_pearson_r": lsqr_metrics["overall_oof_pearson_r"],
                "cbag_mean_absolute_value": lsqr_metrics["overall_cbag_mean_absolute_value"],
                "cbag_age_pearson_r": lsqr_metrics["overall_cbag_age_pearson_r"],
            }
        )
        lsqr_predictions[representation] = lsqr_prediction
        cbag_component = lsqr_prediction[["row_index", "true_age", "outer_fold", "cbag"]].rename(
            columns={"cbag": representation}
        )
        if cbag_wide is None:
            cbag_wide = cbag_component
        else:
            cbag_wide = cbag_wide.merge(
                cbag_component.drop(columns=["true_age", "outer_fold"]),
                on="row_index",
                how="inner",
                validate="one_to_one",
            )

    stability = pd.DataFrame(stability_rows)
    stability["stable_by_predefined_ico_threshold"] = (
        stability["mae_difference_lsqr_minus_auto"].abs().le(0.02)
        & stability["oof_prediction_pearson_r"].ge(0.999)
    )
    stability.to_csv(args.output_dir / "auto_vs_lsqr_stability.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(
        args.output_dir / "auto_vs_lsqr_fold_alpha_comparison.csv", index=False
    )
    pd.concat(combined_curves, ignore_index=True).to_csv(
        args.output_dir / "lsqr_inner_alpha_mae_curves.csv", index=False
    )
    boundary = pd.concat(boundary_rows, ignore_index=True)
    boundary.to_csv(args.output_dir / "lsqr_alpha_boundary_check.csv", index=False)
    pd.DataFrame(cbag_summary_rows).to_csv(args.output_dir / "lsqr_cbag_summary.csv", index=False)
    assert cbag_wide is not None
    cbag_wide.to_csv(args.output_dir / "lsqr_foldwise_cbag_by_subject.csv", index=False)

    rng = np.random.default_rng(args.seed)
    bootstrap_rows: list[dict[str, object]] = []
    for representation_a, representation_b in PREDEFINED_COMPARISONS:
        prediction_a = lsqr_predictions[representation_a]
        prediction_b = lsqr_predictions[representation_b]
        if not np.array_equal(prediction_a["row_index"], prediction_b["row_index"]):
            raise RuntimeError(f"Paired comparison order mismatch: {representation_a}, {representation_b}")
        paired_difference = prediction_a["absolute_error"].to_numpy() - prediction_b[
            "absolute_error"
        ].to_numpy()
        indices = rng.integers(
            0, len(paired_difference), size=(args.bootstrap_replicates, len(paired_difference))
        )
        bootstrap_delta = paired_difference[indices].mean(axis=1)
        lower, upper = np.percentile(bootstrap_delta, [2.5, 97.5])
        p_lower = (np.sum(bootstrap_delta <= 0) + 1) / (args.bootstrap_replicates + 1)
        p_upper = (np.sum(bootstrap_delta >= 0) + 1) / (args.bootstrap_replicates + 1)
        bootstrap_rows.append(
            {
                "representation_a": representation_a,
                "representation_b": representation_b,
                "n": len(paired_difference),
                "delta_mae_a_minus_b": float(paired_difference.mean()),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "bootstrap_two_sided_p": float(min(1.0, 2 * min(p_lower, p_upper))),
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": args.seed,
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap["fdr_bh_q"] = fdr_bh(bootstrap["bootstrap_two_sided_p"].to_numpy())
    bootstrap["fdr_bh_significant_0_05"] = bootstrap["fdr_bh_q"] < 0.05
    bootstrap.to_csv(args.output_dir / "predefined_paired_bootstrap.csv", index=False)

    report = {
        "status": "complete",
        "n_representations": len(REPRESENTATIONS),
        "n_subjects": int(stability["n"].iloc[0]),
        "lsqr_solver_fixed": True,
        "alpha_grid": "logspace(-4, 6, 21), unchanged from auto run",
        "n_selected_at_minimum_alpha": int(boundary["selected_at_minimum_boundary"].sum()),
        "n_selected_at_maximum_alpha": int(boundary["selected_at_maximum_boundary"].sum()),
        "n_predefined_paired_comparisons": len(PREDEFINED_COMPARISONS),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
    }
    (args.output_dir / "lsqr_stability_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
