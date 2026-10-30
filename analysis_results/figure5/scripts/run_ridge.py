#!/usr/bin/env python3
"""Run one age-prediction representation with shared nested CV folds."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_FILES = {
    "regional_mean_md": "X_regional_mean_md.npy",
    "regional_thickness": "X_regional_thickness.npy",
    "regional_profile": "X_regional_profile.npy",
    "ico3_md": "X_ico3_md.npy",
    "ico4_md": "X_ico4_md.npy",
    "ico4_column_mean_md": "X_ico4_column_mean_md.npy",
    "ico4_patch_mean_md": "X_ico4_patch_mean_md.npy",
    "ico5_md": "X_ico5_md.npy",
    "ico3_column_length": "X_ico3_column_length.npy",
    "ico3_curvature": "X_ico3_curvature.npy",
    "ico4_column_length": "X_ico4_column_length.npy",
    "ico4_curvature": "X_ico4_curvature.npy",
}
REPRESENTATIONS = {
    "regional_mean_md": ["regional_mean_md"],
    "regional_thickness": ["regional_thickness"],
    "regional_profile": ["regional_profile"],
    "regional_profile_thickness": ["regional_profile", "regional_thickness"],
    "ico3_md": ["ico3_md"],
    "ico4_md": ["ico4_md"],
    "ico4_column_mean_md": ["ico4_column_mean_md"],
    "ico4_patch_mean_md": ["ico4_patch_mean_md"],
    "ico5_md": ["ico5_md"],
    "ico3_md_length": ["ico3_md", "ico3_column_length"],
    "ico3_md_curvature": ["ico3_md", "ico3_curvature"],
    "ico3_md_length_curvature": ["ico3_md", "ico3_column_length", "ico3_curvature"],
    "ico4_md_length": ["ico4_md", "ico4_column_length"],
    "ico4_md_curvature": ["ico4_md", "ico4_curvature"],
    "ico4_md_length_curvature": ["ico4_md", "ico4_column_length", "ico4_curvature"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=sorted(REPRESENTATIONS), required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-outer-folds", type=int)
    parser.add_argument("--alpha-count", type=int, default=21)
    parser.add_argument("--solver", choices=("auto", "lsqr"), default="auto")
    return parser.parse_args()


def safe_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.isclose(np.std(y_true), 0) or np.isclose(np.std(y_pred), 0):
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def load_representation(feature_dir: Path, representation: str) -> np.ndarray:
    arrays = [
        np.load(feature_dir / "matrices" / BASE_FILES[component], mmap_mode="r")
        for component in REPRESENTATIONS[representation]
    ]
    if len({array.shape[0] for array in arrays}) != 1:
        raise RuntimeError("Component row counts differ")
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate([np.asarray(array) for array in arrays], axis=1)


def inner_select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    id_to_index: dict[str, int],
    inner_folds: list[dict[str, object]],
    alphas: np.ndarray,
    solver: str,
) -> tuple[float, list[float]]:
    scores = np.zeros((len(inner_folds), len(alphas)), dtype=float)
    for split_index, split in enumerate(inner_folds):
        train = np.asarray([id_to_index[s] for s in split["train_subject_ids"]], dtype=int)
        validation = np.asarray([id_to_index[s] for s in split["validation_subject_ids"]], dtype=int)
        preprocessing = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        x_train = preprocessing.fit_transform(x[train])
        x_validation = preprocessing.transform(x[validation])
        for alpha_index, alpha in enumerate(alphas):
            model = Ridge(alpha=float(alpha), solver=solver)
            model.fit(x_train, y[train])
            scores[split_index, alpha_index] = mean_absolute_error(
                y[validation], model.predict(x_validation)
            )
        del x_train, x_validation
    mean_scores = scores.mean(axis=0)
    best_index = int(np.argmin(mean_scores))
    return float(alphas[best_index]), mean_scores.tolist()


def save_figures(predictions: pd.DataFrame, representation: str, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.5, 5.0))
    axis.scatter(predictions["true_age"], predictions["predicted_age"], s=20, alpha=0.75)
    lower = float(min(predictions["true_age"].min(), predictions["predicted_age"].min()))
    upper = float(max(predictions["true_age"].max(), predictions["predicted_age"].max()))
    axis.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="True age (years)", ylabel="Predicted age (years)", title=representation)
    fig.tight_layout()
    fig.savefig(output_dir / "predicted_vs_true_age.png", dpi=200)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    residual = predictions["predicted_age"] - predictions["true_age"]
    axis.scatter(predictions["true_age"], residual, s=20, alpha=0.75)
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="True age (years)", ylabel="Residual (predicted - true)", title=representation)
    fig.tight_layout()
    fig.savefig(output_dir / "residual_vs_age.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    start = time.time()
    output_dir = args.output_root / args.representation
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = pd.read_csv(args.feature_dir / "subject_order.csv", dtype={"subject_id": str})
    ids = subjects["subject_id"].tolist()
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate subject IDs")
    y = pd.to_numeric(subjects["age_at_scan"], errors="raise").to_numpy(dtype=float)
    x = load_representation(args.feature_dir, args.representation)
    if x.shape[0] != len(y):
        raise RuntimeError("Feature and outcome row counts differ")
    id_to_index = {subject_id: index for index, subject_id in enumerate(ids)}
    outer_payload = json.loads((args.fold_dir / "outer_folds.json").read_text(encoding="utf-8"))
    inner_payload = json.loads(
        (args.fold_dir / "inner_folds_by_outer_fold.json").read_text(encoding="utf-8")
    )
    folds = outer_payload["folds"]
    if args.max_outer_folds is not None:
        folds = folds[: args.max_outer_folds]
    alphas = np.logspace(-4, 6, args.alpha_count)
    oof = np.full(len(y), np.nan, dtype=float)
    prediction_fold = np.full(len(y), -1, dtype=int)
    selected_alpha = np.full(len(y), np.nan, dtype=float)
    raw_bag = np.full(len(y), np.nan, dtype=float)
    cbag = np.full(len(y), np.nan, dtype=float)
    corrected_predicted_age = np.full(len(y), np.nan, dtype=float)
    fold_results: list[dict[str, object]] = []
    alpha_search: dict[str, object] = {}
    alpha_curve_rows: list[dict[str, object]] = []
    cbag_coefficient_rows: list[dict[str, object]] = []
    for fold in folds:
        fold_index = int(fold["outer_fold"])
        train = np.asarray([id_to_index[s] for s in fold["train_subject_ids"]], dtype=int)
        test = np.asarray([id_to_index[s] for s in fold["test_subject_ids"]], dtype=int)
        inner_folds = inner_payload["outer_folds"][str(fold_index)]["folds"]
        best_alpha, mean_inner_mae = inner_select_alpha(
            x, y, id_to_index, inner_folds, alphas, args.solver
        )
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=best_alpha, solver=args.solver)),
            ]
        )
        pipeline.fit(x[train], y[train])
        prediction = pipeline.predict(x[test])
        training_prediction = pipeline.predict(x[train])
        training_bag = training_prediction - y[train]
        age_slope, intercept = np.polyfit(y[train], training_bag, deg=1)
        test_raw_bag = prediction - y[test]
        test_cbag = test_raw_bag - (intercept + age_slope * y[test])
        oof[test] = prediction
        prediction_fold[test] = fold_index
        selected_alpha[test] = best_alpha
        raw_bag[test] = test_raw_bag
        cbag[test] = test_cbag
        corrected_predicted_age[test] = y[test] + test_cbag
        fold_results.append(
            {
                "outer_fold": fold_index,
                "n_train": len(train),
                "n_test": len(test),
                "selected_alpha": best_alpha,
                "mae": float(mean_absolute_error(y[test], prediction)),
                "pearson_r": safe_r(y[test], prediction),
                "cbag_mean_absolute_value": float(np.mean(np.abs(test_cbag))),
                "cbag_age_pearson_r": safe_r(y[test], test_cbag),
            }
        )
        cbag_coefficient_rows.append(
            {
                "outer_fold": fold_index,
                "training_n": len(train),
                "training_bag_intercept": float(intercept),
                "training_bag_age_slope": float(age_slope),
            }
        )
        for alpha, inner_mae in zip(alphas, mean_inner_mae):
            alpha_curve_rows.append(
                {
                    "outer_fold": fold_index,
                    "alpha": float(alpha),
                    "mean_inner_validation_mae": float(inner_mae),
                    "selected": bool(np.isclose(alpha, best_alpha)),
                    "selected_at_minimum_boundary": bool(
                        np.isclose(alpha, best_alpha) and np.isclose(best_alpha, alphas[0])
                    ),
                    "selected_at_maximum_boundary": bool(
                        np.isclose(alpha, best_alpha) and np.isclose(best_alpha, alphas[-1])
                    ),
                }
            )
        alpha_search[str(fold_index)] = {
            "alphas": alphas.tolist(),
            "mean_inner_validation_mae": mean_inner_mae,
            "selected_alpha": best_alpha,
        }
        print(f"completed_outer_fold={fold_index}", flush=True)
    evaluated = np.isfinite(oof)
    predictions = pd.DataFrame(
        {
            "row_index": subjects.loc[evaluated, "row_index"].to_numpy(dtype=int),
            "true_age": y[evaluated],
            "predicted_age": oof[evaluated],
            "outer_fold": prediction_fold[evaluated],
            "selected_alpha": selected_alpha[evaluated],
            "absolute_error": np.abs(oof[evaluated] - y[evaluated]),
            "brain_age_gap": raw_bag[evaluated],
            "cbag": cbag[evaluated],
            "cbag_corrected_predicted_age": corrected_predicted_age[evaluated],
        }
    )
    predictions.to_csv(output_dir / f"{args.representation}_oof_predictions.csv", index=False)
    pd.DataFrame(fold_results).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(alpha_curve_rows).to_csv(output_dir / "inner_alpha_mae_curves.csv", index=False)
    pd.DataFrame(cbag_coefficient_rows).to_csv(
        output_dir / "foldwise_cbag_coefficients.csv", index=False
    )
    (output_dir / "alpha_search.json").write_text(
        json.dumps(alpha_search, indent=2) + "\n", encoding="utf-8"
    )
    metrics = {
        "status": "complete" if evaluated.all() else "partial_dry_run",
        "representation": args.representation,
        "ridge_solver": args.solver,
        "matrix_shape": list(x.shape),
        "n_evaluated": int(evaluated.sum()),
        "overall_oof_mae": float(mean_absolute_error(y[evaluated], oof[evaluated])),
        "overall_oof_pearson_r": safe_r(y[evaluated], oof[evaluated]),
        "foldwise_cbag_method": (
            "Within each outer fold, regress in-sample outer-training BAG on chronological age; "
            "subtract that training-derived linear bias from held-out outer-test BAG"
        ),
        "overall_cbag_mean_absolute_value": float(np.mean(np.abs(cbag[evaluated]))),
        "overall_cbag_age_pearson_r": safe_r(y[evaluated], cbag[evaluated]),
        "alpha_grid": alphas.tolist(),
        "fold_metrics": fold_results,
        "elapsed_seconds": time.time() - start,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    save_figures(predictions, args.representation, output_dir)
    print(json.dumps({"status": metrics["status"], "representation": args.representation, "n": int(evaluated.sum())}))


if __name__ == "__main__":
    main()
