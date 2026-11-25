#!/usr/bin/env python3
"""Nested-CV brain-age comparison of warped delta-MD and raw ico4 MD.

This reuses the Figure 5 fold construction and Ridge modelling specification:
age-stratified nested 5x5 CV, seed 20261027, 21 alphas from 1e-4 to 1e6,
training-fold median imputation, training-fold standardization, and Ridge with
solver='auto'. The two representations share the exact same folds and ages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
FIGURE5_SCRIPTS = REPO / "analysis_results/figure5/scripts"
sys.path.insert(0, str(FIGURE5_SCRIPTS))
from build_folds import age_bins  # noqa: E402


DEFAULT_DELTA = REPO / "analysis_results/figure6_new/outputs/warped_delta_382DBdHx"
DEFAULT_MD = Path(
    "/mnt/newStor/paros/paros_WORK/xintong/results/"
    "figure5_age_prediction/outputs/features"
)
DEFAULT_OUTPUT = (
    REPO / "analysis_results/figure6_new/outputs/brain_age_delta_vs_raw_md_156"
)
POSITION_KEYS = [
    "hemisphere", "ico_level", "ico_node_id", "fsaverage_vertex_id",
    "dk_label_id", "dk_label_name", "depth_index", "depth_fraction",
    "depth_direction",
]
FOLD_SEED = 20261027
BOOTSTRAP_SEED = 20261028
N_SPLITS = 5
ALPHAS = np.logspace(-4, 6, 21)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-run", type=Path, default=DEFAULT_DELTA)
    parser.add_argument("--md-features", type=Path, default=DEFAULT_MD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-outer-folds", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--solver", choices=("auto", "lsqr"), default="auto")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.isclose(np.std(y_true), 0) or np.isclose(np.std(y_pred), 0):
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def make_folds(subject_ids: np.ndarray, ages: np.ndarray, output: Path) -> tuple[list, dict]:
    labels, n_bins = age_bins(ages, N_SPLITS)
    outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED)
    outer_payload = {
        "seed": FOLD_SEED, "n_splits": N_SPLITS,
        "stratification": "age quantile bins; Figure 5 build_folds.py policy",
        "n_age_bins": n_bins, "folds": [],
    }
    inner_payload = {"seed": FOLD_SEED, "n_splits": N_SPLITS, "outer_folds": {}}
    summary = []
    seen = []
    for outer_fold, (train, test) in enumerate(outer_cv.split(subject_ids, labels)):
        train_ids = subject_ids[train].tolist()
        test_ids = subject_ids[test].tolist()
        seen.extend(test_ids)
        outer_payload["folds"].append({
            "outer_fold": outer_fold,
            "train_subject_ids": train_ids,
            "test_subject_ids": test_ids,
        })
        for split, indices in (("train", train), ("test", test)):
            summary.append({
                "outer_fold": outer_fold, "split": split, "n": len(indices),
                "age_mean": float(ages[indices].mean()),
                "age_sd": float(ages[indices].std(ddof=1)),
                "age_min": float(ages[indices].min()),
                "age_max": float(ages[indices].max()),
            })
        inner_labels, inner_bins = age_bins(ages[train], N_SPLITS)
        inner_cv = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED + outer_fold + 1
        )
        inner_folds = []
        for inner_fold, (inner_train, validation) in enumerate(
            inner_cv.split(train, inner_labels)
        ):
            inner_folds.append({
                "inner_fold": inner_fold,
                "train_subject_ids": subject_ids[train[inner_train]].tolist(),
                "validation_subject_ids": subject_ids[train[validation]].tolist(),
            })
        inner_payload["outer_folds"][str(outer_fold)] = {
            "n_age_bins": inner_bins, "folds": inner_folds,
        }
    if len(seen) != len(subject_ids) or len(set(seen)) != len(subject_ids):
        raise RuntimeError("Outer folds do not partition the 156 subjects exactly once")
    write_json(output / "outer_folds.json", outer_payload)
    write_json(output / "inner_folds_by_outer_fold.json", inner_payload)
    pd.DataFrame(summary).to_csv(output / "fold_age_summary.csv", index=False)
    return outer_payload["folds"], inner_payload


def training_feature_mask(x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.any(np.isfinite(x[rows]), axis=0)


def fit_representation(
    name: str,
    x: np.ndarray,
    ages: np.ndarray,
    subjects: pd.DataFrame,
    folds: list[dict[str, object]],
    inner_payload: dict[str, object],
    output: Path,
    solver: str,
    max_outer_folds: int | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    representation_dir = output / name
    representation_dir.mkdir(parents=True, exist_ok=True)
    ids = subjects.subject_id.tolist()
    id_to_index = {subject_id: index for index, subject_id in enumerate(ids)}
    use_folds = folds if max_outer_folds is None else folds[:max_outer_folds]
    oof = np.full(len(subjects), np.nan)
    prediction_fold = np.full(len(subjects), -1, dtype=int)
    selected_alpha = np.full(len(subjects), np.nan)
    fold_rows = []
    alpha_rows = []
    feature_rows = []
    started = time.monotonic()

    for fold in use_folds:
        outer_fold = int(fold["outer_fold"])
        train = np.asarray([id_to_index[s] for s in fold["train_subject_ids"]], dtype=int)
        test = np.asarray([id_to_index[s] for s in fold["test_subject_ids"]], dtype=int)
        inner_folds = inner_payload["outer_folds"][str(outer_fold)]["folds"]
        scores = np.zeros((len(inner_folds), len(ALPHAS)))
        for inner_index, split in enumerate(inner_folds):
            inner_train = np.asarray(
                [id_to_index[s] for s in split["train_subject_ids"]], dtype=int
            )
            validation = np.asarray(
                [id_to_index[s] for s in split["validation_subject_ids"]], dtype=int
            )
            mask = training_feature_mask(x, inner_train)
            feature_rows.append({
                "representation": name, "outer_fold": outer_fold,
                "stage": "inner", "inner_fold": inner_index,
                "n_training": len(inner_train), "n_validation": len(validation),
                "n_input_features": x.shape[1], "n_usable_features": int(mask.sum()),
                "n_all_missing_training_features": int((~mask).sum()),
            })
            preprocessing = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            x_train = preprocessing.fit_transform(np.asarray(x[inner_train][:, mask]))
            x_validation = preprocessing.transform(np.asarray(x[validation][:, mask]))
            for alpha_index, alpha in enumerate(ALPHAS):
                model = Ridge(alpha=float(alpha), solver=solver)
                model.fit(x_train, ages[inner_train])
                scores[inner_index, alpha_index] = mean_absolute_error(
                    ages[validation], model.predict(x_validation)
                )
            del x_train, x_validation
        mean_scores = scores.mean(axis=0)
        best_index = int(np.argmin(mean_scores))
        best_alpha = float(ALPHAS[best_index])
        for alpha, score in zip(ALPHAS, mean_scores):
            alpha_rows.append({
                "representation": name, "outer_fold": outer_fold,
                "alpha": float(alpha), "mean_inner_validation_mae": float(score),
                "selected": bool(np.isclose(alpha, best_alpha)),
            })

        mask = training_feature_mask(x, train)
        feature_rows.append({
            "representation": name, "outer_fold": outer_fold,
            "stage": "outer", "inner_fold": np.nan,
            "n_training": len(train), "n_validation": len(test),
            "n_input_features": x.shape[1], "n_usable_features": int(mask.sum()),
            "n_all_missing_training_features": int((~mask).sum()),
        })
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=best_alpha, solver=solver)),
        ])
        pipeline.fit(np.asarray(x[train][:, mask]), ages[train])
        prediction = pipeline.predict(np.asarray(x[test][:, mask]))
        oof[test] = prediction
        prediction_fold[test] = outer_fold
        selected_alpha[test] = best_alpha
        fold_rows.append({
            "representation": name, "outer_fold": outer_fold,
            "n_train": len(train), "n_test": len(test),
            "selected_alpha": best_alpha,
            "mae": float(mean_absolute_error(ages[test], prediction)),
            "rmse": float(mean_squared_error(ages[test], prediction) ** 0.5),
            "pearson_r": safe_r(ages[test], prediction),
            "r2": float(r2_score(ages[test], prediction)),
            "n_usable_features": int(mask.sum()),
        })
        print(json.dumps({"representation": name, "completed_outer_fold": outer_fold}), flush=True)

    evaluated = np.isfinite(oof)
    predictions = subjects.loc[evaluated, ["row_index", "subject_id", "actual_age"]].copy()
    predictions["representation"] = name
    predictions["outer_fold"] = prediction_fold[evaluated]
    predictions["selected_alpha"] = selected_alpha[evaluated]
    predictions["predicted_age"] = oof[evaluated]
    predictions["error"] = predictions.predicted_age - predictions.actual_age
    predictions["absolute_error"] = predictions.error.abs()
    predictions["squared_error"] = predictions.error ** 2
    predictions.to_csv(representation_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(representation_dir / "outer_fold_metrics.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(representation_dir / "inner_alpha_mae_curves.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(representation_dir / "fold_feature_qc.csv", index=False)
    metrics = {
        "status": "complete" if evaluated.all() else "partial_smoke",
        "representation": name, "n_evaluated": int(evaluated.sum()),
        "matrix_shape": list(x.shape), "solver": solver,
        "alpha_grid": ALPHAS.tolist(),
        "mae": float(mean_absolute_error(ages[evaluated], oof[evaluated])),
        "rmse": float(mean_squared_error(ages[evaluated], oof[evaluated]) ** 0.5),
        "pearson_r": safe_r(ages[evaluated], oof[evaluated]),
        "r2": float(r2_score(ages[evaluated], oof[evaluated])),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(representation_dir / "metrics.json", metrics)
    return predictions, metrics


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    required = [
        args.delta_run / "delta_md.npy", args.delta_run / "participants.csv",
        args.delta_run / "positions.csv", args.md_features / "matrices/X_ico4_md.npy",
        args.md_features / "subject_order.csv",
        args.md_features / "feature_metadata/features_ico4_md.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    participants = pd.read_csv(args.delta_run / "participants.csv", dtype={"subject_id": str})
    subjects = participants.loc[participants.model_role.eq("older_cn_target")].copy()
    subjects = subjects.sort_values("scoring_row_index").reset_index(drop=True)
    if len(subjects) != 156 or subjects.subject_id.nunique() != 156:
        raise RuntimeError("Expected 156 unique older-CN targets")
    if set(subjects.matrix_source) != {"figure5_6"}:
        raise RuntimeError("Older-CN targets are not exclusively matrix_source=figure5_6")
    source_order = pd.read_csv(args.md_features / "subject_order.csv", dtype={"subject_id": str})
    source_lookup = source_order.set_index("row_index")
    located = source_lookup.loc[subjects.matrix_row_index.astype(int)]
    if not np.array_equal(subjects.subject_id.to_numpy(), located.subject_id.to_numpy()):
        raise RuntimeError("matrix_row_index does not reproduce older-CN subject IDs")
    if not np.allclose(subjects.actual_age, located.age_at_scan, atol=0, rtol=0):
        raise RuntimeError("Delta participant and raw-MD ages disagree")
    model_subjects = subjects[[
        "subject_id", "scoring_row_index", "matrix_row_index", "matrix_source",
        "model_role", "actual_age",
    ]].copy()
    model_subjects.insert(0, "row_index", np.arange(len(model_subjects), dtype=int))
    model_subjects.to_csv(output / "subject_order.csv", index=False)

    positions = pd.read_csv(args.delta_run / "positions.csv")
    raw_meta = pd.read_csv(args.md_features / "feature_metadata/features_ico4_md.csv")
    if len(positions) != 98427 or len(raw_meta) != 98427:
        raise RuntimeError("Expected 98,427 positions in both representations")
    if positions.duplicated(POSITION_KEYS).any() or raw_meta.duplicated(POSITION_KEYS).any():
        raise RuntimeError("Duplicate position identity key")
    raw_key_to_column = {
        tuple(row): int(index)
        for index, row in enumerate(raw_meta[POSITION_KEYS].itertuples(index=False, name=None))
    }
    raw_reorder = np.asarray([
        raw_key_to_column.get(tuple(row), -1)
        for row in positions[POSITION_KEYS].itertuples(index=False, name=None)
    ], dtype=int)
    if np.any(raw_reorder < 0) or len(np.unique(raw_reorder)) != 98427:
        raise RuntimeError("Could not establish a one-to-one position alignment")
    alignment = positions[POSITION_KEYS + ["feature_index"]].copy()
    alignment = alignment.rename(columns={"feature_index": "delta_feature_index"})
    alignment.insert(0, "aligned_feature_index", np.arange(98427, dtype=int))
    alignment["raw_feature_index"] = raw_meta.iloc[raw_reorder].feature_index.to_numpy(int)
    alignment["raw_column_before_reorder"] = raw_reorder
    alignment["already_identical_order"] = raw_reorder == np.arange(98427)
    alignment.to_csv(output / "position_alignment.csv", index=False)

    delta_source = np.load(args.delta_run / "delta_md.npy", mmap_mode="r")
    raw_source = np.load(args.md_features / "matrices/X_ico4_md.npy", mmap_mode="r")
    if delta_source.shape not in ((180, 4687, 21), (180, 98427)):
        raise RuntimeError(f"Unexpected delta matrix shape: {delta_source.shape}")
    delta = np.asarray(
        delta_source[subjects.scoring_row_index.astype(int)]
    ).reshape(156, 98427)
    raw = np.asarray(
        raw_source[subjects.matrix_row_index.astype(int)][:, raw_reorder]
    )
    ages = subjects.actual_age.to_numpy(float)

    missing_rows = []
    for name, matrix in (("delta_md", delta), ("raw_md", raw)):
        missing_rows.append({
            "representation": name, "n_subjects": matrix.shape[0],
            "n_features": matrix.shape[1], "n_values": matrix.size,
            "n_missing": int(np.isnan(matrix).sum()),
            "missing_fraction": float(np.isnan(matrix).mean()),
            "n_all_missing_features_full_cohort": int(np.isnan(matrix).all(axis=0).sum()),
            "subject_missing_fraction_min": float(np.isnan(matrix).mean(axis=1).min()),
            "subject_missing_fraction_max": float(np.isnan(matrix).mean(axis=1).max()),
        })
    pd.DataFrame(missing_rows).to_csv(output / "input_missingness_qc.csv", index=False)
    write_json(output / "input_alignment_report.json", {
        "status": "complete", "n_subjects": 156, "n_positions": 98427,
        "subject_ids_unique": True, "subject_rows_match_raw_md_order": True,
        "ages_match": True, "position_keys": POSITION_KEYS,
        "position_order_already_identical": bool(np.array_equal(raw_reorder, np.arange(98427))),
        "young_cn_in_ridge_queue": 0,
        "excluded_leaky_delta_derivatives": [
            "age_sex_extra_md.npy", "age_sex_thickness_extra_md.npy",
            "age_sex_z.npy", "age_sex_thickness_z.npy",
        ],
        "input_sha256": {
            "participants": sha256(args.delta_run / "participants.csv"),
            "positions": sha256(args.delta_run / "positions.csv"),
            "delta_md": sha256(args.delta_run / "delta_md.npy"),
            "raw_subject_order": sha256(args.md_features / "subject_order.csv"),
            "raw_position_metadata": sha256(
                args.md_features / "feature_metadata/features_ico4_md.csv"
            ),
            "raw_md": sha256(args.md_features / "matrices/X_ico4_md.npy"),
        },
    })

    folds, inner_payload = make_folds(
        subjects.subject_id.to_numpy(), ages, output
    )
    delta_predictions, delta_metrics = fit_representation(
        "delta_md", delta, ages, model_subjects, folds, inner_payload,
        output, args.solver, args.max_outer_folds,
    )
    raw_predictions, raw_metrics = fit_representation(
        "raw_md", raw, ages, model_subjects, folds, inner_payload,
        output, args.solver, args.max_outer_folds,
    )
    full_run = args.max_outer_folds is None
    if full_run:
        paired = delta_predictions.merge(
            raw_predictions,
            on=["row_index", "subject_id", "actual_age", "outer_fold"],
            suffixes=("_delta_md", "_raw_md"), validate="one_to_one",
        ).sort_values("row_index").reset_index(drop=True)
        if len(paired) != 156 or paired.subject_id.nunique() != 156:
            raise RuntimeError("Representations do not have 156 paired OOF predictions")
        paired["absolute_error_difference_delta_minus_raw"] = (
            paired.absolute_error_delta_md - paired.absolute_error_raw_md
        )
        paired.to_csv(output / "paired_oof_predictions.csv", index=False)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        difference = paired.absolute_error_difference_delta_minus_raw.to_numpy(float)
        bootstrap_indices = rng.integers(
            0, len(paired), size=(args.bootstrap_replicates, len(paired))
        )
        bootstrap = difference[bootstrap_indices].mean(axis=1)
        low, high = np.percentile(bootstrap, [2.5, 97.5])
        comparison = {
            "status": "complete", "n_paired": 156,
            "estimand": "MAE_delta_md_minus_MAE_raw_md",
            "estimate": float(difference.mean()),
            "ci95_lower": float(low), "ci95_upper": float(high),
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "interpretation": "negative values favor delta_md",
        }
        write_json(output / "paired_mae_bootstrap.json", comparison)
        pd.DataFrame([delta_metrics, raw_metrics]).to_csv(
            output / "overall_metrics.csv", index=False
        )
        write_json(output / "run_report.json", {
            "status": "complete", "n_subjects": 156,
            "shared_nested_cv": "age-stratified 5x5",
            "fold_seed": FOLD_SEED, "solver": args.solver,
            "alpha_grid": ALPHAS.tolist(),
            "preprocessing": "training-fold median imputation then StandardScaler",
            "delta_metrics": delta_metrics, "raw_metrics": raw_metrics,
            "paired_mae": comparison,
            "python": platform.python_version(),
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        })
        print(json.dumps({
            "status": "complete", "n": 156,
            "delta_mae": delta_metrics["mae"], "raw_mae": raw_metrics["mae"],
            "delta_minus_raw_mae": comparison["estimate"],
            "ci95": [comparison["ci95_lower"], comparison["ci95_upper"]],
            "output_dir": str(output),
        }))
    else:
        write_json(output / "smoke_report.json", {
            "status": "complete", "outer_folds_run": args.max_outer_folds,
            "delta_metrics": delta_metrics, "raw_metrics": raw_metrics,
        })
        print(json.dumps({"status": "smoke_complete", "output_dir": str(output)}))


if __name__ == "__main__":
    main()
