#!/usr/bin/env python3
"""Compute leakage-resistant foldwise cBAG using cross-fitted outer-training predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_ridge import REPRESENTATIONS, load_representation, safe_r


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=sorted(REPRESENTATIONS), required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--solver", choices=("lsqr",), default="lsqr")
    return parser.parse_args()


def pipeline(alpha: float, solver: str) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, solver=solver)),
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.representation
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = pd.read_csv(args.feature_dir / "subject_order.csv", dtype={"subject_id": str})
    ids = subjects["subject_id"].tolist()
    id_to_index = {subject_id: index for index, subject_id in enumerate(ids)}
    y = subjects["age_at_scan"].to_numpy(dtype=float)
    x = load_representation(args.feature_dir, args.representation)
    outer = json.loads((args.fold_dir / "outer_folds.json").read_text(encoding="utf-8"))
    inner = json.loads(
        (args.fold_dir / "inner_folds_by_outer_fold.json").read_text(encoding="utf-8")
    )
    fold_metrics = pd.read_csv(args.model_root / args.representation / "fold_metrics.csv")
    selected_alpha = dict(zip(fold_metrics["outer_fold"], fold_metrics["selected_alpha"]))
    raw_prediction = pd.read_csv(
        args.model_root
        / args.representation
        / f"{args.representation}_oof_predictions.csv"
    ).set_index("row_index")
    cbag = np.full(len(y), np.nan)
    coefficients: list[dict[str, object]] = []
    for outer_fold in outer["folds"]:
        fold_index = int(outer_fold["outer_fold"])
        train = np.asarray([id_to_index[s] for s in outer_fold["train_subject_ids"]], dtype=int)
        test = np.asarray([id_to_index[s] for s in outer_fold["test_subject_ids"]], dtype=int)
        crossfit_prediction = np.full(len(y), np.nan)
        for split in inner["outer_folds"][str(fold_index)]["folds"]:
            inner_train = np.asarray(
                [id_to_index[s] for s in split["train_subject_ids"]], dtype=int
            )
            validation = np.asarray(
                [id_to_index[s] for s in split["validation_subject_ids"]], dtype=int
            )
            model = pipeline(float(selected_alpha[fold_index]), args.solver)
            model.fit(x[inner_train], y[inner_train])
            crossfit_prediction[validation] = model.predict(x[validation])
        if not np.isfinite(crossfit_prediction[train]).all():
            raise RuntimeError(f"outer fold {fold_index}: incomplete cross-fitted training predictions")
        crossfit_training_bag = crossfit_prediction[train] - y[train]
        age_slope, intercept = np.polyfit(y[train], crossfit_training_bag, deg=1)
        fold_raw_prediction = raw_prediction.loc[test, "predicted_age"].to_numpy()
        fold_raw_bag = fold_raw_prediction - y[test]
        cbag[test] = fold_raw_bag - (intercept + age_slope * y[test])
        coefficients.append(
            {
                "outer_fold": fold_index,
                "selected_alpha": float(selected_alpha[fold_index]),
                "calibration_n": len(train),
                "crossfit_bag_intercept": float(intercept),
                "crossfit_bag_age_slope": float(age_slope),
                "crossfit_training_bag_age_pearson_r": safe_r(y[train], crossfit_training_bag),
            }
        )
        print(f"completed_crossfit_cbag_outer_fold={fold_index}", flush=True)
    if not np.isfinite(cbag).all():
        raise RuntimeError("cBAG is incomplete")
    row_index = subjects["row_index"].to_numpy(dtype=int)
    raw_ordered = raw_prediction.loc[row_index]
    output = pd.DataFrame(
        {
            "row_index": row_index,
            "true_age": y,
            "outer_fold": raw_ordered["outer_fold"].to_numpy(dtype=int),
            "raw_predicted_age": raw_ordered["predicted_age"].to_numpy(),
            "raw_brain_age_gap": raw_ordered["predicted_age"].to_numpy() - y,
            "crossfit_cbag": cbag,
            "crossfit_cbag_corrected_predicted_age": y + cbag,
        }
    )
    output.to_csv(output_dir / f"{args.representation}_crossfit_cbag.csv", index=False)
    pd.DataFrame(coefficients).to_csv(
        output_dir / "crossfit_cbag_fold_coefficients.csv", index=False
    )
    metrics = {
        "status": "complete",
        "representation": args.representation,
        "solver": args.solver,
        "n": len(y),
        "method": (
            "For each outer fold, obtain 5-fold cross-fitted predictions for the outer-training "
            "subjects using that outer fold's selected alpha; regress cross-fitted training BAG on "
            "age; subtract the training-derived linear bias from held-out outer-test BAG"
        ),
        "crossfit_cbag_mean_absolute_value": float(np.mean(np.abs(cbag))),
        "crossfit_cbag_age_pearson_r": safe_r(y, cbag),
        "crossfit_cbag_mean": float(np.mean(cbag)),
        "crossfit_cbag_sd": float(np.std(cbag, ddof=1)),
    }
    (output_dir / "crossfit_cbag_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
