#!/usr/bin/env python3
"""Merge independent outer-fold Ridge shards into one standard model output.

Description
-----------
Merge independent outer-fold Ridge shards into one standard model output.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--representation`` command-line input
- ``--shard-root`` command-line input
- ``--output-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure5/scripts/merge_ridge_outer_fold_shards.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_ridge import safe_r, save_figures  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-outer-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dirs = sorted(args.shard_root.glob(f"shard_*/{args.representation}"))
    if not model_dirs:
        raise FileNotFoundError(f"No shard outputs under {args.shard_root}")

    predictions: list[pd.DataFrame] = []
    fold_metrics: list[pd.DataFrame] = []
    alpha_curves: list[pd.DataFrame] = []
    cbag_coefficients: list[pd.DataFrame] = []
    alpha_search: dict[str, object] = {}
    metrics_by_shard: list[dict[str, object]] = []
    for model_dir in model_dirs:
        predictions.append(
            pd.read_csv(model_dir / f"{args.representation}_oof_predictions.csv")
        )
        fold_metrics.append(pd.read_csv(model_dir / "fold_metrics.csv"))
        alpha_curves.append(pd.read_csv(model_dir / "inner_alpha_mae_curves.csv"))
        cbag_coefficients.append(
            pd.read_csv(model_dir / "foldwise_cbag_coefficients.csv")
        )
        current_search = json.loads((model_dir / "alpha_search.json").read_text())
        overlap = set(alpha_search) & set(current_search)
        if overlap:
            raise RuntimeError(f"Duplicated alpha-search folds: {sorted(overlap)}")
        alpha_search.update(current_search)
        metrics_by_shard.append(json.loads((model_dir / "metrics.json").read_text()))

    prediction = pd.concat(predictions, ignore_index=True).sort_values("row_index")
    folds = pd.concat(fold_metrics, ignore_index=True).sort_values("outer_fold")
    curves = pd.concat(alpha_curves, ignore_index=True).sort_values(
        ["outer_fold", "alpha"]
    )
    coefficients = pd.concat(cbag_coefficients, ignore_index=True).sort_values(
        "outer_fold"
    )
    expected_folds = set(range(args.expected_outer_folds))
    observed_folds = set(folds.outer_fold.astype(int))
    if observed_folds != expected_folds or set(alpha_search) != {
        str(value) for value in expected_folds
    }:
        raise RuntimeError(
            f"Incomplete outer folds: metrics={sorted(observed_folds)}, "
            f"alpha={sorted(alpha_search)}"
        )
    if folds.outer_fold.duplicated().any() or coefficients.outer_fold.duplicated().any():
        raise RuntimeError("An outer fold appears in more than one shard")
    if prediction.row_index.duplicated().any() or len(prediction) != 196:
        raise RuntimeError(
            f"Expected 196 unique OOF predictions, found {len(prediction)}"
        )
    if set(prediction.outer_fold.astype(int)) != expected_folds:
        raise RuntimeError("OOF predictions do not cover every expected outer fold")
    if prediction[["true_age", "predicted_age"]].isna().any().any():
        raise RuntimeError("Merged OOF predictions contain missing ages")

    output_dir = args.output_root / args.representation
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(
        output_dir / f"{args.representation}_oof_predictions.csv", index=False
    )
    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    curves.to_csv(output_dir / "inner_alpha_mae_curves.csv", index=False)
    coefficients.to_csv(output_dir / "foldwise_cbag_coefficients.csv", index=False)
    (output_dir / "alpha_search.json").write_text(
        json.dumps(alpha_search, indent=2) + "\n", encoding="utf-8"
    )

    y = prediction.true_age.to_numpy(dtype=float)
    predicted = prediction.predicted_age.to_numpy(dtype=float)
    cbag = prediction.cbag.to_numpy(dtype=float)
    shard_elapsed_seconds = [
        float(item["elapsed_seconds"]) for item in metrics_by_shard
    ]
    maximum_shard_elapsed_seconds = float(max(shard_elapsed_seconds))
    peak_shard_rss_kb = int(max(item["peak_rss_kb"] for item in metrics_by_shard))
    metrics = {
        "status": "complete",
        "representation": args.representation,
        "ridge_solver": metrics_by_shard[0]["ridge_solver"],
        "matrix_shape": metrics_by_shard[0]["matrix_shape"],
        "n_evaluated": len(prediction),
        "overall_oof_mae": float(mean_absolute_error(y, predicted)),
        "overall_oof_pearson_r": safe_r(y, predicted),
        "foldwise_cbag_method": metrics_by_shard[0]["foldwise_cbag_method"],
        "overall_cbag_mean_absolute_value": float(np.mean(np.abs(cbag))),
        "overall_cbag_age_pearson_r": safe_r(y, cbag),
        "alpha_grid": metrics_by_shard[0]["alpha_grid"],
        "fold_metrics": folds.to_dict(orient="records"),
        "parallel_outer_fold_shards": len(model_dirs),
        # Canonical summary fields use parallel wall time and maximum per-shard RSS.
        "elapsed_seconds": maximum_shard_elapsed_seconds,
        "peak_rss_kb": peak_shard_rss_kb,
        # Preserve the shard-level provenance used to derive the canonical fields.
        "shard_elapsed_seconds": shard_elapsed_seconds,
        "maximum_shard_elapsed_seconds": maximum_shard_elapsed_seconds,
        "peak_shard_rss_kb": peak_shard_rss_kb,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    save_figures(prediction, args.representation, output_dir)
    report = {
        "status": "complete",
        "representation": args.representation,
        "n_shards": len(model_dirs),
        "outer_folds": sorted(observed_folds),
        "n_oof_predictions": len(prediction),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "parallel_merge_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
