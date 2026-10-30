#!/usr/bin/env python3
"""Compare regional MD profile alone with profile plus regional thickness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPRESENTATIONS = ("regional_profile", "regional_profile_thickness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cbag-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261028)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, pd.DataFrame] = {}
    model_rows: list[dict[str, object]] = []
    cbag_rows: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []
    for representation in REPRESENTATIONS:
        model_dir = args.model_root / representation
        metrics = json.loads((model_dir / "metrics.json").read_text())
        model_rows.append(
            {
                "representation": representation,
                "n": metrics["n_evaluated"],
                "n_features": metrics["matrix_shape"][1],
                "mae": metrics["overall_oof_mae"],
                "pearson_r": metrics["overall_oof_pearson_r"],
                "solver": metrics["ridge_solver"],
            }
        )
        predictions[representation] = pd.read_csv(
            model_dir / f"{representation}_oof_predictions.csv"
        ).sort_values("row_index")
        cbag = json.loads(
            (args.cbag_root / representation / "crossfit_cbag_metrics.json").read_text()
        )
        cbag_rows.append(
            {
                "representation": representation,
                "n": cbag["n"],
                "crossfit_cbag_mean_absolute_value": cbag[
                    "crossfit_cbag_mean_absolute_value"
                ],
                "crossfit_cbag_age_pearson_r": cbag["crossfit_cbag_age_pearson_r"],
                "crossfit_cbag_mean": cbag["crossfit_cbag_mean"],
                "crossfit_cbag_sd": cbag["crossfit_cbag_sd"],
            }
        )
        curve = pd.read_csv(model_dir / "inner_alpha_mae_curves.csv")
        curve.insert(0, "representation", representation)
        curves.append(curve)
    pd.DataFrame(model_rows).to_csv(args.output_dir / "model_summary.csv", index=False)
    pd.DataFrame(cbag_rows).to_csv(args.output_dir / "crossfit_cbag_summary.csv", index=False)
    all_curves = pd.concat(curves, ignore_index=True)
    all_curves.to_csv(args.output_dir / "inner_alpha_mae_curves.csv", index=False)
    selected = all_curves.loc[all_curves["selected"]]
    selected.to_csv(args.output_dir / "selected_alphas.csv", index=False)

    a = predictions["regional_profile"]
    b = predictions["regional_profile_thickness"]
    if not np.array_equal(a["row_index"], b["row_index"]):
        raise RuntimeError("Subject order mismatch")
    difference = a["absolute_error"].to_numpy() - b["absolute_error"].to_numpy()
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(difference), (args.bootstrap_replicates, len(difference)))
    bootstrap = difference[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    p_lower = (np.sum(bootstrap <= 0) + 1) / (args.bootstrap_replicates + 1)
    p_upper = (np.sum(bootstrap >= 0) + 1) / (args.bootstrap_replicates + 1)
    comparison = pd.DataFrame(
        [
            {
                "representation_a": "regional_profile",
                "representation_b": "regional_profile_thickness",
                "n": len(difference),
                "delta_mae_a_minus_b": float(difference.mean()),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "bootstrap_two_sided_p": float(min(1.0, 2 * min(p_lower, p_upper))),
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": args.seed,
                "comparison_status": "additional comparison requested after original seven were specified",
            }
        ]
    )
    comparison.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    report = {
        "status": "complete",
        "n_subjects": len(difference),
        "n_profile_features": 1428,
        "n_thickness_features_added": 68,
        "n_combined_features": 1496,
        "bootstrap_replicates": args.bootstrap_replicates,
        "selected_at_minimum_alpha": int(selected["selected_at_minimum_boundary"].sum()),
        "selected_at_maximum_alpha": int(selected["selected_at_maximum_boundary"].sum()),
    }
    (args.output_dir / "analysis_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
