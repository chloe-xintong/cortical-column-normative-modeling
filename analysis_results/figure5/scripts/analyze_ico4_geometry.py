#!/usr/bin/env python3
"""Summarize ico4 geometry increments with paired bootstrap and FDR.

Description
-----------
Summarize ico4 geometry increments with paired bootstrap and FDR.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--model-root`` command-line input
- ``--cbag-root`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure5/scripts/analyze_ico4_geometry.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_lsqr_stability import fdr_bh


REPRESENTATIONS = [
    "ico4_md",
    "ico4_md_length",
    "ico4_md_curvature",
    "ico4_md_length_curvature",
]
COMPARISONS = [
    ("ico4_md", "ico4_md_length"),
    ("ico4_md", "ico4_md_curvature"),
    ("ico4_md", "ico4_md_length_curvature"),
]


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
    summary_rows: list[dict[str, object]] = []
    cbag_rows: list[dict[str, object]] = []
    alpha_curves: list[pd.DataFrame] = []
    for representation in REPRESENTATIONS:
        model_dir = args.model_root / representation
        metrics = json.loads((model_dir / "metrics.json").read_text())
        prediction = pd.read_csv(model_dir / f"{representation}_oof_predictions.csv").sort_values(
            "row_index"
        )
        predictions[representation] = prediction
        summary_rows.append(
            {
                "representation": representation,
                "n": metrics["n_evaluated"],
                "mae": metrics["overall_oof_mae"],
                "pearson_r": metrics["overall_oof_pearson_r"],
                "solver": metrics["ridge_solver"],
                "n_features": metrics["matrix_shape"][1],
            }
        )
        cbag_metrics = json.loads(
            (args.cbag_root / representation / "crossfit_cbag_metrics.json").read_text()
        )
        cbag_rows.append(
            {
                "representation": representation,
                "n": cbag_metrics["n"],
                "crossfit_cbag_mean_absolute_value": cbag_metrics[
                    "crossfit_cbag_mean_absolute_value"
                ],
                "crossfit_cbag_age_pearson_r": cbag_metrics["crossfit_cbag_age_pearson_r"],
                "crossfit_cbag_mean": cbag_metrics["crossfit_cbag_mean"],
                "crossfit_cbag_sd": cbag_metrics["crossfit_cbag_sd"],
            }
        )
        curve = pd.read_csv(model_dir / "inner_alpha_mae_curves.csv")
        curve.insert(0, "representation", representation)
        alpha_curves.append(curve)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "ico4_geometry_model_summary.csv", index=False)
    pd.DataFrame(cbag_rows).to_csv(args.output_dir / "ico4_geometry_crossfit_cbag_summary.csv", index=False)
    all_curves = pd.concat(alpha_curves, ignore_index=True)
    all_curves.to_csv(args.output_dir / "ico4_geometry_inner_alpha_mae_curves.csv", index=False)
    selected = all_curves.loc[all_curves["selected"]].copy()
    selected.to_csv(args.output_dir / "ico4_geometry_selected_alphas.csv", index=False)

    rng = np.random.default_rng(args.seed)
    comparison_rows: list[dict[str, object]] = []
    for representation_a, representation_b in COMPARISONS:
        a = predictions[representation_a]
        b = predictions[representation_b]
        if not np.array_equal(a["row_index"], b["row_index"]):
            raise RuntimeError(f"Subject order mismatch: {representation_a}, {representation_b}")
        difference = a["absolute_error"].to_numpy() - b["absolute_error"].to_numpy()
        indices = rng.integers(0, len(difference), (args.bootstrap_replicates, len(difference)))
        bootstrap = difference[indices].mean(axis=1)
        lower, upper = np.percentile(bootstrap, [2.5, 97.5])
        p_lower = (np.sum(bootstrap <= 0) + 1) / (args.bootstrap_replicates + 1)
        p_upper = (np.sum(bootstrap >= 0) + 1) / (args.bootstrap_replicates + 1)
        comparison_rows.append(
            {
                "representation_a": representation_a,
                "representation_b": representation_b,
                "n": len(difference),
                "delta_mae_a_minus_b": float(difference.mean()),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "bootstrap_two_sided_p": float(min(1.0, 2 * min(p_lower, p_upper))),
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": args.seed,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison["fdr_bh_q"] = fdr_bh(comparison["bootstrap_two_sided_p"].to_numpy())
    comparison["fdr_bh_significant_0_05"] = comparison["fdr_bh_q"] < 0.05
    comparison.to_csv(args.output_dir / "ico4_geometry_paired_bootstrap.csv", index=False)
    report = {
        "status": "complete",
        "n_subjects": int(next(iter(predictions.values())).shape[0]),
        "n_models": len(REPRESENTATIONS),
        "n_predefined_comparisons": len(COMPARISONS),
        "bootstrap_replicates": args.bootstrap_replicates,
        "n_selected_at_minimum_alpha": int(selected["selected_at_minimum_boundary"].sum()),
        "n_selected_at_maximum_alpha": int(selected["selected_at_maximum_boundary"].sum()),
    }
    (args.output_dir / "ico4_geometry_analysis_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
