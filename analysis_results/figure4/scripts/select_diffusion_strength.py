#!/usr/bin/env python3
"""Select Figure 4B mesh-diffusion strength using outcome-blind ACF error.

Description
-----------
Select Figure 4B mesh-diffusion strength using outcome-blind ACF error.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--candidate`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/select_diffusion_strength.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_candidate(value: str) -> tuple[float, Path]:
    strength, path = value.split("=", 1)
    return float(strength), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", required=True,
        help="Candidate formatted as STRENGTH=OUTPUT_ROOT",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for candidate in args.candidate:
        strength, root = parse_candidate(candidate)
        for subject_dir in sorted((root / "subjects").glob("*")):
            completion = subject_dir / "completion.json"
            acf_path = subject_dir / "figure4_surface_pca_spatial_acf.csv"
            curve_path = subject_dir / "figure4_surface_pca_null_subject_bins.csv"
            if not completion.exists() or not acf_path.exists() or not curve_path.exists():
                continue
            status = json.loads(completion.read_text())
            if status.get("status") != "complete":
                continue
            acf = pd.read_csv(acf_path)
            targets = json.loads(
                (subject_dir / "figure4_surface_pca_targets.json").read_text()
            )
            keys = ["hemisphere", "depth_index", "distance_bin_index"]
            real = acf.loc[acf.data_type.eq("real")].set_index(keys).spatial_acf
            # Calibration jobs share the same seed and rotation. Use the first
            # realization for every strength so Monte Carlo variation cannot
            # influence the selected smoothing parameter.
            surrogate = acf.loc[
                acf.data_type.eq("surrogate") & acf.replicate.eq(0)
            ].set_index(keys).spatial_acf
            difference = surrogate - real
            difference_table = difference.rename("difference").reset_index()
            difference_table["distance_range_mm"] = pd.cut(
                difference_table["distance_bin_index"],
                bins=[-1, 2, 3, 9, 15],
                labels=["0-6", "6-8", "8-20", "20-30"],
            )
            range_metrics = {}
            for distance_range, range_table in difference_table.groupby(
                "distance_range_mm", observed=True
            ):
                values = range_table["difference"]
                key = str(distance_range).replace("-", "_")
                range_metrics[f"acf_bias_{key}_mm"] = float(values.mean())
                range_metrics[f"acf_rmse_{key}_mm"] = float(
                    np.sqrt(np.mean(values**2))
                )
            curve = pd.read_csv(curve_path)
            rows.append({
                "diffusion_strength": strength,
                "diffusion_steps": int(targets.get("diffusion_steps", 1)),
                "subject_id": subject_dir.name,
                "acf_mae": float(difference.abs().mean()),
                "acf_rmse": float(np.sqrt(np.mean(difference**2))),
                "acf_bias": float(difference.mean()),
                "acf_q95_abs_error": float(difference.abs().quantile(.95)),
                "count_mismatch_rows": int((
                    (curve.anchor_count != curve.real_anchor_count)
                    | (curve.pair_count != curve.real_pair_count)
                ).sum()),
                "all_curve_values_finite": bool(np.isfinite(
                    curve[["real_dissimilarity", "null_dissimilarity", "null_minus_real"]]
                ).all().all()),
                **range_metrics,
            })
    by_subject = pd.DataFrame(rows)
    if by_subject.empty:
        raise RuntimeError("No completed calibration candidates found")
    summary = (
        by_subject.groupby(["diffusion_strength", "diffusion_steps"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            acf_mae_mean=("acf_mae", "mean"),
            acf_mae_median=("acf_mae", "median"),
            acf_rmse_mean=("acf_rmse", "mean"),
            acf_bias_mean=("acf_bias", "mean"),
            acf_q95_abs_error_mean=("acf_q95_abs_error", "mean"),
            count_mismatch_rows=("count_mismatch_rows", "sum"),
            all_curve_values_finite=("all_curve_values_finite", "all"),
        )
        .sort_values(["diffusion_steps", "diffusion_strength"])
    )
    range_columns = [
        column for column in by_subject.columns
        if column.endswith("_mm")
        and (column.startswith("acf_bias_") or column.startswith("acf_rmse_"))
    ]
    range_summary = (
        by_subject.groupby(["diffusion_strength", "diffusion_steps"], as_index=False)[
            range_columns
        ].mean()
    )
    summary = summary.merge(
        range_summary,
        on=["diffusion_strength", "diffusion_steps"],
        validate="one_to_one",
    )
    eligible = summary.loc[
        summary.count_mismatch_rows.eq(0) & summary.all_curve_values_finite
    ]
    selected = eligible.sort_values(
        ["acf_rmse_mean", "diffusion_steps", "diffusion_strength"]
    ).iloc[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_subject.to_csv(args.output_dir / "diffusion_calibration_by_subject.csv", index=False)
    summary.to_csv(args.output_dir / "diffusion_calibration_summary.csv", index=False)
    decision = {
        "selection_rule": "minimum mean subject distance-bin ACF RMSE; Figure 4B effect not inspected",
        "selected_diffusion_strength": float(selected.diffusion_strength),
        "selected_diffusion_steps": int(selected.diffusion_steps),
        "selected_acf_mae_mean": float(selected.acf_mae_mean),
        "n_subjects": int(selected.n_subjects),
        "preliminary": True,
    }
    (args.output_dir / "diffusion_calibration_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
