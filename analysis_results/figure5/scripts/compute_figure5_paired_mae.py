#!/usr/bin/env python3
"""Compute exploratory spatial-resolution and ico4-ablation MAE families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPARISONS = [
    {
        "family": "spatial_resolution",
        "family_label": "Exploratory spatial-resolution comparisons",
        "representation_a": "regional_profile",
        "representation_b": "ico3_md",
        "comparison": "Regional 21-depth profile − ico3",
    },
    {
        "family": "spatial_resolution",
        "family_label": "Exploratory spatial-resolution comparisons",
        "representation_a": "regional_profile",
        "representation_b": "ico4_md",
        "comparison": "Regional 21-depth profile − ico4",
    },
    {
        "family": "spatial_resolution",
        "family_label": "Exploratory spatial-resolution comparisons",
        "representation_a": "ico4_md",
        "representation_b": "ico5_md",
        "comparison": "ico4 − ico5",
    },
    {
        "family": "ico4_ablation",
        "family_label": "Exploratory ico4-ablation comparisons",
        "representation_a": "ico4_column_mean_md",
        "representation_b": "ico4_md",
        "comparison": "ico4 column mean MD − ico4",
    },
    {
        "family": "ico4_ablation",
        "family_label": "Exploratory ico4-ablation comparisons",
        "representation_a": "ico4_md",
        "representation_b": "ico4_md_length",
        "comparison": "ico4 − ico4 + length",
    },
    {
        "family": "ico4_ablation",
        "family_label": "Exploratory ico4-ablation comparisons",
        "representation_a": "ico4_md",
        "representation_b": "ico4_md_curvature",
        "comparison": "ico4 − ico4 + curvature",
    },
    {
        "family": "ico4_ablation",
        "family_label": "Exploratory ico4-ablation comparisons",
        "representation_a": "ico4_md",
        "representation_b": "ico4_md_length_curvature",
        "comparison": "ico4 − ico4 + length + curvature",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261028)
    return parser.parse_args()


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
    representations = sorted(
        {item[key] for item in COMPARISONS for key in ("representation_a", "representation_b")}
    )
    predictions: dict[str, pd.DataFrame] = {}
    reference: pd.DataFrame | None = None
    for representation in representations:
        path = args.model_root / representation / f"{representation}_oof_predictions.csv"
        frame = pd.read_csv(path).sort_values("row_index").reset_index(drop=True)
        required = {"row_index", "true_age", "predicted_age", "absolute_error"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"{path} lacks columns: {sorted(required - set(frame.columns))}")
        if len(frame) != 196 or frame["row_index"].duplicated().any():
            raise RuntimeError(f"{representation}: expected 196 unique OOF subjects")
        if reference is None:
            reference = frame
        elif not np.array_equal(frame["row_index"], reference["row_index"]):
            raise RuntimeError(f"{representation}: subject order differs")
        elif not np.allclose(frame["true_age"], reference["true_age"], atol=0, rtol=0):
            raise RuntimeError(f"{representation}: true ages differ")
        predictions[representation] = frame

    assert reference is not None
    rng = np.random.default_rng(args.seed)
    shared_indices = rng.integers(
        0, len(reference), size=(args.bootstrap_replicates, len(reference))
    )
    rows: list[dict[str, object]] = []
    for comparison_order, item in enumerate(COMPARISONS):
        a = predictions[item["representation_a"]]["absolute_error"].to_numpy(dtype=float)
        b = predictions[item["representation_b"]]["absolute_error"].to_numpy(dtype=float)
        paired_difference = a - b
        bootstrap = paired_difference[shared_indices].mean(axis=1)
        lower, upper = np.percentile(bootstrap, [2.5, 97.5])
        p_lower = (np.sum(bootstrap <= 0) + 1) / (args.bootstrap_replicates + 1)
        p_upper = (np.sum(bootstrap >= 0) + 1) / (args.bootstrap_replicates + 1)
        rows.append(
            {
                **item,
                "comparison_order": comparison_order,
                "n": len(paired_difference),
                "delta_mae_a_minus_b": float(paired_difference.mean()),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "bootstrap_two_sided_p": float(min(1.0, 2 * min(p_lower, p_upper))),
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": args.seed,
                "bootstrap_indices_shared_across_comparisons": True,
                "delta_interpretation": "positive values favor representation_b (lower MAE)",
            }
        )

    result = pd.DataFrame(rows)
    result["fdr_bh_q"] = np.nan
    result["fdr_bh_significant_0_05"] = pd.array([pd.NA] * len(result), dtype="boolean")
    result["fdr_status"] = "included in exploratory/secondary comparison family"
    for family in ("spatial_resolution", "ico4_ablation"):
        mask = result["family"].eq(family)
        q_values = fdr_bh(result.loc[mask, "bootstrap_two_sided_p"].to_numpy())
        result.loc[mask, "fdr_bh_q"] = q_values
        result.loc[mask, "fdr_bh_significant_0_05"] = q_values < 0.05
    result = result.sort_values("comparison_order").reset_index(drop=True)

    csv_path = args.output_dir / "Figure5_paired_MAE_comparisons.csv"
    result.to_csv(csv_path, index=False)
    report = {
        "status": "complete",
        "n_subjects": len(reference),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "bootstrap_indices_shared_across_all_comparisons": True,
        "fdr_method": "Benjamini-Hochberg",
        "fdr_families": {
            "spatial_resolution": 3,
            "ico4_ablation": 4,
        },
        "note": (
            "These comparison families were organized after inspection of the results and must be "
            "described as exploratory/secondary, not as preregistered or prespecified."
        ),
        "output_csv": str(csv_path),
    }
    (args.output_dir / "Figure5_paired_MAE_comparisons_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    print(result[["comparison", "delta_mae_a_minus_b", "ci95_lower", "ci95_upper", "bootstrap_two_sided_p", "fdr_bh_q"]].to_string(index=False))


if __name__ == "__main__":
    main()
