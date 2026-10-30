#!/usr/bin/env python3
"""Combine Figure 5 model metrics and paired OOF absolute errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPRESENTATIONS = [
    "regional_mean_md",
    "regional_thickness",
    "regional_profile",
    "ico3_md",
    "ico4_md",
    "ico4_column_mean_md",
    "ico5_md",
    "ico3_md_length",
    "ico3_md_curvature",
    "ico3_md_length_curvature",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    paired: pd.DataFrame | None = None
    for representation in REPRESENTATIONS:
        model_dir = args.model_root / representation
        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        metric_rows.append(
            {
                "representation": representation,
                "n": metrics["n_evaluated"],
                "mae": metrics["overall_oof_mae"],
                "pearson_r": metrics["overall_oof_pearson_r"],
                "elapsed_seconds": metrics["elapsed_seconds"],
                "peak_rss_kb": metrics["peak_rss_kb"],
            }
        )
        prediction = pd.read_csv(model_dir / f"{representation}_oof_predictions.csv")
        component = prediction[["row_index", "true_age", "absolute_error"]].rename(
            columns={"absolute_error": representation}
        )
        paired = component if paired is None else paired.merge(
            component.drop(columns="true_age"), on="row_index", how="inner", validate="one_to_one"
        )
    pd.DataFrame(metric_rows).sort_values("mae").to_csv(args.output_dir / "model_summary.csv", index=False)
    assert paired is not None
    paired.to_csv(args.output_dir / "paired_absolute_errors.csv", index=False)
    print(json.dumps({"status": "complete", "n_representations": len(metric_rows), "n_paired": len(paired)}))


if __name__ == "__main__":
    main()
