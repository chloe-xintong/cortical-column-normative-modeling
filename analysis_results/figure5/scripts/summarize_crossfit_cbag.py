#!/usr/bin/env python3
"""Combine leakage-resistant cross-fitted cBAG outputs across representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_ridge import REPRESENTATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cbag-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    wide: pd.DataFrame | None = None
    for representation in REPRESENTATIONS:
        model_dir = args.cbag_root / representation
        metrics = json.loads((model_dir / "crossfit_cbag_metrics.json").read_text())
        rows.append(metrics)
        values = pd.read_csv(model_dir / f"{representation}_crossfit_cbag.csv")
        component = values[["row_index", "true_age", "outer_fold", "crossfit_cbag"]].rename(
            columns={"crossfit_cbag": representation}
        )
        wide = component if wide is None else wide.merge(
            component.drop(columns=["true_age", "outer_fold"]),
            on="row_index",
            how="inner",
            validate="one_to_one",
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "lsqr_crossfit_cbag_summary.csv", index=False)
    assert wide is not None
    wide.to_csv(args.output_dir / "lsqr_crossfit_cbag_by_subject.csv", index=False)
    (args.output_dir / "CBAG_METHOD_NOTE.txt").write_text(
        "Use lsqr_crossfit_cbag_summary.csv and lsqr_crossfit_cbag_by_subject.csv "
        "for Figure 5 and downstream analyses. These use 5-fold cross-fitted predictions "
        "within each outer-training set to estimate age-bias correction before applying it "
        "to the held-out outer-test fold. The earlier lsqr_cbag_summary.csv and "
        "lsqr_foldwise_cbag_by_subject.csv used in-sample outer-training predictions and are "
        "retained only as an overfitting diagnostic; do not use them for inference.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "n_representations": len(rows), "n": len(wide)}))


if __name__ == "__main__":
    main()
