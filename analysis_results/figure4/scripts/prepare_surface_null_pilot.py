#!/usr/bin/env python3
"""Select deterministic registration-cost quantiles for a Figure 4 null pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registration-table", type=Path, required=True)
    parser.add_argument("--number", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    costs = pd.read_csv(args.registration_table, dtype={"subject_id": str})[
        ["subject_id", "bbregister_mincost"]
    ]
    eligible = (
        manifest.loc[manifest.input_complete]
        .merge(costs, on="subject_id", validate="one_to_one")
        .dropna(subset=["bbregister_mincost"])
        .sort_values(["bbregister_mincost", "subject_id"])
        .reset_index(drop=True)
    )
    indices = np.linspace(0, len(eligible) - 1, args.number).round().astype(int)
    pilot = eligible.iloc[indices].copy()
    pilot.insert(1, "pilot_selection_rank", np.arange(args.number))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pilot.to_csv(args.output, index=False)
    print(pilot[["subject_id", "bbregister_mincost"]].to_string(index=False))


if __name__ == "__main__":
    main()
