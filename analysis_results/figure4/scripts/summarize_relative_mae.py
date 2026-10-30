#!/usr/bin/env python3
"""Summarize and plot the Figure 4A symmetric relative MAE curve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_SEED = 20261022
N_BOOTSTRAP = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject_ids = pd.read_csv(args.manifest, dtype={"subject_id": str})[
        "subject_id"
    ].drop_duplicates().tolist()
    tables = []
    missing = []
    failed = []
    for subject_id in subject_ids:
        subject_dir = args.results_root / "subjects" / subject_id
        path = subject_dir / "figure4_relative_mae_subject_bins.csv"
        if path.is_file():
            tables.append(pd.read_csv(path, dtype={"subject_id": str}))
        elif (subject_dir / "failure.json").is_file():
            failed.append(subject_id)
        else:
            missing.append(subject_id)
    if not tables:
        raise RuntimeError("No subject-level relative-MAE tables found")
    subject = pd.concat(tables, ignore_index=True)
    duplicated = subject.duplicated(["subject_id", "distance_bin_index"])
    if duplicated.any():
        raise ValueError("Duplicate subject x distance-bin rows")
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for (bin_index, distance_bin), frame in subject.groupby(
        ["distance_bin_index", "distance_bin"]
    ):
        values = frame["symmetric_relative_mae_pct"].to_numpy(float)
        indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
        boot = np.median(values[indices], axis=1)
        rows.append(
            {
                "distance_bin_index": bin_index,
                "distance_bin": distance_bin,
                "distance_bin_center_mm": 2 * bin_index + 1,
                "n_subjects": len(values),
                "median": np.median(values),
                "q25": np.quantile(values, .25),
                "q75": np.quantile(values, .75),
                "bootstrap_ci95_low": np.quantile(boot, .025),
                "bootstrap_ci95_high": np.quantile(boot, .975),
                "anchor_count_total": int(frame["anchor_count"].sum()),
                "pair_count_total": int(frame["pair_count"].sum()),
            }
        )
    group = pd.DataFrame(rows).sort_values("distance_bin_index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject.to_csv(args.output_dir / "figure4_relative_mae_subject_bins.csv", index=False)
    group.to_csv(args.output_dir / "figure4_relative_mae_group_summary.csv", index=False)
    status = {
        "status": "complete" if not missing and not failed else "incomplete",
        "subjects_expected": len(subject_ids),
        "subjects_completed": subject["subject_id"].nunique(),
        "subjects_missing": missing,
        "subjects_failed": failed,
        "definition": (
            "200 * profile_MAE / (median_abs_profile_anchor + "
            "median_abs_profile_neighbor)"
        ),
        "unit": "percent",
        "aggregation": "pair -> anchor x distance-bin median -> subject -> group",
    }
    (args.output_dir / "figure4_relative_mae_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )

    sys.path.insert(0, str(args.repo_root))
    from qc.publication_style import (  # pylint: disable=import-outside-toplevel
        FIGURE_WIDTH_IN,
        PALETTE,
        save_publication_figure,
        set_publication_style,
        style_axes,
    )
    set_publication_style()
    x = group["distance_bin_center_mm"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.25))
    ax.plot(x, group["median"], color=PALETTE["blue"], marker="o", markersize=3)
    ax.fill_between(
        x, group["q25"], group["q75"],
        color=PALETTE["blue"], alpha=.2, linewidth=0,
    )
    ax.set(
        xlabel="White-surface geodesic distance (mm)",
        ylabel="Symmetric relative profile MAE (%)",
        xlim=(0, 30),
        xticks=np.arange(0, 31, 5),
    )
    style_axes(ax, grid=True)
    fig.subplots_adjust(left=.12, right=.98, bottom=.19, top=.98)
    save_publication_figure(fig, args.output_dir / "Figure4A_relative_profile_mae")
    plt.close(fig)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
