#!/usr/bin/env python3
"""Summarize the full-cohort formal ico5 within-DK K=15 real curves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected = pd.read_csv(args.manifest, dtype={"subject_id": str})["subject_id"].tolist()
    tables = []
    missing = []
    for subject_id in expected:
        path = args.results_root / "subjects" / subject_id / "figure4_ico5_k15_real_subject_bins.csv"
        if path.is_file():
            tables.append(pd.read_csv(path))
        else:
            missing.append(subject_id)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} formal real subjects: {missing[:10]}")
    subject = pd.concat(tables, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject.to_csv(args.output_dir / "figure4A_ico5_k15_real_subject_bins.csv", index=False)
    rows = []
    for keys, table in subject.groupby(
        ["distance_bin_index", "distance_bin", "pair_type"], observed=True
    ):
        bin_index, bin_label, pair_type = keys
        row = {
            "distance_bin_index": bin_index, "distance_bin": bin_label,
            "pair_type": pair_type, "n_subjects": table.subject_id.nunique(),
            "anchor_count_total": int(table.n_anchors.sum()),
            "pair_count_total": int(table.n_pairs.sum()),
        }
        for source, label in (
            ("median_dissimilarity", "profile_dissimilarity"),
            ("median_profile_mae", "profile_mae"),
            ("median_profile_rmse", "profile_rmse"),
            ("median_geodesic_distance_mm", "geodesic_distance_mm"),
        ):
            row[f"{label}_median"] = table[source].median()
            row[f"{label}_q25"] = table[source].quantile(.25)
            row[f"{label}_q75"] = table[source].quantile(.75)
        rows.append(row)
    group = pd.DataFrame(rows).sort_values("distance_bin_index")
    group.to_csv(args.output_dir / "figure4A_ico5_k15_real_group_summary.csv", index=False)

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from qc.publication_style import (
        FIGURE_WIDTH_IN, PALETTE, save_publication_figure, set_publication_style,
        style_axes,
    )
    set_publication_style()
    x = group.distance_bin_index.to_numpy() * 2 + 1
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.35), constrained_layout=True)
    ax.plot(x, group.profile_dissimilarity_median, color=PALETTE["blue"], linewidth=1.5)
    ax.fill_between(
        x, group.profile_dissimilarity_q25, group.profile_dissimilarity_q75,
        color=PALETTE["blue"], alpha=.2, linewidth=0,
    )
    ax.set(
        xlabel="White-surface geodesic distance (mm)",
        ylabel="Profile-shape dissimilarity (1 − r)",
    )
    style_axes(ax)
    save_publication_figure(fig, args.output_dir / "Figure4A_ico5_within_DK_K15")
    plt.close(fig)
    (args.output_dir / "figure4A_ico5_k15_status.json").write_text(json.dumps({
        "status": "full_cohort_complete", "n_subjects": len(expected),
        "anchor_definition": "official_fsaverage5_fixed",
        "candidate_definition": "within_DK_region", "k": 15,
        "aggregation": "pair -> anchor x bin median -> bilateral subject median -> group median/IQR",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
