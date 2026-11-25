#!/usr/bin/env python3
"""Build quantitative QC rankings for the 188 reviewed regional-pilot subjects.

Description
-----------
Build quantitative QC rankings for the 188 reviewed regional-pilot subjects.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--source-audit`` command-line input
- ``--subject-order`` command-line input
- ``--regional-profile`` command-line input
- ``--option3-subject-root`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/build_qc_ranking.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--subject-order", type=Path, required=True)
    parser.add_argument("--regional-profile", type=Path, required=True)
    parser.add_argument("--option3-subject-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    audit = pd.read_csv(args.source_audit, dtype={"subject_id": str})
    subjects = audit.loc[audit.included, "subject_id"].tolist()
    if len(subjects) != 188 or "D0271" not in subjects:
        raise RuntimeError("QC ranking input must be the original 188-person cohort including D0271")
    order = pd.read_csv(args.subject_order, dtype={"subject_id": str})
    rows = order.set_index("subject_id").loc[subjects, "row_index"].to_numpy(int)
    profile = np.asarray(np.load(args.regional_profile, mmap_mode="r")[rows], float)
    median = np.median(profile, axis=0)
    mad = np.median(np.abs(profile - median), axis=0)
    fallback = np.std(profile, axis=0, ddof=0)
    scale = np.where(mad > 0, 1.4826 * mad, fallback)
    robust_z = (profile - median) / scale

    output: list[dict[str, object]] = []
    missing: list[str] = []
    for i, subject_id in enumerate(subjects):
        subject_root = args.option3_subject_root / subject_id
        mapping_path = subject_root / "option3_mapping_qc_subject_hemi.csv"
        region_path = subject_root / "option3_label_qc_region.csv"
        if not mapping_path.exists() or not region_path.exists():
            missing.append(subject_id)
            continue
        mapping = pd.read_csv(mapping_path)
        region = pd.read_csv(region_path)
        denom = float(mapping.n_analysis_mask_vertices.sum())
        rdenom = float(region.n_vertices.sum())
        az = np.abs(robust_z[i])
        output.append({
            "subject_id": subject_id,
            "retained_in_final_187": subject_id != "D0271",
            "exclusion_reason": "source_image_structural_abnormality" if subject_id == "D0271" else "",
            "zero_length_column_proportion": float(mapping.zero_length_count_analysis_mask.sum() / denom),
            "noncortical_triangle_proportion": float(region.n_noncortical_triangle.sum() / rdenom),
            "valid_coverage": float(np.average(
                mapping.complete_valid_profile_pct_analysis_mask / 100.0,
                weights=mapping.n_analysis_mask_vertices,
            )),
            "regional_profile_robust_z_max_abs": float(az.max()),
            "regional_profile_robust_z_q99_abs": float(np.quantile(az, 0.99)),
            "regional_profile_robust_z_median_abs": float(np.median(az)),
            "regional_profile_robust_z_rms": float(np.sqrt(np.mean(robust_z[i] ** 2))),
            "regional_profile_robust_z_fraction_abs_gt_3": float(np.mean(az > 3)),
        })
    if missing:
        raise RuntimeError(f"Missing Option3 QC files for: {missing}")
    table = pd.DataFrame(output)
    anomaly_metrics = [
        "zero_length_column_proportion", "noncortical_triangle_proportion",
        "regional_profile_robust_z_max_abs", "regional_profile_robust_z_q99_abs",
        "regional_profile_robust_z_rms", "regional_profile_robust_z_fraction_abs_gt_3",
    ]
    for column in anomaly_metrics:
        table[f"rank_{column}"] = table[column].rank(method="min", ascending=False).astype(int)
    table["rank_low_valid_coverage"] = table.valid_coverage.rank(method="min", ascending=True).astype(int)
    rank_columns = [x for x in table if x.startswith("rank_")]
    table["composite_mean_anomaly_rank"] = table[rank_columns].mean(axis=1)
    table["composite_rank"] = table.composite_mean_anomaly_rank.rank(method="min", ascending=True).astype(int)
    table = table.sort_values(["composite_rank", "subject_id"]).reset_index(drop=True)
    table.to_csv(args.output_dir / "cohort_qc_ranking_188.csv", index=False)
    d0271 = table.loc[table.subject_id.eq("D0271")].iloc[0]
    profile_metrics = [
        "regional_profile_robust_z_max_abs", "regional_profile_robust_z_q99_abs",
        "regional_profile_robust_z_rms", "regional_profile_robust_z_fraction_abs_gt_3",
    ]
    relative = table[profile_metrics].div(d0271[profile_metrics].astype(float), axis=1)
    candidates = table.loc[table.subject_id.ne("D0271") & relative.max(axis=1).ge(0.5)].copy()
    candidates["max_profile_metric_fraction_of_D0271"] = relative.loc[candidates.index].max(axis=1)
    candidates = candidates.sort_values("max_profile_metric_fraction_of_D0271", ascending=False)
    candidates["manual_review_note"] = (
        "Ranking only; no automatic exclusion. Review source images if expert QC considers severity comparable to D0271."
    )
    candidates.to_csv(args.output_dir / "manual_source_image_qc_review_candidates.csv", index=False)
    pd.DataFrame([d0271]).to_csv(args.output_dir / "D0271_quantitative_qc_rank.csv", index=False)
    print({"n_ranked": len(table), "D0271_composite_rank": int(d0271.composite_rank),
           "candidate_rows_for_manual_review": len(candidates)})


if __name__ == "__main__":
    main()
