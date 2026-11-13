#!/usr/bin/env python3
"""Build descriptive statistics for the user-selected all-diagnosis pools.

Description
-----------
Build descriptive statistics for the user-selected all-diagnosis pools.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- No required command-line inputs; optional inputs are defined by the script.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure5/cohort_selection/build_selected_pool_statistics.py --help
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HIGH = "HIGH (<=30d)"
MEDIUM = "MEDIUM (31-90d)"
LOW = "LOW (>90d)"
AMBIGUOUS = "REVIEW (ambiguous)"
AGE_BANDS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89"]


def bilateral_coverage(profile_root: Path, subject_id: str) -> float:
    n_mask = 0
    n_valid = 0
    for hemisphere in ("lh", "rh"):
        path = profile_root / subject_id / "profiles" / f"{subject_id}_{hemisphere}_md_profiles.npz"
        with np.load(path, allow_pickle=False) as profile:
            mask = profile["in_analysis_mask"].astype(bool)
            valid = profile["column_valid"].astype(bool)
        n_mask += int(mask.sum())
        n_valid += int((mask & valid).sum())
    return n_valid / n_mask


def count_pct(mask: pd.Series, denominator: int) -> tuple[int, float]:
    count = int(mask.sum())
    return count, 100.0 * count / denominator if denominator else np.nan


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    project = repo.parents[1]
    selection_dir = Path(__file__).resolve().parent
    output_dir = selection_dir / "outputs" / "rule_statistics"
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(selection_dir / "outputs" / "cohort_manifest.csv", dtype={"subject_id": str})
    visit = pd.read_csv(
        repo / "/mnt/newStor/paros/paros_WORK/xintong/result/cortical_trajectory_analysis/analysis_results/cohort_audit/outputs/date_match_visit_audit_325.csv",
        dtype={"subject_id": str},
    )
    active = pd.read_csv(
        repo / "manifests/figure4_registration_qc_pass_320_subjects.csv",
        dtype={"subject_id": str},
    )
    active_ids = set(active["subject_id"])

    coverage_path = output_dir / "complete_option3_coverage_325.csv"
    option3_ids = cohort.loc[cohort["option3_qc_pass"], "subject_id"].tolist()
    if coverage_path.is_file():
        coverage = pd.read_csv(coverage_path, dtype={"subject_id": str})
    else:
        existing = pd.read_csv(
            output_dir / "registration_qc_pass_coverage_320.csv",
            dtype={"subject_id": str},
        )
        coverage_lookup = dict(
            zip(existing["subject_id"], existing["recomputed_bilateral_coverage_fraction"])
        )
        profile_root = project / "results/fsaverage_result/columns"
        for subject_id in option3_ids:
            if subject_id not in coverage_lookup:
                coverage_lookup[subject_id] = bilateral_coverage(profile_root, subject_id)
        coverage = pd.DataFrame(
            {
                "subject_id": option3_ids,
                "recomputed_bilateral_coverage_fraction": [
                    coverage_lookup[subject_id] for subject_id in option3_ids
                ],
            }
        )
        coverage.to_csv(coverage_path, index=False)
    if set(coverage["subject_id"]) != set(option3_ids):
        raise RuntimeError("The complete-profile coverage cache does not match the 325-person pool")

    visit_columns = [
        "subject_id",
        "n_unique_visit_dates",
        "visit_date_span_days",
        "same_date_diagnosis_conflict",
        "diagnosis_differs_across_distinct_dates",
    ]
    table = cohort.merge(visit[visit_columns], on="subject_id", how="left", validate="one_to_one")
    table = table.merge(coverage, on="subject_id", how="left", validate="one_to_one")
    table["coverage_ge_0_95"] = table["recomputed_bilateral_coverage_fraction"].ge(0.95)
    table["long_gap_diagnosis_discordance"] = (
        table["n_unique_visit_dates"].ge(2)
        & table["visit_date_span_days"].ge(365)
        & table["diagnosis_differs_across_distinct_dates"].eq(True)
    )
    table["age_band"] = pd.cut(
        pd.to_numeric(table["age_at_scan"], errors="coerce"),
        bins=[20, 30, 40, 50, 60, 70, 80, 90],
        labels=AGE_BANDS,
        right=False,
        include_lowest=True,
    )
    table["visit_count_group"] = np.select(
        [
            table["n_unique_visit_dates"].eq(1),
            table["n_unique_visit_dates"].ge(2),
        ],
        ["one distinct date", "multiple distinct dates"],
        default="missing visit audit",
    )
    table["longitudinal_status"] = np.select(
        [
            table["n_unique_visit_dates"].isna(),
            table["same_date_diagnosis_conflict"].eq(True),
            table["long_gap_diagnosis_discordance"],
            table["diagnosis_differs_across_distinct_dates"].eq(True),
        ],
        [
            "missing visit audit",
            "same-date diagnosis conflict",
            "long-gap diagnosis discordance",
            "other distinct-date diagnosis discordance",
        ],
        default="no observed diagnosis discordance",
    )
    table["coverage_status"] = np.select(
        [
            table["recomputed_bilateral_coverage_fraction"].isna(),
            table["coverage_ge_0_95"],
        ],
        ["missing profile coverage", ">=95%"],
        default="<95%",
    )

    table["has_diagnosis_label_conflict"] = (
        table["same_date_diagnosis_conflict"].eq(True)
        | table["diagnosis_differs_across_distinct_dates"].eq(True)
    )
    qc_pass = table["subject_id"].isin(active_ids)
    coverage_pass = table["coverage_ge_0_95"]
    no_observed_label_conflict = qc_pass & coverage_pass & ~table["has_diagnosis_label_conflict"]
    different_label_review = qc_pass & coverage_pass & table["has_diagnosis_label_conflict"]
    all_qial_coverage_pass = qc_pass & coverage_pass

    pools = [
        (
            "all_metadata_448",
            "All metadata subjects",
            pd.Series(True, index=table.index),
            "Complete metadata universe; imaging, QIAL confidence, diagnosis, visit audit, and profile coverage may be missing.",
        ),
        (
            "complete_dwi_325",
            "Complete Option 3 DWI pool",
            table["option3_qc_pass"],
            "Complete Option 3 DWI profiles before manual registration QC.",
        ),
        (
            "qc_pass_dwi_320",
            "QC-passed analyzable DWI reference",
            table["subject_id"].isin(active_ids),
            "All currently analyzable complete-DWI subjects before diagnosis, match-confidence, and coverage filtering.",
        ),
        (
            "no_label_conflict_259",
            "No-observed-label-conflict candidate",
            no_observed_label_conflict,
            "All diagnoses and QIAL levels are allowed; retains single-date subjects and multiple-date subjects with the same diagnosis label, after registration and coverage QC.",
        ),
        (
            "different_label_review_54",
            "Different-label review subset",
            different_label_review,
            "All diagnoses and QIAL levels are allowed; contains 50 subjects with labels differing across distinct dates and four with conflicting labels on duplicate rows from the same date.",
        ),
        (
            "all_qial_all_labels_313",
            "All-QIAL all-label maximum candidate",
            all_qial_coverage_pass,
            "Largest currently computable registration-QC- and coverage-passing cohort; accepts every diagnosis label, QIAL level, and label-consistency status.",
        ),
    ]
    expected_counts = [448, 325, 320, 259, 54, 313]
    if [int(mask.sum()) for _, _, mask, _ in pools] != expected_counts:
        raise RuntimeError("Selected cohort-pool counts changed unexpectedly")

    definitions: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    age_rows: list[dict[str, object]] = []

    for order, (pool_id, label, mask, definition) in enumerate(pools):
        subset = table.loc[mask].copy()
        n = len(subset)
        definitions.append(
            {
                "display_order": order,
                "pool_id": pool_id,
                "pool_label": label,
                "n_subjects": n,
                "definition": definition,
            }
        )
        age = pd.to_numeric(subset["age_at_scan"], errors="coerce").dropna()
        cov = pd.to_numeric(
            subset["recomputed_bilateral_coverage_fraction"], errors="coerce"
        ).dropna()
        summary: dict[str, object] = {
            "pool_id": pool_id,
            "pool_label": label,
            "n_subjects": n,
            "age_n": len(age),
            "age_missing_n": n - len(age),
            "age_mean": age.mean(),
            "age_sd": age.std(ddof=1),
            "age_median": age.median(),
            "age_q25": age.quantile(0.25),
            "age_q75": age.quantile(0.75),
            "age_min": age.min(),
            "age_max": age.max(),
            "visit_audit_n": int(subset["n_unique_visit_dates"].notna().sum()),
            "visit_audit_missing_n": int(subset["n_unique_visit_dates"].isna().sum()),
            "coverage_n": len(cov),
            "coverage_missing_n": n - len(cov),
            "coverage_mean": cov.mean(),
            "coverage_min": cov.min(),
        }
        for code in (1, 2, 9):
            count, pct = count_pct(subset["SUBJECT_SEX"].eq(code), n)
            summary[f"sex_code_{code}_n"] = count
            summary[f"sex_code_{code}_pct"] = pct
        summary["sex_missing_n"] = int(subset["SUBJECT_SEX"].isna().sum())
        for confidence, key in [
            (HIGH, "qial_high"),
            (MEDIUM, "qial_medium"),
            (LOW, "qial_low"),
            (AMBIGUOUS, "qial_ambiguous"),
        ]:
            count, pct = count_pct(subset["QIAL_match_confidence"].eq(confidence), n)
            summary[f"{key}_n"] = count
            summary[f"{key}_pct"] = pct
        summary["qial_missing_n"] = int(subset["QIAL_match_confidence"].isna().sum())
        summaries.append(summary)

        variables = [
            ("QIAL_match_confidence", "QIAL match confidence"),
            ("diagnosis_at_scan", "Matched-visit diagnosis"),
            ("SUBJECT_SEX", "SUBJECT_SEX code"),
            ("visit_count_group", "Number of distinct visit dates"),
            ("longitudinal_status", "Longitudinal diagnosis audit"),
            ("coverage_status", "Profile coverage status"),
        ]
        for variable, variable_label in variables:
            values = pd.Series(
                ["missing" if pd.isna(value) else str(value) for value in subset[variable]],
                index=subset.index,
            )
            for level, count in values.value_counts(sort=False).items():
                group_rows.append(
                    {
                        "pool_id": pool_id,
                        "pool_label": label,
                        "variable": variable_label,
                        "level": level,
                        "n": int(count),
                        "pct": 100.0 * int(count) / n,
                    }
                )

        age_values = subset["age_band"].astype(object)
        for age_band in AGE_BANDS + ["missing"]:
            count = int(
                age_values.isna().sum()
                if age_band == "missing"
                else age_values.eq(age_band).sum()
            )
            age_rows.append(
                {
                    "pool_id": pool_id,
                    "pool_label": label,
                    "age_band": age_band,
                    "n": count,
                    "pct": 100.0 * count / n,
                }
            )

    pd.DataFrame(definitions).to_csv(output_dir / "selected_pool_definitions.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "selected_pool_summary.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output_dir / "selected_pool_group_counts.csv", index=False)
    pd.DataFrame(age_rows).to_csv(output_dir / "selected_pool_age_band_counts.csv", index=False)

    report = {
        "status": "complete",
        "selected_pool_counts": {
            row["pool_id"]: row["n_subjects"] for row in definitions
        },
        "current_196_and_cn_restriction_deliberately_omitted_from_display": True,
        "label_consistency_partition": {
            "qc_pass_before_coverage_no_observed_conflict": 266,
            "qc_pass_before_coverage_different_label": 54,
            "coverage_pass_no_observed_conflict": 259,
            "coverage_pass_different_label": 54,
            "coverage_pass_union_all_qial": 313,
        },
        "percent_denominator": "total N within each selected pool; missing is shown explicitly",
    }
    (output_dir / "selected_pool_statistics_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
