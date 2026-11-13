#!/usr/bin/env python3
"""Compare Figure 5 cohort composition under explicit visit-match rules.

Description
-----------
The current 196-subject primary cohort is not changed.  Every Figure 5 rule
below requires the existing manual registration QC pass, complete Option 3
profiles, strict matched-visit CN status, and at least 95% bilateral profile
coverage.  Only the visit-match rule varies.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--cohort-manifest`` command-line input
- ``--visit-audit`` command-line input
- ``--figure4-active-manifest`` command-line input
- ``--profile-root`` command-line input
- ``--current-final-subjects`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure5/cohort_selection/build_cohort_rule_statistics.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HIGH = "HIGH (<=30d)"
MEDIUM = "MEDIUM (31-90d)"
LOW = "LOW (>90d)"
AMBIGUOUS = "REVIEW (ambiguous)"
LONG_GAP_DAYS = 365
AGE_BAND_EDGES = [20, 30, 40, 50, 60, 70, 80, 90]
AGE_BAND_LABELS = [
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70-79",
    "80-89",
]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    repo = repository_root()
    project = repo.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "cohort_manifest.csv",
    )
    parser.add_argument(
        "--visit-audit",
        type=Path,
        default=repo
        / "analysis_results"
        / "cohort_audit"
        / "outputs"
        / "date_match_visit_audit_325.csv",
    )
    parser.add_argument(
        "--figure4-active-manifest",
        type=Path,
        default=repo / "manifests" / "figure4_registration_qc_pass_320_subjects.csv",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=project / "results" / "fsaverage_result" / "columns",
    )
    parser.add_argument(
        "--current-final-subjects",
        type=Path,
        default=project
        / "results"
        / "figure5_age_prediction"
        / "outputs"
        / "features"
        / "subject_order.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "rule_statistics",
    )
    return parser.parse_args()


def read_ids(path: Path) -> set[str]:
    table = pd.read_csv(path, dtype={"subject_id": str})
    if "subject_id" not in table:
        raise ValueError(f"Missing subject_id in {path}")
    if table["subject_id"].duplicated().any():
        raise ValueError(f"Duplicate subject IDs in {path}")
    return set(table["subject_id"])


def bilateral_profile_coverage(profile_root: Path, subject_id: str) -> float:
    n_mask = 0
    n_valid = 0
    for hemisphere in ("lh", "rh"):
        path = (
            profile_root
            / subject_id
            / "profiles"
            / f"{subject_id}_{hemisphere}_md_profiles.npz"
        )
        with np.load(path, allow_pickle=False) as profile:
            mask = profile["in_analysis_mask"].astype(bool)
            valid = profile["column_valid"].astype(bool)
        n_mask += int(mask.sum())
        n_valid += int((mask & valid).sum())
    return n_valid / n_mask


def count_and_percent(mask: pd.Series, denominator: int) -> tuple[int, float]:
    count = int(mask.sum())
    percent = 100.0 * count / denominator if denominator else np.nan
    return count, percent


def summarize_rule(rule_id: str, label: str, table: pd.DataFrame) -> dict[str, object]:
    n = len(table)
    age = pd.to_numeric(table["age_at_scan"], errors="coerce").dropna()
    row: dict[str, object] = {
        "rule_id": rule_id,
        "rule_label": label,
        "n_subjects": n,
        "age_n": len(age),
        "age_mean": age.mean(),
        "age_sd": age.std(ddof=1),
        "age_median": age.median(),
        "age_q25": age.quantile(0.25),
        "age_q75": age.quantile(0.75),
        "age_min": age.min(),
        "age_max": age.max(),
    }
    for code in (1, 2, 9):
        count, percent = count_and_percent(table["SUBJECT_SEX"].eq(code), n)
        row[f"sex_code_{code}_n"] = count
        row[f"sex_code_{code}_pct"] = percent
    missing_sex, missing_sex_pct = count_and_percent(table["SUBJECT_SEX"].isna(), n)
    row["sex_missing_n"] = missing_sex
    row["sex_missing_pct"] = missing_sex_pct

    confidence_columns = {
        HIGH: "qial_high",
        MEDIUM: "qial_medium",
        LOW: "qial_low",
        AMBIGUOUS: "qial_ambiguous",
    }
    for level, prefix in confidence_columns.items():
        count, percent = count_and_percent(table["QIAL_match_confidence"].eq(level), n)
        row[f"{prefix}_n"] = count
        row[f"{prefix}_pct"] = percent

    row["single_distinct_visit_date_n"] = int(
        table["n_unique_visit_dates"].eq(1).sum()
    )
    row["multiple_distinct_visit_dates_n"] = int(
        table["n_unique_visit_dates"].ge(2).sum()
    )
    row["same_date_diagnosis_conflict_n"] = int(
        table["same_date_diagnosis_conflict"].sum()
    )
    row["long_gap_distinct_date_diagnosis_discordance_n"] = int(
        table["long_gap_diagnosis_discordance"].sum()
    )
    row["coverage_mean"] = table["recomputed_bilateral_coverage_fraction"].mean()
    row["coverage_min"] = table["recomputed_bilateral_coverage_fraction"].min()
    return row


def grouped_counts(
    rule_id: str,
    label: str,
    table: pd.DataFrame,
    variable: str,
    display_variable: str,
) -> list[dict[str, object]]:
    # Build an ordinary string series so ordered categoricals (used for age
    # bands) can share the same explicit missing label as other variables.
    values = pd.Series(
        ["missing" if pd.isna(value) else str(value) for value in table[variable]],
        index=table.index,
    )
    counts = values.value_counts(dropna=False, sort=False)
    n = len(table)
    return [
        {
            "rule_id": rule_id,
            "rule_label": label,
            "variable": display_variable,
            "level": level,
            "n": int(count),
            "pct": 100.0 * int(count) / n if n else np.nan,
        }
        for level, count in counts.items()
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.cohort_manifest, dtype={"subject_id": str})
    visit = pd.read_csv(args.visit_audit, dtype={"subject_id": str})
    active_ids = read_ids(args.figure4_active_manifest)
    final_ids = read_ids(args.current_final_subjects)

    required_cohort = {
        "subject_id",
        "age_at_scan",
        "SUBJECT_SEX",
        "QIAL_match_confidence",
        "qial_imaging_match",
        "option3_qc_pass",
        "age_diagnosis_complete",
        "cn_at_scan",
        "date_match_primary_or_approved_low",
        "diagnosis_at_scan",
    }
    required_visit = {
        "subject_id",
        "n_unique_visit_dates",
        "visit_date_span_days",
        "same_date_diagnosis_conflict",
        "diagnosis_differs_across_distinct_dates",
        "all_metadata_rows_strict_cn",
    }
    if missing := sorted(required_cohort - set(cohort.columns)):
        raise ValueError(f"Cohort manifest missing columns: {missing}")
    if missing := sorted(required_visit - set(visit.columns)):
        raise ValueError(f"Visit audit missing columns: {missing}")

    table = cohort.merge(
        visit[list(required_visit)], on="subject_id", how="left", validate="one_to_one"
    )
    table = table.loc[table["subject_id"].isin(active_ids)].copy()
    if len(table) != 320:
        raise RuntimeError(f"Expected 320 registration-QC-pass subjects, found {len(table)}")

    coverage_cache_path = args.output_dir / "registration_qc_pass_coverage_320.csv"
    if coverage_cache_path.is_file():
        coverage_cache = pd.read_csv(
            coverage_cache_path, dtype={"subject_id": str}
        )
        table = table.merge(
            coverage_cache,
            on="subject_id",
            how="left",
            validate="one_to_one",
        )
        if table["recomputed_bilateral_coverage_fraction"].isna().any():
            raise RuntimeError("Coverage cache does not contain all 320 subjects")
    else:
        table["recomputed_bilateral_coverage_fraction"] = [
            bilateral_profile_coverage(args.profile_root, subject_id)
            for subject_id in table["subject_id"]
        ]
        table[
            ["subject_id", "recomputed_bilateral_coverage_fraction"]
        ].to_csv(coverage_cache_path, index=False)
    table["coverage_ge_0_95"] = table[
        "recomputed_bilateral_coverage_fraction"
    ].ge(0.95)
    table["long_gap_diagnosis_discordance"] = (
        table["n_unique_visit_dates"].ge(2)
        & table["visit_date_span_days"].ge(LONG_GAP_DAYS)
        & table["diagnosis_differs_across_distinct_dates"]
    )

    common_figure5 = (
        table["qial_imaging_match"]
        & table["option3_qc_pass"]
        & table["age_diagnosis_complete"]
        & table["cn_at_scan"]
        & table["coverage_ge_0_95"]
    )
    current = common_figure5 & table["date_match_primary_or_approved_low"]
    high_or_medium = table["QIAL_match_confidence"].isin([HIGH, MEDIUM])
    low_or_ambiguous = table["QIAL_match_confidence"].isin([LOW, AMBIGUOUS])
    low_ambiguous_visit_rule = common_figure5 & (
        high_or_medium
        | (low_or_ambiguous & ~table["long_gap_diagnosis_discordance"])
    )
    global_visit_consistency = (
        common_figure5 & ~table["long_gap_diagnosis_discordance"]
    )
    all_qial_cn = common_figure5
    common_figure5_before_coverage = (
        table["qial_imaging_match"]
        & table["option3_qc_pass"]
        & table["age_diagnosis_complete"]
        & table["cn_at_scan"]
    )
    current_before_coverage = (
        common_figure5_before_coverage
        & table["date_match_primary_or_approved_low"]
    )
    current_plus_eligible_ambiguous = current | (
        all_qial_cn & table["QIAL_match_confidence"].eq(AMBIGUOUS)
    )

    table["in_figure4_qc_pass_reference"] = True
    table["in_figure5_current_primary"] = current
    table["in_figure5_low_ambiguous_visit_rule_365d"] = low_ambiguous_visit_rule
    table["in_figure5_global_visit_consistency_365d"] = global_visit_consistency
    table["in_figure5_all_qial_cn_sensitivity"] = all_qial_cn
    table["in_saved_figure5_final_196"] = table["subject_id"].isin(final_ids)
    if set(table.loc[current, "subject_id"]) != final_ids:
        raise RuntimeError("Reconstructed current primary rule does not equal saved 196")

    expansion_rows = [
        {
            "section": "data availability",
            "display_order": 0,
            "pool_id": "all_metadata_subjects",
            "pool_label": "All metadata subjects",
            "n_subjects": len(cohort),
            "possible_addition_vs_current_196": np.nan,
            "current_interpretation": "Theoretical metadata universe, not an imaging-analysis cohort.",
            "possible_future_action": "Obtain or link imaging for the 79 subjects without a current QIAL imaging match.",
        },
        {
            "section": "data availability",
            "display_order": 1,
            "pool_id": "qial_imaging_matched",
            "pool_label": "Current QIAL imaging match",
            "n_subjects": int(cohort["qial_imaging_match"].sum()),
            "possible_addition_vs_current_196": np.nan,
            "current_interpretation": "Imaging-linked pool before requiring complete Option 3 DWI profiles.",
            "possible_future_action": "Complete or recover Option 3 profiles for the 44 imaging-linked subjects outside the 325-profile cohort.",
        },
        {
            "section": "data availability",
            "display_order": 2,
            "pool_id": "complete_option3_profiles",
            "pool_label": "Complete Option 3 DWI profiles",
            "n_subjects": int(
                (cohort["qial_imaging_match"] & cohort["option3_qc_pass"]).sum()
            ),
            "possible_addition_vs_current_196": np.nan,
            "current_interpretation": "Current complete DWI-profile pool before manual registration QC.",
            "possible_future_action": "Review or reprocess the five manual registration-QC failures; they are not currently analysis-eligible.",
        },
        {
            "section": "data availability",
            "display_order": 3,
            "pool_id": "registration_qc_pass_reference",
            "pool_label": "QC-passed complete-DWI reference",
            "n_subjects": len(table),
            "possible_addition_vs_current_196": np.nan,
            "current_interpretation": "Recommended meaning of all currently analyzable DWI subjects.",
            "possible_future_action": "Apply diagnosis, visit-match, and coverage rules according to the analysis question.",
        },
        {
            "section": "review/recalculation scenario",
            "display_order": 4,
            "pool_id": "current_primary",
            "pool_label": "Current final primary cohort",
            "n_subjects": int(current.sum()),
            "possible_addition_vs_current_196": 0,
            "current_interpretation": "Saved Figure 5 models and results use this cohort.",
            "possible_future_action": "No action; this is the frozen current analysis set.",
        },
        {
            "section": "review/recalculation scenario",
            "display_order": 5,
            "pool_id": "current_before_coverage",
            "pool_label": "Current rule before 95% coverage QC",
            "n_subjects": int(current_before_coverage.sum()),
            "possible_addition_vs_current_196": int(
                current_before_coverage.sum() - current.sum()
            ),
            "current_interpretation": "Four otherwise eligible subjects fail only the current coverage threshold.",
            "possible_future_action": "Recompute or recover DWI profiles for D0072, D0094, D0191, and D0222.",
        },
        {
            "section": "review/recalculation scenario",
            "display_order": 6,
            "pool_id": "current_plus_eligible_ambiguous",
            "pool_label": "Current cohort plus eligible ambiguous matches",
            "n_subjects": int(current_plus_eligible_ambiguous.sum()),
            "possible_addition_vs_current_196": int(
                current_plus_eligible_ambiguous.sum() - current.sum()
            ),
            "current_interpretation": "Adds ambiguous subjects who already pass strict-CN and coverage requirements.",
            "possible_future_action": "Confirm 21 eligible source-ambiguous subjects. They are the remainder of 28 after two registration-QC exclusions, four same-date diagnosis conflicts, and one MCI exclusion.",
        },
        {
            "section": "review/recalculation scenario",
            "display_order": 7,
            "pool_id": "targeted_low_ambiguous_review",
            "pool_label": "Targeted LOW/ambiguous review scenario",
            "n_subjects": int(low_ambiguous_visit_rule.sum()),
            "possible_addition_vs_current_196": int(
                low_ambiguous_visit_rule.sum() - current.sum()
            ),
            "current_interpretation": "Adds 21 eligible ambiguous and seven eligible LOW subjects under the remembered longitudinal rule.",
            "possible_future_action": "Confirm the 21 eligible ambiguous matches and adjudicate seven single-date LOW cases.",
        },
        {
            "section": "review/recalculation scenario",
            "display_order": 8,
            "pool_id": "all_qial_cn_coverage_pass",
            "pool_label": "All-QIAL strict-CN coverage-pass ceiling",
            "n_subjects": int(all_qial_cn.sum()),
            "possible_addition_vs_current_196": int(
                all_qial_cn.sum() - current.sum()
            ),
            "current_interpretation": "Largest currently computable Figure 5 CN cohort without a QIAL-confidence exclusion.",
            "possible_future_action": "Requires accepting 21 eligible ambiguous plus 14 LOW subjects: seven single-date LOW and seven long-gap diagnosis-discordant LOW.",
        },
    ]

    expected_expansion_counts = [448, 369, 325, 320, 196, 200, 217, 224, 231]
    observed_expansion_counts = [row["n_subjects"] for row in expansion_rows]
    if observed_expansion_counts != expected_expansion_counts:
        raise RuntimeError(
            "Unexpected availability/review counts: "
            f"{observed_expansion_counts} != {expected_expansion_counts}"
        )

    rules = [
        (
            "figure4_qc_pass_reference",
            "Figure 4 registration-QC-pass reference",
            table["in_figure4_qc_pass_reference"],
            False,
            False,
            "Complete Option 3 profiles and manual registration QC pass; no date, diagnosis, or 95% coverage restriction.",
        ),
        (
            "figure5_current_primary",
            "Figure 5 current primary",
            table["in_figure5_current_primary"],
            True,
            True,
            "HIGH/MEDIUM, or LOW with >=2 distinct dates and every visit strict CN; ambiguous excluded.",
        ),
        (
            "figure5_low_ambiguous_visit_rule_365d",
            "Figure 5 LOW/ambiguous visit-rule sensitivity",
            table["in_figure5_low_ambiguous_visit_rule_365d"],
            True,
            True,
            "Keep HIGH/MEDIUM. For LOW/ambiguous, exclude only when >=2 distinct dates span >=365 days and diagnosis differs across dates; same-date duplicates are not longitudinal visits.",
        ),
        (
            "figure5_global_visit_consistency_365d",
            "Figure 5 global visit-consistency sensitivity",
            table["in_figure5_global_visit_consistency_365d"],
            True,
            True,
            "Apply the >=2 dates, >=365-day span, and diagnosis-discordance exclusion to every QIAL confidence level, including HIGH/MEDIUM.",
        ),
        (
            "figure5_all_qial_cn_sensitivity",
            "Figure 5 all-QIAL CN sensitivity",
            table["in_figure5_all_qial_cn_sensitivity"],
            True,
            True,
            "No QIAL confidence restriction; retain every QIAL-matched strict-CN subject passing profile, registration, and coverage QC.",
        ),
    ]

    definition_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    age_band_rows: list[dict[str, object]] = []
    for order, (rule_id, label, mask, strict_cn_required, coverage_required, description) in enumerate(rules):
        subset = table.loc[mask].copy()
        definition_rows.append(
            {
                "display_order": order,
                "rule_id": rule_id,
                "rule_label": label,
                "n_subjects": len(subset),
                "manual_registration_qc_pass_required": True,
                "strict_cn_required": strict_cn_required,
                "coverage_ge_0_95_required": coverage_required,
                "visit_rule": description,
                "primary_or_sensitivity": (
                    "primary" if rule_id == "figure5_current_primary" else "reference/sensitivity"
                ),
            }
        )
        summary_rows.append(summarize_rule(rule_id, label, subset))
        subset["visit_count_group"] = np.where(
            subset["n_unique_visit_dates"].ge(2), "multiple distinct dates", "one distinct date"
        )
        subset["age_band"] = pd.cut(
            pd.to_numeric(subset["age_at_scan"], errors="coerce"),
            bins=AGE_BAND_EDGES,
            labels=AGE_BAND_LABELS,
            right=False,
            include_lowest=True,
        )
        subset["longitudinal_status"] = np.select(
            [
                subset["same_date_diagnosis_conflict"].to_numpy(dtype=bool),
                subset["long_gap_diagnosis_discordance"].to_numpy(dtype=bool),
                subset["diagnosis_differs_across_distinct_dates"].to_numpy(
                    dtype=bool
                ),
            ],
            [
                "same-date diagnosis conflict",
                "long-gap diagnosis discordance",
                "other distinct-date diagnosis discordance",
            ],
            default="no observed diagnosis discordance",
        )
        for variable, display_variable in [
            ("QIAL_match_confidence", "QIAL match confidence"),
            ("diagnosis_at_scan", "Matched-visit diagnosis"),
            ("SUBJECT_SEX", "SUBJECT_SEX code"),
            ("age_band", "Age band (years)"),
            ("visit_count_group", "Number of distinct visit dates"),
            ("longitudinal_status", "Longitudinal diagnosis audit"),
        ]:
            group_rows.extend(
                grouped_counts(rule_id, label, subset, variable, display_variable)
            )
        age_counts = subset["age_band"].value_counts(sort=False, dropna=False)
        for age_band in AGE_BAND_LABELS:
            count = int(age_counts.get(age_band, 0))
            age_band_rows.append(
                {
                    "rule_id": rule_id,
                    "rule_label": label,
                    "age_band": age_band,
                    "n": count,
                    "pct": 100.0 * count / len(subset) if len(subset) else np.nan,
                }
            )

    membership_columns = [
        "subject_id",
        "age_at_scan",
        "SUBJECT_SEX",
        "QIAL_match_confidence",
        "diagnosis_at_scan",
        "n_unique_visit_dates",
        "visit_date_span_days",
        "same_date_diagnosis_conflict",
        "diagnosis_differs_across_distinct_dates",
        "long_gap_diagnosis_discordance",
        "recomputed_bilateral_coverage_fraction",
        "coverage_ge_0_95",
        "in_figure4_qc_pass_reference",
        "in_figure5_current_primary",
        "in_figure5_low_ambiguous_visit_rule_365d",
        "in_figure5_global_visit_consistency_365d",
        "in_figure5_all_qial_cn_sensitivity",
        "in_saved_figure5_final_196",
    ]
    table[membership_columns].sort_values("subject_id", kind="stable").to_csv(
        args.output_dir / "cohort_rule_subject_membership.csv", index=False
    )
    pd.DataFrame(definition_rows).to_csv(
        args.output_dir / "cohort_rule_definitions.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "cohort_rule_demographic_summary.csv", index=False
    )
    pd.DataFrame(group_rows).to_csv(
        args.output_dir / "cohort_rule_group_counts.csv", index=False
    )
    pd.DataFrame(age_band_rows).to_csv(
        args.output_dir / "cohort_rule_age_band_counts.csv", index=False
    )
    pd.DataFrame(expansion_rows).to_csv(
        args.output_dir / "cohort_expansion_review_summary.csv", index=False
    )

    report = {
        "status": "complete",
        "long_gap_days_for_visit_consistency_sensitivity": LONG_GAP_DAYS,
        "rule_counts": {
            row["rule_id"]: row["n_subjects"] for row in definition_rows
        },
        "availability_and_review_counts": {
            row["pool_id"]: row["n_subjects"] for row in expansion_rows
        },
        "invariants": {
            "manual_registration_qc_pass_required_for_all_rules": True,
            "current_primary_matches_saved_subject_order_196": True,
            "figure5_rules_require_strict_cn": True,
            "figure5_rules_require_coverage_ge_0_95": True,
        },
        "note": (
            "Only the current-primary 196-subject rule is used by the saved Figure 5 models. "
            "The alternative sets are descriptive sensitivity cohorts; no models were rerun."
        ),
    }
    (args.output_dir / "cohort_rule_statistics_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
