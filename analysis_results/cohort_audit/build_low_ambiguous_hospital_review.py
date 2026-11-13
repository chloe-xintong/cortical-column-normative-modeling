#!/usr/bin/env python3
"""Build LOW/ambiguous subject and visit lists for hospital data review.

Description
-----------
Build LOW/ambiguous subject and visit lists for hospital data review.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--metadata`` command-line input
- ``--visit-audit`` command-line input
- ``--exclusions`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/cohort_audit/build_low_ambiguous_hospital_review.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LOW = "LOW (>90d)"
AMBIGUOUS = "REVIEW (ambiguous)"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    repo = repository_root()
    project = repo.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=project / "data" / "ADRC_harmonized_metadata_QIAL_imaging_matches.csv",
    )
    parser.add_argument(
        "--visit-audit",
        type=Path,
        default=repo / "analysis_results" / "cohort_audit" / "outputs" / "date_match_visit_audit_325.csv",
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=repo / "analysis_results" / "cohort_audit" / "outputs" / "figure_subject_exclusions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "analysis_results" / "cohort_audit" / "outputs" / "hospital_review",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return "missing"
    text = str(value).strip()
    return text if text else "missing"


def visit_detail(group: pd.DataFrame) -> str:
    parts = []
    for row in group.sort_values(["visit_date_parsed", "VISITNUM"], kind="stable").itertuples():
        qial = "QIAL-selected" if row.QIAL_has_imaging_data == 1 else "not-selected"
        parts.append(
            f"{clean_text(row.VISIT_DATE)} [{clean_text(row.VISITNUM)}: "
            f"{clean_text(row.DX_Label_harmonized)}; {qial}]"
        )
    return " | ".join(parts)


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata, low_memory=False)
    audit = pd.read_csv(args.visit_audit, dtype={"subject_id": str})
    exclusions = pd.read_csv(args.exclusions, dtype={"subject_id": str})

    review = audit.loc[
        audit["QIAL_match_confidence"].isin([LOW, AMBIGUOUS])
    ].copy()
    review_ids = set(review["subject_id"])

    visits = metadata.loc[
        metadata["CONNECTOME_KEY_CLEAN"].isin(review_ids)
    ].copy()
    visits = visits.rename(columns={"CONNECTOME_KEY_CLEAN": "subject_id"})
    visits["visit_date_parsed"] = pd.to_datetime(visits["VISIT_DATE"], errors="coerce")

    same_day_counts = visits.groupby(
        ["subject_id", "VISIT_DATE"], dropna=False
    )["subject_id"].transform("size")
    visits["same_day_record_count"] = same_day_counts
    visits["same_day_multiple_records"] = same_day_counts.gt(1)

    confidence = review.set_index("subject_id")["QIAL_match_confidence"]
    visits["QIAL_match_confidence"] = visits["subject_id"].map(confidence)
    visits = visits.merge(
        review[
            [
                "subject_id",
                "same_date_diagnosis_conflict",
                "diagnosis_differs_across_distinct_dates",
            ]
        ],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    summary_rows: list[dict[str, object]] = []
    for subject_id, group in visits.groupby("subject_id", sort=True):
        audit_row = review.loc[review["subject_id"].eq(subject_id)].iloc[0]
        repeated = group.loc[group["same_day_multiple_records"]]
        repeated_dates = sorted(repeated["VISIT_DATE"].dropna().astype(str).unique())
        same_day_multiple = bool(repeated_dates)
        same_date_conflict = bool(audit_row["same_date_diagnosis_conflict"])
        cross_date_conflict = bool(
            audit_row["diagnosis_differs_across_distinct_dates"]
        )
        ambiguous = audit_row["QIAL_match_confidence"] == AMBIGUOUS

        issue_codes = []
        if same_day_multiple:
            issue_codes.append("same_day_multiple_records")
        if same_date_conflict:
            issue_codes.append("same_date_diagnosis_conflict")
        if cross_date_conflict:
            issue_codes.append("diagnosis_conflict_across_dates")
        if ambiguous and not same_day_multiple:
            issue_codes.append("source_ambiguous_without_same_day_duplicate")
        if not issue_codes:
            issue_codes.append("LOW_scan_to_visit_gap_over_90d_only")

        questions = []
        if same_day_multiple:
            questions.append("Confirm whether same-day rows are separate visits or duplicate records")
        if same_date_conflict:
            questions.append("Reconcile conflicting diagnoses recorded on the same date")
        if cross_date_conflict:
            questions.append("Confirm the diagnosis at each distinct visit date")
        if ambiguous:
            questions.append("Confirm which clinical visit should be linked to the MRI scan")
        else:
            questions.append("Confirm the MRI scan date and the intended linked clinical visit")

        summary_rows.append(
            {
                "subject_id": subject_id,
                "QIAL_match_confidence": audit_row["QIAL_match_confidence"],
                "hospital_review_issue": ";".join(issue_codes),
                "hospital_question": "; ".join(questions),
                "matched_visit_date": audit_row["matched_visit_date"],
                "diagnosis_at_matched_visit": audit_row["diagnosis_at_scan"],
                "n_metadata_rows": int(audit_row["n_metadata_rows"]),
                "n_unique_visit_dates": int(audit_row["n_unique_visit_dates"]),
                "same_day_multiple_records": same_day_multiple,
                "same_day_multiple_dates": "|".join(repeated_dates),
                "same_date_diagnosis_conflict": same_date_conflict,
                "diagnosis_conflict_across_dates": cross_date_conflict,
                "visit_date_span_days": int(audit_row["visit_date_span_days"]),
                "all_visit_details": visit_detail(group),
                "manual_registration_qc_excluded": bool(
                    audit_row["manual_registration_qc_excluded"]
                ),
                "in_active_figure4_320": bool(audit_row["in_active_figure4_320"]),
                "in_final_figure5_196": bool(audit_row["in_final_figure5_196"]),
            }
        )

    summary = pd.DataFrame(summary_rows)
    confidence_order = pd.Categorical(
        summary["QIAL_match_confidence"], categories=[AMBIGUOUS, LOW], ordered=True
    )
    summary = (
        summary.assign(_confidence_order=confidence_order)
        .sort_values(["_confidence_order", "subject_id"], kind="stable")
        .drop(columns="_confidence_order")
    )

    visit_columns = [
        "subject_id",
        "QIAL_match_confidence",
        "PTID",
        "VISITNUM",
        "VISIT_DATE",
        "VISIT_AGE",
        "DX_Label_harmonized",
        "NORMCOG_01",
        "DEMENTIA_01",
        "MCI_01",
        "AD_01",
        "QIAL_has_imaging_data",
        "same_day_record_count",
        "same_day_multiple_records",
        "same_date_diagnosis_conflict",
        "diagnosis_differs_across_distinct_dates",
    ]
    visit_output = visits[visit_columns].sort_values(
        ["QIAL_match_confidence", "subject_id", "VISIT_DATE", "VISITNUM"],
        kind="stable",
    )

    focused = summary.loc[
        summary["same_day_multiple_records"]
        | summary["same_date_diagnosis_conflict"]
        | summary["diagnosis_conflict_across_dates"]
        | summary["hospital_review_issue"].eq(
            "source_ambiguous_without_same_day_duplicate"
        )
    ].copy()

    previous_exclusions = exclusions.loc[
        exclusions["figure"].eq("Figure 5")
        & exclusions["reason_code"].eq("date_match_not_currently_approved"),
        [
            "subject_id",
            "reason_code",
            "reason",
            "figure5_metadata_reason_all_flags",
        ],
    ].copy()
    previously_excluded = previous_exclusions.merge(
        summary,
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    previous_order = pd.Categorical(
        previously_excluded["QIAL_match_confidence"],
        categories=[AMBIGUOUS, LOW],
        ordered=True,
    )
    previously_excluded = (
        previously_excluded.assign(_confidence_order=previous_order)
        .sort_values(["_confidence_order", "subject_id"], kind="stable")
        .drop(columns="_confidence_order")
    )
    previously_excluded_ids = set(previously_excluded["subject_id"])
    previously_excluded_visits = visit_output.loc[
        visit_output["subject_id"].isin(previously_excluded_ids)
    ].copy()

    if len(summary) != 75:
        raise RuntimeError(f"Expected 75 LOW/ambiguous subjects, found {len(summary)}")
    if int(summary["same_day_multiple_records"].sum()) != 27:
        raise RuntimeError("Expected 27 subjects with same-day multiple records")
    if int(summary["same_date_diagnosis_conflict"].sum()) != 4:
        raise RuntimeError("Expected four same-date diagnosis conflicts")
    if int(summary["diagnosis_conflict_across_dates"].sum()) != 13:
        raise RuntimeError("Expected 13 cross-date diagnosis conflicts")
    if len(focused) != 41:
        raise RuntimeError(f"Expected 41 focused-review subjects, found {len(focused)}")
    if len(previously_excluded) != 52:
        raise RuntimeError(
            f"Expected 52 prior date-rule exclusions, found {len(previously_excluded)}"
        )
    prior_counts = previously_excluded["QIAL_match_confidence"].value_counts()
    if prior_counts.to_dict() != {AMBIGUOUS: 26, LOW: 26}:
        raise RuntimeError(f"Unexpected prior exclusion composition: {prior_counts.to_dict()}")
    if int(previously_excluded["same_day_multiple_records"].sum()) != 26:
        raise RuntimeError("Expected 26 prior exclusions with same-day multiple records")
    if int(previously_excluded["same_date_diagnosis_conflict"].sum()) != 4:
        raise RuntimeError("Expected four prior exclusions with same-date conflicts")
    if int(previously_excluded["diagnosis_conflict_across_dates"].sum()) != 12:
        raise RuntimeError("Expected 12 prior exclusions with cross-date conflicts")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "low_ambiguous_all_75_subjects.csv", index=False)
    focused.to_csv(
        args.output_dir / "low_ambiguous_focused_hospital_review_41_subjects.csv",
        index=False,
    )
    visit_output.to_csv(
        args.output_dir / "low_ambiguous_visit_details_148_rows.csv", index=False
    )
    previously_excluded.to_csv(
        args.output_dir
        / "figure5_previously_excluded_low_ambiguous_52_subjects.csv",
        index=False,
    )
    previously_excluded_visits.to_csv(
        args.output_dir
        / "figure5_previously_excluded_low_ambiguous_visit_details.csv",
        index=False,
    )

    print(f"All LOW/ambiguous subjects: {len(summary)}")
    print(f"Focused hospital review: {len(focused)}")
    print(f"Same-day multiple records: {int(summary['same_day_multiple_records'].sum())}")
    print(f"Same-date diagnosis conflicts: {int(summary['same_date_diagnosis_conflict'].sum())}")
    print(
        "Diagnosis conflicts across dates: "
        f"{int(summary['diagnosis_conflict_across_dates'].sum())}"
    )
    print(f"Previously excluded by Figure 5 date rule: {len(previously_excluded)}")


if __name__ == "__main__":
    main()
