#!/usr/bin/env python3

"""Build the fixed Figure 5 metadata-eligible ADRC cohort.

This stage identifies the clinical visit selected by the QIAL imaging match,
defines age/diagnosis at that matched visit, and intersects the result with the
325-subject Option 3 QC manifest.  It does not yet apply representation
availability, coverage, registration, or surface-reconstruction exclusions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ACCEPTED_MATCH_CONFIDENCE = ("HIGH (<=30d)", "MEDIUM (31-90d)")
NORMAL_DIAGNOSIS = "Normal"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    repo = repository_root()
    project = repo.parents[1]
    parser = argparse.ArgumentParser(
        description="Create the Figure 5 metadata cohort and audit tables."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=project
        / "data"
        / "ADRC_harmonized_metadata_QIAL_imaging_matches.csv",
    )
    parser.add_argument(
        "--option3-manifest",
        type=Path,
        default=repo / "manifests" / "figure4_valid_325_subjects.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "analysis_results" / "figure5" / "cohort_selection" / "outputs",
    )
    parser.add_argument(
        "--registration-qc-exclusions",
        type=Path,
        default=repo
        / "analysis_results"
        / "figure5"
        / "cohort_selection"
        / "manual_registration_qc_exclusions.csv",
    )
    return parser.parse_args()


def require_columns(table: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def flag_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if not row["qial_imaging_match"]:
        reasons.append("no_QIAL_imaging_match")
    if not row["option3_qc_pass"]:
        reasons.append("not_in_Option3_325")
    if row["qial_imaging_match"] and not row[
        "date_match_primary_or_approved_low"
    ]:
        reasons.append("date_match_not_within_90d")
    if row["qial_imaging_match"] and not row["age_diagnosis_complete"]:
        reasons.append("missing_age_or_diagnosis")
    if row["qial_imaging_match"] and row["age_diagnosis_complete"] and not row["cn_at_scan"]:
        reasons.append("not_cognitively_normal_at_matched_visit")
    if row["manual_registration_qc_excluded"]:
        reasons.append("manual_registration_qc_failure")
    return ";".join(reasons)


def count_step(
    rows: list[dict[str, object]],
    step: int,
    name: str,
    mask: pd.Series,
    previous_count: int,
) -> int:
    count = int(mask.sum())
    rows.append(
        {
            "step": step,
            "criterion": name,
            "n_remaining": count,
            "n_excluded_at_step": previous_count - count,
        }
    )
    return count


def build_low_longitudinal_audit(
    metadata: pd.DataFrame,
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit all clinical visits for LOW-confidence Option 3 subjects."""
    low = cohort.loc[
        cohort["qial_imaging_match"]
        & cohort["option3_qc_pass"]
        & cohort["QIAL_match_confidence"].eq("LOW (>90d)")
    ].copy()

    visits = metadata.loc[
        metadata["CONNECTOME_KEY_CLEAN"].isin(low["subject_id"])
    ].copy()
    visits["visit_date_parsed"] = pd.to_datetime(
        visits["VISIT_DATE"], errors="coerce"
    )
    visits["clinical_visit_cn"] = (
        visits["DX_Label_harmonized"].eq(NORMAL_DIAGNOSIS)
        & visits["NORMCOG_01"].eq(1)
        & visits["DEMENTIA_01"].eq(0)
        & visits["MCI_01"].eq(0)
        & visits["AD_01"].eq(0)
    )
    visits = visits.sort_values(
        ["CONNECTOME_KEY_CLEAN", "visit_date_parsed", "VISITNUM"],
        kind="stable",
        na_position="last",
    )

    subject_rows: list[dict[str, object]] = []
    for subject_id, group in visits.groupby("CONNECTOME_KEY_CLEAN", sort=True):
        dates = (
            group["visit_date_parsed"]
            .dropna()
            .dt.strftime("%Y-%m-%d")
            .drop_duplicates()
            .tolist()
        )
        diagnoses = (
            group["DX_Label_harmonized"]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .tolist()
        )
        n_missing_diagnosis = int(group["DX_Label_harmonized"].isna().sum())
        all_visits_cn = bool(group["clinical_visit_cn"].all())
        if len(dates) <= 1:
            longitudinal_group = "single_clinical_visit"
        elif all_visits_cn:
            longitudinal_group = "multiple_visits_all_cn"
        else:
            longitudinal_group = "multiple_visits_not_all_cn"

        subject_rows.append(
            {
                "subject_id": subject_id,
                "n_metadata_rows": len(group),
                "n_unique_visit_dates": len(dates),
                "visit_dates": "|".join(dates),
                "n_unique_diagnoses": len(diagnoses),
                "diagnosis_sequence": "|".join(diagnoses),
                "n_missing_diagnosis_rows": n_missing_diagnosis,
                "diagnosis_consistent_across_visits": (
                    len(diagnoses) == 1 and n_missing_diagnosis == 0
                ),
                "all_visits_cn": all_visits_cn,
                "longitudinal_review_group": longitudinal_group,
            }
        )

    subject_audit = pd.DataFrame(subject_rows).merge(
        low[
            [
                "subject_id",
                "PTID",
                "visit_id",
                "matched_visit_date",
                "diagnosis_at_scan",
                "cn_at_scan",
                "manual_registration_qc_excluded",
            ]
        ],
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    subject_audit["further_inclusion_review_candidate"] = (
        subject_audit["cn_at_scan"]
        & subject_audit["longitudinal_review_group"].isin(
            ["single_clinical_visit", "multiple_visits_all_cn"]
        )
    )
    subject_audit["further_inclusion_review_reason"] = ""
    candidate = subject_audit["further_inclusion_review_candidate"]
    subject_audit.loc[candidate, "further_inclusion_review_reason"] = (
        subject_audit.loc[candidate, "longitudinal_review_group"]
    )
    subject_audit["approved_for_primary_by_longitudinal_rule"] = (
        subject_audit["cn_at_scan"]
        & subject_audit["longitudinal_review_group"].eq(
            "multiple_visits_all_cn"
        )
    )
    subject_audit["included_in_primary_after_registration_qc"] = (
        subject_audit["approved_for_primary_by_longitudinal_rule"]
        & ~subject_audit["manual_registration_qc_excluded"]
    )
    subject_audit = subject_audit.sort_values("subject_id", kind="stable")

    visits["subject_id"] = visits["CONNECTOME_KEY_CLEAN"]
    visits["is_qial_matched_visit"] = visits["QIAL_has_imaging_data"].eq(1)
    visit_details = visits[
        [
            "subject_id",
            "PTID",
            "PTID_clean",
            "VISITNUM",
            "VISIT_DATE",
            "VISIT_AGE",
            "DX_Label_harmonized",
            "NORMCOG_01",
            "DEMENTIA_01",
            "MCI_01",
            "AD_01",
            "clinical_visit_cn",
            "is_qial_matched_visit",
            "QIAL_match_confidence",
        ]
    ].copy()

    summary = (
        subject_audit.groupby(
            ["cn_at_scan", "longitudinal_review_group"], dropna=False
        )
        .size()
        .rename("n_subjects")
        .reset_index()
    )
    return subject_audit, visit_details, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata, low_memory=False)
    option3 = pd.read_csv(args.option3_manifest, dtype={"subject_id": str})
    registration_exclusions = pd.read_csv(
        args.registration_qc_exclusions, dtype={"subject_id": str}
    )

    metadata_columns = [
        "PTID",
        "PTID_clean",
        "CONNECTOME_KEY_CLEAN",
        "VISITNUM",
        "VISIT_DATE",
        "VISIT_AGE",
        "DX_Label_harmonized",
        "SUBJECT_SEX",
        "NORMCOG_01",
        "DEMENTIA_01",
        "MCI_01",
        "AD_01",
        "QIAL_has_imaging_data",
        "QIAL_match_confidence",
    ]
    require_columns(metadata, metadata_columns, "metadata")
    require_columns(option3, ["subject_id"], "Option 3 manifest")
    require_columns(
        registration_exclusions,
        ["subject_id", "qc_status", "qc_reason"],
        "manual registration QC exclusions",
    )

    if not set(metadata["QIAL_has_imaging_data"].dropna().unique()).issubset({0, 1}):
        raise ValueError("QIAL_has_imaging_data contains values other than 0/1")
    if option3["subject_id"].duplicated().any():
        raise ValueError("Option 3 manifest contains duplicate subject IDs")
    if registration_exclusions["subject_id"].duplicated().any():
        raise ValueError("Registration QC exclusion list contains duplicate IDs")
    if not registration_exclusions["qc_status"].eq("fail").all():
        raise ValueError("Every registration QC exclusion must have qc_status=fail")

    metadata = metadata.copy()
    metadata["visit_date_parsed"] = pd.to_datetime(
        metadata["VISIT_DATE"], errors="coerce"
    )
    metadata["qial_imaging_match"] = metadata["QIAL_has_imaging_data"].eq(1)

    # QIAL normally selects exactly one clinical row per imaged subject.  The
    # stable date sort implements the prespecified earliest-eligible rule if a
    # future metadata release contains more than one QIAL-positive row.
    qial = metadata.loc[metadata["qial_imaging_match"]].copy()
    qial = qial.sort_values(
        ["CONNECTOME_KEY_CLEAN", "visit_date_parsed", "VISITNUM"],
        kind="stable",
        na_position="last",
    )
    qial["n_qial_matched_visits"] = qial.groupby(
        "CONNECTOME_KEY_CLEAN"
    )["CONNECTOME_KEY_CLEAN"].transform("size")
    qial["selected_earliest_qial_visit"] = ~qial.duplicated(
        "CONNECTOME_KEY_CLEAN", keep="first"
    )
    selected_qial = qial.loc[qial["selected_earliest_qial_visit"]].copy()

    # Keep one auditable row for every metadata subject.  Subjects without a
    # QIAL match retain their earliest metadata visit but cannot be eligible.
    fallback = metadata.sort_values(
        ["CONNECTOME_KEY_CLEAN", "visit_date_parsed", "VISITNUM"],
        kind="stable",
        na_position="last",
    ).drop_duplicates("CONNECTOME_KEY_CLEAN", keep="first")
    selected_ids = set(selected_qial["CONNECTOME_KEY_CLEAN"])
    fallback = fallback.loc[~fallback["CONNECTOME_KEY_CLEAN"].isin(selected_ids)]
    fallback["n_qial_matched_visits"] = 0
    fallback["selected_earliest_qial_visit"] = False
    cohort = pd.concat([selected_qial, fallback], ignore_index=True, sort=False)

    if cohort["CONNECTOME_KEY_CLEAN"].duplicated().any():
        raise RuntimeError("Subject-level cohort is not one row per subject")

    option3_ids = set(option3["subject_id"])
    cohort["subject_id"] = cohort["CONNECTOME_KEY_CLEAN"]
    cohort["visit_id"] = cohort["VISITNUM"]
    cohort["matched_visit_date"] = cohort["visit_date_parsed"].dt.strftime(
        "%Y-%m-%d"
    )
    cohort["age_at_scan"] = pd.to_numeric(cohort["VISIT_AGE"], errors="coerce").where(
        cohort["qial_imaging_match"]
    )
    cohort["age_at_scan_source"] = ""
    cohort.loc[cohort["qial_imaging_match"], "age_at_scan_source"] = (
        "VISIT_AGE_at_QIAL_matched_clinical_visit"
    )
    cohort["diagnosis_at_scan"] = cohort["DX_Label_harmonized"].where(
        cohort["qial_imaging_match"]
    )
    cohort["diagnosis_at_scan_source"] = ""
    cohort.loc[cohort["qial_imaging_match"], "diagnosis_at_scan_source"] = (
        "DX_Label_harmonized_at_QIAL_matched_clinical_visit"
    )
    cohort["option3_qc_pass"] = cohort["subject_id"].isin(option3_ids)
    cohort["date_match_primary"] = cohort["QIAL_match_confidence"].isin(
        ACCEPTED_MATCH_CONFIDENCE
    )
    cohort["age_diagnosis_complete"] = (
        cohort["age_at_scan"].notna() & cohort["diagnosis_at_scan"].notna()
    )
    cohort["cn_at_scan"] = (
        cohort["diagnosis_at_scan"].eq(NORMAL_DIAGNOSIS)
        & cohort["NORMCOG_01"].eq(1)
        & cohort["DEMENTIA_01"].eq(0)
        & cohort["MCI_01"].eq(0)
        & cohort["AD_01"].eq(0)
    )
    registration_reason = registration_exclusions.set_index("subject_id")[
        "qc_reason"
    ]
    cohort["manual_registration_qc_excluded"] = cohort["subject_id"].isin(
        registration_reason.index
    )
    cohort["manual_registration_qc_reason"] = (
        cohort["subject_id"].map(registration_reason).fillna("")
    )

    low_subject_audit, low_visit_details, low_audit_summary = (
        build_low_longitudinal_audit(metadata, cohort)
    )
    approved_low_ids = set(
        low_subject_audit.loc[
            low_subject_audit["approved_for_primary_by_longitudinal_rule"],
            "subject_id",
        ]
    )
    cohort["low_multiple_visits_all_cn_included"] = cohort["subject_id"].isin(
        approved_low_ids
    )
    cohort["date_match_primary_or_approved_low"] = (
        cohort["date_match_primary"]
        | cohort["low_multiple_visits_all_cn_included"]
    )
    cohort["metadata_primary_eligible"] = (
        cohort["qial_imaging_match"]
        & cohort["option3_qc_pass"]
        & cohort["date_match_primary_or_approved_low"]
        & cohort["age_diagnosis_complete"]
        & cohort["cn_at_scan"]
    )
    cohort["metadata_sensitivity_eligible"] = (
        cohort["qial_imaging_match"]
        & cohort["option3_qc_pass"]
        & cohort["age_diagnosis_complete"]
        & cohort["cn_at_scan"]
    )
    cohort["figure5_primary_eligible"] = (
        cohort["metadata_primary_eligible"]
        & ~cohort["manual_registration_qc_excluded"]
    )
    cohort["figure5_sensitivity_eligible"] = (
        cohort["metadata_sensitivity_eligible"]
        & ~cohort["manual_registration_qc_excluded"]
    )
    cohort["metadata_exclusion_reason"] = cohort.apply(flag_reason, axis=1)

    low_subject_audit.to_csv(
        args.output_dir / "low_longitudinal_diagnosis_audit.csv", index=False
    )
    low_visit_details.to_csv(
        args.output_dir / "low_longitudinal_visit_details.csv", index=False
    )
    low_audit_summary.to_csv(
        args.output_dir / "low_longitudinal_audit_summary.csv", index=False
    )

    output_columns = [
        "subject_id",
        "PTID",
        "PTID_clean",
        "visit_id",
        "matched_visit_date",
        "age_at_scan",
        "age_at_scan_source",
        "diagnosis_at_scan",
        "diagnosis_at_scan_source",
        "SUBJECT_SEX",
        "QIAL_has_imaging_data",
        "QIAL_match_confidence",
        "n_qial_matched_visits",
        "selected_earliest_qial_visit",
        "qial_imaging_match",
        "option3_qc_pass",
        "date_match_primary",
        "low_multiple_visits_all_cn_included",
        "date_match_primary_or_approved_low",
        "age_diagnosis_complete",
        "cn_at_scan",
        "metadata_primary_eligible",
        "metadata_sensitivity_eligible",
        "manual_registration_qc_excluded",
        "manual_registration_qc_reason",
        "figure5_primary_eligible",
        "figure5_sensitivity_eligible",
        "metadata_exclusion_reason",
    ]
    cohort = cohort[output_columns].sort_values("subject_id", kind="stable")
    cohort.to_csv(args.output_dir / "cohort_manifest.csv", index=False)

    primary = cohort.loc[cohort["figure5_primary_eligible"]].copy()
    sensitivity = cohort.loc[cohort["figure5_sensitivity_eligible"]].copy()
    primary.to_csv(args.output_dir / "figure5_primary_metadata_cn.csv", index=False)
    sensitivity.to_csv(
        args.output_dir / "figure5_all_qial_cn_sensitivity.csv", index=False
    )

    step_rows: list[dict[str, object]] = []
    all_subjects = pd.Series(True, index=cohort.index)
    remaining = count_step(
        step_rows, 0, "Unique subjects in metadata", all_subjects, len(cohort)
    )
    mask = cohort["qial_imaging_match"]
    remaining = count_step(
        step_rows, 1, "QIAL_has_imaging_data == 1", mask, remaining
    )
    mask &= cohort["option3_qc_pass"]
    remaining = count_step(
        step_rows, 2, "Intersect Option 3 QC 325", mask, remaining
    )
    mask &= cohort["date_match_primary_or_approved_low"]
    remaining = count_step(
        step_rows,
        3,
        "QIAL HIGH/MEDIUM or approved LOW with multiple all-CN visits",
        mask,
        remaining,
    )
    mask &= cohort["age_diagnosis_complete"]
    remaining = count_step(
        step_rows, 4, "Nonmissing matched-visit age and diagnosis", mask, remaining
    )
    mask &= cohort["cn_at_scan"]
    remaining = count_step(
        step_rows, 5, "Cognitively normal at matched visit", mask, remaining
    )
    mask &= ~cohort["manual_registration_qc_excluded"]
    count_step(
        step_rows,
        6,
        "Pass manual T1-DWI registration visualization QC",
        mask,
        remaining,
    )
    pd.DataFrame(step_rows).to_csv(
        args.output_dir / "exclusion_summary.csv", index=False
    )

    confidence_summary = (
        cohort.loc[cohort["qial_imaging_match"] & cohort["option3_qc_pass"]]
        .groupby(["QIAL_match_confidence", "diagnosis_at_scan"], dropna=False)
        .size()
        .rename("n_subjects")
        .reset_index()
    )
    confidence_summary.to_csv(
        args.output_dir / "date_match_diagnosis_summary.csv", index=False
    )

    rules = {
        "metadata_source": str(args.metadata.resolve()),
        "option3_manifest": str(args.option3_manifest.resolve()),
        "manual_registration_qc_exclusions": str(
            args.registration_qc_exclusions.resolve()
        ),
        "qial_imaging_rule": "QIAL_has_imaging_data == 1",
        "visit_selection_rule": (
            "Earliest VISIT_DATE among QIAL-positive rows; current input has "
            "exactly one QIAL-positive row per matched subject."
        ),
        "age_at_scan_definition": (
            "VISIT_AGE from the QIAL-matched clinical row; the source file does "
            "not expose a separate scan date or exact date difference."
        ),
        "diagnosis_at_scan_definition": (
            "DX_Label_harmonized from the same QIAL-matched clinical row."
        ),
        "primary_date_match_confidence": list(ACCEPTED_MATCH_CONFIDENCE),
        "primary_date_window_days": 90,
        "approved_low_primary_exception": (
            "A LOW (>90d) subject may enter the primary cohort only when the "
            "QIAL-matched visit is CN, there are at least two unique "
            "VISIT_DATE values, and every available clinical visit satisfies "
            "the strict harmonized CN rule. Single-visit LOW subjects remain "
            "review-only."
        ),
        "primary_cn_rule": (
            "DX_Label_harmonized == Normal, NORMCOG_01 == 1, and "
            "DEMENTIA_01/MCI_01/AD_01 == 0"
        ),
        "sensitivity_rule": (
            "All QIAL-positive Option 3 subjects with complete matched-visit "
            "age/diagnosis and CN status, including LOW and REVIEW matches."
        ),
        "low_longitudinal_review_rule": (
            "Among QIAL-positive Option 3 LOW (>90d) subjects who are CN at "
            "the matched visit, flag for further inclusion review if there is "
            "only one unique VISIT_DATE or if all clinical visits satisfy the "
            "strict harmonized CN rule. Multiple-visit all-CN LOW subjects are "
            "included in the primary cohort; single-visit LOW subjects remain "
            "review-only."
        ),
        "registration_qc_rule": (
            "Exclude every subject listed with qc_status=fail in "
            "manual_registration_qc_exclusions.csv."
        ),
        "not_yet_applied": [
            "representation common-availability intersection",
            "profile coverage threshold",
            "surface reconstruction QC exclusion",
            "protocol restriction or covariate",
        ],
    }
    (args.output_dir / "filter_rules.json").write_text(
        json.dumps(rules, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Metadata subjects: {len(cohort)}")
    print(f"QIAL imaging matches: {int(cohort['qial_imaging_match'].sum())}")
    print(f"Option 3 intersection: {int((cohort['qial_imaging_match'] & cohort['option3_qc_pass']).sum())}")
    print(f"Primary cohort after manual registration QC: {len(primary)}")
    print(f"All-QIAL sensitivity cohort after manual registration QC: {len(sensitivity)}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
