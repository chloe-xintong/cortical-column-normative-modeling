#!/usr/bin/env python3
"""Build one auditable subject-flow record for Figures 2, 4, and 5.

Description
-----------
This script reports the currently implemented cohort rules.  It does not
change any analysis cohort.  In particular, the source label
``REVIEW (ambiguous)`` is preserved as supplied; it is not re-derived because
the supplied QIAL table has no scan date or exact scan-to-visit day gap.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--metadata`` command-line input
- ``--figure5-manifest`` command-line input
- ``--figure4-historical-manifest`` command-line input
- ``--figure4-active-manifest`` command-line input
- ``--registration-exclusions`` command-line input
- ``--figure2-ab-subjects`` command-line input
- ``--figure2-registration-manifest`` command-line input
- ``--figure5-candidate-coverage`` command-line input
- ``--figure5-final-subjects`` command-line input
- ``--profile-root`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/cohort_audit/build_cohort_audit.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ACCEPTED_MATCH_CONFIDENCE = {"HIGH (<=30d)", "MEDIUM (31-90d)"}
AMBIGUOUS_MATCH_CONFIDENCE = "REVIEW (ambiguous)"
LOW_MATCH_CONFIDENCE = "LOW (>90d)"
NORMAL_DIAGNOSIS = "Normal"


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
        "--figure5-manifest",
        type=Path,
        default=repo
        / "analysis_results"
        / "figure5"
        / "cohort_selection"
        / "outputs"
        / "cohort_manifest.csv",
    )
    parser.add_argument(
        "--figure4-historical-manifest",
        type=Path,
        default=repo / "manifests" / "figure4_valid_325_subjects.csv",
    )
    parser.add_argument(
        "--figure4-active-manifest",
        type=Path,
        default=repo / "manifests" / "figure4_registration_qc_pass_320_subjects.csv",
    )
    parser.add_argument(
        "--registration-exclusions",
        type=Path,
        default=repo / "manifests" / "figure4_registration_qc_exclusions.csv",
    )
    parser.add_argument(
        "--figure2-ab-subjects",
        type=Path,
        default=repo / "manifests" / "o2_fs_50_fsaverage_subjects.txt",
    )
    parser.add_argument(
        "--figure2-registration-manifest",
        type=Path,
        default=repo / "manifests" / "registration_qc_adrc_subjects.csv",
    )
    parser.add_argument(
        "--figure5-candidate-coverage",
        type=Path,
        default=project
        / "results"
        / "figure5_age_prediction"
        / "outputs"
        / "features"
        / "candidate_coverage_screen.csv",
    )
    parser.add_argument(
        "--figure5-final-subjects",
        type=Path,
        default=project
        / "results"
        / "figure5_age_prediction"
        / "outputs"
        / "features"
        / "subject_order.csv",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=project / "results" / "fsaverage_result" / "columns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "analysis_results" / "cohort_audit" / "outputs",
    )
    return parser.parse_args()


def require_columns(table: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(table.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def read_ids(path: Path) -> set[str]:
    table = pd.read_csv(path, dtype={"subject_id": str})
    require_columns(table, {"subject_id"}, str(path))
    if table["subject_id"].duplicated().any():
        raise ValueError(f"Duplicate subject_id values in {path}")
    return set(table["subject_id"])


def strict_cn(table: pd.DataFrame) -> pd.Series:
    return (
        table["DX_Label_harmonized"].eq(NORMAL_DIAGNOSIS)
        & table["NORMCOG_01"].eq(1)
        & table["DEMENTIA_01"].eq(0)
        & table["MCI_01"].eq(0)
        & table["AD_01"].eq(0)
    )


def join_unique(values: pd.Series) -> str:
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned.loc[cleaned.ne("")]
    return "|".join(pd.unique(cleaned))


def build_visit_audit(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata["visit_date_parsed"] = pd.to_datetime(
        metadata["VISIT_DATE"], errors="coerce"
    )
    metadata["strict_cn"] = strict_cn(metadata)

    rows: list[dict[str, object]] = []
    for subject_id, group in metadata.groupby("CONNECTOME_KEY_CLEAN", sort=True):
        dated = group.loc[group["visit_date_parsed"].notna()].copy()
        unique_dates = sorted(dated["visit_date_parsed"].drop_duplicates())
        span_days = (
            int((unique_dates[-1] - unique_dates[0]).days)
            if len(unique_dates) >= 2
            else 0
        )

        per_date_diagnoses = (
            dated.groupby("visit_date_parsed", sort=True)["DX_Label_harmonized"]
            .apply(lambda values: set(values.dropna().astype(str).str.strip()) - {""})
        )
        same_date_conflict = bool(
            any(len(diagnoses) > 1 for diagnoses in per_date_diagnoses)
        )
        single_value_by_date = [
            next(iter(diagnoses))
            for diagnoses in per_date_diagnoses
            if len(diagnoses) == 1
        ]
        distinct_date_discordance = bool(
            len(per_date_diagnoses) >= 2
            and (
                len(set(single_value_by_date)) > 1
                or any(len(diagnoses) != 1 for diagnoses in per_date_diagnoses)
            )
        )

        qial_rows = group.loc[group["QIAL_has_imaging_data"].eq(1)]
        rows.append(
            {
                "subject_id": subject_id,
                "n_metadata_rows": len(group),
                "n_qial_positive_rows": len(qial_rows),
                "n_unique_visit_dates": len(unique_dates),
                "visit_date_span_days": span_days,
                "visit_dates": "|".join(
                    date.strftime("%Y-%m-%d") for date in unique_dates
                ),
                "diagnosis_sequence_all_rows": join_unique(
                    group["DX_Label_harmonized"]
                ),
                "same_date_diagnosis_conflict": same_date_conflict,
                "diagnosis_differs_across_distinct_dates": (
                    distinct_date_discordance
                ),
                "all_metadata_rows_strict_cn": bool(group["strict_cn"].all()),
            }
        )
    return pd.DataFrame(rows)


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


def add_step(
    summary_rows: list[dict[str, object]],
    figure: str,
    panel: str,
    step: int,
    criterion: str,
    before: set[str],
    after: set[str],
    reason_code: str,
    notes: str = "",
) -> None:
    if not after.issubset(before):
        raise RuntimeError(f"{figure} {panel} step {step} is not a nested set")
    summary_rows.append(
        {
            "figure": figure,
            "panel": panel,
            "step": step,
            "criterion": criterion,
            "n_before": len(before),
            "n_excluded_at_step": len(before - after),
            "n_remaining": len(after),
            "reason_code": reason_code,
            "notes": notes,
        }
    )


def add_exclusions(
    rows: list[dict[str, object]],
    figure: str,
    panel: str,
    step: int,
    reason_code: str,
    reason: str,
    excluded: set[str],
    cohort: pd.DataFrame,
) -> None:
    lookup = cohort.set_index("subject_id")
    for subject_id in sorted(excluded):
        metadata_reason = ""
        confidence = ""
        if subject_id in lookup.index:
            metadata_reason = str(lookup.at[subject_id, "metadata_exclusion_reason"])
            confidence = str(lookup.at[subject_id, "QIAL_match_confidence"])
            if confidence == "nan":
                confidence = ""
        rows.append(
            {
                "figure": figure,
                "panel": panel,
                "step": step,
                "subject_id": subject_id,
                "reason_code": reason_code,
                "reason": reason,
                "QIAL_match_confidence": confidence,
                "figure5_metadata_reason_all_flags": metadata_reason,
            }
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata, low_memory=False)
    cohort = pd.read_csv(args.figure5_manifest, dtype={"subject_id": str})
    coverage = pd.read_csv(
        args.figure5_candidate_coverage, dtype={"subject_id": str}
    )
    registration_exclusions = pd.read_csv(
        args.registration_exclusions, dtype={"subject_id": str}
    )

    require_columns(
        metadata,
        {
            "CONNECTOME_KEY_CLEAN",
            "VISIT_DATE",
            "DX_Label_harmonized",
            "NORMCOG_01",
            "DEMENTIA_01",
            "MCI_01",
            "AD_01",
            "QIAL_has_imaging_data",
        },
        "metadata",
    )
    require_columns(
        cohort,
        {
            "subject_id",
            "qial_imaging_match",
            "option3_qc_pass",
            "QIAL_match_confidence",
            "date_match_primary_or_approved_low",
            "age_diagnosis_complete",
            "cn_at_scan",
            "manual_registration_qc_excluded",
            "metadata_primary_eligible",
            "metadata_exclusion_reason",
        },
        "Figure 5 cohort manifest",
    )
    require_columns(
        coverage,
        {"subject_id", "coverage_fraction", "passes_minimum_coverage"},
        "Figure 5 coverage table",
    )
    require_columns(
        registration_exclusions,
        {"subject_id", "qc_status", "qc_reason"},
        "registration exclusions",
    )

    all_metadata = set(cohort["subject_id"])
    qial = set(cohort.loc[cohort["qial_imaging_match"], "subject_id"])
    option3 = read_ids(args.figure4_historical_manifest)
    figure4 = read_ids(args.figure4_active_manifest)
    final196 = read_ids(args.figure5_final_subjects)
    registration_failed = set(registration_exclusions["subject_id"])

    if option3 != set(cohort.loc[cohort["option3_qc_pass"], "subject_id"]):
        raise RuntimeError("Option 3 manifest and Figure 5 option3_qc_pass disagree")
    if option3 - registration_failed != figure4:
        raise RuntimeError("Active Figure 4 manifest is not historical 325 minus QC failures")

    # Canonical cross-figure ordering makes every Figure 5 exclusion exclusive:
    # data link -> profile availability -> registration -> date rule -> required
    # metadata -> diagnosis -> coverage.  Reordering independent filters does
    # not change the final set, but it makes the five registration failures
    # visible together instead of attributing four of them to an earlier date
    # filter.
    f5_imaging = all_metadata & qial
    f5_profiles = f5_imaging & option3
    f5_registration = f5_profiles - registration_failed
    approved_date = set(
        cohort.loc[cohort["date_match_primary_or_approved_low"], "subject_id"]
    )
    f5_date = f5_registration & approved_date
    complete = set(cohort.loc[cohort["age_diagnosis_complete"], "subject_id"])
    f5_complete = f5_date & complete
    cn = set(cohort.loc[cohort["cn_at_scan"], "subject_id"])
    f5_cn = f5_complete & cn
    coverage_pass = set(
        coverage.loc[coverage["passes_minimum_coverage"], "subject_id"]
    )
    f5_final = f5_cn & coverage_pass
    if f5_final != final196:
        raise RuntimeError("Reconstructed Figure 5 cohort does not match subject_order.csv")

    figure2_ab = {
        line.strip()
        for line in args.figure2_ab_subjects.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    figure2_c = read_ids(args.figure2_registration_manifest)

    summary_rows: list[dict[str, object]] = []
    summary_rows.append(
        {
            "figure": "Figure 2",
            "panel": "A-B",
            "step": 0,
            "criterion": "Fixed method-comparison cohort with both hemispheres usable",
            "n_before": len(figure2_ab),
            "n_excluded_at_step": 0,
            "n_remaining": len(figure2_ab),
            "reason_code": "fixed_method_comparison_cohort",
            "notes": "No whole-subject exclusion in the exported A-B analysis; invalid vertices are handled within subject.",
        }
    )
    add_step(
        summary_rows,
        "Figure 2",
        "C downstream eligibility",
        1,
        "Pass manual T1-DWI registration visualization QC",
        figure2_c,
        figure2_c - registration_failed,
        "manual_registration_qc_failure",
        "Figure 2C still displays all 327 QC candidates and marks the five failures; 322 pass for downstream use.",
    )
    add_step(
        summary_rows,
        "Figure 4",
        "A-D",
        0,
        "Historical complete Option 3 profile cohort",
        option3,
        option3,
        "complete_option3_profiles",
    )
    add_step(
        summary_rows,
        "Figure 4",
        "A-D",
        1,
        "Pass manual T1-DWI registration visualization QC",
        option3,
        figure4,
        "manual_registration_qc_failure",
        "Date confidence and diagnosis are descriptive only and do not filter Figure 4.",
    )

    f5_steps = [
        (0, "Unique subjects in metadata", all_metadata, all_metadata, "metadata_subject"),
        (1, "QIAL_has_imaging_data == 1", all_metadata, f5_imaging, "no_QIAL_imaging_match"),
        (2, "Complete Option 3 profile cohort", f5_imaging, f5_profiles, "not_in_Option3_325"),
        (3, "Pass manual T1-DWI registration visualization QC", f5_profiles, f5_registration, "manual_registration_qc_failure"),
        (4, "QIAL HIGH/MEDIUM or currently approved LOW exception", f5_registration, f5_date, "date_match_not_currently_approved"),
        (5, "Nonmissing matched-visit age and diagnosis", f5_date, f5_complete, "missing_age_or_diagnosis"),
        (6, "Strict cognitively normal status at matched visit", f5_complete, f5_cn, "not_cognitively_normal_at_matched_visit"),
        (7, "At least 95% bilateral valid-column coverage", f5_cn, f5_final, "profile_coverage_below_0.95"),
    ]
    for step, criterion, before, after, reason in f5_steps:
        add_step(
            summary_rows,
            "Figure 5",
            "A-C and primary models",
            step,
            criterion,
            before,
            after,
            reason,
            "Canonical exclusive ordering; final set is identical to the implemented 196-subject cohort.",
        )

    exclusion_rows: list[dict[str, object]] = []
    add_exclusions(
        exclusion_rows,
        "Figure 2",
        "C downstream eligibility",
        1,
        "manual_registration_qc_failure",
        "Visually confirmed T1-DWI registration failure",
        figure2_c & registration_failed,
        cohort,
    )
    add_exclusions(
        exclusion_rows,
        "Figure 4",
        "A-D",
        1,
        "manual_registration_qc_failure",
        "Visually confirmed T1-DWI registration failure",
        option3 - figure4,
        cohort,
    )
    f5_reason_text = {
        1: "No QIAL imaging match",
        2: "Not in the complete 325-subject Option 3 profile cohort",
        3: "Visually confirmed T1-DWI registration failure",
        4: "Neither HIGH/MEDIUM nor a currently approved LOW exception",
        5: "Missing age or diagnosis at the selected matched visit",
        6: "Does not satisfy the strict matched-visit CN definition",
        7: "Bilateral valid-column coverage is below 95%",
    }
    for step, _, before, after, reason in f5_steps[1:]:
        add_exclusions(
            exclusion_rows,
            "Figure 5",
            "A-C and primary models",
            step,
            reason,
            f5_reason_text[step],
            before - after,
            cohort,
        )

    visit_audit = build_visit_audit(metadata)
    date_audit = cohort.merge(
        visit_audit, on="subject_id", how="left", validate="one_to_one"
    )
    date_audit["in_historical_figure4_325"] = date_audit["subject_id"].isin(option3)
    date_audit["in_active_figure4_320"] = date_audit["subject_id"].isin(figure4)
    date_audit["in_final_figure5_196"] = date_audit["subject_id"].isin(final196)
    date_audit["source_match_is_ambiguous"] = date_audit[
        "QIAL_match_confidence"
    ].eq(AMBIGUOUS_MATCH_CONFIDENCE)
    date_audit["has_multiple_distinct_visit_dates"] = date_audit[
        "n_unique_visit_dates"
    ].ge(2)
    date_audit["longitudinal_three_part_rule_evaluable"] = False
    date_audit["longitudinal_three_part_rule_note"] = (
        "A numeric definition of 'long visit gap' is not specified; use "
        "n_unique_visit_dates, visit_date_span_days, and "
        "diagnosis_differs_across_distinct_dates after fixing that threshold."
    )

    date_columns = [
        "subject_id",
        "in_historical_figure4_325",
        "in_active_figure4_320",
        "in_final_figure5_196",
        "QIAL_match_confidence",
        "source_match_is_ambiguous",
        "matched_visit_date",
        "diagnosis_at_scan",
        "cn_at_scan",
        "date_match_primary_or_approved_low",
        "low_multiple_visits_all_cn_included",
        "manual_registration_qc_excluded",
        "n_metadata_rows",
        "n_qial_positive_rows",
        "n_unique_visit_dates",
        "has_multiple_distinct_visit_dates",
        "visit_date_span_days",
        "visit_dates",
        "diagnosis_sequence_all_rows",
        "same_date_diagnosis_conflict",
        "diagnosis_differs_across_distinct_dates",
        "all_metadata_rows_strict_cn",
        "longitudinal_three_part_rule_evaluable",
        "longitudinal_three_part_rule_note",
        "metadata_exclusion_reason",
    ]
    option3_date_audit = date_audit.loc[
        date_audit["in_historical_figure4_325"], date_columns
    ].sort_values("subject_id", kind="stable")
    ambiguous = option3_date_audit.loc[
        option3_date_audit["source_match_is_ambiguous"]
    ].copy()
    low_audit = option3_date_audit.loc[
        option3_date_audit["QIAL_match_confidence"].eq(LOW_MATCH_CONFIDENCE)
    ].copy()

    # This is an impact audit, not an implemented cohort change.  It answers
    # which currently excluded subjects would be reconsidered if source
    # ambiguity alone were not an exclusion and a single distinct clinical
    # date were not an exclusion.  Current matched-visit CN and registration
    # rules remain in force.
    reconsider = option3_date_audit.loc[
        option3_date_audit["in_active_figure4_320"]
        & option3_date_audit["cn_at_scan"]
        & ~option3_date_audit["date_match_primary_or_approved_low"]
        & (
            option3_date_audit["source_match_is_ambiguous"]
            | (
                option3_date_audit["QIAL_match_confidence"].eq(
                    LOW_MATCH_CONFIDENCE
                )
                & option3_date_audit["n_unique_visit_dates"].eq(1)
            )
        )
    ].copy()
    reconsider["reconsideration_reason"] = np.where(
        reconsider["source_match_is_ambiguous"],
        "source_ambiguous_but_no_distinct_date_diagnosis_discordance",
        "LOW_with_one_distinct_clinical_date",
    )
    reconsider["recomputed_bilateral_coverage_fraction"] = [
        bilateral_profile_coverage(args.profile_root, subject_id)
        for subject_id in reconsider["subject_id"]
    ]
    reconsider["recomputed_coverage_ge_0_95"] = reconsider[
        "recomputed_bilateral_coverage_fraction"
    ].ge(0.95)
    reconsider_columns = [
        "subject_id",
        "reconsideration_reason",
        "QIAL_match_confidence",
        "matched_visit_date",
        "diagnosis_at_scan",
        "n_unique_visit_dates",
        "visit_date_span_days",
        "diagnosis_differs_across_distinct_dates",
        "same_date_diagnosis_conflict",
        "recomputed_bilateral_coverage_fraction",
        "recomputed_coverage_ge_0_95",
    ]

    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "figure_cohort_step_summary.csv", index=False
    )
    pd.DataFrame(exclusion_rows).to_csv(
        args.output_dir / "figure_subject_exclusions.csv", index=False
    )
    option3_date_audit.to_csv(
        args.output_dir / "date_match_visit_audit_325.csv", index=False
    )
    ambiguous.to_csv(args.output_dir / "ambiguous_subjects_28.csv", index=False)
    reconsider[reconsider_columns].to_csv(
        args.output_dir / "candidate_rule_change_impact_not_applied.csv",
        index=False,
    )

    summary = {
        "status": "complete",
        "ambiguous_definition": (
            "QIAL_match_confidence == 'REVIEW (ambiguous)' in the supplied source; "
            "not re-derived in this repository"
        ),
        "n_ambiguous_historical_figure4_325": int(len(ambiguous)),
        "n_ambiguous_active_figure4_320": int(
            ambiguous["in_active_figure4_320"].sum()
        ),
        "n_ambiguous_final_figure5_196": int(
            ambiguous["in_final_figure5_196"].sum()
        ),
        "ambiguous_unique_visit_date_counts": {
            str(int(key)): int(value)
            for key, value in ambiguous["n_unique_visit_dates"]
            .value_counts()
            .sort_index()
            .items()
        },
        "n_ambiguous_with_distinct_date_diagnosis_discordance": int(
            ambiguous["diagnosis_differs_across_distinct_dates"].sum()
        ),
        "n_ambiguous_with_same_date_diagnosis_conflict": int(
            ambiguous["same_date_diagnosis_conflict"].sum()
        ),
        "low_historical_figure4_325": {
            "n_total": int(len(low_audit)),
            "n_currently_approved": int(
                low_audit["date_match_primary_or_approved_low"].sum()
            ),
            "n_cn_single_distinct_date": int(
                (
                    low_audit["cn_at_scan"]
                    & low_audit["n_unique_visit_dates"].eq(1)
                ).sum()
            ),
            "n_cn_multiple_dates_not_all_cn": int(
                (
                    low_audit["cn_at_scan"]
                    & low_audit["n_unique_visit_dates"].ge(2)
                    & ~low_audit["all_metadata_rows_strict_cn"]
                ).sum()
            ),
            "n_not_cn_at_matched_visit": int((~low_audit["cn_at_scan"]).sum()),
        },
        "candidate_rule_change_not_applied": {
            "n_reconsidered_after_current_CN_and_registration_rules": int(
                len(reconsider)
            ),
            "n_reconsidered_passing_recomputed_coverage": int(
                reconsider["recomputed_coverage_ge_0_95"].sum()
            ),
            "current_final_n": len(final196),
            "candidate_final_n_if_all_reconsidered_subjects_are_accepted": int(
                len(final196) + reconsider["recomputed_coverage_ge_0_95"].sum()
            ),
            "warning": (
                "Impact calculation only; no cohort, fold, model, or figure was changed. "
                "The source meaning of REVIEW (ambiguous) and the long-gap threshold "
                "must be approved first."
            ),
        },
        "current_figure5_date_step_exclusions_before_registration_reordering": {
            AMBIGUOUS_MATCH_CONFIDENCE: int(
                (
                    cohort["option3_qc_pass"]
                    & cohort["QIAL_match_confidence"].eq(AMBIGUOUS_MATCH_CONFIDENCE)
                    & ~cohort["date_match_primary_or_approved_low"]
                ).sum()
            ),
            LOW_MATCH_CONFIDENCE: int(
                (
                    cohort["option3_qc_pass"]
                    & cohort["QIAL_match_confidence"].eq(LOW_MATCH_CONFIDENCE)
                    & ~cohort["date_match_primary_or_approved_low"]
                ).sum()
            ),
        },
        "figure2_ab_n": len(figure2_ab),
        "figure2_c_qc_candidates_n": len(figure2_c),
        "figure4_historical_n": len(option3),
        "figure4_active_n": len(figure4),
        "figure5_final_n": len(final196),
        "source_files": {
            "metadata": str(args.metadata.resolve()),
            "figure5_manifest": str(args.figure5_manifest.resolve()),
            "figure4_historical_manifest": str(
                args.figure4_historical_manifest.resolve()
            ),
            "figure4_active_manifest": str(args.figure4_active_manifest.resolve()),
            "figure5_candidate_coverage": str(
                args.figure5_candidate_coverage.resolve()
            ),
            "figure5_final_subjects": str(args.figure5_final_subjects.resolve()),
            "profile_root_for_candidate_coverage": str(args.profile_root.resolve()),
        },
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
