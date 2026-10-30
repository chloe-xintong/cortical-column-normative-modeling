# Figure 5 cohort selection

This directory fixes the metadata-stage inclusion rules before any Figure 5
model is fitted.

Run:

```bash
/home/xintong/.conda/envs/cortical_trajectory_analysis/bin/python \
  analysis_results/figure5/cohort_selection/build_figure5_metadata_cohort.py
```

The primary metadata cohort requires:

1. `QIAL_has_imaging_data == 1`;
2. membership in `manifests/figure4_valid_325_subjects.csv`;
3. QIAL date-match confidence `HIGH (<=30d)` or `MEDIUM (31-90d)`, or a
   prespecified LOW exception with at least two distinct clinical dates and
   strict CN status at every available visit;
4. nonmissing age and diagnosis at the matched clinical visit; and
5. harmonized cognitively normal status at that visit.
6. passage of manual T1-to-DWI registration visualization QC.

`age_at_scan` is operationally defined as `VISIT_AGE` on the QIAL-matched
clinical row. The supplied QIAL CSV does not contain a separate scan-date or
exact day-difference column. The output therefore retains match confidence and
the source definition explicitly.

The fixed manual registration failures are stored in
`manual_registration_qc_exclusions.csv`; they are automatically removed every
time the cohort is rebuilt.

For QIAL `LOW (>90d)` matches in the Option 3 cohort, the builder also audits
all available clinical rows in the metadata. It writes:

- `low_longitudinal_diagnosis_audit.csv`: one row per LOW subject, including
  the number of unique `VISIT_DATE` values, longitudinal diagnosis consistency,
  and the further-review flag;
- `low_longitudinal_visit_details.csv`: the underlying visit-level records;
- `low_longitudinal_audit_summary.csv`: counts by matched-visit CN status and
  longitudinal review group.

A LOW subject is flagged for further inclusion review when the matched visit is
CN and either (a) there is only one unique clinical visit date or (b) multiple
clinical visits all satisfy the strict harmonized CN rule. Group (b) is
included in the primary cohort. Group (a) remains review-only and is not
automatically included.

This stage does **not** yet apply the common-representation intersection,
coverage threshold, surface-reconstruction QC exclusion, or protocol rule.

## Alternative-rule descriptive statistics

`build_cohort_rule_statistics.py` compares the unchanged 196-person primary
cohort with explicitly labeled visit-rule sensitivity cohorts after the shared
registration, strict-CN, and coverage QC requirements. The readable summary is
in `RULE_STATISTICS.md`; machine-readable tables are under
`outputs/rule_statistics/`. The same tables are loaded in Section 2A of the
Figure 5 plotting notebook. Alternative cohorts are descriptive only and do
not replace the saved folds or model outputs.
