# Cross-figure cohort and exclusion audit

This directory provides one subject-level record of cohort flow for Figures 2,
4, and 5. It reports the currently implemented analysis sets; it does not
silently change a cohort or rerun a model.

## Main finding: `ambiguous` is currently over-broad for Figure 5

Here, `ambiguous` means exactly:

```text
QIAL_match_confidence == "REVIEW (ambiguous)"
```

That label is supplied by the source QIAL table. This repository cannot
independently reconstruct it because the table does not contain a scan date or
an exact scan-to-clinical-visit day difference.

Current counts are:

- historical Figure 4 profile cohort: 28 / 325 ambiguous;
- active Figure 4 cohort: 26 / 320 ambiguous (D0128 and D0163 were removed by
  registration QC);
- final Figure 5 cohort: 0 / 196 ambiguous.

The current Figure 5 code excludes all 28 ambiguous subjects in the historical
325-person pool at the date-match step. This is not the same as the proposed
longitudinal rule “exclude only when there are multiple distinct visits, the
visits are far apart, and diagnoses differ.” Among the 28 source-ambiguous
subjects:

- 25 have one unique `VISIT_DATE`;
- 3 have two unique `VISIT_DATE` values;
- 0 have diagnoses that differ across distinct visit dates;
- 4 have conflicting diagnosis labels on duplicate rows from the same date.

The four same-date conflicts are D0233, D0269, D0272, and D0276. They require
within-visit reconciliation; they are not evidence of change across time.

Therefore none of the 28 would be excluded for longitudinal discordance under
that three-part rule, regardless of the eventual “far apart” threshold. Other
independent criteria—matched-visit CN status, registration QC, and coverage—can
still exclude a subject.

## Current Figure 5 LOW rule

`LOW (>90d)` is a scan-to-selected-clinical-visit confidence category, whereas
longitudinal visit span is the time between a subject's distinct clinical
dates. These must not be treated as the same quantity.

The current code admits a LOW subject only if the selected visit is strictly
CN, there are at least two distinct clinical dates, and every available visit
is strictly CN. In the historical 325-person pool there are 47 LOW subjects:
19 satisfy this exception and 28 do not. The 28 current date-step exclusions
contain both single-visit subjects and longitudinally discordant subjects.

Among the LOW subjects whose selected visit is CN:

- 7 have multiple distinct dates and are not consistently CN: D0071, D0080,
  D0095, D0107, D0110, D0117, and D0141;
- 8 have only one distinct date: D0103, D0143, D0202, D0292, D0349, D0352,
  D0388, and D0398;
- the other 19 have multiple dates and are consistently CN.

D0388 independently fails registration QC. Thus, under the remembered
longitudinal rule, seven of the eight single-date LOW subjects would become
date-eligible before coverage is checked; D0388 would remain excluded for
registration.

Using the current matched-visit diagnosis and registration rules, the proposed
interpretation would reconsider 28 subjects: 21 source-ambiguous CN subjects
and 7 single-date LOW CN subjects. All 28 have recomputed bilateral profile
coverage at or above 95%. The candidate final Figure 5 cohort would therefore
be 224 rather than 196. This is an impact calculation only: the cohort, folds,
models, and figures have not been changed. Same-date conflicts and the original
meaning of the source ambiguous label still require a decision.

This is stricter than “exclude only when multiple sufficiently separated
visits disagree,” because a single-visit LOW subject is currently excluded.

## Recommended terminology and pending decision

Use separate fields for separate problems:

- `source_match_is_ambiguous`: the supplied QIAL label;
- `n_unique_visit_dates`: distinct clinical visits, based on date rather than
  raw row count;
- `visit_date_span_days`: longitudinal span between first and last date;
- `same_date_diagnosis_conflict`: duplicate records on one date disagree;
- `diagnosis_differs_across_distinct_dates`: longitudinal diagnoses disagree.

Before changing Figure 5, one number still needs to be fixed: the minimum
`visit_date_span_days` that means “far apart.” The audit intentionally does not
reuse 90 days automatically: the existing 90-day cutoff describes scan-to-visit
matching, not the interval between two clinical visits.

After that threshold is approved, the proposed longitudinal exclusion can be
implemented as the conjunction of:

1. at least two unique clinical dates;
2. visit-date span at or above the fixed threshold; and
3. diagnosis discordance across distinct dates.

Same-date duplicate rows should be reconciled as within-visit data conflicts,
not counted as longitudinal visits.

## Current subject flow

The canonical audit order makes each exclusion exclusive. Reordering
independent filters does not alter the final participant sets.

| Figure/panel | Step | Remaining | Excluded at step | Meaning |
| --- | ---: | ---: | ---: | --- |
| Figure 2A-B | fixed comparison cohort | 50 | 0 | No whole-subject exclusion in the exported analysis |
| Figure 2C | registration QC candidates | 327 | 5 flagged | All 327 are displayed; 322 pass for downstream use |
| Figure 4A-D | complete Option 3 profiles | 325 | 0 | Historical input |
| Figure 4A-D | registration QC | 320 | 5 | Active primary cohort |
| Figure 5 | unique metadata subjects | 448 | 0 | Starting pool |
| Figure 5 | QIAL imaging match | 369 | 79 | No imaging match |
| Figure 5 | complete Option 3 profiles | 325 | 44 | Not in profile cohort |
| Figure 5 | registration QC | 320 | 5 | Visual T1-DWI failure |
| Figure 5 | current date rule | 268 | 52 | 26 ambiguous + 26 non-approved LOW after prior registration removal |
| Figure 5 | complete age/diagnosis | 268 | 0 | Required matched-visit metadata |
| Figure 5 | strict CN status | 200 | 68 | Not CN at selected visit |
| Figure 5 | at least 95% coverage | 196 | 4 | D0072, D0094, D0191, D0222 |

Figure 3 is not implemented. Figure 4 does not filter on date confidence or
diagnosis. Figure 5A-C and its primary model comparisons share the same final
196 participants.

The five visually confirmed registration failures are D0128, D0163, D0248,
D0296, and D0388. The Figure 2C pass count of 322 and the Figure 4 count of 320
have different starting pools: Figure 2C audits 327 registrations, while
Figure 4 starts from the separate 325-subject complete-profile cohort.

## Generated files

- `outputs/figure_cohort_step_summary.csv`: counts and definitions at every
  step;
- `outputs/figure_subject_exclusions.csv`: every excluded subject, figure,
  step, and reason;
- `outputs/date_match_visit_audit_325.csv`: date/visit audit for all 325
  complete-profile subjects;
- `outputs/ambiguous_subjects_28.csv`: focused audit of the 28 source-ambiguous
  subjects;
- `outputs/candidate_rule_change_impact_not_applied.csv`: the 28 subjects who
  would be reconsidered under the remembered interpretation, with recomputed
  coverage;
- `outputs/audit_summary.json`: machine-readable checks and headline counts.

Rebuild with:

```bash
python analysis_results/cohort_audit/build_cohort_audit.py
```
