# Figure 5 cohort statistics under alternative visit rules

The saved Figure 5 models continue to use the current 196-subject primary
cohort. The alternative cohorts below are descriptive sensitivity sets only;
no folds or models were rerun.

Manual T1-DWI registration QC pass is mandatory in every rule. Every Figure 5
rule also requires complete Option 3 profiles, complete matched-visit age and
diagnosis, strict matched-visit CN status, and at least 95% bilateral valid
column coverage. Only the visit-match rule changes.

## Rule definitions and counts

| Rule | Visit rule | N |
| --- | --- | ---: |
| Current primary | HIGH/MEDIUM, or LOW with at least two distinct dates and every visit strict CN; ambiguous excluded | 196 |
| LOW/ambiguous visit-rule sensitivity | Keep HIGH/MEDIUM. For LOW/ambiguous, exclude only when there are at least two distinct dates, span is at least 365 days, and diagnosis differs across dates | 224 |
| Global visit-consistency sensitivity | Apply the same 365-day longitudinal-discordance exclusion to every confidence level, including HIGH/MEDIUM | 200 |
| All-QIAL CN sensitivity | No confidence restriction | 231 |

The global rule excludes 24 HIGH/MEDIUM participants retained by the targeted
LOW/ambiguous rule. These are subjects whose selected scan-adjacent visit is CN
but whose diagnosis changes at a later clinical visit. A later longitudinal
change does not make the original scan-to-visit match unreliable, so this rule
should be treated as a deliberately stringent sensitivity analysis rather than
as the preferred primary definition.

## Data availability and all-diagnosis label review

The data-availability audit must not impose a CN restriction. Normal, MCI, AD,
Other/Impaired, and Unknown labels all remain in scope. The relevant question
is whether repeated records for the same subject have the same diagnosis label
or different labels.

| Type | Pool or review subset | N | Definition |
| --- | --- | ---: | --- |
| Data availability | All metadata subjects | 448 | Complete metadata universe; 79 currently lack a QIAL imaging match |
| Data availability | Complete Option 3 DWI profiles | 325 | Complete DWI-profile pool before manual registration QC |
| Data availability | QC-passed complete-DWI reference | 320 | Current all-subject DWI reference before label-consistency and coverage decisions |
| Candidate | No observed label conflict, coverage pass | 259 | Single-date subjects or multiple-date subjects with the same label; all diagnosis and QIAL categories allowed |
| Review subset | Different diagnosis label | 54 | 50 differ across distinct dates and four conflict on duplicate rows from the same date |
| Review ceiling | All-QIAL, all-label, coverage pass | 313 | Union of the 259 no-conflict candidates and 54 different-label review subjects |

Before the 95% coverage requirement, the 320-person QC-passed reference
contains 266 subjects with no observed label conflict and 54 with different
labels. Seven of the 266 fail coverage; none of the 54 fails coverage. Thus the
coverage-passing partition is `259 + 54 = 313`.

Within the 54 different-label subjects, 50 have diagnosis labels that differ
across distinct visit dates, while D0233, D0269, D0272, and D0276 have
conflicting labels on duplicate records from the same date. These two conflict
types should remain separate during adjudication.

## Demographic and match composition

`SUBJECT_SEX` is reported as stored because its codebook has not been verified.

| Rule | N | Age, mean ± SD | Age, median [IQR] | Sex code 1 / 2 / 9 | HIGH / MEDIUM / LOW / ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current primary | 196 | 58.09 ± 10.53 | 56 [51, 67] | 65 / 130 / 1 | 71 / 107 / 18 / 0 |
| LOW/ambiguous visit rule | 224 | 56.18 ± 11.57 | 55 [49, 65.25] | 72 / 151 / 1 | 71 / 107 / 25 / 21 |
| Global visit consistency | 200 | 55.53 ± 11.52 | 54 [48.75, 64] | 63 / 136 / 1 | 60 / 94 / 25 / 21 |
| All-QIAL CN | 231 | 56.50 ± 11.58 | 55 [49, 66] | 74 / 156 / 1 | 71 / 107 / 32 / 21 |

All four Figure 5 cohorts contain only strict matched-visit CN participants.
The current primary cohort is older on average because the current date rule
removes all ambiguous matches and single-date LOW matches.

## Age-band composition

Age bands are left-closed ten-year intervals based on matched-visit
`VISIT_AGE`: 20–29 means `20 <= age < 30`, continuing through 80–89.

| Rule | 20–29 | 30–39 | 40–49 | 50–59 | 60–69 | 70–79 | 80–89 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current primary | 1 (0.5%) | 7 (3.6%) | 30 (15.3%) | 79 (40.3%) | 44 (22.4%) | 35 (17.9%) | 0 (0.0%) |
| LOW/ambiguous visit rule | 2 (0.9%) | 15 (6.7%) | 44 (19.6%) | 81 (36.2%) | 47 (21.0%) | 35 (15.6%) | 0 (0.0%) |
| Global visit consistency | 2 (1.0%) | 15 (7.5%) | 40 (20.0%) | 75 (37.5%) | 41 (20.5%) | 27 (13.5%) | 0 (0.0%) |
| All-QIAL CN | 2 (0.9%) | 15 (6.5%) | 44 (19.0%) | 82 (35.5%) | 50 (21.6%) | 38 (16.5%) | 0 (0.0%) |
| Figure 4 QC-pass reference | 2 (0.6%) | 20 (6.2%) | 56 (17.5%) | 104 (32.5%) | 73 (22.8%) | 64 (20.0%) | 1 (0.3%) |

## Figure 4 QC-pass reference

The Figure 4 reference contains 320 participants after the five fixed manual
registration failures are removed. It intentionally does not apply the Figure
5 CN, date-confidence, or 95% coverage restrictions.

- age: 57.83 ± 11.79 years; median 58 [50, 67]; range 28–80;
- sex codes 1 / 2 / 9: 117 / 202 / 1;
- QIAL HIGH / MEDIUM / LOW / ambiguous: 105 / 144 / 45 / 26;
- matched-visit diagnoses: Normal 235, MCI 23, AD 7,
  Other/Impaired non-demented 26, Unknown 29.

## Machine-readable outputs

- `outputs/rule_statistics/cohort_rule_definitions.csv`
- `outputs/rule_statistics/cohort_rule_demographic_summary.csv`
- `outputs/rule_statistics/cohort_rule_group_counts.csv`
- `outputs/rule_statistics/cohort_rule_age_band_counts.csv`
- `outputs/rule_statistics/cohort_rule_subject_membership.csv`
- `outputs/rule_statistics/registration_qc_pass_coverage_320.csv`
- `outputs/rule_statistics/cohort_rule_statistics_report.json`

Rebuild with:

```bash
python analysis_results/figure5/cohort_selection/build_cohort_rule_statistics.py
```
