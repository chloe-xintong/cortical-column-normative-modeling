# Normal–MCI–AD cohort selection for downstream deviation analyses

Status: frozen descriptive specification, 2026-11-10.

This document records the cohort rules agreed for downstream normative-deviation and cognition analyses. The frozen Figure 6 Normal reference model is not changed. Only the visit rule for MCI and AD targets is relaxed as described below.

## 1. Common imaging-QC source pool

MCI and AD candidates must belong to the existing all-diagnosis imaging-QC cohort in
`manifests/figure4_coverage95_pass_313_subjects.csv`.

The common imaging flow is:

| Step | N | Normal | MCI | AD | Other/impaired | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Complete Option 3 profiles | 325 | 238 | 23 | 8 | 26 | 30 |
| Manual T1-to-DWI registration QC pass | 320 | 235 | 23 | 7 | 26 | 29 |
| Bilateral valid-column coverage at least 95% | 313 | 231 | 21 | 7 | 26 | 28 |

The coverage definition is
`sum(in_analysis_mask & column_valid) / sum(in_analysis_mask)` across both hemispheres.
No age restriction is applied in this candidate-pool table.

## 2. Frozen Normal reference cohort

The Normal reference cohort remains exactly the frozen Figure 6 cohort of 186 participants.
No Normal participant is added and no fold assignment is changed.

The existing Figure 6 criteria remain in force:

1. cognitively normal at the QIAL-matched clinical visit;
2. member of the Figure 5 primary cohort, including its original visit-match rule;
3. complete Option 3 profiles and at least 95% bilateral valid-column coverage;
4. passed manual T1-to-DWI registration QC;
5. age eligibility `40 <= age < 80` (observed reference age range 41--78 years);
6. sex code 1 or 2;
7. required regional profile and FreeSurfer thickness inputs available;
8. D0271 and D0285 excluded for `source_image_structural_abnormality`;
9. original five outer folds retained, with held-out sizes 38, 38, 37, 36, and 37.

The normative model remains the Figure 6 Warped M2 model:

```text
raw node/depth MD ~ linear age + categorical sex code + local FreeSurfer thickness
```

Reference-sample deviations must continue to use held-out out-of-fold z-scores. Full-cohort
in-sample z-scores must not replace them.

Canonical reference files:

- `analysis_results/figure6/ico4_warped_m2_186/frozen_cohort_manifest.csv`
- `analysis_results/figure6/ico4_warped_m2_186/outer_fold_assignments.csv`
- `analysis_results/figure6/ico4_warped_m2_186/oof/calibration_overall.csv`

## 3. Relaxed MCI target rule

The starting imaging-QC pool contains 21 subjects whose QIAL-matched diagnosis is MCI.
Twenty are retained and D0142 is excluded.

Retained MCI subjects:

```text
D0052 D0126 D0129 D0139 D0156 D0183 D0185 D0190 D0204 D0242
D0243 D0257 D0259 D0290 D0315 D0316 D0354 D0357 D0435 D0439
```

The relaxed rule permits:

- HIGH (`<=30d`) or MEDIUM (`31--90d`) QIAL matches;
- a single LOW/ambiguous MCI visit when there is no contradictory record closer to the scan;
- longitudinal diagnostic change when the MCI visit is the visit closest to the scan;
- D0156 under the exact-date adjudication below.

### D0156 adjudication: include as MCI

- DWI scan: 2023-11-28;
- MCI visit: 2023-08-10, 110 days before the scan;
- later Normal visit: 2025-05-12, 531 days after the scan;
- the two clinical visits are 641 days apart.

The MCI visit is substantially closer to the scan and is therefore used as the scan-time
diagnosis. Record `visit_rule_exception = nearest_exact_scan_date_110d`.

### D0142 adjudication: exclude

- DWI scan: 2024-03-22;
- MCI visit: 2023-07-13, 253 days before the scan;
- Other/impaired visit: 2025-04-08, 382 days after the scan;
- the two clinical visits are 635 days apart.

Although MCI is the closer diagnosis, both visits are remote from the scan. D0142 is excluded
from the primary and relaxed MCI cohorts. It may be reported only in a separately labelled
sensitivity analysis if that decision is made later.

Final pre-age-filter MCI count: **20**.

## 4. Relaxed AD target rule

All seven subjects whose QIAL-matched diagnosis is AD are retained:

```text
D0029 D0064 D0070 D0083 D0176 D0247 D0370
```

The relaxed rule permits:

- HIGH or MEDIUM QIAL matches;
- a single LOW AD visit when there is no contradictory record closer to the scan;
- repeated consistent AD visits;
- D0029 under the exact-date adjudication below.

### D0029 adjudication: include as AD

- DWI scan: 2024-06-28;
- AD visit: 2024-03-21, 99 days before the scan;
- Other/impaired visit: 2022-09-19, 648 days before the scan;
- MCI visit: 2021-03-04, 1,212 days before the scan;
- consecutive clinical-visit gaps are 564 and 549 days.

AD is unambiguously the diagnosis closest to the scan. Record
`visit_rule_exception = nearest_exact_scan_date_99d`.

Final pre-age-filter AD count: **7**.

## 5. Frozen counts and common analysis age range

The reference and target counts serve different roles and must not be combined into a new
training cohort:

| Role | Group | Pre-age-filter N | Age handling |
| --- | --- | ---: | --- |
| Frozen normative training/reference | Normal | 186 | Existing Figure 6 rule; unchanged |
| Downstream target candidates | MCI | 20 | Count before a target age-support filter |
| Downstream target candidates | AD | 7 | Count before a target age-support filter |

The pre-age-filter descriptive total is 213 subjects (186 Normal + 20 MCI + 7 AD), but only the
186 Normal participants train the normative model.

The common downstream analysis range is **41--78 years, inclusive**. This matches the observed
age support of the frozen Figure 6 Normal reference cohort. Applying this common range:

- retains all 186 frozen Normal reference participants;
- removes D0183 (MCI, age 36) and D0290 (MCI, age 39), leaving MCI=18;
- removes D0176 (AD, age 80), leaving AD=6.

The final age-supported analysis cohort is therefore:

| Group | Final N |
| --- | ---: |
| Normal | 186 |
| MCI | 18 |
| AD | 6 |
| **Total** | **210** |

This common analysis-range restriction does not modify the frozen Normal model, its original
five folds, or the pre-age-filter diagnostic adjudications above.

## 6. Cognitive-data handling

- Cognition must come from the same QIAL-matched visit used for the scan-time diagnosis.
- Do not substitute cognition from a different visit.
- Treat `MOCATOTS == 88` as missing.
- Keep Global Cognition Composite and MoCA analyses separate and report their complete-case N.
- For retained MCI and AD targets, the current matched-visit cognition fields are available;
  final complete-case counts must be rechecked after the target age-support filter and after
  z-score availability is established.

Under the agreed 41--78-year analysis range, Global Cognition Composite is available for all
210 participants. MoCA is available for 209 participants; D0127 in the frozen Normal group has
`MOCATOTS == 88` and is treated as missing.

## 7. Required audit fields

Every downstream subject-level analysis table should retain at least:

```text
subject_id
diagnosis_at_scan
scan_date
matched_visit_date
scan_visit_abs_days
QIAL_match_confidence
n_unique_visit_dates
longitudinal_diagnosis_change
visit_rule_exception
reference_or_target
outer_fold
coverage_fraction
source_image_qc_status
```

The exact DWI dates for D0029, D0142, and D0156 were verified from their raw DWI JSON/BXH
sidecars under `/mnt/newStor/paros/paros_WORK/tmp_ADRC_MUSE/<subject_id>/`. The T1 sidecars for
these subjects contain anonymized placeholder dates and must not be used for visit-distance
calculations.

