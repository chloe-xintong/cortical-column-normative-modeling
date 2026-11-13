# Figure 6 ico4 execution plan (not submitted)

No ico4 x 21 array job was submitted in this pilot. The following plan is
conditional on clearing the regional Gaussian-tail QC gate.

## Pre-execution gates

1. Verify the source codebook for `SUBJECT_SEX`; keep `sex_code_1` and
   `sex_code_2` labels until that verification is documented.
2. Audit D0271's native DWI, Option 3 mapping/coverage, and MD profiles in left
   isthmus cingulate, lingual, and parahippocampal cortex. Apply only an
   independently defined imaging-QC rule; do not exclude it because of its z.
3. Repeat the regional tail diagnostic after source QC. If tails persist in
   valid data, freeze one sensitivity model (for example a robust Student-t
   likelihood or an explicitly validated non-Gaussian normative model). Do
   not choose the sensitivity using Figure 7, diagnosis, or cognition.
4. Freeze M2 as the primary mean formula and M0 as the no-thickness
   sensitivity, unless the preceding data-QC step changes the valid cohort.

## Frozen primary design

- Cohort: the audited 188 CN participants, subject order and exclusions fixed.
- Folds: reuse `fold_assignments.json` exactly at subject level.
- Outcomes: 4,687 ico4 centers x 21 depths, pial to white, no depth
  downsampling and no response imputation.
- Mean formula: `MD ~ linear age + categorical sex_code + local FreeSurfer
  cortical thickness`.
- Variance: constant Gaussian PCNToolkit BLR for the primary analysis, subject
  to the gate above.
- Preprocessing: fit age, local-thickness, and response scaling within each
  training fold only. The thickness value is the native FreeSurfer thickness
  mapped to each ico4 center and shared by that center's 21 response depths.
- Outputs: predictive mean, noise variance, posterior parameter variance,
  total predictive variance, raw-unit log density, and z for every held-out
  subject-center-depth.

## Compute layout

Fit the 21 depths for one ico4 center together so that local thickness and
fold transformations are shared and audited. Use chunked array tasks (for
example 10–25 centers per task) rather than one task per outcome. Each shard
must write atomically and contain explicit hemisphere, ico4 vertex/center ID,
DK label, and depth metadata. Do not use anonymous feature indices alone.

Before the full array, run a fixed smoke set covering both hemispheres, all
three Figure 3-selected DK regions, medial/lateral locations, and the flagged
regional neighborhoods. Validate exact source values, one-and-only-one OOF
coverage, variance decomposition, and agreement with the regional M2 results
after regional aggregation.

## Finalization checks

1. Merge shards and require exactly `188 x 4,687 x 21` primary OOF rows.
2. Fail on duplicate keys, missing predictions, nonfinite/negative variances,
   failed fits, or fold leakage.
3. Recompute the same overall/age/sex/depth/DK calibration tables with 10,000
   subject bootstraps; add spatial summaries without treating centers as
   independent inferential units.
4. Compare M2 with the prespecified M0 sensitivity and the frozen tail model,
   if required. Do not select different formulas per center or depth.
5. Record exact jobs, resources, hashes, versions, runtimes, and failures in a
   new formal Figure 6 directory. Keep this regional pilot immutable.
