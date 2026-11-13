# Option 2-FS ico4 brain-age baselines

The two representations are `o2fs_ico4_md` (98,427 features) and
`o2fs_ico4_md_length` (103,114 features). Both use all 196 reference CN subjects,
the exact saved subject order and masked 4,687-node ico4 vertex order from
`figure5_age_prediction/outputs`. MD features are ordered by hemisphere,
vertex, then depth; the length block follows the MD block in the combined model.

## Sources and validation

`scripts/option2_fs_baselines.py prepare` reuses existing Option 2-FS MD
outputs under `results/qc/o2_fs`. Missing outputs are generated using the
existing Figure 2 `run_vol2surf.py`, in a separate experiment source directory.
Canonical depth 0 is `projfrac=1.0` (pial); depth 20 is `projfrac=0.0` (white).
No depth reversal based on similarity to Option 3 is performed.

Length is computed at corresponding native vertices as
`norm(pial_xyz - white_xyz)`, saved as a native scalar and mapped to fsaverage
with `mri_surf2surf --surfreg sphere.reg --mapmethod nnfr`. This is the Option 2
surface-registration workflow. The mapped scalar is selected at the fixed
ico4 vertices. FreeSurfer thickness is not used.

`build` requires complete sources for every subject and checks the saved
vertex metadata, analysis mask, depth metadata, subject order and nested
5×5 fold partitions. It writes 4,704 independent source-value spot checks
covering both hemispheres, three vertices and depths 0/10/20 in every subject,
plus mapped length values. Native length values are also checked against
native endpoint distances during preparation. Matrix dimensions and all
spot checks must pass before `features/validation.json` is issued.
MD NaNs are preserved and audited by subject/hemisphere and feature; matching
NaNs pass source-value checks. Infinite MD values and nonfinite column
lengths still fail validation. Imputation remains inside training folds in
the shared Ridge pipeline, with no whole-cohort imputation.

`train` verifies the validation report and hashes of matrices, metadata,
subject order, vertex list and folds before calling the shared `run_ridge.py`.
It uses the existing fold-wise median imputation, standardization, LSQR Ridge,
and `logspace(-4, 6, 21)` alpha selection by inner validation MAE. No folds
are regenerated. SLURM `afterok` dependencies additionally gate training on
successful feature validation.

## Outputs

The experiment root is
`results/figure5_age_prediction/outputs/option2_fs_baselines/`.

- `source/`: missing MD outputs, native and fsaverage length scalars, mapping
  logs, commands and source hashes.
- `features/`: fixed-order matrices, copied metadata, source audit, spot checks
  and validation report.
- `models_lsqr/`: OOF predictions, selected alphas, inner alpha curves and
  fold metrics for both new representations.
- `comparison/`: OOF predictions with subject IDs for all four models, MAE,
  RMSE, Pearson r, R², selected alphas, subject-wise paired absolute errors,
  paired bootstrap summaries and all bootstrap draws.

The three predefined comparisons are Option 2 MD minus Option 3 MD,
Option 2 MD+length minus Option 3 MD+length, and Option 2 MD minus Option 2
MD+length. Differences are MAE(a) minus MAE(b), so negative favors a.
The bootstrap resamples paired subject OOF absolute errors 10,000 times
with seed 20261102 and reports percentile 95% intervals. This inference is
conditional on the fixed folds and fitted models; models are not refitted
within bootstrap replicates.

## Execution

The shared job wrapper is `slurm/option2_fs_baselines.sbatch`. Stages are
`prepare` (subject array index 0–195), `build`, `train` (model array index
0–1), and `compare`. Pass the stage and experiment root as positional
arguments to the wrapper. Submission identifiers and dependencies are
recorded in `config/submission_option2_fs_20261102.json`.
