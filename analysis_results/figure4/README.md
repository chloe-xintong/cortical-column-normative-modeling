# Figure 4 spatial-scale analysis

The pair builder uses mesh-edge geodesic distances on the fsaverage white
surface. It never creates a full vertex-by-vertex distance matrix. A fixed,
spatially distributed anchor set and fixed random seed produce one reusable
pair table for every subject.

The primary profile metric is `1 - Pearson r` across the complete 21-depth O3
MD profile. Pair values are summarized first within anchor and distance bin,
then within subject, and finally across subjects.

- Figure 4A uses all valid pair types.
- Figure 4B uses an acquisition-and-sampling null as the primary model:
  Gaussian fields are generated on each subject's native MD grid, a
  pre-specified 0/1/2/3-mm FWHM PSF family is applied, and the fields are
  sampled through the real O3 coordinates with trilinear interpolation and
  the exact real-data mask. The grid shape, affine, voxel size, and complete
  acquired field of view are inherited from the subject's MD image; the O3
  coordinates already encode the actual T1-to-DWI transform. Pooled bilateral
  subject-level affine recoloring
  then restores the 21-depth mean and covariance before the unchanged Figure
  4A anchor/subject aggregation. The full PSF family is reported and is not
  selected by agreement with the observed D(r) curve. Subject and realization
  seeds are held fixed across PSF conditions, so the PSF-family comparison is
  paired on the underlying Gaussian fields.
  The main contrast is computed within subject and realization as
  `D_null(r) - D_real(r)`. A secondary local-excess contrast uses the
  20--30-mm median as the far-distance reference:
  `E(r) = D_far - D(r) = corr(r) - corr_far`, followed by
  `E_real(r) - E_null(r)`.
  The local-excess curve and its subject-level 0--10-mm AUC are the primary
  Figure 4B display. Absolute `D_null - D_real`, which mixes local organization
  with platform offsets, is supplementary.
  A sparse 20--100-mm real-data check uses 256 uniform anchors per hemisphere
  and at most 10 neighbors per broad distance bin. In the 325-subject cohort,
  the 60--100 minus 20--30-mm median increase is small (0.00581; bootstrap 95%
  CI 0.00490--0.00688) and remains below the prespecified 0.01 material-change
  margin. Thus 20--30 mm is treated as an approximate, not perfectly flat,
  reference plateau.
  A supplementary common-shape check summarizes each valid column's
  correlation with its subject-specific mean laminar profile for real and
  acquisition-null profiles, with removal of the two endpoint depths as a
  sensitivity analysis.
- The supplementary surface surrogate independently rotates every
  PCA score map, thereby disrupting cross-component colocalization and intact
  profile neighborhoods. Scores are harmonically extended to the complete
  fsaverage sphere before rotation and the subject-specific valid mask is
  applied afterward; nearest-unused medial-wall reassignment is not used.
  Subject/hemisphere-specific covariance recoloring restores zero-lag depth
  covariance. Final depth-map ACF must be validated by distance range before
  any expanded run. The older shared-rotation pilot is diagnostic only and
  is invalid for profile-similarity-versus-distance inference because it
  nearly rotates the complete profile field intact.
  The interpolation-only sensitivity null uses `PSF_FWHM_MM=0`; any explicit
  voxel-space PSF null requires an externally justified positive FWHM.
  Absolute profile MAE is also summarized over the same 0--30-mm bins. The
  real curve is read from each subject's `figure4_real_subject_bins.csv`.
  For the FWHM=0 null, `median_profile_mae` is first median-aggregated across
  realizations within subject and distance bin; both curves are then shown as
  the across-subject median and IQR at the fixed 1, 3, ..., 29-mm bin centers.
  The real-versus-null panel uses only subjects available in both conditions.
  No MAE local-excess reference is imposed at this stage.
  A separate whole-profile spatial-permutation sensitivity null shuffles
  intact 21-depth rows bijectively among the final valid vertices within each
  hemisphere. It exactly preserves the observed profile multiset and all
  within-profile depth structure, but destroys profile-to-vertex assignment
  and local surface organization. Because these profiles have already passed
  through O3 sampling, this null is evaluated directly on the surface and is
  not rasterized back into DWI space or sampled a second time. It answers a
  different question from the acquisition-and-sampling null and must not be
  interpreted as a voxel-sharing model.
- Figure 4C fits a bounded exponential correlation-length model and reports
  anchor split-half reliability.
- The frozen Figure 4D analysis uses the same official ico5 anchors,
  full-resolution fsaverage white-mesh geodesic distances, cortex/annotation
  mask, and DK labels as formal Figure 4A--C. It builds a reusable dense
  0--10-mm candidate neighbourhood once. For each subject, finite,
  nonconstant profile validity is applied before matching. Each eligible
  anchor × 2-mm bin has both same-DK and different-DK candidates. Up to 15
  unique within/across candidates are paired deterministically by the smallest
  absolute geodesic-distance difference, with a 0.1-mm caliper. The frozen
  wide matched table is then used for all profile statistics.

  The actual frozen A--C bin implementation is right-closed:
  `(0,2], (2,4], ..., (8,10]`; Figure 4D reproduces it exactly. Profile
  effects are reduced as matched-pair mean -> anchor-bin mean -> bilateral
  subject-region-bin mean -> equal-region subject-bin mean -> group
  median/IQR. The primary effect is
  `delta_D = (1-r_across) - (1-r_within)`. No Gaussian null is used for D.

  Scripts and stages:

  1. `build_boundary_candidate_neighborhoods.py`: one reusable dense geometry
     table per hemisphere.
  2. `generate_boundary_matched_pairs.py`: subject-specific validity filtering,
     deterministic matching, frozen pair parquet, and strict QC assertions.
  3. `compute_boundary_matched_effect.py`: pair, anchor, region, and subject
     profile-effect tables; no intermediate level is discarded.
  4. `summarize_boundary_matched_effect.py`: group curve, 0--10-mm subject
     scalar, bootstrap CI, pair QC, and publication outputs.

  Example CLI for one subject after the geometry cache exists:

  ```bash
  python analysis_results/figure4/scripts/generate_boundary_matched_pairs.py \
    --manifest manifests/figure4_valid_325_subjects.csv --index 0 \
    --candidate-dir /path/to/boundary_matched/geometry \
    --output-root /path/to/boundary_matched/pairs --k 15 --caliper-mm 0.1
  python analysis_results/figure4/scripts/compute_boundary_matched_effect.py \
    --manifest manifests/figure4_valid_325_subjects.csv --index 0 \
    --pair-root /path/to/boundary_matched/pairs \
    --output-root /path/to/boundary_matched/effects
  ```

  Validation order is mandatory: unit tests, one-subject pilot, five-subject
  consistency pilot, then the 325-subject arrays. A--C pair files are never
  overwritten.

The notebook under `notebooks/` only reads cached tables. The full-cohort SLURM
workflow intentionally does not execute it; the investigator runs it manually.

## Active registration- and coverage-QC-passing cohort

The primary Figure 4 analysis excludes the five visually confirmed
T1–DWI registration failures D0128, D0163, D0248, D0296 and D0388 from the
historical 325-participant cohort, then applies the same bilateral cleaned-profile
coverage rule as Figure 3 and Figure 5: `sum(in_analysis_mask & column_valid) /
sum(in_analysis_mask) >= 0.95`. Use
`manifests/figure4_coverage95_pass_313_subjects.csv`; the exact coverage audit is
saved under `analysis_results/figure4/outputs/coverage95_pass_313/`.

`notebooks/figure4_formal_ico5_k15.ipynb` displays the reaggregated coverage-gated
A–D results under `analysis_results/figure4/outputs/coverage95_pass_313/`.
Historical outputs are not used as a sensitivity analysis in the active notebook.
Group curves, bootstrap intervals and
split-half reliability are recalculated using cached subject-level outputs:

```bash
python analysis_results/figure4/scripts/refresh_registration_qc_cohort.py
python analysis_results/figure4/scripts/write_registration_qc_notebook.py
```

No profile sampling, null realization generation or subject fit is repeated.
All four primary panels are checked against the exact retained participant set.
Other historical region/supplement notebooks retain their original cohorts and
must not be presented as the primary analysis. Figure 5 remains the
same 196 CN participants, already excluding all five failed registrations.
