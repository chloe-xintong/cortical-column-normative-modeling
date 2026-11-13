# ico4 geodesic Voronoi patch-mean test

This is an exploratory Figure 5 supplement. It compares the original retained
ico4 center-column representation with a template-fixed local patch mean while
holding the 4,687 x 21 feature count, 196-subject order, nested folds,
fold-local median imputation, standardization, alpha grid, and LSQR Ridge
solver fixed.

Each valid full-resolution fsaverage vertex is assigned to its nearest retained
ico4 center by exact mesh-edge geodesic distance on `fsaverage` white. For each
subject, patch, and depth, the feature is the mean MD over finite, valid columns
in that fixed patch. Invalid columns are excluded in place rather than shifted;
zero valid coverage produces `NaN` for fold-local imputation.

Run after the primary Figure 5 LSQR `ico4_md` result exists:

```bash
bash analysis_results/figure5/supplement/patch_test/submit.sh
```

Outputs are isolated under
`results/figure5_age_prediction/outputs/supplement/patch_test/`. They include:

- fixed membership in compressed CSV and NPZ formats;
- vertex count, area, median radius, and maximum radius for every patch;
- left/right hemisphere geometry summaries and empty-patch checks;
- the full subject x patch/depth coverage matrix plus feature/subject summaries;
- direct source-value spot checks and structural QC;
- LSQR nested-CV OOF MAE and Pearson r using the primary shared folds;
- a 10,000-replicate paired subject bootstrap of
  `delta MAE = patch mean - ico4` (negative favors patch mean).

Patch construction uses only the fsaverage template, cortical analysis mask,
and retained ico4 centers. It never uses age or prediction results.
