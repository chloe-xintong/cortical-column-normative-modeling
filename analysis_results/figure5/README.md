# Figure 5 age prediction

This directory contains a standalone, reproducible pipeline for building fixed
regional/ico3/ico4/ico5 features from cleaned full-fsaverage Option 3 profiles
and evaluating them with shared nested cross-validation folds.

No MD volume is resampled. Missing profile positions remain `NaN` until the
fold-local scikit-learn imputer is fitted.

The cohort source is the fixed CN manifest produced under
`analysis_results/figure5/cohort_selection/outputs/figure5_primary_metadata_cn.csv`.

Primary outputs are written to
`/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs`.
After the local three-row dry run passes, submit with:

```bash
bash analysis_results/figure5/slurm/submit_pipeline.sh
```

The dependency order is feature build -> matrix QC -> folds -> Ridge models ->
summary. ico5 is a separate 128 GB job; the other eight unique model inputs
run as an array capped at four simultaneous tasks. `ico3_md` is computed once
and reused as both the ico-level comparison and the MD-only incremental model.

The numerical-stability rerun preserves the original `models/` outputs and
writes fixed-`solver="lsqr"` results to `models_lsqr/`. Submit it with:

```bash
bash analysis_results/figure5/slurm/submit_lsqr_stability.sh
```

The dependent analysis writes auto-vs-LSQR prediction comparisons, fold-level
alpha changes, all inner alpha-MAE curves, alpha-boundary flags, seven
pre-specified paired bootstrap comparisons with BH-FDR, and foldwise cBAG to
`stability_lsqr/`.

For cBAG, use only `lsqr_crossfit_cbag_summary.csv` and
`lsqr_crossfit_cbag_by_subject.csv`. The correction is estimated from 5-fold
cross-fitted predictions within each outer-training set and then applied to the
held-out outer-test fold. In-sample training calibration is retained only as an
overfitting diagnostic and must not be used for Figure 5 or downstream risk and
cognition analyses.

## Formal Figure 5 panels

Run `notebooks/plot_figure5.ipynb` to calculate four metrics from the 196 merged
OOF predictions. Figure 5A is the complete 11-representation MAE overview,
ordered from lowest to highest MAE. Figure 5B shows the experimental spatial
sequence (regional 21-depth profile, ico3, ico4, ico5), and Figure 5C shows the
ico4 column-mean, full-depth, and geometry-feature sequence. Each panel is
exported independently at publication scale.

The spatial-resolution family contains three paired-MAE comparisons, and the
ico4-ablation family contains four. Benjamini–Hochberg FDR correction is applied
separately within these two families. Because the families were organized after
inspection of the results, they are described as exploratory/secondary rather
than prespecified. Panels B and C show selected q values; the complete 3+4 table
is exported as `Figure5_exploratory_FDR_statistics.csv`. Pearson r, RMSE, and R²
are exported as complete 11-model Supplementary Figure S5 panels. R² is
calculated with `sklearn.metrics.r2_score`, not as Pearson r squared.

### ico4 column-mean MD comparator

`ico4_column_mean_md` collapses each retained ico4 column's 21-depth MD profile
to one mean-MD scalar, producing 4,687 features per subject. This is distinct
from both the full ico4 21-depth representation and the exploratory geodesic
Voronoi patch-mean representation. It uses the same folds, alpha grid, LSQR
Ridge implementation, and cross-fitted cBAG workflow as the primary models.

```bash
bash analysis_results/figure5/slurm/submit_ico4_column_mean_md.sh
```

## Exploratory ico4 geodesic Voronoi patch mean

The pre-specified supplement compares the original ico4 center-column model
with a fixed fsaverage white-surface geodesic Voronoi patch mean. Every valid
full-resolution vertex is assigned to its nearest retained ico4 center using
mesh-edge geodesic distance. The 21-depth mean is then computed within each
patch, so the patch model has the same number and order of features as ico4:
4,687 patches x 21 depths. Invalid subject columns are excluded in place; a
patch/depth with no valid columns remains `NaN` for fold-local imputation.

Run the independent supplement after the main Figure 5 LSQR outputs exist:

```bash
bash analysis_results/figure5/supplement/patch_test/submit.sh
```

This dependency graph builds and saves fixed membership, patch geometry and
patch/depth coverage QC, runs the shared folds and alpha grid with fixed LSQR
Ridge, and writes the pre-specified paired subject bootstrap for
`delta MAE = patch mean - ico4` (negative favors patch mean). It does not add
the exploratory model to the primary Figure 5 summary.
