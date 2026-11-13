# Figure 3: cortical-depth MD profiles and thickness-related confounding

This analysis characterizes thickness-dependent variation in Option 3 MD
profiles and quantifies depth-specific regional structure remaining after
accounting for cortical thickness. It does not compare depth counts or
interpret 21 depth samples as histological layers.

## Directory layout

- `computation/`: analysis, plotting and verification code, the Slurm wrapper,
  and the frozen submission record.
- `notebooks/`: the executed presentation notebook with embedded final figures
  and tables.
- Formal numerical outputs remain in the independent output directory recorded
  by `computation/submission_20261108.json`; reorganizing this source directory
  does not move or rewrite those results.

## Run

```bash
python analysis_results/figure3/computation/run_all_figure3.py \
  --profiles-dir /path/to/option3_subject_root \
  --subjects-dir /path/to/native_freesurfer_subjects \
  --mapping-dir /path/to/option3_subject_root \
  --qc-exclusions manifests/figure4_registration_qc_exclusions.csv \
  --fsaverage-dir /path/to/fsaverage \
  --output-dir /path/to/figure3_outputs
```

An optional `--subject-list` CSV (`subject_id` column) or text file restricts
candidate discovery. Without it, every subject directory is examined. The
Figure 5 CN list is not used. Missing bilateral inputs and known registration
QC failures are audited before inclusion. The discovery audit currently finds
325 complete profiles plus native thickness inputs. After the existing
five-subject registration exclusion list, the primary cohort also requires
bilateral cleaned-profile coverage of at least 95%; final inclusion is
confirmed by file contents, mapping and coverage checks during execution.

For SLURM, pass the Python interpreter through `PYTHON_BIN` and the source
runner path as the first positional argument to `run_figure3.sbatch`; forward
all analysis arguments after the runner path. The job requests four CPU cores,
64 GB RAM and 24 hours. No machine-specific input paths are embedded in the
analysis or plotting scripts. NumPy, pandas, SciPy, nibabel, matplotlib, and
nilearn are required. Use `--stage prepare|analyze|stability|plot` to rerun a
stage from saved artifacts. `--max-subjects` is only for a separately located
smoke test; the metadata explicitly marks it.

## Shared cohort, mask and validity

- fsaverage vertex order is checked against both hemispheres' exact DK
  annotations, stored profile metadata and column mappings.
- The common mask is the original Option 3 atlas mask, excluding
  unknown/medial-wall labels. Every panel uses this same fixed mask and each
  subject's same within-mask valid vertices. `--minimum-vertex-coverage` is an explicit
  opt-in sensitivity restriction, not the primary analysis.
- Current cleaned `column_valid` is honored. Any NaN, Inf or nonpositive
  MD at any depth invalidates the whole subject–vertex profile. Missing
  depths are never interpolated or imputed.
- Subject coverage uses the same pre-specified rule as Figure 5: the bilateral
  sum of `in_analysis_mask & column_valid` divided by the bilateral sum of
  `in_analysis_mask` must be at least 0.95. `--minimum-subject-coverage`
  defaults to 0.95. An entirely missing subject-region cell remains NaN, with its
  coverage recorded; it does not remove an otherwise valid subject.
- Native `lh.thickness` and `rh.thickness` are mapped with the exact stored
  Option 3 triangle indices and barycentric weights. Weight sums, weight
  bounds, triangle indices, finite thickness and agreement with stored
  mapped thickness are checked. A broad admissibility range `(0, 8]` mm
  is configurable with `--thickness-min/max`; the actual distribution and
  excluded counts are saved. Thickness is never replaced by column length.
- Actual column endpoints as well as metadata verify depth 0 is pial and
  depth 1 is white. Subject IDs and hemisphere names are explicitly checked.

## Panel definitions

A: Regional MD means are computed within each subject, then averaged across
subjects. The whole-cortex display shows mean ± between-subject SD; regional
A and D displays show the group mean with a two-sided 95% t confidence
interval based on subject-level values. All 68 hemisphere-DK regions and
their between-subject SD are retained in the numerical tables.
The separate panel-A exports begin with a whole-cortex profile: valid atlas
vertices are averaged within subject and depth, then subjects are weighted
equally to obtain the group mean and between-subject SD.
The three illustrated regions are selected exclusively from group-mean
FreeSurfer thickness, nearest the linear 10th/50th/90th percentiles across
all valid bilateral DK regions. Extrema are forbidden; ties choose lower
thickness then exact name. The selection is saved before MD tables are read.
An additional full-range display selects the true minimum-thickness region,
the region nearest the linear median, and the true maximum-thickness region,
again using thickness alone. These are pericalcarine, caudal middle frontal,
and temporal pole in the earlier cohort; the formal coverage-gated selection
is recomputed from the retained cohort. Missing cells are reported and are not
imputed or used to replace a thickness-selected region.
Each bilateral thickness/profile is the equal left/right mean of within-region
vertex means; subjects are then weighted equally. Missing bilateral cells are
not filled. The selected regions require complete retained-cohort coverage at all
depths. Main A and D use the same frozen regions, order, axes and colors.
A includes a compact inset of all DK region mean thickness values,
with only the three selected regions labelled. A is descriptive: no thickness
group comparisons, per-depth significance tests, or cross-region vertex pooling.
Display names use normal spaces (lateral occipital, caudal middle frontal,
frontal pole, temporal pole); CSV identifiers retain the exact FreeSurfer names. Selection
and other method notes are in the exported `*_caption.txt` files and notebook
captions, rather than a methods paragraph inside the plot. Panel titles are
`A Raw cortical-depth profiles` and `D Thickness-adjusted profiles`.

The panel-A presentation assets are five independent figures rather than a
horizontal multipanel row: whole cortex; P10/P50/P90 regional curves; their
standalone DK-thickness selection/curve key; minimum/median/maximum regional
curves; and their standalone selection/curve key. Both regional selections use
the same blue–orange–red mapping from lower to higher thickness; all lines are
solid and no black, yellow or gray curve is used.

Panel D mirrors both thickness-only selections in two independent adjusted-profile
figures: `Figure3D_01_quantile_adjusted_profiles` and
`Figure3D_02_extreme_adjusted_profiles`. A short data-driven descriptive
interpretation is saved with the figures; formal residual-region inference
remains separate.

`select_representative_regions.py` saves `figure3A_region_thickness_ranking.csv`
(all regions, ranks, actual percentiles and selections), per-subject bilateral
thickness and MD tables, and `qc/figure3AD_selected_region_depth_coverage.csv`.
Actual percentile is `100*(rank-1)/(R-1)`; target thickness quantiles use NumPy's
linear empirical quantile. The JSON selection audit records these definitions,
cohort size, number of DK regions and frozen A/D colors/order. Some nonselected
regions have fewer available bilateral cells due to existing missing hemispheres;
their actual sample counts are retained in the ranking, without imputation.

B: Each valid 21-depth profile first subtracts its own mean. Subject-normalized
first and second moments yield the global covariance with equal total weight
per subject. Normal PCA feature centering is applied to this covariance;
neither depth features nor profiles are variance-scaled or normalized.
Only one subject's full-resolution data is processed at a time. The PCA
loadings and explained variance are saved. The population covariance uses
normalized empirical subject weights. PC1 is signed for positive covariance
with white-side (depth ≥0.8) minus pial-side (depth ≤0.2) MD.

The B loading display retains only PC1 and PC2 as solid blue and orange lines.
Explained variance for PC1–PC5 is exported as a separate bar chart rather than
sharing a horizontal multipanel figure with the loadings. The group-mean
cortical score maps focus on PC1 and PC2.

The original saved PC1 score products use the earlier requested equation
`profile_shape @ loading`; their convention and sign flip remain recorded.
For the final paired B/D tracking, raw and adjusted scores instead use the
explicit frozen PCA reference
`(profile_shape - original_pca_feature_mean) @ original_loading` for both
conditions. The reference subtraction is spatially constant within a
component and therefore does not change within-subject spatial correlation or
spatial SD, but it makes the before/after projection definition exact. Mean,
sample SD, SE, mean/SD Cohen's d and valid counts from the original PC1
products remain available; Cohen's d maps are supplementary.

C: The main subject-level display compares the distributions of within-subject
spatial Spearman correlations between thickness and fixed PC1/PC2 scores.
Diamonds and error bars report the median and its 10,000-subject-bootstrap 95%
interval. Depth-specific MD–thickness regression uses exact within-subject
demeaning, equivalent to subject fixed effects. Main C shows thickness partial
R² with its subject-bootstrap interval; the signed beta curve is supplementary
because it describes direction rather than explained variance. Older binned
group-map and individual-subject displays remain optional diagnostic artifacts,
not main Figure 3 results.

The C curvature supplement uses the same fixed PC1/PC2 scores and valid
vertices. Native subject-specific FreeSurfer `lh.curv`/`rh.curv` values are
mapped with each subject's Option 3 triangle IDs and barycentric weights.
FreeSurfer signed curvature is positive in sulcal/folded-in locations and
negative in gyral/folded-out locations. The absolute metric is the weighted
mean of absolute native curvature, so signs do not cancel during mapping.
Within each subject and component, the supplement reports spatial Spearman rho
and fits `PC score ~ thickness + curvature`; standardized curvature beta and
curvature partial R² are summarized with 10,000-subject-bootstrap intervals.
Signed curvature is the primary supplementary check and absolute curvature is
secondary. Curvature is not regressed from the primary Figure 3 results.

D: Adjustment is `MD - beta_depth * (thickness - subject_mean_thickness)`;
it preserves subject-level mean MD and original MD units. All adjusted
subject profiles and regional summaries are saved. Therefore the whole-cortex
mean curve is unchanged by construction and is not repeated in D. Both A
regional selection sets are repeated with 95% CIs and A-matched axes.

The original PC1/PC2 loadings and original PCA feature-centering reference are
then frozen and used to project raw and adjusted profiles; PCA is not refit.
For each component, D reports the raw–adjusted within-subject spatial
correlation and the adjusted/raw spatial score SD, plus corresponding
group-mean cortical maps. This distinguishes preservation of spatial ordering
from preservation of score amplitude. Region partial R²
compares subject intercepts with subject plus hemisphere-DK region effects
on the observed subject × 68 region cells, using exact unbalanced-design
fixed-effects OLS. The subject intercepts and region-dummy Gram matrices are
formed using each subject's observed regions only. Missing regions are never
imputed. Bootstrap resamples subjects and
**refits thickness beta in every replicate** before adjusted regional R².
Regional partial R² is supplementary because it quantifies residual regional
MD differences rather than retention of a fixed profile-shape component.

All Figure 3 score surface maps use the same standard `RdBu_r` colors as the
final Figure 2 surface maps. Figure 2 maps nonnegative agreement values to
their stated positive ranges; Figure 3 uses symmetric ranges around zero
because PCA scores are signed. In B, PC1 and PC2 each have their own range;
all hemispheres and views within a PC share it. The same PC-specific range is
reused in D, where raw and adjusted maps therefore remain directly comparable.
The figure captions explicitly warn that
color intensity cannot compare absolute amplitude across PCs; the spatial-SD
ratios provide the quantitative amplitude comparison.

## Inference and stability

The default is 10,000 subject resamples with seed 20261102. Confidence
intervals are percentile 95% intervals. There are no vertex-wise pooled
p-values. PCA split-half repeats 100 random subject partitions using seed
20261103; both halves refit the equally weighted covariance PCA, use the
same sign convention, and compare loading correlation, group-map correlation
and explained variance ratios. Blocked disk-cache reads form split maps
without materializing all subject × vertex × depth observations in RAM.

Column-length sensitivity is explicitly secondary and separate from native
FreeSurfer thickness. Its subject correlations, depth-specific coefficients
and regional partial R² are saved; its adjusted R² curve is descriptive.

## Artifacts

- `tables/`: all requested A–D CSV files, `figure3_subjects.csv`, a `(2, V)`
  `figure3_analysis_mask.npy` (lh then rh), vertex coverage, group scatter,
  sensitivity, `figure3S_pc1_split_half_stability.csv`, and subject/group
  whole-cortex panel-A profiles.
- `surface_maps/`: bilateral MGH maps for mean/SD/SE/d/n of PC1 and thickness,
  plus raw and adjusted fixed-reference PC1/PC2 group maps.
- `figure3B_subject_pc1_scores/`: full-resolution subject scores and masks.
- `adjusted_profiles/`: bilateral thickness-adjusted subject profiles with
  vertex IDs, depth direction and validity mask.
- `curvature_supplement/`: per-subject PC1/PC2 signed/absolute curvature
  associations, multivariable effects controlling thickness, mapping QC,
  group bootstrap summaries and completion metadata.
- `qc/`: candidate exclusions, input mapping checks, native thickness hashes,
  region coverage, thickness distributions, PCA metadata, plotting manifests,
  fixed-component tracking metadata, and `figure3_completeness_audit.json`.
- `cache/`: disk-backed raw MD, mapped thickness/length and sufficient
  statistics. This avoids reloading compressed source arrays for every panel.
- `figures/`: five separate `Figure3A_01`–`Figure3A_05` assets; PC1/PC2
  loadings, separate explained-variance bars, and PC1/PC2 maps for B;
  PC1/PC2 thickness correlations and depth-specific
  partial R² for C; two adjusted regional curve sets, fixed-component maps and
  retention metrics for D; and supplementary beta, regional-effect and PCA-QC
  figures. Each required primary asset is exported as PDF/SVG/300-dpi PNG.
  Surface panels are rasterized within vector files to keep exports manageable;
  text and curves remain vectors.

All figures are regenerated from saved CSV tables or MGH maps. Repeated sample-size
labels are omitted from the primary figures because every curve uses the same
retained cohort. `plot_figure3.py` accepts only
the output root and fsaverage path, independent of the raw profiles.

`notebooks/figure3_results.ipynb` is the executed presentation notebook. It
follows A original curves → B shape modes → C thickness association → D fixed-
component retention, followed by the signed beta, regional partial R² and PC1
split-half stability as supplements. The embedded output is regenerated after
the completeness audit.
It reads `computation/submission_20261108.json` for the output location; override this
with the `FIGURE3_OUTPUT_DIR` environment variable. It refuses to display an
incomplete analysis. Resume an existing prepared cohort using `--stage resume`;
this runs analysis, stability and plots without repeating input preparation.

## Verification

### Paired PC1 versus PC2 thickness associations

Run `analysis_results/figure3/computation/compare_pc1_pc2_thickness.py --output-dir /path/to/figure3_outputs` to
compare within-subject spatial Spearman correlations on identical valid
vertices. Existing PC1 scores are retained; absent PC2 scores are reconstructed
with the saved PC2 loading and cached MD, without refitting PCA. Subject×vertex
score matrices are saved in `pc1_pc2_thickness/pc1_scores.npy` and `pc2_scores.npy`.

`subject_thickness_pc1_pc2_correlations.csv` contains signed and absolute
correlations and paired absolute differences. `component_correlation_summary.csv`
reports medians, quartiles, IQR widths, and 10,000-subject-bootstrap 95% intervals
for median rho. `paired_absolute_correlation_comparison.csv` reports the median
of paired differences, its bootstrap interval and two-sided paired Wilcoxon
signed-rank test (`zero_method=wilcox`, `method=auto`). This is not the difference
of marginal medians. The fixed seed is 20261103. Inference conditions on the
fitted PCA and masks; no vertex-level p-values are used.

The paired signed/absolute violin + boxplots with subject-pair lines are saved
as `Figure3S_thickness_PC1_PC2_paired.{pdf,svg,png}` as optional supplementary
diagnostics. The final notebook uses the component-specific subject
distributions and confidence intervals directly in main C.

`analysis_results/figure3/computation/plot_pc2_vs_thickness.py --output-dir /path/to/figure3_outputs` adds
`Figure3S_PC2_vs_thickness.{pdf,svg,png}`: the subject-level rho distribution
and individual PC2–thickness scatterplots from subjects nearest the 25th,
50th and 75th rho percentiles. Example panels show 2,000 sampled vertices and
40 within-subject equal-count bin means; annotations use all valid vertices.
Selection metadata and bin values are saved in `pc1_pc2_thickness/`.

```bash
python -m unittest discover -s analysis_results/figure3/computation -p 'test_*.py' -v
```

Tests verify profile centering, equal-subject covariance, PC1 sign, fixed-
effect thickness slopes and regional partial R² against explicit dummy-
variable least squares, including repeated subjects in bootstrap samples,
incomplete region grids and bootstrap-specific adjusted outcomes.

Run `audit_figure3_completeness.py` after exporting and executing the notebook.
It checks the frozen N=313, the 0.95 subject-coverage threshold, A/D interval
contracts and matched selections, all 42 primary figure files, eight fixed-
component surface maps, required tables, and notebook execution status.
