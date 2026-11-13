# Figure 2 full-cohort comparison

`full_cohort.py` extends A/B without replacing the historical 50-subject
outputs. Subject discovery scans complete bilateral production Option 3
profiles and excludes the five known registration failures. It does not
apply the Figure 5 CN filter. The current inventory contains 320 subjects.

The old O2-Column outputs cover 50 subjects; comparable archived native
inputs can generate O2-Column for 293. The remaining 27 do not have complete
comparable native inputs and are explicitly recorded as method-specific
unavailability, not newly failed subjects. They remain in O2-FS versus O3.
O2-FS reuses 215 subjects' existing outputs (including the Figure 5 expansion)
and generates the missing 105 with the existing `run_vol2surf.py`.

Both result sets are saved:

- `max_available/`: maximal subjects for each comparison, expected N=293
  for O2-Column versus O3 and N=320 for O2-FS versus O3.
- `common_cohort/`: the intersection, expected N=293, for sample-matched
  between-method interpretation. The final counts are calculated from
  completed bilateral results rather than asserted in figure text.

All method definitions are imported from `figure2_method_comparison.py`:
21 pial-to-white depths, fsaverage cortex intersected with valid DK labels,
declared and finite profile validity, pairwise masks, depth-wise profile
correlation, MAE, symmetric normalized MAE and shared barycentric absolute
curvature. No new positivity or coverage threshold is silently added.
O2-FS manifests additionally verify `projfrac=1-depth` and exact subject,
hemisphere, template and interpolation metadata. The original native-column
reader and interpolation routine generate missing O2-Column outputs, so
native profile cleaning is not reconstructed from incompatible legacy files.

The D0007 pilot reproduces both original comparisons' masks and all vertex
metrics across 327,684 rows to absolute tolerance 1e-12. No PCA or model fitting
is involved. Subsequent unexpected file/format errors stop the dependent
pipeline, rather than silently dropping subjects.

## Execution

Run `full_cohort.py inventory|subject|aggregate` with `--results-root`,
`--subjects-dir`, `--fsaverage-dir`, `--qc-exclusions` and `--output-dir`.
The subject stage additionally takes `--index`, indexing `subjects.csv`.
Optional `--native-mapping-root` and `--freesurfer-home` override derived paths.
`full_cohort.sbatch` accepts the runner path, stage and forwarded arguments;
it caps BLAS threading at one and uses the interpreter from `PYTHON_BIN`.
Submission IDs and the output location are in `full_cohort_submission.json`.

Each subject has its own source provenance, complete marker, coverage tables,
subject summaries, curvature bins, correlation maps and per-vertex Parquet
files under `subjects/<id>/`. These Parquet files preserve the original vertex
metrics schema but are partitioned by subject to avoid a giant in-memory table.

Aggregation saves equal-subject median/IQR summaries and percentile 95% CIs
from 10,000 subject resamples (seed 20261103), plus subject-level curvature
Spearman summaries and two-sided Wilcoxon tests. Curvature bins are defined
within each subject after pooling its two hemispheres. No vertex-level
p-values are used as cohort inference.

Surface maps preserve Fisher-mean correlation, median, quartiles and valid
subject counts at each vertex. Rendering uses red–white–blue (RdBu_r), with
r=0.98–1.00 for O2-Column and r=0.8–1.00 for O2-FS, and requires 80% subject coverage. Separate maps and agreement/curvature
plots are exported for each analysis set in PNG/PDF/SVG. The new
`figure2_full_cohort.ipynb` reads precomputed subject summaries and static
figures; it never loads all vertices for all subjects simultaneously.
Sample counts remain in the notebook tables rather than inside the figures.
Curvature dependence is exported separately for O2-Column versus O3 and
O2-FS versus O3, each with its own bin-profile and subject-correlation panels.
Agreement distributions are also exported separately for each method pair.
Violin plots contain opaque boxes in a darker shade of the same color and
orange median lines, without overlaid subject scatter points.
The historical notebook remains unchanged. Existing registration QC panels
already use the larger registration cohort and are not refitted by this A/B
extension; their five exclusions are honored here.

Each comparison has one canonical surface plot, using the requested red-blue
colors and method-specific limits stated on the colorbar. The O2-Column
0.8–1 version is replaced, and the separate detail copy is removed. All
hemispheres/views of one comparison share the same limits; use numeric r
values and subject summaries, rather than hue across differently scaled
methods. Matching `*_display.json` files record actual limits and saturation
counts; source data and masks are unchanged.
