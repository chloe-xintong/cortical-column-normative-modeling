# Figure analyses and results

Paper figures follow the shared [figure style standard](FIGURE_STYLE_STANDARD.md),
including semantic colors, slender subject-level violins, and content-aware
panel dimensions rather than default square or equal-width layouts.

This directory groups figure-specific analysis code, notebooks, job scripts,
and repository-local outputs. The active organization is:

| Directory | Purpose | Main notebook |
| --- | --- | --- |
| `figure2/` | Mapping validation: O2/O3 agreement, curvature dependence, registration QC | `figure2/figure2_AB_method_comparison.ipynb` |
| `figure3/` | Cortical-depth MD shape, FreeSurfer thickness confounding, and adjusted regional structure | `figure3/notebooks/figure3_results.ipynb` |
| `figure4/` | Spatial profile similarity, sampling null, correlation length, matched boundaries | `figure4/notebooks/figure4_formal_ico5_k15.ipynb` |
| `figure5/` | Age-prediction representation comparisons and feature ablations | `figure5/notebooks/plot_figure5.ipynb` |

The cross-figure participant flow, every subject-level exclusion, and the
QIAL ambiguous/LOW visit audit are centralized in
`cohort_audit/README.md` and its generated `outputs/` tables.

Figure 3 has a standalone command-line runner; see `figure3/README.md` for
its shared-cohort QC, statistical definitions, and reproducible figure exports.

## Data and execution

Figure 4's active registration-QC-passing cohort contains 320 participants.
Figure 5 uses the same 196 cognitively normal participants as before the move.
The move does not change any cohort, metric, model output, or plot content.

Shared plotting utilities remain in `qc/publication_style.py`; manifests remain
in `manifests/`. External datasets and results retain their existing locations
under `/mnt/newStor/paros/paros_WORK/xintong/results/`. Figure 2's directory name
is now `figure2`; its external historical result directory still uses
`qc/template_registration_comparison/`.

For compatibility with previous commands and serialized paths, the old
repository directories are symbolic links to the new locations:

- `qc/template_registration_comparison` → `analysis_results/figure2`
- `qc/figure4` → `analysis_results/figure4`
- `qc/figure5` → `analysis_results/figure5`

Use the new locations for ongoing work. `migration_report.json` records the
moved directories and updated files. Existing notebook outputs may display the
old paths as historical execution text; the executable source uses the new
repository locations.

Figure 2's full QC-passed cohort extension is documented in
`figure2/FULL_COHORT.md` and displayed in `figure2/figure2_full_cohort.ipynb`.
It saves maximum-available pairwise and common-subject analyses separately,
while retaining the original 50-subject notebook and results.

The active figures use fixed scientific color roles from
`qc/publication_style.py`, rather than a six-color rotation. Deep blue is the
default single-series color with faint interval bands. Red emphasizes the
fixed final `ico4_md_length` model or the real data in the primary real-versus-null comparison;
it never encodes statistical significance. Named methods/models keep their
shared colors across panels and notebooks. Peer curves use blue/orange with
distinct line styles and markers. Figure 3's three selected regions retain the
same colors, line styles and markers in A and D. Background grids are disabled;
lines are about 1.8 pt, markers about 3 pt, interval alpha 0.12, and left/bottom
axes are thin dark gray. Text stays neutral. Heatmaps
have independent numeric encodings: `project_blue` is light-to-dark blue for
nonnegative magnitudes, and `project_diverging` has a neutral `#F7F7F7` zero,
negative blue and positive red. Signed ranges are symmetric; correlation
matrices use −1 to 1. Missing cells are gray. Comparable panels share limits;
display clipping is documented and never changes the underlying data.

`figure_style_qc/semantic_palette_policy.json` records the complete role mapping.
`figure_style_qc/semantic_palette_accessibility.pdf` shows the redundant encodings
in ordinary color, grayscale, and protanopia/deuteranopia simulations. Regenerate
this diagnostic with `check_semantic_palette.py`; it is a visual review aid,
not a blanket accessibility certification.

`figure_style_qc/heatmap_rules.json` records the heatmap rules, including the
fixed across-minus-within subtraction and matrix-layout conventions. Regenerate
the colorbar review sheet and scientific-scale checks with
`check_heatmap_rules.py`. Current Figure 2 agreement maps retain shared 0.8–1
display limits; Figure 3 signed maps share a pooled bilateral 98th-percentile
absolute-value display limit per metric, with exact values saved alongside
the Figure 3 QC outputs.

Orange (`#FC8C5A`) also encodes a derived Real-versus-Null/reference result,
such as the Figure 4B residual local-excess curve. Shared `DERIVED_STYLE` and
`DERIVED_BAND` specify 1.9-pt lines, 3-pt circles, and orange interval alpha
0.16; the ordinary 1.8-pt/0.12 defaults still apply elsewhere. This categorical
role does not override the signed/nonnegative rules for heatmaps.

Current plots omit panel-letter labels (A/B/C/D and supplementary variants).
Descriptive titles and stable analysis/output filenames are retained. The shared
`panel_label` entry point is disabled, and export finalization removes title prefixes.
