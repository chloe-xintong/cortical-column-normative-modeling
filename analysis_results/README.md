# Figure analyses and results

This directory groups figure-specific analysis code, notebooks, job scripts,
and repository-local outputs. The active organization is:

| Directory | Purpose | Main notebook |
| --- | --- | --- |
| `figure2/` | Mapping validation: O2/O3 agreement, curvature dependence, registration QC | `figure2/figure2_AB_method_comparison.ipynb` |
| `figure4/` | Spatial profile similarity, sampling null, correlation length, matched boundaries | `figure4/notebooks/figure4_formal_ico5_k15.ipynb` |
| `figure5/` | Age-prediction representation comparisons and feature ablations | `figure5/notebooks/plot_figure5.ipynb` |

The cross-figure participant flow, every subject-level exclusion, and the
QIAL ambiguous/LOW visit audit are centralized in
`cohort_audit/README.md` and its generated `outputs/` tables.

Figure 3 has not yet been implemented.

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
