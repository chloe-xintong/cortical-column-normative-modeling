# Figure 6 analysis

This is the single entry point for all Figure 6 results. The former
`analysis_results/figure6_normative_model/` tree has been merged here; there is no longer a
separate normative-model result root.

## Frozen primary analysis

- Cohort: 186 CN participants.
- Source-image exclusions: D0271 and D0285, both
  `source_image_structural_abnormality`.
- Original five folds retained: 38, 38, 37, 36, and 37 held-out participants.
- Primary model: PCNToolkit 1.3.0 Warped M2 with linear age, categorical sex code, and local
  FreeSurfer cortical thickness.
- Formal spatial analysis: 4,687 bilateral ico4 nodes × 21 depths.

## Directory guide

- `regional_warped_m2_186/`: common-cohort regional model selection, Gaussian-versus-warped
  comparison, calibration, and six-versus-21-depth summaries.
- `ico4_warped_m2_186/`: formal ico4 OOF predictions, calibration, full-cohort models,
  original-scale centile predictions, and Figure 6A/6B outputs.
- `notebooks/`: all Figure 6 notebooks, including model selection and age-conditioned analysis.
- `notebook_figures/` and `plotting_inputs/`: exported statistical figures and their plotting
  tables.
- `qc/`: source-image review material, including the two confirmed exclusions.
- `scripts/` and `slurm/`: reproducible analysis and scheduling code.
- top-level regional tables, `models/`, `shards/`, `figure6.png`, and `figure6.pdf`: the original
  188-person regional pilot retained as historical output. Its original documentation is
  `README_regional_pilot_188.md`.

## Current completion state

The combined 21-depth OOF analysis contains 18,100,551 valid predictions with no duplicate,
missing, or extra valid keys, no nonfinite final predictions, and no Gaussian fallback. Overall
mean z is -0.020500, SD(z) is 1.003680, and observed 50%, 80%, and 90% coverage is 49.77%,
80.30%, and 90.38%. Bootstrap intervals resample participants.

All 98,427 full-cohort node-by-depth models completed, and every serialized model passed its
restoration check. Original-MD-scale 5th, 50th, and 95th percentile predictions are available
for all nodes and depths over ages 45–76 for both observed sex-code levels.

Figure 6C is complete under the frozen standardized estimand: the original-scale predicted-median
mean at age 76 minus age 45, using the same 186 participants' empirical sex and node-local
thickness at both ages, scaled by 10/31 to a per-decade quantity. The three seven-depth bands are
0--6, 7--13, and 14--20. Results, support audits, surface files, and PNG/PDF maps are in
`ico4_warped_m2_186/figure6c/`. Of 4,687 fixed-mask nodes, 4,517 have complete thickness for all
186 participants; the other 170 are explicitly unavailable rather than imputed or reweighted.

## Final notebook set

- `notebooks/figure6_DK_region_model_selection_186.ipynb`: DK-region model selection and
  Gaussian-versus-warped calibration evidence.
- `notebooks/figure6_ico4_final_model_results_186.ipynb`: final ico4 OOF calibration,
  full-cohort model validation, conditional centile curves, and age-effect preparation.

These are the only retained Figure 6 notebooks. Historical 188-person, superseded 187-person,
and individual M0--M3 intermediate notebooks have been removed.

## Reports

- `summary/figure6_overall_summary.pdf` (also SVG and 300-dpi PNG): integrated four-panel Figure 6
  overview.
- `summary/figure6_numerical_summary.csv` and `.md`: machine-readable and formatted numerical
  summaries spanning DK-region model selection and final ico4 performance.
- `ico4_warped_m2_186/age_conditioned_completion_report.md`
- `figure6_186_analysis_summary.md`

The source meaning of sex codes 1 and 2 remains unverified, so current outputs use the neutral
labels `Sex code 1` and `Sex code 2`.
