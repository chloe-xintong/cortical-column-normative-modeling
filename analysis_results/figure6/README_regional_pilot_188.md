# Historical Figure 6 regional pilot (188 participants)

This file documents the original pilot retained for provenance. It is not the current frozen
186-participant Figure 6 analysis; use `README.md` as the unified Figure 6 entry point.

The legacy top-level tables, `models/`, and `shards/` documented below contain the completed
**regional × 21-depth pilot only**. That historical pilot did not run or submit the full
ico4 × 21 analysis and did not modify Figure 3–5 results.

## Cohort and inputs

The starting point is the final Figure 5 `subject_order.csv` (196 CN
participants). Applying `40 <= age < 80` yields 188 participants, age 41–78:
63 with `sex_code_1` and 125 with `sex_code_2`. The one code-9 participant in
the 196-person list is D0343 (age 30) and is therefore excluded by age before
modeling. The sex-code mapping remains unverified; no male/female labels are
used.

## Notebook and model layout

The historical 188-participant notebooks were removed after consolidation. The tables below are
retained for provenance; the frozen 186-participant model-selection notebook is
`notebooks/figure6_DK_region_model_selection_186.ipynb`.

1. `models/01_M0_linear_age_no_thickness/` — establish the linear/no-thickness baseline;
2. `models/02_M1_spline_age_no_thickness/` — compare M1 with M0 to choose the age form;
3. `models/03_M2_linear_age_with_thickness/` — test thickness under the selected linear age form;
4. `models/04_M3_spline_age_with_thickness/` — cross-check the age-form decision after thickness adjustment.

Each model directory retains its exact model specification, README, and model-filtered calibration, age/depth, age-effect,
comparison, and fold-diagnostic tables. The large OOF prediction table stays
centralized to avoid duplicating it four times. `model_order.json` records the
ordering and role of every model.

All included participants have all 1,428 response values (68 separate
hemisphere x DK regions x 21 pial-to-white depths) and all 68 regional native
FreeSurfer cortical-thickness values. No response or thickness value was
imputed. The regional definitions and ordering exactly match the Figure 5
matrices and the Figure 3 DK inputs. Thickness is native FreeSurfer thickness
mapped through the stored Option 3 correspondence, not column length.

New Figure 6 folds were used because the age-filtered Figure 5 folds contained
an uneven joint age-bin x sex-code cell. The new fixed-seed folds are jointly
stratified:

| Fold | 40–49 | 50–59 | 60–69 | 70–79 | sex_code_1 | sex_code_2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 16 | 9 | 8 | 13 | 25 |
| 1 | 7 | 16 | 8 | 7 | 12 | 26 |
| 2 | 7 | 16 | 8 | 7 | 12 | 26 |
| 3 | 6 | 15 | 9 | 7 | 13 | 24 |
| 4 | 5 | 16 | 10 | 6 | 13 | 24 |

## Model implementation

Each hemisphere x DK region is modeled separately, with its 21 depths sharing
the same regional thickness covariate. Sex is categorical: code 1 is the
reference and `sex_code_2` is a 0/1 indicator. The four predefined Gaussian,
constant-residual-scale models are M0 linear/no thickness, M1 natural cubic
spline (`df=3`)/no thickness, M2 linear/FreeSurfer thickness, and M3 spline/
FreeSurfer thickness.

The implementation uses official PCNToolkit BLR 1.3.0. PCNToolkit 1.4.0 was
current at execution time but requires Python >=3.12; the project interpreter
is Python 3.11, so 1.3.0 was pinned in an isolated environment rather than
changing a shared environment. Recreate it with:

```bash
conda env create -f environment.yml
```

The natural spline is Patsy `cr(age, df=3, constraints='center')`. Its knots,
constraints, and all age/thickness scaling parameters are fitted only on each
training fold and reused unchanged for held-out subjects. The saved fold
transformations are in `shards/fold_transforms_all.json`.

PCNToolkit's predictive variance is used directly:

```text
predictive variance = 1 / beta + diag(Phi A^-1 Phi^T)
z = (observed MD - predictive mean) / sqrt(predictive variance)
```

The first term is observation/noise variance and the second is posterior
parameter uncertainty. Both are transformed back to MD units. NLPD is the
negative Gaussian log density in raw MD units; it can be negative because a
continuous density in small MD units can exceed one. Lower NLPD is better.

## Main OOF results

All 1,360 region x model x fold tasks completed. Every participant has exactly
one OOF prediction for every model, region, and depth (1,073,856 rows total),
with no failures, missing predictions, or duplicate keys. CIs use 10,000
subject-level bootstrap resamples (seed 20261107), never feature-level
resampling.

| Model | mean z | SD z | 50% coverage | 80% coverage | 90% coverage | mean NLPD |
|---|---:|---:|---:|---:|---:|---:|
| M0 | -0.0023 | 1.0184 | 0.5469 | 0.8308 | 0.9142 | -8.5309 |
| M1 | -0.0047 | 1.0177 | 0.5473 | 0.8312 | 0.9146 | -8.5306 |
| M2 | -0.0040 | 1.0177 | 0.5439 | 0.8283 | 0.9125 | -8.5524 |
| M3 | -0.0062 | 1.0150 | 0.5432 | 0.8291 | 0.9132 | -8.5550 |

### Uniform formula selection

Linear age is selected. M1 did not improve M0 NLPD (difference +0.00033,
95% bootstrap CI -0.00766 to 0.00840) or calibration. M3 had only a small,
uncertain NLPD advantage over M2 (-0.00259, CI -0.01093 to 0.00574), without
consistent calibration improvement or protection from edge-age behavior.

Thickness is retained in the uniform primary mean formula. Relative to M0,
M2 improved mean held-out NLPD by -0.02151 (CI -0.04182 to -0.00804), moved all
three central coverage estimates slightly toward their targets, improved mean
NLPD in 44/68 regions and at every depth, and still improved NLPD after
removing endpoints (-0.02291) or D0271 (-0.01337). M0/M2 age effects correlate
at 0.988 but their median absolute difference is about 10% of the median
absolute M0 age effect; individual OOF z-scores correlate at 0.976. These
changes are larger than numerical jitter.

The recommended uniform mean formula is therefore:

```text
MD ~ linear age + categorical sex_code + local FreeSurfer cortical thickness
```

M0 remains the prespecified no-thickness sensitivity model.

## Gaussian calibration finding and ico4 gate

M2 central coverage is close to nominal, but the Gaussian assumption is **not
fully supported**. Overall skewness is 0.90 and excess kurtosis is 12.93. The
problem is localized mainly to left isthmus cingulate, left parahippocampal,
and left lingual profiles, particularly middle-to-white depths. The worst M2
outcomes are left isthmus cingulate depths 0.75–0.90 (SD z about 1.81–1.83,
skew about 10, excess kurtosis above 120). D0271 supplies the largest extreme
profiles, but no post-hoc subject exclusion was made.

The full ico4 run is therefore gated on (1) source-profile/mapping QC for
D0271 and the flagged region-depths without outcome-based exclusion, and (2)
a frozen robust or non-Gaussian sensitivity specification if the source data
are valid. This pilot does not auto-switch to GAMLSS and submits no ico4 job.
See `ICO4_EXECUTION_PLAN.md`.

## Reproduction

The single SLURM driver is `slurm/run_regional_pilot.sbatch`; it runs input
preparation, all four OOF models, 10,000 bootstraps, diagnostics, the selected
centile fit, plotting, and notebook generation. Scripts accept explicit input
paths. Key products are:

- cohort/fold audits: `cohort_audit.csv`, `cohort_summary.csv`,
  `fold_assignments.json`, `fold_balance.csv`, `missingness_audit.csv`;
- OOF data: `oof_predictions_long.parquet`, `oof_zscores_long.parquet`;
- calibration: `calibration_by_*.csv`, including region-depth diagnostics;
- comparisons: `model_comparison_bootstrap.csv`,
  `thickness_sensitivity_summary.csv`, `model_selection_decision.json`;
- curves/figure: `selected_region_centile_data.csv`, `figure6.png`, and
  `figure6.pdf`;
- notebooks: historical pilot notebooks removed; use the frozen 186-participant notebooks listed
  in the unified `README.md`;
- provenance: `regional_input_manifest.json`, `model_specifications.json`,
  `run_metadata.json`, and `shards/`.

Region-level CSVs contain point estimates; bootstrap CIs are supplied for
overall, age-group, sex-code, and depth-band summaries and every predefined
model contrast. This avoids implying feature independence while keeping the
large region table tractable.
