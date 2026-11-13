# Cortical-depth sampling density sensitivity (6 vs 11 vs 21)

## Purpose

This supplementary analysis tests whether the principal profile and age-prediction findings depend on using 21 normalized cortical-depth samples. It is a sensitivity analysis of profile parameterization density, not a search for an optimal number of samples. The 21-point representation is treated as the existing reference representation, not as ground truth.

The 11-point profile retains indices `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]` (normalized depths 0.0, 0.1, ..., 1.0). The 6-point profile retains indices `[0, 4, 8, 12, 16, 20]` (normalized depths 0.0, 0.2, ..., 1.0). Both are exact subsets of the existing 21-point Option 3 profiles. Column construction and MD sampling were not rerun.

## Profile fidelity

For every observed subject × hemisphere × DK-region profile in the Figure 3 cohort, each reduced profile was linearly interpolated back to the 21-point grid. Raw-profile RMSE, mean-centered RMSE, mean-centered Pearson correlation, and maximum absolute difference were computed against the original 21-point profile. DK-region/hemisphere values were first summarized by their median within each subject; group summaries then use the median and IQR across subjects, so every subject has equal weight. Panel A uses the same selected regions, bilateral averaging, 320-subject cohort, and equal-subject group mean as Figure 3A.

Mean-centered reconstruction remained close to the reference representation. The across-subject median of the within-subject regional median RMSE was 3.294e-06 for 6 points and 9.241e-07 for 11 points; the corresponding median correlations were 0.999646 and 0.999969.

## Age prediction

Regional profile and ico4 MD-only representations were compared in the fixed 196-subject Figure 5 cohort. The saved nested 5 × 5 folds, LSQR Ridge implementation, `logspace(-4, 6, 21)` alpha grid, and fold-local median imputation and standardization were held fixed. Existing complete 21-point OOF predictions were reused after validating their solver, alpha grid, matrix dimensions, subject order, outcomes, and outer-fold assignments. Six- and 11-point models were rerun with the same code and folds. Every subject has exactly one OOF prediction per model.

| Representation | Depth points | MAE (years) | RMSE (years) | Pearson r | R² |
|---|---:|---:|---:|---:|---:|
| Regional profile | 6 | 5.702 | 7.392 | 0.710 | 0.504 |
| Regional profile | 11 | 5.819 | 7.553 | 0.697 | 0.482 |
| Regional profile | 21 | 5.638 | 7.314 | 0.717 | 0.515 |
| ico4 MD | 6 | 5.300 | 6.920 | 0.762 | 0.566 |
| ico4 MD | 11 | 5.289 | 6.931 | 0.761 | 0.564 |
| ico4 MD | 21 | 5.291 | 6.939 | 0.761 | 0.563 |

Positive ΔMAE means that the reduced-depth representation had greater OOF absolute error than the 21-point reference.

| Representation | Comparison | ΔMAE (years) | 95% paired-bootstrap CI |
|---|---|---:|---:|
| Regional profile | 6 vs 21 | 0.064 | [-0.017, 0.149] |
| Regional profile | 11 vs 21 | 0.181 | [0.032, 0.342] |
| ico4 MD | 6 vs 21 | 0.009 | [-0.035, 0.054] |
| ico4 MD | 11 vs 21 | -0.001 | [-0.018, 0.015] |

The ico4 MD prediction metrics were essentially unchanged across the three parameterizations. For the regional representation, 6 versus 21 points had a small ΔMAE whose interval included zero, whereas 11 versus 21 points had a small positive ΔMAE of 0.181 years (95% CI 0.032 to 0.342). This non-monotonic pattern does not identify an optimal sampling density and is reported without expanding to additional densities.

## Interpretation boundary

Changing the number of retained profile samples does not change the 1-mm DWI acquisition resolution. Adjacent depth values are highly correlated because of voxel resolution and trilinear interpolation. Therefore, this analysis evaluates the numerical parameterization density of a cortical-depth profile; it does not resolve independent histological layers. Similar performance across densities may be described as: “The principal findings were insensitive to the use of 6, 11, or 21 normalized cortical-depth samples.” It must not be interpreted as showing that any density is optimal.

## Figure

- A: Figure 3A selected-region group mean profiles; 21 points are continuous curves and reduced representations use thinner point-connected lines.
- B: subject-level medians across DK region/hemisphere profiles for mean-centered reconstruction RMSE and correlation.
- C: OOF MAE with subject-bootstrap 95% CIs for the two MD-only prediction representations.
- D: paired ΔMAE relative to 21 points with 10,000 subject-level paired-bootstrap CIs.

The figure contains no sample-size labels. Profile fidelity uses 320 subjects; age prediction uses 196 subjects.
