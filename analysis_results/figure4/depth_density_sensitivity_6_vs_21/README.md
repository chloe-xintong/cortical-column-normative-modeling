# Figure 4 depth-density sensitivity: 6 versus 21 points

This supplementary real-profile analysis tests whether Figure 4's spatial dissimilarity curve and fitted correlation length depend on retaining 21 normalized cortical-depth samples. Six-point profiles are exact subsets of the existing Option 3 profiles at indices `[0, 4, 8, 12, 16, 20]` (depths 0.0, 0.2, ..., 1.0). Column construction and MD sampling were not rerun.

The analysis uses the same 320 registration-QC-passing participants, official ico5 anchors, within-DK K=15 pair table, 2-mm distance bins, subject masks, pair seed, fixed split assignment, and bounded exponential fit as the formal Figure 4 analysis. The 21-point representation is the existing reference representation, not ground truth.

All 320 subjects had acceptable paired full-data fits. Median correlation length was 3.010 mm for 6 points and 2.868 mm for 21 points. The paired median difference (6 minus 21) was 0.127 mm, with a 95% subject-bootstrap CI of [0.119, 0.136] mm. The subject-level 6-versus-21 correlation of fitted lengths was 0.983; the median within-subject correlation between complete 6- and 21-point distance curves was 0.999498.

The spatial increase remained present with 6 points, but the absolute dissimilarity scale shifted downward. Group-median dissimilarity in the 0–2-mm bin was 0.0417 for 6 points and 0.0522 for 21 points; the median 20–30-mm levels were 0.1249 and 0.1475. Thus, the near-to-far increases were 0.0833 and 0.0953, respectively. The qualitative spatial pattern is preserved, while the absolute dissimilarity values and fitted λ are not identical across parameterizations.

The acquisition-and-sampling null is not recomputed here. Its original 100 realizations per subject were saved only as 21-point aggregate pair metrics, not as reusable realization-level profiles. Consequently, this supplement evaluates the real-profile parameterization sensitivity of Figure 4A/C and does not constitute a 6-point reanalysis of the Figure 4B null.

Changing retained profile points does not change the 1-mm DWI acquisition resolution and does not imply independent histological layers or an optimal sampling density.

## Figure

- A: group median and IQR dissimilarity curves for 6 and 21 points.
- B: group median and IQR profile-MAE curves.
- C: paired subject-level dissimilarity difference with a 95% subject-bootstrap CI.
- D: paired subject correlation lengths; the dashed line is identity.

No sample sizes or numerical results are printed inside the figure.

The executed results notebook is `analysis_results/figure4/notebooks/figure4_depth_density_sensitivity_6_vs_21.ipynb`.
