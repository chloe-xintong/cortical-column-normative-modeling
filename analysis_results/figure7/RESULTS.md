# Figure 7 analysis record

- Selected model: Figure 6 ico4 Warped M2, zero-based `outer_fold=1`.
- Model refit: no.
- Mixed-fold z-scores: no.
- Held-out CN: saved Fold 1 OOF z-scores.
- Target participants: same recovered fixed Fold 1 prediction function.
- Analysis cohort: N=111; Figure 7A N=63; Figure 7B N=109.
- Primary burden: participant fraction of valid positions with `|z|>2`.
- Statistical unit: participant.
- Interpretation: fixed internal held-out analysis.

Primary results and their interpretation are reported in `METHODS_RESULTS.md`.
The executed notebook is `notebooks/figure7_fixed_fold1_analysis.ipynb`.

## Main files

- `inputs/cohort_flow_coverage95_n313.csv`: all formal QC participants and
  explicit inclusion/exclusion reason.
- `inputs/final_analysis_roster.csv`: frozen N=111 Figure 7 roster.
- `outputs/participant_deviation_burden.csv`: valid positions and positive,
  negative, and total burden per participant.
- `outputs/figure7a_primary_results.csv`: adjusted MCI–CN and AD–CN contrasts.
- `outputs/figure7b_primary_and_diagnosis_sensitivity.csv`: cognition models.
- `outputs/robustness_results.csv`: coverage≥98% and directional checks.
- `outputs/covariate_balance_by_group.csv`: age, sex, coverage, and overlap.
- `outputs/model_recovery_validation.json`: fixed Fold 1 recovery audit.
- `outputs/all_normal_oof_burden_cognition.csv`: all 186 frozen CN, each with
  the burden from that participant's single held-out fold.
- `outputs/all_normal_oof_cognition_results.csv`: all-Normal primary and
  fold-fixed-effect sensitivity regressions.
- `outputs/all_normal_oof_validation.json`: five-fold OOF provenance audit.
- `figures/Figure7A_diagnostic_group_comparison.{png,pdf}`.
- `figures/Figure7A_adjusted_group_contrasts.{png,pdf}`.
- `figures/Figure7B_raw_global_cognition_association.{png,pdf}`.
- `figures/Figure7B_adjusted_global_cognition_association.{png,pdf}`.
- `figures/Figure7B_diagnosis_adjusted_global_cognition_association.{png,pdf}`.
- `figures/Figure7B_sensitivity_all_normal_oof.{png,pdf}`.

