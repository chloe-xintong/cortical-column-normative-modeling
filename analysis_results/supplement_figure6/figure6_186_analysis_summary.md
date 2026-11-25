# Figure 6 normative-model analysis summary (frozen 186-person cohort)

## 1. Frozen cohort and QC

- Final cohort: 186 CN participants.
- Source-image exclusions: D0271 and D0285, both recorded as
  `source_image_structural_abnormality`.
- The retained participants keep their original five-fold assignments; no participant was
  reassigned after exclusion.
- Fold test sizes: 38, 38, 37, 36, and 37.
- All retained participants pass the prespecified bilateral valid-coverage threshold of 95%.
- No further exclusion was made from statistical z scores, model residuals, or fit warnings.

## 2. Regional model comparison on the common 186-person cohort

All regional models use the same 186 participants, five folds, 68 hemisphere-specific DK
regions, and 21 normalized depths. There are 265,608 held-out observations per model.

| Comparison | Mean-log-loss difference | Subject-bootstrap 95% CI | Conclusion |
|---|---:|---:|---|
| M1 - M0 (spline vs linear age, no thickness) | +0.000390 | [-0.008315, +0.009655] | No stable spline benefit |
| M3 - M2 (spline vs linear age, with thickness) | -0.001967 | [-0.010866, +0.007417] | No stable spline benefit |
| M2 - M0 (local thickness added) | -0.015627 | [-0.021215, -0.010069] | Stable predictive-density benefit from thickness |
| Warped M2 - Gaussian M2 | -0.022927 | [-0.034351, -0.012741] | Stable predictive-density benefit from warp |

Mean log loss is `-logp` and is lower-is-better. These paired bootstrap analyses resample
participants, not individual region-by-depth observations.

### Gaussian M2 versus Warped M2

| OOF metric, all 21 depths | Gaussian M2 | Warped M2 |
|---|---:|---:|
| mean z | -0.00473 | -0.01150 |
| SD(z) | 1.00200 | 1.00668 |
| 50% coverage | 53.41% | 50.44% |
| 80% coverage | 82.25% | 80.51% |
| 90% coverage | 90.91% | 90.42% |
| coverage MAE | 2.188% | 0.457% |
| skewness | 0.427 | -0.001 |
| excess kurtosis | 2.506 | 0.835 |
| fraction \(|z|>3\) | 0.825% | 0.524% |
| fraction \(|z|>4\) | 0.209% | 0.0926% |
| Q-Q RMSE (0.001--0.999 quantile grid) | 0.0963 | 0.0349 |

The warp materially improves central coverage, symmetry, tail calibration, Q-Q agreement, and
predictive density. The six anchor depths reproduce the same conclusion and are a subset of the
21-depth analysis, not a separate competing model.

The recommended regional model therefore remains:

`Warped M2: MD ~ linear age + categorical sex_code + local FreeSurfer thickness`, using
PCNToolkit 1.3.0, `WarpSinhArcsinh`, and homoskedastic variance.

## 3. Formal ico4 Warped-M2 OOF analysis

The fixed Figure 5 ico4 mask contains 4,687 bilateral nodes and 21 depths. The formal run was
split into six anchor depths and the remaining 15 depths, but together they form one frozen
analysis.

| Stage | Fits successful / expected | Failed | Warning fits | OOF predictions |
|---|---:|---:|---:|---:|
| Smoke test (40 nodes, all depths) | 4,200 / 4,200 | 0 | 150 | 152,502 |
| Six anchor depths | 140,610 / 140,610 | 0 | 5,498 | 5,171,586 |
| Remaining 15 depths | 351,525 / 351,525 | 0 | 12,514 | 12,928,965 |
| Formal 21-depth total | 492,135 / 492,135 | 0 | 18,012 | 18,100,551 |

Across the formal run there are no duplicate, missing, or extra OOF keys, no nonfinite final
predictions, and no silent Gaussian fallbacks. The 18,012 warning fits (3.66%) concern candidate
optimization evaluations; their final parameters and predictions passed the finite-value checks.

### Overall 21-depth ico4 calibration

The following combined values are exact count-weighted summaries of the disjoint anchor and
remaining-depth stages. The pooled SD uses the stage counts, means, and sample variances.

| Metric | Combined result |
|---|---:|
| OOF predictions | 18,100,551 |
| mean z | -0.02050 |
| SD(z) | 1.00368 |
| 50% coverage | 49.77% |
| 80% coverage | 80.30% |
| 90% coverage | 90.38% |
| coverage MAE | 0.302% |
| fraction \(|z|>2\) | 4.323% |
| fraction \(|z|>3\) | 0.426% |
| fraction \(|z|>4\) | 0.0656% |
| mean predictive log likelihood (`log_likelihood`, higher is better) | 7.37221 |

The six-anchor and remaining-15-depth stages are highly concordant: their SD(z) values are
1.00350 and 1.00375, and their 90% coverages are 90.40% and 90.37%, respectively. This supports
combining them as the prespecified 21-depth result.

## 4. Interpretation across spatial scales

- Both regional and ico4 Warped M2 results have near-zero mean z, SD(z) near one, and coverage
  close to nominal values.
- The ico4 result does not show a loss of calibration when moving from 68 regions to 4,687 nodes.
- The regional six-depth subset and the ico4 anchor-depth stage both agree with their complete
  21-depth counterparts; the extra 15 depths add depth resolution without changing the model
  conclusion.
- Predictive log likelihood must not be compared numerically between the regional and ico4
  analyses because they contain different spatial outcomes and availability patterns. A formal
  cross-scale uncertainty comparison must also resample participants, rather than treating the
  millions of node-by-depth predictions as independent observations.

## 5. Current completion boundary

Completed:

- source-image QC and frozen 186-person cohort;
- common-cohort regional Gaussian M0--M3 and Warped M2 OOF analyses;
- regional subject-level paired bootstrap comparisons;
- ico4 smoke test and all 492,135 formal 21-depth OOF fits;
- merged 21-depth ico4 OOF index and exact valid-key integrity validation;
- combined calibration tables by age group, depth, hemisphere, and node by depth;
- worst-location and left-parahippocampal diagnostics;
- participant-bootstrap uncertainty for the primary ico4 summaries;
- all 98,427 full-cohort node-by-depth Warped M2 fits;
- recoverable PCNToolkit parameter shards and original-MD-scale 5th, 50th, and 95th centile
  predictions over the supported age range;
- Figure 6A representative-node curves;
- Figure 6C original-scale standardized 45--76 age effects, three seven-depth bands, bilateral
  ico4 surface data, support audits, and PNG/PDF map.

Figure 6C uses the same 186 participants' stored sex code and node-local thickness at both ages,
then scales the age-76 minus age-45 mean of conditional predicted medians by 10/31. Of 4,687
fixed-mask nodes, 4,517 have complete thickness for the exact 186-person estimand; 170 are
explicitly unavailable and gray rather than imputed or reweighted. Neutral numeric sex-code
labels remain appropriate until an external source codebook is documented. A future claim of
depth specificity would still require a direct within-node depth-band contrast with subject-level
uncertainty estimation; the current maps are descriptive effect-size maps.

## 6. Metric-definition note

Regional outputs store both `logp` (predictive log likelihood; higher is better) and
`mean_log_loss = -logp` (MLL/mean log loss; lower is better). The ico4 validator reports
`mean_log_likelihood` directly, so higher is better.

Two regional files currently calculate Q-Q RMSE with different grids: the direct calibration
table uses every ordered z score, whereas the Gaussian-versus-warped comparison uses 999
quantiles from 0.001 to 0.999. The cross-model comparison above uses the latter consistently for
both models. These definitions should be given distinct column names before the final notebook is
frozen.
