# Fixed-pair coverage and common-anchor sensitivity

Completed on the 313-participant coverage-QC-passing primary Figure 4 cohort.
No null generation, pair resampling, imputation or minimum-pair-count increase.

## Main findings

- The common template set contains **18,121/18,742 anchors (96.69%)**.
- Template active-anchor counts range from 18,123 to 18,742 across bins:
  18,740 in (0,2] mm and 18,123 in (28,30] mm.
- Including zero counts, pair count median and IQR are 15 [15,15] in every bin.
  Exact fractions with 0, 1–2, 3–14 and 15 pairs are in
  `template_coverage_by_bin.csv`.
- The five largest DK regions account for 34.17% of all anchors and 35.35%
  of common anchors; total variation in bilateral DK composition is 0.03029.
  Thus the intersection is not concentrated into only a few regions, but
  small-region losses are substantial: frontalpole 0/41; temporalpole 19/99;
  transversetemporal 24/113; parsorbitalis 33/114; entorhinal 35/120.
  There are 33 represented bilateral DK regions rather than 34.
- After individual masks, original active-anchor count median [IQR] is
  18,550 [18,476,18,596] in (0,2] mm and 17,939 [17,869,17,980] in (28,30] mm.
  Valid pair count is 274,598 [273,138,275,400] and
  265,191 [263,148,266,305], respectively. All 15 bins are exported.

| Lambda analysis | Passed/total | Median (mm) | IQR (mm) |
|---|---:|---:|---:|
| Original | 313/313 | 2.858480 | 2.670483–3.034620 |
| Common anchors | 313/313 | 2.939181 | 2.749902–3.113688 |

Among the 313 paired acceptable fits, common-minus-original lambda has
median **0.074726 mm**, IQR **0.045980–0.103016 mm**, and participant-bootstrap
95% CI **0.070101–0.078659 mm** (10,000 resamples; seed 20261119).
The median of individual percentage changes is +2.641%; 303/313 differences
are positive. The difference between the two marginal group medians need
not equal the median of participant-paired differences.

The increasing-to-plateau curve shape is retained. Paired delta-D medians
range from −0.001737 to −0.000014 across bins (−0.000624 in the nearest bin).
Anchor composition has a modest systematic effect on dissimilarity and
lambda; it does not appear to account for the overall distance dependence.
This is a sensitivity result, not an equivalence test or a bound on all
possible effects of regional composition. No per-bin hypothesis tests were
performed. Individual validity can still vary across bins within the common
template set, and small-region depletion limits anatomical generalization.

## Reproducibility and validation

```bash
python analysis_results/figure4/scripts/analyze_common_anchor_sensitivity.py
```

The source pair-file SHA256 matches the frozen formal table. A complete
18,742×15 grid includes template zero-pair bins. The independent official
split assignment verifies the full anchor universe. All 313 anchor-bin
caches have exactly the fixed template pair counts before individual masks.
Cache validity, effective pair counts and anchor counts are checked against
the original primary whole-cortex results for every participant and bin.
The maximum reconstructed dissimilarity difference is 1.49e-8 and maximum
original lambda reconstruction error is 3.25e-8 mm. A direct profile-mask
spot-check for D0007 verifies every bilateral anchor-bin count.

Although caches come from the region analysis, they are reused **before**
region aggregation. The formal anchor-weighted bilateral subject median is
used; no equal-region weighting is introduced. Anchor-bin profile and
distance medians remain unchanged when restricting anchor membership.
Curves are displayed at fixed centers 1,3,...,29 mm. Fits use each subject's
actual median geodesic coordinates, the original bounded exponential fit,
at least six valid bins, R²≥0.5 and exclusion of parameter-boundary fits.

`run_report.json` records provenance and numerical QC. PNG/PDF figures,
subject curves, counts, DK composition, fits and paired differences are
separate files in this directory. The notebook reader is
`analysis_results/figure4/notebooks/figure4_common_anchor_sensitivity.ipynb`.

## Actual pairs per anchor × bin

`actual_pair_counts_template.csv` reports mean, median, IQR and fractions
with exactly 15, zero, or 1–2 fixed pairs, with all 18,742 anchors in each bin.
`actual_pair_counts_individual_pooled.csv` applies the individual profile
masks and pools 313×18,742 equally weighted subject-anchor observations per
bin. Template-absent anchor-bins remain zero. Its quantiles describe these
observations, not the distribution of subject-level means. The complete
subject-level summaries are `actual_pair_counts_by_subject_bin.csv`.
Reproduce with `python analysis_results/figure4/scripts/summarize_actual_pair_counts.py`.
