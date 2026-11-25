# Figure 4B null and nearest-distance gain audit

Active cohort: 313 participants from `manifests/figure4_coverage95_pass_313_subjects.csv`.
Reproduce with `python analysis_results/figure4/scripts/audit_figure4b_nearest_gain.py`.
The audit reads cached results only; no null is regenerated.

## Null definition

The primary null is native-DWI-grid Gaussian noise with **FWHM = 0 mm**.
`compute_spatial_null.py` calls `gaussian_filter` only when FWHM > 0.
Real O3 coordinates, trilinear interpolation and identical real/null validity
masks are used. Bilaterally pooled valid profiles are centered, whitened by
the synthetic depth covariance and recolored by the real depth covariance,
then shifted by the real 21-depth mean. The covariance matrices are 21×21;
the transformation has an eigenvalue floor for numerical stability and
outputs float32 profiles. It does not fit spatial smoothness or the observed
distance curve. It does not exactly match depth quantiles or full distributions.

Use **“acquisition-and-sampling null, FWHM = 0 mm; depth-mean/covariance matched”**,
not “smoothness-matched.” The surface-PCA surrogate is a separate analysis.
The current main plotting script labels this simply “Null model.”

## Platform correction and aggregation

For subject s and realization r:

`P_real_s = median(D_real_s(b))`, for bin indices b = 10,...,14.

`P_null_s,r = median(D_null_s,r(b))`, over the same five bins.

These are five 2-mm bins spanning (20,30] mm, not a mean, midpoint value,
or median across subjects. Define

`E_s,r(d) = [P_real_s − D_real_s(d)] − [P_null_s,r − D_null_s,r(d)]`.

First take the median E across the 100 realizations within each subject,
then the median across subjects. Platform correction precedes realization
aggregation; taking the platform of an already aggregated null curve is not
guaranteed to give the same answer. Differences of group medians likewise
need not equal the median of individual differences.

The nearest bin is **(0,2] mm**. Its subject-level gain has:

- Median 0.0690683; IQR 0.0590899–0.0835183.
- Subject-bootstrap percentile 95% CI **0.0673314–0.0713471**.
- 10,000 resamples; seed 20261119; resampling unit is participant.
- Positive gain in 313/313 participants.

This CI is conditional on the cached 100-realization null per participant.
The figure's original shaded band is between-subject IQR, not this CI.
The existing 0–10-mm AUC CI answers a different summary question.

## Random-realization stability

The 313 retained participants each have 100 realizations, with 31,300 unique
cached subject/realization seed tokens. The implementation derives seeds
from base seed, subject ID and replicate via BLAKE2b; it does not reuse one
noise field across subjects. This audit checks existing independent fields,
not newly generated runs with a different base seed.

Nearest-bin group medians using the first 10/25/50/100 realizations are
0.069118 / 0.069096 / 0.068963 / 0.069068. Ten disjoint batches of ten
realizations give medians from 0.0688483 to 0.0692858. The 100 single-
realization cohort medians range from 0.0684363 to 0.0699909. These ranges
describe Monte Carlo stability; they are not population confidence intervals.

Historical validation JSON retains some 325-subject seed counts and maxima,
as its validation note states. The audit report recomputes retained-cohort
counts and QC maxima. No original cache is overwritten.

## Suggested interpretation

After plateau correction, real data retain local similarity gain beyond the
current acquisition-and-sampling null. This does not establish that all
imaging/sampling effects are removed or isolate a purely physiological effect.

The 20–30-mm interval is an approximate reference plateau. Residual covariance
and native-grid sampling are represented, but empirical spatial smoothness
has not been fitted. No multiple-distance hypothesis tests were performed.
