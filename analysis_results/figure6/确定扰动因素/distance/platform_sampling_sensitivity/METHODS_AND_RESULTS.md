# Distant-platform sampling sensitivity

## Methods

This real-data-only sensitivity analysis extends the existing ico4 local-
similarity pilot. It uses the same frozen 37 young CN, 4,687 ico4 anchors,
full-resolution same-hemisphere candidate vertices, subject validity masks,
white-surface geodesic distances, node order, and 2-mm distance bins.

Only the five original platform bins, (20,22] through (28,30] mm, were
resampled. Platform K values were 30, 100, and 300 for fixed seeds 0--4.
Sampling used the original base seed 20261119 and was nested without
replacement. All 3,515,250 old rank<=30 rows were regenerated and verified
against the existing K=30 pair tables. Candidate exhaustion retained all
available candidates and never duplicated a pair.

Within each distant bin, valid 21-depth profile Pearson correlations were
summarized by their median. The platform `p_real` remained the median of the
five bin medians; distant pairs were not pooled across bins. Effective pair
counts, valid platform-bin counts, candidate exhaustion, and platform
completeness were retained.

To isolate platform propagation, the local correlation was fixed to the
existing K=30, seed-0 `r_real` in each 0--10 mm bin. Only the platform seed and
platform K changed in `L_platform_only = r_local_fixed - p_real`. Primary
numeric comparisons use the exact subject-anchor intersection with all five
platform bins complete for all five seeds and all three K values.

## Results

### Platform stability and saturation

Median across-seed SD of `p_real` was 0.02136 at K=30, 0.01213 at K=100,
and 0.00656 at K=300. The absolute reductions were 0.00923 from 30 to 100
and 0.00557 from 100 to 300. Thus improvement shows diminishing absolute
returns, but K=100 is not a complete plateau: K=300 still reduced median seed
SD by about 46% relative to K=100.

Same-seed position-level median absolute changes were 0.01407 for K100-K30,
0.00837 for K300-K100, and 0.01529 for K300-K30. Median signed changes were
0.00007, 0.00000, and 0.00020, respectively. Increasing K therefore changes
individual platforms without an appreciable cortex-wide directional shift.

The common-complete subject-anchor fraction was 98.93%. Platform completeness
was also 98.93% at each K. Candidate exhaustion was absent at K=30/100 and
occurred in 0.47% of subject-anchor-seed rows at K=300. It was concentrated in
medial orbitofrontal and rostral anterior cingulate anchors, with smaller
effects in right entorhinal and parahippocampal cortex. Exhausted K=300 pools
must not be interpreted as exceptionally stable merely because all seeds use
nearly the same candidates.

### Propagation into 0--10 mm L_real

Because local `r_real` was fixed, platform-only seed SD of `L_real` was nearly
constant across the five local distance bins. Its median was 0.02136 at K=30,
0.01213 at K=100, and 0.00656 at K=300. The corresponding median seed ranges
were 0.05261, 0.02991, and 0.01611.

Compared descriptively with the old K=30 result in which local and platform
samples both varied, platform-only K=30 SD was:

- essentially the whole observed SD at (0,2], where the local K=30 candidate
  pool is often exhausted and local seed variation is zero;
- 72% of combined SD at (2,4];
- 60%, 56%, and 53% at (4,6], (6,8], and (8,10].

The analogous squared-SD ratios were 52%, 37%, 31%, and 28% over (2,4] through
(8,10]. These are descriptive contrasts, not an independence-based variance
decomposition. At K=100, residual platform-only SD was 0.01213; at K=300 it
was 0.00656. Local-pair sampling therefore becomes the larger remaining source
at most bins as platform K increases, but platform sampling is still visible
at K=300.

### Temporal, entorhinal, and cingulate anchors

All requested regions improved. From K=30 to K=300, median platform seed SD
fell approximately:

- right/left entorhinal: 0.0743/0.0708 to 0.0170/0.0166 (77%/77% reduction);
- right/left temporal pole: 0.0510/0.0464 to 0.0127/0.0123 (75%/74%);
- inferior temporal: about 0.047--0.049 to 0.014--0.015 (about 71%);
- rostral anterior cingulate: about 0.031 to 0.007 (about 77%);
- caudal anterior, posterior, and isthmus cingulate: typically about 70--74%.

Despite the improvement, entorhinal, temporal pole, and inferior temporal
anchors retain larger absolute platform SD than many other regions. K=300
candidate exhaustion also affects rostral anterior cingulate and right
entorhinal locations, so their improvement must be read together with actual
effective pair counts.

## Interpretation

The gain is approaching saturation in absolute terms but has not plateaued at
K=100. K=300 gives a further meaningful reduction in platform and propagated
local-feature variability. It also introduces a small, spatially structured
candidate-exhaustion problem. No pass threshold is imposed. These results do
not include a null or perturbation model.
