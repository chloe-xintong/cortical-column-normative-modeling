# Ico4 real-data local-similarity sampling-stability pilot

## Scope and configuration

This pilot uses the already frozen 37-person young-CN roster from
`analysis_results/figure6/figure6_new/outputs/full/young_reference_roster.csv`.
All 37 subject IDs map one-to-one to the formal 313-person Figure 4 manifest.
No new young-group definition was applied.

The 4,687 masked ico4 nodes are fixed anchors in the stored `node_order`.
Each record retains hemisphere, ico4 node ID, full-resolution fsaverage vertex
ID, and DK label. Candidate neighbors are all valid full-resolution vertices
in the same hemisphere, without a DK restriction. Distances are shortest paths
on the full-resolution fsaverage white mesh in (0,2], ..., (28,30] mm bins.

Five fixed seeds (0--4; base seed 20261119) independently permute each complete
anchor-bin candidate set without replacement. Ranks 1--15 define K=15 and
ranks 1--30 define K=30, so the two samples are nested. Candidate exhaustion is
retained as-is. Geometry is fixed across subjects before applying each formal
clean-profile validity mask.

Pearson correlation uses all 21 profile depths. Both vertices must pass the
formal subject mask, all depths must be finite, and both centered profiles must
have nonzero norm. `r_real` is the median valid pair correlation in each
subject-anchor-bin. `p_real` is the median of the five 20--30 mm bin medians;
the number of valid platform bins and five-bin completeness are stored.
`L_real = r_real - p_real`. It has no null correction and is not called E.

## Main results

### Local r_real

For K=15, the median across-seed SD of `r_real` over subject-anchor locations
was 0.0115, 0.0345, 0.0441, 0.0490, and 0.0516 in the five 0--10 mm bins.
For K=30 it was 0.0000, 0.0200, 0.0279, 0.0317, and 0.0338. The zero at
(0,2] for K=30 largely reflects candidate exhaustion: when all available
candidates are selected, every seed is identical. It is not evidence of high
statistical precision.

At K=15 the median sampling-SD / between-young-subject-SD ratio for `r_real`
was 0.052--0.140 across 0--10 mm; at K=30 it was 0.000--0.091. No anchor-bin
had between-subject SD below the recorded numerical near-zero criterion
(1e-8), so these ratios were not driven by a zero denominator.

Same-seed K=30 minus K=15 differences had median signed shift 0.0000 in every
distance bin. Median absolute differences increased from 0.0059 at (0,2] to
0.0212 at (8,10] and approximately 0.0236 at (28,30]. Thus K changes local
values materially at individual locations even though there is no detectable
global directional shift.

### Platform and L_real

The median across-seed SD of `p_real` was 0.0302 for K=15 and 0.0214 for K=30.
For same-seed K comparisons, the platform median signed shift was -0.00002
(mean 0.00138), while the median absolute difference was 0.01696. Platform
completeness for all five seeds was 98.93% of subject-anchor-K records.

Platform subtraction increases sampling sensitivity. For K=15, median
across-seed SD of `L_real` was 0.0383--0.0638 across 0--10 mm; K=30 reduced it
to 0.0223--0.0424. The median sampling/between-subject ratio for `L_real`
rose from 0.163 at (0,2] to 0.637 at (8,10] for K=15. In the 20--30 mm bins it
was approximately 0.99--1.15 for K=15 and 0.82--1.03 for K=30. Therefore the
raw correlation is substantially more sampling-stable than the platform-
subtracted feature, and K=15 does not uniformly stabilize `L_real` or the
platform at the anchor level.

Median absolute K differences for `L_real` were 0.0230--0.0326 over 0--10 mm.
Median signed differences were only 0.00047--0.00071, again indicating little
systematic shift but non-negligible location-specific changes.

### Coverage and spatial pattern

At K=15, locations with only 1--5 or 6--14 effective pairs had median
`r_real` seed SD of 0.179 and 0.121, versus 0.049 when all 15 were valid.
K=30 reduced the median to 0.033 when all 30 pairs were valid and 0.037 for
16--29 pairs. Seed SD can be exactly zero in exhausted candidate pools because
all seeds use the same pairs; these locations remain low-information and must
not be classified as stable from seed variance alone.

The largest K=30 reductions in local `r_real` seed SD occurred bilaterally in
entorhinal and temporal pole, followed by inferior temporal,
parahippocampal/fusiform, medial orbitofrontal, and anterior cingulate regions.
The same regions generally showed the largest platform-SD reductions. The
lowest platform-completeness fractions occurred in caudal/rostral anterior
cingulate, temporal pole, entorhinal, posterior cingulate, and inferior
temporal anchors (approximately 88--95% in the least complete region-hemi
groups).

Spatial-map correlations are supplied only as a supplement. For K=15, median
seed-0 versus other-seed map correlations over 0--10 mm were about
0.918--0.940 for `r_real`, but only 0.512--0.917 for complete-platform
`L_real`. K=30 increased these to about 0.965--0.997 and 0.691--0.982,
respectively. These high global correlations do not override the anchor-level
absolute differences and variance ratios above.

## Interpretation before null computation

K=15 estimates raw local correlation reasonably consistently in aggregate,
but it is not uniformly stable at individual anchors, especially in
low-coverage temporal/limbic locations. K=30 provides a clear reduction in
seed variability without a meaningful global signed shift. The platform and
`L_real` remain appreciably sampling-sensitive even at K=30 in some locations.

Before computing a null, downstream specifications should explicitly handle:

- candidate-exhausted bins, whose zero seed variance is structural;
- effective-pair count and common-valid-position requirements;
- incomplete five-bin platforms, with separate flags rather than silent use;
- region-specific instability in entorhinal, temporal pole, inferior temporal,
  anterior/posterior cingulate, and nearby medial/orbitofrontal locations;
- whether to use K=30, exclude/flag unstable anchor-bins, or regularize the
  platform estimator. This pilot deliberately sets no pass/fail threshold.

No null, young-reference model, group test, or perturbation network was run.
