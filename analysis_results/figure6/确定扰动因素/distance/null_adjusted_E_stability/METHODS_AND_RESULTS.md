# Small-scale null-adjusted E stability pilot

## Configuration

Ten participants were selected without replacement from the frozen 37-person
young-CN roster using NumPy seed 20261121. All 4,687 ico4 anchors were retained.
Pair seeds were the first three existing seeds (0, 1, 2), without outcome-
based selection. Local bins use nested K=30 pairs and the five 20--30 mm
platform bins use nested K=300 pairs.

The null implementation directly reuses the formal Figure 4B functions and
settings: one Gaussian random field over the complete native DWI FOV, sampling
at the real O3 DWI voxel coordinates with order-1 interpolation, FWHM=0, and
exact bilateral subject-level matching of the real 21-depth mean and covariance.
The formal base seed 20261020 and `blake2b64:F4AcqNullV2` subject/replicate seed
scheme were retained. Each subject has 20 independent realizations. A single
null field per subject/realization was reused for all three pair seeds.

The earlier real pilot used a slightly wider profile mask than formal Figure
4B. To guarantee identical real/null validity, real features were recomputed
with the formal intersection of acquisition geometry validity and final real
profile validity (`sampling_valid`, analysis mask, mapping, column validity,
finite depths, and nonzero profile variance). An explicit old-vs-formal mask
audit is saved.

For real and every null realization, pair correlations were first summarized
within each anchor-bin. Each anchor's platform was then the median of its five
far-bin medians. Local enhancement was `L = r - p`, separately for real and
null. Per-realization `E = L_real - L_null`; final E is the median across 20
realizations. Negative E values are retained.

## Pair-sampling sensitivity

Using the same 20 null fields for all pair seeds, the median location-level
pairwise absolute E difference was:

- (0,2]: 0.0112
- (2,4]: 0.0323
- (4,6]: 0.0398
- (6,8]: 0.0437
- (8,10]: 0.0463

The corresponding median across-pair-seed SD was 0.0081, 0.0234, 0.0287,
0.0316, and 0.0335. Pair sensitivity therefore increases across 0--10 mm.
The small (0,2] value requires caution: about 72% of positions exhaust the
local K=30 geometry pool, so different pair seeds often use the same complete
candidate set.

Pair sensitivity was spatially heterogeneous. Median pairwise absolute E
differences across the five local bins were largest in right and left
entorhinal cortex (0.171 and 0.129), temporal pole (about 0.092--0.115), and
inferior temporal cortex (about 0.103--0.106). Anterior cingulate values ranged
from about 0.056 to 0.085; posterior/isthmus cingulate values were about
0.043--0.052. These regions remain prominent after null adjustment.

## Null-realization sensitivity

For fixed pair tables, median absolute differences between the two independent
10-realization batches were very similar across pair seeds and measured:

- (0,2]: 0.0216
- (2,4]: 0.0209
- (4,6]: 0.0196
- (6,8]: 0.0188
- (8,10]: 0.0184

Each 10-realization batch differed from the combined-20 E by a median of about
0.0083--0.0097. Individual-realization E remained much noisier: its median SD
was about 0.120 at (0,2], declining to 0.066 at (8,10]. Thus aggregation across
realizations is essential; 20 realizations improve over either 10-realization
batch but do not eliminate null Monte Carlo error.

At (0,2], null-batch sensitivity exceeds pair-seed sensitivity. From (2,4]
outward, pair-seed differences exceed 10-vs-10 null-batch differences and are
the larger observed instability source. This comparison uses absolute
differences rather than sign agreement, so near-zero E values are not treated
as failures merely for changing sign.

## Coverage

The common analysis-complete fraction across the three pair seeds was
approximately 98.94--98.98% across the five local bins. Median effective local
pair counts were 25 at (0,2] and 30 in the other bins. Platform candidate
exhaustion affected about 0.47% of positions; local exhaustion was concentrated
almost entirely in (0,2]. No position or DK region was automatically excluded.

## Interpretation

This pilot does not establish a pass threshold. It shows that null-adjusted E
has two separable sources of instability: Monte Carlo variation remains around
0.02 for two independent batches of 10 nulls, while K=30 local pair sampling
creates larger location-level differences beyond 2 mm and is especially
important in entorhinal, temporal-pole, inferior-temporal, and cingulate
anchors. A later production specification should consider both more null
realizations and local-pair stabilization; increasing null realizations alone
will not resolve the observed pair-seed sensitivity.

No young-reference model or group comparison was fitted.
