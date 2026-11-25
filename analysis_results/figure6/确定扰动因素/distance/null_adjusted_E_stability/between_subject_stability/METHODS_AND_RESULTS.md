# Between-subject stability of null-adjusted E

## Scope and definitions

This diagnostic reads only the cached 10-subject results from
`null_adjusted_E_stability`. It creates no pairs, null realizations, image
samples, disease comparisons, or reference models.

For every anchor-bin and pair of conditions, analysis used the exact common
finite subjects. Let A and B denote either two pair seeds using the same 20
nulls, or the two independent 10-null batches at a fixed pair seed.

- Common offset is the across-subject mean of `E_A - E_B`.
- Centered residual SD is the across-subject SD after removing that offset.
- Reference between-subject SD is the SD of `(E_A + E_B)/2`.
- Pearson and Spearman correlations compare E across common subjects.

Correlations and residual/between-SD ratios are left missing when fewer than
four common subjects are available, either condition is constant, or the
between-subject SD is no larger than the numerical epsilon 1e-8. These are
computational guards, not scientific pass thresholds.

For hemisphere×DK summaries, each subject and comparison first used the exact
common finite anchors in that region/bin, then separately took the median E
under A and B. This represents regional aggregation of local neighborhoods; it
is not a within-DK-only pair analysis.

## Anchor-bin results

Across locations, median common offsets were close to zero for both sources of
variation. Pair-seed offset medians ranged from about -0.00011 to 0.00060 over
0--10 mm; null-batch medians ranged from about -0.00008 to 0.00018. The main
instability is therefore subject-specific change rather than a shared shift.

For pair-seed comparisons, median centered residual SD increased from 0.0161
at (0,2] to 0.0743 at (8,10]. At the same time, median between-subject SD
declined from 0.201 to 0.0876. The residual/between-SD ratio consequently rose
from 0.090 to 0.856. Median Pearson correlation declined from 0.997 to 0.741,
and median Spearman correlation from 0.976 to 0.661.

For the two null batches, centered residual SD was more uniform, declining
from 0.0362 to 0.0291. Residual/between-SD ratios rose from 0.179 to 0.301.
Pearson remained 0.986--0.962 and Spearman remained 0.915--0.903. Thus the two
10-null batches generally preserve individual differences and ranking better
than changing the K=30 pair sample, especially beyond 2 mm.

Only about 0.11--0.15% of anchor-level entries had fewer than four common
subjects, and about 0.02--0.06% had near-zero between-subject SD. These entries
are explicitly flagged and excluded only from correlation/ratio calculations;
they remain in the detailed output.

## Hemisphere×DK summaries

Regional medians reduced the absolute centered residual SD. For pair-seed
comparisons it was approximately 0.0066 at (0,2] and 0.011--0.013 in the other
bins. However, regional aggregation also reduced between-subject SD. The
pair-seed residual/between-SD ratio still increased from 0.178 to 0.870;
Pearson fell from 0.987 to 0.710 and Spearman from 0.964 to 0.630. Regional
aggregation therefore does not restore stable subject ranking at longer local
distances.

For null batches, DK-level Pearson decreased from 0.973 to 0.894 and Spearman
from 0.927 to 0.818. Residual/between-SD ratios increased from 0.260 to 0.507.
This is weaker than the anchor-level correlations because regional E has less
between-subject variance after spatial median aggregation, although its
absolute residual SD is smaller.

## Focus regions

Values below are medians across the five local bins and the three prespecified
comparisons.

Pair-seed preservation was weakest in several cingulate and temporal/limbic
regions. Examples include:

- left caudal anterior cingulate: Pearson 0.59, Spearman 0.55, residual/between
  ratio about 1.05;
- right posterior cingulate: Pearson 0.64, Spearman 0.56, ratio about 0.94;
- right entorhinal: Pearson 0.81, Spearman 0.70, ratio about 0.73;
- left entorhinal: Pearson 0.91, Spearman 0.88, ratio about 0.46;
- temporal pole: Pearson about 0.89--0.96 but Spearman only about 0.76--0.77;
- inferior temporal: Pearson about 0.83--0.89, Spearman about 0.70--0.79.

The null-batch comparisons were generally more stable in entorhinal, temporal
pole, inferior temporal, and anterior cingulate regions, commonly with Pearson
above 0.94. Posterior/isthmus cingulate remained more sensitive to the two
10-null batches, with Pearson roughly 0.85--0.93 and smaller regional
between-subject variance.

Each listed region retained all 10 common subjects in these summaries. The
minimum number of common anchors varied by region (for example about 9--13 in
temporal-pole/entorhinal groups versus about 58 in inferior temporal), so the
small-region estimates should be interpreted as 10-person pilot diagnostics.

## Interpretation

Subject differences are not primarily affected by a common additive offset.
Null Monte Carlo batching introduces measurable noise but mostly preserves
individual ordering. Pair choice is the more important threat to preserving
between-subject differences beyond 2 mm, and DK median aggregation does not
remove this problem because it also compresses the biological between-subject
spread. No pass threshold or automatic regional exclusion is applied.
