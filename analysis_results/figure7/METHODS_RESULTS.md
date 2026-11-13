# Figure 7 concise methods and results

## Methods

We performed a fixed internal held-out analysis using the existing Figure 6
ico4 Warped-M2 model from zero-based `outer_fold=1`. Saved OOF z-scores were
reused for the 38 held-out cognitively normal (CN) participants. The same fixed
Fold 1 prediction function evaluated age-eligible non-Normal participants
without model refitting or use of another fold. Total deviation burden was the
participant-level fraction of valid node-depth positions with `|z| > 2`;
positive and negative burdens were directional secondary measures.

The formal coverage-at-least-95% N=313 cohort was restricted to ages 41–78.
The final cohort comprised 38 held-out CN, 19 MCI, 6 AD, 22 Other/Impaired, and
26 Unknown participants. Fold 1 training CN and all other Normal participants
were excluded. Figure 7A used participant-level OLS with HC3 standard errors,
adjusting for age, sex code, and bilateral profile coverage. Holm correction
covered the MCI–CN and AD–CN primary contrasts. Figure 7B related Global
Cognition Composite to total burden standardized within complete cases, with
the same covariates; diagnosis adjustment was a sensitivity analysis.

The original Figure 6 OOF run did not serialize Fold 1 posterior objects. The
fixed latent mean and variance prediction functions were therefore recovered
from the saved Fold 1 medians, intervals, warp, standardization, and held-out
design points. Recovery was accepted only when saved held-out z-scores were
reproduced within numerical tolerance. Held-out CN burden always used the
saved OOF z-score itself. No hyperparameter optimization or response-model fit
was run.

## Results

Figure 7A included 63 participants (38 held-out CN, 19 MCI, 6 AD). The adjusted
MCI–CN total-burden difference was **0.12 percentage points** (95% CI −1.15 to
1.38; unadjusted and Holm-adjusted p=0.858). The adjusted AD–CN difference was
**3.83 points** (95% CI −1.60 to 9.26; raw p=0.167; Holm p=0.333). The AD
estimate was imprecise because only six participants were available.

Figure 7B included 109 participants with Global Cognition Composite; D0276 and
D0277 (both Unknown) lacked the score. The unadjusted coefficient per 1-SD
higher total burden was **−0.629** composite units (95% CI −1.157 to −0.101;
p=0.020). The prespecified age-, sex-, and coverage-adjusted coefficient was
**−0.620** (95% CI −1.335 to 0.095; p=0.089). After additionally adjusting for
diagnosis, the sensitivity coefficient was **−0.396** (95% CI −0.721 to −0.071;
p=0.017). Thus the primary association did not exclude zero, while the raw and
diagnosis-adjusted estimates are descriptive and sensitivity results,
respectively; neither replaces the prespecified primary inference.

Restricting to coverage≥98% retained 100 participants. The corresponding
primary cognition coefficient was −0.723 (95% CI −1.440 to −0.006; p=0.048),
and the diagnosis-adjusted coefficient was −0.375 (95% CI −0.738 to −0.011;
p=0.043); these remain sensitivity results. Positive burden showed a stronger
AD direction than negative burden, but neither directional AD–CN comparison
was conventionally significant (positive p=0.054; negative p=0.474).

Age and coverage ranges overlapped across groups, but AD participants were
older than held-out CN (standardized mean difference 0.90) and had lower mean
coverage (−0.73); five of six AD participants had sex code 2. Other/Impaired
participants were also older (0.85) and predominantly sex code 1. These
imbalances motivate the adjusted models and caution against causal or
external-validation interpretations.

Fixed-function recovery succeeded at 98,385/98,427 node-depth positions. At 42
positions (two nodes), the stored held-out design did not have sufficient rank
to recover a target prediction function; target z-scores were explicitly
unavailable there rather than imputed or refit. Across recovered positions, the
maximum saved-z reproduction error was `8.7e-14`. D0271 and D0285 were not in
the prespecified analysis cohort, so their exclusion sensitivity was not
applicable. MoCA was not analyzed in the main figure; values of 88, including
D0076 and D0127, were stored as missing.

These findings are a **fixed internal held-out analysis**, not external
independent validation.

## Exploratory all-Normal five-fold OOF analysis

At the user's request, we additionally analyzed all 186 frozen Figure 6
reference CN. Every participant contributed exactly one saved OOF z-score set,
from the fold in which that participant was held out (fold sizes 38, 38, 37,
36, and 37); no participant's scores were averaged across folds and no model
was refit. All 186 were aged 41–78, passed the formal coverage≥95% criterion,
and had Global Cognition Composite available.

In the model adjusted for age, sex code, and bilateral profile coverage, the
coefficient per 1-SD higher total OOF burden was **0.022** composite units (95%
CI −0.077 to 0.121; p=0.664). Adding OOF-fold fixed effects gave **0.027**
(95% CI −0.077 to 0.131; p=0.609). Thus, this exploratory analysis did not find
evidence of a linear burden–cognition association within the frozen Normal
reference group. It is kept separate from, and does not replace, the fixed
Fold 1 Figure 7 primary analysis.

