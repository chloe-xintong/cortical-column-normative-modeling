# Figure 7 — fixed Fold 1 internal held-out analysis

This directory evaluates the existing Figure 6 ico4 Warped-M2 model from
zero-based `outer_fold=1`. It never refits a model, mixes folds, or changes the
frozen Figure 6 artifacts.

The Figure 6 OOF run did not serialize posterior model objects. It did save, for
every node/depth/fold, the training-fold transformations and warp together with
held-out predictive medians, intervals, and z-scores. Figure 7 recovers the
fixed Fold 1 latent mean and variance prediction functions from those saved
artifacts. Recovery is accepted only when it reproduces every available saved
held-out z-score within numerical tolerance. Held-out CN burdens use the saved
OOF z-scores directly; recovered functions are used only for non-Normal targets.

The source pool is the formal coverage-at-least-95% N=313 cohort, restricted to
ages 41–78. Main Figure 7 analyses include Fold 1 held-out CN and all age-eligible
MCI, AD, Other/Impaired, and Unknown participants. Fold 1 training CN and all
other Normal participants are excluded.

An explicitly exploratory sensitivity analysis uses all 186 frozen reference
CN. Each participant contributes only the saved z-scores from that person's own
held-out fold in the five-fold OOF analysis; no participant-level scores are
averaged across folds. This does not replace the fixed-Fold-1 primary analysis.

Plotting and final presentation are contained in
`notebooks/figure7_fixed_fold1_analysis.ipynb`. Cached analysis tables are under
`outputs/`; scripts under `scripts/` prepare inputs, evaluate the fixed model,
compute participant-level statistics, and build the notebook.

