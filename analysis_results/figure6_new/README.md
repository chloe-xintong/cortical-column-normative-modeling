# Figure 6 young-reference pilot

This directory contains an additive pilot and does not alter the frozen Figure 6 Warped-M2
analysis. The pilot selects CN younger than 50 years from the pre-age-filter Figure 5/6 ico4
candidate matrix, applies the existing coverage and imaging-QC rules, and fits at every selected
ico4 node and depth:

```text
raw MD ~ categorical sex code + node-local FreeSurfer cortical thickness
```

Age is used only to select the young reference. Signed deviation is
`(observed MD - predicted MD) / training-reference residual SD`.

## Smoke test

From the repository root:

```bash
/home/apps/ubuntu-22.04/anaconda3/bin/python \
  analysis_results/figure6_new/run_young_reference.py \
  --mode smoke \
  --smoke-nodes 4
```

The default smoke output is `analysis_results/figure6_new/outputs/smoke/`. It runs all eligible
reference and target participants but only four of the existing Figure 6 smoke-test nodes, with
all 21 depths at each node.

## Full run

```bash
/home/apps/ubuntu-22.04/anaconda3/bin/python \
  analysis_results/figure6_new/run_young_reference.py \
  --mode full \
  --output-dir analysis_results/figure6_new/outputs/full
```

The full run uses all 4,687 fixed-mask ico4 nodes and 21 depths. It is intended for a compute
node rather than a login node.

## Primary outputs

- `cohort_report.json` and `young_reference_candidate_audit.csv`: selection counts and audit.
- `young_reference_loo_depth_band_burden.csv`: exact leave-one-subject-out young-CN burdens.
- `young_reference_loo_z_young.npy`: signed young-CN LOO z-score matrix. Rows use
  `young_reference_roster.csv:loo_row_index`; columns use
  `scored_node_depth_metadata.csv:output_position`; unavailable positions are `NaN`.
- `young_reference_parameters.parquet` and `.npz`: full young-reference coefficients and residual
  SD at every fitted position.
- `target_z_young.npy`: signed target-by-position z-score matrix.
- `scoring_target_roster.csv` and `scored_node_depth_metadata.csv`: explicit row/column indices for
  the z-score matrix.
- `participant_depth_band_burden.csv`: Figure 7-ready wide participant table containing actual
  age, diagnosis, subject key, coverage, valid denominators, and `|z_young| > 2` fractions for
  outer (0--6), middle (7--13), inner (14--20), and all depths.
- `run_report.json`: model definition, input hashes, dimensions, fit counts, and LOO validation.

The current frozen downstream target definition uses 41--78-year support for MCI/AD, excludes
D0142, and retains D0156. D0183, D0290, and D0176 are outside that support. D0271 and D0285 are
excluded from both reference and older-CN target roles by the existing Figure 6 source-image QC.

## Full LOO model comparison

`compare_loo_models.py` compares the existing linear thickness-adjusted reference (A), the
historical PCNToolkit 1.3.0 `WarpSinhArcsinh` model with identical covariates (B), and a linear
sex-only model (C). Each of the 37 held-out CN has a separately fitted model at every position.

Smoke test:

```bash
XDG_CACHE_HOME=/tmp/figure6-new-cache MPLCONFIGDIR=/tmp/figure6-new-mpl \
  .venv/figure6-pcntoolkit-1.3.0/bin/python \
  analysis_results/figure6_new/compare_loo_models.py \
  --action smoke --smoke-nodes 1 \
  --output-dir analysis_results/figure6_new/outputs/loo_model_comparison_smoke
```

Full node array and final aggregation:

```bash
mkdir -p analysis_results/figure6_new/outputs/loo_model_comparison_full/logs
for offset in 0 1000 2000 3000; do
  sbatch --export=ALL,NODE_OFFSET="$offset" \
    analysis_results/figure6_new/run_loo_model_comparison_full.sbatch
done
sbatch --array=0-686%20 --export=ALL,NODE_OFFSET=4000 \
  analysis_results/figure6_new/run_loo_model_comparison_full.sbatch

XDG_CACHE_HOME=/tmp/figure6-new-cache MPLCONFIGDIR=/tmp/figure6-new-mpl \
  .venv/figure6-pcntoolkit-1.3.0/bin/python \
  analysis_results/figure6_new/compare_loo_models.py \
  --action summarize \
  --output-dir analysis_results/figure6_new/outputs/loo_model_comparison_full
```
