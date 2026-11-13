#!/usr/bin/env python3
"""Create the cached-results notebook for the frozen formal Figure 4A-D analysis.

Description
-----------
Create the cached-results notebook for the frozen formal Figure 4A-D analysis.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- No required command-line inputs; optional inputs are defined by the script.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/create_figure4_formal_notebook.py --help
"""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("analysis_results/figure4/notebooks/figure4_formal_ico5_k15.ipynb")


def lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def markdown(identifier: str, text: str) -> dict:
    return {"cell_type": "markdown", "id": identifier, "metadata": {}, "source": lines(text)}


def code(identifier: str, text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": identifier,
        "metadata": {}, "outputs": [], "source": lines(text),
    }


def main() -> None:
    cells = [
        markdown("title", """
# Figure 4A–D: formal frozen ico5 analysis

This notebook reads cached full-cohort results and does not recompute profiles, geodesic neighbourhoods, real curves, null realizations, or boundary matches. Figure 4A–C use official ico5 anchors mapped one-to-one to fsaverage, within-DK-region candidates, nested sampling with $K=15$, and pair-sampling seed 0. The acquisition-and-sampling null uses 100 independently seeded realizations per subject. Figure 4D uses the same ico5 anchors and dense valid candidates, with deterministic one-to-one matching of within-region and across-boundary candidates from the same anchor at geodesic distances matched within 0.1 mm.
"""),
        code("setup", """
from pathlib import Path
import json

import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path('/mnt/newStor/paros/paros_WORK/xintong/results/qc/full_cohort_spatial_scale/figure4_ico5_formal_k15')
PAIR_QC_DIR = ROOT / 'pair_qc'
REAL_DIR = ROOT / 'real' / 'cohort'
RELATIVE_MAE_DIR = ROOT / 'relative_mae' / 'cohort'
NULL_DIR = ROOT / 'acquisition_null_fwhm0_seedfix' / 'cohort'
CORRELATION_DIR = ROOT / 'correlation_length' / 'cohort'
BOUNDARY_DIR = ROOT / 'boundary_matched' / 'formal' / 'cohort'
COMPARISON_DIR = ROOT / 'comparison'

def show_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    display(Image(filename=str(path)))

print(f'Frozen results: {ROOT}')
"""),
        markdown("pair-qc-title", """
## Pair-table QC

Required checks are performed on the single frozen directed-pair list used by Figure 4A–C. The C split is fixed once within each hemisphere × DK-region stratum, and the same anchor remains in the same half across all distance bins.
"""),
        code("pair-qc", """
pair_qc = json.loads((PAIR_QC_DIR / 'formal_pair_qc.json').read_text())
coverage = pd.read_csv(PAIR_QC_DIR / 'formal_k15_coverage_by_distance.csv')
display(pd.DataFrame([pair_qc]))
display(coverage)
assert pair_qc['status'] == 'pass'
"""),
        markdown("figure-a-title", """
## Figure 4A: real profile-shape dissimilarity

For every subject, pairwise $D=1-r$ and symmetric relative profile MAE are reduced as pair → anchor × 2-mm-bin median → bilateral subject median. Curves are group medians and bands are subject-level IQRs. Symmetric relative MAE is $200\,\mathrm{MAE}(x,y)/(\mathrm{median}|x|+\mathrm{median}|y|)$ and is reported as a percentage.
"""),
        code("figure-a", """
real_group = pd.read_csv(REAL_DIR / 'figure4A_ico5_k15_real_group_summary.csv')
relative_mae_group = pd.read_csv(
    RELATIVE_MAE_DIR / 'figure4_relative_mae_group_summary.csv'
)
display(real_group)
display(relative_mae_group)
show_image(REAL_DIR / 'Figure4A_ico5_K15_with_relative_MAE.png')
"""),
        markdown("figure-b-title", """
## Figure 4B: seed-fixed acquisition-and-sampling null

The null is Gaussian noise on each subject's native DWI grid, sampled through the real O3 coordinates with trilinear interpolation and recolored to match the subject-specific laminar mean and 21×21 depth covariance. The primary plateau is 20–30 mm. The main result is the residual local-excess similarity $L_{real}(d)-L_{null}(d)$; it is not interpreted as a purely physiological effect.
"""),
        code("figure-b", """
null_validation = json.loads((NULL_DIR / 'figure4B_acquisition_validation_summary.json').read_text())
auc_group = pd.read_csv(NULL_DIR / 'figure4B_local_excess_auc_group_summary.csv')
far_windows = pd.read_csv(NULL_DIR / 'figure4B_far_window_sensitivity_auc_group.csv')
convergence = pd.read_csv(NULL_DIR / 'figure4B_acquisition_convergence.csv')
display(pd.DataFrame([null_validation]))
display(auc_group)
display(far_windows)
display(convergence)
show_image(NULL_DIR / 'Figure4B_ico5_K15_formal_main.png')
"""),
        markdown("figure-b-supp-title", """
### Figure 4B supplementary subject-level AUC

Each point is one subject's 0–10-mm residual local-excess AUC; the horizontal line and vertical segment show the median and IQR.
"""),
        code("figure-b-supp", """
show_image(NULL_DIR / 'Figure4B_ico5_K15_AUC_supplement.png')
"""),
        markdown("figure-c-title", """
## Figure 4C: correlation length and split-half reproducibility

The exponential model is $D(d)=D_\infty-A\exp(-d/\lambda)$. Full-data and fixed stratified half-curves use the same fitting bounds and acceptance criterion ($R^2\geq0.5$ with no parameter-boundary hit).
"""),
        code("figure-c", """
full_fit = pd.read_csv(CORRELATION_DIR / 'figure4C_full_fit_summary.csv')
reliability = pd.read_csv(CORRELATION_DIR / 'figure4C_fixed_split_reliability.csv')
fit_qc = pd.read_csv(CORRELATION_DIR / 'figure4C_fit_qc_summary.csv')
display(full_fit)
display(reliability)
display(fit_qc)
show_image(CORRELATION_DIR / 'Figure4C_ico5_K15_correlation_length.png')
"""),
        markdown("figure-d-title", """
## Figure 4D: within-region versus across-boundary matched pairs

For each subject, hemisphere, ico5 anchor, and 2-mm distance bin from 0–10 mm, within-region and across-boundary candidates are deterministically matched one-to-one by white-surface geodesic distance (maximum mismatch 0.1 mm; up to 15 matches). The displayed effect is $\Delta D=D_{across}-D_{within}$ after matched pairs → anchor-bin mean → bilateral subject-region-bin mean → equal-region subject-bin mean. The curve is the group median with subject-level IQR; positive values indicate greater across-boundary profile-shape dissimilarity.
"""),
        code("figure-d", """
boundary_status = json.loads((BOUNDARY_DIR / 'figure4D_status.json').read_text())
boundary_group = pd.read_csv(BOUNDARY_DIR / 'figure4D_group_summary.csv')
boundary_scalar = pd.read_csv(BOUNDARY_DIR / 'figure4D_0_10_group_summary.csv')
boundary_qc = pd.read_csv(BOUNDARY_DIR / 'figure4D_group_pair_qc_by_bin.csv')
display(pd.DataFrame([boundary_status['pair_qc']]))
display(boundary_group)
display(boundary_scalar)
display(boundary_qc)
assert boundary_status['status'] == 'complete'
assert boundary_status['subjects'] == 325
assert boundary_status['pair_qc']['status'] == 'pass'
show_image(BOUNDARY_DIR / 'Figure4D_boundary_matched_0_10.png')
"""),
        markdown("comparison-title", """
## Legacy versus formal frozen analysis

This table is a robustness comparison only. Because anchor density, pair scope, $K$, and null seeds changed together, differences cannot be uniquely attributed to any single design change.
"""),
        code("comparison", """
comparison = pd.read_csv(COMPARISON_DIR / 'figure4_legacy_vs_formal_comparison.csv')
display(comparison)
"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "cortical_trajectory_analysis",
                "language": "python", "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
