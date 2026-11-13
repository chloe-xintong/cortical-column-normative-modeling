#!/usr/bin/env python3
"""Redraw paper subject-level violins from saved statistical results.

Description
-----------
Redraw paper subject-level violins from saved statistical results.

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
python analysis_results/render_subject_violins.py --help
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from analysis_results.figure2.plot_full_cohort import plot_all
from analysis_results.figure3.figure3_association_plot import export_subject_rho
from analysis_results.figure3.plot_pc2_vs_thickness import plot as plot_pc2
from analysis_results.figure3.compare_pc1_pc2_thickness import plot_paired


FIGURE2 = Path("/mnt/newStor/paros/paros_WORK/xintong/results/qc/template_registration_comparison/figure2_full_cohort_fsaverage")
FSAVERAGE = Path("/mnt/newStor/paros/paros_WORK/xintong/data/freesurfer/subjects/fsaverage")
FIGURE3 = Path("/mnt/newStor/paros/paros_WORK/xintong/results/figure3_depth_thickness")


def main():
    plot_all(FIGURE2, FSAVERAGE, statistics_only=True)
    export_subject_rho(FIGURE3)
    plot_pc2(FIGURE3)
    paired = FIGURE3 / "pc1_pc2_thickness"
    table = pd.read_csv(paired / "subject_thickness_pc1_pc2_correlations.csv")
    comparison = pd.read_csv(paired / "paired_absolute_correlation_comparison.csv").iloc[0].to_dict()
    plot_paired(table, comparison, paired)
    caption = ("Violin plots show the kernel density distribution across subjects. "
               "Internal boxes indicate the interquartile range, center lines indicate the median, "
               "and whiskers extend to the most extreme observations within 1.5 times the "
               "interquartile range. Gray points, where present, are boxplot outliers; all subjects "
               "contribute to the density and statistical summaries.\n")
    (FIGURE2 / "figures" / "Figure2_violin_caption.txt").write_text(caption)
    (FIGURE3 / "figures" / "Figure3_violin_caption.txt").write_text(caption)


if __name__ == "__main__":
    main()
