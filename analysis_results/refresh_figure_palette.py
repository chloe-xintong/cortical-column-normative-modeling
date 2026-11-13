"""Redraw active figures from cached results with the shared six-color palette.

Description
-----------
Redraw active figures from cached results with the shared six-color palette.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``figure`` command-line input

Outputs
-------
- In-memory values returned to the calling code.

Usage
-----
python analysis_results/refresh_figure_palette.py --help
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault('MPLBACKEND', 'Agg')


def run(script, *args):
    subprocess.run([sys.executable, str(REPO / script), *map(str, args)], check=True)


def refresh(figure):
    if figure == 2:
        config = json.loads((REPO / 'analysis_results/figure2/full_cohort_submission.json').read_text())
        run('analysis_results/figure2/plot_full_cohort.py', '--output-dir', config['output_dir'],
            '--fsaverage-dir', '/home/apps/freesurfer/subjects/fsaverage')
    elif figure == 3:
        import pandas as pd
        root = Path(json.loads((REPO / 'analysis_results/figure3/computation/submission_20261102.json').read_text())['output_dir'])
        run('analysis_results/figure3/computation/plot_figure3.py', '--output-dir', root,
            '--fsaverage-dir', '/home/apps/freesurfer/subjects/fsaverage')
        run('analysis_results/figure3/computation/figure3_profile_plot.py', '--output-dir', root)
        run('analysis_results/figure3/computation/figure3_region_effect_plot.py', '--output-dir', root)
        sys.path.insert(0, str(REPO / 'analysis_results/figure3/computation'))
        from compare_pc1_pc2_thickness import plot_paired
        out = root / 'pc1_pc2_thickness'
        plot_paired(pd.read_csv(out / 'subject_thickness_pc1_pc2_correlations.csv'),
                    json.loads((out / 'complete.json').read_text())['comparison'], root / 'figures')
        run('analysis_results/figure3/computation/plot_pc2_vs_thickness.py', '--output-dir', root)
    elif figure == 4:
        root = REPO / '/mnt/newStor/paros/paros_WORK/xintong/result/cortical_trajectory_analysis/analysis_results/figure4/outputs/registration_qc_pass_320'
        run('analysis_results/figure4/scripts/plot_ico5_formal_a_with_relative_mae.py',
            '--real-summary', root / 'real/cohort/figure4A_ico5_k15_real_group_summary.csv',
            '--relative-summary', root / 'relative_mae/cohort/figure4_relative_mae_group_summary.csv',
            '--output-dir', root / 'real/cohort', '--repo-root', REPO)
        run('analysis_results/figure4/scripts/plot_ico5_formal_b.py',
            '--cohort-dir', root / 'acquisition_null_fwhm0_seedfix/cohort')
        sys.path.insert(0, str(REPO / 'analysis_results/figure4/scripts'))
        from summarize_ico5_formal_c import plot_cached as plot_c
        from summarize_boundary_matched_effect import plot_cached as plot_d
        plot_c(root / 'correlation_length/cohort')
        plot_d(root / 'boundary_matched/formal/cohort', REPO)
        run('analysis_results/figure4/scripts/write_registration_qc_notebook.py')
    print(f'Figure {figure}: cached-result redraw complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure', type=int, choices=[2, 3, 4])
    refresh(parser.parse_args().figure)
