"""Subject-level effect-size display shared by the figure and notebook.

Description
-----------
Subject-level effect-size display shared by the figure and notebook.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure3/computation/figure3_association_plot.py --help
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from qc.publication_style import subject_violin


def draw_subject_rho(ax, subjects, summary):
    values = subjects.spearman_rho.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or subjects.subject_id.duplicated().any():
        raise ValueError('Expected one finite spatial rho per unique subject')
    if len(values) != int(summary.n_subjects):
        raise ValueError('Subject count differs from association summary')
    subject_violin(ax, values, position=1, orientation="horizontal",
                   single_group=True, showfliers=True)
    bound = max(.15, float(np.max(np.abs(values))) + .02)
    ax.axvline(0, color='#888888', linestyle='--', linewidth=1, zorder=0)
    ax.set(xlim=(-bound,bound), ylim=(.3,1.7), yticks=[],
           xlabel='Within-subject spatial Spearman ρ\n(PC1 vs thickness)')
    ax.text(.02,.03,'statistically detectable but negligible\nsubject-level association',
            transform=ax.transAxes,fontsize=10,va='bottom')
    ax.set_title('Thickness and profile shape',loc='left',fontweight='normal')
    return ax


def export_subject_rho(root):
    """Export the standalone panel used by the notebook, without kernel caches."""
    from pathlib import Path
    import sys
    import pandas as pd
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from qc.publication_style import set_publication_style, finalize_figure
    root = Path(root)
    subjects = pd.read_csv(root / 'tables/figure3C_subject_spatial_correlations.csv')
    summary = pd.read_csv(root / 'tables/figure3C_group_association_summary.csv').set_index('statistic')
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7.09, 3.2))
    draw_subject_rho(ax, subjects, summary.loc['median_spearman_rho'])
    finalize_figure(fig)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(root / 'figures' / f'Figure3C_subject_thickness_association.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    export_subject_rho(parser.parse_args().output_dir)
