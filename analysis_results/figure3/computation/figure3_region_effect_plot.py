"""Cached Figure 3D regional-effect display with fixed raw/adjusted colors.

Description
-----------
Cached Figure 3D regional-effect display with fixed raw/adjusted colors.

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
python analysis_results/figure3/computation/figure3_region_effect_plot.py --help
"""
from pathlib import Path
import sys
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qc.publication_style import PROFILE_STYLES, set_publication_style, finalize_figure


def draw_region_effect(ax, regional):
    if len(regional) != 21 or not np.allclose(regional.normalized_depth, np.linspace(0, 1, 21)):
        raise ValueError('Expected the existing 21 pial-to-white depths')
    for key, label in [('raw', 'Raw MD'), ('adjusted', 'Thickness adjusted')]:
        columns = [f'{key}_partial_r2', f'{key}_ci95_lower', f'{key}_ci95_upper']
        if not np.isfinite(regional[columns].to_numpy(float)).all():
            raise ValueError('Nonfinite saved regional effects or confidence intervals')
        # Keep color and marker semantics, but use solid lines for both raw
        # and thickness-adjusted regional-effect curves in this display.
        style = {**PROFILE_STYLES[key], 'linestyle': '-'}
        ax.plot(regional.normalized_depth, regional[columns[0]], **style,
                markevery=4, linewidth=1.8, label=label)
        ax.fill_between(regional.normalized_depth, regional[columns[1]], regional[columns[2]],
                        color=style['color'], alpha=.12, linewidth=0)
    ax.set(xlabel='Normalized depth (pial → white)', ylabel='Region partial R²',
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    return ax


def export(root):
    root = Path(root)
    table = pd.read_csv(root / 'tables/figure3D_region_effect_by_depth.csv')
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7.09, 3.2))
    draw_region_effect(ax, table)
    finalize_figure(fig)
    fig.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(root / 'figures' / f'Figure3D_raw_adjusted_region_effect.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    export(parser.parse_args().output_dir)
