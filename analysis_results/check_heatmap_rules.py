"""Validate scientific heatmap encodings and export their display specification.

Description
-----------
Validate scientific heatmap encodings and export their display specification.

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
python analysis_results/check_heatmap_rules.py --help
"""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from qc.publication_style import (
    PALETTE, CMAP_SEQUENTIAL, CMAP_DIVERGING, heatmap_norm, set_publication_style,
)


def main():
    out = REPO / '/mnt/newStor/paros/paros_WORK/xintong/result/cortical_trajectory_analysis/analysis_results/figure_style_qc'
    out.mkdir(exist_ok=True)
    signed = heatmap_norm('signed', vmax=2)
    assert signed(0) == .5 and signed(-2) == 0 and signed(2) == 1
    np.testing.assert_allclose(CMAP_DIVERGING(signed(0)), to_rgba('#F7F7F7'))
    correlation = heatmap_norm('correlation')
    assert (correlation.vmin, correlation.vcenter, correlation.vmax) == (-1, 0, 1)
    for cmap in (CMAP_SEQUENTIAL, CMAP_DIVERGING):
        np.testing.assert_allclose(cmap(np.ma.masked), to_rgba(PALETTE['missing']))
    sequential_rgb = CMAP_SEQUENTIAL(np.linspace(0, 1, 257))[:, :3]
    luminance = sequential_rgb @ [.2126, .7152, .0722]
    assert np.all(np.diff(luminance) < 0), 'High values must consistently be darker'
    set_publication_style()
    fig, axes = plt.subplots(3, 1, figsize=(7.09, 3.3), constrained_layout=True)
    for ax, cmap, norm, label in [
        (axes[0], plt.colormaps['RdBu_r'], heatmap_norm('nonnegative', vmin=.8, vmax=1),
         'O2-FS profile agreement, r (red–blue display limits 0.8–1)'),
        (axes[1], plt.colormaps['RdBu_r'], heatmap_norm('nonnegative', vmin=.98, vmax=1),
         'O2-Column profile agreement, r (red–blue display limits 0.98–1)'),
        (axes[2], CMAP_DIVERGING, correlation,
         'Signed correlation, r (zero = neutral; fixed limits −1 to 1)'),
    ]:
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax,
                     orientation='horizontal', label=label)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'heatmap_color_rules.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    policy = dict(
        status='complete', checks=['Exact neutral zero', 'Symmetric signed limits',
            'Fixed correlation limits', 'Gray missing values', 'Monotonic sequential darkness'],
        nonnegative=dict(cmap='project_blue', stops=['#E6F1F3','#90BEE0','#4B74B2'],
            direction='Higher values are darker; never reverse similarity versus distance',
            applies_to=['Nonnegative MD/thickness', 'Distance/dissimilarity', 'Nonnegative similarity/agreement',
                       'MAE/RMSE', 'Coverage/proportions', 'Nonnegative effects', 'p/q values']),
        signed=dict(cmap='project_diverging', stops=['#4B74B2','#90BEE0','#E6F1F3',
                     '#F7F7F7','#FFDF92','#FC8C5A','#DB3124'],
            negative='Blue', zero='#F7F7F7', positive='Red', norm='TwoSlopeNorm centered at zero with symmetric limits',
            correlation_matrix_limits=[-1,0,1]),
        shared_scales='Comparable panels must share cmap, vmin, vmax and norm. Any percentile clipping uses pooled panels; document thresholds; never change original data.',
        figure2='User-requested red-blue override: RdBu_r, O2-Column r=0.98–1.00 and O2-FS r=0.8–1.00; one canonical version per comparison, no old O2-Column scale or separate detail copy. Limits shared within comparison, source data unchanged.',
        figure3='Signed PC1 mean and Cohen d; one pooled bilateral absolute 98th percentile per metric; all views share symmetric limits; exact limits in qc/heatmap_display_scales.json.',
        differences='Keep delta D = D_across - D_within. Red indicates greater across-region dissimilarity; blue indicates greater within-region dissimilarity. Never flip subtraction to obtain red.',
        missing='#D9D9D9; mask, do not replace with zero or minimum',
        symmetric_matrix_layout='Use lower triangle and mask constant diagonal where appropriate; hemisphere/anatomical DK order; thin hemisphere separators; no dense cell borders or value labels; complete metric colorbar labels.',
    )
    (out / 'heatmap_rules.json').write_text(json.dumps(policy, indent=2) + '\n')
    print('Heatmap contracts verified:', out)


if __name__ == '__main__':
    main()
