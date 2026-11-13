"""Export a grayscale/CVD review sheet for the fixed scientific encodings.

Description
-----------
CVD: Machado et al. (2009), severity 100, applied in linear sRGB.
Matrix source: https://raw.githubusercontent.com/njsmith/colorspacious/master/colorspacious/cvd.py
This diagnostic checks redundant encodings; it is not a universal accessibility certification.

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
python analysis_results/check_semantic_palette.py --help
"""
from pathlib import Path
import sys
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from qc.publication_style import (
    PALETTE, SERIES_COLORS, MODEL_COLORS, METHOD_COLORS, COMPARISON_COLORS,
    REGION_STYLES, PROFILE_STYLES, DERIVED_STYLE, DERIVED_BAND, set_publication_style,
)

MATRICES = {
    'Protanopia': np.array([[.152286,1.052583,-.204868],
                            [.114503,.786281,.099216],[-.003882,-.048116,1.051998]]),
    'Deuteranopia': np.array([[.367322,.860646,-.227968],
                             [.280085,.672501,.047413],[-.011820,.042940,.968881]]),
}


def simulate(color, mode):
    srgb = np.array(to_rgb(color))
    linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
    if mode == 'Grayscale':
        transformed = np.repeat(linear @ [.2126,.7152,.0722], 3)
    elif mode in MATRICES:
        transformed = MATRICES[mode] @ linear
    else:
        return srgb
    transformed = np.clip(transformed, 0, 1)
    return np.where(transformed <= .0031308, transformed * 12.92,
                    1.055 * transformed ** (1/2.4) - .055)


def main():
    out = REPO / '/mnt/newStor/paros/paros_WORK/xintong/result/cortical_trajectory_analysis/analysis_results/figure_style_qc'
    out.mkdir(exist_ok=True)
    assert PALETTE['red'] not in SERIES_COLORS and PALETTE['pale_blue'] not in SERIES_COLORS
    assert MODEL_COLORS['ico4_md_length'] == PALETTE['red']
    assert PALETTE['pale_blue'] not in MODEL_COLORS.values()
    roles = [
        ('Primary real', dict(color=PALETTE['red'], linestyle='-', marker='o')),
        ('Technical null', dict(color=PALETTE['blue'], linestyle='--', marker='s')),
        ('Derived comparison', dict(color=PALETTE['orange'], linestyle='-', marker='o')),
        *[(name, style) for name, style in REGION_STYLES.items()],
    ]
    set_publication_style()
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharey=True, constrained_layout=True)
    for ax, mode in zip(axes, ['Original', 'Grayscale', *MATRICES]):
        for i, (name, style) in enumerate(roles):
            visual = dict(style, color=simulate(style['color'], mode))
            ax.plot([0,.5,1], [i,i,i], **visual, linewidth=1.8)
        ax.set(title=mode, xticks=[], yticks=range(len(roles)),
               yticklabels=[name for name, _ in roles], xlim=(-.05,1.05))
        ax.spines[['top','right','bottom']].set_visible(False)
    for ext in ['png','pdf','svg']:
        fig.savefig(out / f'semantic_palette_accessibility.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    policy = dict(
        status='complete', palette=PALETTE, default_categories=SERIES_COLORS,
        method_colors=METHOD_COLORS, comparison_colors=COMPARISON_COLORS,
        model_colors=MODEL_COLORS, region_styles=REGION_STYLES, profile_styles=PROFILE_STYLES,
        derived_style=DERIVED_STYLE, derived_band=DERIVED_BAND,
        rules={
            'single_series': 'Deep-blue data with a faint same-color interval band, alpha 0.12.',
            'emphasis': 'Red highlights the fixed final ico4 MD + length model or the primary real-vs-null object; never statistical significance.',
            'paired_series': 'Primary red solid/circle, secondary deep-blue dashed/square; peers blue/orange with distinct line styles.',
            'method_identity': 'Named method/model mappings take precedence over the anonymous single-series default.',
            'intervals': 'Use pale/light blue for backgrounds and intervals; pale blue is forbidden for thin data marks.',
            'text': 'Neutral #333333, Arial/Helvetica with Liberation Sans fallback; no colored labels.',
            'derived_results': 'Orange represents curves or points derived by comparing, correcting or subtracting Real and Null/reference. Derived curve width 1.9 pt and same-orange interval alpha 0.16; heatmaps retain their independent numeric rules.',
            'statistics': 'p/q values, confidence intervals and neutral text retain inference; color is not a significance flag.',
            'heatmaps': 'Keep standard scientific colormaps independently of categorical colors; nonnegative maps use project_blue (low pale to high dark); signed maps use project_diverging with neutral #F7F7F7 at zero, symmetric shared limits and gray missing values.',
            'panel_letters': 'Do not display A/B/C/D or supplementary panel letters in plots; retain descriptive titles and stable filenames.',
            'paper_style': 'No background grids, only left/bottom axes, 0.8-pt dark-gray spines, ~1.8-pt lines, ~3-pt markers, alpha 0.12 intervals; paper-sized fonts.',
            'prior_layout': 'Keep Figure 2 method pairs separate, dark box bodies/orange medians/no scatter; no N in Figures 2 and 4; separate Figure 4A metrics; no Figure 4B threshold line.',
        },
        accessibility_review='Original, linear-sRGB luminance grayscale, Machado protanopia/deuteranopia severity 100. Red and blue still require shape/dash redundancy.',
        sources=['https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/',
                 'https://raw.githubusercontent.com/njsmith/colorspacious/master/colorspacious/cvd.py'],
    )
    (out / 'semantic_palette_policy.json').write_text(json.dumps(policy, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
