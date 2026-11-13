#!/usr/bin/env python3
"""Plot the completed Option 2-FS comparison for the Figure 5 supplement.

Description
-----------
Plot the completed Option 2-FS comparison for the Figure 5 supplement.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--comparison-dir`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure5/scripts/plot_option2_fs_supplement.py --help
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qc.publication_style import MODEL_COLORS, set_publication_style, finalize_figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
COMPARISON = Path('/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs/option2_fs_baselines/comparison')
OUTPUT = REPO / '/mnt/newStor/paros/paros_WORK/xintong/result/cortical_trajectory_analysis/analysis_results/figure5/outputs/figures/supplement'
MODELS = ['o2fs_ico4_md', 'ico4_md', 'o2fs_ico4_md_length', 'ico4_md_length']
LABELS = ['Option 2-FS\nMD', 'Option 3\nMD', 'Option 2-FS\nMD +\nlength', 'Option 3\nMD +\nlength']
COLORS = [MODEL_COLORS[model] for model in MODELS]
PAIRS = [('o2fs_ico4_md', 'ico4_md'), ('o2fs_ico4_md_length', 'ico4_md_length'),
         ('o2fs_ico4_md', 'o2fs_ico4_md_length')]


def make_figure(comparison=COMPARISON, output=OUTPUT):
    comparison, output = Path(comparison), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(comparison / 'model_metrics.csv').set_index('model').loc[MODELS]
    paired = pd.read_csv(comparison / 'paired_bootstrap_differences.csv').set_index(['model_a', 'model_b']).loc[PAIRS].reset_index()
    rng = np.random.default_rng(20261102)
    indices = rng.integers(0, 196, (10000, 196))
    ci_rows = []
    reference = None
    for model in MODELS:
        p = pd.read_csv(comparison / f'{model}_oof_predictions.csv')
        if len(p) != 196 or p.subject_id.duplicated().any() or not np.isfinite(p.predicted_age).all():
            raise ValueError(f'Incomplete predictions: {model}')
        alignment = p[['subject_id', 'row_index', 'true_age', 'outer_fold']]
        if reference is None:
            reference = alignment
        elif not reference.equals(alignment):
            raise ValueError('Paired subject/fold order mismatch')
        errors = np.abs(p.predicted_age.to_numpy() - p.true_age.to_numpy())
        if not np.isclose(errors.mean(), metrics.loc[model, 'mae']):
            raise ValueError('Saved metric differs from OOF predictions')
        low, high = np.percentile(errors[indices].mean(axis=1), [2.5, 97.5])
        ci_rows.append(dict(model=model, mae=errors.mean(), ci95_lower=low, ci95_upper=high))
    ci = pd.DataFrame(ci_rows).set_index('model')
    for row in paired.itertuples():
        if not np.isclose(row.delta_mae_a_minus_b,
                          metrics.loc[row.model_a, 'mae'] - metrics.loc[row.model_b, 'mae']):
            raise ValueError('Paired difference differs from model MAEs')
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial','Helvetica','Liberation Sans','DejaVu Sans'], 'font.size': 14, 'axes.labelsize': 17,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'axes.linewidth': 1.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.8), gridspec_kw={'width_ratios': [1, 1.2]})
    a, b = axes
    for i, model in enumerate(MODELS):
        row = ci.loc[model]
        a.errorbar(i, row.mae, yerr=[[row.mae-row.ci95_lower], [row.ci95_upper-row.mae]],
                   fmt='o' if 'length' not in model else 's', color=COLORS[i],
                   markeredgecolor='none', markersize=3.2, ecolor='#555555',
                   elinewidth=.8, capsize=2)
        a.annotate(f'{row.mae:.3f}', (i, row.ci95_upper), xytext=(0, 9),
                   textcoords='offset points', ha='center', fontsize=7)
    a.set_xticks(range(4), LABELS)
    a.set_xlim(-0.5, 3.5)
    a.set_ylim(ci.ci95_lower.min() - .15, ci.ci95_upper.max() + .3)
    a.set_ylabel('OOF MAE (years)')
    a.set_title('ico4 age-prediction error', loc='left', fontweight='bold', fontsize=9, pad=10)
    a.grid(False)
    pair_labels = ['O2-FS − O3: MD', 'O2-FS − O3: MD + length', 'O2-FS: MD − (MD + length)']
    for i, row in enumerate(paired.itertuples()):
        b.errorbar(row.delta_mae_a_minus_b, i,
                   xerr=[[row.delta_mae_a_minus_b-row.ci95_lower], [row.ci95_upper-row.delta_mae_a_minus_b]],
                   fmt='o', color='#4B74B2', markersize=3.2, elinewidth=.8, capsize=2)
        b.annotate(f'{row.delta_mae_a_minus_b:+.3f} [{row.ci95_lower:+.3f}, {row.ci95_upper:+.3f}]',
                   (row.delta_mae_a_minus_b, i), xytext=(0, 13),
                   textcoords='offset points', ha='center', fontsize=7)
    b.axvline(0, color='#888888', linestyle='--', linewidth=1.3)
    b.set_yticks(range(3), pair_labels)
    b.set_ylim(2.6, -.65)
    b.set_xlim(min(-.16, paired.ci95_lower.min() - .06), paired.ci95_upper.max() + .07)
    b.set_xlabel('Paired MAE difference (years)')
    b.set_title('Paired differences (95% CI)', loc='left', fontweight='bold', fontsize=9, pad=10)
    b.grid(False)
    fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.86, wspace=.72)
    fig.text(.5, .035, '196 CN participants · shared ico4 mask and nested 5 × 5 folds · LSQR Ridge\n'
             '10,000 paired-subject bootstrap samples; intervals conditional on the fixed fitted folds.',
             ha='center', fontsize=7, color='#555555')
    stem = 'Figure5S_Option2FS_ico4_baselines'
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'{stem}.{ext}', dpi=600, bbox_inches='tight')
    metrics.to_csv(output / f'{stem}_metrics.csv')
    ci.to_csv(output / f'{stem}_MAE_CI.csv')
    paired.to_csv(output / f'{stem}_paired_bootstrap.csv', index=False)
    return fig, metrics.reset_index(), paired


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir', type=Path, default=COMPARISON)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    fig, _, _ = make_figure(args.comparison_dir, args.output_dir)
    plt.close(fig)
    print(args.output_dir)
