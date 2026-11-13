#!/usr/bin/env python3
"""Audit the final Figure 3 A–D result set and executed notebook.

Description
-----------
Audit the final Figure 3 A–D result set and executed notebook.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input
- ``--notebook`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure3/computation/audit_figure3_completeness.py --help
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


PRIMARY_FIGURES = (
    'Figure3A_01_whole_cortex_profile',
    'Figure3A_02_quantile_region_profiles',
    'Figure3A_03_quantile_selection_key',
    'Figure3A_04_extreme_region_profiles',
    'Figure3A_05_extreme_selection_key',
    'Figure3B_01_PC1_PC2_loadings',
    'Figure3B_02_explained_variance',
    'Figure3B_03_PC1_PC2_group_score_maps',
    'Figure3C_01_PC1_PC2_thickness_correlations',
    'Figure3C_02_thickness_partial_r2',
    'Figure3D_01_quantile_adjusted_profiles',
    'Figure3D_02_extreme_adjusted_profiles',
    'Figure3D_03_PC1_raw_adjusted_maps',
    'Figure3D_04_PC2_raw_adjusted_maps',
    'Figure3D_05_fixed_component_retention',
)

CURVATURE_SUPPLEMENT_FIGURES = (
    'Figure3S_C01_PC_curvature_spatial_rho',
    'Figure3S_C02_PC_curvature_adjusted_beta',
    'Figure3S_C03_PC_curvature_partial_r2',
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def finite_csv(path, expected_rows=None):
    table = pd.read_csv(path)
    if expected_rows is not None:
        require(len(table) == expected_rows,
                f'{path.name}: expected {expected_rows} rows, found {len(table)}')
    numeric = table.select_dtypes(include=[np.number])
    require(np.isfinite(numeric.to_numpy()).all(), f'{path.name}: nonfinite numeric value')
    return table


def audit(root, notebook):
    tables, qc, figures = root/'tables', root/'qc', root/'figures'
    preparation = json.loads((qc/'preparation.json').read_text())
    analysis = json.loads((qc/'analysis.json').read_text())
    a_manifest = json.loads((qc/'figure3A_separate_plot_manifest.json').read_text())
    d_manifest = json.loads((qc/'figure3D_two_selection_sets_manifest.json').read_text())
    tracking = json.loads((qc/'figure3_fixed_component_tracking.json').read_text())
    subjects = pd.read_csv(tables/'figure3_subjects.csv', dtype={'subject_id': str})
    n = len(subjects)

    require(n == preparation['n_subjects'] == analysis['n_subjects'] == 313,
            'Frozen cohort is not N=313')
    require(preparation['minimum_subject_coverage'] == .95,
            'Subject-level coverage threshold is not 0.95')
    require(analysis['status'] == 'complete', 'Core analysis is incomplete')
    require(analysis['n_missing_subject_region_cells'] == 0,
            'Unexpected missing subject-region cells')

    require(a_manifest['whole_cortex_interval'] == 'Between-subject SD',
            'A whole-cortex interval is not between-subject SD')
    require(a_manifest['regional_interval'].startswith('Two-sided 95% t confidence interval'),
            'A regional interval is not 95% CI')
    require(d_manifest['interval'].startswith('Two-sided 95% t confidence interval'),
            'D regional interval is not 95% CI')
    require(len(a_manifest['quantile_regions']) == len(a_manifest['extreme_regions']) == 3,
            'A does not retain both three-region selection sets')
    require(a_manifest['quantile_regions'] == d_manifest['quantile_regions'],
            'A/D quantile regions differ')
    require(a_manifest['extreme_regions'] == d_manifest['extreme_regions'],
            'A/D full-range regions differ')
    require(a_manifest['raw_adjusted_axes'].startswith('Identical') and
            d_manifest['raw_adjusted_axes'].startswith('Identical'),
            'A/D axes are not locked within selection set')
    require(d_manifest['common_palette'] == ['blue', 'orange', 'red'] and
            d_manifest['all_lines_solid'], 'A/D curve style contract failed')

    whole = finite_csv(tables/'figure3A_whole_cortex_group_profile.csv', 21)
    require(whole.n_subjects.eq(n).all(), 'Whole-cortex profile does not use all subjects')
    a_regions = finite_csv(tables/'figure3A_bilateral_group_profiles_ci95.csv')
    d_regions = finite_csv(tables/'figure3D_bilateral_adjusted_group_profiles_ci95.csv')
    for name, table in [('A', a_regions), ('D', d_regions)]:
        require({'ci95_lower', 'ci95_upper', 'ci95_halfwidth'}.issubset(table.columns),
                f'{name}: missing 95% CI columns')
        require((table.ci95_lower <= table.mean_md).all() and
                (table.mean_md <= table.ci95_upper).all(), f'{name}: invalid CI bounds')

    variance = finite_csv(tables/'figure3B_pca_explained_variance.csv')
    require({1, 2}.issubset(set(variance.component)), 'PC1/PC2 variance missing')
    finite_csv(tables/'figure3B_pca_loadings.csv', 21)
    effects = finite_csv(tables/'figure3C_depth_specific_thickness_effect.csv', 21)
    require({'partial_r2_ci95_lower', 'partial_r2_ci95_upper',
             'beta_ci95_lower', 'beta_ci95_upper'}.issubset(effects.columns),
            'C depth-effect intervals missing')
    pc_summary = finite_csv(root/'pc1_pc2_thickness/component_correlation_summary.csv', 2)
    require(set(pc_summary.component) == {'PC1', 'PC2'} and pc_summary.n_subjects.eq(n).all(),
            'C PC1/PC2 subject summary is incomplete')

    require(tracking['status'] == 'complete' and tracking['components'] == ['PC1', 'PC2'],
            'Fixed-component tracking is incomplete')
    require(tracking['loading_source'] == 'Original frozen PCA loadings' and
            tracking['centering_reference'].startswith('Original pca_feature_mean'),
            'D did not freeze the original PCA reference')
    require(tracking.get('map_colormap', '').startswith('RdBu_r') and
            tracking.get('map_normalization', '').startswith('symmetric around zero'),
            'B/D score maps do not use the final Figure 2 RdBu_r color standard')
    b_scales = tracking.get('B_component_display_vmax', {})
    d_scales = tracking.get('D_component_display_vmax', {})
    require(set(b_scales) == {'PC1', 'PC2'} and
            all(np.isfinite(list(b_scales.values()))) and
            all(value > 0 for value in b_scales.values()) and
            not np.isclose(b_scales['PC1'], b_scales['PC2']),
            'B does not use valid separate PC1/PC2 display ranges')
    require(set(d_scales) == {'PC1', 'PC2'} and
            all(np.isfinite(list(d_scales.values()))) and
            all(value > 0 for value in d_scales.values()),
            'D fixed within-component display ranges are missing')
    require(all(np.isclose(b_scales[component], d_scales[component],
                           rtol=0, atol=1e-15)
                for component in ('PC1', 'PC2')),
            'B and D do not reuse the same fixed range within each PC')
    require((figures/'Figure3B_PC1_PC2_group_score_maps_caption.txt').is_file(),
            'B separate-scale caption is missing')
    finite_csv(tables/'figure3D_fixed_component_subject_retention.csv', 2*n)
    finite_csv(tables/'figure3D_fixed_component_retention_summary.csv', 4)
    finite_csv(tables/'figure3D_fixed_component_group_map_retention.csv', 2)

    curvature_root = root/'curvature_supplement'
    curvature_status = json.loads((curvature_root/'complete.json').read_text())
    require(curvature_status['status'] == 'complete' and
            curvature_status['n_subjects'] == n and
            curvature_status['statistical_unit'] == 'subject',
            'Curvature supplement is incomplete or uses the wrong statistical unit')
    require(curvature_status['primary_analysis_changed'] is False,
            'Curvature supplement unexpectedly changed the primary analysis')
    require(curvature_status['legacy_signed_max_abs_difference'] == 0 and
            curvature_status['legacy_absolute_max_abs_difference'] == 0,
            'Curvature mapping differs from the independent legacy validation')
    curvature_metrics = finite_csv(
        curvature_root/'subject_pc_curvature_metrics.csv', 4*n)
    require(curvature_metrics.subject_id.nunique() == n and
            set(curvature_metrics.component) == {'PC1', 'PC2'} and
            set(curvature_metrics.curvature_metric) == {'signed', 'absolute'},
            'Subject-level curvature metrics are incomplete')
    finite_csv(curvature_root/'pc_curvature_group_summary.csv', 12)

    missing_files = []
    for stem in PRIMARY_FIGURES:
        for extension in ('png', 'pdf', 'svg'):
            path = figures/f'{stem}.{extension}'
            if not path.is_file() or path.stat().st_size < 1000:
                missing_files.append(str(path))
    for stem in CURVATURE_SUPPLEMENT_FIGURES:
        for extension in ('png', 'pdf', 'svg'):
            path = figures/f'{stem}.{extension}'
            if not path.is_file() or path.stat().st_size < 1000:
                missing_files.append(str(path))
    for component in (1, 2):
        for hemisphere in ('lh', 'rh'):
            for condition in ('raw', 'adjusted'):
                path = root/'surface_maps'/(
                    f'{hemisphere}.figure3_PC{component}_{condition}_fixed_mean.mgh')
                if not path.is_file() or path.stat().st_size < 1000:
                    missing_files.append(str(path))
    require(not missing_files, 'Missing or empty outputs: ' + ', '.join(missing_files))

    nb = json.loads(notebook.read_text())
    errors = [output for cell in nb['cells'] if cell.get('cell_type') == 'code'
              for output in cell.get('outputs', []) if output.get('output_type') == 'error']
    code_cells = [cell for cell in nb['cells'] if cell.get('cell_type') == 'code']
    require(not errors, 'Executed notebook contains errors')
    require(all(cell.get('execution_count') is not None for cell in code_cells),
            'Notebook contains unexecuted code cells')
    source = '\n'.join(''.join(cell.get('source', [])) for cell in nb['cells'])
    for stem in PRIMARY_FIGURES:
        require(stem in source, f'Notebook does not present {stem}')
    for stem in CURVATURE_SUPPLEMENT_FIGURES:
        require(stem in source, f'Notebook does not present {stem}')

    return {
        'status': 'complete',
        'n_subjects': n,
        'subject_coverage_threshold': preparation['minimum_subject_coverage'],
        'panel_A': {
            'whole_cortex_interval': 'between-subject SD',
            'regional_interval': '95% t CI',
            'selection_sets': ['P10/P50/P90', 'minimum/median/maximum'],
        },
        'panel_B': ['PC1/PC2 loading curves',
                    'separate PC1-PC5 explained-variance bars',
                    'PC1/PC2 group cortical score maps'],
        'panel_C': ['PC1/PC2 within-subject thickness correlations with CI',
                    'depth-specific thickness partial R2 with CI'],
        'panel_C_curvature_supplement': {
            'status': 'complete',
            'n_subjects': n,
            'metrics': ['signed curvature', 'absolute curvature'],
            'model': 'PC score ~ thickness + curvature within subject',
            'inference': 'subject bootstrap',
            'primary_analysis_changed': False,
        },
        'panel_D': ['two A-matched adjusted regional curve sets with 95% CI',
                    'fixed original PC1/PC2 maps before/after adjustment',
                    'subject spatial correlation and adjusted/raw spatial SD'],
        'surface_map_colormap': 'RdBu_r (same as final Figure 2)',
        'surface_map_normalization': 'separate PC1/PC2 symmetric ranges; same PC range reused across B and D',
        'B_component_display_vmax': b_scales,
        'D_component_display_vmax': d_scales,
        'primary_figure_stems': len(PRIMARY_FIGURES),
        'primary_figure_files': len(PRIMARY_FIGURES)*3,
        'curvature_supplement_figure_files': len(CURVATURE_SUPPLEMENT_FIGURES)*3,
        'fixed_component_surface_maps': 8,
        'notebook_code_cells_executed': len(code_cells),
        'missing_required_outputs': [],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--notebook', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output_dir, args.notebook)
    destination = args.output_dir/'qc/figure3_completeness_audit.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
