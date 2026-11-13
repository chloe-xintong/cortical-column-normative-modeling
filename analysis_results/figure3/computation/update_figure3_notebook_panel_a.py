#!/usr/bin/env python3
"""Replace Figure 3A notebook presentation with the separate vertical exports.

Description
-----------
Replace Figure 3A notebook presentation with the separate vertical exports.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``notebook`` command-line input

Outputs
-------
- In-memory values returned to the calling code.

Usage
-----
python analysis_results/figure3/computation/update_figure3_notebook_panel_a.py --help
"""
from pathlib import Path
import argparse

import nbformat


PANEL_A_MARKDOWN = r"""## A · Whole-cortex and thickness-selected regional profiles

Panel A is presented as a vertical sequence of independent figures rather than a horizontal multipanel row. First, the whole-cortex curve summarizes the overall pial-to-white MD pattern: valid analysis-mask vertices are averaged within each subject and depth, followed by an equal-subject group mean and between-subject SD.

Two thickness-only regional selections follow. The first uses the regions nearest the linear P10/P50/P90 group-mean bilateral FreeSurfer thickness quantiles, excluding the extrema. The second spans the complete observed range using the true minimum, the region nearest the linear median, and the true maximum. Selection is frozen before MD profiles or MD coverage are inspected. Each curve figure has its own standalone DK-thickness selection and curve key. Both sets use the same blue–orange–red mapping from lower to higher thickness.

Bilateral regional values are equal left/right means within subject, followed by equal-subject summaries. The same prespecified subject-level coverage gate is applied to the complete analysis cohort, and missing bilateral cells are not imputed.
"""


COHORT_MARKDOWN = r"""## Cohort, coverage, and missing-region QC

A profile is invalid as a whole if any depth is NaN, Inf, nonpositive, or invalid under the existing Option 3 QC mask. Missing depths are never interpolated or imputed. Native FreeSurfer cortical thickness is mapped with the stored Option 3 sphere triangles and barycentric weights; column length is a distinct sensitivity variable.

After the fixed registration-QC exclusions, the primary Figure 3 cohort uses the same subject-level rule as Figure 5: bilateral `sum(in_analysis_mask & column_valid) / sum(in_analysis_mask) ≥ 0.95`. Regional fixed-effects models use observed subject–region cells only. Bootstrap resamples subjects, not vertices.
"""


PANEL_A_CODE = r"""panel_a_manifest = json.loads((QC / 'figure3A_separate_plot_manifest.json').read_text())
assert panel_a_manifest['status'] == 'complete'
assert panel_a_manifest['layout'] == 'five separate figures; no horizontal multi-panel row'
assert panel_a_manifest['selection_uses_md'] is False
assert panel_a_manifest['whole_cortex_interval'] == 'Between-subject SD'
assert panel_a_manifest['regional_interval'].startswith('Two-sided 95% t confidence interval')

panel_a_displays = [
    ('Whole-cortex mean MD–depth profile', 'Figure3A_01_whole_cortex_profile'),
    ('P10/P50/P90 regional examples', 'Figure3A_02_quantile_region_profiles'),
    ('P10/P50/P90 DK-thickness selection and curve key', 'Figure3A_03_quantile_selection_key'),
    ('Minimum/median/maximum regional examples', 'Figure3A_04_extreme_region_profiles'),
    ('Minimum/median/maximum DK-thickness selection and curve key', 'Figure3A_05_extreme_selection_key'),
]
for title, stem in panel_a_displays:
    path = FIGURES / f'{stem}.png'
    if not path.is_file():
        raise RuntimeError(f'Missing separate Figure 3A export: {path}')
    display(Markdown(f'### {title}'))
    display(Image(filename=str(path), width=760))

display(Markdown('**Caption.** ' + (FIGURES / 'Figure3A_separate_displays_caption.txt').read_text()))

whole_cortex = pd.read_csv(TABLES / 'figure3A_whole_cortex_group_profile.csv')
assert len(whole_cortex) == 21 and whole_cortex.n_subjects.eq(N).all()
assert (whole_cortex.ci95_lower <= whole_cortex.mean_md).all()
assert (whole_cortex.ci95_upper >= whole_cortex.mean_md).all()
display(Markdown('### Whole-cortex numerical summary'))
display(whole_cortex.loc[whole_cortex.depth_index.isin([0, 5, 10, 15, 20])])

ranking = pd.read_csv(TABLES / 'figure3A_region_thickness_ranking.csv')
selection = ranking.loc[ranking.selected_for_figure3].sort_values('target_percentile')
selected = selection.region_name.tolist()
extreme_selection = json.loads((QC / 'figure3A_extreme_region_selection.json').read_text())
assert extreme_selection['selection_uses_md'] is False
extreme_qc = pd.read_csv(QC / 'figure3A_extreme_region_coverage_qc.csv')
assert extreme_qc.complete_21_depth_grid.all() and extreme_qc.no_imputation.all()
assert set(extreme_qc.region_name) == set(panel_a_manifest['extreme_regions'])
display(Markdown('### Thickness-only selections and coverage QC'))
display(selection[['region_name', 'selected_level', 'group_mean_thickness', 'thickness_rank',
                   'target_percentile', 'thickness_percentile', 'n_thickness_subjects', 'color']])
display(extreme_qc)
"""


PANEL_D_MARKDOWN = r"""## D · Thickness-adjusted profiles for both thickness selections

Thickness adjustment is shown separately for the same two thickness-only selections used in A: P10/P50/P90 and minimum/median/maximum. Region choice, blue–orange–red mapping, bilateral weighting and subject weighting are unchanged. The adjustment is

$$MD^{adj}_{s,v,d}=MD_{s,v,d}-\beta_d(T_{s,v}-\bar T_s).$$

It retains the original MD units and each subject's overall MD level. All A and D curve shading is the two-sided 95% t confidence interval for the across-subject mean. This operation is separate from the within-profile centering used for PCA. The brief interpretation below is descriptive; formal inference on residual regional structure is reported separately.
"""


PANEL_D_CODE = r"""panel_d_manifest = json.loads((QC / 'figure3D_two_selection_sets_manifest.json').read_text())
assert panel_d_manifest['status'] == 'complete'
assert panel_d_manifest['n_subjects'] == N
assert panel_d_manifest['common_palette'] == ['blue', 'orange', 'red']
assert panel_d_manifest['all_lines_solid'] is True
assert panel_d_manifest['interval'].startswith('Two-sided 95% t confidence interval')

panel_d_displays = [
    ('P10/P50/P90 after thickness adjustment', 'Figure3D_01_quantile_adjusted_profiles'),
    ('Minimum/median/maximum after thickness adjustment', 'Figure3D_02_extreme_adjusted_profiles'),
]
for title, stem in panel_d_displays:
    path = FIGURES / f'{stem}.png'
    if not path.is_file():
        raise RuntimeError(f'Missing Figure 3D export: {path}')
    display(Markdown(f'### {title}'))
    display(Image(filename=str(path), width=760))

summary = pd.read_csv(TABLES / 'figure3D_two_selection_sets_summary.csv')
assert len(summary) == 4 and set(summary.profile_type) == {'raw', 'adjusted'}
display(Markdown('### Interpretation'))
display(Markdown((FIGURES / panel_d_manifest['interpretation_file']).read_text()))
"""


def replace_cell(cells, headings, source, cell_type):
    if isinstance(headings,str):
        headings=(headings,)
    matches=[i for i,cell in enumerate(cells)
             if any(cell['source'].lstrip().startswith(heading) for heading in headings)]
    if len(matches)!=1:
        raise RuntimeError(f'Expected one cell beginning one of {headings!r}, found {matches}')
    index=matches[0]
    cells[index]=(nbformat.v4.new_markdown_cell(source) if cell_type=='markdown'
                  else nbformat.v4.new_code_cell(source))


def update(path):
    notebook=nbformat.read(path,as_version=4)
    cells=notebook.cells
    replace_cell(cells,'## Cohort, coverage, and missing-region QC',
                 COHORT_MARKDOWN,'markdown')
    replace_cell(cells,('## Full Figure 3','## A · Whole-cortex and thickness-selected'),
                 PANEL_A_MARKDOWN,'markdown')
    full_figure_code=next(i for i,cell in enumerate(cells)
                          if ("Figure3_depth_profiles_thickness.png" in cell['source']
                              or "panel_a_manifest =" in cell['source']))
    cells[full_figure_code]=nbformat.v4.new_code_cell(PANEL_A_CODE)
    replace_cell(cells,('## A and D ·','## D · Thickness-adjusted profiles for the frozen',
                        '## D · Thickness-adjusted profiles for both'),
                 PANEL_D_MARKDOWN,'markdown')
    old_ad_code=next(i for i,cell in enumerate(cells)
                     if ("Figure3_AD_regional_profiles.png" in cell['source']
                         or ("selection_qc =" in cell['source'] and "adjusted =" in cell['source'])
                         or "panel_d_manifest =" in cell['source']))
    cells[old_ad_code]=nbformat.v4.new_code_cell(PANEL_D_CODE)
    header=cells[0]['source']
    old=('All panels share the QC-passed cohort, original fsaverage analysis mask, exact DK labels, '
         'pial-to-white depth order, and whole-profile validity rule.')
    old_expanded=(old+' Panel A is displayed as five separate vertically ordered figures: whole cortex, '
                  'P10/P50/P90 curves and key, and minimum/median/maximum curves and key.')
    new=('All panels share the registration-QC-passed, ≥95% subject-coverage cohort, original '
         'fsaverage analysis mask, exact DK labels, pial-to-white depth order, and whole-profile '
         'validity rule. Panel A is displayed as five separate vertically ordered figures: whole cortex, '
         'P10/P50/P90 curves and key, and minimum/median/maximum curves and key.')
    if new not in header:
        if old_expanded in header:
            cells[0]['source']=header.replace(old_expanded,new)
        elif old in header:
            cells[0]['source']=header.replace(old,new)
        else:
            raise RuntimeError('Notebook header text changed; refusing an ambiguous replacement')
    setup=cells[1]['source']
    cells[1]['source']=setup.replace('submission_20261102.json','submission_20261108.json')
    # The cluster's older nbconvert schema rejects the optional cell-id field
    # emitted by newer nbformat releases. Cell ordering is the stable identity
    # used by this notebook, so remove ids for backward-compatible execution.
    for cell in cells:
        cell.pop('id',None)
    nbformat.write(notebook,path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook',type=Path)
    args=parser.parse_args()
    update(args.notebook)
