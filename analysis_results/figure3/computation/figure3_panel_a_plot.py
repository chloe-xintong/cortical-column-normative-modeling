#!/usr/bin/env python3
"""Export Figure 3A displays as separate, vertically presented figures.

Description
-----------
Export Figure 3A displays as separate, vertically presented figures.

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
python analysis_results/figure3/computation/figure3_panel_a_plot.py --help
"""
from pathlib import Path
import argparse
import json
import sys

import matplotlib as mpl
if __name__ == '__main__':
    mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qc.publication_style import PALETTE, apply_semantic_style, finalize_figure
from select_representative_regions import (display_region_name, prepare,
                                            select_extreme_regions, REGION_STYLES)


EXTREME_STYLES = {
    'minimum': dict(color=PALETTE['blue'], linestyle='-', marker='o'),
    'median': dict(color=PALETTE['orange'], linestyle='-', marker='s'),
    'maximum': dict(color=PALETTE['red'], linestyle='-', marker='^'),
}


def save(fig, figures, stem):
    finalize_figure(fig)
    for ext in ('pdf','svg','png'):
        fig.savefig(figures/f'{stem}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def whole_cortex_profiles(root):
    """Subject-first means across all valid atlas vertices, then group mean/CI."""
    tables=root/'tables'
    subjects=pd.read_csv(tables/'figure3_subjects.csv',dtype={'subject_id':str})
    mask=np.load(tables/'figure3_analysis_mask.npy').reshape(-1).astype(bool)
    raw=np.load(root/'cache/raw_md.npy',mmap_mode='r')
    thickness=np.load(root/'cache/thickness.npy',mmap_mode='r')
    if raw.shape!=(len(subjects),len(mask),21) or thickness.shape!=(len(subjects),len(mask)):
        raise ValueError('Whole-cortex cache dimensions do not match the frozen cohort and mask')
    rows=[]
    for i,subject in enumerate(subjects.subject_id):
        valid=mask & np.isfinite(thickness[i])
        values=np.asarray(raw[i,valid],dtype=float)
        if not valid.any() or values.shape[1]!=21 or not np.isfinite(values).all():
            raise ValueError(f'{subject}: invalid whole-cortex profile cache')
        mean=values.mean(axis=0)
        rows.extend(dict(subject_id=subject,depth_index=d,normalized_depth=d/20,
                         mean_md=value,n_valid_vertices=int(valid.sum()))
                    for d,value in enumerate(mean))
    subject_table=pd.DataFrame(rows)
    group=(subject_table.groupby(['depth_index','normalized_depth'],sort=True).mean_md
           .agg(mean_md='mean',sd_md='std',n_subjects='count').reset_index())
    if len(group)!=21 or not group.n_subjects.eq(len(subjects)).all():
        raise ValueError('Whole-cortex group profile is incomplete')
    group=add_ci95_columns(group)
    subject_table.to_csv(tables/'figure3A_subject_whole_cortex_profiles.csv',index=False)
    group.to_csv(tables/'figure3A_whole_cortex_group_profile.csv',index=False)
    return group


def profile_values(table,region,n_expected=None):
    rows=table.loc[table.region.eq(region)].sort_values('depth_index')
    values=rows[['normalized_depth','mean_md','sd_md','n_subjects']].to_numpy(float)
    if (values.shape!=(21,4) or not np.isfinite(values).all()
            or not np.allclose(values[:,0],np.linspace(0,1,21))
            or np.any(values[:,2]<0) or np.unique(values[:,3]).size!=1):
        raise ValueError(f'{region}: invalid group profile')
    if n_expected is not None and not np.all(values[:,3]==n_expected):
        raise ValueError(f'{region}: expected {n_expected} subjects at every depth')
    return values


def mean_ci95_halfwidth(sd,n):
    """Two-sided 95% t interval for an across-subject mean."""
    sd=np.asarray(sd,dtype=float);n=np.asarray(n,dtype=float)
    if np.any(~np.isfinite(sd)) or np.any(~np.isfinite(n)) or np.any(sd<0) or np.any(n<2):
        raise ValueError('95% CI requires finite SD and at least two subjects')
    return student_t.ppf(.975,n-1)*sd/np.sqrt(n)


def add_ci95_columns(table):
    result=table.copy()
    result['ci95_halfwidth']=mean_ci95_halfwidth(result.sd_md,result.n_subjects)
    result['ci95_lower']=result.mean_md-result.ci95_halfwidth
    result['ci95_upper']=result.mean_md+result.ci95_halfwidth
    return result


def ci95_values(values):
    display=values.copy()
    display[:,2]=mean_ci95_halfwidth(values[:,2],values[:,3])
    return display


def profile_limits(curves):
    low=min(float(np.min(x[:,1]-x[:,2])) for x in curves)
    high=max(float(np.max(x[:,1]+x[:,2])) for x in curves)
    span=high-low
    if not np.isfinite([low,high]).all() or span<=0:
        raise ValueError('Cannot determine a valid MD axis range')
    return low-.04*span,high+.08*span


def style_profile_axis(ax,ylim,title):
    ax.set(xlim=(0,1),ylim=ylim,xlabel='Normalized depth (0 pial → 1 white)',
           ylabel='MD (original image units)')
    ax.set_title(title,loc='left',fontweight='bold')
    ax.xaxis.set_major_locator(FixedLocator(np.linspace(0,1,6)))
    formatter=ScalarFormatter(useMathText=False);formatter.set_powerlimits((0,0))
    ax.yaxis.set_major_formatter(formatter)


def draw_whole_cortex(group):
    values=group[['normalized_depth','mean_md','sd_md','n_subjects']].to_numpy(float)
    fig,ax=plt.subplots(figsize=(5.7,3.15))
    x,mean,sd,_=values.T
    ax.plot(x,mean,color=PALETTE['blue'],lw=1.9,marker='o',markevery=4)
    ax.fill_between(x,mean-sd,mean+sd,color=PALETTE['blue'],alpha=.12)
    style_profile_axis(ax,profile_limits([values]),'Whole-cortex MD–depth profile')
    return fig


def draw_regional_profiles(table,selection,kind,title,n,ylim=None):
    curves=[ci95_values(profile_values(table,row.region_name,n))
            for row in selection.itertuples()]
    fig,ax=plt.subplots(figsize=(5.7,3.15))
    for row,values in zip(selection.itertuples(),curves):
        style=(REGION_STYLES[row.selected_level] if kind=='quantile'
               else EXTREME_STYLES[row.extreme_level])
        x,mean,sd,_=values.T
        ax.plot(x,mean,**style,lw=1.9,markevery=4)
        ax.fill_between(x,mean-sd,mean+sd,color=style['color'],alpha=.12)
    style_profile_axis(ax,profile_limits(curves) if ylim is None else ylim,title)
    return fig


def shared_profile_limits(raw,adjusted,selection,n):
    curves=[]
    for table in (raw,adjusted):
        curves.extend(ci95_values(profile_values(table,row.region_name,n))
                      for row in selection.itertuples())
    return profile_limits(curves)


def draw_selection_key(ranking,selection,kind):
    """Standalone DK-thickness distribution plus matching curve legend."""
    fig=plt.figure(figsize=(6.1,2.7))
    grid=fig.add_gridspec(2,1,height_ratios=(1.35,1),hspace=.36)
    ax=fig.add_subplot(grid[0])
    ax.scatter(ranking.group_mean_thickness,np.zeros(len(ranking)),s=22,
               color='#B0B0B0',zorder=1)
    handles=[]
    for i,row in enumerate(selection.itertuples()):
        if kind=='quantile':
            level=row.selected_level;color=row.color
            style=REGION_STYLES[level]
            prefix=f'P{int(row.target_percentile)}'
        else:
            level=row.extreme_level;color=row.extreme_color
            style=EXTREME_STYLES[level]
            prefix={'minimum':'Minimum','median':'Median','maximum':'Maximum'}[level]
        ax.scatter(row.group_mean_thickness,0,s=62,color=color,edgecolor='white',zorder=3)
        offset=18 if i!=1 else -24
        ax.annotate(prefix,(row.group_mean_thickness,0),xytext=(0,offset),
                    textcoords='offset points',ha='center',va='bottom' if i!=1 else 'top',
                    arrowprops={'arrowstyle':'-','color':color,'lw':.8})
        handles.append(Line2D([0],[0],**style,lw=1.9,markersize=4,
            label=f'{prefix}: {display_region_name(row.region_name)} '
                  f'({row.group_mean_thickness:.2f} mm)'))
    ax.set(yticks=[],ylim=(-.55,.75),xlabel='Group-mean bilateral DK cortical thickness (mm)')
    ax.spines[['left','right','top']].set_visible(False)
    ax.margins(x=.07)
    legend_ax=fig.add_subplot(grid[1]);legend_ax.axis('off')
    legend_ax.legend(handles=handles,loc='center left',frameon=False,fontsize=8,
                     handlelength=2.7,labelspacing=.55)
    fig.suptitle('Regional thickness distribution',y=.995,fontweight='bold')
    return fig


def export_panel_a_figures(root,ranking=None,selected=None):
    root=Path(root);figures=root/'figures';figures.mkdir(exist_ok=True)
    if ranking is None or selected is None:
        ranking,selected=prepare(root)
    n=len(pd.read_csv(root/'tables/figure3_subjects.csv'))
    raw=pd.read_csv(root/'tables/figure3A_bilateral_group_profiles.csv')
    adjusted=pd.read_csv(root/'tables/figure3D_bilateral_adjusted_group_profiles.csv')
    add_ci95_columns(raw).to_csv(
        root/'tables/figure3A_bilateral_group_profiles_ci95.csv',index=False)
    whole=whole_cortex_profiles(root)
    extremes=select_extreme_regions(ranking)
    qc=pd.read_csv(root/'qc/figure3A_extreme_region_coverage_qc.csv')
    extremes=extremes.merge(qc[['region_name','n_profile_subjects','coverage_fraction']],
                            on='region_name',how='left',validate='one_to_one')
    if extremes.n_profile_subjects.isna().any():
        raise ValueError('Extreme-region coverage QC is incomplete')
    quantile_ylim=shared_profile_limits(raw,adjusted,selected,n)
    extreme_ylim=shared_profile_limits(raw,adjusted,extremes,n)

    save(draw_whole_cortex(whole),figures,'Figure3A_01_whole_cortex_profile')
    save(draw_regional_profiles(raw,selected,'quantile',
         'P10/P50/P90 thickness examples',n,quantile_ylim),
         figures,'Figure3A_02_quantile_region_profiles')
    save(draw_selection_key(ranking,selected,'quantile'),figures,
         'Figure3A_03_quantile_selection_key')
    save(draw_regional_profiles(raw,extremes,'extreme',
         'Minimum/median/maximum thickness examples',n,extreme_ylim),
         figures,'Figure3A_04_extreme_region_profiles')
    save(draw_selection_key(ranking,extremes,'extreme'),figures,
         'Figure3A_05_extreme_selection_key')

    caption=(
        'Figure 3A separate displays. The whole-cortex curve first averages MD over all valid '
        'analysis-mask vertices within each subject and depth, then shows the equal-subject group '
        'mean and between-subject SD. Regional curves use equal left/right regional means '
        'within subject followed by equal-subject summaries. P10/P50/P90 regions and the separate '
        'minimum/median/maximum set are selected solely from group-mean bilateral native mapped '
        'FreeSurfer thickness, before MD coverage is considered. Whole-cortex shading is '
        'between-subject SD for the whole cortex; regional A shading is the two-sided 95% t '
        'confidence interval for the across-subject mean. Raw and adjusted regional plots use '
        'identical axes within each selection set. '
        'All displayed curves use the same coverage-gated cohort; missing cells are not imputed. '
        'Normalized depth runs from pial (0) to white (1).\n')
    (figures/'Figure3A_separate_displays_caption.txt').write_text(caption)
    manifest=dict(status='complete',layout='five separate figures; no horizontal multi-panel row',
        whole_cortex_weighting='Vertex mean within subject and depth, then equal-subject mean',
        whole_cortex_interval='Between-subject SD',
        quantile_regions=selected.region_name.tolist(),
        extreme_regions=extremes.region_name.tolist(),
        extreme_profile_subjects=dict(zip(extremes.region_name,extremes.n_profile_subjects.astype(int))),
        selection_uses_md=False,
        regional_interval='Two-sided 95% t confidence interval for the across-subject mean',
        raw_adjusted_axes='Identical within each selection set',
        figures=[f'Figure3A_0{i}_{name}' for i,name in enumerate((
            'whole_cortex_profile','quantile_region_profiles','quantile_selection_key',
            'extreme_region_profiles','extreme_selection_key'),start=1)])
    (root/'qc/figure3A_separate_plot_manifest.json').write_text(
        json.dumps(manifest,indent=2)+'\n')


def selection_range_summary(table, selection, selection_set, profile_type):
    names=selection.region_name.tolist()
    wide=(table.loc[table.region.isin(names)]
          .pivot(index='depth_index',columns='region',values='mean_md')[names])
    if wide.shape!=(21,3) or not np.isfinite(wide.to_numpy()).all():
        raise ValueError(f'{selection_set} {profile_type}: incomplete mean profiles')
    spread=wide.max(axis=1)-wide.min(axis=1)
    maximum_index=int(spread.to_numpy().argmax())
    return dict(
        selection_set=selection_set, profile_type=profile_type,
        mean_between_region_range=float(spread.mean()),
        maximum_between_region_range=float(spread.iloc[maximum_index]),
        maximum_range_depth=maximum_index/20,
        pial_order=' > '.join(display_region_name(x) for x in
                              wide.iloc[0].sort_values(ascending=False).index),
        white_order=' > '.join(display_region_name(x) for x in
                               wide.iloc[-1].sort_values(ascending=False).index))


def export_panel_d_figures(root,ranking=None,selected=None):
    """Export adjusted profiles for both thickness-only selection sets."""
    root=Path(root);figures=root/'figures';figures.mkdir(exist_ok=True)
    if ranking is None or selected is None:
        ranking,selected=prepare(root)
    n=len(pd.read_csv(root/'tables/figure3_subjects.csv'))
    raw=pd.read_csv(root/'tables/figure3A_bilateral_group_profiles.csv')
    adjusted=pd.read_csv(root/'tables/figure3D_bilateral_adjusted_group_profiles.csv')
    add_ci95_columns(adjusted).to_csv(
        root/'tables/figure3D_bilateral_adjusted_group_profiles_ci95.csv',index=False)
    extremes=select_extreme_regions(ranking)
    for selection in (selected,extremes):
        for region in selection.region_name:
            profile_values(adjusted,region,n)

    quantile_ylim=shared_profile_limits(raw,adjusted,selected,n)
    extreme_ylim=shared_profile_limits(raw,adjusted,extremes,n)

    save(draw_regional_profiles(adjusted,selected,'quantile',
         'Thickness-adjusted P10/P50/P90 profiles',n,quantile_ylim),figures,
         'Figure3D_01_quantile_adjusted_profiles')
    save(draw_regional_profiles(adjusted,extremes,'extreme',
         'Adjusted full-range profiles',n,extreme_ylim),figures,
         'Figure3D_02_extreme_adjusted_profiles')

    rows=[]
    for selection_set,selection in (('P10/P50/P90',selected),
                                     ('minimum/median/maximum',extremes)):
        rows.append(selection_range_summary(raw,selection,selection_set,'raw'))
        rows.append(selection_range_summary(adjusted,selection,selection_set,'adjusted'))
    summary=pd.DataFrame(rows)
    summary.to_csv(root/'tables/figure3D_two_selection_sets_summary.csv',index=False)
    q=summary.loc[summary.selection_set.eq('P10/P50/P90')].set_index('profile_type')
    e=summary.loc[summary.selection_set.eq('minimum/median/maximum')].set_index('profile_type')
    e_change=100*(e.loc['adjusted','mean_between_region_range']/
                  e.loc['raw','mean_between_region_range']-1)
    explanation=(
        'P10/P50/P90: after adjustment, frontal pole (red) remains highest, caudal middle '
        'frontal (orange) remains intermediate, and lateral occipital (blue) remains lowest '
        'across depth. The regional ordering is therefore not removed by the within-subject '
        'linear thickness correction.\n\n'
        'Minimum/median/maximum: pericalcarine (blue) is highest near the pial surface and '
        'temporal pole (red) is lowest; the curves converge toward white matter, where caudal '
        f'middle frontal (orange) becomes slightly highest. Adjustment reduces the descriptive '
        f'mean separation across depth by {abs(e_change):.0f}%, so thickness explains part, but '
        'not all, of the full-range regional contrast. Formal residual regional inference is '
        'reported separately.\n')
    (figures/'Figure3D_two_selection_sets_interpretation.txt').write_text(explanation)
    manifest=dict(
        status='complete', n_subjects=n,
        adjustment='Within-subject depth-specific linear thickness correction',
        interval='Two-sided 95% t confidence interval for the across-subject mean',
        raw_adjusted_axes='Identical within each selection set',
        common_palette=['blue','orange','red'], all_lines_solid=True,
        quantile_regions=selected.region_name.tolist(),
        extreme_regions=extremes.region_name.tolist(),
        figures=['Figure3D_01_quantile_adjusted_profiles',
                 'Figure3D_02_extreme_adjusted_profiles'],
        interpretation_file='Figure3D_two_selection_sets_interpretation.txt',
        interpretation_is_descriptive=True)
    (root/'qc/figure3D_two_selection_sets_manifest.json').write_text(
        json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    apply_semantic_style()
    ranking,selected=prepare(args.output_dir)
    export_panel_a_figures(args.output_dir,ranking,selected)
    export_panel_d_figures(args.output_dir,ranking,selected)
