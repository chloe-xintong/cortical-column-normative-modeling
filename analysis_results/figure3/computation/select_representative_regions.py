#!/usr/bin/env python3
"""Select A/D regions exclusively from bilateral FreeSurfer thickness.

Description
-----------
Select A/D regions exclusively from bilateral FreeSurfer thickness.

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
python analysis_results/figure3/computation/select_representative_regions.py --help
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qc.publication_style import PALETTE
import argparse
import json
import numpy as np
import pandas as pd

LEVELS = ('thin','intermediate','thick')
# Figure 3 uses only the established project palette. The quantile examples
# use its lighter colors; the full-range examples use disjoint darker colors.
REGION_STYLES = {
    'thin': dict(color=PALETTE['blue'], linestyle='-', marker='o'),
    'intermediate': dict(color=PALETTE['orange'], linestyle='-', marker='s'),
    'thick': dict(color=PALETTE['red'], linestyle='-', marker='^'),
}
COLORS = tuple(REGION_STYLES[level]['color'] for level in LEVELS)
EXTREME_LEVELS = ('minimum','median','maximum')
EXTREME_COLORS = COLORS
DISPLAY_NAMES = {'lateraloccipital':'lateral occipital',
                 'caudalmiddlefrontal':'caudal middle frontal', 'frontalpole':'frontal pole',
                 'temporalpole':'temporal pole'}


def display_region_name(name):
    return DISPLAY_NAMES.get(name,name.replace('_',' '))


def write_profile_caption(root):
    root=Path(root)
    text=('Regions selected at the 10th, 50th, and 90th percentiles of cortical thickness. '
          'Selection uses only group-mean FreeSurfer thickness, choosing the nearest available DK regions '
          'to the linear empirical quantiles and excluding the extrema. '
          'The inset shows all bilateral DK region means in gray and the selected regions in matching colors. '
          'Raw and thickness-adjusted profiles are shown for identical regions, colors and order. '
          'Left/right within-region means are weighted equally within subject; subjects are then weighted equally. '
          'Curves show the group mean and shading shows its two-sided 95% t confidence interval. '
          'Normalized depth runs from pial (0) to white (1). '
          'Raw regional profiles are descriptive and contain no significance tests; formal inference is provided by the thickness-association and residual-region analyses.\n')
    (root/'figures/Figure3_AD_regional_profiles_caption.txt').write_text(text)
    heatmap_note = ('PC score and standardized-score maps use the same RdBu_r colors as the final Figure 2 surface maps, with a symmetric signed scale centered at zero (neutral white): blue is negative and red is positive. Display limits use the 98th percentile of absolute values pooled across compared maps and are identical across corresponding hemisphere/view panels; only color display is clipped, never source values. Missing vertices are gray. Exact limits are saved in QC metadata. ')
    (root/'figures/Figure3_depth_profiles_thickness_caption.txt').write_text(text+
        heatmap_note + 'The profile-shape display shows the subject-weighted mean-centered PC1 loading, explained variance and bilateral inflated mean-score maps. '
        'The thickness-association display reports subject-level spatial correlations and within-subject depth-specific thickness coefficients. '
        'Thickness is native FreeSurfer thickness mapped with the Option 3 barycentric weights. '
        'Regional partial R² compares subject fixed effects with subject-plus-region fixed effects on observed cells. '
        'Confidence intervals use subject bootstrap; adjusted regional analyses refit thickness coefficients in each replicate.\n')


def rank_and_select(thickness):
    """Input contains only region identifiers and group-mean thickness, not MD.

    Linear empirical quantiles; nearest thickness wins. Numerically tied
    distances resolve to lower thickness, then exact region name.
    Actual percentile uses the compatible plotting position (rank-1)/(R-1).
    """
    table=thickness.sort_values(['group_mean_thickness','region_name'],kind='stable').reset_index(drop=True).copy()
    if len(table)<5 or table.region_name.duplicated().any() or not np.isfinite(table.group_mean_thickness).all():
        raise ValueError('Need at least five unique valid DK regions')
    table['thickness_rank']=np.arange(1,len(table)+1)
    table['thickness_percentile']=np.arange(len(table))/(len(table)-1)*100
    table['selected_for_figure3']=False;table['selected_level']='';table['color']=''
    table['target_percentile']=np.nan;table['target_thickness']=np.nan
    used=set()
    for target,level,color in zip((10,50,90),LEVELS,COLORS):
        value=float(np.percentile(table.group_mean_thickness,target,method='linear'))
        candidates=table.loc[~table.index.isin([0,len(table)-1,*used])]
        distance=np.abs(candidates.group_mean_thickness-value)
        choice=int(candidates.index[np.isclose(distance,distance.min(),atol=1e-12,rtol=0)][0])
        used.add(choice)
        table.loc[choice,['selected_for_figure3','selected_level','color','target_percentile','target_thickness']]=[True,level,color,target,value]
    return table


def select_extreme_regions(ranking):
    """Select the minimum, nearest linear median, and maximum thickness only.

    This function intentionally accepts only the frozen thickness ranking.  It
    must not inspect MD profiles or their coverage when choosing regions.
    """
    table=ranking.sort_values(['group_mean_thickness','region_name'],kind='stable').reset_index(drop=True)
    if len(table)<3 or table.region_name.duplicated().any():
        raise ValueError('Need at least three unique thickness-ranked regions')
    median=float(np.percentile(table.group_mean_thickness,50,method='linear'))
    distance=np.abs(table.group_mean_thickness-median)
    middle=int(table.index[np.isclose(distance,distance.min(),atol=1e-12,rtol=0)][0])
    indices=(0,middle,len(table)-1)
    if len(set(indices))!=3:
        raise ValueError('Minimum, median and maximum thickness selections are not distinct')
    chosen=table.loc[list(indices)].copy()
    chosen['extreme_level']=EXTREME_LEVELS
    chosen['extreme_color']=EXTREME_COLORS
    chosen['target_percentile']=(0.,50.,100.)
    chosen['target_thickness']=(float(table.iloc[0].group_mean_thickness),median,
                                float(table.iloc[-1].group_mean_thickness))
    return chosen


def bilateral_subject_profiles(table):
    """Equal-weight left/right region means, never cross-region vertex pooling."""
    keys=['subject_id','region','depth_index','normalized_depth']
    if table.duplicated(keys+['hemisphere']).any():raise ValueError('Duplicate hemisphere-region profiles')
    wide=table.pivot(index=keys,columns='hemisphere',values='md')
    if not {'lh','rh'}.issubset(wide.columns):raise ValueError('Both hemispheres required')
    # Missing hemisphere => unavailable bilateral cell; no imputation/fallback.
    result=((wide.lh+wide.rh)/2).rename('md').reset_index()
    return result


def prepare(root):
    root=Path(root);tables=root/'tables';qc=root/'qc'
    subjects=pd.read_csv(tables/'figure3_subjects.csv')
    n=len(subjects)
    nodes=pd.read_csv(qc/'vertex_order_and_labels.csv')
    regions=nodes.loc[nodes.atlas_mask,['hemisphere','dk_label_name','region_index']].drop_duplicates().sort_values('region_index').reset_index(drop=True)
    # Load only thickness from the ZIP-backed sufficient-statistics cache.
    # No MD matrix, profile table or profile shape is read until selection is saved.
    with np.load(root/'cache/subject_sufficient_statistics.npz') as data:
        thickness=data['regional_thickness']
    if thickness.shape!=(n,len(regions)):raise ValueError('Thickness subject-region dimensions differ')
    bilateral=[];subject_rows=[]
    for name in sorted(regions.dk_label_name.unique()):
        idx=regions.index[regions.dk_label_name.eq(name)].to_numpy()
        if len(idx)!=2 or set(regions.iloc[idx].hemisphere)!= {'lh','rh'}:
            raise ValueError('Expected exact left/right DK homologues')
        values=thickness[:,idx].mean(axis=1)
        if not np.isfinite(values).any():continue
        bilateral.append(dict(region_name=name,group_mean_thickness=float(np.nanmean(values)),
            sd_subject_thickness=float(np.nanstd(values,ddof=1)),n_thickness_subjects=int(np.isfinite(values).sum())))
        subject_rows.extend(dict(subject_id=s,region_name=name,bilateral_thickness=float(t)) for s,t in zip(subjects.subject_id,values))
    ranking=rank_and_select(pd.DataFrame(bilateral))
    extremes=select_extreme_regions(ranking)
    ranking['selected_for_figure3_extreme']=ranking.region_name.isin(extremes.region_name)
    ranking['extreme_level']='';ranking['extreme_color']=''
    for row in extremes.itertuples():
        hit=ranking.region_name.eq(row.region_name)
        ranking.loc[hit,['extreme_level','extreme_color']]=[row.extreme_level,row.extreme_color]
    ranking.to_csv(tables/'figure3A_region_thickness_ranking.csv',index=False)
    pd.DataFrame(subject_rows).to_csv(tables/'figure3A_subject_bilateral_thickness.csv',index=False)
    selected=ranking.loc[ranking.selected_for_figure3].sort_values('target_percentile')
    # Freeze the thickness-only selection before accessing either raw/adjusted MD.
    selection=dict(n_subjects=n,n_valid_dk_regions=len(ranking),
        selection_input='Per-subject equal-hemisphere mean of native mapped FreeSurfer thickness; then equal subject mean; no MD input',
        percentile_definition='Linear empirical thickness quantiles; actual percentile = 100*(rank-1)/(R-1)',
        tie_break='Nearest thickness; ties within 1e-12 go to lower thickness then name; extrema forbidden',
        bilateral_definition='(left region mean + right region mean)/2; both hemispheres required; unavailable bilateral cells remain NaN',
        regions=selected[['region_name','group_mean_thickness','thickness_rank','thickness_percentile','target_percentile','selected_level','color','n_thickness_subjects']].to_dict('records'),
        selection_uses_md=False,significance_tests_in_panel_a=False)
    (qc/'figure3A_region_selection.json').write_text(json.dumps(selection,indent=2,allow_nan=False)+'\n')
    extreme_selection=dict(n_subjects=n,n_valid_dk_regions=len(ranking),
        selection_input='Group-mean bilateral native mapped FreeSurfer thickness only; no MD or coverage input',
        selection_rule='Minimum, nearest region to the linear empirical 50th percentile, and maximum',
        tie_break='Median ties within 1e-12 go to lower thickness then exact region name',
        coverage_policy='Selection is frozen before MD coverage is checked; missing bilateral cells remain unavailable and are never imputed',
        regions=extremes[['region_name','group_mean_thickness','thickness_rank','thickness_percentile',
                          'target_percentile','extreme_level','extreme_color','n_thickness_subjects']].to_dict('records'),
        selection_uses_md=False)
    (qc/'figure3A_extreme_region_selection.json').write_text(
        json.dumps(extreme_selection,indent=2,allow_nan=False)+'\n')
    # Keep the public bilateral thickness table consistent with the new definition.
    ranking.rename(columns={'region_name':'region','group_mean_thickness':'mean_thickness',
        'sd_subject_thickness':'SD_thickness','n_thickness_subjects':'n_subjects'})[
        ['region','mean_thickness','SD_thickness','n_subjects']].to_csv(tables/'figure3A_bilateral_region_thickness.csv',index=False)
    counts=[]
    for label,source,subject_file,group_file in [
        ('raw','figure3A_subject_regional_profiles.csv','figure3A_bilateral_subject_profiles.csv','figure3A_bilateral_group_profiles.csv'),
        ('adjusted','figure3D_subject_adjusted_regional_profiles.csv','figure3D_bilateral_subject_adjusted_profiles.csv','figure3D_bilateral_adjusted_group_profiles.csv')]:
        frame=pd.read_csv(tables/source)
        if set(frame.subject_id)!=set(subjects.subject_id):raise ValueError('MD cohort differs from selection cohort')
        bilateral_md=bilateral_subject_profiles(frame)
        bilateral_md.to_csv(tables/subject_file,index=False)
        group=bilateral_md.groupby(['region','depth_index','normalized_depth'],sort=False).md.agg(
            mean_md='mean',sd_md='std',n_subjects='count').reset_index()
        group.to_csv(tables/group_file,index=False)
        for row in selected.itertuples():
            values=group.loc[group.region.eq(row.region_name)]
            if len(values)!=21 or not values.n_subjects.eq(n).all():
                raise ValueError(f'Selected {row.region_name} lacks the complete {n}-subject depth grid')
            for cell in values.itertuples():
                counts.append(dict(panel='A' if label=='raw' else 'D',n_final_subjects=n,n_valid_dk_regions=len(ranking),
                    region_name=row.region_name,selected_level=row.selected_level,color=row.color,
                    group_mean_thickness=row.group_mean_thickness,actual_percentile=row.thickness_percentile,
                    target_percentile=row.target_percentile,depth_index=cell.depth_index,
                    normalized_depth=cell.normalized_depth,n_valid_subjects=cell.n_subjects))
    pd.DataFrame(counts).to_csv(qc/'figure3AD_selected_region_depth_coverage.csv',index=False)
    extreme_counts=[]
    raw_group=pd.read_csv(tables/'figure3A_bilateral_group_profiles.csv')
    coverage=pd.read_csv(qc/'region_coverage.csv')
    for row in extremes.itertuples():
        values=raw_group.loc[raw_group.region.eq(row.region_name)].sort_values('depth_index')
        if (len(values)!=21 or not np.array_equal(values.depth_index.to_numpy(),np.arange(21))
                or not np.isfinite(values[['mean_md','sd_md']].to_numpy()).all()
                or values.n_subjects.nunique()!=1 or int(values.n_subjects.iloc[0])<1):
            raise ValueError(f'Extreme region {row.region_name} lacks a valid 21-depth bilateral profile')
        region_coverage=coverage.loc[coverage.region.eq(row.region_name)]
        if len(region_coverage)!=2*n:
            raise ValueError(f'Extreme region {row.region_name} lacks hemisphere-specific coverage QC')
        zero=region_coverage.loc[region_coverage.n_valid_vertices.eq(0)]
        positive=region_coverage.loc[region_coverage.n_valid_vertices.gt(0),'n_valid_vertices']
        extreme_counts.append(dict(
            region_name=row.region_name,extreme_level=row.extreme_level,color=row.extreme_color,
            group_mean_thickness=row.group_mean_thickness,n_thickness_subjects=int(row.n_thickness_subjects),
            n_profile_subjects=int(values.n_subjects.iloc[0]),n_expected_subjects=n,
            coverage_fraction=float(values.n_subjects.iloc[0]/n),n_zero_hemisphere_cells=len(zero),
            zero_coverage_subjects=';'.join(sorted(zero.subject_id.astype(str).unique())),
            minimum_positive_vertex_count=int(positive.min()),median_positive_vertex_count=float(positive.median()),
            complete_21_depth_grid=True,no_imputation=True))
    pd.DataFrame(extreme_counts).to_csv(qc/'figure3A_extreme_region_coverage_qc.csv',index=False)
    return ranking,selected


def draw_thickness_distribution(ax,ranking,compact=False):
    # The compact inset sits beside a legend that already gives the full region
    # names and thicknesses.  Repeating those long names here made panel A's
    # upper half unnecessarily dense, so the inset only identifies the three
    # pre-specified thickness quantiles.
    ax.scatter(ranking.group_mean_thickness,np.zeros(len(ranking)),s=8 if compact else 20,color='#B0B0B0',zorder=1)
    selected=ranking.loc[ranking.selected_for_figure3].sort_values('target_percentile')
    for i,row in enumerate(selected.itertuples()):
        ax.scatter(row.group_mean_thickness,0,s=28 if compact else 55,color=row.color,edgecolor='white',zorder=3)
        offset=(7 if i!=1 else -4) if compact else (20 if i!=1 else -25)
        label=(f'P{int(row.target_percentile)}' if compact
               else display_region_name(row.region_name))
        ax.annotate(label,(row.group_mean_thickness,0),xytext=(0,offset),
                    textcoords='offset points',ha='center',va='top' if i==1 else 'bottom',fontsize=6.5 if compact else 8.5,color='#222222',
                    arrowprops={'arrowstyle':'-','color':row.color,'lw':.7})
    ax.set(yticks=[],ylim=(-.45,.8) if compact else (-.8,.8))
    ax.set_xlabel('DK thickness (mm)' if compact else 'Group-mean DK cortical thickness (mm)',fontsize=7 if compact else 10)
    ax.tick_params(axis='x',labelsize=7 if compact else 9,length=2 if compact else 3)
    ax.spines[['left','right','top']].set_visible(False)
    ax.margins(x=.08)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    ranking,selected=prepare(p.parse_args().output_dir)
    print(selected.to_string(index=False))
