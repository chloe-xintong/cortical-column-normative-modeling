#!/usr/bin/env python3
"""Render Figure 3 entirely from saved tables and MGH maps (no hardcoded results).

Description
-----------
Render Figure 3 entirely from saved tables and MGH maps (no hardcoded results).

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input
- ``--fsaverage-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure3/computation/plot_figure3.py --help
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qc.publication_style import finalize_figure, PALETTE, PROFILE_STYLES, apply_semantic_style, heatmap_norm
from select_representative_regions import REGION_STYLES
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd

SURFACE_CMAP=plt.colormaps['RdBu_r'].copy()
SURFACE_CMAP.set_bad(PALETTE['missing'])
from nilearn.plotting import plot_surf_stat_map
from figure3_association_plot import draw_subject_rho, export_subject_rho
from figure3_region_effect_plot import draw_region_effect, export as export_region_effect
from figure3_panel_a_plot import (export_panel_a_figures, export_panel_d_figures,
                                  mean_ci95_halfwidth)
from select_representative_regions import (prepare as prepare_regions, draw_thickness_distribution,
    bilateral_subject_profiles, display_region_name, write_profile_caption)


def save(fig, output, stem):
    import textwrap
    for ax in fig.axes:
        if ax.name != '3d' and ax.get_xlabel():
            width=max(18,int(fig.get_figwidth()*ax.get_position().width*72/4.25))
            ax.set_xlabel('\n'.join(textwrap.fill(line,width=width,break_long_words=False) for line in ax.get_xlabel().split('\n')))
    finalize_figure(fig)
    for ext in ('pdf','svg','png'):
        fig.savefig(output/f'{stem}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def bilateral_profiles(root,filename):
    profiles=pd.read_csv(root/'tables'/filename)
    subject=bilateral_subject_profiles(profiles)
    group=subject.groupby(['region','depth_index','normalized_depth'],sort=False).md.agg(mean_md='mean',sd_md='std',n_subjects='count').reset_index()
    return group


def draw_profiles(ax,table,selected,title,n,ylim):
    for row in selected.itertuples():
        region=row.region_name
        t=table.loc[table.region.eq(region)].sort_values('depth_index')
        x=t.normalized_depth.to_numpy();m=t.mean_md.to_numpy()
        s=mean_ci95_halfwidth(t.sd_md.to_numpy(),t.n_subjects.to_numpy())
        label=display_region_name(region)
        ax.plot(x,m,**REGION_STYLES[row.selected_level],markevery=4,label=label,lw=1.8)
        ax.fill_between(x,m-s,m+s,color=row.color,alpha=.12)
    ax.set(xlim=(0,1),ylim=ylim,xlabel='Normalized depth (0 pial → 1 white)',ylabel='MD (original image units)')
    ax.set_title(f'{title}\nMean with 95% CI',loc='left',fontweight='bold')
    ax.legend(frameon=False,fontsize=8.5,labelspacing=.35,handlelength=2.5)
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))


def render_surface(ax,fsaverage,hemi,view,values,vmax):
    mesh=fsio.read_geometry(fsaverage/'surf'/f'{hemi}.inflated')
    curv=fsio.read_morph_data(fsaverage/'surf'/f'{hemi}.curv')
    plot_surf_stat_map(mesh,values,hemi='left' if hemi=='lh' else 'right',view=view,
        bg_map=curv,bg_on_data=False,cmap=SURFACE_CMAP,symmetric_cbar=True,vmin=-vmax,vmax=vmax,
        colorbar=False,axes=ax,figure=ax.figure,engine='matplotlib',title=f'{hemi.upper()} {view}')
    for collection in ax.collections:collection.set_rasterized(True)


def plot_all(root,fsaverage,main_only=False):
    root,fsaverage=Path(root),Path(fsaverage);out=root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],'font.size':11,'axes.titlesize':12,'axes.labelsize':11,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    apply_semantic_style()
    export_subject_rho(root)
    export_region_effect(root)
    n=len(pd.read_csv(root/'tables/figure3_subjects.csv'))
    ranking,selected=prepare_regions(root)
    export_panel_a_figures(root,ranking,selected)
    export_panel_d_figures(root,ranking,selected)
    selected_names=selected.region_name.tolist()
    raw=pd.read_csv(root/'tables/figure3A_bilateral_group_profiles.csv')
    adjusted=pd.read_csv(root/'tables/figure3D_bilateral_adjusted_group_profiles.csv')
    load=pd.read_csv(root/'tables/figure3B_pca_loadings.csv')
    ev=pd.read_csv(root/'tables/figure3B_pca_explained_variance.csv')
    association=pd.read_csv(root/'tables/figure3C_group_association_summary.csv').set_index('statistic')
    subject_association=pd.read_csv(root/'tables/figure3C_subject_spatial_correlations.csv')
    bins=pd.read_csv(root/'tables/figure3C_binned_group_scatter.csv')
    effect=pd.read_csv(root/'tables/figure3C_depth_specific_thickness_effect.csv')
    regional=pd.read_csv(root/'tables/figure3D_region_effect_by_depth.csv')
    chosen=pd.concat([raw.loc[raw.region.isin(selected_names)],adjusted.loc[adjusted.region.isin(selected_names)]])
    chosen_ci=mean_ci95_halfwidth(chosen.sd_md.to_numpy(),chosen.n_subjects.to_numpy())
    low=(chosen.mean_md.to_numpy()-chosen_ci).min()
    high=(chosen.mean_md.to_numpy()+chosen_ci).max();span=high-low
    ylim=(low-.04*span,high+.08*span)
    fig=plt.figure(figsize=(7.09,10.5));grid=fig.add_gridspec(2,2,hspace=.4,wspace=.4)
    a=fig.add_subplot(grid[0,0]);draw_profiles(a,raw,selected,'Raw cortical-depth profiles',n,ylim)
    thickness_ax=a.inset_axes([.49,.59,.48,.13])
    thickness_ax.set_facecolor((1,1,1,.90))
    draw_thickness_distribution(thickness_ax,ranking,compact=True)
    bg=grid[0,1].subgridspec(3,2,height_ratios=[1,1.1,1.1],hspace=1.1,wspace=.02)
    loading=fig.add_subplot(bg[0,:]);loading.plot(load.normalized_depth,load.PC1,color='#4B74B2',lw=1.8)
    loading.axhline(0,color='#aaaaaa',lw=.7)
    loading.set(xlabel='Normalized depth (pial → white)',ylabel='PC1 loading',xlim=(0,1))
    loading.set_title(f'Mean-centered profile shape\nPC1 ({ev.iloc[0].explained_variance_ratio:.1%} variance)',loc='left',fontweight='bold')
    maps={h:np.asarray(nib.load(root/'surface_maps'/f'{h}.figure3B_pc1_mean.mgh').dataobj).reshape(-1) for h in ('lh','rh')}
    vmax=float(np.nanpercentile(np.abs(np.concatenate(list(maps.values()))),98))
    display_scales={"PC1_mean":dict(vmin=-vmax,vcenter=0,vmax=vmax,clipping="98th percentile of absolute values pooled across both hemispheres",cmap="RdBu_r",positive="Positive PC1 score",negative="Negative PC1 score")}
    for row,hemi in enumerate(('lh','rh')):
        for col,view in enumerate(('lateral','medial')):
            ax=fig.add_subplot(bg[row+1,col],projection='3d');render_surface(ax,fsaverage,hemi,view,maps[hemi],vmax)
    cbar=fig.colorbar(plt.cm.ScalarMappable(norm=heatmap_norm("signed",vmax=vmax),cmap=SURFACE_CMAP),
                      ax=[ax for ax in fig.axes if getattr(ax,'name','')=='3d'],shrink=.65,pad=.01)
    cbar.set_label('Group mean PC1 score (MD units)')
    cg=grid[1,0].subgridspec(2,1,hspace=.7)
    c1=fig.add_subplot(cg[0]);c2=fig.add_subplot(cg[1])
    rho=association.loc['median_spearman_rho']
    draw_subject_rho(c1,subject_association,rho)
    c2.plot(effect.normalized_depth,effect.beta_thickness,color=PALETTE['blue'],lw=1.8)
    c2.fill_between(effect.normalized_depth,effect.beta_ci95_lower,effect.beta_ci95_upper,color=PALETTE['light_blue'],alpha=.12)
    c2.axhline(0,color='#aaaaaa',lw=.8)
    c2.set(xlabel='Normalized depth (pial → white)',ylabel='Thickness β (MD units / mm)',xlim=(0,1))
    dg=grid[1,1].subgridspec(2,1,hspace=.7)
    d1=fig.add_subplot(dg[0]);d2=fig.add_subplot(dg[1])
    draw_profiles(d1,adjusted,selected,'Thickness-adjusted profiles',n,ylim)
    draw_region_effect(d2,regional)
    save(fig,out,'Figure3_depth_profiles_thickness')
    write_profile_caption(root)
    if main_only:
        return
    fig,ax=plt.subplots(figsize=(3.45,3.1))
    ax.errorbar(bins.mean_thickness,bins.mean_pc1_score,yerr=bins.vertex_se,
                fmt='o',ms=4,capsize=2,color='#4B74B2')
    ax.set(xlabel='Group mean cortical thickness (mm)',ylabel='Group mean PC1 score',
           title='Descriptive group-average relationship (40 equal-count bins)')
    fig.text(.5,.015,'Error bars: vertex SE; not subject-level inference.\n'
             'Aggregation does not represent the typical within-subject effect.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.1,1,1));save(fig,out,'Figure3S_group_average_pc1_thickness')
    stable=pd.read_csv(root/'tables/figure3S_pc1_split_half_stability.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.09,2.8))
    axes[0].hist(stable.loading_correlation,bins=15,color='#4B74B2');axes[0].set(xlabel='Split-half PC1 loading correlation',ylabel='Repetitions')
    axes[1].hist(stable.group_surface_map_correlation,bins=15,color='#90BEE0');axes[1].set(xlabel='Split-half group surface-map correlation')
    axes[2].scatter(stable.explained_variance_ratio_half1,stable.explained_variance_ratio_half2,s=15,color='#4B74B2')
    axes[2].set(xlabel='PC1 variance ratio: half 1',ylabel='PC1 variance ratio: half 2')
    fig.suptitle(f'PC1 subject split-half stability ({len(stable)} repetitions)');fig.tight_layout()
    save(fig,out,'Figure3S_pc1_stability')
    sm=pd.read_csv(root/'tables/figure3S_subject_sensitivity_associations.csv')
    gm=pd.read_csv(root/'tables/figure3B_group_pc1_surface_statistics.csv')
    gm=gm.loc[gm.n_valid_subjects.gt(0)].sort_values('mean_thickness')
    points=[]
    for ids in np.array_split(np.arange(len(gm)),40):
        part=gm.iloc[ids];points.append((part.mean_thickness.mean(),part.mean_md.mean(),part.mean_md.std(ddof=1)/np.sqrt(len(part))))
    points=np.asarray(points)
    fig,axes=plt.subplots(1,2,figsize=(7.09,3.2))
    axes[0].errorbar(points[:,0],points[:,1],yerr=points[:,2],fmt='o',ms=4,color='#4B74B2')
    axes[0].set(xlabel='Group mean cortical thickness (mm)',ylabel='Group mean depth-averaged MD',title='Descriptive vertex bins (mean ± SE)')
    vals=sm.loc[sm.feature.eq('mean_md'),'spearman_with_thickness']
    axes[1].hist(vals,bins=20,color='#4B74B2');axes[1].set(xlabel='Subject spatial Spearman ρ: mean MD vs thickness',ylabel='Subjects')
    fig.tight_layout();save(fig,out,'Figure3S_mean_md_vs_thickness')
    sensitivity=pd.read_csv(root/'tables/figure3S_column_length_sensitivity.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.09,3.2))
    vals=sm.loc[sm.feature.eq('column_length'),'spearman_with_pc1']
    axes[0].hist(vals,bins=20,color='#4B74B2');axes[0].set(xlabel='Subject spatial ρ: PC1 vs column length',ylabel='Subjects')
    axes[1].plot(sensitivity.normalized_depth,sensitivity.beta_length,color='#4B74B2')
    axes[1].fill_between(sensitivity.normalized_depth,sensitivity.beta_ci95_lower,sensitivity.beta_ci95_upper,color='#90BEE0',alpha=.12)
    axes[1].set(xlabel='Normalized depth',ylabel='Length β (MD units / mm)')
    axes[2].plot(regional.normalized_depth,regional.raw_partial_r2,**PROFILE_STYLES['raw'],markevery=4,label='Raw')
    axes[2].plot(regional.normalized_depth,regional.adjusted_partial_r2,**PROFILE_STYLES['adjusted'],markevery=4,label='Thickness adjusted')
    axes[2].plot(sensitivity.normalized_depth,sensitivity.length_adjusted_region_partial_r2,**PROFILE_STYLES['length_adjusted'],markevery=4,label='Length adjusted')
    axes[2].set(xlabel='Normalized depth',ylabel='Region partial R²',ylim=(0,1));axes[2].legend(frameon=False,fontsize=9)
    fig.suptitle('Column length sensitivity only; primary adjustment uses FreeSurfer thickness')
    fig.tight_layout();save(fig,out,'Figure3S_column_length_sensitivity')
    # Requested supplemental standardized score map (Cohen's d).
    fig=plt.figure(figsize=(7.09,4.5))
    dm={h:np.asarray(nib.load(root/'surface_maps'/f'{h}.figure3B_pc1_cohens_d.mgh').dataobj).reshape(-1) for h in ('lh','rh')}
    vmax=float(np.nanpercentile(np.abs(np.concatenate(list(dm.values()))),98))
    display_scales["PC1_cohens_d"]=dict(vmin=-vmax,vcenter=0,vmax=vmax,clipping="98th percentile of absolute values pooled across both hemispheres",cmap="RdBu_r",positive="Positive standardized PC1 score",negative="Negative standardized PC1 score")
    for row,h in enumerate(('lh','rh')):
        for col,view in enumerate(('lateral','medial')):
            render_surface(fig.add_subplot(2,2,row*2+col+1,projection='3d'),fsaverage,h,view,dm[h],vmax)
    fig.colorbar(plt.cm.ScalarMappable(norm=heatmap_norm("signed",vmax=vmax),cmap=SURFACE_CMAP),ax=fig.axes,shrink=.7,label="PC1 Cohen's d")
    save(fig,out,'Figure3S_pc1_cohens_d')
    (root/'qc/heatmap_display_scales.json').write_text(json.dumps(display_scales,indent=2)+'\n')
    write_profile_caption(root)
    write={'status':'complete','n_subjects':n,'selected_region_thickness_order':selected_names,
           'region_selection':'Nearest linear 10th/50th/90th thickness quantiles; no MD input; frozen A/D colors and order',
           'figures':['Figure3_depth_profiles_thickness','Figure3S_group_average_pc1_thickness','Figure3S_pc1_stability','Figure3S_mean_md_vs_thickness','Figure3S_column_length_sensitivity','Figure3S_pc1_cohens_d'],
           'panel_c_primary':'Subject-level spatial rho: violin and dark-blue box, orange median; no scatter or N labels',
           'panel_c_interpretation':'statistically detectable but negligible subject-level association',
           'formats':['pdf','svg','png'],'png_dpi':300}
    (root/'qc/plot_manifest.json').write_text(json.dumps(write,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--fsaverage-dir',type=Path,required=True)
    p.add_argument('--main-only',action='store_true')
    a=p.parse_args();plot_all(a.output_dir,a.fsaverage_dir,a.main_only)
