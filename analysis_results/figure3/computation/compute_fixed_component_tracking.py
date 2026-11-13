#!/usr/bin/env python3
"""Complete Figure 3B–D tracking of fixed original PC1/PC2 components.

Description
-----------
Complete Figure 3B–D tracking of fixed original PC1/PC2 components.

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
python analysis_results/figure3/computation/compute_fixed_component_tracking.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.freesurfer.io as fsio
from nilearn.plotting import plot_surf_stat_map
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qc.publication_style import (PALETTE, finalize_figure,
                                  heatmap_norm, set_publication_style,
                                  subject_violin)


# Match the final Figure 2 cortical maps exactly. Figure 3 uses the same
# colors with a signed, zero-centered normalization because PCA scores can be
# positive or negative.
SURFACE_CMAP = plt.colormaps['RdBu_r'].copy()
SURFACE_CMAP.set_bad(PALETTE['missing'])


def save(fig, directory, stem):
    finalize_figure(fig)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(directory/f'{stem}.{ext}', dpi=300, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)


def spatial_correlation(a, b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    a=a-a.mean();b=b-b.mean()
    denominator=np.sqrt(a@a)*np.sqrt(b@b)
    if not np.isfinite(denominator) or denominator<=0:
        raise ValueError('Undefined spatial correlation')
    return float((a@b)/denominator)


def save_surface(path, values):
    nib.save(nib.MGHImage(np.asarray(values,dtype=np.float32).reshape(-1,1,1),
                          np.eye(4)),path)


def render_surface(ax, fsaverage, hemi, view, values, vmax, title):
    mesh=fsio.read_geometry(fsaverage/'surf'/f'{hemi}.inflated')
    curv=fsio.read_morph_data(fsaverage/'surf'/f'{hemi}.curv')
    plot_surf_stat_map(
        mesh, values, hemi='left' if hemi=='lh' else 'right', view=view,
        bg_map=curv, bg_on_data=False, cmap=SURFACE_CMAP,
        symmetric_cbar=True, vmin=-vmax, vmax=vmax, colorbar=False,
        axes=ax, figure=ax.figure, engine='matplotlib', title=title)
    for collection in ax.collections:
        collection.set_rasterized(True)


def map_scale(arrays):
    values=np.concatenate([np.abs(np.asarray(x)[np.isfinite(x)]) for x in arrays])
    vmax=float(np.percentile(values,98))
    if not np.isfinite(vmax) or vmax<=0:
        raise ValueError('Invalid surface-map display scale')
    return vmax


def plot_raw_component_maps(root, fsaverage, raw_maps, display_scales=None):
    figures=root/'figures'
    scales=(display_scales if display_scales is not None else
            {f'PC{k+1}':map_scale([raw_maps[k,h] for h in ('lh','rh')])
             for k in range(2)})
    fig=plt.figure(figsize=(7.09,4.5))
    grid=fig.add_gridspec(2,5,width_ratios=(1,1,1,1,.075),
                         wspace=.03,hspace=.16)
    views=(('lh','lateral'),('lh','medial'),('rh','lateral'),('rh','medial'))
    for row in range(2):
        vmax=scales[f'PC{row+1}']
        row_axes=[]
        for col,(hemi,view) in enumerate(views):
            ax=fig.add_subplot(grid[row,col],projection='3d');row_axes.append(ax)
            render_surface(ax,fsaverage,hemi,view,raw_maps[row,hemi],vmax,
                           f'{hemi.upper()} {view}' if row==0 else '')
            if col==0:
                ax.text2D(-.12,.5,f'PC{row+1}',transform=ax.transAxes,
                          rotation=90,va='center',ha='center',fontsize=9)
        cax=fig.add_subplot(grid[row,4])
        bar=fig.colorbar(
            plt.cm.ScalarMappable(norm=heatmap_norm('signed',vmax=vmax),
                                  cmap=SURFACE_CMAP),cax=cax)
        bar.set_label(f'PC{row+1} score (MD units)')
    fig.suptitle('PC1 and PC2 group cortical score maps',fontweight='bold')
    save(fig,figures,'Figure3B_03_PC1_PC2_group_score_maps')
    caption=(
        'PC1 and PC2 use separate colorbar ranges so that the spatial pattern '
        'of each component remains visible. Each range is symmetric about zero, '
        'and all hemispheres and views within a PC share that range. Both use '
        'the same RdBu_r colors as Figure 2. Because the two PCs have different '
        'numeric ranges, color intensity must not be used to compare their '
        'absolute score amplitudes; quantitative score metrics provide that comparison.\n')
    (figures/'Figure3B_PC1_PC2_group_score_maps_caption.txt').write_text(caption)
    return scales


def plot_raw_adjusted_maps(root, fsaverage, raw_maps, adjusted_maps,
                           components=(0, 1), display_scales=None):
    figures=root/'figures';scales={}
    views=(('lh','lateral'),('lh','medial'),('rh','lateral'),('rh','medial'))
    for component in components:
        vmax=(display_scales[f'PC{component+1}'] if display_scales is not None
              else map_scale([raw_maps[component,h] for h in ('lh','rh')]+[
                              adjusted_maps[component,h] for h in ('lh','rh')]))
        scales[f'PC{component+1}']=vmax
        fig=plt.figure(figsize=(7.09,4.5));grid=fig.add_gridspec(2,4,wspace=.01,hspace=.16)
        axes=[]
        for row,(condition,maps) in enumerate((('Raw',raw_maps),
                                                ('Adjusted',adjusted_maps))):
            for col,(hemi,view) in enumerate(views):
                ax=fig.add_subplot(grid[row,col],projection='3d');axes.append(ax)
                title=f'{hemi.upper()} {view}' if row==0 else ''
                render_surface(ax,fsaverage,hemi,view,maps[component,hemi],vmax,
                               title)
                if col==0:
                    ax.text2D(-.12,.5,condition,transform=ax.transAxes,
                              rotation=90,va='center',ha='center',fontsize=9)
        bar=fig.colorbar(plt.cm.ScalarMappable(norm=heatmap_norm('signed',vmax=vmax),
                         cmap=SURFACE_CMAP),ax=axes,shrink=.62,pad=.01)
        bar.set_label(f'Fixed-reference PC{component+1} score (MD units)')
        fig.suptitle(f'PC{component+1} before and after thickness adjustment',fontweight='bold')
        save(fig,figures,f'Figure3D_0{component+3}_PC{component+1}_raw_adjusted_maps')
    caption=(
        'Within each PC, raw and thickness-adjusted maps use one fixed colorbar '
        'range pooled across both conditions, hemispheres and views. The range '
        'is symmetric about zero and uses RdBu_r. PC1 and PC2 retain separate '
        'ranges; color intensity must not be used to compare absolute amplitude '
        'between components. Amplitude change is reported by the adjusted/raw '
        'spatial score SD ratio.\n')
    (figures/'Figure3D_fixed_score_maps_caption.txt').write_text(caption)
    return scales


def bootstrap_summary(table, repetitions, seed):
    rng=np.random.default_rng(seed);rows=[]
    for component,part in table.groupby('component',sort=True):
        for metric in ('spatial_correlation_raw_adjusted','adjusted_to_raw_sd_ratio'):
            values=part[metric].to_numpy(float);n=len(values)
            samples=rng.integers(0,n,(repetitions,n))
            boot=np.median(values[samples],axis=1)
            lo,hi=np.percentile(boot,[2.5,97.5]);q25,q75=np.percentile(values,[25,75])
            rows.append(dict(component=component,metric=metric,n_subjects=n,
                median=float(np.median(values)),q25=float(q25),q75=float(q75),
                ci95_lower=float(lo),ci95_upper=float(hi),
                bootstrap_repetitions=repetitions,seed=seed))
    return pd.DataFrame(rows)


def violin_with_ci(ax, table, summary, metric, ylabel, reference=None):
    colors=(PALETTE['blue'],PALETTE['orange'])
    for position,component in enumerate(('PC1','PC2')):
        values=table.loc[table.component.eq(component),metric].to_numpy(float)
        subject_violin(ax,values,position=position,color=colors[position],
                       single_group=False,showfliers=True)
        row=summary.loc[summary.component.eq(component)&summary.metric.eq(metric)].iloc[0]
        ax.errorbar(position,row['median'],
                    yerr=[[row['median']-row.ci95_lower],
                          [row.ci95_upper-row['median']]],
                    fmt='D',color=colors[position],markeredgecolor='white',
                    markersize=5,capsize=3,linewidth=1.3,zorder=5)
    if reference is not None:
        ax.axhline(reference,color='#777777',linestyle='--',linewidth=.8)
    ax.set(xticks=[0,1],xticklabels=['PC1','PC2'],ylabel=ylabel,xlim=(-.5,1.5))


def plot_tracking_metrics(root, table, summary):
    fig,axes=plt.subplots(1,2,figsize=(7.09,3.2))
    violin_with_ci(axes[0],table,summary,'spatial_correlation_raw_adjusted',
                   'Raw–adjusted spatial correlation',reference=1)
    violin_with_ci(axes[1],table,summary,'adjusted_to_raw_sd_ratio',
                   'Adjusted / raw spatial SD',reference=1)
    axes[0].set_title('Spatial pattern retention')
    axes[1].set_title('Score amplitude retention')
    fig.suptitle('Fixed original PCA components after thickness adjustment')
    fig.tight_layout(rect=(0,0,1,.93))
    save(fig,root/'figures','Figure3D_05_fixed_component_retention')


def plot_loadings(root, loadings, variance):
    """Export PC1/PC2 loadings and explained variance as separate figures."""
    colors=(PALETTE['blue'],PALETTE['orange'])
    markers=('o', 's')
    fig,ax=plt.subplots(figsize=(5.7,3.15))
    for index in range(2):
        ax.plot(loadings.normalized_depth,loadings[f'PC{index+1}'],
                color=colors[index],linestyle='-',
                marker=markers[index],markevery=4,
                linewidth=2.6,label=f'PC{index+1}')
    ax.axhline(0,color='#777777',linewidth=.8)
    ax.set(xlim=(0,1),xlabel='Normalized depth (0 pial → 1 white)',ylabel='Loading',
           title='Main cortical-depth shape components')
    ax.legend(loc='center left',bbox_to_anchor=(1.015,.5),frameon=True,
              facecolor='white',edgecolor='#C7C7C7',framealpha=.97,
              fancybox=False,borderpad=.55,labelspacing=.5,
              handlelength=2.4,handletextpad=.65)
    fig.subplots_adjust(right=.78)
    save(fig,root/'figures','Figure3B_01_PC1_PC2_loadings')

    shown=variance.loc[variance.component.between(1,5)].copy()
    bar_colors=[PALETTE['blue'],PALETTE['orange'],
                PALETTE['light_blue'],'#C8B6D9','#B8B8B8']
    fig,ax=plt.subplots(figsize=(5.7,3.15))
    ax.bar(shown.component,shown.explained_variance_ratio,color=bar_colors,
           width=.68)
    ax.set(xlabel='Principal component',ylabel='Explained variance ratio',
           xticks=shown.component,title='Profile-shape variance explained')
    ax.set_ylim(0,float(shown.explained_variance_ratio.max())*1.12)
    save(fig,root/'figures','Figure3B_02_explained_variance')


def plot_thickness_associations(root):
    comparison=root/'pc1_pc2_thickness'
    subjects=pd.read_csv(comparison/'subject_thickness_pc1_pc2_correlations.csv')
    summary=pd.read_csv(comparison/'component_correlation_summary.csv')
    fig,ax=plt.subplots(figsize=(5.7,3.2));colors=(PALETTE['blue'],PALETTE['orange'])
    fills=(PALETTE['pale_blue'],'#FDE3D6')
    outline=PALETTE['reference']
    for position,(component,column) in enumerate((('PC1','rho_thickness_pc1'),
                                                   ('PC2','rho_thickness_pc2'))):
        values=subjects[column].to_numpy(float)
        subject_violin(ax,values,position=position,color=colors[position],
                       single_group=False,showfliers=True,width=.42,
                       fill_color=fills[position],edge_color=outline,alpha=1,
                       box_facecolor=fills[position],box_edgecolor=outline)
        row=summary.loc[summary.component.eq(component)].iloc[0]
        ax.errorbar(position,row.median_rho,
                    yerr=[[row.median_rho-row.median_rho_ci95_lower],
                          [row.median_rho_ci95_upper-row.median_rho]],
                    fmt='D',color=colors[position],markeredgecolor='white',
                    markersize=5,capsize=3,linewidth=1.3,zorder=5)
    ax.axhline(0,color='#777777',linestyle='--',linewidth=.8)
    ax.set(xticks=[0,1],xticklabels=['PC1','PC2'],
           ylabel='Within-subject spatial Spearman ρ',
           title='Thickness association of fixed profile-shape scores',xlim=(-.5,1.5))
    save(fig,root/'figures','Figure3C_01_PC1_PC2_thickness_correlations')

    effect=pd.read_csv(root/'tables/figure3C_depth_specific_thickness_effect.csv')
    fig,ax=plt.subplots(figsize=(5.7,3.15))
    ax.plot(effect.normalized_depth,effect.partial_r2,color=PALETTE['blue'],
            marker='o',markevery=4,linewidth=1.9)
    ax.fill_between(effect.normalized_depth,effect.partial_r2_ci95_lower,
                    effect.partial_r2_ci95_upper,color=PALETTE['blue'],alpha=.12)
    ax.set(xlim=(0,1),ylim=(0,max(.01,float(effect.partial_r2_ci95_upper.max())*1.08)),
           xlabel='Normalized depth (0 pial → 1 white)',
           ylabel='Thickness partial R²',title='Depth-specific variance explained by thickness')
    save(fig,root/'figures','Figure3C_02_thickness_partial_r2')

    fig,ax=plt.subplots(figsize=(5.7,3.15))
    ax.plot(effect.normalized_depth,effect.beta_thickness,color=PALETTE['blue'],
            marker='o',markevery=4,linewidth=1.9)
    ax.fill_between(effect.normalized_depth,effect.beta_ci95_lower,
                    effect.beta_ci95_upper,color=PALETTE['blue'],alpha=.12)
    ax.axhline(0,color='#777777',linestyle='--',linewidth=.8)
    ax.set(xlim=(0,1),xlabel='Normalized depth (0 pial → 1 white)',
           ylabel='Thickness β (MD units / mm)',title='Direction of depth-specific thickness association')
    save(fig,root/'figures','Figure3S_thickness_beta_by_depth')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--fsaverage-dir',type=Path,required=True)
    parser.add_argument('--bootstrap-repetitions',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=20261108)
    args=parser.parse_args();root=args.output_dir;tables=root/'tables';qc=root/'qc'
    figures=root/'figures';figures.mkdir(exist_ok=True)
    set_publication_style()

    subjects=pd.read_csv(tables/'figure3_subjects.csv',dtype={'subject_id':str})
    raw=np.load(root/'cache/raw_md.npy',mmap_mode='r')
    thickness=np.load(root/'cache/thickness.npy',mmap_mode='r')
    mask=np.load(tables/'figure3_analysis_mask.npy').reshape(-1).astype(bool)
    loadings=pd.read_csv(tables/'figure3B_pca_loadings.csv')
    variance=pd.read_csv(tables/'figure3B_pca_explained_variance.csv')
    beta=pd.read_csv(tables/'figure3C_depth_specific_thickness_effect.csv').beta_thickness.to_numpy(float)
    with np.load(root/'cache/subject_sufficient_statistics.npz') as cached:
        thickness_means=cached['t_means'].astype(float)
    n,v,_=raw.shape
    if n!=len(subjects) or thickness.shape!=(n,v) or len(mask)!=v:
        raise ValueError('Figure 3 cache dimensions differ from the frozen cohort')
    reference=loadings.pca_feature_mean.to_numpy(float)
    vectors=loadings[['PC1','PC2']].to_numpy(float)
    beta_centered=beta-beta.mean()
    sums_raw=np.zeros((2,v));sums_adjusted=np.zeros((2,v));counts=np.zeros(v,int)
    rows=[]
    for index,subject in enumerate(subjects.subject_id):
        valid=mask&np.isfinite(thickness[index])
        x=np.asarray(raw[index,valid],float)
        shape=x-x.mean(axis=1,keepdims=True)
        raw_scores=(shape-reference)@vectors
        delta=np.asarray(thickness[index,valid],float)-thickness_means[index]
        adjusted_shape=shape-delta[:,None]*beta_centered
        adjusted_scores=(adjusted_shape-reference)@vectors
        for component in range(2):
            a=raw_scores[:,component];b=adjusted_scores[:,component]
            raw_sd=float(a.std(ddof=1));adjusted_sd=float(b.std(ddof=1))
            rows.append(dict(subject_id=subject,component=f'PC{component+1}',
                n_valid_vertices=int(valid.sum()),
                spatial_correlation_raw_adjusted=spatial_correlation(a,b),
                raw_spatial_sd=raw_sd,adjusted_spatial_sd=adjusted_sd,
                adjusted_to_raw_sd_ratio=adjusted_sd/raw_sd))
            sums_raw[component,valid]+=a;sums_adjusted[component,valid]+=b
        counts[valid]+=1
        print(f'fixed_component_subject={index+1}/{n}',flush=True)

    raw_mean=np.divide(sums_raw,counts,out=np.full_like(sums_raw,np.nan),where=counts>0)
    adjusted_mean=np.divide(sums_adjusted,counts,out=np.full_like(sums_adjusted,np.nan),where=counts>0)
    half=v//2;raw_maps={};adjusted_maps={}
    for component in range(2):
        for hemi_index,hemi in enumerate(('lh','rh')):
            sl=slice(hemi_index*half,(hemi_index+1)*half)
            raw_maps[component,hemi]=raw_mean[component,sl]
            adjusted_maps[component,hemi]=adjusted_mean[component,sl]
            save_surface(root/'surface_maps'/f'{hemi}.figure3_PC{component+1}_raw_fixed_mean.mgh',
                         raw_maps[component,hemi])
            save_surface(root/'surface_maps'/f'{hemi}.figure3_PC{component+1}_adjusted_fixed_mean.mgh',
                         adjusted_maps[component,hemi])

    subject_metrics=pd.DataFrame(rows)
    subject_metrics.to_csv(tables/'figure3D_fixed_component_subject_retention.csv',index=False)
    summary=bootstrap_summary(subject_metrics,args.bootstrap_repetitions,args.seed)
    summary.to_csv(tables/'figure3D_fixed_component_retention_summary.csv',index=False)
    map_rows=[]
    for component in range(2):
        good=mask&np.isfinite(raw_mean[component])&np.isfinite(adjusted_mean[component])
        a=raw_mean[component,good];b=adjusted_mean[component,good]
        map_rows.append(dict(component=f'PC{component+1}',n_vertices=int(good.sum()),
            group_map_spatial_correlation=spatial_correlation(a,b),
            raw_group_map_spatial_sd=float(a.std(ddof=1)),
            adjusted_group_map_spatial_sd=float(b.std(ddof=1)),
            adjusted_to_raw_group_map_sd_ratio=float(b.std(ddof=1)/a.std(ddof=1))))
    pd.DataFrame(map_rows).to_csv(tables/'figure3D_fixed_component_group_map_retention.csv',index=False)

    plot_loadings(root,loadings,variance)
    component_scales={f'PC{k+1}':map_scale(
        [raw_maps[k,h] for h in ('lh','rh')]+[
         adjusted_maps[k,h] for h in ('lh','rh')]) for k in range(2)}
    b_scales=plot_raw_component_maps(
        root,args.fsaverage_dir,raw_maps,component_scales)
    d_scales=plot_raw_adjusted_maps(
        root,args.fsaverage_dir,raw_maps,adjusted_maps,
        display_scales=component_scales)
    plot_tracking_metrics(root,subject_metrics,summary)
    plot_thickness_associations(root)
    status=dict(status='complete',n_subjects=n,components=['PC1','PC2'],
        loading_source='Original frozen PCA loadings',
        centering_reference='Original pca_feature_mean saved with the frozen loadings',
        score_definition='(within-profile-centered MD - original PCA feature mean) dot original loading',
        adjustment='Original depth-specific beta applied before within-profile centering and fixed projection',
        retention_metrics=['raw-adjusted within-subject spatial Pearson correlation',
                           'adjusted/raw within-subject spatial score SD'],
        bootstrap_repetitions=args.bootstrap_repetitions,seed=args.seed,
        B_component_display_vmax=b_scales,
        B_scale_policy='Separate symmetric zero-centered range for each PC; same PC-specific range reused in D',
        D_component_display_vmax=d_scales,
        D_scale_policy='Fixed within PC across B raw map and D raw/adjusted maps, hemispheres and views',
        map_colormap='RdBu_r, matching final Figure 2 surface maps',
        map_normalization='symmetric around zero for signed PCA scores',
        map_clipping='98th percentile of absolute values pooled across compared maps; source values unchanged')
    (qc/'figure3_fixed_component_tracking.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':
    main()
