"""Plot the full-cohort Figure 2 results from saved subject tables and maps.

Description
-----------
Plot the full-cohort Figure 2 results from saved subject tables and maps.

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
python analysis_results/figure2/plot_full_cohort.py --help
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qc.publication_style import finalize_figure, PALETTE, COMPARISON_COLORS, apply_semantic_style, CMAP_SEQUENTIAL, heatmap_norm, subject_violin
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from nilearn.plotting import plot_surf_stat_map

NAMES={'O2Column_vs_O3':'O2-Column versus O3','O2FS_vs_O3':'O2-FS versus O3'}
SURFACE_RANGES={'O2Column_vs_O3':(.98,1.),'O2FS_vs_O3':(.8,1.)}
SURFACE_CMAP=plt.colormaps['RdBu_r'].copy()
SURFACE_CMAP.set_bad(PALETTE['missing'])
COLORS=COMPARISON_COLORS
DARK_COLORS={'O2Column_vs_O3':'#975436','O2FS_vs_O3':'#304A72'}


def violin_box(ax,values,slug):
    subject_violin(ax,values,position=0,single_group=True,showfliers=True)
    ax.set(xticks=[],xlabel='')


def save(fig,root,stem):
    finalize_figure(fig)
    for ext in ('png','pdf','svg'):fig.savefig(root/f'{stem}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def plot_all(root,fsaverage,statistics_only=False):
    root,fsaverage=Path(root),Path(fsaverage)
    out=root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],'font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    apply_semantic_style()
    for cohort in ('common_cohort','max_available'):
        data=pd.read_csv(root/cohort/'subject_metrics.csv')
        curve=pd.read_csv(root/cohort/'curvature_summary.csv')
        label='Matched common cohort' if cohort=='common_cohort' else 'Maximum available sample for each comparison'
        for slug,name in NAMES.items():
            # Both panels are single subject-level distributions with equal
            # scientific status, so equal narrow panels are appropriate.
            fig,axes=plt.subplots(1,2,figsize=(5.2,3.8),
                gridspec_kw={'width_ratios':[1,1]})
            for ax,metric,ylabel in zip(axes,['profile_r','profile_snmae_pct'],['Subject median profile correlation','Subject median sNMAE (%)']):
                values=data.loc[data.comparison.eq(slug)&data.metric.eq(metric),'subject_median'].dropna().to_numpy()
                violin_box(ax,values,slug)
                ax.set_ylabel(ylabel)
            fig.suptitle(f'Profile agreement: {name}',fontweight='normal')
            fig.tight_layout(rect=(0,0,1,.92))
            save(fig,out,f'Figure2_{cohort}_{slug}_agreement')
        for slug,name in NAMES.items():
            c=curve.loc[curve.comparison.eq(slug)].sort_values('curvature_bin')
            fig,ax=plt.subplots(figsize=(5.2,3.25))
            ax.plot(c.median_curvature_across_subjects,c.median_mae_across_subjects,'o-',color=COLORS[slug])
            ax.fill_between(c.median_curvature_across_subjects,c.q25_mae_across_subjects,c.q75_mae_across_subjects,color=PALETTE['light_blue'] if slug=='O2FS_vs_O3' else COLORS[slug],alpha=.12)
            ax.set(xlabel='Subject-specific curvature bin median',ylabel='Subject median profile MAE (MD units)')
            fig.suptitle(f'Curvature dependence: {name}',fontweight='normal')
            fig.tight_layout(rect=(0,0,1,.92))
            save(fig,out,f'Figure2_{cohort}_{slug}_curvature')

            fig,ax=plt.subplots(figsize=(2.6,3.6))
            values=data.loc[data.comparison.eq(slug)&data.metric.eq('curvature_mae_spearman'),'subject_median'].dropna().to_numpy()
            violin_box(ax,values,slug)
            ax.axhline(0,color='#888888',ls='--',lw=.8)
            ax.set(xticks=[],xlabel='',ylabel='Within-subject Spearman ρ (curvature versus MAE)')
            fig.suptitle(f'Curvature–error association: {name}',fontweight='normal')
            fig.tight_layout(rect=(0,0,1,.92))
            save(fig,out,f'Figure2_{cohort}_{slug}_curvature_association')
        if not statistics_only:
            for slug in NAMES:
                plot_surface(root,fsaverage,cohort,slug)
    write_surface_caption(root)
    (root/'results_complete.json').write_text(json.dumps(dict(status='complete',
        subject_statistics='analysis_complete.json',figure_formats=['png','pdf','svg'],png_dpi=300,
        agreement_surface='RdBu_r: O2-Column r=0.98–1.00; O2-FS r=0.8–1.00; one version per comparison; missing vertices gray'),indent=2)+'\n')


def write_surface_caption(root):
    out=Path(root)/'figures'
    (out/'Figure2_surface_maps_caption.txt').write_text(
        'Blue–white–red surface agreement maps (RdBu_r): O2-Column versus O3 uses r=0.98–1.00; O2-FS versus O3 uses r=0.8–1.00. '
        'Blue is lower and red is higher within the stated display range. All hemispheres and views of a given comparison use identical limits. '
        'Ranges differ between comparisons, so use numeric r values and subject summaries for between-method comparisons. '
        'Values outside the display limits are color-saturated; source correlations and masks are unchanged. Missing vertices are gray.\n')
    marker=Path(root)/'results_complete.json'
    if marker.exists():
        record=json.loads(marker.read_text())
        record['agreement_surface']='RdBu_r: O2-Column r=0.98–1.00; O2-FS r=0.8–1.00; one canonical version per comparison'
        record['surface_ranges']=SURFACE_RANGES
        marker.write_text(json.dumps(record,indent=2)+'\n')


def clean_obsolete_detail_maps(root):
    # These redundant generated versions were explicitly rejected by the user.
    out=Path(root)/'figures'
    for cohort in ('common_cohort','max_available'):
        for ext in ('png','pdf','svg','json'):
            name=f'Figure2A_{cohort}_O2Column_vs_O3_surface_detail'
            p=out/(name+'_display.json' if ext=='json' else name+'.'+ext)
            if p.is_file():p.unlink()
    p=out/'Figure2_O2Column_surface_detail_caption.txt'
    if p.is_file():p.unlink()


def plot_surface(root,fsaverage,cohort,slug):
    vmin,vmax=SURFACE_RANGES[slug]
    root,fsaverage=Path(root),Path(fsaverage)
    out=root/'figures'
    label='Matched common cohort' if cohort=='common_cohort' else 'Maximum available sample for each comparison'
    fig=plt.figure(figsize=(7.09,2.8));axes=[];source_values={}
    for index,(h,hemi,view) in enumerate([('lh','left','lateral'),('lh','left','medial'),('rh','right','lateral'),('rh','right','medial')]):
        table=pd.read_csv(root/cohort/f'{slug}_agreement_map_{h}.csv')
        values=table.mean_profile_r_fisher.to_numpy(dtype=float,copy=True)
        values[table.valid_subject_fraction.to_numpy()<.8]=np.nan
        source_values[h]=values
        ax=fig.add_subplot(1,4,index+1,projection='3d');axes.append(ax)
        mesh=fsio.read_geometry(fsaverage/'surf'/f'{h}.inflated')
        sulc=fsio.read_morph_data(fsaverage/'surf'/f'{h}.sulc')
        view_title=f'{"Left" if h == "lh" else "Right"} {view}'
        plot_surf_stat_map(mesh,values,hemi=hemi,view=view,bg_map=sulc,bg_on_data=True,
            cmap=SURFACE_CMAP,vmin=vmin,vmax=vmax,colorbar=False,axes=ax,figure=fig,title=f'{h.upper()} {view}')
        for collection in ax.collections:collection.set_rasterized(True)
        ax.set_title(view_title)
    fig.suptitle(f'Vertex-wise profile agreement: {NAMES[slug]}',fontweight='normal')
    fig.colorbar(plt.cm.ScalarMappable(norm=heatmap_norm("nonnegative",vmin=vmin,vmax=vmax),cmap=SURFACE_CMAP),ax=axes,
                 shrink=.55,label='Fisher-mean profile r')
    stem=f'Figure2A_{cohort}_{slug}_surface'
    save(fig,out,stem)
    finite=np.concatenate([values[np.isfinite(values)] for values in source_values.values()])
    display=dict(cohort=cohort,comparison=slug,cmap=SURFACE_CMAP.name,vmin=vmin,vmax=vmax,
        hemisphere_and_view_limits='identical',n_valid_vertices=len(finite),
        n_below_vmin=int(np.sum(finite<vmin)),fraction_below_vmin=float(np.mean(finite<vmin)),
        n_above_vmax=int(np.sum(finite>vmax)),source_min=float(finite.min()),source_max=float(finite.max()),
        note='Color saturation only; no changes to source correlations or analysis mask')
    (out/f'{stem}_display.json').write_text(json.dumps(display,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--fsaverage-dir',type=Path,required=True)
    parser.add_argument('--surfaces-only',action='store_true',help='Replace surface maps with the final red-blue versions')
    parser.add_argument('--statistics-only',action='store_true',help='Reuse previously rendered surface maps')
    args=parser.parse_args()
    if args.surfaces_only:
        apply_semantic_style()
        for cohort in ('common_cohort','max_available'):
            for slug in NAMES:
                plot_surface(args.output_dir,args.fsaverage_dir,cohort,slug)
        write_surface_caption(args.output_dir)
        clean_obsolete_detail_maps(args.output_dir)
    else:
        plot_all(args.output_dir,args.fsaverage_dir,args.statistics_only)
        if not args.statistics_only:
            clean_obsolete_detail_maps(args.output_dir)
