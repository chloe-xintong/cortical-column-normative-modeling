"""Deterministic A/D notebook display, independent of inherited layout/DPI.

Description
-----------
Deterministic A/D notebook display, independent of inherited layout/DPI.

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
python analysis_results/figure3/computation/figure3_profile_plot.py --help
"""
import matplotlib as mpl
if __name__ == '__main__':
    mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator, ScalarFormatter
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qc.publication_style import finalize_figure

import numpy as np


def draw_raw_adjusted_profiles(raw, adjusted, n_subjects, selected, ranking):
    from select_representative_regions import draw_thickness_distribution, display_region_name, REGION_STYLES
    from figure3_panel_a_plot import mean_ci95_halfwidth
    names=selected.region_name.tolist()
    colors=dict(zip(names,selected.color))
    curves = []
    for panel,data in [('raw',raw),('adjusted',adjusted)]:
        for region in names:
            rows = data.loc[data.region.eq(region)].sort_values('depth_index')
            values = rows[['normalized_depth','mean_md','sd_md']].to_numpy(float,copy=True)
            if (values.shape != (21,3) or not np.isfinite(values).all()
                    or not np.allclose(values[:,0],np.linspace(0,1,21))
                    or np.any(values[:,2]<0) or not rows.n_subjects.eq(n_subjects).all()):
                raise ValueError(f'{panel} {region}: incomplete/nonfinite profiles, invalid SD, depths or subject count')
            values[:,2]=mean_ci95_halfwidth(values[:,2],rows.n_subjects.to_numpy())
            curves.append(values)
    low = min(float(np.min(x[:,1]-x[:,2])) for x in curves)
    high = max(float(np.max(x[:,1]+x[:,2])) for x in curves)
    span = high-low
    if not np.isfinite([low,high]).all() or span<=0:
        raise ValueError('Cannot determine a finite, nondegenerate MD plotting range')
    limits = (low-.04*span, high+.08*span)
    # Avoid tight/constrained layout and automatic tick-space calculations at
    # notebook display time. Neither limits nor ticks depend on a renderer bbox.
    style = {'figure.dpi':120., 'savefig.dpi':120., 'figure.autolayout':False,
             'figure.constrained_layout.use':False, 'font.family':'sans-serif', 'font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],
             'font.size':11., 'xtick.labelsize':11., 'ytick.labelsize':11.,
             'axes.labelsize':11., 'axes.titlesize':12., 'text.usetex':False,
             'axes.spines.top':False, 'axes.spines.right':False}
    with mpl.rc_context(style):
        fig,axes = plt.subplots(1,2,figsize=(7.09,3.1),dpi=120.,sharex=True,sharey=True,layout=None)
        fig.subplots_adjust(left=.085,right=.985,bottom=.19,top=.80,wspace=.12)
        ticks = MaxNLocator(nbins=6).tick_values(*limits)
        for panel,(ax,title) in enumerate(zip(axes,['Raw cortical-depth profiles','Thickness-adjusted profiles'])):
            for j,region in enumerate(names):
                values = curves[3*panel+j];x,mean,sd = values.T
                ax.plot(x,mean,**REGION_STYLES[selected.loc[selected.region_name.eq(region),'selected_level'].iloc[0]],markevery=4,label=display_region_name(region))
                ax.fill_between(x,mean-sd,mean+sd,color=colors[region],alpha=.12)
            ax.set(xlabel='Normalized depth (0 pial → 1 white)',xlim=(0,1),ylim=limits)
            # Matplotlib 3.11 may auto-position the right shared-y title at
            # infinity when its hidden scientific-offset bbox is consulted.
            ax.set_title(title,y=1.02,pad=10)
            ax.xaxis.set_major_locator(FixedLocator(np.linspace(0,1,6)))
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            formatter = ScalarFormatter(useMathText=False);formatter.set_powerlimits((0,0))
            ax.yaxis.set_major_formatter(formatter)
        axes[0].set_ylabel('MD (original image units)')
        axes[1].legend(frameon=False,fontsize=10)
        fig.suptitle('Mean with 95% CI',fontsize=12)
        thickness_ax=axes[0].inset_axes([.49,.55,.48,.23])
        thickness_ax.set_facecolor((1,1,1,.90))
        draw_thickness_distribution(thickness_ax,ranking,compact=True)
        return fig,axes


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    import pandas as pd
    parser = argparse.ArgumentParser(description='Render A/D profiles without a notebook display backend')
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    root = args.output_dir
    from select_representative_regions import prepare, write_profile_caption
    ranking,selected=prepare(root)
    mpl.rcdefaults()
    mpl.rcParams.update({'figure.dpi':120.,'savefig.dpi':300.})
    raw = pd.read_csv(root/'tables/figure3A_bilateral_group_profiles.csv')
    adjusted = pd.read_csv(root/'tables/figure3D_bilateral_adjusted_group_profiles.csv')
    n = len(pd.read_csv(root/'tables/figure3_subjects.csv'))
    fig,_ = draw_raw_adjusted_profiles(raw,adjusted,n,selected,ranking)
    finalize_figure(fig)
    for ext in ('png','pdf','svg'):
        fig.savefig(root/'figures'/f'Figure3_AD_regional_profiles.{ext}',dpi=300,bbox_inches=None)
    plt.close(fig)
    write_profile_caption(root)
