#!/usr/bin/env python3
"""PC2–thickness display using subject-level inference and individual examples.

Description
-----------
PC2–thickness display using subject-level inference and individual examples.

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
python analysis_results/figure3/computation/plot_pc2_vs_thickness.py --help
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qc.publication_style import finalize_figure, subject_violin

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def plot(root):
    root=Path(root);out=root/'pc1_pc2_thickness'
    subjects=pd.read_csv(root/'tables/figure3_subjects.csv')
    associations=pd.read_csv(out/'subject_thickness_pc1_pc2_correlations.csv')
    if subjects.subject_id.tolist()!=associations.subject_id.tolist():
        raise ValueError('Subject order mismatch')
    summary=pd.read_csv(out/'component_correlation_summary.csv').set_index('component').loc['PC2']
    t=np.load(root/'cache/thickness.npy',mmap_mode='r')
    pc2=np.load(out/'pc2_scores.npy',mmap_mode='r')
    mask=np.load(root/'tables/figure3_analysis_mask.npy').reshape(-1)
    if t.shape!=pc2.shape or t.shape!=(len(subjects),len(mask)):
        raise ValueError('Thickness/PC2 subject-vertex dimensions differ')
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],'font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(7.09,6))
    ax=axes.flat[0];rho=associations.rho_thickness_pc2.to_numpy(float)
    subject_violin(ax,rho,position=1,single_group=True,showfliers=True)
    rng=np.random.default_rng(20261103)
    ax.axhline(0,color='#888888',ls='--',lw=1)
    ax.set(xticks=[],xlabel='',ylabel='Within-subject spatial Spearman ρ',
        ylim=(min(-.03,rho.min()-.03),rho.max()+.04),title='Subject-level PC2–thickness association')
    ax.text(.04,.96,f'Median ρ = {summary.median_rho:.3f}\n95% subject-bootstrap CI\n'
        f'[{summary.median_rho_ci95_lower:.3f}, {summary.median_rho_ci95_upper:.3f}]',
        transform=ax.transAxes,va='top',fontsize=10)
    used=set();selected=[];bins=[]
    for panel,(quantile,ax) in enumerate(zip([.25,.5,.75],axes.flat[1:]),start=1):
        target=np.quantile(rho,quantile)
        index=next(int(i) for i in np.argsort(np.abs(rho-target),kind='stable') if int(i) not in used)
        used.add(index);subject=subjects.iloc[index].subject_id
        valid=mask & np.isfinite(t[index]) & np.isfinite(pc2[index])
        x=np.asarray(t[index,valid],float);y=np.asarray(pc2[index,valid],float)
        full_rho=float(spearmanr(x,y).statistic)
        if not np.isclose(full_rho,rho[index],atol=1e-10):
            raise ValueError('Example correlation differs from saved full-vertex correlation')
        sample=rng.choice(len(x),size=min(2000,len(x)),replace=False)
        ax.scatter(x[sample],y[sample],s=7,alpha=.16,color='#4B74B2',linewidths=0,rasterized=True)
        centers=[]
        for bi,ids in enumerate(np.array_split(np.argsort(x,kind='stable'),40)):
            centers.append((x[ids].mean(),y[ids].mean()))
            bins.append(dict(subject_id=subject,quantile=quantile,bin=bi,n_vertices=len(ids),
                mean_thickness=x[ids].mean(),mean_pc2_score=y[ids].mean()))
        centers=np.asarray(centers)
        ax.plot(centers[:,0],centers[:,1],color='#4B74B2',lw=1.6,marker='s',markevery=5,linestyle='--',label='Within-subject bin means')
        ax.set(xlabel='Native mapped cortical thickness (mm)',ylabel='PC2 score (MD units)',
            title=f'{subject}: spatial ρ = {full_rho:.3f}')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax.legend(frameon=False,fontsize=9)
        selected.append(dict(subject_id=subject,row_index=index,selection_quantile=quantile,
            target_rho=target,actual_rho=full_rho,n_valid_vertices=len(x),n_displayed_vertices=len(sample)))
    fig.suptitle('PC2 versus cortical thickness',fontweight='normal')
    fig.text(.5,.015,'Distribution: inference across subjects (10,000 bootstrap samples). Individual examples: subjects nearest the 25th/50th/75th rho percentiles.\n'
        'Points: 2,000 sampled vertices; dashed lines: 40 equal-count bins within that subject. Rho uses all valid vertices; no vertex-level p-values.',
        ha='center',fontsize=9)
    finalize_figure(fig)
    fig.tight_layout(rect=(0,.09,1,.95))
    for ext in ('png','pdf','svg'):
        fig.savefig(root/'figures'/f'Figure3S_PC2_vs_thickness.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(selected).to_csv(out/'pc2_thickness_representative_subjects.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'pc2_thickness_example_bins.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    plot(p.parse_args().output_dir)
