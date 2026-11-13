#!/usr/bin/env python3
"""Paired subject-level spatial thickness correlations for fixed PC1 and PC2.

Description
-----------
Paired subject-level spatial thickness correlations for fixed PC1 and PC2.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure3/computation/compare_pc1_pc2_thickness.py --help
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qc.publication_style import finalize_figure, subject_violin

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


def paired_summary(table, repetitions=10000, seed=20261103):
    n = len(table)
    if n < 2 or table.subject_id.duplicated().any():
        raise ValueError('At least two unique subjects are required')
    rho = table[['rho_thickness_pc1','rho_thickness_pc2']].to_numpy(float)
    if not np.isfinite(rho).all():
        raise ValueError('Nonfinite subject correlations')
    delta = np.abs(rho[:,1])-np.abs(rho[:,0])
    if not np.allclose(delta,table.delta_abs_rho):
        raise ValueError('Paired differences do not match the saved correlations')
    indices = np.random.default_rng(seed).integers(0,n,(repetitions,n))
    summaries = []
    for k, name in enumerate(('PC1','PC2')):
        values = rho[:,k]
        bootstrap = np.median(values[indices],axis=1)
        q25,q75 = np.percentile(values,[25,75])
        lo,hi = np.percentile(bootstrap,[2.5,97.5])
        summaries.append(dict(component=name,n_subjects=n,median_rho=float(np.median(values)),
            rho_q25=float(q25),rho_q75=float(q75),rho_iqr=float(q75-q25),
            median_rho_ci95_lower=float(lo),median_rho_ci95_upper=float(hi),
            median_abs_rho=float(np.median(np.abs(values))),bootstrap_samples=repetitions,seed=seed))
    boot_delta = np.median(delta[indices],axis=1)
    lo,hi = np.percentile(boot_delta,[2.5,97.5])
    if np.all(delta == 0):
        statistic,pvalue = 0.,1.
    else:
        test = wilcoxon(delta,alternative='two-sided',zero_method='wilcox',method='auto')
        statistic,pvalue = float(test.statistic),float(test.pvalue)
    comparison = dict(n_subjects=n,median_paired_difference=float(np.median(delta)),
        ci95_lower=float(lo),ci95_upper=float(hi),wilcoxon_statistic=statistic,wilcoxon_pvalue=pvalue,
        wilcoxon_alternative='two-sided',wilcoxon_zero_method='wilcox',wilcoxon_method='auto',
        n_nonzero_pairs=int(np.count_nonzero(delta)),bootstrap_samples=repetitions,seed=seed,
        difference_definition='abs(rho_thickness_pc2) - abs(rho_thickness_pc1)',
        bootstrap_statistic='median of paired subject differences, not difference of marginal medians')
    return pd.DataFrame(summaries),comparison,boot_delta


def plot_paired(table,comparison,output):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],'font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes = plt.subplots(1,2,figsize=(5.2,3.8),
        gridspec_kw={'width_ratios':[1,1]})
    colors = ['#4B74B2','#FC8C5A']
    for ax,cols,title,ylabel in [
        (axes[0],['rho_thickness_pc1','rho_thickness_pc2'],'Signed spatial correlations','Within-subject spatial Spearman ρ'),
        (axes[1],['abs_rho_pc1','abs_rho_pc2'],'Paired association magnitudes','Absolute within-subject spatial Spearman ρ')]:
        data = table[cols].to_numpy(float)
        for k,color in enumerate(colors):
            subject_violin(ax,data[:,k],position=k,color=color,
                           single_group=False,showfliers=True)
        ax.axhline(0,color='#888888',ls='--',lw=.8)
        ax.set(xticks=[0,1],xticklabels=['PC1','PC2'],ylabel=ylabel,title=title,xlim=(-.5,1.5))
    signed = table[['rho_thickness_pc1','rho_thickness_pc2']].to_numpy()
    bound = min(1.,max(.15,float(np.abs(signed).max())+.04))
    axes[0].set_ylim(-bound,bound)
    axes[1].set_ylim(-.015,bound)
    fig.suptitle('Thickness–profile component associations',fontweight='normal')
    c = comparison
    fig.text(.5,.035,f'Median paired Δ|ρ| (PC2 − PC1) = {c["median_paired_difference"]:.3f}; '
        f'95% subject-bootstrap CI [{c["ci95_lower"]:.3f}, {c["ci95_upper"]:.3f}]\n'
        f'Two-sided paired Wilcoxon p = {c["wilcoxon_pvalue"]:.3g}; 10,000 subject resamples; no vertex-level inference.',
        ha='center',fontsize=10)
    finalize_figure(fig)
    fig.tight_layout(rect=(0,.15,1,.94))
    for ext in ('pdf','svg','png'):
        fig.savefig(output/f'Figure3S_thickness_PC1_PC2_paired.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def run(root,seed=20261103,repetitions=10000):
    root = Path(root)
    out = root/'pc1_pc2_thickness';out.mkdir(exist_ok=True)
    subjects = pd.read_csv(root/'tables/figure3_subjects.csv',dtype={'subject_id':str})
    n = len(subjects)
    if not np.array_equal(subjects.row_index,np.arange(n)):
        raise ValueError('Noncanonical subject order')
    raw = np.load(root/'cache/raw_md.npy',mmap_mode='r')
    thickness = np.load(root/'cache/thickness.npy',mmap_mode='r')
    mask = np.load(root/'tables/figure3_analysis_mask.npy').reshape(-1)
    loadings = pd.read_csv(root/'tables/figure3B_pca_loadings.csv')
    if raw.shape != (n,len(mask),21) or thickness.shape != (n,len(mask)):
        raise ValueError('Cached subject/vertex dimensions do not match')
    if not np.allclose(loadings.normalized_depth,np.linspace(0,1,21)):
        raise ValueError('Noncanonical depth order')
    w = loadings[['PC1','PC2']].to_numpy(float)
    if not np.allclose(w[:,0],np.load(root/'cache/pc1_loading.npy')):
        raise ValueError('PC1 loading differs from saved PC1 scores')
    # Canonical subject×vertex caches include NaNs outside each subject's mask.
    pc1 = np.lib.format.open_memmap(out/'pc1_scores.npy',mode='w+',dtype='float32',shape=thickness.shape)
    pc2 = np.lib.format.open_memmap(out/'pc2_scores.npy',mode='w+',dtype='float32',shape=thickness.shape)
    pc1[:]=np.nan;pc2[:]=np.nan
    rows,qc = [],[]
    for i,subject in enumerate(subjects.subject_id):
        with np.load(root/'figure3B_subject_pc1_scores'/f'{subject}_pc1_scores.npz') as saved:
            if str(saved['subject_id'].item())!=subject:raise ValueError('PC1 subject identity mismatch')
            first = saved['scores'].reshape(-1).astype(float)
            saved_valid = saved['valid_mask'].reshape(-1).astype(bool)
        valid = mask & np.isfinite(thickness[i]) & np.isfinite(first)
        if not np.array_equal(valid,saved_valid):raise ValueError('PC1 and thickness validity masks differ')
        x = np.asarray(raw[i,valid],dtype=float)
        if not np.isfinite(x).all() or np.any(x<=0):raise ValueError('Invalid depth profile in score reconstruction')
        shape = x-x.mean(axis=1,keepdims=True)
        projected = shape@w
        if not np.allclose(first[valid],projected[:,0],rtol=1e-5,atol=1e-10):
            raise ValueError('Saved PC1 scores do not reproduce from fixed loadings')
        pc1[i,valid]=first[valid];pc2[i,valid]=projected[:,1]
        # Use the saved float32 score matrices for both correlations.
        t = np.asarray(thickness[i,valid],float)
        a,b = np.asarray(pc1[i,valid],float),np.asarray(pc2[i,valid],float)
        rho1 = float(spearmanr(t,a).statistic);rho2 = float(spearmanr(t,b).statistic)
        if not np.isfinite([rho1,rho2]).all():raise ValueError('Undefined subject-level spatial rho')
        rows.append(dict(subject_id=subject,rho_thickness_pc1=rho1,rho_thickness_pc2=rho2,
            abs_rho_pc1=abs(rho1),abs_rho_pc2=abs(rho2),delta_abs_rho=abs(rho2)-abs(rho1)))
        qc.append(dict(subject_id=subject,row_index=i,n_common_valid_vertices=int(valid.sum()),
            max_pc1_reconstruction_error=float(np.max(np.abs(first[valid]-projected[:,0])))))
        print(f'pc1_pc2_subject={i+1}/{n}',flush=True)
    pc1.flush();pc2.flush()
    table = pd.DataFrame(rows)
    old = pd.read_csv(root/'tables/figure3C_subject_spatial_correlations.csv').set_index('subject_id')
    # Historical correlations used float64 projections; allow float32 ranking precision.
    old_error = np.max(np.abs(table.rho_thickness_pc1.to_numpy()-old.loc[table.subject_id,'spearman_rho'].to_numpy()))
    if old_error>1e-5:raise ValueError('PC1 spatial correlations differ from previous analysis')
    table.to_csv(out/'subject_thickness_pc1_pc2_correlations.csv',index=False)
    summary,comparison,boot = paired_summary(table,repetitions,seed)
    summary.to_csv(out/'component_correlation_summary.csv',index=False)
    pd.DataFrame([comparison]).to_csv(out/'paired_absolute_correlation_comparison.csv',index=False)
    pd.DataFrame({'replicate':np.arange(repetitions),'median_delta_abs_rho':boot}).to_csv(out/'paired_bootstrap_differences.csv',index=False)
    pd.DataFrame(qc).to_csv(out/'subject_vertex_alignment_qc.csv',index=False)
    plot_paired(table,comparison,root/'figures')
    payload = dict(status='complete',n_subjects=n,seed=seed,bootstrap_samples=repetitions,
        score_shape=list(thickness.shape),statistical_unit='subject',
        score_source='PC1 copied from existing files; PC2 reconstructed from existing raw MD cache and saved fixed PC2 loading, without PCA refit',
        score_convention='profile-wise mean-centered MD dot saved loading; no additional score centering or normalization',
        pc2_sign='Existing saved PCA sign, largest-magnitude loading positive; absolute comparison is sign-invariant',
        vertex_support='Identical within-subject valid vertices for thickness, PC1 and PC2',
        max_difference_from_previous_pc1_rho=float(old_error),
        bootstrap_conditions='Fixed fitted PCA and masks; paired resampling of subjects; no vertex-level p-values',
        comparison=comparison)
    (out/'complete.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(summary.to_string(index=False));print(json.dumps(comparison))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--bootstrap-samples',type=int,default=10000)
    a=p.parse_args();run(a.output_dir,a.seed,a.bootstrap_samples)
