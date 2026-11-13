#!/usr/bin/env python3
"""Supplementary Figure 3C analysis of fixed PC scores and cortical curvature.

Description
-----------
Supplementary Figure 3C analysis of fixed PC scores and cortical curvature.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input
- ``--legacy-curvature-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure3/computation/compute_curvature_supplement.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qc.publication_style import (PALETTE, finalize_figure, set_publication_style,
                                  subject_violin)


FS_SIGN_SOURCE = 'https://surfer.nmr.mgh.harvard.edu/fswiki/UserContributions/FAQ'
GROUPS = (
    ('PC1', 'signed', 0.0, PALETTE['blue'], 'PC1\nsigned'),
    ('PC2', 'signed', 1.0, PALETTE['orange'], 'PC2\nsigned'),
    ('PC1', 'absolute', 2.7, PALETTE['light_blue'], 'PC1\n|curvature|'),
    ('PC2', 'absolute', 3.7, '#FDBB9A', 'PC2\n|curvature|'),
)


def save(fig, directory, stem):
    finalize_figure(fig)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(directory/f'{stem}.{extension}', dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)


def mapped_curvature(mapping_path, curvature_path):
    curvature=np.asarray(fsio.read_morph_data(str(curvature_path)),dtype=float)
    with np.load(mapping_path,allow_pickle=False) as data:
        target=np.asarray(data['target_vertex_id'],dtype=np.int64)
        source=np.asarray(data['source_vertex_ids'],dtype=np.int64)
        weights=np.asarray(data['barycentric_weights'],dtype=float)
        mapping_valid=np.asarray(data['mapping_valid'],dtype=bool)
    if not np.array_equal(target,np.arange(len(target))):
        raise ValueError(f'{mapping_path}: target vertices are not in full template order')
    if source.shape!=(len(target),3) or weights.shape!=source.shape:
        raise ValueError(f'{mapping_path}: invalid triangle/weight shape')
    in_range=np.all((source>=0)&(source<len(curvature)),axis=1)
    finite=np.zeros(len(target),bool)
    finite[in_range]=np.isfinite(curvature[source[in_range]]).all(axis=1)
    weight_valid=(np.isfinite(weights).all(axis=1)&
                  (np.abs(weights.sum(axis=1)-1)<=1e-6)&
                  np.all((weights>=-1e-6)&(weights<=1+1e-6),axis=1))
    valid=mapping_valid&in_range&finite&weight_valid
    signed=np.full(len(target),np.nan);absolute=np.full(len(target),np.nan)
    signed[valid]=np.einsum('ij,ij->i',weights[valid],
                            curvature[source[valid]])
    absolute[valid]=np.einsum('ij,ij->i',weights[valid],
                              np.abs(curvature[source[valid]]))
    return signed,absolute,valid,curvature


def pearson(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    a=a-a.mean();b=b-b.mean()
    denominator=np.sqrt(a@a)*np.sqrt(b@b)
    if not np.isfinite(denominator) or denominator<=0:
        raise ValueError('Undefined correlation')
    return float((a@b)/denominator)


def spearman_from_ranks(score_rank, predictor_rank):
    return pearson(score_rank,predictor_rank)


def multivariable_effect(score,thickness,curvature):
    """OLS score ~ intercept + thickness + curvature from sufficient statistics."""
    y=np.asarray(score,float);t=np.asarray(thickness,float);c=np.asarray(curvature,float)
    yc=y-y.mean();tc=t-t.mean();cc=c-c.mean()
    yy=float(yc@yc);tt=float(tc@tc);ccc=float(cc@cc)
    ty=float(tc@yc);cy=float(cc@yc);t_c=float(tc@cc)
    if min(yy,tt,ccc)<=0:
        raise ValueError('Constant score or predictor')
    xtx=np.array([[tt,t_c],[t_c,ccc]],float)
    condition=float(np.linalg.cond(xtx))
    beta=np.linalg.solve(xtx,np.array([ty,cy],float))
    sse_full=max(0.0,yy-float(beta@np.array([ty,cy])))
    sse_thickness=max(0.0,yy-ty*ty/tt)
    full_r2=float(np.clip(1-sse_full/yy,0,1))
    thickness_r2=float(np.clip(1-sse_thickness/yy,0,1))
    partial=float(np.clip((sse_thickness-sse_full)/sse_thickness,0,1))
    sd_y=np.sqrt(yy/(len(y)-1));sd_t=np.sqrt(tt/(len(y)-1));sd_c=np.sqrt(ccc/(len(y)-1))
    return dict(
        raw_beta_thickness=float(beta[0]),raw_beta_curvature=float(beta[1]),
        standardized_beta_thickness=float(beta[0]*sd_t/sd_y),
        standardized_beta_curvature=float(beta[1]*sd_c/sd_y),
        thickness_only_r2=thickness_r2,full_model_r2=full_r2,
        incremental_r2_curvature=float(max(0,full_r2-thickness_r2)),
        partial_r2_curvature=partial,
        partial_correlation_curvature=float(np.sign(beta[1])*np.sqrt(partial)),
        pearson_thickness_curvature=pearson(t,c),
        predictor_xtx_condition_number=condition)


def bootstrap_summary(table,repetitions,seed):
    rng=np.random.default_rng(seed);rows=[]
    metrics=('spatial_spearman_rho','standardized_beta_curvature',
             'partial_r2_curvature')
    for (component,curvature_metric),part in table.groupby(
            ['component','curvature_metric'],sort=False):
        for metric in metrics:
            values=part[metric].to_numpy(float);n=len(values)
            indices=rng.integers(0,n,(repetitions,n))
            boot=np.median(values[indices],axis=1)
            lo,hi=np.percentile(boot,[2.5,97.5]);q25,q75=np.percentile(values,[25,75])
            rows.append(dict(component=component,curvature_metric=curvature_metric,
                metric=metric,n_subjects=n,median=float(np.median(values)),
                q25=float(q25),q75=float(q75),ci95_lower=float(lo),
                ci95_upper=float(hi),bootstrap_repetitions=repetitions,seed=seed))
    return pd.DataFrame(rows)


def plot_metric(root,table,summary,metric,ylabel,title,stem,zero=False):
    fig,ax=plt.subplots(figsize=(6.1,3.25))
    positions=[];labels=[]
    for component,curvature_metric,position,color,label in GROUPS:
        values=table.loc[(table.component==component)&
                         (table.curvature_metric==curvature_metric),metric].to_numpy(float)
        row=summary.loc[(summary.component==component)&
                        (summary.curvature_metric==curvature_metric)&
                        (summary.metric==metric)].iloc[0]
        subject_violin(ax,values,position=position,color=color,
                       single_group=False,showfliers=True)
        ax.errorbar(position,row['median'],
                    yerr=[[row['median']-row.ci95_lower],
                          [row.ci95_upper-row['median']]],
                    fmt='D',color=color,markeredgecolor='white',markersize=5,
                    capsize=3,linewidth=1.3,zorder=5)
        positions.append(position);labels.append(label)
    if zero:
        ax.axhline(0,color='#A8A8A8',linestyle='--',linewidth=.8,zorder=0)
    ax.axvline(1.85,color='#D0D0D0',linewidth=.7,zorder=0)
    ax.set(xticks=positions,xticklabels=labels,xlim=(-.6,4.3),ylabel=ylabel,title=title)
    save(fig,root/'figures',stem)


def validate_legacy(subject,hemisphere,signed,absolute,legacy_root):
    path=legacy_root/subject/'fsaverage'/f'{subject}_{hemisphere}_curvature.npz'
    if not path.is_file():
        return np.nan,np.nan,False
    with np.load(path,allow_pickle=False) as data:
        previous_signed=np.asarray(data['signed_mean_curvature'],float)
        previous_absolute=np.asarray(data['absolute_mean_curvature'],float)
    valid=np.isfinite(signed)&np.isfinite(previous_signed)
    valid_abs=np.isfinite(absolute)&np.isfinite(previous_absolute)
    return (float(np.max(np.abs(signed[valid]-previous_signed[valid]))),
            float(np.max(np.abs(absolute[valid_abs]-previous_absolute[valid_abs]))),True)


def process_subject(subject,index,input_rows,mask,thickness,score1,score2,
                    legacy_root,subject_dir):
    signed_parts=[];absolute_parts=[];mapping_rows=[]
    for hemisphere in ('lh','rh'):
        row=input_rows.loc[(subject,hemisphere)]
        curvature_path=Path(row.native_thickness_file).with_name(f'{hemisphere}.curv')
        signed,absolute,mapping_valid,native=mapped_curvature(
            Path(row.mapping_file),curvature_path)
        legacy_signed,legacy_absolute,has_legacy=validate_legacy(
            subject,hemisphere,signed,absolute,legacy_root)
        signed_parts.append(signed);absolute_parts.append(absolute)
        mapping_rows.append(dict(subject_id=subject,hemisphere=hemisphere,
            curvature_file=str(curvature_path),mapping_file=row.mapping_file,
            n_template_vertices=len(signed),n_valid_curvature=int(mapping_valid.sum()),
            curvature_min=float(np.nanmin(signed)),curvature_max=float(np.nanmax(signed)),
            native_positive_fraction=float(np.mean(native>0)),
            native_negative_fraction=float(np.mean(native<0)),
            legacy_validation_available=has_legacy,
            legacy_signed_max_abs_difference=legacy_signed,
            legacy_absolute_max_abs_difference=legacy_absolute))
    signed=np.concatenate(signed_parts);absolute=np.concatenate(absolute_parts)
    t=np.asarray(thickness[index],float);scores=(np.asarray(score1[index],float),
                                                np.asarray(score2[index],float))
    valid=mask&np.isfinite(t)&np.isfinite(signed)&np.isfinite(absolute)
    for score in scores:
        valid&=np.isfinite(score)
    if valid.sum()<1000:
        raise ValueError(f'{subject}: only {valid.sum()} valid curvature vertices')
    t=t[valid];signed=signed[valid];absolute=absolute[valid]
    score_values=[score[valid] for score in scores]
    score_ranks=[rankdata(score,method='average') for score in score_values]
    curvature_values={'signed':signed,'absolute':absolute}
    curvature_ranks={name:rankdata(value,method='average')
                     for name,value in curvature_values.items()}
    rows=[]
    for component,(score,score_rank) in enumerate(zip(score_values,score_ranks),start=1):
        for curvature_metric,curvature in curvature_values.items():
            effect=multivariable_effect(score,t,curvature)
            rows.append(dict(subject_id=subject,component=f'PC{component}',
                curvature_metric=curvature_metric,n_valid_vertices=int(valid.sum()),
                spatial_spearman_rho=spearman_from_ranks(
                    score_rank,curvature_ranks[curvature_metric]),
                spatial_pearson_r=pearson(score,curvature),**effect))
    metrics=pd.DataFrame(rows);mapping_qc=pd.DataFrame(mapping_rows)
    metrics.to_csv(subject_dir/f'{subject}_metrics.csv',index=False)
    mapping_qc.to_csv(subject_dir/f'{subject}_mapping_qc.csv',index=False)
    return metrics,mapping_qc


def interpretation(summary):
    s=summary.set_index(['component','curvature_metric','metric'])
    pieces=[]
    for metric,label in (('signed','signed curvature'),('absolute','absolute curvature')):
        for component in ('PC1','PC2'):
            rho=s.loc[(component,metric,'spatial_spearman_rho')]
            beta=s.loc[(component,metric,'standardized_beta_curvature')]
            r2=s.loc[(component,metric,'partial_r2_curvature')]
            pieces.append(
                f'{component}, {label}: median spatial rho {rho["median"]:.3f} '
                f'(95% CI {rho.ci95_lower:.3f} to {rho.ci95_upper:.3f}); '
                f'median standardized curvature beta after thickness adjustment '
                f'{beta["median"]:.3f} (95% CI {beta.ci95_lower:.3f} to '
                f'{beta.ci95_upper:.3f}); median curvature partial R2 '
                f'{r2["median"]:.3f} (95% CI {r2.ci95_lower:.3f} to '
                f'{r2.ci95_upper:.3f}).')
    return ('\n'.join(pieces)+'\n\nAssociations are observational. They do not distinguish '
            'anatomical organization from sampling geometry, and curvature is not '
            'removed from the primary Figure 3 analysis.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--legacy-curvature-root',type=Path,required=True)
    parser.add_argument('--bootstrap-repetitions',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=20261109)
    args=parser.parse_args();root=args.output_dir
    supplement=root/'curvature_supplement';subject_dir=supplement/'subjects'
    subject_dir.mkdir(parents=True,exist_ok=True);(root/'figures').mkdir(exist_ok=True)
    set_publication_style()
    subjects=pd.read_csv(root/'tables/figure3_subjects.csv',dtype={'subject_id':str})
    inputs=pd.read_csv(root/'qc/subject_input_qc.csv',dtype={'subject_id':str})
    inputs=inputs.set_index(['subject_id','hemisphere']).sort_index()
    mask=np.load(root/'tables/figure3_analysis_mask.npy').reshape(-1).astype(bool)
    thickness=np.load(root/'cache/thickness.npy',mmap_mode='r')
    score1=np.load(root/'pc1_pc2_thickness/pc1_scores.npy',mmap_mode='r')
    score2=np.load(root/'pc1_pc2_thickness/pc2_scores.npy',mmap_mode='r')
    if thickness.shape!=score1.shape or score1.shape!=score2.shape or score1.shape!=(len(subjects),len(mask)):
        raise ValueError('Thickness, fixed PC scores, cohort and mask are misaligned')
    metric_parts=[];qc_parts=[]
    for index,subject in enumerate(subjects.subject_id):
        metric_path=subject_dir/f'{subject}_metrics.csv'
        qc_path=subject_dir/f'{subject}_mapping_qc.csv'
        if metric_path.is_file() and qc_path.is_file():
            metrics=pd.read_csv(metric_path,dtype={'subject_id':str})
            mapping_qc=pd.read_csv(qc_path,dtype={'subject_id':str})
            if len(metrics)!=4 or len(mapping_qc)!=2:
                raise ValueError(f'{subject}: incomplete cached curvature supplement')
        else:
            metrics,mapping_qc=process_subject(
                subject,index,inputs,mask,thickness,score1,score2,
                args.legacy_curvature_root,subject_dir)
        metric_parts.append(metrics);qc_parts.append(mapping_qc)
        print(f'curvature_subject={index+1}/{len(subjects)} {subject}',flush=True)
    metrics=pd.concat(metric_parts,ignore_index=True)
    mapping_qc=pd.concat(qc_parts,ignore_index=True)
    if len(metrics)!=len(subjects)*4 or metrics.subject_id.nunique()!=len(subjects):
        raise ValueError('Incomplete subject-level curvature metrics')
    summary=bootstrap_summary(metrics,args.bootstrap_repetitions,args.seed)
    metrics.to_csv(supplement/'subject_pc_curvature_metrics.csv',index=False)
    summary.to_csv(supplement/'pc_curvature_group_summary.csv',index=False)
    mapping_qc.to_csv(supplement/'curvature_mapping_qc.csv',index=False)
    plot_metric(root,metrics,summary,'spatial_spearman_rho',
                'Within-subject spatial Spearman ρ',
                'Fixed PC scores and cortical curvature',
                'Figure3S_C01_PC_curvature_spatial_rho',zero=True)
    plot_metric(root,metrics,summary,'standardized_beta_curvature',
                'Standardized curvature β',
                'Curvature effect after controlling thickness',
                'Figure3S_C02_PC_curvature_adjusted_beta',zero=True)
    plot_metric(root,metrics,summary,'partial_r2_curvature',
                'Curvature partial R²',
                'Additional score variance explained beyond thickness',
                'Figure3S_C03_PC_curvature_partial_r2',zero=False)
    text=interpretation(summary)
    (root/'figures/Figure3S_C_curvature_interpretation.txt').write_text(text)
    legacy=mapping_qc.loc[mapping_qc.legacy_validation_available.astype(bool)]
    status=dict(status='complete',n_subjects=len(subjects),statistical_unit='subject',
        components=['PC1','PC2'],curvature_metrics=['signed','absolute'],
        curvature_source='native FreeSurfer lh.curv/rh.curv mapped with each subject Option 3 triangle IDs and barycentric weights',
        signed_curvature_definition='weighted signed native FreeSurfer curv; positive=sulcal/folded-in, negative=gyral/folded-out',
        absolute_curvature_definition='weighted mean of absolute native curvature, not absolute value after interpolation',
        freesurfer_sign_source=FS_SIGN_SOURCE,
        score_source='fixed PC1/PC2 scores from the same Figure 3 cohort and valid vertices; intercept makes the saved score-origin convention irrelevant to spatial effects',
        models=['within-subject spatial Spearman correlation',
                'within-subject OLS PC score ~ thickness + curvature'],
        curvature_effect_metrics=['standardized beta','partial R2 relative to thickness-only model'],
        bootstrap_repetitions=args.bootstrap_repetitions,seed=args.seed,
        inference='median and percentile 95% CI from subject bootstrap',
        minimum_valid_vertices=int(metrics.n_valid_vertices.min()),
        median_valid_vertices=float(metrics.groupby('subject_id').n_valid_vertices.first().median()),
        legacy_validation_subject_hemispheres=len(legacy),
        legacy_signed_max_abs_difference=float(legacy.legacy_signed_max_abs_difference.max()) if len(legacy) else None,
        legacy_absolute_max_abs_difference=float(legacy.legacy_absolute_max_abs_difference.max()) if len(legacy) else None,
        primary_analysis_changed=False,
        interpretation='association check only; does not distinguish anatomy from sampling geometry and curvature is not regressed from primary results')
    (supplement/'complete.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':
    main()
