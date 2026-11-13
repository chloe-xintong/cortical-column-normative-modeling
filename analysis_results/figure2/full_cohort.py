#!/usr/bin/env python3
"""Extend Figure 2 A/B to all available QC-passed subjects, retaining old inputs.

Description
-----------
Extend Figure 2 A/B to all available QC-passed subjects, retaining old inputs.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``stage`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure2/full_cohort.py --help
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

import figure2_method_comparison as original

HERE=Path(__file__).resolve().parent
COMPARISONS={'O2Column_vs_O3':'O2-Column','O2FS_vs_O3':'O2-FS'}


def run(command,env=None):
    subprocess.run([str(x) for x in command],check=True,env=env)


def dump(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def existing_fs(args,subject,hemi):
    candidates=[args.results_root/'qc/o2_fs'/subject/'fsaverage'/hemi/'vol2surf_manifest.json',
        args.results_root/'figure5_age_prediction/outputs/option2_fs_baselines/source'/subject/hemi/'md/vol2surf_manifest.json']
    for p in candidates:
        if p.is_file():return p
    return None


def inventory(args):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    excluded=set(pd.read_csv(args.qc_exclusions).subject_id)
    rows=[];included=[]
    root=args.results_root/'fsaverage_result/columns'
    for p in sorted(root.iterdir()):
        if not p.is_dir():continue
        s=p.name
        complete=all((p/'profiles'/f'{s}_{h}_md_profiles.npz').is_file() for h in ('lh','rh'))
        available_column=all(original.method_profile_path(args.results_root,'O2-Column',s,h,'fsaverage').is_file()
                             for h in ('lh','rh'))
        native=all(any((args.native_mapping_root/s/h/'regions').glob('*_native_to_fsaverage.csv.gz')) for h in ('lh','rh'))
        basic=all((args.subjects_dir/s/'surf'/f'{h}.curv').is_file() and
                  (args.subjects_dir/s/'surf'/f'{h}.white').is_file() and
                  (args.subjects_dir/s/'surf'/f'{h}.pial').is_file() for h in ('lh','rh'))
        md=(args.results_root.parent/'data/processed/dwi'/s/f'{s}_md.nii.gz').is_file()
        reg=(p/'registration'/f'{s}_DWI2T1.dat').is_file()
        status='registration_qc_fail' if s in excluded else ('missing_O3' if not complete else
                 ('missing_production_inputs' if not(basic and md and reg) else 'included'))
        row=dict(subject_id=s,status=status,o2_column_existing=available_column,
                 o2_column_input_available=available_column or native,native_source_available=native,
                 o2_fs_existing=all(existing_fs(args,s,h) is not None for h in ('lh','rh')),
                 o2_column_unavailable_reason='' if available_column or native else 'Missing comparable archived native-column inputs')
        rows.append(row)
        if status=='included':included.append(row)
    pd.DataFrame(rows).to_csv(args.output_dir/'candidate_inventory.csv',index=False)
    table=pd.DataFrame(included)
    if table.empty:raise ValueError('No QC-passed complete Option 3 subjects')
    table.insert(0,'row_index',np.arange(len(table)))
    table.to_csv(args.output_dir/'subjects.csv',index=False)
    dump(args.output_dir/'inventory.json',dict(n_included=len(table),n_candidates=len(rows),
        registration_qc_exclusions=sorted(excluded),
        n_o2_column_available=int(table.o2_column_input_available.sum()),
        n_o2_fs_reusable=int(table.o2_fs_existing.sum()),
        strategy='Maximal valid sample for each pair, plus the matched common subject cohort; no CN restriction',
        old_50_subject_outputs='Read-only, retained unchanged'))
    dump(args.output_dir/'configuration.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    print((args.output_dir/'inventory.json').read_text())


def fs_profiles(manifest_path,subject,hemi):
    d=json.loads(manifest_path.read_text());outputs=d['outputs']
    if (d['subject_id']!=subject or d['hemisphere']!=hemi or d['template']!='fsaverage'
            or d['depth_direction']!='pial_to_white' or d['interp']!='trilinear' or len(outputs)!=21
            or not np.allclose([x['depth'] for x in outputs],original.CANONICAL_DEPTHS)
            or not np.allclose([x['projfrac'] for x in outputs],1-original.CANONICAL_DEPTHS)):
        raise ValueError(f'Invalid O2-FS source/depth manifest: {manifest_path}')
    values=np.column_stack([np.asarray(nib.load(x['output']).dataobj).reshape(-1) for x in outputs])
    return original.ensure_target_ids(np.arange(163842),values,np.ones(163842,bool),163842)


def prepare_subject(args,subject,has_column):
    dest=args.output_dir/'subjects'/subject;dest.mkdir(parents=True,exist_ok=True)
    paths={};env=os.environ.copy()
    env.update(FREESURFER_HOME=str(args.freesurfer_home),PATH=f'{args.freesurfer_home}/bin:'+env['PATH'])
    with tempfile.TemporaryDirectory(prefix='figure2-full-') as temp:
        sd=Path(temp);(sd/subject).symlink_to(args.subjects_dir/subject)
        (sd/'fsaverage').symlink_to(args.fsaverage_dir);env['SUBJECTS_DIR']=str(sd)
        for h in ('lh','rh'):
            mapping=args.results_root/'fsaverage_result/columns'/subject/'columns'/f'{subject}_{h}_columns.npz'
            manifest=existing_fs(args,subject,h)
            if manifest is None:
                md_dir=dest/'sources'/h/'o2_fs'
                if not (md_dir/'vol2surf_manifest.json').is_file():
                    run([sys.executable,HERE/'o2_fs/run_vol2surf.py','--subject-id',subject,
                         '--subjects-dir',sd,'--metric-image',args.results_root.parent/'data/processed/dwi'/subject/f'{subject}_md.nii.gz',
                         '--reg',args.results_root/'fsaverage_result/columns'/subject/'registration'/f'{subject}_DWI2T1.dat',
                         '--template','fsaverage','--hemisphere',h,
                         '--depths',','.join(f'{d:.2f}' for d in original.CANONICAL_DEPTHS),'--output-dir',md_dir],env)
                manifest=md_dir/'vol2surf_manifest.json'
            paths[f'fs_{h}']=str(manifest)
            curvature=args.results_root/'qc/template_registration_comparison/curvature/shared_barycentric'/subject/'fsaverage'/f'{subject}_{h}_curvature.npz'
            if not curvature.is_file():
                curvature=dest/'sources'/h/'shared_curvature.npz'
                if not curvature.is_file():
                    run([sys.executable,HERE/'curvature/map_barycentric_curvature.py','--subject-id',subject,
                         '--hemisphere',h,'--subjects-dir',args.subjects_dir,'--mapping-file',mapping,'--output',curvature])
            paths[f'curvature_{h}']=str(curvature)
            if has_column:
                column=original.method_profile_path(args.results_root,'O2-Column',subject,h,'fsaverage')
                if not column.is_file():
                    column=dest/'sources'/h/f'{subject}_{h}_profiles.npz'
                    if not column.is_file():
                        run([sys.executable,HERE/'o2_column/generate_o2_column_profiles.py','--subject-id',subject,
                             '--hemisphere',h,'--template','fsaverage','--mapping-root',args.native_mapping_root,
                             '--mapping-file',mapping,'--output',column])
                paths[f'column_{h}']=str(column)
    return paths


def subject_stage(args):
    subjects=pd.read_csv(args.output_dir/'subjects.csv')
    row=subjects.iloc[args.index];s=row.subject_id;has_column=bool(row.o2_column_input_available)
    dest=args.output_dir/'subjects'/s;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').is_file():print('Already complete:',s);return
    paths=prepare_subject(args,s,has_column)
    writers={};summaries=[];coverage=[];bins=[];r_maps={};pooled={k:[] for k in COMPARISONS}
    try:
        for h in ('lh','rh'):
            cortex=original.load_cortex_mask(args.fsaverage_dir.parent,'fsaverage',h)
            o3=original.load_method_profiles(args.results_root,'O3',s,h,'fsaverage',len(cortex))
            fs=fs_profiles(Path(paths[f'fs_{h}']),s,h)
            column=original.load_npz_profiles(Path(paths[f'column_{h}']),len(cortex)) if has_column else None
            with np.load(paths[f'curvature_{h}']) as data:
                curvature=original.align_scalar_values(data['target_vertex_id'],data['absolute_mean_curvature'],data['valid_mask'],len(cortex))
            common=cortex & o3.declared_valid & o3.finite_profile & fs.declared_valid & fs.finite_profile
            if column is not None:common &= column.declared_valid & column.finite_profile
            else:common[:]=False
            for slug,method in COMPARISONS.items():
                other=column if method=='O2-Column' else fs
                if other is None:continue
                pair=original.build_pair_data(other,o3,cortex)
                metrics=original.calculate_vertex_metrics(pair,curvature,common)
                if not pair.pairwise_valid.any():raise ValueError(f'{s} {h} {slug}: no valid paired vertices')
                for field,value in [('hemisphere',h),('subject_id',s),('comparison',slug)]:metrics.insert(0,field,value)
                table=pa.Table.from_pandas(metrics,preserve_index=False)
                if slug not in writers:writers[slug]=pq.ParquetWriter(dest/f'{slug}_vertex_metrics.parquet',table.schema)
                writers[slug].write_table(table)
                cov=original.coverage_row(s,h,slug,pair,common)
                cov['n_undefined_profile_correlation']=int((metrics.valid & metrics.profile_r.isna()).sum())
                coverage.append(cov);r_maps[f'{slug}_{h}']=metrics.profile_r.to_numpy(dtype=np.float64)
                pooled[slug].append(metrics.loc[metrics.valid,['profile_r','profile_mae','profile_snmae_pct','absolute_mean_curvature']])
        for slug,parts in pooled.items():
            if not parts:continue
            if len(parts)!=2:raise ValueError('Incomplete bilateral comparison')
            full=pd.concat(parts,ignore_index=True)
            for metric in ['profile_r','profile_mae','profile_snmae_pct']:
                summaries.append(dict(subject_id=s,comparison=slug,metric=metric,
                    subject_median=float(full[metric].median()),n_valid_vertices=int(full[metric].notna().sum())))
            valid=full.loc[np.isfinite(full.absolute_mean_curvature)&np.isfinite(full.profile_mae)]
            edges=original.assign_curvature_bins(valid.absolute_mean_curvature.to_numpy(),10)
            groups=original.bin_index(valid.absolute_mean_curvature.to_numpy(),edges)
            for b in np.unique(groups):
                p=valid.loc[groups==b]
                bins.append(dict(subject_id=s,comparison=slug,curvature_bin=int(b),
                    median_profile_mae=float(p.profile_mae.median()),
                    median_absolute_mean_curvature=float(p.absolute_mean_curvature.median()),n_vertices=len(p)))
            rho=float(spearmanr(valid.absolute_mean_curvature,valid.profile_mae).statistic)
            summaries.append(dict(subject_id=s,comparison=slug,metric='curvature_mae_spearman',
                subject_median=rho,n_valid_vertices=len(valid)))
        pd.DataFrame(summaries).to_csv(dest/'subject_metrics.csv',index=False)
        pd.DataFrame(coverage).to_csv(dest/'coverage.csv',index=False)
        pd.DataFrame(bins).to_csv(dest/'curvature_bins.csv',index=False)
        np.savez_compressed(dest/'correlation_maps.npz',**r_maps)
        dump(dest/'complete.json',dict(subject_id=s,status='complete',o2_column_available=has_column,
            depth_direction='pial_to_white',source_paths=paths,
            methods='Unchanged Figure 2 loading, cortex mask, pairwise validity and vertex metrics'))
    finally:
        for writer in writers.values():writer.close()


def aggregate(args):
    subjects=pd.read_csv(args.output_dir/'subjects.csv')
    ids=subjects.subject_id.tolist()
    for s in ids:
        if not (args.output_dir/'subjects'/s/'complete.json').is_file():raise ValueError(f'Incomplete subject task: {s}')
    all_metrics=pd.concat([pd.read_csv(args.output_dir/'subjects'/s/'subject_metrics.csv') for s in ids],ignore_index=True)
    all_bins=pd.concat([pd.read_csv(args.output_dir/'subjects'/s/'curvature_bins.csv') for s in ids],ignore_index=True)
    coverage=pd.concat([pd.read_csv(args.output_dir/'subjects'/s/'coverage.csv') for s in ids],ignore_index=True)
    common=set(ids)
    for slug in COMPARISONS:
        common &= set(all_metrics.loc[all_metrics.comparison.eq(slug),'subject_id'])
    for label,cohort in [('max_available',set(ids)),('common_cohort',common)]:
        out=args.output_dir/label;out.mkdir(exist_ok=True)
        metrics=all_metrics.loc[all_metrics.subject_id.isin(cohort)].copy()
        bins=all_bins.loc[all_bins.subject_id.isin(cohort)].copy()
        cov=coverage.loc[coverage.subject_id.isin(cohort)].copy()
        metrics.to_csv(out/'subject_metrics.csv',index=False);bins.to_csv(out/'curvature_subject_bins.csv',index=False)
        cov.to_csv(out/'coverage.csv',index=False)
        rng=np.random.default_rng(20261103);rows=[]
        for (slug,metric),p in metrics.groupby(['comparison','metric'],sort=False):
            values=p.subject_median.to_numpy();values=values[np.isfinite(values)]
            boots=np.median(values[rng.integers(0,len(values),(10000,len(values)))],axis=1)
            lo,hi=np.percentile(boots,[2.5,97.5])
            rows.append(dict(comparison=slug,metric=metric,n_subjects=len(values),median=float(np.median(values)),
                q25=float(np.percentile(values,25)),q75=float(np.percentile(values,75)),ci95_lower=float(lo),ci95_upper=float(hi)))
            if metric=='curvature_mae_spearman':
                test=original.wilcoxon(values,alternative='two-sided',method='auto')
                rows[-1].update(subject_wilcoxon_statistic=float(test.statistic),subject_wilcoxon_pvalue=float(test.pvalue))
        pd.DataFrame(rows).to_csv(out/'subject_summary.csv',index=False)
        bins.groupby(['comparison','curvature_bin']).agg(n_subjects=('subject_id','nunique'),
            median_mae_across_subjects=('median_profile_mae','median'),
            q25_mae_across_subjects=('median_profile_mae',lambda x:x.quantile(.25)),
            q75_mae_across_subjects=('median_profile_mae',lambda x:x.quantile(.75)),
            median_curvature_across_subjects=('median_absolute_mean_curvature','median')).reset_index().to_csv(out/'curvature_summary.csv',index=False)
        pd.DataFrame({'subject_id':[s for s in ids if s in cohort]}).to_csv(out/'subjects.csv',index=False)
        for slug in COMPARISONS:
            chosen=[s for s in ids if s in cohort and s in set(metrics.loc[metrics.comparison.eq(slug),'subject_id'])]
            matrices={h:np.empty((len(chosen),163842),dtype=np.float64) for h in ('lh','rh')}
            for i,s in enumerate(chosen):
                with np.load(args.output_dir/'subjects'/s/'correlation_maps.npz') as data:
                    for h in ('lh','rh'):matrices[h][i]=data[f'{slug}_{h}']
            for h,frame in original.make_cross_subject_summary(matrices).items():
                frame.to_csv(out/f'{slug}_agreement_map_{h}.csv',index=False)
    dump(args.output_dir/'analysis_complete.json',dict(status='complete',n_qc_passed_subjects=len(ids),
        n_common_subjects=len(common),comparison_subject_counts=all_metrics.groupby('comparison').subject_id.nunique().to_dict(),
        bootstrap_repetitions=10000,bootstrap_unit='subject',bootstrap_seed=20261103,
        validity='Existing Figure 2 pairwise masks preserved; subject is retained only with bilateral results',
        max_available_vs_common='Main matched comparison uses common_cohort; maximal per-pair results are also saved',
        missing_O2Column_subjects=subjects.loc[~subjects.o2_column_input_available,'subject_id'].tolist()))
    from plot_full_cohort import plot_all
    plot_all(args.output_dir,args.fsaverage_dir)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['inventory','subject','aggregate'])
    for name in ['results-root','subjects-dir','fsaverage-dir','output-dir','qc-exclusions']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--native-mapping-root',type=Path)
    p.add_argument('--freesurfer-home',type=Path)
    p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.native_mapping_root is None:a.native_mapping_root=a.results_root/'archive/legacy_pipeline/native_column_fsaverage'
    if a.freesurfer_home is None:a.freesurfer_home=a.fsaverage_dir.parents[1]
    {'inventory':inventory,'subject':subject_stage,'aggregate':aggregate}[a.stage](a)
