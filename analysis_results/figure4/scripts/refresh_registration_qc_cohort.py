#!/usr/bin/env python3
"""Reaggregate cached Figure 4 results for a supplied primary-cohort manifest.

Description
-----------
Reaggregate cached Figure 4 results for a supplied primary-cohort manifest.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--output-dir`` command-line input
- ``--source-results`` command-line input
- ``--source-cohort`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/refresh_registration_qc_cohort.py --help
"""
from pathlib import Path
import argparse,subprocess,sys,json,hashlib
import pandas as pd
import numpy as np
REPO=Path(__file__).resolve().parents[3]
SOURCE=Path('/mnt/newStor/paros/paros_WORK/xintong/results/qc/full_cohort_spatial_scale/figure4_ico5_formal_k15')
SCRIPTS=REPO/'analysis_results/figure4/scripts'
sys.path.insert(0,str(SCRIPTS))
from validate_acquisition_sampling_null import summarize_at_n,summarize_local_excess_auc

def run(name,*args):
 print('Running',name,flush=True)
 subprocess.run([sys.executable,str(SCRIPTS/name),*map(str,args)],check=True)

def parse_args():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--manifest',type=Path,default=REPO/'manifests/figure4_coverage95_pass_313_subjects.csv')
 parser.add_argument('--output-dir',type=Path,default=REPO/'/mnt/newStor/paros/paros_WORK/xintong/result/cortical_trajectory_analysis/analysis_results/figure4/outputs/coverage95_pass_313')
 parser.add_argument('--source-results',type=Path,default=SOURCE)
 parser.add_argument('--source-cohort',type=Path,default=REPO/'manifests/figure4_registration_qc_pass_320_subjects.csv')
 return parser.parse_args()

def main():
 args=parse_args();manifest=args.manifest;out=args.output_dir;source=args.source_results
 ids=pd.read_csv(manifest,dtype={'subject_id':str}).subject_id.tolist();n=len(ids)
 if n<4 or len(set(ids))!=n:raise ValueError('Manifest must contain at least four unique subjects')
 out.mkdir(parents=True,exist_ok=True)
 for script,part in [('summarize_ico5_formal_real.py','real'),('summarize_ico5_formal_c.py','correlation_length')]:
  run(script,'--manifest',manifest,'--results-root',source/part,'--output-dir',out/part/'cohort')
 run('summarize_relative_mae.py','--manifest',manifest,'--results-root',source/'relative_mae','--output-dir',out/'relative_mae/cohort','--repo-root',REPO)
 run('plot_ico5_formal_a_with_relative_mae.py','--real-summary',out/'real/cohort/figure4A_ico5_k15_real_group_summary.csv','--relative-summary',out/'relative_mae/cohort/figure4_relative_mae_group_summary.csv','--output-dir',out/'real/cohort','--repo-root',REPO)
 run('summarize_boundary_matched_effect.py','--manifest',manifest,'--pair-root',source/'boundary_matched/formal/pairs','--effect-root',source/'boundary_matched/formal/effects','--output-dir',out/'boundary_matched/formal/cohort','--repo-root',REPO)
 print('Reaggregating cached null realizations',flush=True)
 src=source/'acquisition_null_fwhm0_seedfix/cohort';dst=out/'acquisition_null_fwhm0_seedfix/cohort';dst.mkdir(parents=True,exist_ok=True)
 curves=pd.read_csv(src/'figure4B_acquisition_subject_realization_bins.csv');curves=curves[curves.subject_id.isin(ids)].copy();assert set(curves.subject_id)==set(ids)
 assert curves.groupby('subject_id').replicate.nunique().eq(100).all()
 curves.to_csv(dst/'figure4B_acquisition_subject_realization_bins.csv',index=False)
 groups=[]
 for realization_count in [10,25,50,100]:
  sub,group=summarize_at_n(curves,realization_count);groups.append(group)
 pd.concat(groups).to_csv(dst/'figure4B_acquisition_group_summary.csv',index=False)
 sub.to_csv(dst/'figure4B_acquisition_subject_summary.csv',index=False)
 a,b=summarize_local_excess_auc(sub);a.to_csv(dst/'figure4B_local_excess_auc_subject.csv',index=False);b.to_csv(dst/'figure4B_local_excess_auc_group_summary.csv',index=False)
 # Reuse unchanged subject-level validation, restricted to retained subjects.
 validation=pd.read_csv(src/'figure4B_acquisition_validation_by_subject.csv');validation=validation[validation.subject_id.isin(ids)];validation.to_csv(dst/'figure4B_acquisition_validation_by_subject.csv',index=False)
 far=pd.read_csv(src/'figure4B_far_window_sensitivity_auc_subject.csv');far=far[far.subject_id.isin(ids)];far.to_csv(dst/'figure4B_far_window_sensitivity_auc_subject.csv',index=False)
 v=json.loads((src/'figure4B_acquisition_validation_summary.json').read_text());v['subjects_expected']=n;v['subjects_completed_per_condition']={'0.0':n};v['all_expected_subjects_complete']=True;v['status']='retained_cohort_reaggregated';v['validation_note']='Retained cached subject-level validation; historical global seed and max-error diagnostics refer to source QC.'
 (dst/'figure4B_acquisition_validation_summary.json').write_text(json.dumps(v,indent=2))
 run('plot_ico5_formal_b.py','--cohort-dir',dst)
 old=pd.read_csv(args.source_cohort,dtype={'subject_id':str});excluded=old[~old.subject_id.isin(ids)].subject_id.tolist()
 # Exact participant-set checks across the primary A/B/C/D cached tables.
 paths=[out/'real/cohort/figure4A_ico5_k15_real_subject_bins.csv',dst/'figure4B_acquisition_subject_summary.csv',out/'correlation_length/cohort/figure4C_correlation_length_fits.csv',out/'boundary_matched/formal/cohort/figure4D_subject_bin_effects.csv']
 for p in paths:assert set(pd.read_csv(p).subject_id)==set(ids),p
 report={'source_n':len(old),'retained_n':n,'excluded_subjects':excluded,'exclusion_rule':'subject-level bilateral cleaned-profile coverage >= 0.95 after registration QC','manifest':str(manifest),'source_results':str(source),'output_results':str(out),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'method':'Original subject-level results reused; group medians/IQR, bootstrap CIs, and split-half reliability recalculated. No sampling, null realizations, or fits rerun.','participant_set_checks':f'pass: A/B/C/D all exactly match the {n}-subject manifest.'}
 (out/'cohort_refresh_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
