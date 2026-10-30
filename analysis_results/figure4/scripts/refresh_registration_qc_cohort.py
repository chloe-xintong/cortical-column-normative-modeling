#!/usr/bin/env python3
"""Reaggregate cached Figure 4 results after fixed manual registration exclusions.
Preserves the legacy 325-subject frozen outputs. No subject-level recomputation.
"""
from pathlib import Path
import subprocess,sys,json,hashlib
import pandas as pd
import numpy as np
REPO=Path(__file__).resolve().parents[3]
SOURCE=Path('/mnt/newStor/paros/paros_WORK/xintong/results/qc/full_cohort_spatial_scale/figure4_ico5_formal_k15')
OUT=REPO/'analysis_results/figure4/outputs/registration_qc_pass_320'
MANIFEST=REPO/'manifests/figure4_registration_qc_pass_320_subjects.csv'
SCRIPTS=REPO/'analysis_results/figure4/scripts'
sys.path.insert(0,str(SCRIPTS))
from validate_acquisition_sampling_null import summarize_at_n,summarize_local_excess_auc

def run(name,*args):
 print('Running',name,flush=True)
 subprocess.run([sys.executable,str(SCRIPTS/name),*map(str,args)],check=True)
def main():
 ids=pd.read_csv(MANIFEST).subject_id.tolist();assert len(ids)==len(set(ids))==320
 OUT.mkdir(parents=True,exist_ok=True)
 for script,part in [('summarize_ico5_formal_real.py','real'),('summarize_ico5_formal_c.py','correlation_length')]:
  run(script,'--manifest',MANIFEST,'--results-root',SOURCE/part,'--output-dir',OUT/part/'cohort')
 run('summarize_relative_mae.py','--manifest',MANIFEST,'--results-root',SOURCE/'relative_mae','--output-dir',OUT/'relative_mae/cohort','--repo-root',REPO)
 run('plot_ico5_formal_a_with_relative_mae.py','--real-summary',OUT/'real/cohort/figure4A_ico5_k15_real_group_summary.csv','--relative-summary',OUT/'relative_mae/cohort/figure4_relative_mae_group_summary.csv','--output-dir',OUT/'real/cohort','--repo-root',REPO)
 run('summarize_boundary_matched_effect.py','--manifest',MANIFEST,'--pair-root',SOURCE/'boundary_matched/formal/pairs','--effect-root',SOURCE/'boundary_matched/formal/effects','--output-dir',OUT/'boundary_matched/formal/cohort','--repo-root',REPO)
 print('Reaggregating cached null realizations',flush=True)
 src=SOURCE/'acquisition_null_fwhm0_seedfix/cohort';dst=OUT/'acquisition_null_fwhm0_seedfix/cohort';dst.mkdir(parents=True,exist_ok=True)
 curves=pd.read_csv(src/'figure4B_acquisition_subject_realization_bins.csv');curves=curves[curves.subject_id.isin(ids)].copy();assert set(curves.subject_id)==set(ids)
 assert curves.groupby('subject_id').replicate.nunique().eq(100).all()
 curves.to_csv(dst/'figure4B_acquisition_subject_realization_bins.csv',index=False)
 groups=[]
 for n in [10,25,50,100]:
  sub,group=summarize_at_n(curves,n);groups.append(group)
 pd.concat(groups).to_csv(dst/'figure4B_acquisition_group_summary.csv',index=False)
 sub.to_csv(dst/'figure4B_acquisition_subject_summary.csv',index=False)
 a,b=summarize_local_excess_auc(sub);a.to_csv(dst/'figure4B_local_excess_auc_subject.csv',index=False);b.to_csv(dst/'figure4B_local_excess_auc_group_summary.csv',index=False)
 # Reuse unchanged subject-level validation, restricted to retained subjects.
 validation=pd.read_csv(src/'figure4B_acquisition_validation_by_subject.csv');validation=validation[validation.subject_id.isin(ids)];validation.to_csv(dst/'figure4B_acquisition_validation_by_subject.csv',index=False)
 far=pd.read_csv(src/'figure4B_far_window_sensitivity_auc_subject.csv');far=far[far.subject_id.isin(ids)];far.to_csv(dst/'figure4B_far_window_sensitivity_auc_subject.csv',index=False)
 v=json.loads((src/'figure4B_acquisition_validation_summary.json').read_text());v['subjects_expected']=320;v['subjects_completed_per_condition']={'0.0':320};v['all_expected_subjects_complete']=True;v['status']='retained_cohort_reaggregated';v['validation_note']='Retained cached subject-level validation; historical global seed and max-error diagnostics refer to source QC.'
 (dst/'figure4B_acquisition_validation_summary.json').write_text(json.dumps(v,indent=2))
 run('plot_ico5_formal_b.py','--cohort-dir',dst)
 old=pd.read_csv(REPO/'manifests/figure4_valid_325_subjects.csv');excluded=old[~old.subject_id.isin(ids)].subject_id.tolist()
 # Exact participant-set checks across the primary A/B/C/D cached tables.
 paths=[OUT/'real/cohort/figure4A_ico5_k15_real_subject_bins.csv',dst/'figure4B_acquisition_subject_summary.csv',OUT/'correlation_length/cohort/figure4C_correlation_length_fits.csv',OUT/'boundary_matched/formal/cohort/figure4D_subject_bin_effects.csv']
 for p in paths:assert set(pd.read_csv(p).subject_id)==set(ids),p
 report={'source_n':len(old),'retained_n':len(ids),'excluded_subjects':excluded,'exclusion_source':str(REPO/'analysis_results/figure5/cohort_selection/manual_registration_qc_exclusions.csv'),'manifest':str(MANIFEST),'source_results':str(SOURCE),'output_results':str(OUT),'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),'method':'Original subject-level results reused; group medians/IQR, bootstrap CIs, and split-half reliability recalculated. No sampling, null realizations, or fits rerun.','participant_set_checks':'pass: A/B/C/D all exactly match the 320-subject manifest.'}
 (OUT/'cohort_refresh_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
