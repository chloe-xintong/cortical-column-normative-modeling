#!/usr/bin/env python3
"""Write the active cached-results notebook for the registration-QC-passing cohort."""
from pathlib import Path
import json,base64,shutil
import pandas as pd
REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'analysis_results/figure4/outputs/registration_qc_pass_320'
NOTEBOOK=REPO/'analysis_results/figure4/notebooks/figure4_formal_ico5_k15.ipynb'
LEGACY=NOTEBOOK.with_name('figure4_formal_ico5_k15_legacy_325.ipynb')
if not LEGACY.exists():shutil.copy2(NOTEBOOK,LEGACY)
cells=[]
def md(t):cells.append({'cell_type':'markdown','metadata':{},'source':t.splitlines(True)})
def code(t,outputs):cells.append({'cell_type':'code','execution_count':sum(c['cell_type']=='code' for c in cells)+1,'metadata':{},'source':t.splitlines(True),'outputs':outputs})
def out_table(p):return {'output_type':'execute_result','execution_count':sum(c['cell_type']=='code' for c in cells)+1,'metadata':{},'data':{'text/plain':[pd.read_csv(p).to_string(index=False)]}}
def panel(title,description,part,tables,figure):
 md('## '+title+'\n\n'+description)
 src=f"PART = ROOT / {part!r}\n"
 outs=[]
 for table in tables:src+=f"display(pd.read_csv(PART / {table!r}))\n";outs.append(out_table(ROOT/part/table))
 src+=f"display(Image(filename=str(PART / {figure!r})))\n"
 outs.append({'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode((ROOT/part/figure).read_bytes()).decode(),'text/plain':['<IPython.core.display.Image object>']}})
 code(src,outs)
md('# Figure 4A–D: formal ico5 K=15, registration-QC-passing cohort (n=320)\n\nThe original 325-subject frozen results are retained separately. Five visually confirmed T1–DWI registration failures (D0128, D0163, D0248, D0296, D0388) are removed. This notebook reads newly reaggregated cached subject-level results. Original sampling, subject fits and null realizations are unchanged; cohort summaries, bootstrap confidence intervals and reliability are recalculated.\n\nActive manifest: `manifests/figure4_registration_qc_pass_320_subjects.csv`. Figure 5 remains the same 196 CN participants.')
code(f"from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path({str(ROOT)!r})\nMANIFEST = Path({str(REPO/'manifests/figure4_registration_qc_pass_320_subjects.csv')!r})\nsubjects = pd.read_csv(MANIFEST).subject_id\nassert subjects.nunique() == len(subjects) == 320\nprint(f'Active Figure 4 cohort: {{len(subjects)}} participants')\n",[{'output_type':'stream','name':'stdout','text':['Active Figure 4 cohort: 320 participants\n']}])
panel('Figure 4A: profile-shape dissimilarity and relative MAE','Official ico5 anchors; within-DK candidates; at most 15 pairs per anchor × 2-mm bin. Pair → anchor-bin median → bilateral subject median → cohort median/IQR. Distances use fsaverage white-surface mesh-edge geodesics. 21 depths, pial → white.','real/cohort',['figure4A_ico5_k15_real_group_summary.csv'],'Figure4A_ico5_K15_with_relative_MAE.png')
panel('Figure 4B: seed-fixed acquisition-and-sampling null','FWHM=0 mm; 100 existing independent realizations per retained participant. Local excess uses the 20–30-mm reference. Residual AUC is not a pure physiological estimate.','acquisition_null_fwhm0_seedfix/cohort',['figure4B_local_excess_auc_group_summary.csv'],'Figure4B_ico5_K15_formal_main.png')
panel('Figure 4C: correlation length and fixed anchor split-half stability','D(d)=D∞−A exp(−d/λ). No refitting is required after subject exclusions. Group distribution and split-half statistics are recomputed on retained participants. This is not scan-rescan reliability.','correlation_length/cohort',['figure4C_full_fit_summary.csv','figure4C_fixed_split_reliability.csv','figure4C_fit_qc_summary.csv'],'Figure4C_ico5_K15_correlation_length.png')
panel('Figure 4D: matched within/across-DK boundary effect','Same-anchor one-to-one distance matching, caliper 0.1 mm. Effect ΔD=Dacross−Dwithin. Matched-pair mean → anchor-bin mean → bilateral subject-region-bin mean → equal-region subject-bin mean. This design does not support a consistent positive DK-boundary shape effect.','boundary_matched/formal/cohort',['figure4D_0_10_group_summary.csv','figure4D_group_summary.csv'],'Figure4D_boundary_matched_0_10.png')
md('## Provenance and scope\n\nReaggregation script: `analysis_results/figure4/scripts/refresh_registration_qc_cohort.py`. Cohort audit: `analysis_results/figure4/outputs/registration_qc_pass_320/cohort_refresh_report.json`. No historical 325-subject result files are overwritten. Supplemental and region-level notebooks retain their historical cohorts until explicitly refreshed; they should not be mixed with the active 320-subject primary analysis.')
# Preserve the user-requested additional annotated Figure 4A cells on refresh.
if NOTEBOOK.exists():
    existing = json.loads(NOTEBOOK.read_text())
    annotations = [c for c in existing['cells']
                   if 'figure4_sampling_scale_annotations' in c.get('metadata', {}).get('tags', [])]
    cells[4:4] = annotations
NOTEBOOK.write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':4},indent=1))
print(NOTEBOOK)
