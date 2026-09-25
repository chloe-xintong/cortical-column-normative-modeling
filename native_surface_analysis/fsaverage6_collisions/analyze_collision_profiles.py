#!/usr/bin/env python3
"""Collision-group MD profile, native-space and label-mixing analysis."""
from __future__ import annotations
import argparse, json, itertools
from pathlib import Path
import numpy as np, pandas as pd
from nibabel.freesurfer.io import read_annot
from ..common.config import FSAVERAGE6_MAPPING_ROOT, FREESURFER_SUBJECTS_DIR
from ..common.io import mapping_files, mapping_frame

def valid_bool(s): return s.astype(str).str.lower().isin(("true","1","1.0"))
def corr(a,b):
    if np.std(a)==0 or np.std(b)==0: return np.nan
    return float(np.corrcoef(a,b)[0,1])
def one_subject(subject, out, hemis):
    groups=[]; status=[]; examples=[]
    for hemi in hemis:
        try:
            lab,_,names=read_annot(str(FREESURFER_SUBJECTS_DIR/'fsaverage6'/'label'/f'{hemi}.aparc.annot'))
            target_labels=np.asarray(lab); decoded=[x.decode() if isinstance(x,bytes) else str(x) for x in names]
        except Exception as e:
            status.append(dict(subject_id=subject,hemisphere=hemi,status='failed',reason=f'target annotation: {e}')); continue
        paths=mapping_files(FSAVERAGE6_MAPPING_ROOT,subject,hemi); frames=[]
        for n,path in enumerate(paths,1):
            region=path.name.split(f'{subject}_{hemi}_',1)[1].removesuffix('_native_to_fsaverage.csv.gz')
            try:
                f=mapping_frame(path); frames.append(f)
            except Exception as e: status.append(dict(subject_id=subject,hemisphere=hemi,region=region,status='failed',reason=str(e))); continue
            if n%100==0: print(f'{subject} {hemi}: read {n}/{len(paths)} mapping files',flush=True)
        mapped=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
        if len(mapped):
            mapped=mapped[mapped['template_vertex_id'].notna()].copy(); mapped['template_vertex_id']=mapped.template_vertex_id.astype(int)
        for vertex,m in mapped.groupby('template_vertex_id',sort=False):
                all_regions=m.region.astype(str); counts=all_regions.value_counts(); main=float(counts.iloc[0]/len(m))
                valid=m[valid_bool(m.column_valid)].copy(); md_cols=[f'MD_depth_{i}' for i in range(21)]
                valid['profile_valid']=valid[md_cols].apply(lambda x: np.isfinite(pd.to_numeric(x,errors='coerce').to_numpy()).all(),axis=1)
                vp=valid[valid.profile_valid].copy(); pairs=[]
                for i,j in itertools.combinations(range(len(vp)),2):
                    a=pd.to_numeric(vp.iloc[i][md_cols],errors='coerce').to_numpy(float); b=pd.to_numeric(vp.iloc[j][md_cols],errors='coerce').to_numpy(float); d=np.linalg.norm(pd.to_numeric(vp.iloc[i][['native_white_x','native_white_y','native_white_z']])-pd.to_numeric(vp.iloc[j][['native_white_x','native_white_y','native_white_z']]))
                    pairs.append((corr(a,b),float(np.mean(np.abs(a-b))),float(np.sqrt(np.mean((a-b)**2))),float(d)))
                tlabel=decoded[int(target_labels[int(vertex)])] if 0 <= int(vertex)<len(target_labels) and int(target_labels[int(vertex)])<len(decoded) else 'unknown'
                native_target_agreement=np.mean(all_regions.to_numpy()==tlabel) if len(m) else np.nan
                valid_agreement=np.mean(vp.region.astype(str).to_numpy()==tlabel) if len(vp) else np.nan
                rec=dict(subject_id=subject,hemisphere=hemi,source_region=region,target_vertex_id=int(vertex),n_columns_mapped=len(m),n_profiles_valid=len(vp),multiplicity=min(len(m),6),multiplicity_label=str(len(m) if len(m)<6 else '>=6'),native_region_count=counts.size,majority_native_region=counts.index[0],majority_region_fraction=main,mixed_label=counts.size>1,target_region=tlabel,template_label_agreement_pct=100*native_target_agreement,valid_profile_label_agreement_pct=100*valid_agreement,profile_pairs=len(pairs))
                for key,ix in [('profile_pearson_median',0),('profile_mae_median',1),('profile_rmse_median',2),('native_anchor_distance_median',3)]: rec[key]=float(np.nanmedian([p[ix] for p in pairs])) if pairs else np.nan
                rec['native_anchor_distance_max']=float(np.nanmax([p[3] for p in pairs])) if pairs else np.nan
                groups.append(rec)
                if len(examples)<20 and len(pairs)>=2: examples.append(rec)
        status.append(dict(subject_id=subject,hemisphere=hemi,status='success',reason='',mapping_files=len(paths)))
    subject_out=out/subject; (subject_out/'tables').mkdir(parents=True,exist_ok=True)
    pd.DataFrame(groups).to_csv(subject_out/'tables/collision_group_metrics.csv',index=False)
    pd.DataFrame(status).to_csv(subject_out/'tables/status.csv',index=False)
    pd.DataFrame(examples).to_csv(subject_out/'tables/example_groups.csv',index=False)
    print(f'{subject}: groups={len(groups):,}',flush=True)
    return groups,status

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--subjects',nargs='+',default=['D0097']); ap.add_argument('--hemispheres',nargs='+',default=['lh','rh']); ap.add_argument('--output-dir',type=Path,default=Path('/mnt/newStor/paros/paros_WORK/xintong/results/native_surface_analysis/collision_profile_analysis')); ap.add_argument('--seed',type=int,default=20260924); args=ap.parse_args(); (args.output_dir/'tables').mkdir(parents=True,exist_ok=True)
    allg=[]; alls=[]
    for i,s in enumerate(args.subjects,1): print(f'[{i}/{len(args.subjects)}] starting {s}',flush=True); g,st=one_subject(s,args.output_dir,args.hemispheres); allg.extend(g); alls.extend(st)
    g=pd.DataFrame(allg); s=pd.DataFrame(alls); g.to_csv(args.output_dir/'tables/collision_group_metrics.csv',index=False); s.to_csv(args.output_dir/'tables/status.csv',index=False)
    if len(g):
      covered=g.groupby(['subject_id','hemisphere','target_vertex_id'],as_index=False).size(); g['multiplicity_label']=g['n_columns_mapped'].map(lambda x:str(int(x)) if x<6 else '>=6')
      num=['profile_pearson_median','profile_mae_median','profile_rmse_median','native_anchor_distance_median','native_anchor_distance_max']
      subj=g.groupby('subject_id').agg(groups=('target_vertex_id','size'),covered_vertices=('target_vertex_id','nunique'),groups_with_2_profiles=('n_profiles_valid',lambda x:int((x>=2).sum())),mixed_label_groups=('mixed_label','sum'),**{x:(x,'median') for x in num}).reset_index(); subj.to_csv(args.output_dir/'tables/collision_subject_summary.csv',index=False)
      mult=g.groupby(['multiplicity_label'],sort=False).agg(covered_vertices=('target_vertex_id','size'),groups_with_2_profiles=('n_profiles_valid',lambda x:int((x>=2).sum())),mixed_label_fraction=('mixed_label','mean'),profile_pearson_median=('profile_pearson_median','median'),profile_mae_median=('profile_mae_median','median'),profile_rmse_median=('profile_rmse_median','median'),native_anchor_distance_median=('native_anchor_distance_median','median'),native_anchor_distance_max=('native_anchor_distance_max','median')).reset_index(); mult['covered_vertex_pct']=100*mult.covered_vertices/mult.covered_vertices.sum(); mult.to_csv(args.output_dir/'tables/collision_multiplicity_summary.csv',index=False)
    json.dump(vars(args),open(args.output_dir/'tables/run_config.json','w'),indent=2,default=str); print(f'Wrote {len(g):,} collision groups; output={args.output_dir}',flush=True)
if __name__=='__main__': main()
