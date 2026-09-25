#!/usr/bin/env python3
"""Pilot: compare fixed-depth native spatial averaging scales."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.model_selection import KFold
from ..common.config import SUBJECTS_DIR, PROFILES_ROOT, FSAVERAGE6_MAPPING_ROOT
from ..common.io import mapping_files, mapping_frame, read_surface

METHODS = ("region", "spatial", "random")
KS = (1, 2, 4, 8, 16, 32)

def valid_bool(x):
    return x.astype(str).str.lower().isin(("true", "1", "1.0"))

def edges_to_graph(vertices, faces, roi_ids):
    roi = set(int(x) for x in roi_ids); pairs=set()
    for tri in np.asarray(faces, dtype=np.int64):
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            if int(a) in roi and int(b) in roi and a != b: pairs.add((min(a,b),max(a,b)))
    ids=np.array(sorted(roi),dtype=int); pos={v:i for i,v in enumerate(ids)}
    rr=[]; cc=[]; dd=[]
    for a,b in pairs:
        w=float(np.linalg.norm(vertices[a]-vertices[b])); i,j=pos[a],pos[b]
        rr += [i,j]; cc += [j,i]; dd += [w,w]
    return ids, csr_matrix((dd,(rr,cc)),shape=(len(ids),len(ids)))

def spatial_groups(vertices, faces, ids, k):
    native_ids, graph = edges_to_graph(vertices, faces, ids)
    if len(native_ids) < k or graph.shape[0] == 0: return None, {"status":"invalid_too_few_or_no_edges"}
    ncomp, labels = connected_components(graph, directed=False)
    if ncomp > 1: return None, {"status":"invalid_disconnected_roi", "components":int(ncomp)}
    dist=dijkstra(graph, directed=False, indices=0); seeds=[int(np.argmax(dist))]
    all_dist=np.full(graph.shape[0], np.inf); all_dist[:]=dist
    for _ in range(1,k):
        nearest=np.min(np.vstack([dijkstra(graph,directed=False,indices=s) for s in seeds]),axis=0)
        seeds.append(int(np.argmax(nearest)))
    D=np.vstack([dijkstra(graph,directed=False,indices=s) for s in seeds])
    group=np.argmin(D,axis=0)
    return dict(zip(native_ids,group)), {"status":"valid","components":1,"group_sizes":np.bincount(group,minlength=k).tolist()}

def group_connectedness(vertices, faces, ids, groups):
    native_ids, graph=edges_to_graph(vertices,faces,ids); out=[]
    if graph.shape[0]==0: return out
    for g in sorted(set(groups.values())):
        keep=np.array([groups.get(int(v))==g for v in native_ids]); sub=graph[keep][:,keep]
        n=connected_components(sub,directed=False,return_labels=False) if sub.shape[0] else 0
        out.append((int(g),int(n),int(keep.sum())))
    return out

def metric(y,p):
    e=y-p
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e*e)))

def save_overlay(out, subject, hemi, region, name, nverts, ids, vals):
    a=np.full(nverts,np.nan,dtype=np.float32); a[np.asarray(ids,dtype=int)]=np.asarray(vals,dtype=np.float32)
    path=out/"overlays"/f"{subject}_{hemi}_{region}_{name}.mgh"
    nib.save(nib.freesurfer.mghformat.MGHImage(a[:,None,None],np.eye(4)),str(path))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--subjects',nargs='+',default=['D0231','D0330','D0097','D0388','D0163']); ap.add_argument('--regions',nargs='+',default=['precentral','superiorfrontal','supramarginal','superiortemporal']); ap.add_argument('--hemispheres',nargs='+',default=['lh','rh']); ap.add_argument('--output-dir',type=Path,default=Path('/mnt/newStor/paros/paros_WORK/xintong/results/native_surface_analysis/spatial_averaging_pilot')); ap.add_argument('--seed',type=int,default=20260924); ap.add_argument('--folds',type=int,default=5); ap.add_argument('--random-repeats',type=int,default=20); ap.add_argument('--min-train-per-group',type=int,default=10); args=ap.parse_args()
    for d in ('tables','overlays','figures','logs'): (args.output_dir/d).mkdir(parents=True,exist_ok=True)
    rows=[]; pred_rows=[]; groups_rows=[]; rng=np.random.default_rng(args.seed); total=len(args.subjects)*len(args.hemispheres)*len(args.regions); done=0
    for subject in args.subjects:
      for hemi in args.hemispheres:
       verts,faces=__import__('nibabel').freesurfer.io.read_geometry(str(SUBJECTS_DIR/subject/'surf'/f'{hemi}.white'))
       for region in args.regions:
        done+=1; print(f'[{done}/{total}] {subject} {hemi} {region}',flush=True)
        paths=[p for p in mapping_files(FSAVERAGE6_MAPPING_ROOT,subject,hemi) if p.name.endswith(f'_{region}_native_to_fsaverage.csv.gz')]
        if len(paths)!=1: print('  skipped: mapping file count',len(paths),flush=True); continue
        f=mapping_frame(paths[0]); f=f[valid_bool(f['column_valid'])].copy(); md=pd.to_numeric(f['MD_depth_10'],errors='coerce'); f=f.assign(md50=md).dropna(subset=['md50']); f=f.drop_duplicates('native_vertex_id')
        if len(f)<args.folds: print('  skipped: insufficient valid columns',len(f),flush=True); continue
        ids=f.native_vertex_id.to_numpy(int); y=f.md50.to_numpy(float); folds=np.full(len(f),-1); folds[KFold(args.folds,shuffle=True,random_state=args.seed).split(f).__next__()[0]]=0
        for fold,(tr,te) in enumerate(KFold(args.folds,shuffle=True,random_state=args.seed).split(f)): folds[te]=fold
        full_region=float(np.mean(y)); nverts=len(verts); groups_by_k={}
        for k in KS:
          if k==1: groups_by_k[k]={int(v):0 for v in ids}; diag={'status':'valid','components':1,'group_sizes':[len(ids)]}
          else: groups_by_k[k],diag=spatial_groups(verts,faces,ids,k)
          if groups_by_k[k] is not None:
            for g,nc,ns in group_connectedness(verts,faces,ids,groups_by_k[k]): groups_rows.append(dict(subject=subject,hemisphere=hemi,region=region,method='spatial',k=k,group=g,n_components=nc,n_columns=ns))
          save_overlay(args.output_dir,subject,hemi,region,f'raw_md50',nverts,ids,y)
          for method in METHODS:
            if method in ('spatial','random') and groups_by_k[k] is None:
              rows.append(dict(subject=subject,hemisphere=hemi,region=region,method=method,k=k,repeat=0,mae=np.nan,rmse=np.nan,baseline_mae=np.nan,mae_improvement_pct=np.nan,n_evaluated=0,status='invalid_spatial_partition'))
              continue
            reps=args.random_repeats if method=='random' and k>1 else 1
            for rep in range(reps):
              if method=='region': labels=np.zeros(len(f),int)
              elif method=='spatial':
                if groups_by_k[k] is None: continue
                labels=np.array([groups_by_k[k][int(v)] for v in ids])
              else:
                labels=np.repeat(np.arange(k), np.bincount(np.array([groups_by_k[k][int(v)] for v in ids]),minlength=k)); rng_rep=np.random.default_rng(args.seed+rep+k*1000); rng_rep.shuffle(labels)
              pred=np.full(len(f),np.nan); ok=True
              for fold in range(args.folds):
                tr=folds!=fold; te=~tr
                for g in range(k):
                  sel=tr & (labels==g)
                  if sel.sum()<args.min_train_per_group: ok=False
                if not ok: break
                for g in range(k): pred[te & (labels==g)]=np.mean(y[tr & (labels==g)])
              if not ok or not np.isfinite(pred).all(): continue
              mae,rmse=metric(y,pred); base=np.mean(np.abs(y-full_region)); rows.append(dict(subject=subject,hemisphere=hemi,region=region,method=method,k=k,repeat=rep,mae=mae,rmse=rmse,baseline_mae=base,mae_improvement_pct=np.nan if base==0 else 100*(base-mae)/base,n_evaluated=len(y),status='valid'))
              if rep==0: save_overlay(args.output_dir,subject,hemi,region,f'{method}_k{k}_full_mean',nverts,ids,np.array([np.mean(y[labels==g]) for g in labels])); save_overlay(args.output_dir,subject,hemi,region,f'{method}_k{k}_oof_prediction',nverts,ids,pred); save_overlay(args.output_dir,subject,hemi,region,f'{method}_k{k}_oof_residual',nverts,ids,y-pred)
              if rep==0: pred_rows.extend(dict(subject=subject,hemisphere=hemi,region=region,column_index=int(i),native_vertex_id=int(ids[i]),md50=float(y[i]),method=method,k=k,repeat=rep,fold=int(folds[i]),prediction=float(pred[i]),residual=float(y[i]-pred[i])) for i in range(len(y)))
    pd.DataFrame(rows).to_csv(args.output_dir/'tables/metrics_summary.csv',index=False); pd.DataFrame(pred_rows).to_csv(args.output_dir/'tables/column_predictions.csv',index=False); pd.DataFrame(groups_rows).to_csv(args.output_dir/'tables/spatial_group_connectivity.csv',index=False); json.dump(vars(args),open(args.output_dir/'tables/run_config.json','w'),indent=2,default=str); print(f'Wrote {len(rows):,} metric rows and {len(pred_rows):,} prediction rows',flush=True)
if __name__=='__main__': main()
