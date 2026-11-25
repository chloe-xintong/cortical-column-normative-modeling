#!/usr/bin/env python3
"""Run one outer fold of the fixed ico4 spatial/depth graph-Ridge experiment."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
REPO=Path(__file__).resolve().parents[5]; SCRIPTS=REPO/'analysis_results/figure6/确定扰动因素/distance/scripts'; sys.path.insert(0,str(SCRIPTS))
from validate_graph_ridge_operator import preprocess_fit,transform,solve
AGE=REPO/'analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction'; OUT=AGE/'graph_ridge'
ALPHAS=np.logspace(-4,6,21); RHOS=(0,1,10); FAMILIES={'ordinary':[(0,0)],'spatial':[(0,0),(1,0),(10,0)],'depth':[(0,0),(0,1),(0,10)],'spatial_depth':[(s,d) for s in RHOS for d in RHOS]}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--outer-fold',type=int,required=True); a=p.parse_args(); started=time.time(); out=OUT/f'outer_{a.outer_fold}'; out.mkdir(parents=True,exist_ok=True)
 X=np.load(AGE/'features/matrices/X_ico4_md.npy',mmap_mode='r'); subjects=pd.read_csv(AGE/'features/subject_order.csv',dtype={'subject_id':str}); y=subjects.age_at_scan.to_numpy(float); id2i=dict(zip(subjects.subject_id,range(len(subjects)))); graph=dict(np.load(AGE/'features/ico4_weight_graph_edges.npz'))
 outer=json.loads((AGE/'folds/outer_folds.json').read_text())['folds'][a.outer_fold]; inner=json.loads((AGE/'folds/inner_folds_by_outer_fold.json').read_text())['outer_folds'][str(a.outer_fold)]['folds']
 scores=np.full((len(inner),len(RHOS),len(RHOS),len(ALPHAS)),np.nan); diagnostics=[]
 for si,split in enumerate(inner):
  tr=np.array([id2i[s] for s in split['train_subject_ids']]); va=np.array([id2i[s] for s in split['validation_subject_ids']]); xt,med,mean,scale,empty=preprocess_fit(np.asarray(X[tr])); xv=transform(np.asarray(X[va]),med,mean,scale)
  for ii,rs in enumerate(RHOS):
   for jj,rd in enumerate(RHOS):
    for ai,alpha in enumerate(ALPHAS):
     beta,inter,q,sec=solve(xt,y[tr],float(alpha),rs,rd,graph); pred=inter+xv@beta; scores[si,ii,jj,ai]=mean_absolute_error(y[va],pred); diagnostics.append({'outer_fold':a.outer_fold,'inner_fold':si,'alpha':alpha,'rho_S':rs,'rho_D':rd,'lsqr_istop':q[1],'lsqr_iterations':q[2],'runtime_seconds':sec})
     if q[1] not in (1,2): raise RuntimeError(f'LSQR nonconvergence istop={q[1]}')
 mean=scores.mean(0); grid=[]
 for ii,rs in enumerate(RHOS):
  for jj,rd in enumerate(RHOS):
   for ai,alpha in enumerate(ALPHAS): grid.append({'outer_fold':a.outer_fold,'alpha':alpha,'rho_S':rs,'rho_D':rd,'mean_inner_MAE':mean[ii,jj,ai]})
 grid=pd.DataFrame(grid); selected=[]
 for family,combos in FAMILIES.items():
  q=grid[grid.apply(lambda r:(r.rho_S,r.rho_D) in combos,axis=1)].sort_values(['mean_inner_MAE','alpha','rho_S','rho_D']).iloc[0]; selected.append({'family':family,'alpha':q.alpha,'rho_S':int(q.rho_S),'rho_D':int(q.rho_D),'mean_inner_MAE':q.mean_inner_MAE})
 selected=pd.DataFrame(selected); otr=np.array([id2i[s] for s in outer['train_subject_ids']]); ote=np.array([id2i[s] for s in outer['test_subject_ids']]); xt,med,mn,sc,empty=preprocess_fit(np.asarray(X[otr])); xv=transform(np.asarray(X[ote]),med,mn,sc); cache={}; predrows=[]; finaldiag=[]
 for r in selected.itertuples():
  key=(r.alpha,r.rho_S,r.rho_D)
  if key not in cache: cache[key]=solve(xt,y[otr],r.alpha,r.rho_S,r.rho_D,graph)
  beta,inter,q,sec=cache[key]; pred=inter+xv@beta; finaldiag.append({'outer_fold':a.outer_fold,'family':r.family,'alpha':r.alpha,'rho_S':r.rho_S,'rho_D':r.rho_D,'lsqr_istop':q[1],'lsqr_iterations':q[2],'runtime_seconds':sec})
  for idx,value in zip(ote,pred): predrows.append({'row_index':idx,'subject_id':subjects.iloc[idx].subject_id,'true_age':y[idx],'predicted_age':value,'outer_fold':a.outer_fold,'family':r.family,'selected_alpha':r.alpha,'selected_rho_S':r.rho_S,'selected_rho_D':r.rho_D,'absolute_error':abs(value-y[idx])})
 grid.to_csv(out/'inner_grid_mean_MAE.csv',index=False); selected.to_csv(out/'selected_parameters.csv',index=False); pd.DataFrame(predrows).to_csv(out/'OOF_predictions.csv',index=False); pd.DataFrame(diagnostics).to_parquet(out/'inner_solver_diagnostics.parquet',index=False,compression='zstd'); pd.DataFrame(finaldiag).to_csv(out/'outer_solver_diagnostics.csv',index=False)
 report={'status':'complete','outer_fold':a.outer_fold,'n_train':len(otr),'n_test':len(ote),'alpha_grid':ALPHAS.tolist(),'rho_grid':list(RHOS),'families':FAMILIES,'runtime_seconds':time.time()-started,'all_inner_lsqr_converged':True}
 (out/'completion.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({k:v for k,v in report.items() if k not in {'alpha_grid','families'}},indent=2))
if __name__=='__main__': main()
