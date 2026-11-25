#!/usr/bin/env python3
"""Validate sparse graph-Ridge operators against formal ico4 Ridge OOF predictions."""
import json,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator,lsqr
REPO=Path(__file__).resolve().parents[5]; AGE=REPO/'analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction'
def preprocess_fit(x):
 with warnings.catch_warnings(): warnings.simplefilter('ignore',RuntimeWarning); med=np.nanmedian(x,axis=0)
 empty=~np.isfinite(med); med[empty]=0; z=np.where(np.isfinite(x),x,med); mean=z.mean(0); scale=z.std(0); scale[~(scale>0)]=1
 return ((z-mean)/scale).astype(np.float32),med.astype(np.float32),mean.astype(np.float32),scale.astype(np.float32),empty
def transform(x,med,mean,scale): return ((np.where(np.isfinite(x),x,med)-mean)/scale).astype(np.float32)
def operator(x,alpha,rhos,rhod,g):
 n,p=x.shape; N=int(g['n_nodes']); D=int(g['n_depths']); u=g['spatial_u']; v=g['spatial_v']; ss=float(g['spatial_B_scale']); ds=float(g['depth_B_scale']); es=len(u)*D; ed=N*(D-1)
 a=np.sqrt(alpha); cs=np.sqrt(alpha*rhos); cd=np.sqrt(alpha*rhod); rows=n+p+(es if rhos else 0)+(ed if rhod else 0)
 def mv(beta):
  b=beta.reshape(N,D); out=[x@beta,a*beta]
  if rhos: out.append((cs*ss*(b[u]-b[v])).ravel())
  if rhod: out.append((cd*ds*(b[:,1:]-b[:,:-1])).ravel())
  return np.concatenate(out)
 def rmv(q):
  pos=0; ans=x.T@q[:n]; pos=n; ans=ans+a*q[pos:pos+p]; pos+=p; grad=np.zeros((N,D),float)
  if rhos:
   z=q[pos:pos+es].reshape(len(u),D)*(cs*ss); pos+=es; np.add.at(grad,u,z); np.add.at(grad,v,-z)
  if rhod:
   z=q[pos:pos+ed].reshape(N,D-1)*(cd*ds); grad[:,:-1]-=z; grad[:,1:]+=z
  return ans+grad.ravel()
 return LinearOperator((rows,p),matvec=mv,rmatvec=rmv,dtype=np.float64)
def solve(x,y,alpha,rs,rd,g):
 ym=y.mean(); yc=y-ym; op=operator(x,alpha,rs,rd,g); rhs=np.concatenate([yc,np.zeros(op.shape[0]-len(y))]); t=time.time(); q=lsqr(op,rhs,atol=1e-4,btol=1e-4,show=False); return q[0],ym,q,time.time()-t
def main():
 X=np.load(AGE/'features/matrices/X_ico4_md.npy',mmap_mode='r'); order=pd.read_csv(AGE/'features/subject_order.csv',dtype={'subject_id':str}); folds=json.loads((AGE/'folds/outer_folds.json').read_text())['folds']; graph=dict(np.load(AGE/'features/ico4_weight_graph_edges.npz'))
 formal=pd.read_csv(AGE/'models_lsqr/ico4_md/ico4_md_oof_predictions.csv').set_index('row_index'); foldm=pd.read_csv(AGE/'models_lsqr/ico4_md/fold_metrics.csv').set_index('outer_fold'); id2i=dict(zip(order.subject_id,order.row_index)); rows=[]
 for f in folds:
  k=f['outer_fold']; tr=np.array([id2i[s] for s in f['train_subject_ids']]); te=np.array([id2i[s] for s in f['test_subject_ids']]); xt,med,mean,scale,empty=preprocess_fit(np.asarray(X[tr])); xv=transform(np.asarray(X[te]),med,mean,scale); beta,inter,q,sec=solve(xt,order.age_at_scan.to_numpy()[tr],float(foldm.loc[k].selected_alpha),0,0,graph); pred=inter+xv@beta; ref=formal.loc[te].predicted_age.to_numpy(); rows.append({'outer_fold':k,'alpha':foldm.loc[k].selected_alpha,'n_train':len(tr),'n_test':len(te),'n_empty_training_features':int(empty.sum()),'lsqr_istop':q[1],'lsqr_iterations':q[2],'lsqr_residual_norm':q[3],'runtime_seconds':sec,'max_prediction_absolute_error_vs_formal':float(np.max(np.abs(pred-ref))),'mean_prediction_absolute_error_vs_formal':float(np.mean(np.abs(pred-ref)))})
 result=pd.DataFrame(rows); result.to_csv(AGE/'graph_ridge_baseline_validation.csv',index=False)
 # One prespecified cost/convergence benchmark at fold 0, alpha=0.0316, rhoS=rhoD=1.
 f=folds[0]; tr=np.array([id2i[s] for s in f['train_subject_ids']]); xt,*rest=preprocess_fit(np.asarray(X[tr])); beta,inter,q,sec=solve(xt,order.age_at_scan.to_numpy()[tr],float(foldm.loc[0].selected_alpha),1,1,graph)
 report={'status':'pass' if result.max_prediction_absolute_error_vs_formal.max()<0.01 else 'fail','baseline_max_absolute_error_tolerance':0.01,'maximum_baseline_prediction_absolute_error':result.max_prediction_absolute_error_vs_formal.max(),'benchmark':{'outer_fold':0,'alpha':float(foldm.loc[0].selected_alpha),'rho_S':1,'rho_D':1,'lsqr_istop':int(q[1]),'lsqr_iterations':int(q[2]),'residual_norm':float(q[3]),'runtime_seconds':sec},'full_grid_not_started':True}
 (AGE/'graph_ridge_validation_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(result.to_string(index=False)); print(json.dumps(report,indent=2))
 if report['status']!='pass': raise RuntimeError('rho=0 did not reproduce formal Ridge')
if __name__=='__main__': main()
