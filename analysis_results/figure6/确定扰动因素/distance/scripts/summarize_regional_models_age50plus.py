#!/usr/bin/env python3
"""Summarize formal Figure-5 regional-model OOF predictions for age >=50."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
F5=Path('/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs'); REPO=Path(__file__).resolve().parents[5]; OUT=REPO/'analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction/reduced_diagnostic/regional_age50plus'
MODELS=['regional_mean_md','regional_thickness','regional_profile','regional_profile_thickness']
def main():
 OUT.mkdir(parents=True,exist_ok=True); order=pd.read_csv(F5/'features/subject_order.csv',dtype={'subject_id':str}); frames=[]; rows=[]
 for model in MODELS:
  p=pd.read_csv(F5/'models_lsqr'/model/f'{model}_oof_predictions.csv').merge(order[['row_index','subject_id']],on='row_index',validate='one_to_one'); p=p[p.true_age>=50].copy(); p['model']=model; p['absolute_error']=(p.predicted_age-p.true_age).abs(); frames.append(p)
  y,z=p.true_age,p.predicted_age; rows.append({'model':model,'age_rule':'>=50','n':len(p),'age_min':y.min(),'age_max':y.max(),'MAE':mean_absolute_error(y,z),'RMSE':mean_squared_error(y,z)**.5,'Pearson_r':pearsonr(y,z).statistic,'R2':r2_score(y,z)})
 allp=pd.concat(frames,ignore_index=True); metrics=pd.DataFrame(rows); metrics.to_csv(OUT/'regional_OOF_metrics_age50plus.csv',index=False); allp.to_csv(OUT/'regional_OOF_predictions_age50plus.csv',index=False)
 wide=allp.pivot(index='subject_id',columns='model',values='absolute_error'); n=len(wide); rng=np.random.default_rng(20261028); idx=rng.integers(0,n,size=(10000,n)); comps=[]; draws=[]
 pairs=[('regional_profile','regional_mean_md'),('regional_profile_thickness','regional_profile'),('regional_profile_thickness','regional_thickness')]
 for a,b in pairs:
  d=wide[a].to_numpy()-wide[b].to_numpy(); boot=d[idx].mean(1); label=f'{a}_minus_{b}'; comps.append({'comparison':label,'age_rule':'>=50','model_A':a,'model_B':b,'delta_MAE_A_minus_B':d.mean(),'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975),'n_subjects':n,'bootstrap_replicates':10000,'bootstrap_seed':20261028}); draws.append(pd.DataFrame({'comparison':label,'bootstrap_index':range(10000),'delta_MAE':boot}))
 pd.DataFrame(comps).to_csv(OUT/'regional_paired_fixed_OOF_delta_MAE_age50plus.csv',index=False); pd.concat(draws).to_parquet(OUT/'regional_bootstrap_draws_age50plus.parquet',index=False,compression='zstd'); print(metrics.to_string(index=False)); print(pd.DataFrame(comps).to_string(index=False))
if __name__=='__main__': main()
