#!/usr/bin/env python3
"""Paired fixed-OOF comparisons for ico3+r_real and r_real+E."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
REPO=Path(__file__).resolve().parents[5]; AGE=REPO/'analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction'; OUT=AGE/'reduced_diagnostic'; F5=Path('/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs')
S={'ico3_md':F5/'models_lsqr/ico3_md/ico3_md_oof_predictions.csv','ico3_md_r_real':AGE/'models_diagnostic/ico3_md_r_real/ico3_md_r_real_oof_predictions.csv','r_real':AGE/'models_diagnostic/column_r_real/column_r_real_oof_predictions.csv','E_local':AGE/'models_lsqr/column_E_local/column_E_local_oof_predictions.csv','r_real_E_local':AGE/'models_diagnostic/column_r_real_E_local/column_r_real_E_local_oof_predictions.csv'}
def main():
 order=pd.read_csv(AGE/'features/subject_order.csv',dtype={'subject_id':str}); frames=[]; metrics=[]
 for name,path in S.items():
  p=pd.read_csv(path).merge(order[['row_index','subject_id']],on='row_index',validate='one_to_one').sort_values('row_index'); p['model']=name; p['absolute_error']=(p.predicted_age-p.true_age).abs(); frames.append(p)
  y,z=p.true_age,p.predicted_age; metrics.append({'model':name,'n':len(p),'MAE':mean_absolute_error(y,z),'RMSE':mean_squared_error(y,z)**.5,'Pearson_r':pearsonr(y,z).statistic,'R2':r2_score(y,z)})
 a=pd.concat(frames); wide=a.pivot(index='subject_id',columns='model',values='absolute_error').loc[order.subject_id]; rng=np.random.default_rng(20261028); idx=rng.integers(0,196,size=(10000,196)); rows=[]; draws=[]
 for x,y in [('ico3_md_r_real','ico3_md'),('r_real_E_local','r_real'),('r_real_E_local','E_local')]:
  d=wide[x].to_numpy()-wide[y].to_numpy(); b=d[idx].mean(1); label=f'{x}_minus_{y}'; rows.append({'comparison':label,'model_A':x,'model_B':y,'delta_MAE_A_minus_B':d.mean(),'ci95_low':np.quantile(b,.025),'ci95_high':np.quantile(b,.975),'n_subjects':196,'bootstrap_replicates':10000,'bootstrap_seed':20261028}); draws.append(pd.DataFrame({'comparison':label,'bootstrap_index':range(10000),'delta_MAE':b}))
 pd.DataFrame(metrics).to_csv(OUT/'two_extra_OOF_metrics.csv',index=False); pd.DataFrame(rows).to_csv(OUT/'two_extra_paired_fixed_OOF_delta_MAE.csv',index=False); pd.concat(draws).to_parquet(OUT/'two_extra_bootstrap_draws.parquet',index=False,compression='zstd'); print(pd.DataFrame(metrics).to_string(index=False)); print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
