#!/usr/bin/env python3
"""Evaluate fixed graph-Ridge OOF predictions among participants aged >=50 years."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
REPO=Path(__file__).resolve().parents[5]; ROOT=REPO/'analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction/graph_ridge'
def main():
 p=pd.read_csv(ROOT/'all_family_OOF_predictions.csv'); p=p[p.true_age>=50].copy(); families=['ordinary','spatial','depth','spatial_depth']; rows=[]
 for f in families:
  q=p[p.family.eq(f)]; y,z=q.true_age,q.predicted_age; rows.append({'family':f,'age_rule':'>=50','n':len(q),'age_min':y.min(),'age_max':y.max(),'MAE':mean_absolute_error(y,z),'RMSE':mean_squared_error(y,z)**.5,'Pearson_r':pearsonr(y,z).statistic,'R2':r2_score(y,z)})
 metrics=pd.DataFrame(rows); metrics.to_csv(ROOT/'OOF_metrics_age50plus.csv',index=False)
 wide=p.pivot(index='subject_id',columns='family',values='absolute_error'); n=len(wide); rng=np.random.default_rng(20261028); idx=rng.integers(0,n,size=(10000,n)); comps=[]; draws=[]
 for f in families[1:]:
  d=wide[f].to_numpy()-wide.ordinary.to_numpy(); b=d[idx].mean(1); label=f'{f}_minus_ordinary'; comps.append({'comparison':label,'age_rule':'>=50','model_A':f,'model_B':'ordinary','delta_MAE_A_minus_B':d.mean(),'ci95_low':np.quantile(b,.025),'ci95_high':np.quantile(b,.975),'n_subjects':n,'bootstrap_replicates':10000,'bootstrap_seed':20261028}); draws.append(pd.DataFrame({'comparison':label,'bootstrap_index':range(10000),'delta_MAE':b}))
 pd.DataFrame(comps).to_csv(ROOT/'paired_fixed_OOF_delta_MAE_age50plus.csv',index=False); pd.concat(draws).to_parquet(ROOT/'paired_fixed_OOF_bootstrap_draws_age50plus.parquet',index=False,compression='zstd'); p.to_csv(ROOT/'all_family_OOF_predictions_age50plus.csv',index=False); print(metrics.to_string(index=False)); print(pd.DataFrame(comps).to_string(index=False))
if __name__=='__main__': main()
