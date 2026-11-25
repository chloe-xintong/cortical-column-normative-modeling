#!/usr/bin/env python3
"""Final bounded diagnostic of MD level, raw local correlations, and incremental value."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
REPO=Path(__file__).resolve().parents[5]; AGE=REPO/"analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction"; OUT=AGE/"reduced_diagnostic"
F5=Path('/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs')
SOURCES={
 'E_local':AGE/'models_lsqr/column_E_local/column_E_local_oof_predictions.csv',
 'Plateau':AGE/'models_lsqr/column_plateau/column_plateau_oof_predictions.csv',
 'r_real':AGE/'models_diagnostic/column_r_real/column_r_real_oof_predictions.csv',
 'global_mean_MD':AGE/'models_diagnostic/global_mean_md/global_mean_md_oof_predictions.csv',
 'ico4_depth_mean_MD':F5/'models_lsqr/ico4_column_mean_md/ico4_column_mean_md_oof_predictions.csv',
 'ico4_21depth_MD':AGE/'models_lsqr/ico4_md/ico4_md_oof_predictions.csv',
 'MD_plus_Plateau':AGE/'models_diagnostic/ico4_md_plateau/ico4_md_plateau_oof_predictions.csv',
 'MD_plus_E':AGE/'models_diagnostic/ico4_md_E_local/ico4_md_E_local_oof_predictions.csv',
 'MD_plus_r_real':AGE/'models_diagnostic/ico4_md_r_real/ico4_md_r_real_oof_predictions.csv'}
MODEL_DIRS={
 'E_local':AGE/'models_lsqr/column_E_local','Plateau':AGE/'models_lsqr/column_plateau','r_real':AGE/'models_diagnostic/column_r_real','global_mean_MD':AGE/'models_diagnostic/global_mean_md','ico4_depth_mean_MD':F5/'models_lsqr/ico4_column_mean_md','ico4_21depth_MD':AGE/'models_lsqr/ico4_md','MD_plus_Plateau':AGE/'models_diagnostic/ico4_md_plateau','MD_plus_E':AGE/'models_diagnostic/ico4_md_E_local','MD_plus_r_real':AGE/'models_diagnostic/ico4_md_r_real'}
def main():
 OUT.mkdir(exist_ok=True); order=pd.read_csv(AGE/'features/subject_order.csv',dtype={'subject_id':str}); outer=json.loads((AGE/'folds/outer_folds.json').read_text())
 age=dict(zip(order.subject_id,order.age_at_scan)); base=[]
 for f in outer['folds']:
  mu=float(np.mean([age[s] for s in f['train_subject_ids']]))
  base.extend({'subject_id':s,'true_age':age[s],'predicted_age':mu,'outer_fold':f['outer_fold']} for s in f['test_subject_ids'])
 base=pd.DataFrame(base).merge(order[['row_index','subject_id']],on='subject_id',validate='one_to_one'); frames=[base.assign(model='training_mean')]; alignment=None
 alpha=[]
 for name,path in SOURCES.items():
  p=pd.read_csv(path).merge(order[['row_index','subject_id']],on='row_index',validate='one_to_one'); p['model']=name; frames.append(p)
  fold=pd.read_csv(MODEL_DIRS[name]/'fold_metrics.csv'); curves=pd.read_csv(MODEL_DIRS[name]/'inner_alpha_mae_curves.csv')
  lo,hi=curves.alpha.min(),curves.alpha.max(); fold['model']=name; fold['selected_at_minimum_boundary']=np.isclose(fold.selected_alpha,lo); fold['selected_at_maximum_boundary']=np.isclose(fold.selected_alpha,hi); alpha.append(fold)
 allp=pd.concat(frames,ignore_index=True); allp['absolute_error']=(allp.predicted_age-allp.true_age).abs(); allp['squared_error']=(allp.predicted_age-allp.true_age)**2
 for name,p in allp.groupby('model'):
  a=p[['row_index','subject_id','true_age','outer_fold']].sort_values('row_index').reset_index(drop=True)
  if alignment is None: alignment=a
  elif not a.equals(alignment): raise RuntimeError(f'alignment differs: {name}')
 metrics=[]
 for name,p in allp.groupby('model',sort=False):
  y,z=p.true_age.to_numpy(),p.predicted_age.to_numpy(); metrics.append({'model':name,'n':len(p),'MAE':mean_absolute_error(y,z),'RMSE':mean_squared_error(y,z)**.5,'Pearson_r':pearsonr(y,z).statistic,'R2':r2_score(y,z)})
 metrics=pd.DataFrame(metrics); metrics.to_csv(OUT/'diagnostic_OOF_metrics.csv',index=False); allp.to_csv(OUT/'diagnostic_OOF_predictions.csv',index=False)
 alpha=pd.concat(alpha,ignore_index=True); alpha.to_csv(OUT/'selected_alpha_boundary_audit.csv',index=False)
 comparisons=[('global_mean_MD','training_mean'),('ico4_depth_mean_MD','global_mean_MD'),('ico4_21depth_MD','ico4_depth_mean_MD'),('r_real','E_local'),('MD_plus_Plateau','ico4_21depth_MD'),('MD_plus_E','ico4_21depth_MD'),('MD_plus_r_real','ico4_21depth_MD')]
 wide=allp.pivot(index='subject_id',columns='model',values='absolute_error').loc[order.subject_id]; rng=np.random.default_rng(20261028); idx=rng.integers(0,196,size=(10000,196)); rows=[]; draws=[]
 for a,b in comparisons:
  d=wide[a].to_numpy()-wide[b].to_numpy(); boot=d[idx].mean(1); label=f'{a}_minus_{b}'; rows.append({'comparison':label,'model_A':a,'model_B':b,'delta_MAE_A_minus_B':d.mean(),'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975),'n_subjects':196,'bootstrap_replicates':10000,'bootstrap_seed':20261028}); draws.append(pd.DataFrame({'comparison':label,'bootstrap_index':range(10000),'delta_MAE':boot}))
 pd.DataFrame(rows).to_csv(OUT/'paired_fixed_OOF_delta_MAE.csv',index=False); pd.concat(draws).to_parquet(OUT/'paired_fixed_OOF_bootstrap_draws.parquet',index=False,compression='zstd')
 fig,axes=plt.subplots(3,3,figsize=(11,10),constrained_layout=True); mm=metrics.set_index('model'); lo=allp[['true_age','predicted_age']].min().min(); hi=allp[['true_age','predicted_age']].max().max()
 for ax,name in zip(axes.ravel(),['training_mean','global_mean_MD','ico4_depth_mean_MD','ico4_21depth_MD','r_real','E_local','MD_plus_Plateau','MD_plus_E','MD_plus_r_real']):
  p=allp[allp.model.eq(name)]; ax.scatter(p.true_age,p.predicted_age,s=13,alpha=.65); ax.plot([lo,hi],[lo,hi],'k--',lw=.8); m=mm.loc[name]; ax.set(title=f'{name}\nMAE={m.MAE:.2f}, R²={m.R2:.2f}',xlabel='True age',ylabel='Predicted age'); ax.spines[['top','right']].set_visible(False)
 fig.savefig(OUT/'diagnostic_OOF_scatter.png',dpi=220); plt.close(fig)
 report={'status':'complete','scope':'bounded cached-feature diagnostic','n_subjects':196,'new_pairs':False,'new_null':False,'shared_nested_folds':True,'fold_local_preprocessing':True,'paired_bootstrap':'fixed OOF; shared subject resamples; no refitting','alpha_expansion_performed':False,'minimum_boundary_counts':alpha.groupby('model').selected_at_minimum_boundary.sum().astype(int).to_dict(),'maximum_boundary_counts':alpha.groupby('model').selected_at_maximum_boundary.sum().astype(int).to_dict()}
 (OUT/'run_metadata.json').write_text(json.dumps(report,indent=2)+'\n'); print(metrics.to_string(index=False)); print(pd.DataFrame(rows).to_string(index=False)); print(alpha.groupby('model')[['selected_at_minimum_boundary','selected_at_maximum_boundary']].sum().to_string()); print(f'output_dir={OUT.resolve()}')
if __name__=='__main__': main()
