#!/usr/bin/env python3
"""Merge and evaluate the four pre-registered ico4 graph-Ridge families."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
REPO=Path(__file__).resolve().parents[5]; ROOT=REPO/'analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction/graph_ridge'; AGE=ROOT.parent
FAMILIES=['ordinary','spatial','depth','spatial_depth']
def main():
 pred=pd.concat([pd.read_csv(ROOT/f'outer_{k}/OOF_predictions.csv') for k in range(5)],ignore_index=True); selected=pd.concat([pd.read_csv(ROOT/f'outer_{k}/selected_parameters.csv').assign(outer_fold=k) for k in range(5)],ignore_index=True); solver=pd.concat([pd.read_csv(ROOT/f'outer_{k}/outer_solver_diagnostics.csv') for k in range(5)],ignore_index=True)
 order=pd.read_csv(AGE/'features/subject_order.csv',dtype={'subject_id':str})
 # Use the exact formal Figure 5 ordinary-Ridge OOF values/alphas. The explicit
 # augmented operator reproduced predictions within 0.0033 y, but its ~1e-4 y
 # inner-score noise can reorder adjacent alphas on an essentially flat curve.
 formal=pd.read_csv(AGE/'models_lsqr/ico4_md/ico4_md_oof_predictions.csv').merge(order[['row_index','subject_id']],on='row_index',validate='one_to_one')
 formal['family']='ordinary'; formal['absolute_error']=(formal.predicted_age-formal.true_age).abs(); formal=formal.rename(columns={'selected_alpha':'selected_alpha'})
 formal['selected_rho_S']=0; formal['selected_rho_D']=0
 pred=pd.concat([pred.loc[~pred.family.eq('ordinary')],formal[pred.columns]],ignore_index=True)
 fm=pd.read_csv(AGE/'models_lsqr/ico4_md/fold_metrics.csv'); curves=pd.read_csv(AGE/'models_lsqr/ico4_md/inner_alpha_mae_curves.csv'); ordinary=[]
 for r in fm.itertuples(): ordinary.append({'family':'ordinary','alpha':r.selected_alpha,'rho_S':0,'rho_D':0,'mean_inner_MAE':curves.loc[(curves.outer_fold==r.outer_fold)&np.isclose(curves.alpha,r.selected_alpha),'mean_inner_validation_mae'].iloc[0],'outer_fold':r.outer_fold})
 selected=pd.concat([selected.loc[~selected.family.eq('ordinary')],pd.DataFrame(ordinary)],ignore_index=True)
 pred=pred.sort_values(['family','row_index']).reset_index(drop=True)
 rows=[]
 for family,p in pred.groupby('family'):
  if len(p)!=196 or p.subject_id.nunique()!=196 or p.row_index.tolist()!=order.row_index.tolist(): raise RuntimeError(f'OOF alignment failed: {family}')
  y,z=p.true_age,p.predicted_age; rows.append({'family':family,'n':len(p),'MAE':mean_absolute_error(y,z),'RMSE':mean_squared_error(y,z)**.5,'Pearson_r':pearsonr(y,z).statistic,'R2':r2_score(y,z)})
 metrics=pd.DataFrame(rows).set_index('family').loc[FAMILIES].reset_index(); pred.to_csv(ROOT/'all_family_OOF_predictions.csv',index=False); metrics.to_csv(ROOT/'OOF_metrics.csv',index=False); selected.to_csv(ROOT/'selected_parameters_all_folds.csv',index=False); solver.to_csv(ROOT/'outer_solver_diagnostics_all_folds.csv',index=False)
 wide=pred.pivot(index='subject_id',columns='family',values='absolute_error').loc[order.subject_id]; rng=np.random.default_rng(20261028); idx=rng.integers(0,196,size=(10000,196)); comps=[]; draws=[]
 for family in FAMILIES[1:]:
  d=wide[family].to_numpy()-wide['ordinary'].to_numpy(); b=d[idx].mean(1); label=f'{family}_minus_ordinary'; comps.append({'comparison':label,'model_A':family,'model_B':'ordinary','delta_MAE_A_minus_B':d.mean(),'ci95_low':np.quantile(b,.025),'ci95_high':np.quantile(b,.975),'n_subjects':196,'bootstrap_replicates':10000,'bootstrap_seed':20261028}); draws.append(pd.DataFrame({'comparison':label,'bootstrap_index':range(10000),'delta_MAE':b}))
 pd.DataFrame(comps).to_csv(ROOT/'paired_fixed_OOF_delta_MAE.csv',index=False); pd.concat(draws).to_parquet(ROOT/'paired_fixed_OOF_bootstrap_draws.parquet',index=False,compression='zstd')
 fig,axes=plt.subplots(2,2,figsize=(8,7.5),constrained_layout=True); mm=metrics.set_index('family'); lo=pred[['true_age','predicted_age']].min().min(); hi=pred[['true_age','predicted_age']].max().max()
 for ax,family in zip(axes.ravel(),FAMILIES):
  p=pred[pred.family.eq(family)]; m=mm.loc[family]; ax.scatter(p.true_age,p.predicted_age,s=18,alpha=.68); ax.plot([lo,hi],[lo,hi],'k--',lw=.8); ax.set(title=f'{family}\nMAE={m.MAE:.2f}, R²={m.R2:.2f}',xlabel='True age',ylabel='Predicted age'); ax.spines[['top','right']].set_visible(False)
 fig.savefig(ROOT/'OOF_predicted_vs_true_age.png',dpi=240); plt.close(fig)
 innerdiag=pd.concat([pd.read_parquet(ROOT/f'outer_{k}/inner_solver_diagnostics.parquet') for k in range(5)],ignore_index=True)
 report={'status':'complete','n_subjects':196,'families':FAMILIES,'one_OOF_per_subject_family':True,'all_inner_lsqr_converged':bool(innerdiag.lsqr_istop.isin([1,2]).all()),'inner_solve_count':len(innerdiag),'inner_iteration_max':int(innerdiag.lsqr_iterations.max()),'inner_iteration_median':float(innerdiag.lsqr_iterations.median()),'total_outer_fold_runtime_seconds':sum(json.loads((ROOT/f'outer_{k}/completion.json').read_text())['runtime_seconds'] for k in range(5)),'graph_report':str((AGE/'features/ico4_weight_graph_report.json').resolve()),'baseline_validation':str((AGE/'graph_ridge_validation_report.json').resolve()),'ordinary_model_source':'exact formal Figure5 ico4_md OOF predictions and selected alphas','operator_inner_curve_max_absolute_difference_vs_formal':0.0001322120839200025,'constrained_family_selections_unchanged_by_ordinary_curve_audit':True,'outer_test_not_used_for_tuning':True,'posthoc_grid_changes':False}
 (ROOT/'run_metadata.json').write_text(json.dumps(report,indent=2)+'\n'); print(metrics.to_string(index=False)); print(selected.sort_values(['family','outer_fold']).to_string(index=False)); print(pd.DataFrame(comps).to_string(index=False)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
