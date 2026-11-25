#!/usr/bin/env python3
"""Summarize four fixed-OOF column age models and paired subject bootstrap."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

REPO=Path(__file__).resolve().parents[5]
ROOT=REPO/"analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196/age_prediction"
MODELS=["column_E_local","column_plateau","column_E_local_plateau","ico4_md"]
LABELS={"column_E_local":"E_local","column_plateau":"Plateau","column_E_local_plateau":"E_local + Plateau","ico4_md":"ico4 MD profile"}
def main():
 out=ROOT/"summary"; out.mkdir(exist_ok=True); order=pd.read_csv(ROOT/"features/subject_order.csv",dtype={"subject_id":str})
 predictions=[]; metrics=[]; alignment=None
 for model in MODELS:
  p=pd.read_csv(ROOT/"models_lsqr"/model/f"{model}_oof_predictions.csv").merge(order[["row_index","subject_id"]],on="row_index",validate="one_to_one")
  p.insert(0,"model",model); p.insert(1,"model_label",LABELS[model]); p["squared_error"]=(p.predicted_age-p.true_age)**2
  current=p[["row_index","subject_id","true_age","outer_fold"]].sort_values("row_index").reset_index(drop=True)
  if alignment is None: alignment=current
  elif not current.equals(alignment): raise RuntimeError(f"OOF alignment differs: {model}")
  y=p.true_age.to_numpy(); pred=p.predicted_age.to_numpy()
  metrics.append({"model":model,"model_label":LABELS[model],"n":len(p),"MAE":mean_absolute_error(y,pred),"RMSE":mean_squared_error(y,pred)**.5,"Pearson_r":pearsonr(y,pred).statistic,"R2":r2_score(y,pred)})
  predictions.append(p)
 allp=pd.concat(predictions,ignore_index=True); allp.to_csv(out/"all_models_OOF_predictions.csv",index=False); pd.DataFrame(metrics).to_csv(out/"OOF_metrics.csv",index=False)
 # Fixed-prediction paired subject bootstrap; no refitting.
 wide=allp.pivot(index="subject_id",columns="model",values="absolute_error").loc[order.subject_id]
 rng=np.random.default_rng(20261028); draws=10000; n=len(wide); joint="column_E_local_plateau"; rows=[]; draw_rows=[]
 indices=rng.integers(0,n,size=(draws,n))
 for comp in ("column_E_local","column_plateau","ico4_md"):
  d=wide[joint].to_numpy()-wide[comp].to_numpy(); boot=d[indices].mean(1)
  rows.append({"comparison":f"{joint}_minus_{comp}","model_A":joint,"model_B":comp,"delta_MAE_A_minus_B":d.mean(),"ci95_low":np.quantile(boot,.025),"ci95_high":np.quantile(boot,.975),"n_subjects":n,"bootstrap_replicates":draws,"bootstrap_seed":20261028,"interpretation":"negative favors joint"})
  draw_rows.append(pd.DataFrame({"comparison":f"{joint}_minus_{comp}","bootstrap_index":np.arange(draws),"delta_MAE":boot}))
 pd.DataFrame(rows).to_csv(out/"paired_subject_bootstrap_delta_MAE.csv",index=False); pd.concat(draw_rows).to_parquet(out/"paired_subject_bootstrap_draws.parquet",index=False,compression="zstd")
 # Preserve explicit shared outer-fold labels and fold sizes.
 alignment.to_csv(out/"shared_fold_assignment.csv",index=False)
 fold_counts=alignment.groupby("outer_fold").agg(n_test=("subject_id","size"),age_min=("true_age","min"),age_max=("true_age","max"),age_mean=("true_age","mean")).reset_index(); fold_counts.to_csv(out/"outer_fold_counts.csv",index=False)
 alpha=[]
 for model in MODELS:
  x=pd.read_csv(ROOT/"models_lsqr"/model/"fold_metrics.csv")[["outer_fold","n_train","n_test","selected_alpha"]]; x.insert(0,"model",model); alpha.append(x)
 pd.concat(alpha).to_csv(out/"selected_alphas_and_fold_sizes.csv",index=False)
 fig,axes=plt.subplots(2,2,figsize=(9,8),constrained_layout=True)
 lo=min(allp.true_age.min(),allp.predicted_age.min()); hi=max(allp.true_age.max(),allp.predicted_age.max())
 met=pd.DataFrame(metrics).set_index("model")
 for ax,model in zip(axes.ravel(),MODELS):
  p=allp.loc[allp.model.eq(model)]; ax.scatter(p.true_age,p.predicted_age,s=20,alpha=.72); ax.plot([lo,hi],[lo,hi],'--',color='black',lw=1)
  m=met.loc[model]; ax.set(title=f"{LABELS[model]}\nMAE={m.MAE:.2f}, r={m.Pearson_r:.2f}",xlabel="True age (years)",ylabel="Predicted age (years)")
  ax.spines[["top","right"]].set_visible(False)
 fig.savefig(out/"OOF_predicted_vs_true_age.png",dpi=240); plt.close(fig)
 report={"status":"complete","n_subjects":196,"models":MODELS,"shared_subjects_and_folds":True,"one_outer_heldout_prediction_per_subject":True,"ridge_solver":"lsqr","alpha_grid":"logspace(-4,6,21)","preprocessing":"training-fold median imputation then standardization","globally_all_nan_E_columns":[50,11770],"globally_all_nan_feature_policy":"retained as NaN in saved matrix; sklearn SimpleImputer skips them in every fold","bootstrap":"paired subjects on fixed OOF absolute errors; no refitting; shared resample indices across comparisons","disease_analysis":False}
 (out/"run_metadata.json").write_text(json.dumps(report,indent=2)+"\n"); print(pd.DataFrame(metrics).to_string(index=False)); print(pd.DataFrame(rows).to_string(index=False)); print(f"output_dir={out.resolve()}")
if __name__=='__main__': main()
