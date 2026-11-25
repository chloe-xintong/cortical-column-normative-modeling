#!/usr/bin/env python3
"""Build aligned CN196 E-local, plateau, combined, and ico4-MD Ridge inputs."""
import json,shutil
from pathlib import Path
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[5]
ROOT=REPO/"analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196"
OUT=ROOT/"age_prediction"; F5=Path("/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs")
def main():
 (OUT/"features/matrices").mkdir(parents=True,exist_ok=True); (OUT/"folds").mkdir(parents=True,exist_ok=True)
 order=pd.read_csv(ROOT/"index/subject_order.csv",dtype={"subject_id":str}).sort_values("row_index").reset_index(drop=True)
 node=pd.read_csv(ROOT/"index/ico4_nodes.csv").sort_values("node_order").reset_index(drop=True)
 e=np.full((len(order),len(node)*5),np.nan,np.float32); p=np.full((len(order),len(node)),np.nan,np.float32)
 reference_keys=None
 for i,sid in enumerate(order.subject_id):
  x=pd.read_parquet(ROOT/"subjects"/sid/"E_final.parquet").sort_values(["node_order","distance_bin_index"])
  keys=x[["node_order","hemisphere","anchor_vertex","anchor_dk_label","anchor_dk_region","distance_bin_index","distance_bin"]].reset_index(drop=True)
  if reference_keys is None: reference_keys=keys
  elif not keys.equals(reference_keys): raise RuntimeError(f"feature index differs: {sid}")
  e[i]=x.E_100.to_numpy(np.float32)
  q=x.loc[x.distance_bin_index.eq(0)].sort_values("node_order")
  if q.node_order.tolist()!=node.node_order.tolist(): raise RuntimeError(f"node order differs: {sid}")
  p[i]=q.p_real.to_numpy(np.float32)
 np.save(OUT/"features/matrices/X_column_E_local.npy",e); np.save(OUT/"features/matrices/X_column_plateau.npy",p)
 md=np.load(F5/"features/matrices/X_ico4_md.npy",mmap_mode="r")
 f5order=pd.read_csv(F5/"features/subject_order.csv",dtype={"subject_id":str})
 if not order[["row_index","subject_id","age_at_scan"]].equals(f5order[["row_index","subject_id","age_at_scan"]]): raise RuntimeError("Figure5 subject order differs")
 if md.shape!=(196,4687*21): raise RuntimeError("ico4 MD shape differs")
 shutil.copy2(F5/"features/matrices/X_ico4_md.npy",OUT/"features/matrices/X_ico4_md.npy")
 order.to_csv(OUT/"features/subject_order.csv",index=False); reference_keys.to_csv(OUT/"features/E_local_feature_index.csv",index=False)
 node[["node_order","hemisphere","ico_node_id","fsaverage_vertex_id","dk_label_id","dk_label_name"]].to_csv(OUT/"features/plateau_feature_index.csv",index=False)
 for name in ("outer_folds.json","inner_folds_by_outer_fold.json","fold_age_summary.csv"): shutil.copy2(F5/"folds"/name,OUT/"folds"/name)
 report={"status":"complete","n_subjects":196,"subject_order_exact_Figure5":True,"matrix_shapes":{"column_E_local":list(e.shape),"column_plateau":list(p.shape),"column_E_local_plateau":[196,e.shape[1]+p.shape[1]],"ico4_md":list(md.shape)},"nan_fraction":{"column_E_local":float(np.isnan(e).mean()),"column_plateau":float(np.isnan(p).mean()),"ico4_md":float(np.isnan(md).mean())},"negative_E_fraction_among_finite":float((e[np.isfinite(e)]<0).mean()),"missing_value_policy":"NaN retained; fold-local median imputation in shared Ridge pipeline","fold_source":str((F5/'folds').resolve())}
 (OUT/"matrix_build_report.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2)); print(f"output_dir={OUT.resolve()}")
if __name__=='__main__': main()
