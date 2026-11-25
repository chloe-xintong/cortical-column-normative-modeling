#!/usr/bin/env python3
"""Build cached-data-only matrices for column age-prediction diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[5]; ROOT=REPO/"analysis_results/figure6/确定扰动因素/distance/column_age_prediction_Figure5_CN196"; AGE=ROOT/"age_prediction"; MAT=AGE/"features/matrices"
def main():
 order=pd.read_csv(AGE/"features/subject_order.csv",dtype={"subject_id":str}); index=pd.read_csv(AGE/"features/E_local_feature_index.csv")
 raw=np.full((196,23435),np.nan,np.float32)
 for i,sid in enumerate(order.subject_id):
  x=pd.read_parquet(ROOT/"subjects"/sid/"E_final.parquet").sort_values(["node_order","distance_bin_index"])
  if not x[["node_order","distance_bin_index"]].reset_index(drop=True).equals(index[["node_order","distance_bin_index"]]): raise RuntimeError(f"index mismatch {sid}")
  raw[i]=x.r_real.to_numpy(np.float32)
 md=np.load(MAT/"X_ico4_md.npy",mmap_mode="r"); cube=np.asarray(md).reshape(196,4687,21)
 with np.errstate(invalid="ignore"):
  depth_mean=np.nanmean(cube,axis=2).astype(np.float32); global_mean=np.nanmean(cube,axis=(1,2)).astype(np.float32)[:,None]
 np.save(MAT/"X_column_r_real.npy",raw); np.save(MAT/"X_ico4_depth_mean_md.npy",depth_mean); np.save(MAT/"X_global_mean_md.npy",global_mean)
 report={"status":"complete","source":"existing cached real features and Figure5 ico4 MD only","new_null":False,"matrix_shapes":{"column_r_real":list(raw.shape),"global_mean_md":list(global_mean.shape),"ico4_depth_mean_md":list(depth_mean.shape),"ico4_md_E_local":[196,98427+23435],"ico4_md_plateau":[196,98427+4687]},"nan_fraction":{"column_r_real":float(np.isnan(raw).mean()),"global_mean_md":float(np.isnan(global_mean).mean()),"ico4_depth_mean_md":float(np.isnan(depth_mean).mean())}}
 (AGE/"diagnostic_matrix_report.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
