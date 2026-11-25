#!/usr/bin/env python3
"""Freeze the Figure-5 CN196 column-feature cohort and reuse completed CN37 outputs."""
import hashlib,json,os,shutil
from pathlib import Path
import pandas as pd
REPO=Path(__file__).resolve().parents[5]; D=REPO/"analysis_results/figure6/确定扰动因素/distance"
SOURCE=D/"young_CN37_frozen_spatial_features"; OUT=D/"column_age_prediction_Figure5_CN196"
F5=Path("/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs")
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 for d in (OUT/"index",OUT/"pairs",OUT/"subjects",OUT/"logs"): d.mkdir(parents=True,exist_ok=True)
 copies={F5/"features/subject_order.csv":OUT/"index/subject_order.csv",SOURCE/"index/ico4_nodes.csv":OUT/"index/ico4_nodes.csv",SOURCE/"pairs/seed_00_local_0_10mm_k300.parquet":OUT/"pairs/seed_00_local_0_10mm_k300.parquet",SOURCE/"pairs/seed_00_platform_20_30mm_k300.parquet":OUT/"pairs/seed_00_platform_20_30mm_k300.parquet"}
 for a,b in copies.items(): shutil.copy2(a,b)
 order=pd.read_csv(OUT/"index/subject_order.csv",dtype={"subject_id":str}); reused=[]
 for sid in order.subject_id:
  src=SOURCE/"subjects"/sid; dst=OUT/"subjects"/sid
  if (src/"completion.json").exists() and not dst.exists():
   os.symlink(src.resolve(),dst,target_is_directory=True); reused.append(sid)
  elif (src/"completion.json").exists():
   reused.append(sid)
 report={"status":"frozen","n_subjects":len(order),"n_reused_complete":len(reused),"n_to_compute":len(order)-len(reused),"reused_subject_ids":reused,"pair_seed":0,"local_k_max":300,"platform_k_max":300,"null_realizations":100,"null_base_seed":20261020,"fold_source":str((F5/'folds').resolve()),"files":{str(b.relative_to(OUT)):digest(b) for b in copies.values()}}
 (OUT/"feature_inventory.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({k:v for k,v in report.items() if k not in {'reused_subject_ids','files'}},indent=2)); print(f"output_dir={OUT.resolve()}")
if __name__=='__main__': main()
