#!/usr/bin/env python3
"""Freeze the final seed-0 ico4 local/platform pair and node indices."""
import hashlib, json, shutil
from pathlib import Path
import pandas as pd

REPO=Path(__file__).resolve().parents[5]
D=REPO/"analysis_results/figure6/确定扰动因素/distance"
OUT=D/"young_CN37_frozen_spatial_features"

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"index").mkdir(exist_ok=True); (OUT/"pairs").mkdir(exist_ok=True)
    sources={
      "nodes":D/"local_similarity_stability/ico4_nodes.csv",
      "roster":D/"local_similarity_stability/young_cn37_roster.csv",
      "local_pairs":D/"null_adjusted_E_stability/local_K_convergence/pairs/seed_00_local_k300_nested.parquet",
      "platform_pairs":D/"platform_sampling_sensitivity/pairs/seed_00_platform_k300_nested.parquet"}
    targets={"nodes":OUT/"index/ico4_nodes.csv","roster":OUT/"index/young_cn37_roster.csv",
             "local_pairs":OUT/"pairs/seed_00_local_0_10mm_k300.parquet",
             "platform_pairs":OUT/"pairs/seed_00_platform_20_30mm_k300.parquet"}
    for key,src in sources.items(): shutil.copy2(src,targets[key])
    nodes=pd.read_csv(targets["nodes"]); roster=pd.read_csv(targets["roster"],dtype={"subject_id":str})
    report={"status":"frozen","n_subjects":len(roster),"n_anchors":len(nodes),"pair_seed":0,
      "pair_base_seed":20261119,"null_base_seed":20261020,"null_realizations":100,
      "local_bins":["(0,2]","(2,4]","(4,6]","(6,8]","(8,10]"],"local_k_max":300,
      "platform_bins":["(20,22]","(22,24]","(24,26]","(26,28]","(28,30]"],"platform_k_max":300,
      "same_pairs_all_subjects":True,"replacement":False,"files":{
        k:{"path":str(v.relative_to(OUT)),"sha256":sha256(v),"bytes":v.stat().st_size} for k,v in targets.items()}}
    (OUT/"frozen_config.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="files"},indent=2)); print(f"output_dir={OUT.resolve()}")
if __name__=="__main__": main()
