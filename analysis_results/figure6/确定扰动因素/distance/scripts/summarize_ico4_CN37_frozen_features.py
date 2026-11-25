#!/usr/bin/env python3
"""Validate and summarize the frozen CN37 features and 50-vs-50 null batches."""
import json,sys,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO=Path(__file__).resolve().parents[5]; D=REPO/"analysis_results/figure6/确定扰动因素/distance"
ROOT=D/"young_CN37_frozen_spatial_features"; OUT=ROOT/"summary"; sys.path.insert(0,str(D/"scripts"))
from analyze_ico4_E_between_subject_stability import compare_long,common_anchor_region
KEYS=["node_order","hemisphere","anchor_dk_label","anchor_dk_region","distance_bin_index","distance_bin"]

def dist(frame,groups):
    metrics=["common_offset_mean_A_minus_B","absolute_common_offset","centered_residual_sd","between_subject_sd_pair_average","residual_to_between_sd_ratio","pearson","spearman"]
    rows=[]
    for keys,p in frame.groupby(groups,observed=True,sort=True):
      if not isinstance(keys,tuple): keys=(keys,)
      r=dict(zip(groups,keys)); r["n_locations"]=len(p); r["median_common_subjects"]=p.n_common_subjects.median(); r["minimum_common_subjects"]=p.n_common_subjects.min(); r["correlation_computable_fraction"]=p.pearson.notna().mean()
      for m in metrics:
        v=p[m].dropna()
        for n,q in (("median",.5),("q1",.25),("q3",.75)): r[f"{m}_{n}"]=v.quantile(q) if len(v) else np.nan
      rows.append(r)
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(exist_ok=True); roster=pd.read_csv(ROOT/"index/young_cn37_roster.csv",dtype={"subject_id":str}); old=set(pd.read_csv(D/"null_adjusted_E_stability/pilot_roster.csv",dtype={"subject_id":str}).subject_id)
    writer=None; inventory=[]; target=ROOT/"CN37_E_spatial_features.parquet"
    try:
      for subject in roster.subject_id:
        d=ROOT/"subjects"/subject; c=json.loads((d/"completion.json").read_text()); f=pd.read_parquet(d/"E_final.parquet")
        reps=pd.read_parquet(d/"local_null_realizations.parquet",columns=["replicate","replicate_seed"])
        unique=reps.drop_duplicates().sort_values("replicate")
        okay=(unique.replicate.tolist()==list(range(100)) and len(f)==4687*5 and c["reused_prior_realizations"]==(20 if subject in old else 0))
        if not okay: raise RuntimeError(f"validation failed: {subject}")
        table=pa.Table.from_pandas(f,preserve_index=False)
        if writer is None: writer=pq.ParquetWriter(target,table.schema,compression="zstd")
        writer.write_table(table); inventory.append({"subject_id":subject,"n_final_rows":len(f),"n_replicates":unique.replicate.nunique(),"reused_prior_realizations":c["reused_prior_realizations"],"validation_pass":okay})
    finally:
      if writer: writer.close()
    pd.DataFrame(inventory).to_csv(OUT/"subject_inventory.csv",index=False)
    source=pd.read_parquet(target); source["value_a"]=source.E_batch_0_49; source["value_b"]=source.E_batch_50_99
    anchor=compare_long(source[["subject_id"]+KEYS+["value_a","value_b"]],KEYS,"null_0_49_minus_50_99","null_batch")
    anchor["absolute_common_offset"]=anchor.common_offset_mean_A_minus_B.abs()
    region,regional=common_anchor_region(source,"E_batch_0_49","E_batch_50_99","null_0_49_minus_50_99","null_batch")
    region["absolute_common_offset"]=region.common_offset_mean_A_minus_B.abs()
    anchor.to_parquet(OUT/"anchor_null_batch_stability_detail.parquet",index=False,compression="zstd"); region.to_csv(OUT/"DK_null_batch_stability_detail.csv",index=False); regional.to_parquet(OUT/"subject_DK_batch_medians.parquet",index=False,compression="zstd")
    dist(anchor,["distance_bin_index","distance_bin"]).to_csv(OUT/"anchor_null_batch_stability_by_distance.csv",index=False)
    dist(anchor,["hemisphere","anchor_dk_region","distance_bin_index","distance_bin"]).to_csv(OUT/"anchor_null_batch_stability_by_DK.csv",index=False)
    dist(region,["distance_bin_index","distance_bin"]).to_csv(OUT/"DK_summary_null_batch_stability_by_distance.csv",index=False)
    coverage=source.groupby(["distance_bin_index","distance_bin"],observed=True).agg(n_subject_anchor=("E_100","size"),E100_valid_fraction=("E_100",lambda x:x.notna().mean()),batch_both_valid_fraction=("E_batch_0_49",lambda x:(x.notna() & source.loc[x.index,"E_batch_50_99"].notna()).mean()),n_valid_real_pairs_median=("n_valid_real_pairs","median"),candidate_exhausted_fraction=("candidate_exhausted","mean"),platform_complete_real_fraction=("platform_complete_real","mean")).reset_index()
    coverage.to_csv(OUT/"coverage_by_distance.csv",index=False)
    report={"status":"complete","n_subjects":37,"n_anchors":4687,"n_local_bins":5,"n_final_rows":len(source),"null_realizations":100,"batch_comparison":"0-49 vs 50-99","prior_20_reused_subjects":sum(x["reused_prior_realizations"]==20 for x in inventory),"all_subject_validations_passed":all(x["validation_pass"] for x in inventory),"missing_preserved":True,"automatic_region_exclusion":False,"reference_model_fitted":False}
    (OUT/"run_metadata.json").write_text(json.dumps(report,indent=2)+"\n")
    print(dist(anchor,["distance_bin_index","distance_bin"]).to_string(index=False)); print(coverage.to_string(index=False)); print(f"output_dir={OUT.resolve()}")
if __name__=="__main__": main()
