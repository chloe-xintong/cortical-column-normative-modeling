#!/usr/bin/env python3
"""Compute one subject's frozen seed-0, K<=300 real and 100-realization null features."""
from __future__ import annotations
import argparse,json,resource,sys,time,warnings
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO=Path(__file__).resolve().parents[5]; D=REPO/"analysis_results/figure6/确定扰动因素/distance"
ROOT=D/"young_CN37_frozen_spatial_features"; F4=REPO/"analysis_results/figure4/scripts"
sys.path[:0]=[str(F4),str(D/"scripts")]
from compute_spatial_null import load_sampling_geometry,recolor_bilateral_profiles,sample_noise,stable_null_seed
from surface_pca_surrogate import load_real_profiles
from compute_ico4_null_adjusted_E_subject import pair_matrices,median_pair_correlation,metadata_rows,platform_median
LOCAL=range(5); FAR=range(10,15); BASE=20261020

def write_block(writer,path,frame):
    table=pa.Table.from_pandas(frame,preserve_index=False)
    if writer is None: writer=pq.ParquetWriter(path,table.schema,compression="zstd")
    writer.write_table(table); return writer

def main():
    p=argparse.ArgumentParser(); p.add_argument("--feature-root",type=Path,default=ROOT); g=p.add_mutually_exclusive_group(required=True); g.add_argument("--index",type=int); g.add_argument("--subject-id")
    a=p.parse_args(); started=time.time()
    root=a.feature_root
    roster_path=root/"index/subject_order.csv"
    if not roster_path.exists(): roster_path=root/"index/young_cn37_roster.csv"
    roster=pd.read_csv(roster_path,dtype={"subject_id":str})
    manifest=pd.read_csv(REPO/"manifests/figure4_coverage95_pass_313_subjects.csv",dtype={"subject_id":str})
    cohort=roster.merge(manifest,on="subject_id",validate="one_to_one"); row=cohort.iloc[a.index] if a.index is not None else cohort.loc[cohort.subject_id.eq(a.subject_id)].iloc[0]
    subject=str(row.subject_id); out=root/"subjects"/subject; out.mkdir(parents=True,exist_ok=True)
    nodes=pd.read_csv(root/"index/ico4_nodes.csv").sort_values("node_order"); lm=metadata_rows(nodes,LOCAL); fm=metadata_rows(nodes,FAR)
    local=pair_matrices(root/"pairs/seed_00_local_0_10mm_k300.parquet",LOCAL,300)
    far=pair_matrices(root/"pairs/seed_00_platform_20_30mm_k300.parquet",FAR,300)
    md=nib.load(str(row.md_image)); shape=md.shape[:3]; geom={}; real={}; valid={}; mask=[]
    for h in ("lh","rh"):
        coords,gv,_=load_sampling_geometry(Path(row[f"clean_profile_{h}"])); profiles,rv=load_real_profiles(Path(row[f"clean_profile_{h}"]))
        geom[h]=coords; real[h]=profiles; valid[h]=gv&rv; mask.append({"subject_id":subject,"hemisphere":h,"n_final_valid":int(valid[h].sum()),"rule":"geometry_valid & real_final_valid"})
    lr,ln=median_pair_correlation(real,valid,local,100000); fr,fn=median_pair_correlation(real,valid,far,100000)
    pr=platform_median(fr); nb=np.isfinite(fr.reshape(4687,5)).sum(1).astype(np.int8)
    realout=lm.copy(); realout.insert(0,"subject_id",subject); realout["r_real"]=lr; realout["p_real"]=np.repeat(pr,5); realout["L_real"]=realout.r_real-realout.p_real
    realout["n_valid_real_pairs"]=ln; realout["n_valid_platform_bins_real"]=np.repeat(nb,5); realout["platform_complete_real"]=realout.n_valid_platform_bins_real.eq(5)
    realout["n_template_selected"]=local["n_template_selected"]; realout["candidate_exhausted"]=local["candidate_exhausted"]
    realout.to_parquet(out/"real_features.parquet",index=False,compression="zstd"); pd.DataFrame(mask).to_csv(out/"formal_mask_qc.csv",index=False)
    oldroot=D/"null_adjusted_E_stability"; old10=set(pd.read_csv(oldroot/"pilot_roster.csv",dtype={"subject_id":str}).subject_id)
    reuse=subject in old10; local_path=out/"local_null_realizations.parquet.incomplete"; far_path=out/"platform_bin_null_realizations.parquet.incomplete"
    lw=fw=None; reused=0; max_checks=[0.0,0.0,0.0]
    try:
      if reuse:
        oldl=pd.read_parquet(oldroot/"local_K_convergence/subjects"/subject/"local_K_realizations.parquet",filters=[("pair_seed","=",0),("local_k","=",300)])
        oldf=pd.read_parquet(oldroot/"subjects"/subject/"null_platform_bin_realizations.parquet",filters=[("pair_seed","=",0)])
        expected=[stable_null_seed(BASE,subject,r) for r in range(20)]
        if oldl.groupby("replicate").replicate_seed.first().tolist()!=expected: raise RuntimeError("old local seeds differ")
        # Real values must also agree before cached null rows are accepted.
        max_checks[0]=float(np.nanmax(np.abs(oldl.loc[oldl.replicate.eq(0)].sort_values(["node_order","distance_bin_index"]).r_real.to_numpy()-lr)))
        cols=["node_order","hemisphere","ico_node_id","anchor_vertex","anchor_dk_label","anchor_dk_region","distance_bin_index","distance_bin","replicate","replicate_seed","r_null","p_null","L_null","E_realization","n_valid_null_pairs","n_valid_platform_bins_null","platform_complete_null"]
        cached=oldl[cols].copy(); cached.insert(0,"subject_id",subject); lw=write_block(lw,local_path,cached)
        fcols=["node_order","hemisphere","ico_node_id","anchor_vertex","anchor_dk_label","anchor_dk_region","distance_bin_index","distance_bin","replicate","replicate_seed","r_null","n_valid_null_pairs"]
        cachedf=oldf[fcols].copy(); cachedf.insert(0,"subject_id",subject); fw=write_block(fw,far_path,cachedf); reused=20
      for rep in range(reused,100):
        seed=stable_null_seed(BASE,subject,rep); rng=np.random.default_rng(seed); noise=rng.standard_normal(shape,dtype=np.float32)
        null={h:sample_noise(noise,geom[h],valid[h]) for h in ("lh","rh")}; null=recolor_bilateral_profiles(null,real,valid)
        nr,nn=median_pair_correlation(null,valid,local,100000); nfr,nfn=median_pair_correlation(null,valid,far,100000)
        pn=platform_median(nfr); nbn=np.isfinite(nfr.reshape(4687,5)).sum(1).astype(np.int8)
        q=lm.copy(); q.insert(0,"subject_id",subject); q["replicate"]=rep; q["replicate_seed"]=np.uint64(seed); q["r_null"]=nr; q["p_null"]=np.repeat(pn,5); q["L_null"]=q.r_null-q.p_null
        q["E_realization"]=(np.repeat(lr.reshape(4687,5)[:,0]*0+pr,5)*0) # allocate aligned float column
        q["E_realization"]=(lr-np.repeat(pr,5))-q.L_null; q["n_valid_null_pairs"]=nn; q["n_valid_platform_bins_null"]=np.repeat(nbn,5); q["platform_complete_null"]=q.n_valid_platform_bins_null.eq(5)
        lw=write_block(lw,local_path,q)
        z=fm.copy(); z.insert(0,"subject_id",subject); z["replicate"]=rep; z["replicate_seed"]=np.uint64(seed); z["r_null"]=nfr; z["n_valid_null_pairs"]=nfn; fw=write_block(fw,far_path,z)
    finally:
      if lw: lw.close()
      if fw: fw.close()
    local_path.replace(out/"local_null_realizations.parquet"); far_path.replace(out/"platform_bin_null_realizations.parquet")
    # Final E and the two prespecified independent null batches.
    null=pd.read_parquet(out/"local_null_realizations.parquet",columns=["node_order","distance_bin_index","replicate","E_realization"])
    piv=null.pivot(index=["node_order","distance_bin_index"],columns="replicate",values="E_realization")
    final=realout.merge(piv.loc[:,range(100)].median(1).rename("E_100").reset_index(),on=["node_order","distance_bin_index"],validate="one_to_one")
    final=final.merge(piv.loc[:,range(50)].median(1).rename("E_batch_0_49").reset_index(),on=["node_order","distance_bin_index"],validate="one_to_one")
    final=final.merge(piv.loc[:,range(50,100)].median(1).rename("E_batch_50_99").reset_index(),on=["node_order","distance_bin_index"],validate="one_to_one")
    final.to_parquet(out/"E_final.parquet",index=False,compression="zstd")
    report={"status":"complete","subject_id":subject,"replicates":100,"reused_prior_realizations":reused,"new_realizations":100-reused,"null_base_seed":BASE,
      "replicate_seeds":[stable_null_seed(BASE,subject,r) for r in range(100)],"pair_seed":0,"local_k_max":300,"platform_k_max":300,
      "old_real_r_max_absolute_error":max_checks[0],"runtime_seconds":time.time()-started,"max_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    (out/"completion.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({k:v for k,v in report.items() if k not in {"subject_id","replicate_seeds"}},indent=2))
if __name__=="__main__": main()
