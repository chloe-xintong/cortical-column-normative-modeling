#!/usr/bin/env python3
"""Replay cached null fields and compute nested local K=30/100/300 E."""

from __future__ import annotations

import json
import resource
import sys
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[5]
DISTANCE = REPO / "analysis_results/figure6/确定扰动因素/distance"
F4 = REPO / "analysis_results/figure4/scripts"
LOCAL_SCRIPTS = DISTANCE / "scripts"
sys.path[:0] = [str(F4), str(LOCAL_SCRIPTS)]
from compute_spatial_null import load_sampling_geometry, recolor_bilateral_profiles, sample_noise, stable_null_seed  # noqa: E402
from surface_pca_surrogate import load_real_profiles  # noqa: E402
from compute_ico4_null_adjusted_E_subject import pair_matrices, metadata_rows  # noqa: E402


PAIR_SEEDS=(0,1,2); K_VALUES=(30,100,300); NULL_BASE_SEED=20261020


def correlation_values(profiles, valid, matrices, chunk_size=100_000):
    values=np.full_like(matrices["lh_anchor"],np.nan,dtype=np.float32)
    for hemi in ("lh","rh"):
        anchor=matrices[f"{hemi}_anchor"].ravel(); neighbor=matrices[f"{hemi}_neighbor"].ravel()
        output=values.ravel()
        for start in range(0,len(anchor),chunk_size):
            stop=min(start+chunk_size,len(anchor)); a=anchor[start:stop]; n=neighbor[start:stop]
            ids=np.flatnonzero((a>=0)&(n>=0))
            if not len(ids): continue
            aa,nn=a[ids],n[ids]; ids=ids[valid[hemi][aa]&valid[hemi][nn]]
            if not len(ids): continue
            aa,nn=a[ids],n[ids]
            left=profiles[hemi][aa].astype(np.float64); right=profiles[hemi][nn].astype(np.float64)
            left-=left.mean(1,keepdims=True); right-=right.mean(1,keepdims=True)
            denominator=np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1)
            corr=np.full(len(ids),np.nan); okay=denominator>0
            corr[okay]=np.sum(left[okay]*right[okay],axis=1)/denominator[okay]
            output[start+ids]=corr.astype(np.float32)
    return values


def summarize_k(values, k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        median=np.nanmedian(values[:,:k],axis=1).astype(np.float32)
    count=np.isfinite(values[:,:k]).sum(1).astype(np.int16)
    return median,count


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__); g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--index",type=int); g.add_argument("--subject-id")
    a=p.parse_args(); started=time.time()
    parent=DISTANCE/"null_adjusted_E_stability"; pilot=parent/"local_K_convergence"
    roster=pd.read_csv(parent/"pilot_roster.csv",dtype={"subject_id":str})
    manifest=pd.read_csv(REPO/"manifests/figure4_coverage95_pass_313_subjects.csv",dtype={"subject_id":str})
    cohort=roster.merge(manifest,on="subject_id",validate="one_to_one")
    row=cohort.iloc[a.index] if a.index is not None else cohort.loc[cohort.subject_id.eq(a.subject_id)].iloc[0]
    subject=str(row.subject_id); out=pilot/"subjects"/subject; out.mkdir(parents=True,exist_ok=True)
    nodes=pd.read_csv(DISTANCE/"local_similarity_stability/ico4_nodes.csv").sort_values("node_order")
    meta=metadata_rows(nodes,range(5))
    matrices={s:pair_matrices(pilot/"pairs"/f"seed_{s:02d}_local_k300_nested.parquet",range(5),300)
              for s in PAIR_SEEDS}
    cached=pd.read_parquet(parent/"subjects"/subject/"null_adjusted_local_realizations.parquet")
    platform=cached.drop_duplicates(["replicate","pair_seed","node_order"])[[
        "replicate","replicate_seed","pair_seed","node_order","p_real","p_null",
        "n_valid_platform_bins_real","n_valid_platform_bins_null","platform_complete_real",
        "platform_complete_null","platform_any_candidate_exhausted"]]
    md=nib.load(str(row.md_image)); shape=md.shape[:3]
    geometries={}; real={}; final_valid={}
    for hemi in ("lh","rh"):
        coordinates,geometry_valid,_=load_sampling_geometry(Path(row[f"clean_profile_{hemi}"]))
        profiles,real_valid=load_real_profiles(Path(row[f"clean_profile_{hemi}"]))
        geometries[hemi]=coordinates; real[hemi]=profiles; final_valid[hemi]=geometry_valid&real_valid
    real_values={}
    for seed in PAIR_SEEDS:
        values=correlation_values(real,final_valid,matrices[seed]); real_values[seed]={}
        for k in K_VALUES: real_values[seed][k]=summarize_k(values,k)
    completion=json.loads((parent/"subjects"/subject/"completion.json").read_text())
    expected=[stable_null_seed(NULL_BASE_SEED,subject,r) for r in range(20)]
    if completion["replicate_seeds"]!=expected: raise RuntimeError("Cached formal replicate seeds differ")
    k30_checks=[]; writer=None
    temporary=out/"local_K_realizations.parquet.incomplete"
    try:
        for replicate,rep_seed in enumerate(expected):
            replicate_outputs=[]
            rng=np.random.default_rng(rep_seed); noise=rng.standard_normal(shape,dtype=np.float32)
            null={h:sample_noise(noise,geometries[h],final_valid[h]) for h in ("lh","rh")}
            null=recolor_bilateral_profiles(null,real,final_valid)
            for seed in PAIR_SEEDS:
                values=correlation_values(null,final_valid,matrices[seed])
                pform=platform.loc[(platform.replicate==replicate)&(platform.pair_seed==seed)].sort_values("node_order")
                if len(pform)!=4687: raise RuntimeError("Cached platform rows incomplete")
                for k in K_VALUES:
                    r_null,n_null=summarize_k(values,k); r_real,n_real=real_values[seed][k]
                    current=meta.copy(); current["replicate"]=replicate; current["replicate_seed"]=np.uint64(rep_seed)
                    current["pair_seed"]=seed; current["local_k"]=k
                    current["r_real"]=r_real; current["r_null"]=r_null
                    current["p_real"]=np.repeat(pform.p_real.to_numpy(np.float32),5)
                    current["p_null"]=np.repeat(pform.p_null.to_numpy(np.float32),5)
                    current["L_real"]=current.r_real-current.p_real; current["L_null"]=current.r_null-current.p_null
                    current["E_realization"]=current.L_real-current.L_null
                    current["n_valid_real_pairs"]=n_real; current["n_valid_null_pairs"]=n_null
                    current["n_template_selected"]=np.minimum(matrices[seed]["n_geometry"],k).astype(np.int16)
                    current["candidate_exhausted"]=matrices[seed]["n_geometry"]<=k
                    current["n_valid_platform_bins_real"]=np.repeat(pform.n_valid_platform_bins_real.to_numpy(np.int8),5)
                    current["n_valid_platform_bins_null"]=np.repeat(pform.n_valid_platform_bins_null.to_numpy(np.int8),5)
                    current["platform_complete_real"]=np.repeat(pform.platform_complete_real.to_numpy(bool),5)
                    current["platform_complete_null"]=np.repeat(pform.platform_complete_null.to_numpy(bool),5)
                    current["platform_candidate_exhausted"]=np.repeat(pform.platform_any_candidate_exhausted.to_numpy(bool),5)
                    replicate_outputs.append(current)
                    if k==30:
                        old=cached.loc[(cached.replicate==replicate)&(cached.pair_seed==seed)].sort_values(["node_order","distance_bin_index"])
                        check=np.nanmax(np.abs(old.r_null.to_numpy()-r_null))
                        check_e=np.nanmax(np.abs(old.E_realization.to_numpy()-current.E_realization.to_numpy()))
                        k30_checks.append((check,check_e))
            block=pd.concat(replicate_outputs,ignore_index=True); block.insert(0,"subject_id",subject)
            table=pa.Table.from_pandas(block,preserve_index=False)
            if writer is None: writer=pq.ParquetWriter(temporary,table.schema,compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None: writer.close()
    maximum=np.nanmax(np.asarray(k30_checks),axis=0)
    if np.any(maximum>2e-6): raise RuntimeError(f"K30 replay mismatch: {maximum}")
    temporary.replace(out/"local_K_realizations.parquet")
    report={"status":"complete","subject_id":subject,"local_k":list(K_VALUES),"pair_seeds":list(PAIR_SEEDS),
            "replicates":20,"replicate_seeds_reused_exactly":True,"platform_cache_reused":True,
            "maximum_k30_r_null_absolute_error":float(maximum[0]),
            "maximum_k30_E_absolute_error":float(maximum[1]),"runtime_seconds":time.time()-started,
            "max_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    (out/"completion.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="subject_id"},indent=2))


if __name__=="__main__": main()
