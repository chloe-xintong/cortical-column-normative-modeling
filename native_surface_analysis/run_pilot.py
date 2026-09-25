#!/usr/bin/env python3
"""Run a read-only D0097 pilot for input checks and fsaverage6 collisions."""
from pathlib import Path
import argparse
import gzip
import numpy as np
import pandas as pd
from .common.config import SUBJECTS_DIR, PROFILES_ROOT, FSAVERAGE6_MAPPING_ROOT
from .common.io import mapping_files, mapping_frame, profile_path, read_md_profile, read_surface, read_aparc

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--subject", default="D0097")
    ap.add_argument("--output-dir", type=Path, default=Path("/mnt/newStor/paros/paros_WORK/xintong/results/native_surface_analysis/pilot_D0097"))
    args=ap.parse_args()
    rows=[]; groups=[]
    total=sum(len(mapping_files(FSAVERAGE6_MAPPING_ROOT,args.subject,h)) for h in ("lh","rh"))
    processed=0
    for hemi in ("lh","rh"):
        white=read_surface(SUBJECTS_DIR,args.subject,hemi,"white")
        pial=read_surface(SUBJECTS_DIR,args.subject,hemi,"pial")
        inflated=read_surface(SUBJECTS_DIR,args.subject,hemi,"inflated")
        native_labels,names=read_aparc(SUBJECTS_DIR/args.subject,hemi)
        if not (len(white)==len(pial)==len(inflated)==len(native_labels)):
            raise ValueError(f"{hemi}: native surface/annotation vertex mismatch")
        files=mapping_files(FSAVERAGE6_MAPPING_ROOT,args.subject,hemi)
        for path in files:
            region=path.name.split(f"{args.subject}_{hemi}_",1)[1].removesuffix("_native_to_fsaverage.csv.gz")
            frame=mapping_frame(path)
            profile=read_md_profile(profile_path(PROFILES_ROOT,args.subject,hemi,region))
            valid=frame[frame.column_valid.astype(str).str.lower().isin(["true","1","1.0"])].copy()
            if len(profile) != int(frame.cleaned_region_row.max()+1 if len(frame) else 0):
                # Empty profiles are valid; otherwise mapping rows must point to them.
                if len(profile) and "cleaned_region_row" in frame:
                    raise ValueError(f"{hemi}/{region}: profile rows do not match mapping")
            if (valid.native_vertex_id<0).any() or (valid.native_vertex_id>=len(white)).any():
                raise ValueError(f"{hemi}/{region}: native vertex out of bounds")
            counts=valid.template_vertex_id.value_counts()
            for vertex,n in counts[counts>=2].items():
                members=valid[valid.template_vertex_id.eq(vertex)]
                profiles=[]
                for idx in members.cleaned_region_row.astype(int):
                    if 0 <= idx < len(profile) and np.isfinite(profile[idx]).all(): profiles.append(profile[idx])
                pair_mae=[]
                for i in range(len(profiles)):
                    for j in range(i+1,len(profiles)): pair_mae.append(float(np.mean(np.abs(profiles[i]-profiles[j]))))
                groups.append({"subject_id":args.subject,"hemisphere":hemi,"region":region,"template_vertex_id":int(vertex),"n_columns_mapped":int(n),"n_profiles_valid":len(profiles),"profile_pair_mae_median":float(np.median(pair_mae)) if pair_mae else np.nan,"native_region_count":int(members.region.nunique())})
            rows.append({"subject_id":args.subject,"hemisphere":hemi,"region":region,"mapping_rows":len(frame),"valid_rows":len(valid),"md_rows":len(profile),"native_vertices":len(white)})
            processed += 1
            if processed % 10 == 0 or processed == total:
                pct=100*processed/max(1,total); width=30; filled=int(width*pct/100)
                print(f"[{'#'*filled}{'-'*(width-filled)}] {pct:6.2f}% ({processed}/{total})", flush=True)
    out=args.output_dir; (out/"tables").mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"tables/input_check_summary.csv",index=False)
    pd.DataFrame(groups).to_csv(out/"tables/pilot_collision_group_metrics.csv",index=False)
    print(f"validated {len(rows)} subject-region files; collision groups={len(groups)}")
    print(f"output={out}")
if __name__=="__main__": main()
