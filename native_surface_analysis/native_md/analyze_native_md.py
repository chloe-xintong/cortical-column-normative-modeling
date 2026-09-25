#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from ..common.config import FSAVERAGE6_MAPPING_ROOT
from ..common.io import mapping_files, mapping_frame, read_surface
from ..common.config import SUBJECTS_DIR

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--subject",default="D0097")
    ap.add_argument("--regions",nargs="*",default=["precentral","postcentral","superiorfrontal","supramarginal"])
    ap.add_argument("--output-dir",type=Path,default=Path("/mnt/newStor/paros/paros_WORK/xintong/results/native_surface_analysis"))
    args=ap.parse_args(); out=args.output_dir/f"native_md_{args.subject}"; (out/"tables").mkdir(parents=True,exist_ok=True); (out/"overlays").mkdir(parents=True,exist_ok=True)
    rows=[]; depth_idx={"25pct":5,"50pct":10,"75pct":15}
    for hemi in ("lh","rh"):
        n=len(read_surface(SUBJECTS_DIR,args.subject,hemi,"white")); files=[p for p in mapping_files(FSAVERAGE6_MAPPING_ROOT,args.subject,hemi) if any(f"_{r}_" in p.name for r in args.regions)]
        for number,path in enumerate(files,1):
            region=path.name.split(f"{args.subject}_{hemi}_",1)[1].removesuffix("_native_to_fsaverage.csv.gz"); frame=mapping_frame(path)
            valid=frame[frame.column_valid.astype(str).str.lower().isin(["true","1","1.0"])].copy()
            if valid.native_vertex_id.duplicated().any(): raise ValueError(f"{hemi}/{region}: duplicate native vertices; refusing overwrite")
            status=np.full(n,np.nan)
            for label,idx in depth_idx.items():
                col=f"MD_depth_{idx}"
                if col not in valid: raise ValueError(f"Missing {col} in {path}")
                vals=pd.to_numeric(valid[col],errors="coerce").to_numpy()
                ids=valid.native_vertex_id.to_numpy()
                status[ids]=vals
                good=np.isfinite(vals)
                rows.append({"subject_id":args.subject,"hemisphere":hemi,"region":region,"depth_label":label,"depth_index":idx,"n_vertices":int(good.sum()),"missing_pct":100*(~good).sum()/len(vals) if len(vals) else np.nan,"mean_md":float(np.nanmean(vals)) if good.any() else np.nan,"sd_md":float(np.nanstd(vals,ddof=1)) if good.sum()>1 else np.nan,"median_md":float(np.nanmedian(vals)) if good.any() else np.nan})
                m=np.isfinite(status); mean=float(np.nanmean(status[m])) if m.any() else np.nan; centered=status-mean
                nib.save(nib.freesurfer.mghformat.MGHImage(status.astype(np.float32)[:,None,None],np.eye(4)),str(out/"overlays"/f"{args.subject}_{hemi}_{region}_{label}_md.mgh"))
                nib.save(nib.freesurfer.mghformat.MGHImage(centered.astype(np.float32)[:,None,None],np.eye(4)),str(out/"overlays"/f"{args.subject}_{hemi}_{region}_{label}_md_centered.mgh"))
            print(f"{args.subject} {hemi}: {number}/{len(files)} {region}",flush=True)
    pd.DataFrame(rows).to_csv(out/"tables/native_md_summary.csv",index=False); print(f"wrote {len(rows)} depth summaries to {out}")
if __name__=="__main__": main()
