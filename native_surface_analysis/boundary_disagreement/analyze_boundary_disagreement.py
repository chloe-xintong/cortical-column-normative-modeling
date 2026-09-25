#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_annot, read_geometry, write_annot
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from ..common.config import SUBJECTS_DIR, FSAVERAGE6_MAPPING_ROOT
from ..common.io import mapping_files, mapping_frame

def graph_distance(labels):
    n=len(labels); edges=set()
    return None

def boundary_hops(labels, faces):
    n=len(labels); edges=set()
    for tri in faces.astype(int):
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            if a!=b: edges.add((min(a,b),max(a,b)))
    if not edges: return np.full(n,np.inf), np.zeros(n,bool)
    e=np.asarray(list(edges),dtype=int)
    row=np.r_[e[:,0],e[:,1]]; col=np.r_[e[:,1],e[:,0]]
    graph=coo_matrix((np.ones(len(row)),(row,col)),shape=(n,n)).tocsr()
    valid=np.array([x not in {"unknown","corpuscallosum","medialwall",""} for x in labels])
    boundary=np.zeros(n,bool)
    for a,b in e:
        if valid[a] and valid[b] and labels[a]!=labels[b]:
            boundary[a]=boundary[b]=True
    dist=dijkstra(graph,indices=np.flatnonzero(boundary),min_only=True)
    return np.asarray(dist), boundary

def read_names(path):
    labels,_,names=read_annot(str(path))
    names=[x.decode() if isinstance(x,bytes) else str(x) for x in names]
    return np.asarray(labels), names

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--subject",default="D0097")
    ap.add_argument("--template",choices=["fsaverage","fsaverage6"],default="fsaverage6")
    ap.add_argument("--output-dir",type=Path,default=Path("/mnt/newStor/paros/paros_WORK/xintong/results/native_surface_analysis"))
    args=ap.parse_args()
    mapping_root=FSAVERAGE6_MAPPING_ROOT if args.template=="fsaverage6" else Path("/mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage")
    template_dir=Path("/home/apps/freesurfer/subjects")/args.template
    out=args.output_dir/f"boundary_disagreement_{args.template}_{args.subject}"; (out/"tables").mkdir(parents=True,exist_ok=True)
    summaries=[]; hops=[]
    for hemi in ("lh","rh"):
        native_dir=SUBJECTS_DIR/args.subject
        nlabels,names=read_names(native_dir/"label"/f"{hemi}.aparc.annot")
        _,faces=read_geometry(str(native_dir/"surf"/f"{hemi}.white"))
        nname={i:x for i,x in enumerate(names)}; native_names=np.array([nname.get(int(x),"") for x in nlabels])
        dist,boundary=boundary_hops(native_names,faces)
        overlay=np.zeros(len(native_names),dtype=np.int8); overlay[boundary]=3
        tlabels,tnames=read_names(template_dir/"label"/f"{hemi}.aparc.annot")
        tname={i:x for i,x in enumerate(tnames)}
        for number,path in enumerate(mapping_files(mapping_root,args.subject,hemi),1):
            region=path.name.split(f"{args.subject}_{hemi}_",1)[1].removesuffix("_native_to_fsaverage.csv.gz")
            frame=mapping_frame(path)
            frame=frame[frame.column_valid.astype(str).str.lower().isin(["true","1","1.0"])].copy()
            frame=frame[(frame.native_vertex_id>=0)&(frame.native_vertex_id<len(native_names))]
            target_ok=(frame.template_vertex_id>=0)&(frame.template_vertex_id<len(tlabels))
            frame=frame[target_ok].copy()
            native_label=native_names[frame.native_vertex_id.to_numpy()]
            target_label=np.array([tname.get(int(x),"") for x in tlabels[frame.template_vertex_id.to_numpy()]])
            invalid=np.isin(target_label,["","unknown","corpuscallosum","medialwall"])
            disagree=(native_label!=target_label)&~invalid
            vertex_ids=frame.native_vertex_id.to_numpy()
            overlay[vertex_ids[~invalid & ~disagree]]=1
            overlay[vertex_ids[disagree]]=2
            d=dist[frame.native_vertex_id.to_numpy()]
            near=np.isfinite(d)&(d<=2)
            internal=np.isfinite(d)&(d>=3)
            denom=len(frame)
            summaries.append({"subject_id":args.subject,"template":args.template,"hemisphere":hemi,"region":region,"valid_mappings":denom,"disagreements":int(disagree.sum()),"disagreement_pct":100*disagree.sum()/denom if denom else np.nan,"boundary_near_mappings":int(near.sum()),"boundary_near_disagreements":int((disagree&near).sum()),"boundary_near_disagreement_pct":100*(disagree&near).sum()/near.sum() if near.sum() else np.nan,"internal_mappings":int(internal.sum()),"internal_disagreements":int((disagree&internal).sum()),"internal_disagreement_pct":100*(disagree&internal).sum()/internal.sum() if internal.sum() else np.nan,"invalid_target_labels":int(invalid.sum()),"unknown_native_labels":int(np.isin(native_label,["","unknown","corpuscallosum","medialwall"]).sum()),"disagreements_near_boundary_pct":100*(disagree&near).sum()/disagree.sum() if disagree.sum() else np.nan})
            for level in (0,1,2,3):
                sel=np.isfinite(d)&(d==level); hops.append({"subject_id":args.subject,"template":args.template,"hemisphere":hemi,"region":region,"hop_bin":str(level),"valid_mappings":int(sel.sum()),"disagreements":int((disagree&sel).sum()),"disagreement_pct":100*(disagree&sel).sum()/sel.sum() if sel.sum() else np.nan})
            sel=np.isfinite(d)&(d>=4); hops.append({"subject_id":args.subject,"template":args.template,"hemisphere":hemi,"region":region,"hop_bin":">=4","valid_mappings":int(sel.sum()),"disagreements":int((disagree&sel).sum()),"disagreement_pct":100*(disagree&sel).sum()/sel.sum() if sel.sum() else np.nan})
            if number%10==0: print(f"{args.template} {hemi}: {number}/{len(mapping_files(mapping_root,args.subject,hemi))}",flush=True)
        colors=np.array([[180,180,180,255],[70,170,90,255],[210,50,50,255],[0,100,70,255]],dtype=np.int32)
        (out/"overlays").mkdir(parents=True,exist_ok=True)
        write_annot(str(out/"overlays"/f"{args.subject}_{hemi}_boundary_disagreement.annot"),overlay,colors,[b"unmapped",b"agreement",b"disagreement",b"boundary"],fill_ctab=True)
    pd.DataFrame(summaries).to_csv(out/"tables/boundary_agreement_summary.csv",index=False)
    pd.DataFrame(hops).to_csv(out/"tables/boundary_agreement_by_hop.csv",index=False)
    print(f"wrote {len(summaries)} summaries to {out}")
if __name__=="__main__": main()
