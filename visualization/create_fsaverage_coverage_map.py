#!/usr/bin/env python3
"""Create categorical fsaverage coverage overlays from Option-A mappings."""
import argparse, logging
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.freesurfer.io import read_annot, read_geometry, write_annot

LOG = logging.getLogger("coverage_map")
# 0 not_in_analysis (dark gray), 1 uncovered (red), 2 single column
# (light green), 3 collision/conflict (dark green).
COLORS = np.array([[70,70,70,255], [220,60,50,255], [135,210,145,255], [0,105,65,255]], dtype=np.uint8)
NAMES = [b"not_in_analysis", b"uncovered", b"single_column", b"collision"]

def surface_path(root, hemi, name):
    p = Path(root) / "surf" / f"{hemi}.{name}"
    if not p.is_file(): raise FileNotFoundError(f"Missing fsaverage surface: {p}")
    return p

def load_mapping(mapping_root, subject, hemi):
    files = sorted((Path(mapping_root)/subject/hemi/"regions").glob(f"{subject}_{hemi}_*_native_to_fsaverage.csv.gz"))
    if not files: raise FileNotFoundError(f"No mapping files for {subject} {hemi}")
    frames = [pd.read_csv(f, usecols=["subject_id","hemisphere","region","native_vertex_id","column_valid","template_vertex_id"]) for f in files]
    x = pd.concat(frames, ignore_index=True)
    x = x[x.column_valid.astype(str).str.lower().isin(["true","1","1.0"])].copy()
    for c in ["native_vertex_id","template_vertex_id"]: x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["native_vertex_id","template_vertex_id"])
    x[["native_vertex_id","template_vertex_id"]] = x[["native_vertex_id","template_vertex_id"]].astype(int)
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--subject-id", required=True); ap.add_argument("--hemisphere", choices=["lh","rh"], required=True)
    ap.add_argument("--mapping-root", required=True); ap.add_argument("--fsaverage-dir", required=True); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a=ap.parse_args(); logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    hemi, subject = a.hemisphere, a.subject_id
    annot = surface_path(a.fsaverage_dir, hemi, "aparc.annot") if (Path(a.fsaverage_dir)/"surf"/f"{hemi}.aparc.annot").is_file() else Path(a.fsaverage_dir)/"label"/f"{hemi}.aparc.annot"
    if not annot.is_file(): raise FileNotFoundError(f"Missing annotation: {annot}")
    coords, _ = read_geometry(str(surface_path(a.fsaverage_dir, hemi, "inflated")))
    labels, ctab, names = read_annot(str(annot)); n=len(coords)
    m=load_mapping(a.mapping_root, subject, hemi)
    if (m.template_vertex_id<0).any() or (m.template_vertex_id>=n).any(): raise ValueError("Template vertex ID out of range")
    hit=np.bincount(m.template_vertex_id, minlength=n)
    region_names={z for z in m.region.dropna().astype(str)}
    decoded=[z.decode() if isinstance(z,bytes) else str(z) for z in names]
    # nibabel returns annotation labels as color-table row indices.  They are
    # not the packed IDs stored in ctab[:, 4].  Using ctab[:, 4] here makes
    # every vertex miss its parcel and produces n_analysis_vertices == 0.
    name_by_label={i: decoded[i] for i in range(len(decoded))}
    analysis=np.array([name_by_label.get(int(label), "") in region_names and name_by_label.get(int(label), "") not in {"unknown","corpuscallosum"} for label in labels])
    status=np.zeros(n,dtype=np.int8); status[analysis & (hit==0)]=1; status[analysis & (hit==1)]=2; status[analysis & (hit>=2)]=3
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); stem=f"{subject}_{hemi}_coverage_status"
    summary={"subject_id":subject,"hemisphere":hemi,"n_analysis_vertices":int(analysis.sum()),"n_uncovered_vertices":int((status==1).sum()),"n_unique_covered_vertices":int(((status==2)|(status==3)).sum()),"n_collision_vertices":int((status==3).sum()),"n_mapped_native_columns":int(len(m)),"coverage_percent":float(100*((status==2)|(status==3)).sum()/max(1,analysis.sum())),"collision_vertex_percent":float(100*(status==3).sum()/max(1,analysis.sum()))}
    pd.DataFrame([summary]).to_csv(out/f"{stem}_summary.csv",index=False)
    if a.dry_run: LOG.info("DRY RUN: %s", summary); return
    # ``write_annot`` expects categorical row indices here and packs the RGB
    # values into annotation IDs when ``fill_ctab=True``.
    ctab=COLORS.astype(np.int32)
    write_annot(str(out/f"{stem}.annot"), labels=status, ctab=ctab, names=NAMES, fill_ctab=True)
    # FreeSurfer surface overlays use one value per vertex and require a
    # 3-D MGH shape: (n_vertices, 1, 1).
    nib.save(nib.freesurfer.mghformat.MGHImage(status.astype(np.float32)[:,None,None], np.eye(4)), str(out/f"{stem}.mgh"))
    LOG.info("Saved %s", out/f"{stem}.annot")
    LOG.info("%s", summary)
if __name__ == "__main__": main()
