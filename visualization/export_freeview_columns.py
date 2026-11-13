#!/usr/bin/env python3
"""Export selected native-surface-RAS columns as FreeSurfer tube surfaces.

Description
-----------
Export selected native-surface-RAS columns as FreeSurfer tube surfaces.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--columns-npz`` command-line input
- ``--md-profiles-npz`` command-line input
- ``--reference-t1`` command-line input
- ``--white-surface`` command-line input
- ``--pial-surface`` command-line input
- ``--vertex-ids`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python visualization/export_freeview_columns.py --help
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import numpy as np

def args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--columns-npz",type=Path,required=True); p.add_argument("--md-profiles-npz",type=Path,required=True); p.add_argument("--reference-t1",type=Path,required=True); p.add_argument("--white-surface",type=Path,required=True); p.add_argument("--pial-surface",type=Path,required=True); p.add_argument("--vertex-ids",type=int,nargs="+",required=True); p.add_argument("--main-vertex-id",type=int); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--radius-mm",type=float,default=.08); p.add_argument("--sides",type=int,default=10); p.add_argument("--md-range",type=float,nargs=2); return p.parse_args()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def tube(points, values, radius, sides):
    verts=[]
    for i,p in enumerate(points):
        tangent=points[min(i+1,len(points)-1)]-points[max(i-1,0)]; tangent/=np.linalg.norm(tangent)
        ref=np.array([1.,0,0]) if abs(tangent[0])<.8 else np.array([0.,1,0]); u=np.cross(tangent,ref); u/=np.linalg.norm(u); v=np.cross(tangent,u)
        for angle in np.linspace(0,2*np.pi,sides,endpoint=False): verts.append(p+radius*(np.cos(angle)*u+np.sin(angle)*v))
    faces=[]
    for i in range(len(points)-1):
        for j in range(sides):
            a=i*sides+j;b=i*sides+(j+1)%sides;c=(i+1)*sides+j;d=(i+1)*sides+(j+1)%sides;faces.extend(((a,c,b),(b,c,d)))
    return np.asarray(verts,np.float32),np.asarray(faces,np.int32),np.repeat(values,sides).astype(np.float32)
def main():
    a=args(); a.output_dir.mkdir(parents=True,exist_ok=True); c,m=load(a.columns_npz),load(a.md_profiles_npz); main=a.main_vertex_id if a.main_vertex_id is not None else a.vertex_ids[0]
    if main not in a.vertex_ids: raise ValueError("--main-vertex-id must be included in --vertex-ids")
    for p in (a.reference_t1,a.white_surface,a.pial_surface):
        if not p.is_file(): raise FileNotFoundError(p)
    all_md=np.concatenate([m["profiles"][v] for v in a.vertex_ids]); vmin,vmax=a.md_range or (float(np.nanpercentile(all_md,2)),float(np.nanpercentile(all_md,98)))
    outputs=[]
    for vid in a.vertex_ids:
        pts=np.asarray(c["column_points"][vid],float); md=np.asarray(m["profiles"][vid],float)
        if pts.shape!=(21,3) or md.shape!=(21,) or not np.isfinite(pts).all() or not np.isfinite(md).all(): raise ValueError(f"Vertex {vid} is not a finite 21-point column/profile")
        verts,faces,overlay=tube(pts,md,a.radius_mm*(1.35 if vid==main else 1),a.sides)
        stem=a.output_dir/f"column_fsavg-{vid}"; fsio.write_geometry(str(stem.with_suffix('.surf')),verts,faces,volume_info=None); fsio.write_morph_data(str(stem.with_suffix('.md')),overlay); outputs.extend((str(stem.with_suffix('.surf').resolve()),str(stem.with_suffix('.md').resolve())))
    norm=plt.Normalize(vmin,vmax); sm=plt.cm.ScalarMappable(norm=norm,cmap="viridis"); fig,ax=plt.subplots(figsize=(5,.55)); fig.colorbar(sm,cax=ax,orientation="horizontal",label="MD (mm²/s)"); fig.savefig(a.output_dir/"md_colorbar.png",dpi=300,bbox_inches="tight",facecolor="white"); fig.savefig(a.output_dir/"md_colorbar.svg",bbox_inches="tight",facecolor="white"); plt.close(fig)
    out=a.output_dir.resolve()
    colored=[f"-f {out/f'column_fsavg-{v}.surf'}:overlay={out/f'column_fsavg-{v}.md'}:overlay_threshold={vmin},{vmax}:overlay_color=colorwheel" for v in a.vertex_ids]
    mono=[f"-f {out/f'column_fsavg-{v}.surf'}:color=0,114,178" for v in a.vertex_ids]
    base=f"freeview -v {a.reference_t1}:grayscale=0,255 -f {a.white_surface}:edgecolor=white -f {a.pial_surface}:edgecolor=cyan"
    commands={"md_colored":base+" "+" ".join(colored),"monochrome":base+" "+" ".join(mono)}
    (a.output_dir/"freeview_commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n# MD-coloured columns\n"+commands["md_colored"]+"\n\n# Monochrome columns\n"+commands["monochrome"]+"\n")
    meta={"coordinate_space":"FreeSurfer native surface RAS (tkReg RAS)","reference_t1":str(a.reference_t1.resolve()),"white_surface":str(a.white_surface.resolve()),"pial_surface":str(a.pial_surface.resolve()),"template_vertex_ids":a.vertex_ids,"main_template_vertex_id":main,"depth_direction":"pial_to_white","points_per_column":21,"radius_mm":a.radius_mm,"tube_sides":a.sides,"md_color_range_mm2_s":[vmin,vmax],"commands":commands,"outputs":outputs}
    (a.output_dir/"freeview_metadata.json").write_text(json.dumps(meta,indent=2)+"\n"); print(json.dumps(meta,indent=2))
if __name__=="__main__":main()
