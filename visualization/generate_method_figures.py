#!/usr/bin/env python3
"""Generate publication-ready cortical-column method illustrations from real data.

Description
-----------
Generate publication-ready cortical-column method illustrations from real data.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--subjects-dir`` command-line input
- ``--subject-id`` command-line input
- ``--columns-npz`` command-line input
- ``--md-profiles-npz`` command-line input
- ``--hemisphere`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python visualization/generate_method_figures.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FORMATS = ("svg", "pdf", "png")
RED = "#d62728"
COLUMN = "#0072B2"
VERTEX_COLORS = ("#E69F00", "#009E73", "#CC79A7")


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects-dir", type=Path, required=True)
    p.add_argument("--subject-id", required=True)
    p.add_argument("--columns-npz", type=Path, required=True)
    p.add_argument("--md-profiles-npz", type=Path, required=True)
    p.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    p.add_argument("--vertex-id", type=int, help="Template vertex; default: first fully valid profile")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--patch-rings", type=int, default=4)
    p.add_argument("--face-limit", type=int, default=18000)
    return p.parse_args()


def scalar(x) -> str:
    return str(x.item() if isinstance(x, np.ndarray) else x)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    v, f = fsio.read_geometry(str(path))
    return np.asarray(v, float), np.asarray(f, np.int64)


def choose_vertex(c: dict, m: dict, requested: int | None) -> int:
    valid = c["mapping_valid"].astype(bool) & np.isfinite(c["column_points"]).all((1, 2))
    if "column_valid" in c:
        valid &= c["column_valid"].astype(bool)
    valid &= np.isfinite(m["profiles"]).all(1)
    if "column_valid" in m:
        valid &= m["column_valid"].astype(bool)
    if requested is not None:
        if requested < 0 or requested >= len(valid) or not valid[requested]:
            raise ValueError(f"Requested vertex {requested} is not a fully valid 21-point MD column")
        return requested
    ids = np.flatnonzero(valid)
    if not len(ids):
        raise ValueError("No fully valid 21-point MD column was found")
    # Avoid a medial-wall-adjacent first row and select a reproducible, mid-list example.
    return int(ids[len(ids) // 2])


def neighbourhood(faces: np.ndarray, seeds: np.ndarray, rings: int) -> np.ndarray:
    vertices = set(map(int, seeds))
    selected = np.zeros(len(faces), bool)
    for _ in range(rings + 1):
        hit = np.any(np.isin(faces, list(vertices)), axis=1)
        selected |= hit
        vertices.update(np.unique(faces[hit]).tolist())
    return faces[selected]


def normalize(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    return (points - np.asarray(center, float)) / float(radius)


def decimate(faces: np.ndarray, limit: int) -> np.ndarray:
    return faces[::max(1, int(np.ceil(len(faces) / limit)))]


def orient(ax, focus: np.ndarray) -> None:
    u = focus / np.linalg.norm(focus)
    ax.view_init(elev=np.degrees(np.arcsin(np.clip(u[2], -1, 1))), azim=np.degrees(np.arctan2(u[1], u[0])) - 90)


def equal_limits(ax, xyz: np.ndarray, pad: float = .15) -> None:
    lo, hi = xyz.min(0), xyz.max(0)
    radius = max((hi - lo).max() / 2, .25) * (1 + pad)
    mid = (lo + hi) / 2
    ax.set_xlim(mid[0]-radius, mid[0]+radius); ax.set_ylim(mid[1]-radius, mid[1]+radius); ax.set_zlim(mid[2]-radius, mid[2]+radius)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()


def mesh(ax, vertices, faces, color="#e2e2e2", alpha=.55, lw=.08):
    ax.plot_trisurf(*vertices.T, triangles=faces, color=color, edgecolor="#a6a6a6", linewidth=lw, alpha=alpha, shade=False)
    ax.set_axis_off(); ax.set_box_aspect((1, 1, 1))


def sphere_panel(fig, spec, target, target_faces, source, source_faces, vid, tri_ids, mapped, weights, face_limit, compact=False):
    sub = spec.subgridspec(1, 2, wspace=-.20 if compact else .10)
    a = fig.add_subplot(sub[0], projection="3d"); b = fig.add_subplot(sub[1], projection="3d")
    mesh(a, target, decimate(target_faces, face_limit)); mesh(b, source, decimate(source_faces, face_limit))
    tp = target[vid]
    a.scatter(*tp, s=24, c=RED, edgecolors="white", linewidths=.5, depthshade=False, zorder=20)
    tri = source[tri_ids]
    b.add_collection3d(Poly3DCollection([tri], facecolor="#c9dce5" if compact else "#f6d9a7", edgecolor="#526d7a" if compact else "#8c6d31", linewidth=1.2, alpha=.92))
    if not compact:
        for i, (point, color) in enumerate(zip(tri, VERTEX_COLORS), 1):
            b.scatter(*point, s=27, color=color, edgecolors="white", linewidths=.5, depthshade=False)
    lifted = mapped + .018 * mapped / np.linalg.norm(mapped)
    if not compact:
        b.scatter(*lifted, s=30, c=RED, edgecolors="white", linewidths=.6, depthshade=False, zorder=30)
    # Orient each registered sphere around its own corresponding position so
    # the two red markers occupy the same screen location.
    orient(a, tp); orient(b, mapped)
    if compact:
        a.text2D(.5, .86, "Fsaverage sphere", transform=a.transAxes, ha="center", fontsize=10)
        b.text2D(.5, .86, "Subject registered sphere", transform=b.transAxes, ha="center", fontsize=10)
        # Local inset makes the mapped triangle legible without pretending it
        # is large at the whole-sphere scale.
        projected = proj3d.proj_transform(*mapped, b.get_proj())
        display_xy = b.transData.transform(projected[:2])
        mapped_axes = b.transAxes.inverted().transform(display_xy)
        inset_size = .14
        inset_left = float(np.clip(mapped_axes[0] - inset_size/2, .04, 1-inset_size-.04))
        inset_bottom = float(np.clip(mapped_axes[1] - inset_size/2, .04, 1-inset_size-.04))
        inset = b.inset_axes([inset_left, inset_bottom, inset_size, inset_size])
        centered = tri - tri.mean(axis=0)
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        tri_2d = centered @ basis[:2].T
        mapped_2d = (mapped - tri.mean(axis=0)) @ basis[:2].T
        inset.fill(tri_2d[:, 0], tri_2d[:, 1], color="#b9d6e2", alpha=.58, ec="#405a66", lw=1.1)
        inset.scatter(tri_2d[:, 0], tri_2d[:, 1], s=15, c=VERTEX_COLORS, edgecolors="white", linewidths=.4, zorder=3)
        inset.scatter(mapped_2d[0], mapped_2d[1], s=18, c=RED, edgecolors="white", linewidths=.45, zorder=4)
        radius = max(np.linalg.norm(tri_2d - mapped_2d, axis=1).max() * 1.22, 1e-6)
        inset.set_xlim(mapped_2d[0]-radius, mapped_2d[0]+radius)
        inset.set_ylim(mapped_2d[1]-radius, mapped_2d[1]+radius)
        inset.set_aspect("equal"); inset.set_axis_off()
    else:
        a.set_title("Fsaverage sphere", fontsize=10, pad=-2)
        b.set_title("Subject registered sphere", fontsize=10, pad=-2)
    # Keep the process arrow in the whitespace between the two axes.
    left_box, right_box = a.get_position(), b.get_position()
    arrow_mid = (left_box.x1 + right_box.x0) / 2
    arrow_y = (left_box.y0 + left_box.y1) / 2
    fig.add_artist(FancyArrowPatch((arrow_mid-.018, arrow_y), (arrow_mid+.018, arrow_y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=11, color="#555555", lw=1.1))
    if not compact:
        b.text2D(.5, .08, r"$v_1$   $v_2$   $v_3$   " + r"$\bullet$ mapped point", transform=b.transAxes, ha="center", fontsize=8, color="#444444")
        b.text2D(.5, .01, rf"$w_1={weights[0]:.3f},\ w_2={weights[1]:.3f},\ w_3={weights[2]:.3f}$"+"\n"+r"$w_1+w_2+w_3=1$", transform=b.transAxes, ha="center", fontsize=8)
    return a, b


def ribbon_panel(ax, white, pial, patch, tri_ids, weights, points, compact=False):
    display_white, display_pial, display_points = white, pial, points
    if compact:
        local = np.vstack((white[np.unique(patch)], pial[np.unique(patch)]))
        shift = np.array([.36 * np.ptp(local, axis=0).max(), 0., 0.])
        display_pial = pial - shift
        display_white = white + shift
        display_points = np.linspace(points[0] - shift, points[-1] + shift, 21)
    background_alpha = .18 if compact else .34
    ax.add_collection3d(Poly3DCollection(display_pial[patch], facecolor="#8db8c7", edgecolor="#638692", linewidth=.15, alpha=background_alpha))
    ax.add_collection3d(Poly3DCollection(display_white[patch], facecolor="#c6b4a5", edgecolor="#82766d", linewidth=.15, alpha=background_alpha))
    triangle_surfaces = ((display_pial, "#74add1"), (display_white, "#d9a66f"))
    for surf, triangle_color in triangle_surfaces:
        triangle_xyz = surf[tri_ids]
        if compact:
            center = triangle_xyz.mean(axis=0)
            triangle_xyz = center + 2.4 * (triangle_xyz - center)
        ax.add_collection3d(Poly3DCollection([triangle_xyz], facecolor=triangle_color, edgecolor="#303030", linewidth=1.8 if compact else 1.0, alpha=.95 if compact else .75))
        for vertex_index, (xyz, color) in enumerate(zip(triangle_xyz, VERTEX_COLORS)):
            ax.scatter(*xyz, s=25, color=color, edgecolors="white", linewidths=.4, depthshade=False)
            if compact:
                label_xyz = center + 1.42 * (xyz - center)
                ax.text(
                    *label_xyz, rf"$v_{vertex_index+1}$", fontsize=6.2,
                    color="#202020", zorder=1000,
                )
    ax.plot(*display_points.T, color=COLUMN, lw=2.6, zorder=15)
    if not compact:
        ax.scatter(*display_points.T, color=COLUMN, s=22, edgecolors="white", linewidths=.4, depthshade=False, zorder=20)
        ax.scatter(*display_points[[0, -1]].T, color=("#2b8cbe", "#8c510a"), s=36, edgecolors="white", linewidths=.6, depthshade=False)
    if not compact:
        ax.text2D(.03, .12, "0 = pial", transform=ax.transAxes, fontsize=8, color="#2b8cbe")
        ax.text2D(.78, .12, "1 = white", transform=ax.transAxes, fontsize=8, color="#8c510a")
    all_xyz = np.vstack((display_white[np.unique(patch)], display_pial[np.unique(patch)], display_points))
    equal_limits(ax, all_xyz, .06)
    # View perpendicular to the pial-white vector so the two endpoints and
    # surfaces are maximally separated in projection without moving geometry.
    if compact:
        ax.view_init(elev=14, azim=-90)
    else:
        column = display_points[-1] - display_points[0]
        column /= np.linalg.norm(column)
        reference = np.array([0., 0., 1.]) if abs(column[2]) < .85 else np.array([0., 1., 0.])
        view = np.cross(column, reference); view /= np.linalg.norm(view)
        ax.view_init(elev=np.degrees(np.arcsin(view[2])), azim=np.degrees(np.arctan2(view[1], view[0])) - 90)
    if compact:
        ax.text2D(.5, .90, "Column generation", transform=ax.transAxes, ha="center", fontsize=9)
        ax.text2D(.5, .83, "21 points: pial → white", transform=ax.transAxes, ha="center", fontsize=9)
        ax.text2D(.08, .19, "0 = pial", transform=ax.transAxes, ha="left", fontsize=8, color="#2b8cbe")
        ax.text2D(.92, .19, "1 = white", transform=ax.transAxes, ha="right", fontsize=8, color="#9a5a16")
    else:
        ax.set_title("Barycentric endpoint reconstruction\n21 equally spaced points from pial to white", fontsize=10)
        ax.text2D(.5, -.02, r"$p=\sum_{i=1}^{3}w_i p_i,\qquad q=\sum_{i=1}^{3}w_i q_i$", transform=ax.transAxes, ha="center", fontsize=10)


def save(fig, stem: Path) -> list[str]:
    outputs=[]
    for ext in FORMATS:
        path=stem.with_suffix("."+ext); kwargs={"bbox_inches":"tight", "facecolor":"white"}
        if ext == "png": kwargs["dpi"] = 300
        fig.savefig(path, **kwargs); outputs.append(str(path.resolve()))
    plt.close(fig); return outputs


def main() -> None:
    args=arguments(); args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"], "font.size":9, "svg.fonttype":"none", "pdf.fonttype":42})
    c, m = load_npz(args.columns_npz), load_npz(args.md_profiles_npz)
    for name, data in (("columns", c), ("MD profiles", m)):
        if scalar(data["subject_id"]) != args.subject_id or scalar(data["hemisphere"]) != args.hemisphere:
            raise ValueError(f"{name} identity does not match CLI subject/hemisphere")
    vid=choose_vertex(c,m,args.vertex_id); points=np.asarray(c["column_points"][vid],float); depths=np.asarray(c["depth_fractions"],float); md=np.asarray(m["profiles"][vid],float)
    if points.shape != (21,3) or depths.shape != (21,) or md.shape != (21,) or scalar(c["depth_direction"]) != "pial_to_white":
        raise ValueError("Inputs must contain exactly 21 samples ordered pial_to_white")
    root=args.subjects_dir/args.subject_id; hemi=args.hemisphere
    source, sf=surface(root/"surf"/f"{hemi}.sphere.reg"); white,wf=surface(root/"surf"/f"{hemi}.white"); pial,pf=surface(root/"surf"/f"{hemi}.pial")
    target,tf=surface(Path(scalar(c["target_sphere_path"])))
    if not (np.array_equal(sf,wf) and np.array_equal(wf,pf)): raise ValueError("Native surfaces do not share topology")
    target=normalize(target,c["target_sphere_center"],c["target_sphere_radius"]); source=normalize(source,c["source_sphere_center"],c["source_sphere_radius"])
    tri_ids=np.asarray(c["native_triangle_vertex_ids"][vid],int); weights=np.asarray(c["barycentric_weights"][vid],float); mapped=np.asarray(c["closest_point_on_source_sphere"][vid],float)
    reconstructed_pial = weights @ pial[tri_ids]
    reconstructed_white = weights @ white[tri_ids]
    if not (np.allclose(reconstructed_pial, points[0], atol=1e-6) and np.allclose(reconstructed_white, points[-1], atol=1e-6)):
        raise ValueError("Saved endpoints do not match barycentric reconstruction from the native triangles")
    patch=neighbourhood(wf,tri_ids,args.patch_rings); prefix=f"{args.subject_id}_{hemi}_fsavg-{vid}"
    outputs=[]
    fig=plt.figure(figsize=(8.0,3.7)); sphere_panel(fig,fig.add_gridspec(1,1)[0],target,tf,source,sf,vid,tri_ids,mapped,weights,args.face_limit); fig.suptitle("fsaverage sphere → Subject registered sphere",fontsize=12); outputs+=save(fig,args.output_dir/f"{prefix}_01_spherical_mapping")
    fig=plt.figure(figsize=(5.0,4.5)); ax=fig.add_subplot(projection="3d"); ribbon_panel(ax,white,pial,patch,tri_ids,weights,points); outputs+=save(fig,args.output_dir/f"{prefix}_02_endpoint_column")
    fig,ax=plt.subplots(figsize=(5.0,3.2)); ax.plot(depths,md,color=COLUMN,lw=1.8,marker="o",ms=3.5); ax.set(xlim=(0,1),xlabel="Normalized cortical depth",ylabel="MD (mm²/s)",title="Depth-resolved MD profile"); ax.text(0,-.22,"Pial",transform=ax.get_xaxis_transform(),ha="center"); ax.text(1,-.22,"White",transform=ax.get_xaxis_transform(),ha="center"); ax.grid(alpha=.2); fig.tight_layout(); outputs+=save(fig,args.output_dir/f"{prefix}_03_md_profile")
    compact_patch=neighbourhood(wf,tri_ids,max(1,args.patch_rings-2))
    fig=plt.figure(figsize=(7.4,2.55)); gs=fig.add_gridspec(1,2,width_ratios=(2.05,1),wspace=-.18,left=.01,right=.99,bottom=.13,top=.97); sphere_panel(fig,gs[0],target,tf,source,sf,vid,tri_ids,mapped,weights,args.face_limit,compact=True); ax=fig.add_subplot(gs[1],projection="3d"); ribbon_panel(ax,white,pial,compact_patch,tri_ids,weights,points,compact=True)
    fig.text(.335, .22, r"$\mathbf{x}=w_1\mathbf{v}_1+w_2\mathbf{v}_2+w_3\mathbf{v}_3$", ha="center", va="center", fontsize=10)
    outputs+=save(fig,args.output_dir/f"{prefix}_04_mapping_to_column")
    meta={"subject_id":args.subject_id,"hemisphere":hemi,"template_vertex_id":vid,"native_triangle_vertex_ids":tri_ids.tolist(),"barycentric_weights":weights.tolist(),"depth_direction":"pial_to_white","depth_point_count":21,"md_profile_real_data":True,"md_units":"mm^2/s","columns_npz":str(args.columns_npz.resolve()),"md_profiles_npz":str(args.md_profiles_npz.resolve()),"outputs":outputs}
    (args.output_dir/f"{prefix}_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))

if __name__ == "__main__": main()
