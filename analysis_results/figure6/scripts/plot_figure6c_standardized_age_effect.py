#!/usr/bin/env python3
"""Render Figure 6C in native ico4 model space and export surface data.

Description
-----------
Render Figure 6C in native ico4 model space and export surface data.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--result-root`` command-line input
- ``--input-dir`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/plot_figure6c_standardized_age_effect.py --help
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import nibabel as nib
from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiMetaData
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import load_surf_data, load_surf_mesh
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


BANDS = ("Outer", "Middle", "Inner")
N_ICO4 = 2562
DISPLAY_MULTIPLIER = 1_000.0
MISSING_COLOR = "#B8B8B8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--age-min", type=int, default=45)
    parser.add_argument("--age-max", type=int, default=76)
    parser.add_argument(
        "--figure-stem", default="Figure6C_depth_stratified_age_association"
    )
    parser.add_argument(
        "--title", default="Depth-stratified standardized age association"
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_ico4_meshes() -> dict[str, dict[str, np.ndarray]]:
    template = fetch_surf_fsaverage(mesh="fsaverage5")
    meshes: dict[str, dict[str, np.ndarray]] = {}
    for hemi, side in (("lh", "left"), ("rh", "right")):
        sphere = load_surf_mesh(template[f"sphere_{side}"]).coordinates
        inflated = load_surf_mesh(template[f"infl_{side}"]).coordinates
        sulc = np.asarray(load_surf_data(template[f"sulc_{side}"]), float)
        if sphere.shape[0] < N_ICO4 or inflated.shape[0] < N_ICO4 or len(sulc) < N_ICO4:
            raise RuntimeError("fsaverage5 template does not contain the nested ico4 vertices")
        faces = ConvexHull(np.asarray(sphere[:N_ICO4], float)).simplices.astype(np.int32)
        if faces.shape != (5120, 3) or len(np.unique(faces)) != N_ICO4:
            raise RuntimeError("Failed to reconstruct the native ico4 face topology")
        meshes[hemi] = {
            "coordinates": np.asarray(inflated[:N_ICO4], float),
            "faces": faces,
            "sulc": sulc[:N_ICO4],
        }
    return meshes


def build_surface_arrays(
    metadata: pd.DataFrame,
    band: pd.DataFrame,
) -> dict[tuple[str, str], np.ndarray]:
    arrays: dict[tuple[str, str], np.ndarray] = {}
    value_column = "depth_band_mean_average_md_change_per_decade"
    for hemi in ("lh", "rh"):
        hemi_meta = metadata.loc[metadata.hemisphere.eq(hemi)]
        ids = hemi_meta.fsaverage_vertex_id.to_numpy(int)
        if len(ids) != len(np.unique(ids)) or np.any(ids < 0) or np.any(ids >= N_ICO4):
            raise RuntimeError(f"Invalid or duplicate fsaverage vertex IDs for {hemi}")
        for depth_band in BANDS:
            part = band.loc[
                band.hemisphere.eq(hemi) & band.depth_band.astype(str).eq(depth_band)
            ].sort_values("node_index")
            expected = hemi_meta.sort_values("node_index")
            if len(part) != len(expected) or not np.array_equal(
                part.node_index.to_numpy(int), expected.node_index.to_numpy(int)
            ):
                raise RuntimeError(f"Node/order mismatch for {hemi} {depth_band}")
            values = np.full(N_ICO4, np.nan, dtype=np.float64)
            values[part.fsaverage_vertex_id.to_numpy(int)] = part[value_column].to_numpy(float)
            arrays[hemi, depth_band] = values
    return arrays


def save_gifti(
    path: Path,
    arrays: dict[tuple[str, str], np.ndarray],
    hemi: str,
    age_min: int,
    age_max: int,
) -> None:
    image = GiftiImage()
    image.meta = GiftiMetaData({
        "AnatomicalStructurePrimary": "CortexLeft" if hemi == "lh" else "CortexRight",
        "ModelSpace": "ico4 vertices indexed by fsaverage_vertex_id (0..2561)",
        "MapMeaning": f"Average MD change per decade, ages {age_min}-{age_max}",
        "RawUnit": "mm^2/s",
        "MissingValueMeaning": "outside fixed analysis mask or excluded for incomplete 186-subject local thickness",
    })
    for band in BANDS:
        data = np.asarray(arrays[hemi, band], dtype=np.float32)
        image.add_gifti_data_array(GiftiDataArray(
            data=data,
            intent="NIFTI_INTENT_NONE",
            meta=GiftiMetaData({"Name": band}),
        ))
    nib.save(image, path)


def save_surface_data(
    output_dir: Path,
    arrays: dict[tuple[str, str], np.ndarray],
    age_min: int,
    age_max: int,
) -> list[str]:
    surface_dir = output_dir / "surface_data"
    surface_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for hemi in ("lh", "rh"):
        gifti = surface_dir / f"figure6c_average_md_change_per_decade_{hemi}.func.gii"
        save_gifti(gifti, arrays, hemi, age_min, age_max)
        files.append(str(gifti.relative_to(output_dir)))
        mask = np.isfinite(arrays[hemi, "Outer"]).astype(np.float32)
        mask_image = GiftiImage(darrays=[GiftiDataArray(
            data=mask,
            intent="NIFTI_INTENT_NONE",
            meta=GiftiMetaData({"Name": "Figure6C eligible analysis mask"}),
        )])
        mask_path = surface_dir / f"figure6c_eligible_mask_{hemi}.shape.gii"
        nib.save(mask_image, mask_path)
        files.append(str(mask_path.relative_to(output_dir)))
        for band in BANDS:
            mgh = surface_dir / f"figure6c_{band.lower()}_{hemi}.mgh"
            image = nib.MGHImage(
                np.asarray(arrays[hemi, band], dtype=np.float32).reshape(-1, 1, 1),
                np.eye(4),
            )
            nib.save(image, mgh)
            files.append(str(mgh.relative_to(output_dir)))
    return files


def render(
    output_dir: Path,
    meshes: dict[str, dict[str, np.ndarray]],
    arrays: dict[tuple[str, str], np.ndarray],
    vmax_raw: float,
    age_min: int,
    age_max: int,
    title: str,
    figure_stem: str,
) -> None:
    cmap = plt.colormaps["RdBu_r"].copy()
    cmap.set_bad(MISSING_COLOR)
    vmax = vmax_raw * DISPLAY_MULTIPLIER
    fig = plt.figure(figsize=(7.09, 5.25), facecolor="white")
    grid = fig.add_gridspec(
        3, 4, left=0.055, right=0.985, bottom=0.145, top=0.83,
        wspace=-0.02, hspace=0.02,
    )
    views = (
        ("lh", "left", "lateral", "Left lateral"),
        ("lh", "left", "medial", "Left medial"),
        ("rh", "right", "lateral", "Right lateral"),
        ("rh", "right", "medial", "Right medial"),
    )
    axes = []
    for row, band in enumerate(BANDS):
        for col, (hemi, nilearn_hemi, view, view_title) in enumerate(views):
            ax = fig.add_subplot(grid[row, col], projection="3d")
            axes.append(ax)
            mesh = (meshes[hemi]["coordinates"], meshes[hemi]["faces"])
            values = arrays[hemi, band] * DISPLAY_MULTIPLIER
            plot_surf_stat_map(
                mesh,
                values,
                hemi=nilearn_hemi,
                view=view,
                bg_map=meshes[hemi]["sulc"],
                bg_on_data=False,
                darkness=0.55,
                cmap=cmap,
                symmetric_cbar=True,
                vmin=-vmax,
                vmax=vmax,
                colorbar=False,
                axes=ax,
                figure=fig,
                title=view_title if row == 0 else None,
            )
            for collection in ax.collections:
                collection.set_rasterized(True)
            if col == 0:
                ax.text2D(
                    -0.08, 0.5, band, transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=9.5,
                    fontweight="bold",
                )
    color_axis = fig.add_axes([0.20, 0.045, 0.60, 0.025])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(-vmax, vmax), cmap=cmap),
        cax=color_axis,
        orientation="horizontal",
        extend="both",
        extendfrac=0.035,
    )
    colorbar.set_label(
        f"Average MD change per decade, ages {age_min}–{age_max} (×10⁻³ mm²/s)",
        fontsize=9,
    )
    colorbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        title,
        x=0.5, y=0.985, ha="center", fontsize=11, fontweight="bold",
    )
    for extension, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(
            output_dir / f"{figure_stem}.{extension}",
            bbox_inches="tight", facecolor="white", **kwargs,
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.age_min >= args.age_max:
        raise ValueError("--age-min must be less than --age-max")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.result_root / "ico4_node_metadata.csv")
    band = pd.read_parquet(args.input_dir / "node_depth_band_standardized_age_effect.parquet")
    if len(metadata) != 4687 or len(band) != 4687 * 3:
        raise RuntimeError("Unexpected node or depth-band table size")
    arrays = build_surface_arrays(metadata, band)
    finite = np.concatenate([
        values[np.isfinite(values)] for values in arrays.values()
    ])
    if not len(finite) or not np.isfinite(finite).all():
        raise RuntimeError("No finite surface effects")
    full_vmax_raw = float(np.max(np.abs(finite)))
    vmax_raw = float(np.percentile(np.abs(finite), 98))
    if not np.isfinite(vmax_raw) or vmax_raw <= 0:
        raise RuntimeError("Invalid symmetric display range")
    if vmax_raw > full_vmax_raw:
        raise RuntimeError("Percentile display limit exceeds the complete finite range")
    meshes = build_ico4_meshes()
    files = save_surface_data(args.output_dir, arrays, args.age_min, args.age_max)
    render(
        args.output_dir, meshes, arrays, vmax_raw, args.age_min, args.age_max,
        args.title, args.figure_stem,
    )

    mapping_rows = []
    for hemi in ("lh", "rh"):
        part = metadata.loc[metadata.hemisphere.eq(hemi)]
        eligible = np.isfinite(arrays[hemi, "Outer"])
        mapping_rows.append({
            "hemisphere": hemi,
            "fixed_mask_nodes": len(part),
            "ico_node_ids_unique": int(part.ico_node_id.nunique()),
            "fsaverage_vertex_ids_unique": int(part.fsaverage_vertex_id.nunique()),
            "fsaverage_vertex_id_min": int(part.fsaverage_vertex_id.min()),
            "fsaverage_vertex_id_max": int(part.fsaverage_vertex_id.max()),
            "eligible_rendered_nodes": int(eligible.sum()),
            "gray_ico4_vertices": int((~eligible).sum()),
            "ico4_mesh_vertices": N_ICO4,
            "ico4_mesh_faces": int(len(meshes[hemi]["faces"])),
        })
    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(args.output_dir / "surface_mapping_qc.csv", index=False)
    report = {
        "status": "complete",
        "figure_png": f"{args.figure_stem}.png",
        "figure_pdf": f"{args.figure_stem}.pdf",
        "age_range": [args.age_min, args.age_max],
        "surface_data_files": files,
        "rendering_space": "native ico4 model space, 2562 vertices per hemisphere",
        "surface_template": "nested first 2562 fsaverage5 vertices; ico4 topology reconstructed from the matching sphere",
        "mapping": "values assigned explicitly by saved hemisphere and fsaverage_vertex_id; ico_node_id retained and checked in source metadata",
        "interpolation": "none",
        "medial_wall_policy": "no cross-medial-wall interpolation; unavailable vertices remain NaN and render gray",
        "color_map": "RdBu_r (blue negative, white zero, red positive)",
        "display_multiplier": DISPLAY_MULTIPLIER,
        "display_unit": "x10^-3 mm^2/s per decade",
        "symmetric_vmax_raw_mm2_per_s": vmax_raw,
        "full_max_abs_raw_mm2_per_s": full_vmax_raw,
        "p98_to_full_max_abs_ratio": vmax_raw / full_vmax_raw,
        "raw_finite_min": float(finite.min()),
        "raw_finite_max": float(finite.max()),
        "range_rule": "symmetric +/- P98(abs(E)) pooled over all finite effects from all three depth bands and both hemispheres",
        "raw_data_clipped": False,
        "colorbar_extend": "both",
        "n_below_negative_display_limit": int(np.sum(finite < -vmax_raw)),
        "n_above_positive_display_limit": int(np.sum(finite > vmax_raw)),
        "n_color_clipped": int(np.sum(np.abs(finite) > vmax_raw)),
        "fraction_color_clipped": float(np.mean(np.abs(finite) > vmax_raw)),
        "mask_color": MISSING_COLOR,
        "mapping_checks": mapping.to_dict(orient="records"),
        "significance_overlay": False,
    }
    atomic_json(args.output_dir / "rendering_qc.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
