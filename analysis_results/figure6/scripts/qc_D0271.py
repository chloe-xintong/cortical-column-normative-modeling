#!/usr/bin/env python3
"""Create reproducible source-image and profile QC products for D0271.

Description
-----------
This is intentionally subject-specific because D0271 triggered the Figure 6
Gaussian-calibration gate.  It does not modify source data or cohort manifests.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- No required command-line inputs; optional inputs are defined by the script.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/qc_D0271.py --help
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd


SUBJECT = "D0271"
REPO = Path(__file__).resolve().parents[3]
RAW = Path("/mnt/newStor/paros/paros_WORK/tmp_ADRC_MUSE/D0271")
DWI = Path("/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi/D0271")
FS = Path("/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer/D0271")
O3 = Path("/mnt/newStor/paros/paros_WORK/xintong/results/fsaverage_result/columns/D0271")
FEATURES = Path("/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs/features")
O3_QC = Path("/mnt/newStor/paros/paros_WORK/xintong/results/qc/full_cohort_spatial_scale/option3_qc")
REG_QC = Path(
    "/mnt/newStor/paros/paros_WORK/xintong/results/qc/"
    "template_registration_comparison/figure2_method_comparison_fsaverage/qc/"
    "figure2C_registration_input_qc.csv"
)
OUT = REPO / "/mnt/newStor/paros/paros_WORK/xintong/result/cortical_trajectory_analysis/analysis_results/figure6/qc/D0271"
REGIONS = ("isthmuscingulate", "lingual", "parahippocampal")


def image_plane(data: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return data[index, :, :].T
    if axis == 1:
        return data[:, index, :].T
    return data[:, :, index].T


def projected_points(points: np.ndarray, axis: int, index: int, width: float = 0.65):
    near = np.abs(points[:, axis] - index) <= width
    p = points[near]
    if axis == 0:
        return p[:, 1], p[:, 2]
    if axis == 1:
        return p[:, 0], p[:, 2]
    return p[:, 0], p[:, 1]


def occupied_slices(mask: np.ndarray, axis: int) -> list[int]:
    collapsed = np.any(mask, axis=tuple(i for i in range(3) if i != axis))
    nz = np.flatnonzero(collapsed)
    lo, hi = int(nz.min()), int(nz.max())
    return [int(round(x)) for x in np.linspace(lo + 0.2 * (hi - lo), lo + 0.8 * (hi - lo), 3)]


def finite_limits(data: np.ndarray, mask: np.ndarray, q=(1, 99)) -> tuple[float, float]:
    values = data[mask & np.isfinite(data)]
    return tuple(float(x) for x in np.percentile(values, q))


def surface_voxels_t1(t1: nib.spatialimages.SpatialImage) -> dict[str, np.ndarray]:
    ras_to_vox = np.linalg.inv(t1.header.get_vox2ras_tkr())
    out = {}
    for hemi in ("lh", "rh"):
        for boundary in ("white", "pial"):
            vertices, _ = fsio.read_geometry(FS / "surf" / f"{hemi}.{boundary}")
            out[f"{hemi}_{boundary}"] = nib.affines.apply_affine(ras_to_vox, vertices)
    return out


def plot_t1_surface_qc() -> None:
    t1 = nib.load(FS / "mri/T1.mgz")
    data = np.asarray(t1.dataobj, dtype=np.float32)
    brain = np.asarray(nib.load(FS / "mri/brainmask.mgz").dataobj) > 0
    surfaces = surface_voxels_t1(t1)
    lo, hi = finite_limits(data, brain, (0.5, 99.5))
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 10.2), constrained_layout=True)
    for row, (axis, label) in enumerate(zip((2, 1, 0), ("Axial", "Coronal", "Sagittal"))):
        for col, index in enumerate(occupied_slices(brain, axis)):
            ax = axes[row, col]
            ax.imshow(image_plane(data, axis, index), cmap="gray", origin="lower", vmin=lo, vmax=hi)
            for hemi in ("lh", "rh"):
                for boundary, color in (("white", "#DB3124"), ("pial", "#24A7A0")):
                    x, y = projected_points(surfaces[f"{hemi}_{boundary}"], axis, index)
                    ax.scatter(x, y, s=0.18, c=color, linewidths=0, alpha=0.9, rasterized=True)
            ax.set_title(f"{label}: voxel {index}", fontsize=9)
            ax.set_axis_off()
    fig.suptitle("D0271 FreeSurfer surface placement on T1", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.003, "Red: white surface    Teal: pial surface", ha="center", fontsize=9)
    fig.savefig(OUT / "t1_freesurfer_surface_qc.png", dpi=220, facecolor="white")
    plt.close(fig)


def dwi_surfaces(b0_path: Path) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(REPO / "04_profile_sampling"))
    from transform_t1_columns_to_dwi import load_dwi_geometry, load_transform_matrix, transform_points_to_dwi

    transform = load_transform_matrix(O3 / "registration/D0271_DWI2T1.dat")
    _, dwi_tkreg = load_dwi_geometry(b0_path)
    out = {}
    for hemi in ("lh", "rh"):
        for boundary in ("white", "pial"):
            vertices, _ = fsio.read_geometry(FS / "surf" / f"{hemi}.{boundary}")
            out[f"{hemi}_{boundary}"] = transform_points_to_dwi(vertices, transform, dwi_tkreg)
    return out


def plot_registration_qc() -> None:
    b0_path = DWI / "D0271_corrected_mean_b0.nii.gz"
    b0 = nib.load(b0_path).get_fdata(dtype=np.float32)
    mask = nib.load(DWI / "D0271_corrected_mean_b0_brain_mask.nii.gz").get_fdata() > 0
    surfaces = dwi_surfaces(b0_path)
    lo, hi = finite_limits(b0, mask)
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 10.2), constrained_layout=True)
    for row, (axis, label) in enumerate(zip((2, 1, 0), ("Axial", "Coronal", "Sagittal"))):
        for col, index in enumerate(occupied_slices(mask, axis)):
            ax = axes[row, col]
            ax.imshow(image_plane(b0, axis, index), cmap="gray", origin="lower", vmin=lo, vmax=hi)
            for hemi in ("lh", "rh"):
                for boundary, color in (("white", "#DB3124"), ("pial", "#24A7A0")):
                    x, y = projected_points(surfaces[f"{hemi}_{boundary}"], axis, index)
                    ax.scatter(x, y, s=0.18, c=color, linewidths=0, alpha=0.9, rasterized=True)
            ax.set_title(f"{label}: voxel {index}", fontsize=9)
            ax.set_axis_off()
    fig.suptitle("D0271 DWI-to-T1 registration on corrected mean b0", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.003, "Red: white surface    Teal: pial surface", ha="center", fontsize=9)
    fig.savefig(OUT / "dwi_t1_registration_qc.png", dpi=220, facecolor="white")
    plt.close(fig)


def plot_diffusion_maps() -> dict:
    mask = nib.load(DWI / "D0271_corrected_mean_b0_brain_mask.nii.gz").get_fdata() > 0
    paths = {
        "Corrected mean b0": DWI / "D0271_corrected_mean_b0.nii.gz",
        "MD (mm²/s)": DWI / "D0271_md.nii.gz",
        "FA": DWI / "D0271_fa.nii.gz",
        "RD (mm²/s)": DWI / "D0271_rd.nii.gz",
    }
    maps = {name: nib.load(path).get_fdata(dtype=np.float32) for name, path in paths.items()}
    center = np.round(np.mean(np.argwhere(mask), axis=0)).astype(int)
    limits = {
        "Corrected mean b0": finite_limits(maps["Corrected mean b0"], mask),
        "MD (mm²/s)": (0.0, float(np.nanpercentile(maps["MD (mm²/s)"][mask], 99))),
        "FA": (0.0, 1.0),
        "RD (mm²/s)": (0.0, float(np.nanpercentile(maps["RD (mm²/s)"][mask], 99))),
    }
    fig, axes = plt.subplots(4, 3, figsize=(10.8, 12.0), constrained_layout=True)
    for row, (name, data) in enumerate(maps.items()):
        for col, (axis, label) in enumerate(zip((2, 1, 0), ("Axial", "Coronal", "Sagittal"))):
            ax = axes[row, col]
            vmin, vmax = limits[name]
            ax.imshow(image_plane(data, axis, int(center[axis])), cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(f"{name} | {label} {center[axis]}", fontsize=9)
            ax.set_axis_off()
    fig.suptitle("D0271 processed diffusion-map QC", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "diffusion_maps_qc.png", dpi=220, facecolor="white")
    plt.close(fig)

    stats = {}
    for metric in ("md", "fa", "ad", "rd"):
        data = nib.load(DWI / f"D0271_{metric}.nii.gz").get_fdata(dtype=np.float32)[mask]
        finite = data[np.isfinite(data)]
        stats[metric] = {
            "finite_percentage": 100.0 * float(np.isfinite(data).mean()),
            "nonpositive_percentage": 100.0 * float(np.mean(finite <= 0)),
            "q01": float(np.quantile(finite, 0.01)),
            "median": float(np.median(finite)),
            "q99": float(np.quantile(finite, 0.99)),
        }
        if metric == "fa":
            stats[metric]["above_one_percentage"] = 100.0 * float(np.mean(finite > 1))
    return stats


def plot_raw_dwi_qc() -> dict:
    image = nib.load(RAW / "bia5_02050_011.nii.gz")
    mask = nib.load(DWI / "D0271_corrected_mean_b0_brain_mask.nii.gz").get_fdata() > 0
    bvals = np.loadtxt(RAW / "bia5_02050_011.bval").reshape(-1)
    bvecs = np.loadtxt(RAW / "bia5_02050_011.bvec")
    brain_z = np.flatnonzero(np.any(mask, axis=(0, 1)))
    volume_means = []
    slice_means = []
    b800_sum = np.zeros(mask.shape, dtype=np.float64)
    b800_sumsq = np.zeros(mask.shape, dtype=np.float64)
    b800_n = 0
    b0_volumes = []
    for i in range(image.shape[3]):
        volume = np.asarray(image.dataobj[..., i], dtype=np.float32)
        volume_means.append(float(np.mean(volume[mask])))
        slice_means.append([float(np.mean(volume[:, :, z][mask[:, :, z]])) for z in brain_z])
        if bvals[i] == 0:
            b0_volumes.append(volume.copy())
        else:
            b800_sum += volume
            b800_sumsq += np.square(volume, dtype=np.float64)
            b800_n += 1
    mean_b800 = (b800_sum / b800_n).astype(np.float32)
    var_b800 = np.maximum(b800_sumsq / b800_n - np.square(mean_b800, dtype=np.float64), 0)
    cv_b800 = np.zeros(mask.shape, dtype=np.float32)
    cv_b800[mask] = np.sqrt(var_b800[mask]) / np.maximum(mean_b800[mask], 1e-6)
    normalized_slices = np.asarray(slice_means) / np.asarray(volume_means)[:, None]
    diffusion = bvals > 0
    zscores = (normalized_slices[diffusion] - np.mean(normalized_slices[diffusion], axis=0)) / np.maximum(
        np.std(normalized_slices[diffusion], axis=0), 1e-6
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(np.arange(len(volume_means)), np.asarray(volume_means) / np.median(np.asarray(volume_means)[diffusion]), "o-", ms=3)
    axes[0, 0].scatter(np.flatnonzero(~diffusion), np.asarray(volume_means)[~diffusion] / np.median(np.asarray(volume_means)[diffusion]), c="#DB3124", zorder=3, label="b=0")
    axes[0, 0].set(xlabel="Volume", ylabel="Mean brain signal / median b=800", title="Global signal by volume")
    axes[0, 0].legend(frameon=False)
    im = axes[0, 1].imshow(zscores.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-3, vmax=3)
    axes[0, 1].set(xlabel="b=800 volume index", ylabel="Axial brain slice", title="Slice-wise signal z-score")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.8)
    z = int(np.round(np.mean(np.argwhere(mask), axis=0)[2]))
    lo, hi = finite_limits(mean_b800, mask)
    axes[1, 0].imshow(mean_b800[:, :, z].T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
    axes[1, 0].set_title(f"Mean b=800 | axial {z}")
    cv_hi = float(np.nanpercentile(cv_b800[mask], 99))
    axes[1, 1].imshow(cv_b800[:, :, z].T, cmap="magma", origin="lower", vmin=0, vmax=cv_hi)
    axes[1, 1].set_title(f"Across-direction CV | axial {z}")
    for ax in axes[1]:
        ax.set_axis_off()
    fig.suptitle("D0271 raw DWI volume/dropout QC", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "raw_dwi_volume_qc.png", dpi=220, facecolor="white")
    plt.close(fig)

    norms = np.linalg.norm(bvecs, axis=0)
    worst = np.unravel_index(np.argmin(zscores), zscores.shape)
    return {
        "n_volumes": int(image.shape[3]),
        "n_b0": int(np.sum(~diffusion)),
        "n_b800": int(np.sum(diffusion)),
        "max_gradient_norm_error": float(np.max(np.abs(norms[diffusion] - 1))),
        "worst_slice_signal_z": float(zscores[worst]),
        "worst_b800_volume_within_shell": int(worst[0]),
        "worst_axial_voxel": int(brain_z[worst[1]]),
    }


def plot_profile_qc() -> list[dict]:
    subjects = pd.read_csv(FEATURES / "subject_order.csv")
    metadata = pd.read_csv(FEATURES / "feature_metadata/features_regional_profile.csv")
    matrix = np.load(FEATURES / "matrices/X_regional_profile.npy", mmap_mode="r")
    row = int(subjects.loc[subjects.subject_id.eq(SUBJECT), "row_index"].iloc[0])
    depths = np.sort(metadata.depth_fraction.unique())
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True, constrained_layout=True)
    rows = []
    for ax, region in zip(axes, REGIONS):
        meta = metadata[(metadata.hemisphere == "lh") & (metadata.dk_label_name == region)].sort_values("depth_index")
        values = np.asarray(matrix[:, meta.feature_index.to_numpy(dtype=int)], dtype=float)
        subject = values[row]
        q05, q25, median, q75, q95 = np.nanpercentile(values, [5, 25, 50, 75, 95], axis=0)
        ax.fill_between(depths, q05 * 1e3, q95 * 1e3, color="#90BEE0", alpha=0.35, label="Cohort 5–95%")
        ax.fill_between(depths, q25 * 1e3, q75 * 1e3, color="#4B74B2", alpha=0.25, label="Cohort IQR")
        ax.plot(depths, median * 1e3, color="#4B74B2", lw=1.5, label="Cohort median")
        ax.plot(depths, subject * 1e3, color="#DB3124", lw=2.2, marker="o", ms=2.5, label="D0271")
        ax.set(title=f"Left {region}", xlabel="Depth fraction (pial → white)")
        ax.grid(alpha=0.2)
        for depth_index, depth in enumerate(depths):
            vals = values[:, depth_index]
            med = np.nanmedian(vals)
            mad = np.nanmedian(np.abs(vals - med))
            rows.append(
                {
                    "region": region,
                    "depth_index": int(depth_index),
                    "depth_fraction": float(depth),
                    "D0271_md": float(subject[depth_index]),
                    "cohort_median_md": float(med),
                    "robust_z": float(0.6745 * (subject[depth_index] - med) / mad),
                    "percentile": 100.0 * float(np.mean(vals <= subject[depth_index])),
                }
            )
    axes[0].set_ylabel("MD (10⁻³ mm²/s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("D0271 Figure 6 flagged profiles versus 196-subject cohort", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "flagged_profiles_qc.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "flagged_profile_quantiles.csv", index=False)
    return rows


def plot_flagged_spatial_qc() -> dict:
    md = nib.load(DWI / "D0271_md.nii.gz").get_fdata(dtype=np.float32)
    mask = nib.load(DWI / "D0271_corrected_mean_b0_brain_mask.nii.gz").get_fdata() > 0
    vmax = float(np.nanpercentile(md[mask & np.isfinite(md)], 99))
    with np.load(O3 / "profiles/D0271_lh_md_profiles.npz", allow_pickle=False) as profile:
        labels = profile["template_label_name"]
        coords = profile["dwi_voxel_coordinates"]
        values = profile["profiles"]
        sampling_valid = profile["sampling_valid"]
        depth_index = 16
        fig, axes = plt.subplots(3, 3, figsize=(10.8, 9.5), constrained_layout=True)
        result = {}
        for row, region in enumerate(REGIONS):
            selected = (labels == region) & sampling_valid & np.isfinite(values[:, depth_index]) & (values[:, depth_index] > 0)
            points = coords[selected, depth_index]
            center = np.round(np.median(points, axis=0)).astype(int)
            regional_values = values[selected, depth_index]
            result[region] = {
                "depth_fraction": 0.8,
                "n_valid_columns": int(selected.sum()),
                "median_md": float(np.median(regional_values)),
                "mean_md": float(np.mean(regional_values)),
                "q05_md": float(np.quantile(regional_values, 0.05)),
                "q95_md": float(np.quantile(regional_values, 0.95)),
                "median_dwi_voxel": [int(x) for x in center],
            }
            for col, (axis, orientation) in enumerate(zip((2, 1, 0), ("Axial", "Coronal", "Sagittal"))):
                ax = axes[row, col]
                index = int(center[axis])
                ax.imshow(image_plane(md, axis, index), cmap="gray", origin="lower", vmin=0, vmax=vmax)
                x, y = projected_points(points, axis, index, width=0.9)
                ax.scatter(x, y, s=2.0, c="#DB3124", linewidths=0, alpha=0.8, rasterized=True)
                ax.set_title(f"Left {region}\n{orientation} voxel {index}", fontsize=8.5)
                ax.set_axis_off()
    fig.suptitle("Flagged region sampling at depth 0.80 on native MD", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "flagged_regions_spatial_qc.png", dpi=220, facecolor="white")
    plt.close(fig)
    return result


def plot_flagged_t1_surface_qc() -> None:
    t1 = nib.load(FS / "mri/T1.mgz")
    data = np.asarray(t1.dataobj, dtype=np.float32)
    brain = np.asarray(nib.load(FS / "mri/brainmask.mgz").dataobj) > 0
    lo, hi = finite_limits(data, brain, (0.5, 99.5))
    ras_to_vox = np.linalg.inv(t1.header.get_vox2ras_tkr())
    with np.load(O3 / "columns/D0271_lh_columns.npz", allow_pickle=False) as columns:
        labels = columns["template_label_name"]
        valid = columns["column_valid"] & columns["in_analysis_mask"]
        white = nib.affines.apply_affine(ras_to_vox, columns["white_endpoint"])
        pial = nib.affines.apply_affine(ras_to_vox, columns["pial_endpoint"])
        fig, axes = plt.subplots(3, 3, figsize=(10.8, 9.5), constrained_layout=True)
        for row, region in enumerate(REGIONS):
            selected = (labels == region) & valid
            midpoint = 0.5 * (white[selected] + pial[selected])
            center = np.round(np.median(midpoint, axis=0)).astype(int)
            for col, (axis, orientation) in enumerate(zip((2, 1, 0), ("Axial", "Coronal", "Sagittal"))):
                ax = axes[row, col]
                index = int(center[axis])
                ax.imshow(image_plane(data, axis, index), cmap="gray", origin="lower", vmin=lo, vmax=hi)
                xw, yw = projected_points(white[selected], axis, index, width=0.65)
                xp, yp = projected_points(pial[selected], axis, index, width=0.65)
                ax.scatter(xw, yw, s=2.0, c="#DB3124", linewidths=0, alpha=0.85, rasterized=True)
                ax.scatter(xp, yp, s=2.0, c="#24A7A0", linewidths=0, alpha=0.85, rasterized=True)
                ax.set_title(f"Left {region}\n{orientation} voxel {index}", fontsize=8.5)
                ax.set_axis_off()
    fig.suptitle("Flagged-region white/pial placement on native T1", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.003, "Red: white endpoints    Teal: pial endpoints", ha="center", fontsize=9)
    fig.savefig(OUT / "flagged_regions_t1_surface_qc.png", dpi=220, facecolor="white")
    plt.close(fig)


def cohort_mapping_qc() -> dict:
    rows = []
    for path in (O3_QC / "subjects").glob("*/option3_label_qc_region.csv"):
        data = pd.read_csv(path)
        data = data[(data.hemisphere == "lh") & data.target_label_name.isin(REGIONS)].copy()
        data["zero_length_percentage"] = 100 * data.n_zero_length / data.n_vertices
        data["noncortical_triangle_percentage"] = 100 * data.n_noncortical_triangle / data.n_vertices
        rows.append(data)
    all_regions = pd.concat(rows, ignore_index=True)
    output = {}
    for region in REGIONS:
        group = all_regions[all_regions.target_label_name.eq(region)]
        subject = group[group.subject_id.eq(SUBJECT)].iloc[0]
        output[region] = {
            "n_template_region_vertices": int(subject.n_vertices),
            "n_zero_length": int(subject.n_zero_length),
            "zero_length_percentage": float(subject.zero_length_percentage),
            "zero_length_cohort_percentile": 100 * float(np.mean(group.zero_length_percentage <= subject.zero_length_percentage)),
            "noncortical_triangle_percentage": float(subject.noncortical_triangle_percentage),
            "noncortical_cohort_percentile": 100 * float(np.mean(group.noncortical_triangle_percentage <= subject.noncortical_triangle_percentage)),
            "median_target_label_support": float(subject.median_target_label_support),
        }
    return output


def registration_qc() -> dict:
    table = pd.read_csv(REG_QC)
    valid = table[table.mincost_valid]
    subject = valid[valid.subject_id.eq(SUBJECT)].iloc[0]
    return {
        "bbregister_mincost": float(subject.bbregister_mincost),
        "robust_z": float(subject.robust_z),
        "cohort_percentile": 100 * float(np.mean(valid.bbregister_mincost <= subject.bbregister_mincost)),
        "higher_cost_rank": int(np.sum(valid.bbregister_mincost > subject.bbregister_mincost) + 1),
        "n_valid_registrations": int(len(valid)),
        "automatic_cost_outlier": bool(subject.automatic_cost_outlier),
    }


def column_qc() -> dict:
    output = {}
    for hemi in ("lh", "rh"):
        mapping = json.loads((O3 / f"columns/D0271_{hemi}_columns.qc.json").read_text())
        cleaning = json.loads((O3 / f"profiles/D0271_{hemi}_cleaning.json").read_text())
        output[hemi] = {
            "valid_mapping_percentage": float(mapping["valid_mapping_percentage"]),
            "mapping_error_max": float(mapping["mapping_error_max"]),
            "analysis_mask_percentage": float(mapping["analysis_mask_percentage"]),
            "zero_length_in_analysis_mask_percentage": 100 * mapping["n_zero_length_in_analysis_mask"] / mapping["n_in_analysis_mask"],
            "cleaned_column_percentage": 100 * cleaning["n_columns_after_cleaning"] / cleaning["n_columns_before_cleaning"],
            "bad_column_reasons": cleaning["bad_column_reasons"],
        }
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_t1_surface_qc()
    plot_registration_qc()
    diffusion_stats = plot_diffusion_maps()
    raw_dwi_stats = plot_raw_dwi_qc()
    profile_rows = plot_profile_qc()
    spatial = plot_flagged_spatial_qc()
    plot_flagged_t1_surface_qc()
    summary = {
        "subject_id": SUBJECT,
        "purpose": "Figure 6 source-profile/mapping QC gate",
        "source_data_modified": False,
        "registration": registration_qc(),
        "columns": column_qc(),
        "flagged_region_mapping": cohort_mapping_qc(),
        "diffusion_maps_within_brain_mask": diffusion_stats,
        "raw_dwi": raw_dwi_stats,
        "flagged_region_depth_0p8": spatial,
        "flagged_profile_max_robust_z": {
            region: max(row["robust_z"] for row in profile_rows if row["region"] == region)
            for region in REGIONS
        },
        "artifacts": sorted(path.name for path in OUT.iterdir() if path.is_file()),
    }
    (OUT / "quantitative_qc_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
