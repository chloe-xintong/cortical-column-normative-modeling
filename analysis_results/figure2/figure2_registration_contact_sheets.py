#!/usr/bin/env python3
"""Render b0 contact sheets with T1 surfaces transformed into DWI space.

Description
-----------
The transform implementation intentionally reuses
``04_profile_sampling/transform_t1_columns_to_dwi.py`` so the white/pial
vertices use the same FreeSurfer register.dat convention as O3.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--review-queue`` command-line input
- ``--registration-qc`` command-line input
- ``--output-dir`` command-line input
- ``--subject-id`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure2/figure2_registration_contact_sheets.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--registration-qc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    return parser.parse_args()


def slice_locations(brain_mask: np.ndarray, axis: int) -> list[int]:
    # Robustly avoid empty edge slices while retaining inferior/mid/superior
    # coverage in each orthogonal orientation.
    occupied = np.any(brain_mask.astype(bool), axis=tuple(i for i in range(3) if i != axis))
    nonzero = np.flatnonzero(occupied)
    low, high = (int(nonzero.min()), int(nonzero.max())) if len(nonzero) else (0, brain_mask.shape[axis] - 1)
    return [int(round(value)) for value in np.linspace(low + .2 * (high - low), low + .8 * (high - low), 3)]


def image_plane(image: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return image[index, :, :].T
    if axis == 1:
        return image[:, index, :].T
    return image[:, :, index].T


def projected_surface_points(vertices: np.ndarray, axis: int, index: int, width: float = 0.65):
    near = np.abs(vertices[:, axis] - index) <= width
    points = vertices[near]
    if axis == 0:
        return points[:, 1], points[:, 2]
    if axis == 1:
        return points[:, 0], points[:, 2]
    return points[:, 0], points[:, 1]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "04_profile_sampling"))
    from transform_t1_columns_to_dwi import load_dwi_geometry, load_transform_matrix, transform_points_to_dwi

    queue = pd.read_csv(args.review_queue, dtype={"subject_id": str}).set_index("subject_id")
    qc = pd.read_csv(args.registration_qc, dtype={"subject_id": str}).set_index("subject_id")
    if args.subject_id not in queue.index or args.subject_id not in qc.index:
        raise ValueError(f"{args.subject_id} is absent from the review queue or registration QC table")
    review, metadata = queue.loc[args.subject_id], qc.loc[args.subject_id]
    b0_path = Path("/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi") / args.subject_id / f"{args.subject_id}_corrected_mean_b0.nii.gz"
    brain_mask_path = b0_path.with_name(f"{args.subject_id}_corrected_mean_b0_brain_mask.nii.gz")
    b0 = nib.load(str(b0_path))
    b0_data = b0.get_fdata(dtype=np.float32)
    brain_mask = nib.load(str(brain_mask_path)).get_fdata(dtype=np.float32) > 0
    transform = load_transform_matrix(Path(metadata.transform_path))
    _, dwi_tkreg = load_dwi_geometry(b0_path)
    fs_dir = Path(metadata.freesurfer_subject_dir)
    surfaces = {}
    for hemi in ("lh", "rh"):
        for boundary in ("white", "pial"):
            vertices, _ = fsio.read_geometry(str(fs_dir / "surf" / f"{hemi}.{boundary}"))
            surfaces[f"{hemi}_{boundary}"] = transform_points_to_dwi(vertices, transform, dwi_tkreg)

    lo, hi = np.nanpercentile(b0_data[np.isfinite(b0_data) & (b0_data != 0)], [1, 99])
    labels = ("Axial", "Coronal", "Sagittal")
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 10.2), facecolor="white", constrained_layout=True)
    for row, (axis, orientation) in enumerate(zip((2, 1, 0), labels)):
        for column, index in enumerate(slice_locations(brain_mask, axis)):
            ax = axes[row, column]
            ax.imshow(image_plane(b0_data, axis, index), cmap="gray", origin="lower", vmin=lo, vmax=hi, interpolation="nearest")
            for hemi in ("lh", "rh"):
                for boundary, color, name in (("white", "#DB3124", "white"), ("pial", "#FC8C5A", "pial")):
                    x, y = projected_surface_points(surfaces[f"{hemi}_{boundary}"], axis, index)
                    ax.scatter(x, y, s=.22, c=color, linewidths=0, alpha=.90, rasterized=True, label=name if row == 0 and column == 0 else None)
            ax.set_title(f"{orientation}: voxel {index}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
    status = str(metadata.visual_qc_status).replace("_", " ")
    fig.suptitle(
        f"{args.subject_id}  |  bbregister cost = {review.cost:.4f}  |  robust z = {review.robust_z:.2f}  |  QC: {status}",
        fontsize=12, fontweight="bold",
    )
    fig.legend(loc="lower center", ncol=2, frameon=False, markerscale=8, labels=["White surface", "Pial surface"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.subject_id}_registration_qc.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
