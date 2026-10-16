#!/usr/bin/env python3
"""Create compact native-space overlays for manual Option 3 registration QC."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-root", type=Path, required=True)
    parser.add_argument("--o3-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--normal-n", type=int, default=1)
    parser.add_argument("--subject-id", action="append")
    return parser.parse_args()


def scalar_string(value: np.ndarray | str) -> str:
    """Convert a scalar NumPy string value to a regular string."""

    if isinstance(value, np.ndarray):
        value = value.item()
    return str(value)


def select_cases(
    subject_qc: pd.DataFrame,
    top_n: int,
    normal_n: int,
    selected_subjects: list[str] | None,
) -> pd.DataFrame:
    """Select high-OOB, low-coverage, and representative normal cases."""

    table = subject_qc.copy()
    if selected_subjects:
        table = table[table.subject.isin(selected_subjects)].copy()
    rows = []
    for hemisphere, group in table.groupby("hemisphere", sort=True):
        for _, row in group.nlargest(top_n, "oob_vertex_in_mask_pct").iterrows():
            rows.append({**row.to_dict(), "selection_reason": "highest_oob"})
        for _, row in group.nsmallest(top_n, "valid_profile_pct").iterrows():
            rows.append({**row.to_dict(), "selection_reason": "lowest_coverage"})
        normal = group.assign(
            normal_score=(
                (100.0 - group.valid_profile_pct).abs()
                + group.oob_vertex_in_mask_pct
            )
        ).nsmallest(normal_n, "normal_score")
        for _, row in normal.iterrows():
            rows.append({**row.to_dict(), "selection_reason": "representative_normal"})
    selected = pd.DataFrame(rows)
    if selected.empty:
        raise ValueError("No subject-by-hemisphere cases selected")
    selected = selected.drop_duplicates(subset=["subject", "hemisphere"])
    return selected.sort_values(["hemisphere", "subject"]).reset_index(drop=True)


def write_case_outputs(case: pd.Series, o3_root: Path, output_root: Path) -> dict:
    """Write a DWI-space point mask and a Freeview command for one case."""

    subject = str(case.subject)
    hemisphere = str(case.hemisphere)
    profile_path = (
        o3_root
        / subject
        / "fsaverage"
        / "o3_coordinate"
        / f"{subject}_{hemisphere}_profiles.npz"
    )
    with np.load(profile_path, allow_pickle=True) as data:
        dwi_voxel = np.asarray(data["dwi_voxel_coordinates"], dtype=float)
        in_bounds = np.asarray(data["coordinate_in_bounds_mask"], dtype=bool)
        mapping_valid = np.asarray(data["mapping_valid"], dtype=bool)
        md_path = Path(scalar_string(data["metric_image"]))
        registration_path = Path(scalar_string(data["registration"]))

    md_image = nib.load(str(md_path))
    md_shape = md_image.shape[:3]
    brain_mask_path = md_path.parent / (
        f"{subject}_corrected_mean_b0_brain_mask.nii.gz"
    )
    if not brain_mask_path.is_file():
        brain_mask_path = None
    point_mask = np.zeros(md_shape, dtype=np.float32)
    depth_map = np.zeros(md_shape, dtype=np.float32)
    valid_points = (
        mapping_valid[:, None]
        & in_bounds
        & np.isfinite(dwi_voxel).all(axis=2)
    )
    coordinates = dwi_voxel[valid_points]
    depth_values = np.tile(
        np.linspace(0.0, 1.0, dwi_voxel.shape[1]),
        dwi_voxel.shape[0],
    )[valid_points.ravel()]
    voxel_indices = np.rint(coordinates).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, np.asarray(md_shape) - 1)
    point_mask[
        voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
    ] = 1.0
    np.maximum.at(
        depth_map,
        (voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]),
        depth_values.astype(np.float32),
    )

    case_root = output_root / f"{subject}_{hemisphere}"
    case_root.mkdir(parents=True, exist_ok=True)
    point_path = case_root / f"{subject}_{hemisphere}_trajectory_points.nii.gz"
    depth_path = case_root / f"{subject}_{hemisphere}_trajectory_depth.nii.gz"
    nib.save(nib.Nifti1Image(point_mask, md_image.affine), str(point_path))
    nib.save(nib.Nifti1Image(depth_map, md_image.affine), str(depth_path))

    subjects_dir = md_path.parents[2] / "t1_freesurfer"
    surface_dir = subjects_dir / subject / "surf"
    white_path = surface_dir / f"{hemisphere}.white"
    pial_path = surface_dir / f"{hemisphere}.pial"
    command_path = case_root / "freeview_command.sh"
    command = " ".join(
        [
            "freeview",
            f'-v "{md_path}:reg={registration_path}"',
            (
                f'"{brain_mask_path}:reg={registration_path}"'
                if brain_mask_path is not None
                else ""
            ),
            f'"{point_path}"',
            f'"{depth_path}"',
            f'-f "{white_path}" "{pial_path}"',
        ]
    ) + "\n"
    command_path.write_text(command)
    return {
        "subject": subject,
        "hemisphere": hemisphere,
        "selection_reason": case.selection_reason,
        "profile_path": str(profile_path),
        "md_path": str(md_path),
        "brain_mask_path": (
            str(brain_mask_path) if brain_mask_path is not None else ""
        ),
        "registration_path": str(registration_path),
        "trajectory_points_path": str(point_path),
        "trajectory_depth_path": str(depth_path),
        "white_surface_path": str(white_path),
        "pial_surface_path": str(pial_path),
        "freeview_command_path": str(command_path),
        "n_valid_trajectory_points": int(valid_points.sum()),
    }


def main() -> None:
    """Create registration QC overlays for selected cohort cases."""

    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    table_path = args.qc_root / "tables" / "cohort_subject_hemisphere_qc.csv"
    subject_qc = pd.read_csv(table_path)
    selected = select_cases(
        subject_qc,
        args.top_n,
        args.normal_n,
        args.subject_id,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_root / "selected_cases.csv", index=False)
    outputs = []
    for index, (_, case) in enumerate(selected.iterrows(), start=1):
        logging.info(
            "Progress %d/%d subject=%s hemisphere=%s reason=%s",
            index,
            len(selected),
            case.subject,
            case.hemisphere,
            case.selection_reason,
        )
        outputs.append(write_case_outputs(case, args.o3_root, args.output_root))
    pd.DataFrame(outputs).to_csv(
        args.output_root / "overlay_manifest.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
