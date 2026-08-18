"""
Check FreeSurfer outputs required for cortical trajectory generation.

This script verifies that the white and pial surfaces, cortical thickness,
and aparc annotation files are present and aligned for both hemispheres.

For each hemisphere, it checks:
- white and pial surfaces have the same number of vertices
- white and pial surfaces have the same mesh topology
- cortical thickness values match the number of surface vertices
- aparc annotation labels match the number of surface vertices

These checks confirm the vertex-wise correspondence used later in the
trajectory pipeline:

    white[i] <-> pial[i] <-> thickness[i] <-> aparc[i]

Usage:
    python qc/check_freesurfer_outputs.py <freesurfer_subject_dir>
"""
from pathlib import Path
import argparse
import sys

import nibabel.freesurfer.io as fsio
import numpy as np


def check_file(path):
    if not path.exists():
        print(f"  [FAIL] Missing file: {path}")
        return False

    if path.stat().st_size == 0:
        print(f"  [FAIL] Empty file: {path}")
        return False

    return True


def check_hemisphere(subject_dir, hemi):
    print(f"\nChecking {hemi.upper()} hemisphere")

    surf_dir = subject_dir / "surf"
    label_dir = subject_dir / "label"

    white_file = surf_dir / f"{hemi}.white"
    pial_file = surf_dir / f"{hemi}.pial"
    thickness_file = surf_dir / f"{hemi}.thickness"
    aparc_file = label_dir / f"{hemi}.aparc.annot"

    required_files = [
        white_file,
        pial_file,
        thickness_file,
        aparc_file,
    ]

    files_ok = True

    for path in required_files:
        if not check_file(path):
            files_ok = False

    if not files_ok:
        print(f"{hemi.upper()}: FAIL")
        return False

    try:
        white_vertices, white_faces = fsio.read_geometry(white_file)
        pial_vertices, pial_faces = fsio.read_geometry(pial_file)

        thickness = fsio.read_morph_data(thickness_file)

        labels, ctab, names = fsio.read_annot(aparc_file)

    except Exception as exc:
        print(f"  [FAIL] Could not read FreeSurfer data: {exc}")
        print(f"{hemi.upper()}: FAIL")
        return False

    n_white = len(white_vertices)
    n_pial = len(pial_vertices)
    n_thickness = len(thickness)
    n_labels = len(labels)

    vertex_count_match = n_white == n_pial
    topology_match = np.array_equal(white_faces, pial_faces)
    thickness_match = n_thickness == n_white
    aparc_match = n_labels == n_white

    print(f"  white vertices:     {n_white}")
    print(f"  pial vertices:      {n_pial}")
    print(f"  thickness values:   {n_thickness}")
    print(f"  aparc labels:       {n_labels}")
    print(f"  annotation entries: {len(names)}")

    print(
        "  white/pial vertex count:",
        "PASS" if vertex_count_match else "FAIL",
    )

    print(
        "  white/pial topology:",
        "PASS" if topology_match else "FAIL",
    )

    print(
        "  thickness alignment:",
        "PASS" if thickness_match else "FAIL",
    )

    print(
        "  aparc alignment:",
        "PASS" if aparc_match else "FAIL",
    )

    hemisphere_pass = (
        vertex_count_match
        and topology_match
        and thickness_match
        and aparc_match
    )

    print(
        f"{hemi.upper()}:",
        "PASS" if hemisphere_pass else "FAIL",
    )

    return hemisphere_pass


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check FreeSurfer outputs required for cortical trajectory "
            "generation."
        )
    )

    parser.add_argument(
        "subject_dir",
        type=Path,
        help="Path to the FreeSurfer subject directory.",
    )

    args = parser.parse_args()

    subject_dir = args.subject_dir.expanduser().resolve()

    if not subject_dir.exists():
        print(f"Subject directory does not exist: {subject_dir}")
        sys.exit(1)

    print("FreeSurfer output QC")
    print(f"Subject directory: {subject_dir}")

    lh_pass = check_hemisphere(subject_dir, "lh")
    rh_pass = check_hemisphere(subject_dir, "rh")

    overall_pass = lh_pass and rh_pass

    print("\nOverall result:")
    print(
        "FreeSurfer surface QC:",
        "PASS" if overall_pass else "FAIL",
    )

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()