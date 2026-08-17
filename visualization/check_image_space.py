#!/usr/bin/env python3
"""
check_image_space.py

Compare the spatial geometry of two NIfTI images.

The script checks:
1. Image dimensions (shape)
2. Voxel size
3. Orientation
4. Affine transformation matrix

These checks help determine whether two images, such as an ADC map and the DWI
image from which it was derived, occupy the same voxel grid and physical/world
coordinate system.

Example
-------
python check_image_space.py \
    --image1 S00775_adc.nii.gz \
    --image2 S00775_dwi_summed_upsampled.nii.gz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti(path: Path) -> nib.Nifti1Image:
    """
    Load a NIfTI image after checking that the file exists.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    return nib.load(str(path))


def get_spatial_info(img: nib.Nifti1Image) -> dict:
    """
    Extract spatial information from a NIfTI image.
    """
    return {
        "shape": img.shape,
        "voxel_size": img.header.get_zooms()[:3],
        "orientation": nib.aff2axcodes(img.affine),
        "affine": img.affine,
    }


def print_spatial_info(name: str, path: Path, info: dict) -> None:
    """
    Print spatial information for one image.
    """
    print("\n" + "=" * 70)
    print(f"{name}")
    print("=" * 70)

    print(f"Path:        {path}")
    print(f"Shape:       {info['shape']}")
    print(f"Voxel size:  {info['voxel_size']}")
    print(f"Orientation: {info['orientation']}")

    print("\nAffine matrix:")
    print(info["affine"])


def compare_images(info1: dict, info2: dict) -> dict:
    """
    Compare spatial properties of two NIfTI images.
    """

    same_shape = info1["shape"] == info2["shape"]

    same_voxel_size = np.allclose(
        info1["voxel_size"],
        info2["voxel_size"],
    )

    same_orientation = info1["orientation"] == info2["orientation"]

    same_affine = np.allclose(
        info1["affine"],
        info2["affine"],
    )

    max_affine_difference = np.max(
        np.abs(info1["affine"] - info2["affine"])
    )

    return {
        "same_shape": same_shape,
        "same_voxel_size": same_voxel_size,
        "same_orientation": same_orientation,
        "same_affine": same_affine,
        "max_affine_difference": max_affine_difference,
    }


def print_comparison(results: dict) -> None:
    """
    Print comparison results and an overall interpretation.
    """
    print("\n" + "=" * 70)
    print("SPATIAL COMPARISON")
    print("=" * 70)

    print(f"Same shape:       {results['same_shape']}")
    print(f"Same voxel size:  {results['same_voxel_size']}")
    print(f"Same orientation: {results['same_orientation']}")
    print(f"Same affine:      {results['same_affine']}")

    print(
        "Max affine difference: "
        f"{results['max_affine_difference']:.8g}"
    )

    print("\n" + "-" * 70)

    if (
        results["same_shape"]
        and results["same_voxel_size"]
        and results["same_orientation"]
        and results["same_affine"]
    ):
        print(
            "[PASS] The two images have the same spatial geometry.\n"
            "They occupy the same voxel grid and world coordinate system."
        )

    elif results["same_affine"] and results["same_orientation"]:
        print(
            "[WARNING] The images share the same orientation/affine, "
            "but their voxel grids differ.\n"
            "This may indicate resampling, upsampling, or another change "
            "in image dimensions/resolution."
        )

    else:
        print(
            "[FAIL] The images do not have identical spatial geometry.\n"
            "Check whether one image has been registered, resampled, "
            "cropped, reoriented, or otherwise transformed."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare spatial geometry between two NIfTI images."
        )
    )

    parser.add_argument(
        "--image1",
        required=True,
        help="Path to the first NIfTI image.",
    )

    parser.add_argument(
        "--image2",
        required=True,
        help="Path to the second NIfTI image.",
    )

    args = parser.parse_args()

    image1_path = Path(args.image1)
    image2_path = Path(args.image2)

    img1 = load_nifti(image1_path)
    img2 = load_nifti(image2_path)

    info1 = get_spatial_info(img1)
    info2 = get_spatial_info(img2)

    print_spatial_info(
        "IMAGE 1",
        image1_path,
        info1,
    )

    print_spatial_info(
        "IMAGE 2",
        image2_path,
        info2,
    )

    results = compare_images(
        info1,
        info2,
    )

    print_comparison(results)


if __name__ == "__main__":
    main()