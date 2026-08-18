#!/usr/bin/env python3

"""
Compare two NIfTI images and report whether they are geometrically
and numerically identical or meaningfully different.
"""

from pathlib import Path
import argparse

import nibabel as nib
import numpy as np


def load_image(path):
    image = nib.load(path)
    data = image.get_fdata(dtype=np.float32)

    return image, data


def compare_images(path1, path2):
    image1, data1 = load_image(path1)
    image2, data2 = load_image(path2)

    print()
    print(f"Image 1: {path1}")
    print(f"Image 2: {path2}")
    print()

    print("Geometry")
    print(f"Shape 1      : {image1.shape}")
    print(f"Shape 2      : {image2.shape}")
    print(f"Voxel size 1 : {image1.header.get_zooms()[:3]}")
    print(f"Voxel size 2 : {image2.header.get_zooms()[:3]}")
    print(
        f"Orientation 1: {nib.aff2axcodes(image1.affine)}"
    )
    print(
        f"Orientation 2: {nib.aff2axcodes(image2.affine)}"
    )

    shape_same = image1.shape == image2.shape

    affine_same = np.allclose(
        image1.affine,
        image2.affine,
        atol=1e-5,
    )

    voxel_size_same = np.allclose(
        image1.header.get_zooms()[:3],
        image2.header.get_zooms()[:3],
        atol=1e-5,
    )

    orientation_same = (
        nib.aff2axcodes(image1.affine)
        == nib.aff2axcodes(image2.affine)
    )

    print()
    print("Geometry comparison")
    print(f"Same shape       : {shape_same}")
    print(f"Same voxel size  : {voxel_size_same}")
    print(f"Same orientation : {orientation_same}")
    print(f"Same affine      : {affine_same}")

    print()
    print("Intensity statistics")

    for name, data in [
        ("Image 1", data1),
        ("Image 2", data2),
    ]:
        print(
            f"{name}: "
            f"min={np.nanmin(data):.6f}, "
            f"mean={np.nanmean(data):.6f}, "
            f"max={np.nanmax(data):.6f}"
        )

    if not shape_same:
        print()
        print(
            "Voxel-wise comparison skipped because "
            "the image shapes are different."
        )
        return

    difference = np.abs(data1 - data2)

    max_difference = np.nanmax(difference)
    mean_difference = np.nanmean(difference)

    identical = np.array_equal(data1, data2)

    close = np.allclose(
        data1,
        data2,
        rtol=1e-5,
        atol=1e-5,
        equal_nan=True,
    )

    different_voxels = np.count_nonzero(
        difference > 1e-5
    )

    total_voxels = difference.size

    percent_different = (
        different_voxels / total_voxels * 100
    )

    print()
    print("Voxel-wise comparison")
    print(f"Exactly identical : {identical}")
    print(f"Numerically close : {close}")
    print(f"Max difference    : {max_difference:.6f}")
    print(f"Mean difference   : {mean_difference:.6f}")
    print(
        f"Different voxels  : "
        f"{different_voxels} / {total_voxels} "
        f"({percent_different:.6f}%)"
    )

    print()
    print("Conclusion")

    if (
        shape_same
        and affine_same
        and identical
    ):
        print(
            "The two NIfTI images are geometrically "
            "and voxel-wise identical."
        )

    elif (
        shape_same
        and affine_same
        and close
    ):
        print(
            "The two images have the same geometry "
            "and are numerically almost identical."
        )

    elif (
        shape_same
        and affine_same
    ):
        print(
            "The two images have the same geometry "
            "but different voxel intensities."
        )

    else:
        print(
            "The two images differ in image geometry "
            "and should not be treated as duplicates."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare two candidate NIfTI images."
    )

    parser.add_argument(
        "image1",
        type=Path,
        help="First NIfTI image",
    )

    parser.add_argument(
        "image2",
        type=Path,
        help="Second NIfTI image",
    )

    args = parser.parse_args()

    path1 = args.image1.resolve()
    path2 = args.image2.resolve()

    if not path1.exists():
        raise FileNotFoundError(
            f"Image not found: {path1}"
        )

    if not path2.exists():
        raise FileNotFoundError(
            f"Image not found: {path2}"
        )

    compare_images(path1, path2)


if __name__ == "__main__":
    main()