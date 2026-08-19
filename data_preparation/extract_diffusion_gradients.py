"""
Extract diffusion b-values and gradient vectors from a BXH metadata file.

The script reads the diffusiondirection dimension in the BXH file,
validates the gradient count against an optional 4D diffusion NIfTI,
and writes FSL-style .bval and .bvec files.

Inputs:
    - BXH metadata file
    - Optional 4D diffusion NIfTI for validation

Outputs:
    - .bval file
    - .bvec file

Note:
    The extracted gradient vectors are written directly from the BXH
    diffusion metadata. Gradient coordinate conventions should be
    validated before using them for tensor fitting.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np


def strip_namespace(tag):
    return tag.split("}", 1)[-1]


def find_diffusion_dimension(root):
    for element in root.iter():
        if strip_namespace(element.tag) != "dimension":
            continue

        if element.attrib.get("type") == "diffusiondirection":
            return element

    raise ValueError("No diffusiondirection dimension found in BXH file.")


def extract_bvalues(diffusion_dimension):
    for element in diffusion_dimension:
        if strip_namespace(element.tag) != "datapoints":
            continue

        if element.attrib.get("label") == "bvalues":
            values = [float(x) for x in element.text.split()]
            return np.asarray(values, dtype=float)

    raise ValueError("No bvalues found in diffusiondirection dimension.")


def extract_bvectors(diffusion_dimension):
    for element in diffusion_dimension:
        if strip_namespace(element.tag) != "datapoints":
            continue

        if element.attrib.get("label") != "diffusiondirection":
            continue

        vectors = []

        for value in element:
            if strip_namespace(value.tag) != "value":
                continue

            vector = [float(x) for x in value.text.split()]

            if len(vector) != 3:
                raise ValueError(
                    f"Expected 3 values per diffusion vector, got {vector}"
                )

            vectors.append(vector)

        return np.asarray(vectors, dtype=float)

    raise ValueError("No diffusion direction vectors found.")


def extract_measurement_frame(diffusion_dimension):
    for element in diffusion_dimension:
        if strip_namespace(element.tag) != "measurementframe":
            continue

        vectors = []

        for vector_element in element:
            if strip_namespace(vector_element.tag) != "vector":
                continue

            vector = [float(x) for x in vector_element.text.split()]

            if len(vector) != 3:
                raise ValueError("Invalid measurement frame vector.")

            vectors.append(vector)

        frame = np.asarray(vectors, dtype=float)

        if frame.shape != (3, 3):
            raise ValueError(
                f"Expected measurement frame shape (3, 3), got {frame.shape}"
            )

        return frame

    return np.eye(3)


def validate_gradients(bvals, bvecs, nifti_path=None):
    if len(bvals) != len(bvecs):
        raise ValueError(
            f"Number of b-values ({len(bvals)}) does not match "
            f"number of b-vectors ({len(bvecs)})."
        )

    norms = np.linalg.norm(bvecs, axis=1)

    b0_mask = bvals < 50
    diffusion_mask = ~b0_mask

    if np.any(norms[b0_mask] > 1e-3):
        raise ValueError(
            "At least one b=0 volume has a non-zero diffusion vector."
        )

    if np.any(np.abs(norms[diffusion_mask] - 1.0) > 0.02):
        bad_indices = np.where(
            diffusion_mask & (np.abs(norms - 1.0) > 0.02)
        )[0]

        raise ValueError(
            "Some diffusion vectors are not approximately unit length: "
            f"{bad_indices.tolist()}"
        )

    if nifti_path is not None:
        image = nib.load(str(nifti_path))

        if len(image.shape) != 4:
            raise ValueError(
                f"Expected 4D diffusion image, got shape {image.shape}."
            )

        number_of_volumes = image.shape[3]

        if number_of_volumes != len(bvals):
            raise ValueError(
                f"NIfTI has {number_of_volumes} volumes, but BXH contains "
                f"{len(bvals)} gradient entries."
            )

        print(f"NIfTI volumes: {number_of_volumes}")

    print(f"Gradient entries: {len(bvals)}")
    print(f"b=0 volumes: {np.sum(b0_mask)}")
    print(f"diffusion-weighted volumes: {np.sum(diffusion_mask)}")
    print(
        "Diffusion vector norm range: "
        f"{norms[diffusion_mask].min():.6f} - "
        f"{norms[diffusion_mask].max():.6f}"
    )


def save_gradients(bvals, bvecs, output_prefix):
    output_prefix = Path(output_prefix)

    bval_path = output_prefix.with_suffix(".bval")
    bvec_path = output_prefix.with_suffix(".bvec")

    np.savetxt(
        bval_path,
        bvals.reshape(1, -1),
        fmt="%.8g",
    )

    np.savetxt(
        bvec_path,
        bvecs.T,
        fmt="%.10f",
    )

    print(f"Saved b-values: {bval_path}")
    print(f"Saved b-vectors: {bvec_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract diffusion gradient metadata from a BXH file."
    )

    parser.add_argument(
        "bxh",
        type=Path,
        help="Input BXH file",
    )

    parser.add_argument(
        "--nifti",
        type=Path,
        help="Optional diffusion NIfTI for volume-count validation",
    )

    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output prefix for .bval and .bvec files",
    )

    args = parser.parse_args()

    tree = ET.parse(args.bxh)
    root = tree.getroot()

    diffusion_dimension = find_diffusion_dimension(root)

    bvals = extract_bvalues(diffusion_dimension)
    bvecs = extract_bvectors(diffusion_dimension)
    measurement_frame = extract_measurement_frame(diffusion_dimension)

    print("Measurement frame:")
    print(measurement_frame)

    validate_gradients(
        bvals,
        bvecs,
        args.nifti,
    )

    save_gradients(
        bvals,
        bvecs,
        args.output_prefix,
    )


if __name__ == "__main__":
    main()