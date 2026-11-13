#!/usr/bin/env python3
"""Generate O2-Column profiles from native-column profiles and mapping weights.

Description
-----------
Generate O2-Column profiles from native-column profiles and mapping weights.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--subject-id`` command-line input
- ``--hemisphere`` command-line input
- ``--template`` command-line input
- ``--mapping-root`` command-line input
- ``--output`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure2/o2_column/generate_o2_column_profiles.py --help
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import time
from pathlib import Path

import numpy as np


DEPTH_COUNT = 21
WEIGHT_TOLERANCE = 1e-6
METHOD_NAME = "O2-Column"


def read_native_profiles(
    mapping_root: Path,
    subject_id: str,
    hemisphere: str,
) -> dict[int, np.ndarray]:
    """Read cleaned native profiles keyed by native vertex ID."""

    region_root = mapping_root / subject_id / hemisphere / "regions"
    paths = sorted(region_root.glob("*_native_to_fsaverage.csv.gz"))
    if not paths:
        raise FileNotFoundError(
            f"No native-column mapping files found in {region_root}"
        )

    profiles: dict[int, np.ndarray] = {}
    profile_fields = [f"MD_depth_{depth}" for depth in range(DEPTH_COUNT)]

    for path in paths:
        logging.info("Reading native profiles from %s", path.name)
        with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"native_vertex_id", "column_valid", *profile_fields}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing fields {sorted(missing)} in {path}")

            for row in reader:
                if row["column_valid"].lower() != "true":
                    continue

                native_vertex_id = int(row["native_vertex_id"])
                profile = np.asarray(
                    [float(row[field]) for field in profile_fields],
                    dtype=float,
                )
                if not np.isfinite(profile).all():
                    raise ValueError(
                        f"Non-finite native profile for vertex {native_vertex_id}"
                    )

                # Legacy native profiles are white -> pial. Convert once to
                # the project-wide pial -> white convention on input.
                profile = profile[::-1]

                previous = profiles.get(native_vertex_id)
                if previous is not None and not np.allclose(
                    previous,
                    profile,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"Conflicting profiles for native vertex {native_vertex_id}"
                    )
                profiles[native_vertex_id] = profile

    return profiles


def build_profile_matrix(
    source_profiles: dict[int, np.ndarray],
) -> np.ndarray:
    """Convert native-vertex profiles to a vertex-indexed matrix."""

    if not source_profiles:
        raise ValueError("No valid native-column profiles were found")

    max_vertex = max(source_profiles)
    matrix = np.full((max_vertex + 1, DEPTH_COUNT), np.nan, dtype=float)
    for native_vertex_id, profile in source_profiles.items():
        matrix[native_vertex_id] = profile
    return matrix


def generate_profiles(
    mapping_path: Path,
    output_path: Path,
    source_profiles: dict[int, np.ndarray],
    source_profile_root: Path,
    subject_id: str,
    template: str,
    hemisphere: str,
) -> dict[str, object]:
    """Apply validated barycentric weights to native MD profiles."""

    started = time.time()
    with np.load(mapping_path, allow_pickle=False) as mapping:
        target_vertex_id = np.asarray(mapping["target_vertex_id"], dtype=np.int64)
        depth_key = "depths" if "depths" in mapping.files else "depth_fractions"
        depths = np.asarray(mapping[depth_key], dtype=float)
        source_vertex_ids = np.asarray(
            mapping["source_vertex_ids"],
            dtype=np.int64,
        )
        weights = np.asarray(mapping["barycentric_weights"], dtype=float)
        mapping_valid = np.asarray(mapping["mapping_valid"], dtype=bool)

    if depths.size != DEPTH_COUNT:
        raise ValueError(f"Expected {DEPTH_COUNT} depths, found {depths.size}")
    if source_vertex_ids.shape != (target_vertex_id.size, 3):
        raise ValueError("source_vertex_ids shape does not match target vertices")
    if weights.shape != source_vertex_ids.shape:
        raise ValueError("barycentric_weights shape does not match source vertices")
    if mapping_valid.size != target_vertex_id.size:
        raise ValueError("mapping_valid shape does not match target vertices")

    weight_sum_error = np.abs(weights.sum(axis=1) - 1.0)
    weights_valid = np.isfinite(weights).all(axis=1)
    weights_valid &= weight_sum_error <= WEIGHT_TOLERANCE
    weights_valid &= np.all(
        (weights >= -WEIGHT_TOLERANCE) & (weights <= 1.0 + WEIGHT_TOLERANCE),
        axis=1,
    )

    profile_matrix = build_profile_matrix(source_profiles)
    source_ids_in_range = np.all(
        (source_vertex_ids >= 0)
        & (source_vertex_ids < profile_matrix.shape[0]),
        axis=1,
    )
    source_profile_available = np.zeros(target_vertex_id.size, dtype=bool)
    source_profile_available[source_ids_in_range] = np.isfinite(
        profile_matrix[source_vertex_ids[source_ids_in_range]]
    ).all(axis=(1, 2))

    valid = mapping_valid & weights_valid & source_profile_available
    profiles = np.full((target_vertex_id.size, DEPTH_COUNT), np.nan, dtype=float)
    if np.any(valid):
        profiles[valid] = np.einsum(
            "ij,ijk->ik",
            weights[valid],
            profile_matrix[source_vertex_ids[valid]],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        target_vertex_id=target_vertex_id,
        depths=depths,
        md_profiles=profiles,
        source_vertex_ids=source_vertex_ids,
        barycentric_weights=weights,
        mapping_valid=mapping_valid,
        weights_valid=weights_valid,
        source_profile_available=source_profile_available,
        valid_mask=valid,
        method=METHOD_NAME,
        source_profile_root=str(source_profile_root),
        source_mapping=str(mapping_path),
        subject_id=subject_id,
        template=template,
        hemisphere=hemisphere,
        depth_direction="pial_to_white",
    )

    qc = {
        "subject_id": subject_id,
        "template": template,
        "hemisphere": hemisphere,
        "method": METHOD_NAME,
        "depth_direction": "pial_to_white",
        "source_profile_direction": "white_to_pial",
        "source_mapping": str(mapping_path),
        "n_target_vertices": int(target_vertex_id.size),
        "n_valid_mapping": int(mapping_valid.sum()),
        "n_out_of_range_weights": int((mapping_valid & ~weights_valid).sum()),
        "n_source_profile_missing": int(
            (mapping_valid & ~source_profile_available).sum()
        ),
        "n_valid_output_profiles": int(valid.sum()),
        "valid_output_percentage": float(valid.mean() * 100.0),
        "weight_sum_error_median": float(np.nanmedian(weight_sum_error)),
        "weight_sum_error_max": float(np.nanmax(weight_sum_error)),
        "output_finite_value_percentage": float(
            np.isfinite(profiles).mean() * 100.0
        ),
        "elapsed_s": time.time() - started,
    }
    output_path.with_suffix(".qc.json").write_text(json.dumps(qc, indent=2))
    logging.info(
        "Saved %s: %d/%d valid target profiles",
        output_path,
        int(valid.sum()),
        target_vertex_id.size,
    )
    return qc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--hemisphere", choices=["lh", "rh"], required=True)
    parser.add_argument("--template", choices=["fsaverage", "fsaverage6"], required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--coordinate-root", type=Path)
    parser.add_argument("--mapping-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    source_profiles = read_native_profiles(
        args.mapping_root,
        args.subject_id,
        args.hemisphere,
    )
    if args.mapping_file is not None:
        mapping_path = args.mapping_file
    else:
        mapping_path = (
            args.coordinate_root
            / args.subject_id
            / args.template
            / "o3_coordinate"
            / f"{args.subject_id}_{args.hemisphere}_profiles.npz"
        )
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"Missing validated coordinate mapping: {mapping_path}"
        )

    generate_profiles(
        mapping_path,
        args.output,
        source_profiles,
        args.mapping_root,
        args.subject_id,
        args.template,
        args.hemisphere,
    )


if __name__ == "__main__":
    main()
