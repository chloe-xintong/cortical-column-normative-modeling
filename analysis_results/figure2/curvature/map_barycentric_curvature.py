#!/usr/bin/env python3
"""Map native FreeSurfer curvature to template vertices with saved weights.

Description
-----------
Map native FreeSurfer curvature to template vertices with saved weights.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--subject-id`` command-line input
- ``--hemisphere`` command-line input
- ``--subjects-dir`` command-line input
- ``--mapping-file`` command-line input
- ``--output`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure2/curvature/map_barycentric_curvature.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--hemisphere", choices=["lh", "rh"], required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--mapping-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_mapping(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        target_vertex_id = np.asarray(data["target_vertex_id"], dtype=np.int64)
        source_vertex_ids = np.asarray(
            data["source_vertex_ids"],
            dtype=np.int64,
        )
        weights = np.asarray(data["barycentric_weights"], dtype=float)
        mapping_valid = np.asarray(data["mapping_valid"], dtype=bool)

    if source_vertex_ids.shape != (target_vertex_id.size, 3):
        raise ValueError("source_vertex_ids does not match target vertices")
    if weights.shape != source_vertex_ids.shape:
        raise ValueError("barycentric_weights does not match source vertices")
    if mapping_valid.size != target_vertex_id.size:
        raise ValueError("mapping_valid does not match target vertices")
    return target_vertex_id, source_vertex_ids, weights, mapping_valid


def main() -> None:
    args = parse_args()
    surface_root = args.subjects_dir / args.subject_id / "surf"
    curvature_path = surface_root / f"{args.hemisphere}.curv"
    curvature = np.asarray(nib.freesurfer.io.read_morph_data(str(curvature_path)))

    target_vertex_id, source_vertex_ids, weights, mapping_valid = load_mapping(
        args.mapping_file
    )
    source_ids_in_range = np.all(
        (source_vertex_ids >= 0) & (source_vertex_ids < curvature.size),
        axis=1,
    )
    finite_source = np.zeros(target_vertex_id.size, dtype=bool)
    finite_source[source_ids_in_range] = np.isfinite(
        curvature[source_vertex_ids[source_ids_in_range]]
    ).all(axis=1)
    weights_valid = np.isfinite(weights).all(axis=1)
    weights_valid &= np.abs(weights.sum(axis=1) - 1.0) <= 1e-6
    weights_valid &= np.all(
        (weights >= -1e-6) & (weights <= 1.0 + 1e-6),
        axis=1,
    )
    valid = mapping_valid & source_ids_in_range & finite_source & weights_valid

    signed = np.full(target_vertex_id.size, np.nan, dtype=float)
    absolute = np.full(target_vertex_id.size, np.nan, dtype=float)
    signed[valid] = np.einsum(
        "ij,ij->i",
        weights[valid],
        curvature[source_vertex_ids[valid]],
    )
    absolute[valid] = np.einsum(
        "ij,ij->i",
        weights[valid],
        np.abs(curvature[source_vertex_ids[valid]]),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        subject_id=args.subject_id,
        hemisphere=args.hemisphere,
        template="fsaverage",
        method="shared_barycentric_native_curvature",
        target_vertex_id=target_vertex_id,
        source_vertex_ids=source_vertex_ids,
        barycentric_weights=weights,
        mapping_valid=mapping_valid,
        valid_mask=valid,
        signed_mean_curvature=signed,
        absolute_mean_curvature=absolute,
        curvature_surface=str(curvature_path),
        mapping_file=str(args.mapping_file),
    )
    qc = {
        "subject_id": args.subject_id,
        "hemisphere": args.hemisphere,
        "template": "fsaverage",
        "method": "shared_barycentric_native_curvature",
        "curvature_surface": str(curvature_path),
        "mapping_file": str(args.mapping_file),
        "n_target_vertices": int(target_vertex_id.size),
        "n_valid_mapping": int(mapping_valid.sum()),
        "n_valid_curvature": int(valid.sum()),
        "valid_percentage": float(valid.mean() * 100.0),
        "signed_curvature_units": "FreeSurfer native .curv units",
        "absolute_definition": "weighted mean of absolute native curvature",
    }
    args.output.with_suffix(".qc.json").write_text(
        json.dumps(qc, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
