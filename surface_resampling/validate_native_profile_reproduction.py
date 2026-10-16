#!/usr/bin/env python3
"""Verify that the shared MD sampling code reproduces native profiles."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from profile_sampling.sample_profiles import (
    load_metric_image,
    load_region_coordinates,
    sample_trajectories,
)


DEPTH_COUNT = 21


def profile_files(
    profiles_root: Path,
    subject_id: str,
    hemisphere: str,
) -> list[tuple[str, Path]]:
    prefix = f"{subject_id}_{hemisphere}_"
    paths = []
    for path in sorted(
        (profiles_root / subject_id / "profiles").glob(f"{prefix}*_md.csv")
    ):
        region = path.name[len(prefix) : -len("_md.csv")]
        paths.append((region, path))
    return paths


def correlation(reference: np.ndarray, reproduced: np.ndarray) -> float:
    reference_centered = reference - reference.mean()
    reproduced_centered = reproduced - reproduced.mean()
    denominator = np.linalg.norm(reference_centered) * np.linalg.norm(
        reproduced_centered
    )
    if denominator == 0:
        return float("nan")
    return float(np.dot(reference_centered, reproduced_centered) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--profiles-root", type=Path, required=True)
    parser.add_argument("--metric-image", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    volume = load_metric_image(args.metric_image)
    reference_parts = []
    reproduced_parts = []
    file_count = 0

    for hemisphere in ("lh", "rh"):
        for region, profile_path in profile_files(
            args.profiles_root,
            args.subject_id,
            hemisphere,
        ):
            coordinate_path = (
                args.profiles_root
                / args.subject_id
                / "regions"
                / f"{hemisphere}_{region}.mat"
            )
            reference = np.loadtxt(profile_path, delimiter=",", ndmin=2)
            coordinates = load_region_coordinates(coordinate_path, hemisphere)
            reproduced = sample_trajectories(volume, coordinates)
            if reference.shape != reproduced.shape:
                raise ValueError(
                    f"Shape mismatch for {hemisphere}_{region}: "
                    f"reference={reference.shape}, reproduced={reproduced.shape}"
                )
            if reference.shape[1] != DEPTH_COUNT:
                raise ValueError(f"Unexpected depth count in {profile_path}")
            reference_parts.append(reference)
            reproduced_parts.append(reproduced)
            file_count += 1
            logging.info(
                "Reproduced %d/%d columns for %s_%s",
                len(reference),
                reference.shape[0],
                hemisphere,
                region,
            )

    if not reference_parts:
        raise RuntimeError("No native MD profiles were found")

    reference_all = np.vstack(reference_parts)
    reproduced_all = np.vstack(reproduced_parts)
    valid_rows = np.isfinite(reference_all).all(axis=1) & np.isfinite(
        reproduced_all
    ).all(axis=1)
    if not valid_rows.any():
        raise RuntimeError("No complete native profiles were available")

    rng = np.random.default_rng(args.seed)
    valid_indices = np.flatnonzero(valid_rows)
    sample_size = min(args.n_samples, len(valid_indices))
    selected = rng.choice(valid_indices, size=sample_size, replace=False)
    reference_sample = reference_all[selected]
    reproduced_sample = reproduced_all[selected]
    differences = reproduced_sample - reference_sample

    flattened_reference = reference_sample.reshape(-1)
    flattened_reproduced = reproduced_sample.reshape(-1)
    result = {
        "subject_id": args.subject_id,
        "metric_image": str(args.metric_image),
        "profiles_root": str(args.profiles_root),
        "depth_count": DEPTH_COUNT,
        "sampling_order": 1,
        "sampling_mode": "nearest",
        "coordinate_indexing": "0-based after subtracting 1 from transform output",
        "n_profile_files": file_count,
        "n_complete_native_columns": int(valid_rows.sum()),
        "n_sampled_columns": int(sample_size),
        "random_seed": args.seed,
        "pearson_r": correlation(flattened_reference, flattened_reproduced),
        "mae": float(np.mean(np.abs(differences))),
        "rmse": float(np.sqrt(np.mean(differences**2))),
        "maximum_absolute_difference": float(np.max(np.abs(differences))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    logging.info("Saved native profile reproduction QC to %s", args.output)


if __name__ == "__main__":
    main()
