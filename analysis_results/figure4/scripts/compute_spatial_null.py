#!/usr/bin/env python3
"""Compute native-DWI-grid acquisition-and-sampling nulls for Figure 4B."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates

from spatial_metrics import (
    aggregate_anchor_bins,
    aggregate_subject_bins,
    pair_metrics,
)
from surface_pca_surrogate import load_real_profiles, match_mean_and_covariance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--subject-id")
    parser.add_argument("--null-method", default="interpolation_only")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--psf-fwhm-mm", type=float, default=0.0)
    parser.add_argument(
        "--match-laminar-statistics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--preliminary",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=20261020)
    return parser.parse_args()


def select_subject(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.index is not None:
        return manifest.iloc[args.index]
    matches = manifest.loc[manifest["subject_id"].eq(args.subject_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {args.subject_id}")
    return matches.iloc[0]


def stable_null_seed(base_seed: int, subject_id: str, replicate: int | None) -> int:
    """Return a reproducible 64-bit seed with subject/replicate domain separation."""
    level = "subject" if replicate is None else f"replicate:{replicate}"
    payload = f"{base_seed}\0{subject_id}\0{level}".encode("utf-8")
    digest = hashlib.blake2b(
        payload, digest_size=8, person=b"F4AcqNullV2"
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def load_sampling_geometry(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        coordinates = np.asarray(data["dwi_voxel_coordinates"], dtype=float)
        target_ids = np.asarray(data["target_vertex_id"], dtype=np.int64)
        valid = np.asarray(data["coordinate_in_bounds_mask"], dtype=bool).all(axis=1)
        valid &= np.asarray(data["mapping_valid"], dtype=bool)
        if "in_analysis_mask" in data.files:
            valid &= np.asarray(data["in_analysis_mask"], dtype=bool)
        if "column_valid" in data.files:
            valid &= np.asarray(data["column_valid"], dtype=bool)
    if coordinates.shape[1:] != (21, 3):
        raise ValueError(f"Expected n_vertices x 21 x 3 coordinates: {path}")
    n_vertices = int(target_ids.max()) + 1
    aligned_coordinates = np.full((n_vertices, 21, 3), np.nan, dtype=float)
    aligned_valid = np.zeros(n_vertices, dtype=bool)
    aligned_coordinates[target_ids] = coordinates
    aligned_valid[target_ids] = valid
    return aligned_coordinates, aligned_valid, target_ids


def sample_noise(
    noise: np.ndarray,
    coordinates: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    profiles = np.full(coordinates.shape[:2], np.nan, dtype=np.float32)
    valid_ids = np.flatnonzero(valid)
    for depth_index in range(21):
        profiles[valid_ids, depth_index] = map_coordinates(
            noise,
            coordinates[valid_ids, depth_index].T,
            order=1,
            mode="nearest",
        )
    return profiles


def recolor_bilateral_profiles(
    synthetic: dict[str, np.ndarray],
    real: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Match pooled subject-level 21-depth mean and covariance exactly."""
    synthetic_values = np.concatenate(
        [synthetic[h][valid[h]] for h in ("lh", "rh")]
    ).astype(np.float64)
    real_values = np.concatenate(
        [real[h][valid[h]] for h in ("lh", "rh")]
    ).astype(np.float64)
    recolored_values = match_mean_and_covariance(synthetic_values, real_values)
    recolored = {
        h: np.full(real[h].shape, np.nan, dtype=np.float32) for h in ("lh", "rh")
    }
    offset = 0
    for hemisphere in ("lh", "rh"):
        count = int(valid[hemisphere].sum())
        recolored[hemisphere][valid[hemisphere]] = recolored_values[
            offset : offset + count
        ].astype(np.float32)
        offset += count
    return recolored


def depth_statistics(
    subject_id: str,
    replicate: int,
    data_type: str,
    profiles: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
) -> pd.DataFrame:
    values = np.concatenate([profiles[h][valid[h]] for h in ("lh", "rh")])
    return pd.DataFrame({
        "subject_id": subject_id,
        "replicate": replicate,
        "data_type": data_type,
        "depth_index": np.arange(values.shape[1]),
        "mean": values.mean(axis=0),
        "median": np.median(values, axis=0),
        "sd": values.std(axis=0, ddof=1),
        "q05": np.quantile(values, 0.05, axis=0),
        "q95": np.quantile(values, 0.95, axis=0),
    })


def depth_covariance(
    subject_id: str,
    replicate: int,
    data_type: str,
    profiles: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
) -> pd.DataFrame:
    values = np.concatenate([profiles[h][valid[h]] for h in ("lh", "rh")])
    covariance = np.cov(values, rowvar=False)
    correlation = np.corrcoef(values, rowvar=False)
    depth_i, depth_j = np.indices(covariance.shape)
    return pd.DataFrame({
        "subject_id": subject_id,
        "replicate": replicate,
        "data_type": data_type,
        "depth_i": depth_i.ravel(),
        "depth_j": depth_j.ravel(),
        "covariance": covariance.ravel(),
        "correlation": correlation.ravel(),
    })


def profile_mean_correlation_statistics(
    subject_id: str,
    replicate: int,
    data_type: str,
    profiles: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    reference_mean: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize column correlations with the real subject mean profile."""
    values = np.concatenate([profiles[h][valid[h]] for h in ("lh", "rh")])
    summary_rows = []
    histogram_rows = []
    edges = np.linspace(-1.0, 1.0, 101)
    for depth_set, indices in (
        ("all_21_depths", np.arange(21)),
        ("inner_19_depths", np.arange(1, 20)),
    ):
        current = values[:, indices].astype(np.float64)
        reference = reference_mean[indices].astype(np.float64)
        current -= current.mean(axis=1, keepdims=True)
        reference -= reference.mean()
        denominator = np.linalg.norm(current, axis=1) * np.linalg.norm(reference)
        correlations = np.full(len(current), np.nan, dtype=float)
        nonconstant = denominator > 0
        correlations[nonconstant] = (
            current[nonconstant] @ reference / denominator[nonconstant]
        )
        correlations = correlations[np.isfinite(correlations)]
        quantiles = np.quantile(correlations, [.05, .25, .5, .75, .95])
        summary_rows.append({
            "subject_id": subject_id,
            "replicate": replicate,
            "data_type": data_type,
            "depth_set": depth_set,
            "n_columns": len(correlations),
            "mean_correlation": correlations.mean(),
            "q05": quantiles[0],
            "q25": quantiles[1],
            "median_correlation": quantiles[2],
            "q75": quantiles[3],
            "q95": quantiles[4],
        })
        counts, _ = np.histogram(correlations, bins=edges)
        for bin_index, count in enumerate(counts):
            histogram_rows.append({
                "subject_id": subject_id,
                "replicate": replicate,
                "data_type": data_type,
                "depth_set": depth_set,
                "correlation_bin_index": bin_index,
                "correlation_bin_center": (edges[bin_index] + edges[bin_index + 1]) / 2,
                "count": int(count),
                "fraction": count / len(correlations),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(histogram_rows)


def subject_curve(
    profiles_by_hemisphere: dict[str, np.ndarray],
    valid_by_hemisphere: dict[str, np.ndarray],
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    anchor_tables = []
    for hemisphere in ("lh", "rh"):
        hemisphere_pairs = pairs.loc[pairs["hemisphere"].eq(hemisphere)].copy()
        if "pair_type" not in hemisphere_pairs:
            hemisphere_pairs["pair_type"] = "all"
        if "split_half" not in hemisphere_pairs:
            hemisphere_pairs["split_half"] = 1
        metrics = pair_metrics(
            profiles_by_hemisphere[hemisphere],
            valid_by_hemisphere[hemisphere],
            hemisphere_pairs,
        )
        anchor_tables.append(aggregate_anchor_bins(metrics))
    return aggregate_subject_bins(pd.concat(anchor_tables, ignore_index=True))


def main() -> None:
    args = parse_args()
    if args.null_method not in {"acquisition_sampling", "interpolation_only", "psf_smoothed"}:
        raise ValueError("Unsupported null method")
    if args.null_method == "interpolation_only" and args.psf_fwhm_mm != 0:
        raise ValueError("interpolation_only requires PSF_FWHM_MM=0")
    if args.null_method == "psf_smoothed" and args.psf_fwhm_mm <= 0:
        raise ValueError("psf_smoothed requires an explicit positive PSF FWHM")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    row = select_subject(manifest, args)
    subject_id = str(row["subject_id"])
    output_dir = args.output_root / "subjects" / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    completion = output_dir / "completion.json"
    failure = output_dir / "failure.json"
    started = time.time()

    try:
        if not bool(row["input_complete"]):
            raise FileNotFoundError(
                f"Incomplete manifest inputs: {row['missing_inputs']}"
            )
        pairs = pd.read_parquet(args.pair_dir / "figure4_geodesic_pairs_all.parquet")
        md_image = nib.load(str(row["md_image"]))
        shape = md_image.shape[:3]
        affine = np.asarray(md_image.affine, dtype=float)
        voxel_sizes = np.asarray(md_image.header.get_zooms()[:3], dtype=float)
        sigma = args.psf_fwhm_mm / 2.354820045 / voxel_sizes
        subject_seed = stable_null_seed(args.seed, subject_id, None)
        replicate_seeds = [
            stable_null_seed(args.seed, subject_id, replicate)
            for replicate in range(args.replicates)
        ]
        subject_rows = []
        distribution_rows = []
        covariance_rows = []
        laminar_shape_summary_rows = []
        laminar_shape_histogram_rows = []

        geometries = {}
        real_profiles = {}
        final_valid = {}
        mask_rows = []
        for hemisphere in ("lh", "rh"):
            coordinates, geometry_valid, _ = load_sampling_geometry(
                Path(row[f"clean_profile_{hemisphere}"])
            )
            profiles, real_valid = load_real_profiles(
                Path(row[f"clean_profile_{hemisphere}"])
            )
            if len(profiles) != len(coordinates):
                raise ValueError(f"Profile/coordinate vertex mismatch for {hemisphere}")
            geometries[hemisphere] = coordinates
            real_profiles[hemisphere] = profiles
            final_valid[hemisphere] = geometry_valid & real_valid
            mask_rows.append({
                "subject_id": subject_id,
                "hemisphere": hemisphere,
                "geometry_valid_vertices": int(geometry_valid.sum()),
                "real_final_valid_vertices": int(real_valid.sum()),
                "shared_valid_vertices": int(final_valid[hemisphere].sum()),
                "real_null_mask_identical": bool(
                    np.array_equal(final_valid[hemisphere], geometry_valid & real_valid)
                ),
            })

        real_curve = subject_curve(real_profiles, final_valid, pairs).rename(
            columns={
                "median_dissimilarity": "real_dissimilarity",
                "n_anchors": "real_anchor_count",
                "n_pairs": "real_pair_count",
            }
        )
        distribution_rows.append(
            depth_statistics(subject_id, -1, "real", real_profiles, final_valid)
        )
        covariance_rows.append(
            depth_covariance(subject_id, -1, "real", real_profiles, final_valid)
        )
        real_values = np.concatenate(
            [real_profiles[h][final_valid[h]] for h in ("lh", "rh")]
        )
        reference_mean = real_values.mean(axis=0)
        shape_summary, shape_histogram = profile_mean_correlation_statistics(
            subject_id, -1, "real", real_profiles, final_valid, reference_mean
        )
        laminar_shape_summary_rows.append(shape_summary)
        laminar_shape_histogram_rows.append(shape_histogram)

        for replicate in range(args.replicates):
            logging.info(
                "[%s] Null replicate %d/%d", subject_id, replicate + 1, args.replicates
            )
            replicate_seed = replicate_seeds[replicate]
            rng = np.random.default_rng(replicate_seed)
            noise = rng.standard_normal(shape, dtype=np.float32)
            if args.psf_fwhm_mm > 0:
                noise = gaussian_filter(noise, sigma=sigma, mode="reflect")
            null_profiles = {}
            for hemisphere in ("lh", "rh"):
                null_profiles[hemisphere] = sample_noise(
                    noise, geometries[hemisphere], final_valid[hemisphere]
                )
            if args.match_laminar_statistics:
                null_profiles = recolor_bilateral_profiles(
                    null_profiles, real_profiles, final_valid
                )
            distribution_rows.append(
                depth_statistics(
                    subject_id, replicate, "null", null_profiles, final_valid
                )
            )
            covariance_rows.append(
                depth_covariance(
                    subject_id, replicate, "null", null_profiles, final_valid
                )
            )
            shape_summary, shape_histogram = profile_mean_correlation_statistics(
                subject_id, replicate, "null", null_profiles, final_valid,
                reference_mean,
            )
            laminar_shape_summary_rows.append(shape_summary)
            laminar_shape_histogram_rows.append(shape_histogram)
            summary = subject_curve(null_profiles, final_valid, pairs).rename(
                columns={
                    "median_dissimilarity": "null_dissimilarity",
                    "n_anchors": "anchor_count",
                    "n_pairs": "pair_count",
                }
            )
            summary = summary.merge(
                real_curve[[
                    "distance_bin_index", "distance_bin", "pair_type",
                    "real_dissimilarity", "real_anchor_count", "real_pair_count",
                ]],
                on=["distance_bin_index", "distance_bin", "pair_type"],
                validate="one_to_one",
            )
            summary.insert(0, "replicate", replicate)
            summary.insert(1, "replicate_seed", replicate_seed)
            summary.insert(0, "subject_id", subject_id)
            summary["null_minus_real"] = (
                summary["null_dissimilarity"] - summary["real_dissimilarity"]
            )
            far = summary["distance_bin_index"].between(10, 14)
            real_far = float(summary.loc[far, "real_dissimilarity"].median())
            null_far = float(summary.loc[far, "null_dissimilarity"].median())
            summary["real_local_excess_similarity"] = (
                real_far - summary["real_dissimilarity"]
            )
            summary["null_local_excess_similarity"] = (
                null_far - summary["null_dissimilarity"]
            )
            summary["real_minus_null_local_excess"] = (
                summary["real_local_excess_similarity"]
                - summary["null_local_excess_similarity"]
            )
            subject_rows.append(summary)

        output = pd.concat(subject_rows, ignore_index=True)
        output["null_method"] = args.null_method
        output["psf_fwhm_mm"] = args.psf_fwhm_mm
        output["laminar_recolored"] = args.match_laminar_statistics
        output["preliminary"] = args.preliminary
        output.to_csv(
            output_dir / "figure4_null_subject_bins.csv",
            index=False,
        )
        pd.DataFrame(mask_rows).to_csv(
            output_dir / "figure4_null_mask_qc.csv", index=False
        )
        pd.concat(distribution_rows, ignore_index=True).to_csv(
            output_dir / "figure4_null_depth_distribution.csv", index=False
        )
        pd.concat(covariance_rows, ignore_index=True).to_csv(
            output_dir / "figure4_null_depth_covariance.csv", index=False
        )
        pd.concat(laminar_shape_summary_rows, ignore_index=True).to_csv(
            output_dir / "figure4_profile_mean_correlation_summary.csv", index=False
        )
        pd.concat(laminar_shape_histogram_rows, ignore_index=True).to_csv(
            output_dir / "figure4_profile_mean_correlation_histogram.csv", index=False
        )
        completion.write_text(
            json.dumps(
                {
                    "subject_id": subject_id,
                    "status": "complete",
                    "null_method": args.null_method,
                    "psf_fwhm_mm": args.psf_fwhm_mm,
                    "laminar_recolored": args.match_laminar_statistics,
                    "voxel_sizes_mm": voxel_sizes.tolist(),
                    "volume_shape": list(shape),
                    "volume_affine": affine.tolist(),
                    "random_field_support": "complete_native_dwi_fov",
                    "sampling_coordinates": "real_o3_dwi_voxel_coordinates",
                    "interpolation_order": 1,
                    "interpolation_mode": "nearest",
                    "psf_sigma_voxels": sigma.tolist(),
                    "base_seed": args.seed,
                    "subject_seed": subject_seed,
                    "replicate_seed_scheme": "blake2b64:F4AcqNullV2(base_seed,subject_id,replicate)",
                    "replicate_seeds": replicate_seeds,
                    "replicates": args.replicates,
                    "preliminary": args.preliminary,
                    "runtime_seconds": time.time() - started,
                    "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024.0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        failure.unlink(missing_ok=True)
    except Exception as error:
        failure.write_text(
            json.dumps(
                {
                    "subject_id": subject_id,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "runtime_seconds": time.time() - started,
                    "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 1024.0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        completion.unlink(missing_ok=True)
        logging.exception("[%s] Figure 4 null failed", subject_id)
        raise


if __name__ == "__main__":
    main()
