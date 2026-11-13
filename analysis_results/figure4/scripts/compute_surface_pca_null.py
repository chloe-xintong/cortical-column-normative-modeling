#!/usr/bin/env python3
"""Compute one subject's smoothness-matched surface PCA null pilot.

Description
-----------
Compute one subject's smoothness-matched surface PCA null pilot.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--pair-dir`` command-line input
- ``--real-results-root`` command-line input
- ``--subjects-dir`` command-line input
- ``--output-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/compute_surface_pca_null.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from spatial_metrics import aggregate_anchor_bins, aggregate_subject_bins, pair_metrics
from surface_pca_surrogate import (
    fit_pca,
    generate_surrogate,
    harmonic_extend_scores,
    load_real_profiles,
    load_spheres,
    load_white_faces,
    normalized_masked_adjacency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--real-results-root", type=Path, required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--subject-id")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--variance-explained", type=float, default=.99)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--diffusion-strength", type=float, default=0.0)
    parser.add_argument("--diffusion-steps", type=int, default=1)
    parser.add_argument(
        "--rotation-mode", choices=("independent_components", "shared"),
        default="independent_components",
    )
    return parser.parse_args()


def select_subject(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    return manifest.iloc[args.index] if args.index is not None else manifest.loc[manifest.subject_id.eq(args.subject_id)].iloc[0]


def subject_curve(profiles, valid, pairs):
    anchors = []
    for hemisphere in ("lh", "rh"):
        part = pairs.loc[pairs.hemisphere.eq(hemisphere)].copy()
        part["pair_type"] = "all"
        anchors.append(aggregate_anchor_bins(pair_metrics(profiles[hemisphere], valid[hemisphere], part)))
    return aggregate_subject_bins(pd.concat(anchors, ignore_index=True))


def depth_distribution(subject_id, replicate, kind, profiles, valid):
    values = np.concatenate([profiles[h][valid[h]] for h in ("lh", "rh")])
    return pd.DataFrame({
        "subject_id": subject_id, "replicate": replicate, "data_type": kind,
        "depth_index": np.arange(21), "median": np.median(values, axis=0),
        "sd": np.std(values, axis=0, ddof=1), "q05": np.quantile(values, .05, axis=0),
        "q95": np.quantile(values, .95, axis=0),
    })


def covariance_long(subject_id, replicate, kind, profiles, valid):
    values = np.concatenate([profiles[h][valid[h]] for h in ("lh", "rh")])
    correlation = np.corrcoef(values, rowvar=False)
    first, second = np.indices(correlation.shape)
    return pd.DataFrame({"subject_id": subject_id, "replicate": replicate, "data_type": kind,
        "depth_i": first.ravel(), "depth_j": second.ravel(), "correlation": correlation.ravel()})


def spatial_acf(subject_id, replicate, kind, profiles, valid, pairs):
    rows = []
    for hemisphere in ("lh", "rh"):
        pair = pairs.loc[pairs.hemisphere.eq(hemisphere)]
        left = pair.anchor_vertex.to_numpy(dtype=int); right = pair.neighbor_vertex.to_numpy(dtype=int)
        keep = valid[hemisphere][left] & valid[hemisphere][right]
        pair = pair.loc[keep]
        left = left[keep]; right = right[keep]
        for depth in range(21):
            values = profiles[hemisphere][valid[hemisphere], depth].astype(float)
            mean, sd = values.mean(), values.std(ddof=1)
            products = ((profiles[hemisphere][left, depth] - mean) / sd) * ((profiles[hemisphere][right, depth] - mean) / sd)
            local = pd.DataFrame({"distance_bin_index": pair.distance_bin_index.to_numpy(), "spatial_acf": products})
            for bin_index, group in local.groupby("distance_bin_index"):
                rows.append({"subject_id": subject_id, "replicate": replicate, "data_type": kind,
                    "hemisphere": hemisphere, "depth_index": depth, "distance_bin_index": int(bin_index),
                    "spatial_acf": float(group.spatial_acf.mean()), "pair_count": len(group)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args(); started = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    row = select_subject(manifest, args); subject_id = str(row.subject_id)
    output_dir = args.output_root / "subjects" / subject_id; output_dir.mkdir(parents=True, exist_ok=True)
    try:
        pairs = pd.read_parquet(args.pair_dir / "figure4_geodesic_pairs_all.parquet")
        profiles = {}; valid = {}
        for hemisphere in ("lh", "rh"):
            profiles[hemisphere], valid[hemisphere] = load_real_profiles(Path(row[f"clean_profile_{hemisphere}"]))
        spheres = load_spheres(args.subjects_dir)
        faces = load_white_faces(args.subjects_dir)
        adjacencies = {"lh": None, "rh": None}
        if args.diffusion_strength:
            adjacencies = {
                hemisphere: normalized_masked_adjacency(
                    faces[hemisphere], valid[hemisphere]
                )
                for hemisphere in ("lh", "rh")
            }
        mean_profile, loadings, scores, explained = fit_pca(profiles, valid, args.variance_explained)
        extended_scores = {"lh": None, "rh": None}
        if args.rotation_mode == "independent_components":
            extended_scores = {
                hemisphere: harmonic_extend_scores(
                    scores[hemisphere], valid[hemisphere], faces[hemisphere]
                )
                for hemisphere in ("lh", "rh")
            }
        real = subject_curve(profiles, valid, pairs).rename(columns={
            "median_dissimilarity": "real_dissimilarity", "n_anchors": "real_anchor_count", "n_pairs": "real_pair_count"})
        outputs = []; distributions = [depth_distribution(subject_id, -1, "real", profiles, valid)]
        covariances = [covariance_long(subject_id, -1, "real", profiles, valid)]
        acfs = [spatial_acf(subject_id, -1, "real", profiles, valid, pairs)]
        rng = np.random.default_rng(args.seed + sum(map(ord, subject_id)))
        unique_rows = []
        for replicate in range(args.replicates):
            logging.info("[%s] surface PCA realization %d/%d", subject_id, replicate + 1, args.replicates)
            null = {}; unique = {}
            for hemisphere in ("lh", "rh"):
                null[hemisphere], unique[hemisphere], repair_fraction, imputed_fraction = generate_surrogate(
                    mean_profile, loadings, scores[hemisphere], valid[hemisphere], spheres[hemisphere], profiles[hemisphere], rng,
                    do_rank_remap=args.variance_explained < 1.0,
                    adjacency=adjacencies[hemisphere],
                    diffusion_strength=args.diffusion_strength,
                    diffusion_steps=args.diffusion_steps,
                    rotation_mode=args.rotation_mode,
                    extended_scores=extended_scores[hemisphere])
                for component, fraction in enumerate(unique[hemisphere]):
                    unique_rows.append({"subject_id": subject_id, "replicate": replicate, "hemisphere": hemisphere,
                        "component": component, "unique_source_fraction": fraction,
                        "mask_repair_fraction": repair_fraction[component],
                        "imputed_source_fraction": imputed_fraction[component],
                        "diffusion_strength": args.diffusion_strength,
                        "diffusion_steps": args.diffusion_steps,
                        "rotation_mode": args.rotation_mode})
            curve = subject_curve(null, valid, pairs).rename(columns={
                "median_dissimilarity": "null_dissimilarity", "n_anchors": "anchor_count", "n_pairs": "pair_count"})
            merged = curve.merge(real[["distance_bin_index", "distance_bin", "pair_type", "real_dissimilarity", "real_anchor_count", "real_pair_count"]],
                on=["distance_bin_index", "distance_bin", "pair_type"], validate="one_to_one")
            merged.insert(0, "replicate", replicate); merged.insert(0, "subject_id", subject_id)
            merged["null_minus_real"] = merged.null_dissimilarity - merged.real_dissimilarity
            merged["null_method"] = "surface_pca_independent_component_surrogate"; merged["preliminary"] = True
            outputs.append(merged)
            distributions.append(depth_distribution(subject_id, replicate, "surrogate", null, valid))
            covariances.append(covariance_long(subject_id, replicate, "surrogate", null, valid))
            acfs.append(spatial_acf(subject_id, replicate, "surrogate", null, valid, pairs))
        pd.concat(outputs).to_csv(output_dir / "figure4_surface_pca_null_subject_bins.csv", index=False)
        pd.concat(distributions).to_csv(output_dir / "figure4_surface_pca_depth_distribution.csv", index=False)
        pd.concat(covariances).to_csv(output_dir / "figure4_surface_pca_depth_covariance.csv", index=False)
        pd.concat(acfs).to_csv(output_dir / "figure4_surface_pca_spatial_acf.csv", index=False)
        pd.DataFrame(unique_rows).to_csv(output_dir / "figure4_surface_pca_spin_qc.csv", index=False)
        target = {"subject_id": subject_id, "n_components": int(loadings.shape[0]),
            "variance_target": args.variance_explained, "variance_explained": float(explained.sum()),
            "diffusion_strength": args.diffusion_strength,
            "diffusion_steps": args.diffusion_steps,
            "rotation_mode": args.rotation_mode,
            "component_variance_fraction": explained.tolist(), "mean_depth_profile": mean_profile.tolist(),
            "valid_vertices_lh": int(valid["lh"].sum()), "valid_vertices_rh": int(valid["rh"].sum())}
        (output_dir / "figure4_surface_pca_targets.json").write_text(json.dumps(target, indent=2) + "\n")
        completion = {"subject_id": subject_id, "status": "complete", "null_method": "surface_pca_independent_component_surrogate",
            "replicates": args.replicates, "preliminary": True, "runtime_seconds": time.time()-started,
            "diffusion_strength": args.diffusion_strength,
            "diffusion_steps": args.diffusion_steps,
            "rotation_mode": args.rotation_mode,
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        (output_dir / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
        (output_dir / "failure.json").unlink(missing_ok=True)
    except Exception as error:
        (output_dir / "failure.json").write_text(json.dumps({"subject_id": subject_id, "status": "failed",
            "error_type": type(error).__name__, "error": str(error)}, indent=2) + "\n")
        logging.exception("[%s] surface PCA null failed", subject_id); raise


if __name__ == "__main__":
    main()
