#!/usr/bin/env python3
"""Compute one subject's real O3 dissimilarity-distance summaries.

Description
-----------
Compute one subject's real O3 dissimilarity-distance summaries.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--pair-dir`` command-line input
- ``--output-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/compute_real_spatial_curves.py --help
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

from spatial_metrics import (
    aggregate_anchor_bins,
    aggregate_subject_bins,
    pair_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--subject-id")
    return parser.parse_args()


def select_subject(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.index is not None:
        return manifest.iloc[args.index]
    matches = manifest.loc[manifest["subject_id"].eq(args.subject_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {args.subject_id}")
    return matches.iloc[0]


def load_profiles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        profiles = np.asarray(data["profiles"], dtype=float)
        target_ids = np.asarray(data["target_vertex_id"], dtype=np.int64)
        direction = str(data["depth_direction"].item())
        depths = np.asarray(data["depth_fractions"], dtype=float)
        valid = np.asarray(data["sampling_valid"], dtype=bool)
        if "in_analysis_mask" in data.files:
            valid &= np.asarray(data["in_analysis_mask"], dtype=bool)
    if direction != "pial_to_white":
        raise ValueError(f"Expected pial_to_white profiles: {path}")
    if profiles.shape[1] != 21 or not np.allclose(
        depths,
        np.linspace(0.0, 1.0, 21),
    ):
        raise ValueError(f"Expected canonical 21-depth profiles: {path}")
    if np.unique(target_ids).size != len(target_ids):
        raise ValueError(f"Duplicate target vertex IDs: {path}")
    n_vertices = int(target_ids.max()) + 1
    aligned = np.full((n_vertices, 21), np.nan, dtype=float)
    aligned_valid = np.zeros(n_vertices, dtype=bool)
    aligned[target_ids] = profiles
    aligned_valid[target_ids] = valid & np.isfinite(profiles).all(axis=1)
    return aligned, aligned_valid


def add_subject(table: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    table.insert(0, "subject_id", subject_id)
    return table


def main() -> None:
    args = parse_args()
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
        all_pairs = pd.read_parquet(
            args.pair_dir / "figure4_geodesic_pairs_all.parquet"
        )
        boundary_pairs = pd.read_parquet(
            args.pair_dir / "figure4_geodesic_pairs_boundary.parquet"
        )
        all_anchor_tables = []
        boundary_anchor_tables = []
        exclusion_rows = []
        for hemisphere in ("lh", "rh"):
            logging.info("[%s] Loading %s profiles", subject_id, hemisphere)
            profiles, valid = load_profiles(Path(row[f"clean_profile_{hemisphere}"]))
            hemisphere_all = all_pairs.loc[
                all_pairs["hemisphere"].eq(hemisphere)
            ].reset_index(drop=True)
            hemisphere_all = hemisphere_all.copy()
            hemisphere_all["pair_type"] = "all"
            all_metrics = pair_metrics(profiles, valid, hemisphere_all)
            all_anchor_tables.append(aggregate_anchor_bins(all_metrics))

            hemisphere_boundary = boundary_pairs.loc[
                boundary_pairs["hemisphere"].eq(hemisphere)
            ].reset_index(drop=True)
            boundary_metrics = pair_metrics(profiles, valid, hemisphere_boundary)
            boundary_anchor_tables.append(aggregate_anchor_bins(boundary_metrics))
            exclusion_rows.append(
                {
                    "subject_id": subject_id,
                    "hemisphere": hemisphere,
                    "n_profile_vertices": len(profiles),
                    "n_valid_profile_vertices": int(valid.sum()),
                    "n_real_pairs": len(all_metrics),
                    "n_valid_real_pairs": int(all_metrics["pair_valid"].sum()),
                    "n_constant_or_invalid_pairs": int(
                        (~all_metrics["pair_valid"]).sum()
                    ),
                }
            )
            logging.info("[%s] Completed %s real pairs", subject_id, hemisphere)

        all_anchor = add_subject(
            pd.concat(all_anchor_tables, ignore_index=True),
            subject_id,
        )
        boundary_anchor = add_subject(
            pd.concat(boundary_anchor_tables, ignore_index=True),
            subject_id,
        )
        combined_anchor = pd.concat(
            [all_anchor, boundary_anchor],
            ignore_index=True,
        )
        combined_anchor.to_parquet(
            output_dir / "figure4_real_anchor_bins.parquet",
            index=False,
        )
        subject_bins = add_subject(
            aggregate_subject_bins(combined_anchor.drop(columns="subject_id")),
            subject_id,
        )
        subject_bins.to_csv(
            output_dir / "figure4_real_subject_bins.csv",
            index=False,
        )

        split_rows = []
        for split_half, part in all_anchor.groupby("split_half"):
            summary = aggregate_subject_bins(part.drop(columns="subject_id"))
            summary.insert(0, "split_half", split_half)
            summary.insert(0, "subject_id", subject_id)
            split_rows.append(summary)
        pd.concat(split_rows, ignore_index=True).to_csv(
            output_dir / "figure4_real_split_half_subject_bins.csv",
            index=False,
        )
        pd.DataFrame(exclusion_rows).to_csv(
            output_dir / "figure4_real_sampling_qc.csv",
            index=False,
        )
        completion.write_text(
            json.dumps(
                {
                    "subject_id": subject_id,
                    "status": "complete",
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
        logging.exception("[%s] Figure 4 real analysis failed", subject_id)
        raise


if __name__ == "__main__":
    main()
