#!/usr/bin/env python3
"""Compute one subject's formal Figure 4 metrics from six retained depths.

Description
-----------
Compute one subject's formal Figure 4 metrics from six retained depths.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--manifest`` command-line input
- ``--pair-file`` command-line input
- ``--split-assignment`` command-line input
- ``--output-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/depth_density_sensitivity_6_vs_21/scripts/compute_subject.py --help
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from compute_real_spatial_curves import load_profiles, select_subject
from fit_correlation_lengths import fit_curve
from spatial_metrics import aggregate_anchor_bins, aggregate_subject_bins, pair_metrics


DEPTH_INDICES = np.asarray([0, 4, 8, 12, 16, 20], dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--split-assignment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--subject-id")
    parser.add_argument("--min-fit-r2", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    row = select_subject(manifest, args)
    subject_id = str(row["subject_id"])
    output_dir = args.output_root / "subjects" / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    completion = output_dir / "completion.json"
    failure = output_dir / "failure.json"
    try:
        if not bool(row["input_complete"]):
            raise FileNotFoundError(f"Incomplete manifest inputs: {row['missing_inputs']}")
        if DEPTH_INDICES[0] != 0 or DEPTH_INDICES[-1] != 20:
            raise RuntimeError("Six-point subset must retain pial and white endpoints")
        pairs = pd.read_parquet(args.pair_file)
        if not pairs["pair_type"].eq("within_region").all():
            raise RuntimeError("Formal pair table must contain only within-region pairs")
        assignment = pd.read_csv(args.split_assignment)[
            ["hemisphere", "anchor_vertex", "split_half"]
        ]
        anchor_parts: list[pd.DataFrame] = []
        hemisphere_checks: list[dict[str, object]] = []
        for hemisphere in ("lh", "rh"):
            profile_21, valid = load_profiles(Path(row[f"clean_profile_{hemisphere}"]))
            if profile_21.shape[1] != 21:
                raise RuntimeError(f"{hemisphere}: profile last dimension is not 21")
            profile_6 = profile_21[:, DEPTH_INDICES]
            exact = np.array_equal(
                profile_6, np.take(profile_21, DEPTH_INDICES, axis=1), equal_nan=True
            )
            if profile_6.shape[1] != 6 or not exact:
                raise RuntimeError(f"{hemisphere}: six-point exact-subset check failed")
            current = pairs.loc[pairs["hemisphere"].eq(hemisphere)].reset_index(drop=True)
            anchor_parts.append(aggregate_anchor_bins(pair_metrics(profile_6, valid, current)))
            hemisphere_checks.append(
                {
                    "hemisphere": hemisphere,
                    "source_profile_shape": list(profile_21.shape),
                    "six_point_shape": list(profile_6.shape),
                    "exact_shared_values": exact,
                    "valid_vertices": int(valid.sum()),
                }
            )
        anchors = pd.concat(anchor_parts, ignore_index=True)
        subject_bins = aggregate_subject_bins(anchors)
        subject_bins.insert(0, "subject_id", subject_id)
        subject_bins.to_csv(output_dir / "figure4_6point_subject_bins.csv", index=False)

        fit_rows = [
            {
                "subject_id": subject_id,
                "fit_scope": "full",
                **fit_curve(subject_bins, min_fit_r2=args.min_fit_r2),
            }
        ]
        split_anchor = anchors.drop(columns="split_half").merge(
            assignment,
            on=["hemisphere", "anchor_vertex"],
            how="inner",
            validate="many_to_one",
        )
        expected_anchors = anchors[["hemisphere", "anchor_vertex"]].drop_duplicates().shape[0]
        observed_anchors = split_anchor[["hemisphere", "anchor_vertex"]].drop_duplicates().shape[0]
        if observed_anchors != expected_anchors:
            raise RuntimeError("Fixed split assignment does not cover every valid anchor")
        split_parts: list[pd.DataFrame] = []
        for half, table in split_anchor.groupby("split_half", observed=True):
            summary = aggregate_subject_bins(table)
            summary.insert(0, "split_half", int(half))
            summary.insert(0, "subject_id", subject_id)
            split_parts.append(summary)
            fit_rows.append(
                {
                    "subject_id": subject_id,
                    "fit_scope": f"fixed_half_{int(half)}",
                    **fit_curve(summary, min_fit_r2=args.min_fit_r2),
                }
            )
        pd.concat(split_parts, ignore_index=True).to_csv(
            output_dir / "figure4_6point_fixed_split_subject_bins.csv", index=False
        )
        pd.DataFrame(fit_rows).to_csv(
            output_dir / "figure4_6point_correlation_length_fits.csv", index=False
        )
        completion.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "subject_id": subject_id,
                    "depth_indices": DEPTH_INDICES.tolist(),
                    "depth_fractions": (DEPTH_INDICES / 20).tolist(),
                    "pial_and_white_retained": True,
                    "hemisphere_checks": hemisphere_checks,
                    "anchor_definition": "official_fsaverage5_fixed",
                    "candidate_definition": "within_DK_region",
                    "k": 15,
                    "pair_seed": int(pairs["seed"].iat[0]),
                    "split_seed": 20261021,
                    "input_pair_count": int(len(pairs)),
                    "valid_pair_count": int(subject_bins["n_pairs"].sum()),
                    "runtime_seconds": time.time() - started,
                    "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        failure.unlink(missing_ok=True)
        print(json.dumps({"status": "complete", "subject_id": subject_id}), flush=True)
    except Exception as error:
        failure.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "subject_id": subject_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "runtime_seconds": time.time() - started,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        completion.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
