#!/usr/bin/env python3
"""Compute formal Figure 4C fits for one subject from ico5 within-DK K=15 pairs."""

from __future__ import annotations

import argparse
import json
import logging
import resource
import time
from pathlib import Path

import pandas as pd

from compute_real_spatial_curves import load_profiles, select_subject
from fit_correlation_lengths import fit_curve
from spatial_metrics import aggregate_anchor_bins, aggregate_subject_bins, pair_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--split-assignment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--subject-id")
    parser.add_argument("--min-fit-r2", type=float, default=.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    row = select_subject(manifest, args)
    subject_id = str(row.subject_id)
    output_dir = args.output_root / "subjects" / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    completion = output_dir / "completion.json"
    failure = output_dir / "failure.json"
    started = time.time()
    try:
        pairs = pd.read_parquet(args.pair_file)
        assignment = pd.read_csv(args.split_assignment)[[
            "hemisphere", "anchor_vertex", "split_half"
        ]]
        anchor_parts = []
        for hemisphere in ("lh", "rh"):
            profiles, valid = load_profiles(Path(row[f"clean_profile_{hemisphere}"]))
            current = pairs.loc[pairs.hemisphere.eq(hemisphere)].reset_index(drop=True)
            anchor_parts.append(aggregate_anchor_bins(pair_metrics(profiles, valid, current)))
        anchors = pd.concat(anchor_parts, ignore_index=True)
        full = aggregate_subject_bins(anchors)
        fit_rows = [{
            "subject_id": subject_id, "fit_scope": "full",
            **fit_curve(full, min_fit_r2=args.min_fit_r2),
        }]
        split_anchor = anchors.drop(columns="split_half").merge(
            assignment, on=["hemisphere", "anchor_vertex"],
            how="inner", validate="many_to_one",
        )
        if split_anchor[["hemisphere", "anchor_vertex"]].drop_duplicates().shape[0] != anchors[[
            "hemisphere", "anchor_vertex"
        ]].drop_duplicates().shape[0]:
            raise RuntimeError("Split assignment does not cover every valid anchor")
        split_parts = []
        for half, table in split_anchor.groupby("split_half", observed=True):
            summary = aggregate_subject_bins(table)
            summary.insert(0, "split_half", int(half))
            summary.insert(0, "subject_id", subject_id)
            split_parts.append(summary)
            fit_rows.append({
                "subject_id": subject_id,
                "fit_scope": f"fixed_half_{int(half)}",
                **fit_curve(summary, min_fit_r2=args.min_fit_r2),
            })
        pd.concat(split_parts, ignore_index=True).to_csv(
            output_dir / "figure4C_fixed_split_subject_bins.csv", index=False
        )
        pd.DataFrame(fit_rows).to_csv(
            output_dir / "figure4C_correlation_length_fits.csv", index=False
        )
        completion.write_text(json.dumps({
            "status": "complete", "subject_id": subject_id,
            "anchor_definition": "official_fsaverage5_fixed",
            "candidate_definition": "within_DK_region", "k": 15,
            "pair_seed": int(pairs.seed.iat[0]), "split_seed": 20261021,
            "minimum_fit_r2": args.min_fit_r2,
            "runtime_seconds": time.time() - started,
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }, indent=2) + "\n")
        failure.unlink(missing_ok=True)
    except Exception as error:
        failure.write_text(json.dumps({
            "status": "failed", "subject_id": subject_id,
            "error_type": type(error).__name__, "error": str(error),
            "runtime_seconds": time.time() - started,
        }, indent=2) + "\n")
        completion.unlink(missing_ok=True)
        logging.exception("[%s] formal Figure 4C failed", subject_id)
        raise


if __name__ == "__main__":
    main()
