#!/usr/bin/env python3
"""Compute one subject's formal ico5 fixed-anchor, within-DK, K=15 real curve."""

from __future__ import annotations

import argparse
import json
import logging
import resource
import time
from pathlib import Path

import pandas as pd

from compute_real_spatial_curves import load_profiles, select_subject
from spatial_metrics import aggregate_anchor_bins, aggregate_subject_bins, pair_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--index", type=int)
    selection.add_argument("--subject-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    row = select_subject(manifest, args)
    subject_id = str(row["subject_id"])
    output_dir = args.output_root / "subjects" / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    completion = output_dir / "completion.json"
    failure = output_dir / "failure.json"
    started = time.time()
    try:
        pairs = pd.read_parquet(args.pair_file)
        if not pairs["pair_type"].eq("within_region").all():
            raise ValueError("Formal pair table must contain only within-region pairs")
        anchor_parts = []
        for hemisphere in ("lh", "rh"):
            profiles, valid = load_profiles(Path(row[f"clean_profile_{hemisphere}"]))
            current = pairs.loc[pairs["hemisphere"].eq(hemisphere)].reset_index(drop=True)
            anchor_parts.append(aggregate_anchor_bins(pair_metrics(profiles, valid, current)))
        anchor = pd.concat(anchor_parts, ignore_index=True)
        subject = aggregate_subject_bins(anchor)
        subject.insert(0, "subject_id", subject_id)
        subject.to_csv(output_dir / "figure4_ico5_k15_real_subject_bins.csv", index=False)
        completion.write_text(json.dumps({
            "status": "complete",
            "subject_id": subject_id,
            "anchor_definition": "official_fsaverage5_fixed",
            "candidate_definition": "within_DK_region",
            "k": 15,
            "pair_seed": int(pairs["seed"].iat[0]),
            "input_pair_count": int(len(pairs)),
            "valid_pair_count": int(subject["n_pairs"].sum()),
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
        logging.exception("[%s] formal real analysis failed", subject_id)
        raise


if __name__ == "__main__":
    main()
