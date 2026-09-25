#!/usr/bin/env python3
"""Merge fsaverage mapping-QC shards and compute cohort frequencies."""
from pathlib import Path
import argparse
import pandas as pd
from fsaverage_mapping_qc import read_annot, cross_subject_frequency, write_csv, HEMIS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--fsaverage-dir", type=Path, required=True)
    ap.add_argument("--coverage-dir", type=Path, required=True)
    args = ap.parse_args()
    shards = sorted(args.shards_root.glob("shard_*"))
    qc_parts = [pd.read_csv(p/"fsaverage_subject_region_mapping_qc.csv") for p in shards if (p/"fsaverage_subject_region_mapping_qc.csv").is_file()]
    if len(qc_parts) != len(shards) or not qc_parts:
        raise RuntimeError(f"Expected one QC CSV per shard, found {len(qc_parts)}/{len(shards)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qc = pd.concat(qc_parts, ignore_index=True).drop_duplicates(["subject_id","hemisphere","region"])
    qc.to_csv(args.output_dir/"fsaverage_subject_region_mapping_qc.csv", index=False)
    failures = pd.concat([pd.read_csv(p/"fsaverage_mapping_failures.csv") for p in shards if (p/"fsaverage_mapping_failures.csv").is_file()], ignore_index=True)
    failures.to_csv(args.output_dir/"fsaverage_mapping_failures.csv", index=False)
    skipped = pd.concat([pd.read_csv(p/"fsaverage_mapping_skipped_subjects.csv") for p in shards if (p/"fsaverage_mapping_skipped_subjects.csv").is_file()], ignore_index=True)
    skipped.to_csv(args.output_dir/"fsaverage_mapping_skipped_subjects.csv", index=False)
    labels, indices = {}, {}
    for hemi in HEMIS:
        labels[hemi], indices[hemi] = read_annot(args.fsaverage_dir/"label"/f"{hemi}.aparc.annot")
    vertices, regions = cross_subject_frequency(args.coverage_dir/"cross_subject_template_vertex_coverage.csv", args.output_dir/"fsaverage_subject_region_mapping_qc.csv", labels, indices)
    write_csv(args.output_dir/"fsaverage_template_vertex_coverage_frequency.csv", vertices, list(vertices[0]))
    write_csv(args.output_dir/"fsaverage_region_coverage_frequency_summary.csv", regions, list(regions[0]))
    print(f"Merged {len(shards)} shards: {len(qc)} subject-region rows, {len(failures)} failures, {len(vertices)} template vertices")

if __name__ == "__main__":
    main()
