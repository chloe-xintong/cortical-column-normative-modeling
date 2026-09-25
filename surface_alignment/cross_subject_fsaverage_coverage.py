#!/usr/bin/env python3
"""Summarize cross-subject coverage of native columns on fsaverage vertices.

This is deliberately a spatial correspondence QC, not an MD resampling step.
It reads only mapping identity fields and never averages, interpolates, or
rewrites diffusion metric values.  Incomplete subjects are logged and skipped.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_subject_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    values = {line.strip().split(",")[0] for line in path.read_text().splitlines()}
    values.discard(""); values.discard("subject_id")
    return values


def mapping_files(subject_dir: Path) -> list[Path]:
    return sorted(subject_dir.glob("[lr]h/regions/*_native_to_fsaverage.csv.gz"))


def parse_file_identity(path: Path) -> tuple[str, str]:
    hemi = path.parent.parent.name
    suffix = f"_{hemi}_"
    name = path.name
    subject = path.parents[2].name
    start = f"{subject}{suffix}"
    end = "_native_to_fsaverage.csv.gz"
    if not name.startswith(start) or not name.endswith(end):
        raise ValueError(f"Unexpected mapping filename: {path}")
    return hemi, name[len(start):-len(end)]


def process_file(path: Path) -> tuple[dict, dict[tuple[str, str, int], list[int]]]:
    hemi, region = parse_file_identity(path)
    subject = path.parents[2].name
    counts: dict[int, int] = defaultdict(int)
    total = valid = 0
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"column_valid", "template_vertex_id", "mapping_angular_distance_deg"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Mapping file lacks required fields: {path}")
        for row in reader:
            total += 1
            if row["column_valid"] != "True":
                continue
            vertex = int(row["template_vertex_id"])
            distance = float(row["mapping_angular_distance_deg"])
            if vertex < 0 or not np.isfinite(distance):
                raise ValueError(f"Invalid template mapping values in {path}")
            valid += 1
            counts[vertex] += 1
    if total == 0:
        raise ValueError(f"Empty mapping table: {path}")
    summary = {
        "subject_id": subject, "hemisphere": hemi, "region": region,
        "original_columns": total, "valid_columns": valid,
        "invalid_columns": total - valid,
        "unique_template_vertices_valid": len(counts),
        "collision_columns_valid": valid - len(counts),
        "collision_pct_valid": 100.0 * (valid - len(counts)) / valid if valid else float("nan"),
    }
    updates = {(hemi, region, vertex): [1, count, count - 1] for vertex, count in counts.items()}
    return summary, updates


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["subject_id", "reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--subjects-file", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()
    selected = read_subject_filter(args.subjects_file)
    subjects = [p for p in sorted(args.input_root.glob("D*")) if p.is_dir() and (selected is None or p.name in selected)]
    template_stats: dict[tuple[str, str, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    subject_region_rows: list[dict] = []
    skipped: list[dict] = []
    processed_files = 0
    for subject_dir in subjects:
        subject = subject_dir.name
        qc = subject_dir / f"{subject}_mapping_qc.csv"
        files = mapping_files(subject_dir)
        if not qc.is_file() or len(files) != 68:
            skipped.append({"subject_id": subject, "reason": f"incomplete_mapping_output:qc={qc.is_file()},files={len(files)}"})
            continue
        for path in files:
            try:
                row, updates = process_file(path)
            except Exception as exc:
                skipped.append({"subject_id": subject, "reason": f"{path.name}: {exc}"})
                continue
            subject_region_rows.append(row)
            for key, update in updates.items():
                target = template_stats[key]
                target[0] += update[0]  # contributing subjects: one per subject-region file
                target[1] += update[1]  # native valid columns
                target[2] += update[2]  # extra columns sharing a template vertex
            processed_files += 1
            if processed_files % args.progress_every == 0:
                logging.info("Processed %d mapping files; valid subject-region groups=%d", processed_files, len(subject_region_rows))
    template_rows = [
        {"hemisphere": hemi, "region": region, "template_vertex_id": vertex,
         "n_subjects": values[0], "n_valid_native_columns": values[1], "collision_columns": values[2],
         "collision_pct": 100.0 * values[2] / values[1] if values[1] else float("nan")}
        for (hemi, region, vertex), values in template_stats.items()
    ]
    region_rows = []
    for (hemi, region), group in _group_rows(template_rows, ("hemisphere", "region")).items():
        support = np.asarray([r["n_subjects"] for r in group], dtype=float)
        columns = sum(r["n_valid_native_columns"] for r in group)
        collisions = sum(r["collision_columns"] for r in group)
        region_rows.append({"hemisphere": hemi, "region": region,
                            "template_vertices_observed": len(group),
                            "median_subjects_per_template_vertex": float(np.median(support)),
                            "p25_subjects_per_template_vertex": float(np.percentile(support, 25)),
                            "p75_subjects_per_template_vertex": float(np.percentile(support, 75)),
                            "template_vertices_with_2plus_subjects_pct": 100.0 * float(np.mean(support >= 2)),
                            "template_vertices_with_10plus_subjects_pct": 100.0 * float(np.mean(support >= 10)),
                            "valid_native_columns": columns,
                            "collision_columns": collisions,
                            "collision_pct": 100.0 * collisions / columns if columns else float("nan")})
    if args.dry_run:
        logging.info("Dry run: complete subject outputs=%d, files=%d, template vertices=%d, skipped=%d", len({r['subject_id'] for r in subject_region_rows}), processed_files, len(template_rows), len(skipped))
        return 0
    write_csv(args.output_dir / "cross_subject_template_vertex_coverage.csv", template_rows)
    write_csv(args.output_dir / "cross_subject_region_alignment_summary.csv", region_rows)
    write_csv(args.output_dir / "cross_subject_subject_region_coverage.csv", subject_region_rows)
    write_csv(args.output_dir / "cross_subject_skipped_or_incomplete.csv", skipped)
    logging.info("Wrote %d template vertices from %d complete subjects; skipped=%d", len(template_rows), len({r['subject_id'] for r in subject_region_rows}), len(skipped))
    return 0


def _group_rows(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
