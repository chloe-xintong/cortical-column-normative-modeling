#!/usr/bin/env python3
"""Streaming QC for native-column to fsaverage mapping tables.

This program is read-only with respect to mapping tables and cleaned profiles.
It validates that the mapping stage only added spatial identity, then writes
new QC CSV files.  It does not resample, average, or alter diffusion profiles.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import logging
from collections import Counter, defaultdict
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd

DEPTHS = 21
HEMIS = ("lh", "rh")


def read_annot(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing fsaverage annotation: {path}")
    labels, _, names = fsio.read_annot(str(path))
    decoded = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    return np.asarray(labels, dtype=np.int64), {name: i for i, name in enumerate(decoded)}


def load_cleaned_md(profiles_root: Path, subject: str, hemi: str, region: str) -> np.ndarray:
    path = profiles_root / subject / "profiles" / f"{subject}_{hemi}_{region}_md.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cleaned MD profile: {path}")
    if path.stat().st_size == 0:
        return np.empty((0, DEPTHS), dtype=float)
    data = np.asarray(np.loadtxt(path, delimiter=",", ndmin=2), dtype=float)
    if data.ndim != 2 or data.shape[1] != DEPTHS:
        raise ValueError(f"Unexpected cleaned MD shape {data.shape}: {path}")
    return data


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def mapping_files(subject_dir: Path) -> list[Path]:
    return sorted(subject_dir.glob("[lr]h/regions/*_native_to_fsaverage.csv.gz"))


def identity(path: Path) -> tuple[str, str, str]:
    subject = path.parents[2].name
    hemi = path.parent.parent.name
    start, end = f"{subject}_{hemi}_", "_native_to_fsaverage.csv.gz"
    if not path.name.startswith(start) or not path.name.endswith(end):
        raise ValueError(f"Unexpected mapping filename: {path}")
    return subject, hemi, path.name[len(start):-len(end)]


def profile_values(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row[f"MD_depth_{d}"]) for d in range(DEPTHS)], dtype=float)


def inspect_mapping(
    path: Path,
    profiles_root: Path,
    template_labels: dict[str, np.ndarray],
    template_indices: dict[str, dict[str, int]],
) -> tuple[dict, list[dict]]:
    subject, hemi, region = identity(path)
    cleaned = load_cleaned_md(profiles_root, subject, hemi, region)
    labels, indices = template_labels[hemi], template_indices[hemi]
    if region not in indices:
        raise KeyError(f"{region} missing from fsaverage {hemi}.aparc.annot")
    eligible = set(np.flatnonzero(labels == indices[region]).astype(int))
    rows: list[dict] = []
    failure_rows: list[dict] = []
    native_ids: set[int] = set()
    original_rows: set[int] = set()
    valid_target_ids: list[int] = []
    all_target_ids: list[int] = []
    distances: list[float] = []
    expected_cleaned_rows: set[int] = set()
    mapped_valid = invalid_target = md_mismatch = invalid_md = duplicate_native = duplicate_original = 0
    label_match = label_mismatch = 0

    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "subject_id", "hemisphere", "region", "original_region_row",
            "cleaned_region_row", "native_vertex_id", "column_valid",
            "template_vertex_id", "mapping_angular_distance_deg",
        } | {f"MD_depth_{d}" for d in range(DEPTHS)}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Required mapping fields missing: {path}")
        for row in reader:
            rows.append(row)
            if row["subject_id"] != subject or row["hemisphere"] != hemi or row["region"] != region:
                raise ValueError(f"Identity fields disagree with path: {path}")
            original_row = int(row["original_region_row"])
            native_id = int(row["native_vertex_id"])
            if original_row in original_rows:
                duplicate_original += 1
            original_rows.add(original_row)
            if native_id in native_ids:
                duplicate_native += 1
            native_ids.add(native_id)

            target = int(row["template_vertex_id"])
            distance = float(row["mapping_angular_distance_deg"])
            valid = row["column_valid"] == "True"
            target_ok = 0 <= target < len(labels) and np.isfinite(distance)
            if target_ok:
                all_target_ids.append(target)
                distances.append(distance)
            if valid:
                if not target_ok:
                    invalid_target += 1
                    failure_rows.append({"subject_id": subject, "hemisphere": hemi, "region": region,
                                         "original_region_row": original_row, "reason": "invalid_target"})
                    continue
                mapped_valid += 1
                valid_target_ids.append(target)
                if target in eligible:
                    label_match += 1
                else:
                    label_mismatch += 1
                cleaned_row = int(row["cleaned_region_row"])
                expected_cleaned_rows.add(cleaned_row)
                if not (0 <= cleaned_row < cleaned.shape[0]):
                    md_mismatch += 1
                    failure_rows.append({"subject_id": subject, "hemisphere": hemi, "region": region,
                                         "original_region_row": original_row, "reason": "cleaned_row_out_of_range"})
                    continue
                mapped_md = profile_values(row)
                if not np.isfinite(mapped_md).all():
                    invalid_md += 1
                    failure_rows.append({"subject_id": subject, "hemisphere": hemi, "region": region,
                                         "original_region_row": original_row, "reason": "nonfinite_valid_md"})
                elif not np.array_equal(mapped_md, cleaned[cleaned_row], equal_nan=True):
                    md_mismatch += 1
                    failure_rows.append({"subject_id": subject, "hemisphere": hemi, "region": region,
                                         "original_region_row": original_row, "reason": "md_profile_mismatch"})
            else:
                if int(row["cleaned_region_row"]) != -1 or np.isfinite(profile_values(row)).any():
                    md_mismatch += 1
                    failure_rows.append({"subject_id": subject, "hemisphere": hemi, "region": region,
                                         "original_region_row": original_row, "reason": "invalid_column_profile_not_nan"})

    n_original = len(rows)
    if expected_cleaned_rows != set(range(cleaned.shape[0])):
        md_mismatch += 1
        failure_rows.append({"subject_id": subject, "hemisphere": hemi, "region": region,
                             "original_region_row": -1, "reason": "cleaned_row_set_mismatch"})
    valid_counter = Counter(valid_target_ids)
    all_counter = Counter(all_target_ids)
    eligible_valid = set(valid_target_ids) & eligible
    eligible_all = set(all_target_ids) & eligible
    multiplicity = Counter(valid_counter.values())
    collision_vertices = sum(count >= 2 for count in valid_counter.values())
    collision_columns = sum(count for count in valid_counter.values() if count >= 2)
    summary = {
        "subject_id": subject, "hemisphere": hemi, "region": region,
        "original_columns": n_original, "cleaned_md_rows": int(cleaned.shape[0]),
        "valid_columns": int(sum(1 for r in rows if r["column_valid"] == "True")),
        "mapped_valid_columns": mapped_valid,
        "mapping_rate_pct": 100 * mapped_valid / cleaned.shape[0] if cleaned.shape[0] else float("nan"),
        "invalid_columns": n_original - int(sum(1 for r in rows if r["column_valid"] == "True")),
        "invalid_target_count": invalid_target, "md_profile_mismatch_count": md_mismatch,
        "invalid_md_count": invalid_md, "duplicate_native_vertex_rows": duplicate_native,
        "duplicate_original_region_rows": duplicate_original,
        "unique_template_vertices_valid": len(valid_counter),
        "unique_template_vertices_before_cleaning": len(all_counter),
        "eligible_template_vertices": len(eligible),
        "eligible_template_vertices_covered_valid": len(eligible_valid),
        "eligible_template_vertices_covered_before_cleaning": len(eligible_all),
        "regional_coverage_valid_pct": 100 * len(eligible_valid) / len(eligible) if eligible else float("nan"),
        "regional_coverage_before_cleaning_pct": 100 * len(eligible_all) / len(eligible) if eligible else float("nan"),
        "theoretical_max_coverage_before_cleaning_pct": 100 * min(n_original, len(eligible)) / len(eligible) if eligible else float("nan"),
        "theoretical_max_coverage_after_cleaning_pct": 100 * min(mapped_valid, len(eligible)) / len(eligible) if eligible else float("nan"),
        "coverage_efficiency_before_cleaning_pct": 100 * len(eligible_all) / min(n_original, len(eligible)) if min(n_original, len(eligible)) else float("nan"),
        "coverage_efficiency_after_cleaning_pct": 100 * len(eligible_valid) / min(mapped_valid, len(eligible)) if min(mapped_valid, len(eligible)) else float("nan"),
        "coverage_loss_after_cleaning_pct": 100 * (len(eligible_all) - len(eligible_valid)) / len(eligible) if eligible else float("nan"),
        "collision_vertex_rate_pct": 100 * collision_vertices / len(valid_counter) if valid_counter else float("nan"),
        "colliding_column_rate_pct": 100 * collision_columns / mapped_valid if mapped_valid else float("nan"),
        "median_multiplicity": float(np.median(list(valid_counter.values()))) if valid_counter else float("nan"),
        "maximum_multiplicity": max(valid_counter.values(), default=0),
        "multiplicity_1_vertices": multiplicity[1], "multiplicity_2_vertices": multiplicity[2],
        "multiplicity_3_vertices": multiplicity[3],
        "multiplicity_4_vertices": multiplicity[4],
        "multiplicity_5_vertices": multiplicity[5],
        "multiplicity_6plus_vertices": sum(v for k, v in multiplicity.items() if k >= 6),
        "mapping_distance_median_deg": percentile(distances, 50),
        "mapping_distance_p95_deg": percentile(distances, 95),
        "mapping_distance_max_deg": percentile(distances, 100),
        "region_label_agreement_pct": 100 * label_match / mapped_valid if mapped_valid else float("nan"),
        "region_label_mismatch_count": label_mismatch,
    }
    return summary, failure_rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cross_subject_frequency(
    coverage_csv: Path, subject_region_csv: Path, template_labels: dict[str, np.ndarray],
    template_indices: dict[str, dict[str, int]],
) -> tuple[list[dict], list[dict]]:
    vertices = pd.read_csv(coverage_csv)
    units = pd.read_csv(subject_region_csv)
    denominator = units.groupby(["hemisphere", "region"])["subject_id"].nunique().rename("eligible_subjects").reset_index()
    vertices = vertices.merge(denominator, on=["hemisphere", "region"], how="left")
    vertices["coverage_frequency_pct"] = 100 * vertices["n_subjects"] / vertices["eligible_subjects"]
    region_rows: list[dict] = []
    for (hemi, region), group in vertices.groupby(["hemisphere", "region"], sort=True):
        labels, indices = template_labels[hemi], template_indices[hemi]
        eligible_count = int(np.sum(labels == indices[region]))
        frequency = group["coverage_frequency_pct"].to_numpy(dtype=float)
        region_rows.append({
            "hemisphere": hemi, "region": region, "eligible_template_vertices": eligible_count,
            "observed_template_vertices": int(len(group)),
            "median_coverage_frequency_pct": float(np.median(frequency)),
            "p25_coverage_frequency_pct": float(np.percentile(frequency, 25)),
            "p75_coverage_frequency_pct": float(np.percentile(frequency, 75)),
            "vertices_covered_by_all_subjects": int(np.sum(group.n_subjects == group.eligible_subjects)),
            "vertices_covered_by_80pct_subjects": int(np.sum(group.coverage_frequency_pct >= 80)),
            "vertices_covered_by_90pct_subjects": int(np.sum(group.coverage_frequency_pct >= 90)),
            "vertices_covered_by_95pct_subjects": int(np.sum(group.coverage_frequency_pct >= 95)),
        })
    return vertices.to_dict("records"), region_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapping-root", type=Path, required=True)
    ap.add_argument("--profiles-root", type=Path, required=True)
    ap.add_argument("--fsaverage-dir", type=Path, required=True)
    ap.add_argument("--cross-subject-coverage-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    subject_group = ap.add_mutually_exclusive_group()
    subject_group.add_argument("--subjects-file", type=Path)
    subject_group.add_argument("--subject-id")
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--skip-frequency", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    selected = None
    if args.subjects_file:
        selected = {x.strip().split(",")[0] for x in args.subjects_file.read_text().splitlines() if x.strip() and not x.lower().startswith("subject")}
    elif args.subject_id:
        selected = {args.subject_id}
    template_labels, template_indices = {}, {}
    for hemi in HEMIS:
        labels, indices = read_annot(args.fsaverage_dir / "label" / f"{hemi}.aparc.annot")
        template_labels[hemi], template_indices[hemi] = labels, indices
    subjects = [p for p in sorted(args.mapping_root.glob("D*")) if p.is_dir() and (selected is None or p.name in selected)]
    if args.shard_index is not None:
        if args.num_shards is None or not 0 <= args.shard_index < args.num_shards:
            ap.error("--shard-index requires --num-shards and must be in range")
        subjects = [p for i, p in enumerate(subjects) if i % args.num_shards == args.shard_index]
    complete_subjects = [p for p in subjects if len(mapping_files(p)) == 68]
    total_files = len(complete_subjects) * 68
    summaries: list[dict] = []
    failures: list[dict] = []
    skipped: list[dict] = []
    processed = 0
    for subject_dir in subjects:
        files = mapping_files(subject_dir)
        if len(files) != 68:
            skipped.append({"subject_id": subject_dir.name, "reason": f"incomplete_mapping_files={len(files)}"})
            continue
        for path in files:
            try:
                summary, detail = inspect_mapping(path, args.profiles_root, template_labels, template_indices)
                summaries.append(summary); failures.extend(detail)
            except Exception as exc:
                failures.append({"subject_id": subject_dir.name, "hemisphere": "", "region": "",
                                 "original_region_row": -1, "reason": f"{path.name}: {exc}"})
            processed += 1
            if processed % args.progress_every == 0:
                pct = 100.0 * processed / total_files if total_files else 100.0
                bar_width = 30
                filled = int(bar_width * pct / 100.0)
                bar = "#" * filled + "-" * (bar_width - filled)
                logging.info("[%s] %6.2f%% (%d/%d) subject-region files; summaries=%d; failures=%d",
                             bar, pct, processed, total_files, len(summaries), len(failures))
    if args.dry_run:
        logging.info("Dry run complete: subjects=%d summaries=%d failures=%d skipped=%d", len(subjects), len(summaries), len(failures), len(skipped))
        return 1 if failures else 0
    if not summaries:
        raise RuntimeError("No complete mapping subjects were available for QC")
    base_fields = list(summaries[0])
    write_csv(args.output_dir / "fsaverage_subject_region_mapping_qc.csv", summaries, base_fields)
    write_csv(args.output_dir / "fsaverage_mapping_failures.csv", failures, ["subject_id", "hemisphere", "region", "original_region_row", "reason"])
    write_csv(args.output_dir / "fsaverage_mapping_skipped_subjects.csv", skipped, ["subject_id", "reason"])
    frequency_rows = []
    if not args.skip_frequency:
        coverage_csv = args.cross_subject_coverage_dir / "cross_subject_template_vertex_coverage.csv"
        frequency_rows, frequency_regions = cross_subject_frequency(
            coverage_csv, args.output_dir / "fsaverage_subject_region_mapping_qc.csv", template_labels, template_indices
        )
        write_csv(args.output_dir / "fsaverage_template_vertex_coverage_frequency.csv", frequency_rows, list(frequency_rows[0]))
        write_csv(args.output_dir / "fsaverage_region_coverage_frequency_summary.csv", frequency_regions, list(frequency_regions[0]))
    logging.info("Wrote %d subject-region QC rows; %d failures; %d template vertices", len(summaries), len(failures), len(frequency_rows))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
