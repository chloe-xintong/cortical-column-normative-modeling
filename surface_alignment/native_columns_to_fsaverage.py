#!/usr/bin/env python3
"""Create non-resampled native-column to fsaverage correspondence tables.

This program never changes profile CSVs, coordinate MAT files, or cleaning
JSON files.  It reconstructs the original native vertex represented by each
cleaned profile row and assigns that vertex to its nearest fsaverage
registration-sphere vertex.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel.freesurfer.io as fsio
import numpy as np
from scipy.spatial import cKDTree


DEPTHS = 21


@dataclass(frozen=True)
class SurfaceSet:
    white: np.ndarray
    pial: np.ndarray
    sphere_reg: np.ndarray
    inflated: np.ndarray | None = None


def read_surface(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required surface is missing: {path}")
    vertices, _ = fsio.read_geometry(str(path))
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Unexpected surface shape at {path}: {vertices.shape}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"Surface contains NaN/Inf coordinates: {path}")
    return vertices


def load_surfaces(subject_dir: Path, hemisphere: str, *, inflated: bool) -> SurfaceSet:
    surf = subject_dir / "surf"
    white = read_surface(surf / f"{hemisphere}.white")
    pial = read_surface(surf / f"{hemisphere}.pial")
    sphere_reg = read_surface(surf / f"{hemisphere}.sphere.reg")
    optional_inflated = read_surface(surf / f"{hemisphere}.inflated") if inflated else None
    sizes = {white.shape[0], pial.shape[0], sphere_reg.shape[0]}
    if optional_inflated is not None:
        sizes.add(optional_inflated.shape[0])
    if len(sizes) != 1:
        raise ValueError(
            f"Native {hemisphere} vertex counts disagree in {subject_dir}: "
            f"white={len(white)}, pial={len(pial)}, sphere.reg={len(sphere_reg)}"
        )
    return SurfaceSet(white, pial, sphere_reg, optional_inflated)


def unit_sphere(coords: np.ndarray, label: str) -> np.ndarray:
    radii = np.linalg.norm(coords, axis=1)
    if not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError(f"{label} has non-finite or zero-radius sphere coordinates")
    return coords / radii[:, None]


def nearest_template_vertices(native_sphere: np.ndarray, template_sphere: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest template id, normalized template coordinates, angle degrees."""
    native_unit = unit_sphere(native_sphere, "native sphere.reg")
    template_unit = unit_sphere(template_sphere, "fsaverage sphere.reg")
    _, ids = cKDTree(template_unit).query(native_unit, k=1)
    ids = np.asarray(ids, dtype=np.int64)
    dot = np.einsum("ij,ij->i", native_unit, template_unit[ids])
    angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    if not np.isfinite(angle).all():
        raise ValueError("Nearest-neighbour matching produced non-finite angular distances")
    return ids, template_unit, angle


def load_annotation(subject_dir: Path, hemisphere: str) -> tuple[np.ndarray, dict[str, int]]:
    path = subject_dir / "label" / f"{hemisphere}.aparc.annot"
    if not path.is_file():
        raise FileNotFoundError(f"Annotation is missing: {path}")
    labels, _, names = fsio.read_annot(str(path))
    labels = np.asarray(labels, dtype=np.int64)
    names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names]
    return labels, {name: i for i, name in enumerate(names)}


def load_profile(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Cleaned MD profile is missing: {path}")
    if path.stat().st_size == 0:
        return np.empty((0, DEPTHS), dtype=float)
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != DEPTHS:
        raise ValueError(f"Expected N x {DEPTHS} profile at {path}, found {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"Cleaned profile unexpectedly contains NaN/Inf: {path}")
    return data


def load_bad_mask(path: Path, *, allow_missing: bool) -> dict[str, list[int]]:
    if not path.is_file():
        if allow_missing:
            logging.warning("No cleaning mask at %s; assuming no deleted columns", path)
            return {}
        raise FileNotFoundError(f"Cleaning mask is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("indexing") != "0-based trajectory row indices before cleaning":
        raise ValueError(f"Unexpected mask indexing declaration in {path}: {data.get('indexing')!r}")
    result: dict[str, list[int]] = {}
    for region_key, entry in data.get("regions", {}).items():
        indices = [int(i) for i in entry.get("bad_indices", [])]
        if indices != sorted(set(indices)):
            raise ValueError(f"Mask indices are not sorted/unique for {region_key} in {path}")
        result[region_key] = indices
    return result


def profile_files(profile_dir: Path, subject: str, hemisphere: str, metric: str) -> Iterable[tuple[str, Path]]:
    prefix = f"{subject}_{hemisphere}_"
    suffix = f"_{metric}.csv"
    for path in sorted(profile_dir.glob(f"{prefix}*{suffix}")):
        name = path.name
        if name.startswith(prefix) and name.endswith(suffix):
            yield name[len(prefix):-len(suffix)], path


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


def write_csv_gz(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_region_rows(subject: str, hemisphere: str, region: str, native_ids: np.ndarray, profiles: dict[str, np.ndarray], bad_rows: list[int], native: SurfaceSet, template: SurfaceSet, native_unit: np.ndarray, template_ids: np.ndarray, template_unit: np.ndarray, angle: np.ndarray) -> tuple[list[dict], dict]:
    """Build one shared spatial row per column, with all requested metrics."""
    profile = profiles["md"]
    for metric, values in profiles.items():
        if values.shape != profile.shape:
            raise ValueError(
                f"Metric row/depth mismatch for {subject} {hemisphere}_{region}: "
                f"md={profile.shape}, {metric}={values.shape}"
            )
    n_original = len(native_ids)
    bad = np.asarray(bad_rows, dtype=np.int64)
    if np.any(bad < 0) or np.any(bad >= n_original):
        raise IndexError(f"Bad row outside {hemisphere}_{region}: {bad_rows[:10]}")
    keep = np.ones(n_original, dtype=bool)
    keep[bad] = False
    if profile.shape[0] != int(keep.sum()):
        raise ValueError(
            f"Row recovery failed for {subject} {hemisphere}_{region}: cleaned MD rows={profile.shape[0]}, "
            f"original annotation vertices={n_original}, bad rows={len(bad)}"
        )
    cleaned_rows = np.full(n_original, -1, dtype=int)
    cleaned_rows[keep] = np.arange(keep.sum(), dtype=int)
    fields = ["x", "y", "z"]
    rows: list[dict] = []
    for original_row, native_id in enumerate(native_ids):
        native_id = int(native_id)
        template_id = int(template_ids[native_id])
        record = {
            "subject_id": subject, "hemisphere": hemisphere, "region": region,
            "original_region_row": original_row, "cleaned_region_row": int(cleaned_rows[original_row]),
            "native_vertex_id": native_id, "column_valid": bool(keep[original_row]),
            "template_vertex_id": template_id,
            "mapping_angular_distance_deg": float(angle[native_id]),
        }
        for prefix, coords in [("native_white", native.white[native_id]), ("native_pial", native.pial[native_id]), ("registered_sphere", native_unit[native_id]), ("template_sphere", template_unit[template_id]), ("template_white", template.white[template_id]), ("template_pial", template.pial[template_id])]:
            record.update({f"{prefix}_{axis}": float(value) for axis, value in zip(fields, coords)})
        if template.inflated is not None:
            record.update({f"template_inflated_{axis}": float(value) for axis, value in zip(fields, template.inflated[template_id])})
        for metric, metric_profile in profiles.items():
            values = metric_profile[cleaned_rows[original_row]] if keep[original_row] else np.full(DEPTHS, np.nan)
            record.update({f"{metric.upper()}_depth_{depth}": float(value) for depth, value in enumerate(values)})
        rows.append(record)
    valid_template = template_ids[native_ids[keep]]
    all_template = template_ids[native_ids]
    summary = {
        "subject_id": subject, "hemisphere": hemisphere, "region": region,
        "original_columns": n_original, "valid_columns": int(keep.sum()), "invalid_columns": int((~keep).sum()),
        "cleaned_md_rows": int(profile.shape[0]), "unique_template_vertices_valid": int(np.unique(valid_template).size),
        "unique_template_vertices_before_cleaning": int(np.unique(all_template).size),
        "collision_columns_valid": int(keep.sum() - np.unique(valid_template).size),
        "collision_pct_valid": 100.0 * (1.0 - np.unique(valid_template).size / keep.sum()) if keep.any() else float("nan"),
        "template_coverage_pct_before_cleaning": 100.0 * np.unique(all_template).size / n_original if n_original else float("nan"),
        "template_coverage_pct_after_cleaning": 100.0 * np.unique(valid_template).size / keep.sum() if keep.any() else float("nan"),
        "mapping_distance_median_deg": percentile(angle[native_ids], 50),
        "mapping_distance_p95_deg": percentile(angle[native_ids], 95),
        "mapping_distance_max_deg": percentile(angle[native_ids], 100),
        "unmapped_columns": 0,
    }
    return rows, summary


def process_subject(args: argparse.Namespace, subject: str) -> None:
    subject_dir = args.subjects_dir / subject
    profile_dir = args.profiles_root / subject / "profiles"
    output_subject = args.output_root / subject
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"Subject FreeSurfer directory is missing: {subject_dir}")
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Subject cleaned profile directory is missing: {profile_dir}")
    mask = load_bad_mask(profile_dir / args.bad_mask_name, allow_missing=args.allow_missing_mask)
    template = {hemi: load_surfaces(args.fsaverage_dir, hemi, inflated=args.include_inflated) for hemi in args.hemispheres}
    summaries: list[dict] = []
    subject_all_template_ids: set[tuple[str, int]] = set()
    subject_valid_template_ids: set[tuple[str, int]] = set()
    subject_mapping_distances: list[float] = []
    for hemi in args.hemispheres:
        native = load_surfaces(subject_dir, hemi, inflated=False)
        labels, region_index = load_annotation(subject_dir, hemi)
        if len(labels) != len(native.white):
            raise ValueError(f"Annotation/native vertex count mismatch for {subject} {hemi}: {len(labels)} vs {len(native.white)}")
        native_unit = unit_sphere(native.sphere_reg, "native sphere.reg")
        template_ids, template_unit, angle = nearest_template_vertices(native.sphere_reg, template[hemi].sphere_reg)
        found = 0
        for region, profile_path in profile_files(profile_dir, subject, hemi, "md"):
            found += 1
            if region not in region_index:
                raise KeyError(f"Profile region {hemi}_{region} is absent from {hemi}.aparc.annot")
            native_ids = np.flatnonzero(labels == region_index[region])
            profiles = {"md": load_profile(profile_path)}
            for metric in args.metrics:
                if metric == "md":
                    continue
                profiles[metric] = load_profile(profile_dir / f"{subject}_{hemi}_{region}_{metric}.csv")
            region_key = f"{hemi}_{region}"
            rows, summary = build_region_rows(subject, hemi, region, native_ids, profiles, mask.get(region_key, []), native, template[hemi], native_unit, template_ids, template_unit, angle)
            summaries.append(summary)
            subject_all_template_ids.update((hemi, int(row["template_vertex_id"])) for row in rows)
            subject_valid_template_ids.update((hemi, int(row["template_vertex_id"])) for row in rows if row["column_valid"])
            subject_mapping_distances.extend(float(row["mapping_angular_distance_deg"]) for row in rows)
            destination = output_subject / hemi / "regions" / f"{subject}_{hemi}_{region}_native_to_fsaverage.csv.gz"
            if destination.exists() and not args.overwrite and not args.dry_run:
                raise FileExistsError(f"Refusing to overwrite existing mapping: {destination}; use --overwrite")
            logging.info("%s %s: original=%d valid=%d", subject, region_key, summary["original_columns"], summary["valid_columns"])
            if not args.dry_run:
                write_csv_gz(destination, list(rows[0]) if rows else [], rows)
        if not found:
            raise RuntimeError(f"No MD profile files found for {subject} {hemi} in {profile_dir}")
    if summaries:
        subject_summary = {
            "subject_id": subject,
            "regions": len(summaries),
            "original_columns": sum(x["original_columns"] for x in summaries),
            "valid_columns": sum(x["valid_columns"] for x in summaries),
            "invalid_columns": sum(x["invalid_columns"] for x in summaries),
            "cleaned_md_rows": sum(x["cleaned_md_rows"] for x in summaries),
            "unmapped_columns": sum(x["unmapped_columns"] for x in summaries),
            "unique_template_vertices_before_cleaning": len(subject_all_template_ids),
            "unique_template_vertices_valid": len(subject_valid_template_ids),
            "collision_columns_valid": sum(x["valid_columns"] for x in summaries) - len(subject_valid_template_ids),
            "collision_pct_valid": 100.0 * (1.0 - len(subject_valid_template_ids) / sum(x["valid_columns"] for x in summaries)),
            "template_coverage_pct_before_cleaning": 100.0 * len(subject_all_template_ids) / sum(x["original_columns"] for x in summaries),
            "template_coverage_pct_after_cleaning": 100.0 * len(subject_valid_template_ids) / sum(x["valid_columns"] for x in summaries),
            "mapping_distance_median_deg": percentile(np.asarray(subject_mapping_distances), 50),
            "mapping_distance_p95_deg": percentile(np.asarray(subject_mapping_distances), 95),
            "mapping_distance_max_deg": percentile(np.asarray(subject_mapping_distances), 100),
        }
        logging.info("%s totals: original=%d valid=%d invalid=%d", subject, subject_summary["original_columns"], subject_summary["valid_columns"], subject_summary["invalid_columns"])
        if not args.dry_run:
            write_csv(output_subject / f"{subject}_region_mapping_qc.csv", list(summaries[0]), summaries)
            write_csv(output_subject / f"{subject}_mapping_qc.csv", list(subject_summary), [subject_summary])


def parse_subjects(args: argparse.Namespace) -> list[str]:
    if args.subject_id:
        return [args.subject_id]
    subjects = []
    for raw in args.subjects_file.read_text(encoding="utf-8").splitlines():
        value = raw.strip().split(",")[0]
        if value and not value.startswith("#") and value.lower() not in {"subject", "subject_id"}:
            subjects.append(value)
    if not subjects:
        raise ValueError(f"No subject IDs found in {args.subjects_file}")
    return subjects


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject-id")
    group.add_argument("--subjects-file", type=Path, help="One subject ID per line; first CSV column is also accepted.")
    parser.add_argument("--subjects-dir", type=Path, required=True, help="FreeSurfer SUBJECTS_DIR containing native subjects.")
    parser.add_argument("--fsaverage-dir", type=Path, required=True, help="FreeSurfer fsaverage subject directory.")
    parser.add_argument("--profiles-root", type=Path, required=True, help="Root containing <subject>/profiles cleaned CSVs.")
    parser.add_argument("--output-root", type=Path, required=True, help="New mapping output root.")
    parser.add_argument("--metrics", nargs="+", default=["md", "fa", "ad", "rd"], choices=["md", "fa", "ad", "rd"], help="Metrics stored in each shared-coordinate master table.")
    parser.add_argument("--hemispheres", nargs="+", choices=["lh", "rh"], default=["lh", "rh"])
    parser.add_argument("--bad-mask-name", default="bad_trajectory_mask.json")
    parser.add_argument("--include-inflated", action="store_true")
    parser.add_argument("--allow-missing-mask", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate all inputs and row recovery without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of this module's own mapping files.")
    parser.add_argument("--log-file", type=Path, help="Optional log-file path in addition to stderr.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    if "md" not in args.metrics:
        parser.error("--metrics must include md because MD files define the region traversal")
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)
    for path_name in ("subjects_dir", "fsaverage_dir", "profiles_root"):
        path = getattr(args, path_name)
        if not path.is_dir():
            parser.error(f"--{path_name.replace('_', '-')} is not a directory: {path}")
    if args.subjects_file and not args.subjects_file.is_file():
        parser.error(f"--subjects-file is not a file: {args.subjects_file}")
    try:
        for subject in parse_subjects(args):
            process_subject(args, subject)
    except Exception as exc:
        logging.error("Surface alignment failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
