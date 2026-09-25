#!/usr/bin/env python3
"""Resample native-column metric profiles onto fsaverage vertices.

The input is produced by ``native_columns_to_fsaverage.py``.  Spatial mapping
is therefore not repeated here: valid rows are grouped by template vertex and
their metric depth profiles are reduced within each group.
"""
from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np

DEPTHS = 21
METRICS = ("MD", "FA", "AD", "RD")


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def reduce_profiles(rows: list[dict[str, str]], method: str) -> list[dict]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("column_valid", "False").lower() == "true":
            groups[(row["hemisphere"], int(row["template_vertex_id"]))].append(row)

    output: list[dict] = []
    for (hemi, vertex), members in sorted(groups.items()):
        record = {"hemisphere": hemi, "template_vertex_id": vertex,
                  "n_native_columns": len(members),
                  "collision": len(members) > 1}
        for metric in METRICS:
            matrix = np.asarray([[float(r[f"{metric}_depth_{d}"]) for d in range(DEPTHS)]
                                 for r in members], dtype=float)
            if method == "retain":
                # One row per source column is written by the caller.
                continue
            reducer = np.nanmedian if method == "median" else np.nanmean
            values = reducer(matrix, axis=0)
            record.update({f"{metric}_depth_{d}": float(values[d]) for d in range(DEPTHS)})
        output.append(record)
    return output


def fields(method: str) -> list[str]:
    base = ["hemisphere", "template_vertex_id", "n_native_columns", "collision"]
    if method == "retain":
        return base + ["native_vertex_id", "original_region_row"] + [
            f"{m}_depth_{d}" for m in METRICS for d in range(DEPTHS)]
    return base + [f"{m}_depth_{d}" for m in METRICS for d in range(DEPTHS)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapping-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--subject-id")
    p.add_argument("--method", choices=("mean", "median", "retain"), default="mean")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    subjects = [args.subject_id] if args.subject_id else [x.name for x in sorted(args.mapping_root.iterdir()) if x.is_dir()]
    for subject in subjects:
        files = sorted((args.mapping_root / subject).glob("[lr]h/regions/*_native_to_fsaverage.csv.gz"))
        if not files:
            raise FileNotFoundError(f"No mapping tables found for {subject}")
        by_hemi: dict[str, list[dict[str, str]]] = defaultdict(list)
        for path in files:
            rows = read_rows(path)
            for row in rows:
                by_hemi[row["hemisphere"]].append(row)
        for hemi, rows in by_hemi.items():
            destination = args.output_root / subject / f"{subject}_{hemi}_fsaverage_profiles.csv"
            if destination.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {destination}; use --overwrite")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if args.method == "retain":
                result = []
                grouped = defaultdict(list)
                for row in rows:
                    if row.get("column_valid", "False").lower() == "true":
                        grouped[int(row["template_vertex_id"])].append(row)
                for vertex, members in sorted(grouped.items()):
                    for row in members:
                        out = {"hemisphere": hemi, "template_vertex_id": vertex,
                               "n_native_columns": len(members), "collision": len(members) > 1,
                               "native_vertex_id": row["native_vertex_id"],
                               "original_region_row": row["original_region_row"]}
                        out.update({f"{m}_depth_{d}": row[f"{m}_depth_{d}"] for m in METRICS for d in range(DEPTHS)})
                        result.append(out)
            else:
                result = reduce_profiles(rows, args.method)
            with destination.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields(args.method))
                writer.writeheader(); writer.writerows(result)
            print(f"{subject} {hemi}: {len(rows)} columns -> {len(result)} fsaverage vertices ({args.method})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
