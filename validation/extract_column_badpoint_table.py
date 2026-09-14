#!/usr/bin/env python3
"""Extract bad-point locations from raw cortical profiles before cleaning.

The script is deliberately independent of Jupyter. It reads one profile CSV
at a time and writes only bad columns by default, so it does not keep the
cohort in memory and does not modify any input profile.
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PATTERN = re.compile(r"^(?P<subject>D\d+)_(?P<hemisphere>lh|rh)_(?P<region>.+)_(?P<metric>md|fa|ad|rd)\.csv$")


def classify(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf"
    if value == 0:
        return "exact_zero"
    return "valid"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles-root", type=Path, default=Path(
        "/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis"
    ))
    p.add_argument("--output-dir", type=Path, default=Path(
        "/mnt/newStor/paros/paros_WORK/xintong/results/column_quality_qc"
    ))
    p.add_argument("--depths", type=int, default=21)
    p.add_argument("--include-valid", action="store_true",
                   help="also write columns with no bad point; can create a very large file")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "column_badpoint_detail.csv"
    file_path = args.output_dir / "column_badpoint_file_summary.csv"
    if not args.overwrite and (detail_path.exists() or file_path.exists()):
        raise SystemExit(
            "Output already exists. Use --overwrite only when you intentionally want to regenerate:\n"
            f"  {detail_path}\n  {file_path}"
        )

    detail_fields = [
        "subject", "hemisphere", "region", "metric", "column_id",
        "bad_point_count", "bad_depths", "bad_types", "has_consecutive_bad_points",
    ]
    file_fields = [
        "subject", "hemisphere", "region", "metric", "profile_path", "status",
        "total_columns", "total_points", "bad_points", "bad_columns",
        "bad_point_pct", "bad_column_pct", "exact_zero_points", "nan_points", "inf_points",
    ]
    counters = defaultdict(int)
    n_files = 0
    n_detail = 0
    start_time = time.monotonic()
    total_profile_files = sum(
        1 for profile_dir in args.profiles_root.glob("D*/profiles_before_clean")
        for path in profile_dir.glob("*.csv")
        if PATTERN.match(path.name)
    )
    print(f"Total profile files: {total_profile_files}", flush=True)

    with detail_path.open("w", newline="") as detail_f, file_path.open("w", newline="") as file_f:
        detail_writer = csv.DictWriter(detail_f, fieldnames=detail_fields)
        file_writer = csv.DictWriter(file_f, fieldnames=file_fields)
        detail_writer.writeheader()
        file_writer.writeheader()

        for profile_dir in sorted(args.profiles_root.glob("D*/profiles_before_clean")):
            for path in sorted(profile_dir.glob("*.csv")):
                match = PATTERN.match(path.name)
                if not match:
                    continue
                n_files += 1
                if n_files == 1 or n_files % 100 == 0:
                    pct = 100 * n_files / total_profile_files if total_profile_files else 100
                    elapsed = time.monotonic() - start_time
                    rate = n_files / elapsed if elapsed > 0 else 0
                    remaining = (total_profile_files - n_files) / rate if rate > 0 else 0
                    eta_minutes = remaining / 60
                    print(f"[progress] {n_files}/{total_profile_files} files ({pct:.2f}%), "
                          f"elapsed={elapsed / 60:.1f} min, ETA={eta_minutes:.1f} min, "
                          f"detail rows={n_detail}, "
                          f"columns={counters['total_columns']:,}", flush=True)
                meta = match.groupdict()
                base = {**meta, "profile_path": str(path)}
                if path.stat().st_size == 0:
                    file_writer.writerow({**base, "status": "empty", "total_columns": 0,
                                          "total_points": 0, "bad_points": 0, "bad_columns": 0,
                                          "bad_point_pct": "", "bad_column_pct": "",
                                          "exact_zero_points": 0, "nan_points": 0, "inf_points": 0})
                    continue
                try:
                    array = np.loadtxt(path, delimiter=",", ndmin=2)
                    if array.ndim != 2 or array.shape[1] != args.depths:
                        raise ValueError(f"expected {args.depths} depths, got shape {array.shape}")
                except Exception as exc:
                    file_writer.writerow({**base, "status": f"unreadable: {exc}",
                                          "total_columns": 0, "total_points": 0, "bad_points": 0,
                                          "bad_columns": 0, "bad_point_pct": "", "bad_column_pct": "",
                                          "exact_zero_points": 0, "nan_points": 0, "inf_points": 0})
                    continue

                bad = (~np.isfinite(array)) | (array == 0)
                bad_points = int(bad.sum())
                bad_columns = int(np.any(bad, axis=1).sum())
                file_writer.writerow({**base, "status": "valid", "total_columns": array.shape[0],
                                      "total_points": array.size, "bad_points": bad_points,
                                      "bad_columns": bad_columns,
                                      "bad_point_pct": 100 * bad_points / array.size,
                                      "bad_column_pct": 100 * bad_columns / array.shape[0],
                                      "exact_zero_points": int((array == 0).sum()),
                                      "nan_points": int(np.isnan(array).sum()),
                                      "inf_points": int(np.isinf(array).sum())})

                for column_id, row in enumerate(array):
                    mask = (~np.isfinite(row)) | (row == 0)
                    if not args.include_valid and not mask.any():
                        continue
                    depths = np.flatnonzero(mask)
                    types = [classify(float(row[d])) for d in depths]
                    consecutive = bool(np.any(mask[:-1] & mask[1:]))
                    detail_writer.writerow({**meta, "column_id": column_id,
                                            "bad_point_count": int(mask.sum()),
                                            "bad_depths": ";".join(map(str, depths.tolist())),
                                            "bad_types": ";".join(types),
                                            "has_consecutive_bad_points": consecutive})
                    n_detail += 1

                counters["total_columns"] += int(array.shape[0])
                counters["bad_columns"] += bad_columns
                counters["bad_points"] += bad_points
                del array, bad

    print(f"Profile files processed: {n_files}")
    print(f"Detail rows written: {n_detail}")
    print(f"Total columns: {counters['total_columns']}")
    print(f"Columns with bad points: {counters['bad_columns']}")
    print(f"Bad points: {counters['bad_points']}")
    print(f"Detail table: {detail_path}")
    print(f"File summary: {file_path}")


if __name__ == "__main__":
    main()
