#!/usr/bin/env python3
"""Column retention curves and a lightweight interpolation sanity check.

Reads the completed bad-column detail/file-summary CSVs. Raw profile arrays
are touched one file at a time only for the optional interpolation test.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PATTERN = re.compile(r"^(?P<subject>D\d+)_(?P<hemisphere>lh|rh)_(?P<region>.+)_(?P<metric>md|fa|ad|rd)\.csv$")


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qc-root', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/results/column_quality_qc'))
    p.add_argument('--profiles-root', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis'))
    p.add_argument('--depths', type=int, default=21)
    p.add_argument('--max-samples', type=int, default=10000,
                   help='maximum clean columns used for interpolation validation')
    p.add_argument('--seed', type=int, default=20260911)
    return p.parse_args()


def interpolate_at(values: np.ndarray, missing: list[int]) -> np.ndarray:
    observed = np.ones(values.size, dtype=bool)
    observed[missing] = False
    x = np.arange(values.size)
    return np.interp(x[missing], x[observed], values[observed])


def main():
    a = args()
    a.qc_root.mkdir(parents=True, exist_ok=True)
    detail_path = a.qc_root / 'column_badpoint_detail.csv'
    file_path = a.qc_root / 'column_badpoint_file_summary.csv'
    if not detail_path.is_file() or not file_path.is_file():
        raise SystemExit('Missing completed column_badpoint_detail.csv or column_badpoint_file_summary.csv')

    # One row per bad column; columns absent from detail have zero bad points.
    detail = pd.read_csv(detail_path, usecols=['subject', 'bad_point_count'])
    bad_dist = detail.groupby(['subject', 'bad_point_count']).size().unstack(fill_value=0)
    files = pd.read_csv(file_path, usecols=['subject', 'total_columns', 'status'])
    totals = files[files.status == 'valid'].groupby('subject').total_columns.sum()
    rows = []
    for subject, total in totals.items():
        counts = Counter({int(k): int(v) for k, v in bad_dist.loc[subject].items()}) if subject in bad_dist.index else Counter()
        n_bad_columns = sum(counts.values())
        zero_bad = int(total) - n_bad_columns
        row = {'subject': subject, 'total_columns': int(total), 'bad_columns': n_bad_columns}
        for allowed in range(6):
            retained = zero_bad + sum(v for k, v in counts.items() if k <= allowed)
            row[f'retention_allowed_{allowed}'] = retained / int(total) * 100
            row[f'columns_retained_allowed_{allowed}'] = retained
        row['bad_point_pct'] = 100 * sum(k * v for k, v in counts.items()) / (int(total) * a.depths)
        rows.append(row)
    retention = pd.DataFrame(rows).sort_values('subject')
    retention.to_csv(a.qc_root / 'subject_column_retention_curve.csv', index=False)
    summary = []
    for allowed in range(6):
        col = f'retention_allowed_{allowed}'
        summary.append({'allowed_bad_points': allowed, 'median_retention_pct': retention[col].median(),
                        'q1_retention_pct': retention[col].quantile(.25),
                        'q3_retention_pct': retention[col].quantile(.75),
                        'min_retention_pct': retention[col].min(), 'max_retention_pct': retention[col].max()})
    pd.DataFrame(summary).to_csv(a.qc_root / 'population_column_retention_summary.csv', index=False)

    rng = random.Random(a.seed)
    errors = defaultdict(list)
    tested = defaultdict(int)
    paths = sorted(p for d in a.profiles_root.glob('D*/profiles_before_clean') for p in d.glob('*.csv') if PATTERN.match(p.name))
    for path in paths:
        if max(tested.values()) >= a.max_samples:
            break
        try:
            arr = np.loadtxt(path, delimiter=',', ndmin=2)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] != a.depths:
            continue
        clean = arr[np.all(np.isfinite(arr) & (arr != 0), axis=1)]
        if clean.size == 0:
            continue
        match = PATTERN.match(path.name)
        metric = match.group('metric') if match else 'unknown'
        scale = float(np.mean(np.abs(clean)))
        for row in clean:
            for n in (1, 2):
                if tested[(metric, n)] >= a.max_samples:
                    continue
                # Interior points only: endpoints cannot be linearly bracketed.
                missing = rng.sample(range(1, a.depths - 1), n)
                pred = interpolate_at(row, missing)
                mae = float(np.mean(np.abs(pred - row[missing])))
                errors[(metric, n)].append(mae)
                tested[(metric, n)] += 1
        del arr, clean
    interp = pd.DataFrame([{'metric': metric, 'artificial_bad_points': n, 'samples': tested[(metric, n)],
                            'mae_mean': np.mean(errors[(metric, n)]) if errors[(metric, n)] else np.nan,
                            'mae_median': np.median(errors[(metric, n)]) if errors[(metric, n)] else np.nan,
                            'mae_p95': np.percentile(errors[(metric, n)], 95) if errors[(metric, n)] else np.nan}
                           for metric in sorted({key[0] for key in tested}) for n in (1, 2)])
    metric_means = {}
    for path in paths:
        match = PATTERN.match(path.name)
        if not match: continue
        metric_means.setdefault(match.group('metric'), []).append(path)
    # Normalize by the mean absolute value of the corresponding metric.
    scales = {}
    for metric, metric_paths in metric_means.items():
        vals = []
        for path in metric_paths[:50]:
            try:
                x = np.loadtxt(path, delimiter=',', ndmin=2)
                vals.append(float(np.nanmean(np.abs(x[np.isfinite(x)]))))
            except Exception:
                pass
        scales[metric] = np.mean(vals) if vals else np.nan
    interp['nmae_mean'] = interp.apply(lambda r: r.mae_mean / scales.get(r.metric, np.nan), axis=1)
    interp.to_csv(a.qc_root / 'interpolation_validation_summary.csv', index=False)
    print('Retention:', a.qc_root / 'subject_column_retention_curve.csv')
    print('Population summary:', a.qc_root / 'population_column_retention_summary.csv')
    print('Interpolation:', a.qc_root / 'interpolation_validation_summary.csv')
    print('Interpolation samples:', tested)
    print(retention.head())
    print(interp)


if __name__ == '__main__':
    main()
