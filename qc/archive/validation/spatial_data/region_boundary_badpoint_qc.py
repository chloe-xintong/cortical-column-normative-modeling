#!/usr/bin/env python3
"""Compare bad-column rates at cortical region boundaries versus interiors.

Description
-----------
Compare bad-column rates at cortical region boundaries versus interiors.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--profiles-root`` command-line input
- ``--subjects-root`` command-line input
- ``--qc-root`` command-line input
- ``--boundary-layers`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python qc/archive/validation/spatial_data/region_boundary_badpoint_qc.py --help
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd

PROFILE_RE = re.compile(r'^(?P<subject>D\d+)_(?P<hemi>lh|rh)_(?P<region>.+)_(?P<metric>md|fa|ad|rd)\.csv$')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profiles-root', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis'))
    p.add_argument('--subjects-root', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer'))
    p.add_argument('--qc-root', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/results/column_quality_qc'))
    p.add_argument('--boundary-layers', type=int, default=1)
    return p.parse_args()


def boundary_vertices(labels, triangles, layers):
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0)
    different = labels[edges[:, 0]] != labels[edges[:, 1]]
    boundary = np.unique(edges[different].ravel())
    for _ in range(max(0, layers - 1)):
        touching = edges[np.isin(edges, boundary).any(axis=1)]
        boundary = np.unique(np.concatenate((boundary, touching.ravel())))
    return boundary


def main():
    a = parse_args()
    detail_path = a.qc_root / 'column_badpoint_detail.csv'
    file_path = a.qc_root / 'column_badpoint_file_summary.csv'
    if not detail_path.is_file() or not file_path.is_file():
        raise SystemExit('Missing completed badpoint detail/file summary CSVs.')
    a.qc_root.mkdir(parents=True, exist_ok=True)

    files = pd.read_csv(file_path, usecols=['subject', 'hemisphere', 'region', 'metric', 'status', 'total_columns'])
    files = files[files.status.eq('valid')]
    bad = pd.read_csv(detail_path, usecols=['subject', 'hemisphere', 'region', 'metric', 'column_id', 'bad_point_count'])
    bad = bad[bad.bad_point_count > 0]
    bad_by_profile = {tuple(key): set(group.column_id.astype(int))
                      for key, group in bad.groupby(['subject', 'hemisphere', 'region', 'metric'])}

    rows = []
    for subject in sorted(files.subject.unique()):
        subject_dir = a.subjects_root / subject
        for hemi in ('lh', 'rh'):
            annot = subject_dir / 'label' / f'{hemi}.aparc.annot'
            surf = subject_dir / 'surf' / f'{hemi}.white'
            if not annot.is_file() or not surf.is_file():
                continue
            labels, _, names = fsio.read_annot(str(annot))
            coords, faces = fsio.read_geometry(str(surf))
            boundary = set(boundary_vertices(labels, faces, a.boundary_layers).tolist())
            name_by_label = {i: (n.decode() if isinstance(n, bytes) else str(n)) for i, n in enumerate(names)}
            for _, f in files[(files.subject == subject) & (files.hemisphere == hemi)].iterrows():
                label_id = next((i for i, n in name_by_label.items() if n == f.region), None)
                if label_id is None:
                    continue
                vertices = np.flatnonzero(labels == label_id)
                boundary_ids = {i for i, v in enumerate(vertices) if int(v) in boundary}
                total = int(f.total_columns)
                bad_ids = bad_by_profile.get((subject, hemi, f.region, f.metric), set())
                bad_boundary = len(bad_ids.intersection(boundary_ids))
                bad_interior = len(bad_ids.difference(boundary_ids))
                interior_total = total - len(boundary_ids)
                rows.extend([
                    {'subject': subject, 'hemisphere': hemi, 'region': f.region, 'metric': f.metric, 'zone': 'boundary', 'total_columns': len(boundary_ids), 'bad_columns': bad_boundary},
                    {'subject': subject, 'hemisphere': hemi, 'region': f.region, 'metric': f.metric, 'zone': 'interior', 'total_columns': interior_total, 'bad_columns': bad_interior},
                ])
    result = pd.DataFrame(rows)
    result['bad_column_pct'] = 100 * result.bad_columns / result.total_columns.replace(0, np.nan)
    result.to_csv(a.qc_root / 'region_boundary_vs_interior_badpoint_summary.csv', index=False)
    cohort = result.groupby(['metric', 'zone'], as_index=False).agg(total_columns=('total_columns', 'sum'), bad_columns=('bad_columns', 'sum'))
    cohort['bad_column_pct'] = 100 * cohort.bad_columns / cohort.total_columns.replace(0, np.nan)
    cohort.to_csv(a.qc_root / 'cohort_boundary_vs_interior_summary.csv', index=False)
    print('Subjects processed:', result.subject.nunique() if not result.empty else 0)
    print('Rows written:', len(result))
    print('Output:', a.qc_root / 'region_boundary_vs_interior_badpoint_summary.csv')
    print(cohort.to_string(index=False))


if __name__ == '__main__':
    main()
