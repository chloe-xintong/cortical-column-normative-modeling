#!/usr/bin/env python3
"""Apply the shared subject-level cleaned-profile coverage gate to Figure 4.

Description
-----------
Apply the shared subject-level cleaned-profile coverage gate to Figure 4.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--source-manifest`` command-line input
- ``--profiles-dir`` command-line input
- ``--output-manifest`` command-line input
- ``--audit-csv`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/build_coverage95_manifest.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--profiles-dir', type=Path, required=True)
    parser.add_argument('--output-manifest', type=Path, required=True)
    parser.add_argument('--audit-csv', type=Path, required=True)
    parser.add_argument('--minimum-coverage', type=float, default=.95)
    args = parser.parse_args()
    if not 0 <= args.minimum_coverage <= 1:
        parser.error('--minimum-coverage must be in [0, 1]')

    source = pd.read_csv(args.source_manifest, dtype={'subject_id': str})
    if source.empty or source.subject_id.duplicated().any():
        raise ValueError('Source manifest must contain unique subject IDs')
    rows = []
    for subject in source.subject_id:
        n_mask = n_valid = 0
        for hemi in ('lh', 'rh'):
            path = args.profiles_dir / subject / 'profiles' / f'{subject}_{hemi}_md_profiles.npz'
            with np.load(path, allow_pickle=False) as profile:
                mask = profile['in_analysis_mask'].astype(bool)
                valid = profile['column_valid'].astype(bool)
            if mask.shape != valid.shape:
                raise ValueError(f'{subject} {hemi}: coverage arrays differ in shape')
            n_mask += int(mask.sum())
            n_valid += int((mask & valid).sum())
        coverage = n_valid / n_mask
        rows.append(dict(subject_id=subject, n_analysis_vertices=n_mask,
                         n_valid_columns=n_valid, coverage_fraction=coverage,
                         minimum_coverage=args.minimum_coverage,
                         passes_minimum_coverage=coverage >= args.minimum_coverage))

    audit = pd.DataFrame(rows)
    retained = source.loc[source.subject_id.isin(
        audit.loc[audit.passes_minimum_coverage, 'subject_id'])].copy()
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
    retained.to_csv(args.output_manifest, index=False)
    audit.to_csv(args.audit_csv, index=False)
    report = {
        'source_manifest': str(args.source_manifest),
        'source_n': len(source),
        'minimum_coverage': args.minimum_coverage,
        'coverage_definition': 'bilateral sum(in_analysis_mask & column_valid) / sum(in_analysis_mask)',
        'retained_n': len(retained),
        'excluded_n': int((~audit.passes_minimum_coverage).sum()),
        'excluded_subjects': audit.loc[~audit.passes_minimum_coverage, 'subject_id'].tolist(),
        'output_manifest': str(args.output_manifest),
    }
    args.audit_csv.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
