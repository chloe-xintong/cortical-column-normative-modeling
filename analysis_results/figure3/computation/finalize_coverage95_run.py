#!/usr/bin/env python3
"""Validate the completed coverage-gated Figure 3 run and freeze its status.

Description
-----------
Validate the completed coverage-gated Figure 3 run and freeze its status.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input
- ``--submission-json`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure3/computation/finalize_coverage95_run.py --help
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--submission-json', type=Path, required=True)
    parser.add_argument('--expected-subjects', type=int, default=313)
    parser.add_argument('--minimum-coverage', type=float, default=.95)
    args = parser.parse_args()
    root = args.output_dir
    completion = json.loads((root / 'qc/complete.json').read_text())
    if completion.get('status') != 'complete':
        raise RuntimeError('Main Figure 3 analysis is not complete')
    subjects = pd.read_csv(root / 'tables/figure3_subjects.csv', dtype={'subject_id': str})
    if len(subjects) != args.expected_subjects or subjects.subject_id.duplicated().any():
        raise RuntimeError('Final subject table does not match the expected unique cohort')
    screen = pd.read_csv(root / 'qc/candidate_profile_coverage_screen.csv',
                         dtype={'subject_id': str})
    retained = screen.loc[screen.passes_minimum_coverage.astype(bool)]
    if len(retained) != args.expected_subjects:
        raise RuntimeError('Coverage screen retained count differs from the final cohort')
    if retained.coverage_fraction.min() < args.minimum_coverage:
        raise RuntimeError('A retained subject is below the coverage threshold')
    if set(retained.subject_id) != set(subjects.subject_id):
        raise RuntimeError('Coverage screen and final subject table differ')
    extreme = pd.read_csv(root / 'qc/figure3A_extreme_region_coverage_qc.csv')
    if len(extreme) != 3 or not extreme.n_profile_subjects.eq(args.expected_subjects).all():
        raise RuntimeError('Full-range displayed profiles do not share the final cohort')
    stems = (
        'Figure3A_01_whole_cortex_profile',
        'Figure3A_02_quantile_region_profiles',
        'Figure3A_03_quantile_selection_key',
        'Figure3A_04_extreme_region_profiles',
        'Figure3A_05_extreme_selection_key',
        'Figure3B_01_PC1_PC2_loadings',
        'Figure3B_02_explained_variance',
        'Figure3B_03_PC1_PC2_group_score_maps',
        'Figure3C_01_PC1_PC2_thickness_correlations',
        'Figure3C_02_thickness_partial_r2',
        'Figure3D_01_quantile_adjusted_profiles',
        'Figure3D_02_extreme_adjusted_profiles',
        'Figure3D_03_PC1_raw_adjusted_maps',
        'Figure3D_04_PC2_raw_adjusted_maps',
        'Figure3D_05_fixed_component_retention',
    )
    missing = [f'{stem}.{ext}' for stem in stems for ext in ('png', 'pdf', 'svg')
               if not (root / 'figures' / f'{stem}.{ext}').is_file()]
    if missing:
        raise RuntimeError(f'Missing final Figure 3 exports: {missing}')

    submission = json.loads(args.submission_json.read_text())
    submission.update(
        formal_status='complete', completed_date=str(date.today()),
        final_subjects=len(subjects),
        finalization_job=int(os.environ['SLURM_JOB_ID']) if 'SLURM_JOB_ID' in os.environ else None,
        minimum_retained_coverage=float(retained.coverage_fraction.min()),
        notebook_status='executed successfully after coverage-gated recomputation',
    )
    args.submission_json.write_text(json.dumps(submission, indent=2) + '\n')
    print(json.dumps({
        'status': 'complete', 'subjects': len(subjects),
        'minimum_retained_coverage': retained.coverage_fraction.min(),
        'figures_verified': len(stems),
    }, indent=2))


if __name__ == '__main__':
    main()
