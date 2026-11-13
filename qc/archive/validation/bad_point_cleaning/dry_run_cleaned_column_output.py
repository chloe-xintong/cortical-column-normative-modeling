#!/usr/bin/env python3
"""Run cleaning on a copied subject and write the formal column output.

Description
-----------
Run cleaning on a copied subject and write the formal column output.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--subject`` command-line input
- ``--analysis-root`` command-line input
- ``--output-root`` command-line input
- ``--staging-root`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- In-memory values returned to the calling code.

Usage
-----
python qc/archive/validation/bad_point_cleaning/dry_run_cleaned_column_output.py --help
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subject', default='D0398')
    p.add_argument('--analysis-root', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis'))
    p.add_argument('--output-root', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/results/column_profiles'))
    p.add_argument('--staging-root', type=Path, default=None)
    args = p.parse_args()
    source = args.analysis_root / args.subject
    source_profiles = source / 'profiles_before_clean'
    source_regions = source / 'regions'
    output = args.output_root / args.subject
    if output.exists():
        profiles = output / 'profiles'
        regions = output / 'regions'
        complete = (profiles.is_dir() and regions.is_dir()
                    and len(list(profiles.glob('*.csv'))) >= 272
                    and len(list(regions.glob('*.csv'))) >= 68
                    and (profiles / 'bad_trajectory_mask.json').is_file())
        if complete:
            print(f'SKIP complete output: {output}')
            return
        raise SystemExit(f'Output exists but is incomplete; refusing overwrite: {output}')
    staging_root = args.staging_root or (args.output_root / '.tmp')
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / f'{args.subject}_{os.getpid()}'
    output_profiles = output / 'profiles'
    output_regions = output / 'regions'
    if not source_profiles.is_dir() or not source_regions.is_dir():
        raise SystemExit(f'Missing source directories for {args.subject}')
    staging.mkdir(parents=True)
    staging_profiles = staging / 'profiles'
    staging_regions = staging / 'regions'
    shutil.copytree(source_profiles, staging_profiles)
    shutil.copytree(source_regions, staging_regions)
    cmd = [
        'python', 'profile_sampling/clean_bad_trajectories.py',
        '--subject-id', args.subject,
        '--region-dir', str(staging_regions),
        '--profile-dir', str(staging_profiles),
        '--metrics', 'md', 'fa', 'ad', 'rd',
    ]
    try:
        subprocess.run(cmd, check=True)
        if len(list(staging_profiles.glob('*.csv'))) < 272 or not (staging_profiles / 'bad_trajectory_mask.json').is_file():
            raise RuntimeError(f'Incomplete staged output for {args.subject}')
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f'Formal column output: {output_profiles}')
    print(f'Cleaned coordinates: {output_regions}')


if __name__ == '__main__':
    main()
