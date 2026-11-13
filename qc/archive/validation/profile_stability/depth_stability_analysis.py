#!/usr/bin/env python3
"""Streaming MD depth and within-region profile stability summaries.

Description
-----------
Streaming MD depth and within-region profile stability summaries.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--input-root`` command-line input
- ``--output-dir`` command-line input
- ``--max-subjects`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- In-memory values returned to the calling code.

Usage
-----
python qc/archive/validation/profile_stability/depth_stability_analysis.py --help
"""
import argparse, csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

DEPTHS = 21

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-root', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/results/column_profiles'))
    p.add_argument('--output-dir', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/results/depth_stability'))
    p.add_argument('--max-subjects', type=int, default=None)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    depths, regions = [], []; files = sorted(a.input_root.glob('D*/profiles/*_md.csv'))
    if a.max_subjects is not None:
        subjects = sorted({f.parts[-3] for f in files})[:a.max_subjects]
        files = [f for f in files if f.parts[-3] in subjects]
    for no, f in enumerate(files, 1):
        bits = f.stem.split('_'); subject, hemi = bits[:2]; region = '_'.join(bits[2:-1])
        x = np.loadtxt(f, delimiter=',', ndmin=2); x = x[np.isfinite(x).all(axis=1)]
        if x.ndim != 2 or x.shape[1] != DEPTHS or len(x) < 2: continue
        n = len(x); mean = x.mean(0); sd = x.std(0, ddof=1); cv = 100*sd/np.maximum(abs(mean), np.finfo(float).eps)
        q25, med, q75 = np.quantile(x, [.25,.5,.75], axis=0)
        depths.extend(dict(subject=subject, hemisphere=hemi, region=region, depth=d, n_columns=n, mean_md=mean[d], sd_md=sd[d], cv_pct=cv[d], median_md=med[d], q25_md=q25[d], q75_md=q75[d]) for d in range(DEPTHS))
        r = np.array([np.corrcoef(v, mean)[0,1] for v in x]); rmse = np.sqrt(np.mean((x-mean)**2, axis=1))
        regions.append(dict(subject=subject, hemisphere=hemi, region=region, n_columns=n, median_profile_r=np.nanmedian(r), q25_profile_r=np.nanquantile(r,.25), q75_profile_r=np.nanquantile(r,.75), median_profile_rmse=np.median(rmse), q25_profile_rmse=np.quantile(rmse,.25), q75_profile_rmse=np.quantile(rmse,.75), cv_depth_spearman_rho=spearmanr(range(DEPTHS), cv).statistic, cv_depth_slope=np.polyfit(range(DEPTHS), cv, 1)[0]))
        if no % 100 == 0: print(f'Processed {no}/{len(files)} profile files', flush=True)
    for name, rows in [('depth_level_stability.csv', depths), ('region_level_stability.csv', regions)]:
        with (a.output_dir/name).open('w', newline='') as f: w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'Wrote {len(depths):,} depth rows and {len(regions):,} region rows')
if __name__ == '__main__': main()
