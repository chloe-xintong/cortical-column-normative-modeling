#!/usr/bin/env python3
"""Compute cleaned-profile CI/CV summaries outside Jupyter."""
import argparse
import csv
import re
from pathlib import Path
import numpy as np

PATTERN = re.compile(r'^(?P<subject>D\d+)_(?P<hemi>lh|rh)_(?P<region>.+)_(?P<metric>md|fa|ad|rd)\.csv$')
DEPTHS = 21
FIELDS = ['subject','hemi','region','metric','status','n_trajectories','n_depths','mean_value','mean_sd','mean_ci_width','mean_relative_ci_width_pct','mean_cv','regional_cv_pct']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-root', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/results/column_profiles'))
    p.add_argument('--output-dir', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/results/profile_stability'))
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    output = a.output_dir/'regional_ci_cv_summary.csv'
    paths = sorted(a.input_root.glob('D*/profiles/*.csv'))
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader()
        valid = 0
        for no, path in enumerate(paths, 1):
            m = PATTERN.match(path.name)
            if not m: continue
            row = {**m.groupdict(), 'status':'unreadable', 'n_trajectories':0, 'n_depths':0,
                   'mean_value':np.nan, 'mean_sd':np.nan, 'mean_ci_width':np.nan, 'mean_relative_ci_width_pct':np.nan, 'mean_cv':np.nan, 'regional_cv_pct':np.nan}
            try:
                x = np.loadtxt(path, delimiter=',', ndmin=2)
                if x.shape[1] != DEPTHS: row['status'] = 'unexpected_depth_count'; row['n_trajectories'],row['n_depths']=x.shape
                else:
                    x = np.where(np.isfinite(x), x, np.nan); n=np.sum(~np.isnan(x),axis=0)
                    mean=np.nanmean(x,axis=0); sd=np.nanstd(x,axis=0,ddof=1)
                    ci=2*1.96*sd/np.sqrt(np.maximum(n,1)); cv=sd/np.maximum(np.abs(mean),np.finfo(float).eps)
                    relative_ci_pct = 100*ci/np.maximum(np.abs(mean),np.finfo(float).eps)
                    row.update(status='valid',n_trajectories=x.shape[0],n_depths=DEPTHS,mean_value=float(np.nanmean(mean)),mean_sd=float(np.nanmean(sd)),mean_ci_width=float(np.nanmean(ci)),mean_relative_ci_width_pct=float(np.nanmean(relative_ci_pct)),mean_cv=float(np.nanmean(cv)),regional_cv_pct=float(100*np.nanmedian(cv))); valid+=1
            except Exception:
                pass
            writer.writerow(row)
            if no % 500 == 0: print(f'Processed {no}/{len(paths)} files; valid={valid}', flush=True)
    print(f'Wrote {output}; valid profiles={valid}')

if __name__ == '__main__': main()
