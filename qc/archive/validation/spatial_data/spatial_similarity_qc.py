#!/usr/bin/env python3
"""Compute within-region nearby-versus-distant column profile similarity.

Description
-----------
Compute within-region nearby-versus-distant column profile similarity.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--input-root`` command-line input
- ``--dwi-root`` command-line input
- ``--output-dir`` command-line input
- ``--k`` command-line input
- ``--seed`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- In-memory values returned to the calling code.

Usage
-----
python qc/archive/validation/spatial_data/spatial_similarity_qc.py --help
"""
import argparse
import csv
import subprocess
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

DEPTHS = 21

def corr(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a*a).sum(1) * (b*b).sum(1))
    out = np.full(len(a), np.nan)
    ok = den > 0
    out[ok] = (a[ok] * b[ok]).sum(1) / den[ok]
    return out

def load_mrtrix_transform(path):
    result = subprocess.run(['mrinfo', str(path), '-transform'], text=True,
                            capture_output=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        values = line.split()
        if len(values) == 4:
            rows.append([float(v) for v in values])
    if len(rows) != 4:
        raise ValueError(f'Could not parse MRtrix transform from {path}')
    return np.asarray(rows, dtype=float)

def process(path, region_path, affine, k, seed, subject, hemi, region):
    x = np.loadtxt(path, delimiter=',', ndmin=2)
    c = np.loadtxt(region_path, delimiter=',', ndmin=2)
    if x.shape[1] != DEPTHS or c.shape[0] not in (3, 4) or c.shape[1] != x.shape[0]*DEPTHS:
        return None
    valid = np.isfinite(x).all(1)
    voxel = c[:3, ::DEPTHS].T - 1.0
    coords = (affine @ np.c_[voxel, np.ones(len(voxel))].T).T[:, :3]
    valid &= np.isfinite(coords).all(1)
    idx = np.flatnonzero(valid)
    if len(idx) < 2*k + 1:
        return None
    x, coords = x[idx], coords[idx]
    n = len(x)
    tree = cKDTree(coords)
    distances, neighbours = tree.query(coords, k=k+1)
    rng = np.random.default_rng(seed)
    near_r, far_r, near_m, far_m, near_d, far_d, p20_d = [], [], [], [], [], [], []
    far_pool_size = max(k, int(np.ceil(.25*(n-1))))
    for i in range(n):
        dist = np.linalg.norm(coords - coords[i], axis=1)
        near = neighbours[i, 1:k+1]
        candidates = np.flatnonzero(np.arange(n) != i)
        far_pool = candidates[np.argpartition(dist[candidates], -far_pool_size)[-far_pool_size:]]
        far = rng.choice(far_pool, size=k, replace=False)
        xa, xn = x[i:i+1], x[near]
        xf, xd = x[far], x[i:i+1]
        near_r.append(np.nanmedian(corr(np.repeat(xa, k, 0), xn)))
        far_r.append(np.nanmedian(corr(np.repeat(xd, k, 0), xf)))
        near_m.append(np.nanmedian(np.sqrt(np.mean((np.repeat(xa,k,0)-xn)**2,1))))
        far_m.append(np.nanmedian(np.sqrt(np.mean((np.repeat(xd,k,0)-xf)**2,1))))
        near_d.append(float(np.mean(np.linalg.norm(coords[near]-coords[i],axis=1))))
        far_d.append(float(np.mean(np.linalg.norm(coords[far]-coords[i],axis=1))))
        p20_d.append(float(np.linalg.norm(coords[near[-1]]-coords[i])))
    near_r, far_r = np.asarray(near_r), np.asarray(far_r)
    near_m, far_m = np.asarray(near_m), np.asarray(far_m)
    return dict(subject=subject, hemisphere=hemi, region=region, n_columns=n,
                nearest_mean_distance_mm=np.nanmedian(near_d), distant_mean_distance_mm=np.nanmedian(far_d),
                twentieth_neighbor_distance_mm=np.nanmedian(p20_d),
                nearest_profile_corr=np.nanmedian(near_r), distant_profile_corr=np.nanmedian(far_r),
                delta_r=np.nanmedian(near_r-far_r),
                nearest_profile_rmse=np.nanmedian(near_m), distant_profile_rmse=np.nanmedian(far_m),
                delta_rmse=np.nanmedian(far_m-near_m),
                fraction_positive_delta_r=float(np.mean((near_r-far_r)>0)),
                fraction_positive_delta_rmse=float(np.mean((far_m-near_m)>0)),
                coordinate_space='physical_mm')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-root', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/results/column_profiles'))
    ap.add_argument('--dwi-root', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi'))
    ap.add_argument('--output-dir', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/results/spatial_similarity_qc'))
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows=[]; files=sorted(args.input_root.glob('D*/profiles/*_md.csv')); affine_cache={}
    for no, path in enumerate(files, 1):
        parts=path.stem.split('_'); subject=parts[0]; hemi=parts[1]; region='_'.join(parts[2:-1])
        rp=args.input_root/subject/'regions'/f'{hemi}_{region}.csv'
        dwi=args.dwi_root/subject/f'{subject}_04_preprocessed.mif'
        if not rp.is_file() or not dwi.is_file(): continue
        try:
            if subject not in affine_cache:
                affine_cache[subject] = load_mrtrix_transform(dwi)
            affine = affine_cache[subject]
        except Exception as exc:
            print(f'SKIP {subject}: {exc}', flush=True); continue
        row=process(path,rp,affine,args.k,args.seed,subject,hemi,region)
        if row: rows.append(row)
        if no % 10 == 0: print(f'Processed {no}/{len(files)} profile files; valid groups={len(rows)}', flush=True)
    out=args.output_dir/'spatial_similarity_group_summary.csv'
    fields=list(rows[0]) if rows else []
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f'Processed {len(files)} profile files; wrote {len(rows)} groups to {out}')

if __name__ == '__main__': main()
