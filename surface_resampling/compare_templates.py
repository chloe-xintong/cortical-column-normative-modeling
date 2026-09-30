#!/usr/bin/env python3
"""Compare direct fsaverage6 profiles with fsaverage downsampled to fsaverage6."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    values = np.full(a.shape[0], np.nan)
    valid = denom > 0
    values[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return float(np.nanmedian(values)), float(np.nanmean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hemispheres", nargs="+", default=["lh", "rh"])
    parser.add_argument("--depths", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    depths = [float(x) for x in args.depths.split(",")]
    for hemi in args.hemispheres:
        direct_dir = args.root / "fsaverage6" / hemi
        source_dir = args.root / "fsaverage" / hemi
        down_dir = args.root / "fsaverage_to_fsaverage6" / hemi
        down_dir.mkdir(parents=True, exist_ok=True)
        direct = []
        down = []
        for depth in depths:
            stem = f"{args.subject_id}_{hemi}_md_depth-{depth:.2f}.mgh"
            source = source_dir / stem
            target = down_dir / stem
            if not target.exists():
                subprocess.run(["mri_surf2surf", "--srcsubject", "fsaverage", "--trgsubject", "fsaverage6", "--hemi", hemi, "--sval", str(source), "--tval", str(target)], check=True)
            direct.append(np.asarray(nib.load(str(direct_dir / stem)).dataobj).reshape(-1))
            down.append(np.asarray(nib.load(str(target)).dataobj).reshape(-1))
        direct_array = np.column_stack(direct)
        down_array = np.column_stack(down)
        valid = np.isfinite(direct_array).all(axis=1) & np.isfinite(down_array).all(axis=1)
        d = direct_array[valid] - down_array[valid]
        median_corr, mean_corr = corr(direct_array[valid], down_array[valid])
        rows.append({"subject_id": args.subject_id, "hemisphere": hemi, "n_vertices": int(valid.sum()), "profile_correlation_median": median_corr, "profile_correlation_mean": mean_corr, "profile_mae": float(np.mean(np.abs(d))), "profile_rmse": float(np.sqrt(np.mean(d ** 2))), "mean_difference": float(np.mean(d)), "median_absolute_difference": float(np.median(np.abs(d)))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
