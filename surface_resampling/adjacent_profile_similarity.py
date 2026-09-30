#!/usr/bin/env python3
"""Summarize profile similarity across unique template mesh edges."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    parser.add_argument("--surfaces", type=Path, nargs="+")
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for profile_path, surface_path in zip(args.profiles, args.surfaces or []):
        data = np.load(profile_path)
        profiles = np.asarray(data["profiles"], dtype=float)
        _, faces = nib.freesurfer.io.read_geometry(str(surface_path))
        edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], axis=0), axis=1), axis=0)
        a = profiles[edges[:, 0]]
        b = profiles[edges[:, 1]]
        valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        a, b = a[valid], b[valid]
        denom = np.linalg.norm(a - a.mean(1, keepdims=True), axis=1) * np.linalg.norm(b - b.mean(1, keepdims=True), axis=1)
        correlations = np.full(len(a), np.nan)
        nonconstant = denom > 0
        correlations[nonconstant] = np.sum((a[nonconstant] - a[nonconstant].mean(1, keepdims=True)) * (b[nonconstant] - b[nonconstant].mean(1, keepdims=True)), axis=1) / denom[nonconstant]
        errors = a - b
        rows.append({"subject_id": args.subject_id, "template": args.template, "hemisphere": profile_path.stem.split("_")[1], "n_edges": int(len(a)), "correlation_median": float(np.nanmedian(correlations)), "correlation_q1": float(np.nanpercentile(correlations, 25)), "correlation_q3": float(np.nanpercentile(correlations, 75)), "mae_median": float(np.median(np.abs(errors))), "rmse_median": float(np.median(np.sqrt(np.mean(errors ** 2, axis=1))))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
