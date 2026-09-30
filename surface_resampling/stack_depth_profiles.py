#!/usr/bin/env python3
"""Stack depth-specific MGH files into a vertex-by-depth NPZ profile matrix."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--template", choices=["fsaverage", "fsaverage6"], required=True)
    parser.add_argument("--hemisphere", choices=["lh", "rh"], required=True)
    parser.add_argument("--depths", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cortex-mask", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    depths = np.asarray([float(value) for value in args.depths.split(",")], dtype=float)
    matrices = []
    for index, depth in enumerate(depths, start=1):
        path = args.input_dir / f"{args.subject_id}_{args.hemisphere}_md_depth-{depth:.2f}.mgh"
        values = np.asarray(nib.load(str(path)).dataobj).reshape(-1).astype(float)
        matrices.append(values)
        logging.info("[%d/%d] Loaded %s", index, len(depths), path.name)
    profiles = np.column_stack(matrices)
    if args.cortex_mask:
        cortex_mask = np.asarray(nib.load(str(args.cortex_mask)).dataobj).reshape(-1).astype(bool)
        if cortex_mask.size != profiles.shape[0]:
            raise ValueError("Cortex mask vertex count does not match profiles")
    else:
        cortex_mask = np.ones(profiles.shape[0], dtype=bool)
    valid_mask = np.isfinite(profiles).all(axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        profiles=profiles,
        depth_fractions=depths,
        vertex_ids=np.arange(profiles.shape[0], dtype=np.int64),
        cortex_mask=cortex_mask,
        valid_mask=valid_mask,
        template=args.template,
        hemisphere=args.hemisphere,
        subject_id=args.subject_id,
    )
    logging.info("Saved %s with shape %s", args.output, profiles.shape)


if __name__ == "__main__":
    main()
