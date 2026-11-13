#!/usr/bin/env python3
"""Audit spatial coverage of the anchors stored in a Figure 4 pair table.

Description
-----------
This is deliberately read-only with respect to pair construction: it evaluates
the already used anchors against the fsaverage sphere and DK annotation.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--subjects-dir`` command-line input
- ``--pair-table`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure4/scripts/audit_anchor_coverage.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_label
from scipy.spatial import cKDTree


HEMISPHERES = ("lh", "rh")
EXCLUDED_LABELS = {"unknown", "corpuscallosum", "medial_wall", "medialwall"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", default="fsaverage")
    return parser.parse_args()


def annotation_and_mask(template_root: Path, hemisphere: str):
    annot, _, names = fsio.read_annot(str(template_root / "label" / f"{hemisphere}.aparc.annot"))
    cortex = read_label(str(template_root / "label" / f"{hemisphere}.cortex.label"), read_scalars=False).astype(int)
    mask = np.zeros(annot.size, dtype=bool)
    mask[cortex] = True
    labels = np.asarray([
        (name.decode() if isinstance(name, bytes) else str(name)).lower().replace("-", "_").replace(" ", "_")
        for name in names
    ])
    for label_id, name in enumerate(labels):
        if name in EXCLUDED_LABELS:
            mask &= annot != label_id
    return annot, labels, mask


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    tri_area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    area = np.zeros(len(vertices), dtype=float)
    for corner in range(3):
        np.add.at(area, faces[:, corner], tri_area / 3.0)
    return area


def main() -> None:
    args = parse_args()
    template_root = args.subjects_dir / args.template
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(args.pair_table, columns=["hemisphere", "anchor_vertex"])
    summary_rows, region_rows = [], []
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), constrained_layout=True)

    for ax, hemi in zip(axes, HEMISPHERES):
        sphere, _ = fsio.read_geometry(str(template_root / "surf" / f"{hemi}.sphere"))
        white, faces = fsio.read_geometry(str(template_root / "surf" / f"{hemi}.white"))
        annotation, names, mask = annotation_and_mask(template_root, hemi)
        anchors = np.sort(pairs.loc[pairs.hemisphere.eq(hemi), "anchor_vertex"].unique())
        if anchors.size == 0:
            raise ValueError(f"No {hemi} anchors in {args.pair_table}")
        direction = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
        tree = cKDTree(direction[anchors])
        chord, _ = tree.query(direction[mask], k=1)
        angle_deg = np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))
        valid_vertices = np.flatnonzero(mask)
        nearest_angle = np.full(len(sphere), np.nan)
        nearest_angle[valid_vertices] = angle_deg
        areas = vertex_areas(white, faces)

        summary_rows.append({
            "hemisphere": hemi, "n_analysis_vertices": int(mask.sum()), "n_anchors": int(anchors.size),
            "anchor_fraction": float(anchors.size / mask.sum()),
            "nearest_anchor_angle_deg_median": float(np.nanmedian(nearest_angle)),
            "nearest_anchor_angle_deg_q95": float(np.nanquantile(nearest_angle, .95)),
            "nearest_anchor_angle_deg_max": float(np.nanmax(nearest_angle)),
        })
        for label_id in np.unique(annotation[mask]):
            in_region = mask & (annotation == label_id)
            name = names[label_id]
            region_rows.append({
                "hemisphere": hemi, "dk_label": int(label_id), "dk_region": name,
                "vertex_count": int(in_region.sum()), "surface_area_mm2": float(areas[in_region].sum()),
                "area_fraction": float(areas[in_region].sum() / areas[mask].sum()),
                "anchor_count": int(np.isin(anchors, np.flatnonzero(in_region)).sum()),
            })

        # Equirectangular sphere projection gives a compact, reproducible anchor map.
        lon = np.arctan2(direction[:, 1], direction[:, 0])
        lat = np.arcsin(np.clip(direction[:, 2], -1, 1))
        ax.scatter(lon[mask], lat[mask], c="#D9D9D9", s=.12, linewidths=0, rasterized=True)
        ax.scatter(lon[anchors], lat[anchors], c="#DB3124", s=5, linewidths=0, rasterized=True)
        ax.set(title=f"{hemi}: fsaverage sphere anchors", xlabel="Sphere longitude (rad)", ylabel="Sphere latitude (rad)")
        ax.spines[["top", "right"]].set_visible(False)

    regions = pd.DataFrame(region_rows)
    regions["anchor_fraction"] = regions.anchor_count / regions.groupby("hemisphere").anchor_count.transform("sum")
    regions["anchor_minus_area_fraction"] = regions.anchor_fraction - regions.area_fraction
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "figure4_anchor_coverage_summary.csv", index=False)
    regions.sort_values(["hemisphere", "dk_region"]).to_csv(args.output_dir / "figure4_anchor_coverage_by_dk_region.csv", index=False)
    fig.savefig(args.output_dir / "figure4_anchor_coverage_sphere.png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved anchor coverage QC for {len(summary_rows)} hemispheres to {args.output_dir}")


if __name__ == "__main__":
    main()
