#!/usr/bin/env python3
"""Fixed-depth spatial grouping at target effective group sizes."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from ..common.config import FSAVERAGE6_MAPPING_ROOT, SUBJECTS_DIR
from ..common.io import mapping_files, mapping_frame
from .analyze_spatial_averaging import (
    edges_to_graph,
    group_connectedness,
    metric,
    save_overlay,
    spatial_groups,
)


def valid_column_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["column_valid"].astype(str).str.lower().isin(
        ("true", "1", "1.0")
    )


def make_folds(n: int, n_splits: int, seed: int) -> np.ndarray:
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    folds = np.empty(n, dtype=int)
    for position, index in enumerate(indices):
        folds[index] = position % n_splits
    return folds


def analyze_subject(subject: str, args: argparse.Namespace) -> None:
    output = args.output_dir / subject
    for name in ("tables", "overlays", "figures", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)

    metric_rows = []
    group_rows = []
    total_steps = len(args.hemispheres) * len(args.regions) * len(args.target_sizes)
    completed_steps = 0
    for hemisphere in args.hemispheres:
        surface_path = SUBJECTS_DIR / subject / "surf" / f"{hemisphere}.white"
        vertices, faces = nib.freesurfer.io.read_geometry(str(surface_path))
        files = mapping_files(FSAVERAGE6_MAPPING_ROOT, subject, hemisphere)

        for region in args.regions:
            matches = [
                path
                for path in files
                if path.name.endswith(
                    f"_{region}_native_to_fsaverage.csv.gz"
                )
            ]
            if len(matches) != 1:
                print(
                    f"{subject} {hemisphere} {region}: mapping files={len(matches)}",
                    flush=True,
                )
                continue

            frame = mapping_frame(matches[0])
            frame = frame[valid_column_mask(frame)].copy()
            frame["md50"] = pd.to_numeric(
                frame["MD_depth_10"], errors="coerce"
            )
            frame = frame.dropna(subset=["md50"])
            frame = frame.drop_duplicates("native_vertex_id")
            if len(frame) < args.folds:
                continue

            native_ids = frame["native_vertex_id"].to_numpy(int)
            values = frame["md50"].to_numpy(float)
            folds = make_folds(len(frame), args.folds, args.seed)
            for target_size in args.target_sizes:
                completed_steps += 1
                k = max(1, int(np.ceil(len(frame) / target_size)))
                if k == 1:
                    groups = {int(vertex): 0 for vertex in native_ids}
                    diagnostics = {
                        "status": "valid",
                        "components": 1,
                        "group_sizes": [len(native_ids)],
                    }
                else:
                    groups, diagnostics = spatial_groups(
                        vertices,
                        faces,
                        native_ids,
                        k,
                    )
                if groups is None:
                    metric_rows.append(
                        {
                            "subject": subject,
                            "hemisphere": hemisphere,
                            "region": region,
                            "target_group_size": target_size,
                            "k": k,
                            "status": diagnostics["status"],
                        }
                    )
                    continue

                labels = np.array([groups[int(vertex)] for vertex in native_ids])
                group_sizes = np.bincount(labels, minlength=k)
                for group, components, size in group_connectedness(
                    vertices, faces, native_ids, groups
                ):
                    group_rows.append(
                        {
                            "subject": subject,
                            "hemisphere": hemisphere,
                            "region": region,
                            "target_group_size": target_size,
                            "k": k,
                            "group": group,
                            "n_columns": size,
                            "n_components": components,
                        }
                    )

                prediction = np.full(len(frame), np.nan)
                valid = True
                for fold in range(args.folds):
                    training = folds != fold
                    testing = ~training
                    for group in range(k):
                        members = training & (labels == group)
                        if members.sum() < args.min_train_per_group:
                            valid = False
                            break
                    if not valid:
                        break
                    for group in range(k):
                        members = training & (labels == group)
                        prediction[testing & (labels == group)] = np.mean(
                            values[members]
                        )

                if not valid or not np.isfinite(prediction).all():
                    status = "invalid_minimum_training_columns"
                    mae = np.nan
                    rmse = np.nan
                else:
                    status = "valid"
                    mae, rmse = metric(values, prediction)

                metric_rows.append(
                    {
                        "subject": subject,
                        "hemisphere": hemisphere,
                        "region": region,
                        "target_group_size": target_size,
                        "k": k,
                        "n_columns": len(frame),
                        "mean_group_size": float(np.mean(group_sizes)),
                        "min_group_size": int(np.min(group_sizes)),
                        "max_group_size": int(np.max(group_sizes)),
                        "mae": mae,
                        "rmse": rmse,
                        "status": status,
                    }
                )
                if status == "valid":
                    save_overlay(
                        output,
                        subject,
                        hemisphere,
                        region,
                        f"spatial_target_{target_size}_oof_prediction",
                        len(vertices),
                        native_ids,
                        prediction,
                    )
                print(
                    f"[{completed_steps}/{total_steps}] {subject} "
                    f"{hemisphere} {region} target={target_size} "
                    f"k={k} {status}",
                    flush=True,
                )

    pd.DataFrame(metric_rows).to_csv(
        output / "tables/target_group_size_metrics.csv", index=False
    )
    pd.DataFrame(group_rows).to_csv(
        output / "tables/target_group_size_groups.csv", index=False
    )
    print(f"Wrote {subject}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--regions", nargs="+", default=[
        "precentral",
        "superiorfrontal",
        "supramarginal",
        "superiortemporal",
    ])
    parser.add_argument("--hemispheres", nargs="+", default=["lh", "rh"])
    parser.add_argument("--target-sizes", nargs="+", type=int, default=[50, 100, 200, 400])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train-per-group", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/newStor/paros/paros_WORK/xintong/results/"
            "native_surface_analysis/spatial_averaging_target_sizes"
        ),
    )
    args = parser.parse_args()
    analyze_subject(args.subject, args)


if __name__ == "__main__":
    main()
