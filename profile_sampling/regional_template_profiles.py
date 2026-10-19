#!/usr/bin/env python3
"""Aggregate cleaned fsaverage-column profiles by template DK labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--template", default="fsaverage")
    parser.add_argument("--metric-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hemispheres", nargs="+", default=["lh", "rh"])
    return parser.parse_args()


def process_hemisphere(
    args: argparse.Namespace, hemisphere: str
) -> list[dict[str, object]]:
    profile_path = args.profile_dir / (
        f"{args.subject_id}_{hemisphere}_{args.metric_name}_profiles.npz"
    )
    if not profile_path.is_file():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")
    with np.load(profile_path, allow_pickle=True) as data:
        target_ids = np.asarray(data["fsaverage_vertex_id"], dtype=int)
        profiles = np.asarray(data["profiles"], dtype=float)
        saved_label_ids = (
            data["template_label_id"]
            if "template_label_id" in data.files
            else None
        )
        if "template_label_name" in data.files:
            saved_label_names = data["template_label_name"]
        elif "fsaverage_label" in data.files:
            saved_label_names = data["fsaverage_label"]
        else:
            saved_label_names = None

    if saved_label_ids is not None and saved_label_names is not None:
        target_labels = np.asarray(saved_label_ids, dtype=int)
        vertex_label_names = np.asarray(saved_label_names).astype(str)
        label_name_by_id = {
            int(label_id): str(label_name)
            for label_id, label_name in zip(target_labels, vertex_label_names)
        }
    else:
        annotation_path = (
            args.subjects_dir / args.template / "label" / f"{hemisphere}.aparc.annot"
        )
        labels, _, annotation_names = fsio.read_annot(str(annotation_path))
        names = [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in annotation_names
        ]
        target_labels = labels[target_ids]
        label_name_by_id = {
            int(label_id): names[int(label_id)]
            for label_id in np.unique(target_labels)
        }
    excluded = {"unknown", "corpuscallosum", "medial_wall", "medialwall"}
    rows = []
    mean_dir = args.output_dir / "region_mean"
    mean_dir.mkdir(parents=True, exist_ok=True)
    for label_id in np.unique(target_labels):
        label_name = label_name_by_id[int(label_id)]
        if label_name.lower().replace("-", "_") in excluded:
            continue
        keep = target_labels == label_id
        values = profiles[keep]
        valid = np.isfinite(values).all(axis=1)
        values = values[valid]
        if len(values) == 0:
            continue
        mean = values.mean(axis=0)
        std = (
            values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(values.shape[1])
        )
        sem = std / np.sqrt(len(values))
        output_path = mean_dir / (
            f"{args.subject_id}_{hemisphere}_{label_name}_{args.metric_name}_mean.csv"
        )
        np.savetxt(output_path, mean.reshape(-1, 1), delimiter=",")
        rows.append(
            {
                "subject_id": args.subject_id,
                "hemisphere": hemisphere,
                "region": label_name,
                "metric": args.metric_name,
                "n_columns": int(len(values)),
                "depth_count": int(values.shape[1]),
                "mean_profile_path": str(output_path),
                "mean_depth_00": float(mean[0]),
                "mean_depth_50": float(mean[len(mean) // 2]),
                "mean_depth_100": float(mean[-1]),
                "sem_depth_00": float(sem[0]),
                "sem_depth_50": float(sem[len(sem) // 2]),
                "sem_depth_100": float(sem[-1]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = []
    for hemisphere in args.hemispheres:
        rows.extend(process_hemisphere(args, hemisphere))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qa_path = args.output_dir / f"{args.subject_id}_profiles_QA_{args.metric_name}.csv"
    if rows:
        with qa_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} template-DK regional profiles to {args.output_dir}")


if __name__ == "__main__":
    main()
