#!/usr/bin/env python3
"""Audit ico3/4/5 nesting in full fsaverage and write fixed node tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


LEVEL_COUNTS = {3: 642, 4: 2562, 5: 10242}
HEMISPHERES = ("lh", "rh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freesurfer-subjects-dir", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--reference-subject", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--angular-tolerance-degrees", type=float, default=0.01)
    return parser.parse_args()


def normalized(vertices: np.ndarray) -> np.ndarray:
    centered = np.asarray(vertices, dtype=np.float64) - np.median(vertices, axis=0)
    return centered / np.linalg.norm(centered, axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, object] = {
        "status": "complete",
        "mapping_method": (
            "one-to-one nearest coordinate on normalized FreeSurfer sphere; "
            "accepted only when bijective and below angular tolerance"
        ),
        "angular_tolerance_degrees": args.angular_tolerance_degrees,
        "full_template": str((args.freesurfer_subjects_dir / "fsaverage").resolve()),
        "levels": {},
    }
    level_rows: dict[int, list[dict[str, object]]] = {level: [] for level in LEVEL_COUNTS}
    all_pass = True

    for hemisphere in HEMISPHERES:
        full_sphere, _ = fsio.read_geometry(
            args.freesurfer_subjects_dir / "fsaverage" / "surf" / f"{hemisphere}.sphere"
        )
        if len(full_sphere) != 163842:
            raise RuntimeError(f"{hemisphere}: expected 163842 full vertices, got {len(full_sphere)}")
        full_normalized = normalized(full_sphere)
        tree = cKDTree(full_normalized)
        profile_path = (
            args.profile_root
            / args.reference_subject
            / "profiles"
            / f"{args.reference_subject}_{hemisphere}_md_profiles.npz"
        )
        with np.load(profile_path, allow_pickle=False) as profile:
            vertex_ids = profile["fsaverage_vertex_id"].astype(np.int64)
            labels = profile["template_label_id"].astype(np.int64)
            names = profile["template_label_name"].astype(str)
            analysis_mask = profile["in_analysis_mask"].astype(bool)
        if not np.array_equal(vertex_ids, np.arange(163842)):
            raise RuntimeError(f"{hemisphere}: reference profile vertex IDs are not full ordered fsaverage")

        for level, expected_count in LEVEL_COUNTS.items():
            low_path = args.freesurfer_subjects_dir / f"fsaverage{level}" / "surf" / f"{hemisphere}.sphere"
            low_sphere, _ = fsio.read_geometry(low_path)
            if len(low_sphere) != expected_count:
                raise RuntimeError(
                    f"{hemisphere} ico{level}: expected {expected_count}, got {len(low_sphere)}"
                )
            chord, mapped = tree.query(normalized(low_sphere), k=1)
            angular = np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))
            unique = int(np.unique(mapped).size)
            passed = unique == expected_count and float(angular.max()) <= args.angular_tolerance_degrees
            all_pass &= passed
            level_key = f"ico{level}"
            audit["levels"].setdefault(level_key, {})
            audit["levels"][level_key][hemisphere] = {
                "low_sphere_path": str(low_path.resolve()),
                "n_low_vertices": expected_count,
                "n_unique_full_vertices": unique,
                "maximum_angular_error_degrees": float(angular.max()),
                "mean_angular_error_degrees": float(angular.mean()),
                "mapping_is_identity": bool(np.array_equal(mapped, np.arange(expected_count))),
                "mapped_index_min": int(mapped.min()),
                "mapped_index_max": int(mapped.max()),
                "passed": passed,
            }
            for node_id, full_id, error in zip(
                np.arange(expected_count), mapped.astype(np.int64), angular
            ):
                level_rows[level].append(
                    {
                        "hemisphere": hemisphere,
                        "ico_level": level,
                        "ico_node_id": int(node_id),
                        "fsaverage_vertex_id": int(full_id),
                        "dk_label_id": int(labels[full_id]),
                        "dk_label_name": str(names[full_id]),
                        "in_analysis_mask": bool(analysis_mask[full_id]),
                        "angular_mapping_error_degrees": float(error),
                    }
                )

    audit["passed"] = all_pass
    for level, rows in level_rows.items():
        table = pd.DataFrame(rows)
        table.to_csv(args.output_dir / f"ico{level}_vertex_order.csv", index=False)
        audit["levels"][f"ico{level}"]["bilateral"] = {
            "n_before_mask": int(len(table)),
            "n_after_mask": int(table["in_analysis_mask"].sum()),
            "n_after_mask_by_hemisphere": {
                hemisphere: int(
                    table.loc[table["hemisphere"].eq(hemisphere), "in_analysis_mask"].sum()
                )
                for hemisphere in HEMISPHERES
            },
        }
    (args.output_dir / "template_nesting_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if not all_pass:
        raise RuntimeError("Template nesting audit failed; model tasks must not start")
    print(json.dumps({"passed": all_pass, "levels": audit["levels"]}, indent=2))


if __name__ == "__main__":
    main()
