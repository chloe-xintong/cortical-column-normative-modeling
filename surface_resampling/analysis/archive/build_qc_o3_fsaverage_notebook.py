#!/usr/bin/env python3
"""Build the executable D0086 fsaverage O3 QC notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def markdown(source: str) -> dict:
    """Create a markdown notebook cell."""

    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> dict:
    """Create a code notebook cell."""

    return nbf.v4.new_code_cell(dedent(source).strip())


cells = [
    markdown(
        """
        # Option 3 fsaverage cortical-column QC

        This notebook audits the current `O3-Coordinate` workflow for subject
        `D0086` on the `fsaverage` target. It runs both hemispheres when the
        existing outputs are available.

        The notebook reads the saved O3 arrays and supporting FreeSurfer files.
        It does not modify or rerun the O3 generation pipeline, does not run
        fsaverage6, and does not compare O3 with O2-FS.
        """
    ),
    code(
        """
        from pathlib import Path
        import sys

        import numpy as np
        import pandas as pd
        from IPython.display import Image, display


        PROJECT_ROOT = Path(
            "/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
        )
        sys.path.insert(0, str(PROJECT_ROOT / "notebooks"))

        from qc_o3_fsaverage_helpers import run_qc


        CONFIG = {
            "subject_id": "D0086",
            "hemispheres": ["lh", "rh"],
            "target_template": "fsaverage",
            "o3_output_root": (
                "/mnt/newStor/paros/paros_WORK/xintong/results/"
                "o3_coordinate_profiles"
            ),
            "freesurfer_subjects_dir": "/home/apps/freesurfer/subjects",
            "native_subjects_root": (
                "/mnt/newStor/paros/paros_WORK/xintong/data/processed/"
                "t1_freesurfer"
            ),
            "fsaverage_surface_paths": {
                "lh": "/home/apps/freesurfer/subjects/fsaverage/surf/lh.white",
                "rh": "/home/apps/freesurfer/subjects/fsaverage/surf/rh.white",
            },
            "md_volume_path": (
                "/mnt/newStor/paros/paros_WORK/xintong/data/processed/"
                "dwi/D0086/D0086_md.nii.gz"
            ),
            "registration_path": (
                "/mnt/newStor/paros/paros_WORK/xintong/results/column_analysis/"
                "D0086/registration/D0086_DWI2T1.dat"
            ),
            "qc_output_root": "/mnt/newStor/paros/paros_WORK/xintong/results/o3_qc",
            "depths": np.linspace(0.0, 1.0, 21),
            "expected_fsaverage_vertices": 163842,
            "weight_tol": 1e-6,
            "coord_tol_mm": 1e-5,
            "min_valid_profile_fraction": 0.99,
            "min_in_bounds_fraction": 0.99,
            "round_trip_sample_size": 10000,
            "random_seed": 20261005,
        }

        CONFIG
        """
    ),
    markdown(
        """
        ## Provenance and coordinate conventions

        The O3 NPZ files contain the following relevant arrays:

        - `target_vertex_id`: fsaverage vertex IDs;
        - `source_face_id`, `source_vertex_ids`, `barycentric_weights`: native
          surface mapping provenance;
        - `white_anchor`, `pial_anchor`, `native_coordinates`: FreeSurfer
          surface RAS coordinates in subject T1 space;
        - `dwi_voxel_coordinates`: 0-based DWI voxel coordinates after the
          legacy one-based transform output is reduced by one;
        - `md_profiles`: target-vertex by 21-depth MD values.

        Native and template `aparc.annot` files and native FreeSurfer thickness
        files are available for both D0086 hemispheres and are used below.
        """
    ),
    code(
        """
        from datetime import datetime


        def file_record(path):
            path = Path(path)
            return {
                "path": str(path),
                "exists": path.is_file(),
                "modified": (
                    datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                    if path.is_file()
                    else None
                ),
            }


        provenance_paths = []
        for hemisphere in CONFIG["hemispheres"]:
            o3_path = (
                Path(CONFIG["o3_output_root"])
                / CONFIG["subject_id"]
                / "fsaverage"
                / "o3_coordinate"
                / f"{CONFIG['subject_id']}_{hemisphere}_profiles.npz"
            )
            provenance_paths.extend(
                [
                    o3_path,
                    o3_path.with_suffix(".qc.json"),
                    Path(CONFIG["fsaverage_surface_paths"][hemisphere]),
                    Path(CONFIG["native_subjects_root"])
                    / CONFIG["subject_id"]
                    / "surf"
                    / f"{hemisphere}.white",
                    Path(CONFIG["native_subjects_root"])
                    / CONFIG["subject_id"]
                    / "surf"
                    / f"{hemisphere}.pial",
                    Path(CONFIG["native_subjects_root"])
                    / CONFIG["subject_id"]
                    / "surf"
                    / f"{hemisphere}.thickness",
                    Path(CONFIG["native_subjects_root"])
                    / CONFIG["subject_id"]
                    / "label"
                    / f"{hemisphere}.aparc.annot",
                    Path(CONFIG["freesurfer_subjects_dir"])
                    / "fsaverage"
                    / "label"
                    / f"{hemisphere}.aparc.annot",
                ]
            )
        provenance_paths.extend(
            [Path(CONFIG["md_volume_path"]), Path(CONFIG["registration_path"])]
        )
        provenance = pd.DataFrame(file_record(path) for path in provenance_paths)
        display(provenance)
        if not provenance.exists.all():
            missing = provenance.loc[~provenance.exists, "path"].tolist()
            raise FileNotFoundError(f"Missing required QC inputs: {missing}")
        """
    ),
    markdown(
        """
        ## Execute QC

        This cell performs the complete D0086 left/right fsaverage QC and
        writes subject-specific tables, overlays, figures, and a compact JSON
        metrics file under `results/o3_qc/D0086/`.
        """
    ),
    code(
        """
        QC_RESULTS = run_qc(CONFIG)
        display(QC_RESULTS["summary"])
        """
    ),
    code(
        """
        output_root = Path(QC_RESULTS["output_root"])
        summary = pd.read_csv(
            output_root / "D0086_o3_qc_summary_all_hemispheres.csv"
        )
        display(summary)

        depth_tables = []
        for hemisphere in CONFIG["hemispheres"]:
            md_table = pd.read_csv(
                output_root
                / "tables"
                / f"D0086_{hemisphere}_md_depth_summary.csv"
            )
            coordinate_table = pd.read_csv(
                output_root
                / "tables"
                / f"D0086_{hemisphere}_coordinate_depth_qc.csv"
            )
            depth_tables.append(
                md_table.assign(hemisphere=hemisphere).merge(
                    coordinate_table.assign(hemisphere=hemisphere),
                    on=["hemisphere", "depth"],
                )
            )
        depth_summary = pd.concat(depth_tables, ignore_index=True)
        display(depth_summary)
        """
    ),
    code(
        """
        figure_paths = sorted((output_root / "figures").glob("*.png"))
        for figure_path in figure_paths:
            display(Image(filename=str(figure_path), width=900))

        overlay_paths = sorted((output_root / "overlays").glob("*.mgh"))
        display(
            pd.DataFrame(
                {"Freeview_overlay": [str(path) for path in overlay_paths]}
            )
        )
        """
    ),
    markdown(
        """
        ## Final interpretation rule

        PASS/FAIL checks use the configured completeness, weight, coordinate,
        and profile thresholds. Mixed-label triangles and robust column-length
        outliers are descriptive warnings only; they are retained for
        sensitivity analysis and are not automatically removed.
        """
    ),
]


def main() -> None:
    """Write the requested notebook."""

    output = Path("notebooks/qc_o3_fsaverage.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(notebook, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
