#!/usr/bin/env python3
"""Create fsaverage annotations showing native-to-template region agreement."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.freesurfer.io import read_annot, read_geometry, write_annot

COLORS = np.array([[190, 190, 190, 255], [0, 180, 70, 255], [230, 40, 40, 255]], dtype=np.int32)
NAMES = [b"unmapped", b"right", b"wrong"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject-id", required=True)
    p.add_argument("--hemisphere", choices=["lh", "rh"], required=True)
    p.add_argument("--mapping-root", type=Path, required=True)
    p.add_argument("--fsaverage-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    subject, hemi = a.subject_id, a.hemisphere
    annot_path = a.fsaverage_dir / "label" / f"{hemi}.aparc.annot"
    labels, _, names = read_annot(str(annot_path))
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    label_to_name = {i: name for i, name in enumerate(names)}
    n = len(read_geometry(str(a.fsaverage_dir / "surf" / f"{hemi}.inflated"))[0])
    if len(labels) != n:
        raise ValueError("fsaverage annotation and surface vertex counts differ")
    status = np.zeros(n, dtype=np.int8)
    files = sorted((a.mapping_root / subject / hemi / "regions").glob(f"{subject}_{hemi}_*_native_to_fsaverage.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No mapping files found for {subject} {hemi}")
    for path in files:
        region = path.name.split(f"{subject}_{hemi}_", 1)[1].removesuffix("_native_to_fsaverage.csv.gz")
        frame = pd.read_csv(path, usecols=["column_valid", "template_vertex_id"])
        frame = frame[frame.column_valid.astype(str).str.lower().isin(["true", "1", "1.0"])]
        for vertex in frame.template_vertex_id.astype(int):
            if not 0 <= vertex < n:
                continue
            agreement = label_to_name.get(int(labels[vertex]), "") == region
            if status[vertex] == 0:
                status[vertex] = 1 if agreement else 2
            elif not agreement:
                status[vertex] = 2
    ctab = COLORS.copy()
    out = a.output_dir / f"{subject}_{hemi}_region_accuracy.annot"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_annot(str(out), labels=status, ctab=ctab, names=NAMES, fill_ctab=True)
    summary = {"subject_id": subject, "hemisphere": hemi,
               "unmapped": int((status == 0).sum()), "right": int((status == 1).sum()),
               "wrong": int((status == 2).sum()),
               "agreement_pct_among_mapped": float(100 * (status == 1).sum() / max(1, (status > 0).sum()))}
    pd.DataFrame([summary]).to_csv(a.output_dir / f"{subject}_{hemi}_region_accuracy_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
