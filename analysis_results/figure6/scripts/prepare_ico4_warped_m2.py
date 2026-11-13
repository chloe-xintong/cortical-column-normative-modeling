#!/usr/bin/env python3
"""Freeze the 186-person ico4 cohort and build local FreeSurfer thickness.

Description
-----------
Freeze the 186-person ico4 cohort and build local FreeSurfer thickness.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--subject-order`` command-line input
- ``--source-audit`` command-line input
- ``--source-folds`` command-line input
- ``--cohort-manifest`` command-line input
- ``--ico4-md`` command-line input
- ``--ico4-md-metadata`` command-line input
- ``--vertex-order`` command-line input
- ``--profile-root`` command-line input
- ``--freesurfer-root`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/prepare_ico4_warped_m2.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd


EXCLUSIONS = {
    "D0271": "source_image_structural_abnormality",
    "D0285": "source_image_structural_abnormality",
}
DEPTHS = np.linspace(0.0, 1.0, 21)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject-order", type=Path, required=True)
    p.add_argument("--source-audit", type=Path, required=True)
    p.add_argument("--source-folds", type=Path, required=True)
    p.add_argument("--cohort-manifest", type=Path, required=True)
    p.add_argument("--ico4-md", type=Path, required=True)
    p.add_argument("--ico4-md-metadata", type=Path, required=True)
    p.add_argument("--vertex-order", type=Path, required=True)
    p.add_argument("--profile-root", type=Path, required=True)
    p.add_argument("--freesurfer-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--smoke-nodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261108)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    out = a.output_dir
    for d in (out, out / "inputs", out / "smoke", out / "oof", out / "full_models", out / "logs"):
        d.mkdir(parents=True, exist_ok=True)

    subject_order = pd.read_csv(a.subject_order, dtype={"subject_id": str})
    audit = pd.read_csv(a.source_audit, dtype={"subject_id": str})
    cohort = pd.read_csv(a.cohort_manifest, dtype={"subject_id": str})
    if int(audit.included.sum()) != 188:
        raise RuntimeError("Source audit is not the frozen 188-person regional cohort")
    for sid, reason in EXCLUSIONS.items():
        hit = audit.subject_id.eq(sid)
        if hit.sum() != 1 or not bool(audit.loc[hit, "included"].iloc[0]):
            raise RuntimeError(f"Expected one included source row for {sid}")
        audit.loc[hit, "included"] = False
        audit.loc[hit, "exclusion_reason"] = reason
    audit.to_csv(out / "frozen_cohort_manifest.csv", index=False)
    included = audit.loc[audit.included].merge(
        subject_order[["subject_id", "row_index"]], on="subject_id", validate="one_to_one"
    ).merge(cohort[["subject_id", "SUBJECT_SEX"]], on="subject_id", validate="one_to_one")
    included["sex_code"] = included.SUBJECT_SEX.astype(int)
    if len(included) != 186 or set(EXCLUSIONS) & set(included.subject_id):
        raise RuntimeError("Frozen ico4 cohort is not exactly 186 subjects")

    source_folds = json.loads(a.source_folds.read_text(encoding="utf-8"))
    retained = set(included.subject_id)
    folds, assignments = [], []
    for item in source_folds["folds"]:
        fold = int(item["outer_fold"])
        train = [str(s) for s in item["train_subject_ids"] if str(s) in retained]
        test = [str(s) for s in item["test_subject_ids"] if str(s) in retained]
        if set(train) & set(test) or set(train) | set(test) != retained:
            raise RuntimeError(f"Invalid retained partition for fold {fold}")
        folds.append({"outer_fold": fold, "train_subject_ids": train, "test_subject_ids": test})
        assignments.extend({"subject_id": s, "outer_fold": fold} for s in test)
    assign = pd.DataFrame(assignments)
    if len(assign) != 186 or assign.subject_id.nunique() != 186:
        raise RuntimeError("Filtered folds do not partition the retained cohort exactly once")
    write_json(out / "fold_assignments.json", {
        "seed": source_folds.get("seed"), "n_splits": 5,
        "source": str(a.source_folds.resolve()), "source_sha256": sha256(a.source_folds),
        "assignment_policy": "Original 188-person outer folds retained; D0271 and D0285 removed without reassignment.",
        "excluded": EXCLUSIONS, "folds": folds,
    })
    assign.sort_values(["outer_fold", "subject_id"]).to_csv(out / "outer_fold_assignments.csv", index=False)
    balance = assign.merge(included[["subject_id", "age", "age_bin", "sex_code"]], on="subject_id")
    balance.groupby(["outer_fold", "age_bin", "sex_code"], observed=False).size().rename("n").reset_index().to_csv(
        out / "fold_balance.csv", index=False
    )

    md_meta = pd.read_csv(a.ico4_md_metadata)
    vertex = pd.read_csv(a.vertex_order).query("in_analysis_mask == True").reset_index(drop=True)
    md = np.load(a.ico4_md, mmap_mode="r")
    if md.shape != (len(subject_order), 4687 * 21) or len(md_meta) != 4687 * 21 or len(vertex) != 4687:
        raise RuntimeError("Unexpected ico4 matrix or metadata dimensions")
    if not np.array_equal(md_meta.feature_index.to_numpy(), np.arange(len(md_meta))):
        raise RuntimeError("ico4 feature_index is not contiguous")
    for node in range(4687):
        block = md_meta.iloc[node * 21:(node + 1) * 21]
        v = vertex.iloc[node]
        if not np.allclose(block.depth_fraction, DEPTHS):
            raise RuntimeError(f"Noncanonical depth order at node {node}")
        for col in ("hemisphere", "ico_node_id", "fsaverage_vertex_id", "dk_label_id", "dk_label_name"):
            if not np.all(block[col].to_numpy() == v[col]):
                raise RuntimeError(f"MD/vertex mismatch at node {node}: {col}")
    node_meta = vertex[["hemisphere", "ico_level", "ico_node_id", "fsaverage_vertex_id", "dk_label_id", "dk_label_name"]].copy()
    node_meta.insert(0, "node_index", np.arange(4687))
    node_meta.to_csv(out / "ico4_node_metadata.csv", index=False)
    md_meta.to_csv(out / "ico4_node_depth_metadata.csv", index=False)

    thickness_path = out / "inputs" / "X_ico4_local_freesurfer_thickness.npy"
    thickness = np.lib.format.open_memmap(thickness_path, mode="w+", dtype=np.float32,
                                         shape=(len(subject_order), 4687))
    thickness[:] = np.nan
    qc_rows = []
    for row_i, sid in subject_order[["row_index", "subject_id"]].itertuples(index=False):
        parts = []
        for hemi in ("lh", "rh"):
            hemi_nodes = vertex.loc[vertex.hemisphere.eq(hemi)]
            ids = hemi_nodes.fsaverage_vertex_id.to_numpy(int)
            with np.load(a.profile_root / sid / "profiles" / f"{sid}_{hemi}_md_profiles.npz", allow_pickle=False) as p:
                full_ids = p["fsaverage_vertex_id"].astype(int)
                triangles = p["native_triangle_vertex_ids"][ids].astype(int)
                weights = p["barycentric_weights"][ids].astype(float)
                mapping_valid = p["mapping_valid"][ids].astype(bool)
            if not np.array_equal(full_ids, np.arange(163842)):
                raise RuntimeError(f"Unstable fsaverage order: {sid} {hemi}")
            native = fsio.read_morph_data(a.freesurfer_root / sid / "surf" / f"{hemi}.thickness").astype(float)
            valid = mapping_valid & np.all(triangles >= 0, axis=1) & np.all(triangles < len(native), axis=1)
            valid &= np.all(np.isfinite(weights), axis=1) & np.isclose(weights.sum(1), 1.0, atol=1e-5)
            values = np.full(len(ids), np.nan)
            values[valid] = np.sum(native[triangles[valid]] * weights[valid], axis=1)
            values[~np.isfinite(values) | (values <= 0)] = np.nan
            parts.append(values)
            qc_rows.append({"subject_id": sid, "row_index": int(row_i), "hemisphere": hemi,
                            "n_nodes": len(values), "n_valid": int(np.isfinite(values).sum()),
                            "valid_fraction": float(np.isfinite(values).mean())})
        thickness[int(row_i)] = np.concatenate(parts).astype(np.float32)
        if (int(row_i) + 1) % 25 == 0:
            thickness.flush()
    thickness.flush()
    pd.DataFrame(qc_rows).to_csv(out / "local_thickness_mapping_qc.csv", index=False)

    rows = included.row_index.to_numpy(int)
    md_valid = np.isfinite(md[rows]).sum(axis=0)
    thickness_valid = np.isfinite(thickness[rows]).sum(axis=0)
    miss = md_meta[["feature_index", "hemisphere", "ico_node_id", "fsaverage_vertex_id", "dk_label_id", "dk_label_name",
                    "depth_index", "depth_fraction"]].copy()
    miss["n_cohort"] = 186
    miss["n_valid_md"] = md_valid
    miss["n_missing_md"] = 186 - md_valid
    miss["md_missing_fraction"] = (186 - md_valid) / 186
    miss["n_valid_local_thickness"] = np.repeat(thickness_valid, 21)
    miss["n_missing_local_thickness"] = 186 - miss.n_valid_local_thickness
    miss.to_parquet(out / "ico4_node_depth_missingness.parquet", index=False)

    node_coverage = miss.groupby(["hemisphere", "ico_node_id", "dk_label_name"], sort=False).agg(
        node_index=("feature_index", lambda x: int(x.iloc[0] // 21)),
        minimum_valid_md=("n_valid_md", "min"), mean_valid_md=("n_valid_md", "mean"),
        valid_thickness=("n_valid_local_thickness", "first")
    ).reset_index()
    rng = np.random.default_rng(a.seed)
    chosen = []
    # Stratify over hemisphere, DK region, and coverage quartile; explicitly retain left parahippocampal.
    node_coverage["coverage_bin"] = pd.qcut(node_coverage.mean_valid_md, 4, labels=False, duplicates="drop")
    forced = node_coverage.query("hemisphere == 'lh' and dk_label_name == 'parahippocampal'")
    if not forced.empty:
        chosen.append(int(forced.sort_values("mean_valid_md").iloc[len(forced) // 2].node_index))
    groups = list(node_coverage.groupby(["hemisphere", "coverage_bin", "dk_label_name"], observed=True))
    rng.shuffle(groups)
    for _, g in groups:
        candidates = [int(x) for x in g.node_index if int(x) not in chosen]
        if candidates:
            chosen.append(int(rng.choice(candidates)))
        if len(chosen) >= a.smoke_nodes:
            break
    if len(chosen) < a.smoke_nodes:
        pool = [int(x) for x in node_coverage.node_index if int(x) not in chosen]
        chosen.extend(rng.choice(pool, size=a.smoke_nodes - len(chosen), replace=False).astype(int).tolist())
    smoke = node_coverage[node_coverage.node_index.isin(chosen)].copy().sort_values("node_index")
    smoke.to_csv(out / "smoke_nodes.csv", index=False)
    write_json(out / "preflight_report.json", {
        "status": "complete", "cohort_n": 186, "excluded": EXCLUSIONS,
        "fold_test_sizes": {str(k): int(v) for k, v in balance.groupby("outer_fold").size().items()},
        "n_nodes": 4687, "n_depths": 21, "n_node_depths": 4687 * 21,
        "md_valid_minimum": int(md_valid.min()), "md_complete_node_depths": int((md_valid == 186).sum()),
        "thickness_valid_minimum": int(thickness_valid.min()),
        "smoke_nodes": len(smoke), "smoke_seed": a.seed,
        "response": "raw ico4 MD; no mean removal and no imputation",
        "thickness": "native FreeSurfer thickness mapped with stored native triangles and barycentric weights; not column length",
        "fixed_mask": "Figure 5 fixed 4,687-node ico4 analysis mask",
    })
    print(json.dumps({"status": "complete", "cohort_n": 186, "smoke_nodes": len(smoke),
                      "fold_sizes": balance.groupby("outer_fold").size().to_dict()}))


if __name__ == "__main__":
    main()
