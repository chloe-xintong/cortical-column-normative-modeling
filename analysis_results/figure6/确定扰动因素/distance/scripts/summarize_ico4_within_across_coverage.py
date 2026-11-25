#!/usr/bin/env python3
"""Summarize complete within/across-DK candidate coverage for ico4 anchors.

This is a coverage-only analysis.  It builds no correlations, nulls, matched
pairs, or perturbation models, and it never writes a sampled pair table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


REPO = Path(__file__).resolve().parents[5]
SCRIPT_DIR = REPO / "analysis_results/figure4/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_geodesic_pair_tables import (  # noqa: E402
    DISTANCE_EDGES,
    build_graph,
    valid_template_mask,
)
from compute_real_spatial_curves import load_profiles  # noqa: E402


HEMISPHERES = ("lh", "rh")
PAIR_TYPES = ("within", "across")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "manifests/figure4_coverage95_pass_313_subjects.csv",
    )
    parser.add_argument(
        "--subjects-dir", type=Path,
        default=Path("/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer"),
    )
    parser.add_argument(
        "--ico4-node-table", type=Path,
        default=Path("/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs/template/ico4_vertex_order.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO / "analysis_results/figure6/确定扰动因素/distance/candidate_coverage",
    )
    parser.add_argument("--template", default="fsaverage")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--rebuild-geometry-cache", action="store_true")
    return parser.parse_args()


def distance_label(bin_index: int) -> str:
    return f"({DISTANCE_EDGES[bin_index]:.0f},{DISTANCE_EDGES[bin_index + 1]:.0f}]"


def build_geometry_cache(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, sparse.csr_matrix]]:
    cache_dir = args.output_dir / "geometry_cache"
    metadata_path = cache_dir / "ico4_anchor_bins.parquet"
    matrix_paths = {
        pair_type: cache_dir / f"ico4_{pair_type}_candidate_incidence.npz"
        for pair_type in PAIR_TYPES
    }
    if (
        not args.rebuild_geometry_cache
        and metadata_path.is_file()
        and all(path.is_file() for path in matrix_paths.values())
    ):
        metadata = pd.read_parquet(metadata_path)
        matrices = {key: sparse.load_npz(path).tocsr() for key, path in matrix_paths.items()}
        return metadata, matrices

    cache_dir.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(args.ico4_node_table)
    expected_columns = {
        "hemisphere", "ico_level", "ico_node_id", "fsaverage_vertex_id",
        "dk_label_id", "dk_label_name", "in_analysis_mask",
    }
    if not expected_columns.issubset(nodes.columns):
        raise RuntimeError("ico4 node table lacks required columns")
    nodes = nodes.loc[nodes["in_analysis_mask"].astype(bool)].copy()
    if len(nodes) != 4687 or not nodes["ico_level"].eq(4).all():
        raise RuntimeError(f"Expected 4,687 masked ico4 nodes, found {len(nodes)}")

    n_bins = len(DISTANCE_EDGES) - 1
    row_parts: dict[str, list[np.ndarray]] = {key: [] for key in PAIR_TYPES}
    col_parts: dict[str, list[np.ndarray]] = {key: [] for key in PAIR_TYPES}
    metadata_rows: list[dict[str, object]] = []
    row_offset = 0
    template_root = args.subjects_dir / args.template

    for hemisphere in HEMISPHERES:
        white, faces = fsio.read_geometry(template_root / "surf" / f"{hemisphere}.white")
        mask, labels = valid_template_mask(template_root, hemisphere)
        graph = build_graph(white, faces)
        hemi_nodes = nodes.loc[nodes["hemisphere"].eq(hemisphere)].sort_values(
            "ico_node_id", kind="stable"
        )
        anchors = hemi_nodes["fsaverage_vertex_id"].to_numpy(dtype=np.int64)
        if not mask[anchors].all():
            raise RuntimeError(f"{hemisphere}: ico4 anchor outside formal template mask")
        if not np.array_equal(
            labels[anchors], hemi_nodes["dk_label_id"].to_numpy(dtype=np.int64)
        ):
            raise RuntimeError(f"{hemisphere}: ico4 table/template DK labels differ")

        for batch_start in range(0, len(anchors), args.batch_size):
            batch = anchors[batch_start : batch_start + args.batch_size]
            distances = dijkstra(graph, directed=False, indices=batch, limit=30.0)
            for local_index, anchor in enumerate(batch):
                node = hemi_nodes.iloc[batch_start + local_index]
                current = distances[local_index]
                for bin_index, (lower, upper) in enumerate(
                    zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:])
                ):
                    in_bin = mask & (current > lower) & (current <= upper)
                    in_bin[anchor] = False
                    pools = {
                        "within": np.flatnonzero(in_bin & (labels == labels[anchor])),
                        "across": np.flatnonzero(in_bin & (labels != labels[anchor])),
                    }
                    metadata_rows.append({
                        "row_index": row_offset,
                        "hemisphere": hemisphere,
                        "ico4_node_id": int(node["ico_node_id"]),
                        "anchor_vertex": int(anchor),
                        "anchor_dk_label": int(labels[anchor]),
                        "anchor_dk_region": str(node["dk_label_name"]),
                        "distance_bin_index": bin_index,
                        "distance_bin": distance_label(bin_index),
                        "within_n_geometry": int(len(pools["within"])),
                        "across_n_geometry": int(len(pools["across"])),
                    })
                    for pair_type, candidates in pools.items():
                        row_parts[pair_type].append(
                            np.full(len(candidates), row_offset, dtype=np.int32)
                        )
                        col_parts[pair_type].append(candidates.astype(np.int32, copy=False))
                    row_offset += 1

    metadata = pd.DataFrame(metadata_rows)
    shape = (len(metadata), 163842)
    matrices: dict[str, sparse.csr_matrix] = {}
    for pair_type in PAIR_TYPES:
        rows = np.concatenate(row_parts[pair_type])
        cols = np.concatenate(col_parts[pair_type])
        matrix = sparse.coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=shape
        ).tocsr()
        if matrix.max() != 1:
            raise RuntimeError(f"Duplicate {pair_type} candidate incidence")
        sparse.save_npz(matrix_paths[pair_type], matrix, compressed=True)
        matrices[pair_type] = matrix
    metadata.to_parquet(metadata_path, index=False, compression="zstd")
    return metadata, matrices


def load_subject_masks(manifest: pd.DataFrame, hemisphere: str) -> np.ndarray:
    result = np.zeros((len(manifest), 163842), dtype=np.int8)
    for subject_index, row in manifest.iterrows():
        _, valid = load_profiles(Path(row[f"clean_profile_{hemisphere}"]))
        if len(valid) != 163842:
            raise RuntimeError(
                f"{row.subject_id} {hemisphere}: expected 163842 profile vertices, found {len(valid)}"
            )
        result[subject_index] = valid.astype(np.int8)
    return result


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        raise RuntimeError("Cannot summarize an empty denominator")
    return {
        "denominator_n": int(n),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "proportion_0": float(np.mean(values == 0)),
        "proportion_1_2": float(np.mean((values >= 1) & (values <= 2))),
        "proportion_3_5": float(np.mean((values >= 3) & (values <= 5))),
        "proportion_6_14": float(np.mean((values >= 6) & (values <= 14))),
        "proportion_ge_15": float(np.mean(values >= 15)),
    }


def summarize_by_bin(
    metadata: pd.DataFrame,
    values: dict[str, np.ndarray],
    level: str,
    denominator: str,
    anchor_valid: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    for bin_index in range(len(DISTANCE_EDGES) - 1):
        bin_rows = metadata["distance_bin_index"].to_numpy() == bin_index
        for pair_type in PAIR_TYPES:
            current = values[pair_type][bin_rows]
            valid = None if anchor_valid is None else anchor_valid[bin_rows]
            if denominator == "valid_anchor":
                current = current[valid]
            row = {
                "level": level,
                "denominator": denominator,
                "pair_type": pair_type,
                "distance_bin_index": bin_index,
                "distance_bin": distance_label(bin_index),
                "anchor_invalid_n": 0 if valid is None else int((~valid).sum()),
                "anchor_invalid_proportion": 0.0 if valid is None else float((~valid).mean()),
            }
            row.update(distribution_summary(current))
            rows.append(row)
    return pd.DataFrame(rows)


def common_coverage(
    metadata: pd.DataFrame,
    values: dict[str, np.ndarray],
    level: str,
    denominator: str,
    anchor_valid: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    for bin_index in range(len(DISTANCE_EDGES) - 1):
        select = metadata["distance_bin_index"].to_numpy() == bin_index
        within, across = values["within"][select], values["across"][select]
        valid = None if anchor_valid is None else anchor_valid[select]
        if denominator == "valid_anchor":
            within, across = within[valid], across[valid]
        rows.append({
            "level": level,
            "denominator": denominator,
            "distance_bin_index": bin_index,
            "distance_bin": distance_label(bin_index),
            "denominator_n": int(len(within)),
            "anchor_invalid_n": 0 if valid is None else int((~valid).sum()),
            "anchor_invalid_proportion": 0.0 if valid is None else float((~valid).mean()),
            "proportion_both_ge_1": float(np.mean((within >= 1) & (across >= 1))),
            "proportion_both_ge_15": float(np.mean((within >= 15) & (across >= 15))),
        })
    return pd.DataFrame(rows)


def regional_across_summary(
    metadata: pd.DataFrame,
    across: np.ndarray,
    level: str,
    denominator: str,
    anchor_valid: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = metadata[[
        "hemisphere", "anchor_dk_label", "anchor_dk_region",
        "distance_bin_index", "distance_bin",
    ]].copy()
    frame["across"] = across
    if anchor_valid is not None:
        frame["anchor_valid"] = anchor_valid
        if denominator == "valid_anchor":
            frame = frame.loc[frame["anchor_valid"]].copy()
    group = [
        "hemisphere", "anchor_dk_label", "anchor_dk_region",
        "distance_bin_index", "distance_bin",
    ]
    result = frame.groupby(group, observed=True, sort=True).agg(
        denominator_n=("across", "size"),
        across_nonzero_proportion=("across", lambda x: float(np.mean(x >= 1))),
        across_ge_15_proportion=("across", lambda x: float(np.mean(x >= 15))),
    ).reset_index()
    result.insert(0, "denominator", denominator)
    result.insert(0, "level", level)
    return result


def write_interpretation(
    args: argparse.Namespace,
    template_summary: pd.DataFrame,
    subject_summary: pd.DataFrame,
    common: pd.DataFrame,
    regional: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    geometry_across = template_summary.query("pair_type == 'across'").set_index("distance_bin")
    available_across = subject_summary.query(
        "pair_type == 'across' and denominator == 'valid_anchor'"
    ).set_index("distance_bin")
    comparison = pd.DataFrame({
        "geometry": geometry_across["proportion_ge_15"],
        "available_valid_anchor": available_across["proportion_ge_15"],
    })
    comparison["loss"] = comparison["geometry"] - comparison["available_valid_anchor"]
    best_common = common.query(
        "level == 'subject' and denominator == 'valid_anchor'"
    ).sort_values("proportion_both_ge_15", ascending=False).head(5)
    best_regions = regional.query(
        "level == 'subject' and denominator == 'valid_anchor'"
    ).groupby(["hemisphere", "anchor_dk_region"], observed=True).agg(
        mean_across_ge_15=("across_ge_15_proportion", "mean")
    ).reset_index().sort_values("mean_across_ge_15", ascending=False).head(12)
    lines = [
        "# Ico4 within/across DK candidate coverage",
        "",
        "## Methods",
        "",
        f"The cohort is the formal 313-person manifest (`{args.manifest}`). Ico4 anchors are the "
        f"4,687 template-mask nodes in `{args.ico4_node_table}`; neighbors remain all 163,842 "
        "vertices per hemisphere. Distances are shortest paths on the full-resolution fsaverage "
        "white mesh, in the right-closed bins (0,2], ..., (28,30] mm. The formal cortex mask and "
        "aparc/DK labels exclude medial wall/unknown/corpus-callosum vertices. Hemispheres are never "
        "combined as spatial neighborhoods.",
        "",
        "`N_geometry` uses the template mask only. `N_available` additionally requires the formal "
        "clean-profile validity mask for both anchor and neighbor; invalid anchors are zero in the "
        "all-subject-anchor denominator and excluded from the valid-anchor denominator. No candidate "
        "set was truncated to K, no distance matching was performed, and no correlations/nulls/models "
        "were computed.",
        "",
        "## Configuration difference from formal Figure 4",
        "",
        "The formal notebook uses ico5 anchors. This requested pilot intentionally changes anchors "
        "to ico4 while retaining the formal full-resolution neighbors, surface, bins, labels, masks, "
        "and 313-person cohort.",
        "",
        "## N_retained status",
        "",
        "Not computed. The located formal fixed table is ico5 and within-DK only, so it is not "
        "compatible with this ico4 within/across coverage analysis. It was neither reused nor replaced.",
        "",
        "## Result guide",
        "",
        "Exact values are in the CSV files. Across-DK fractions reaching 15 candidates, comparing "
        "template geometry with subject availability among valid anchors:",
        "",
        "```text",
        comparison.reset_index().to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "```",
        "",
        "The five distance bins with the highest subject-level joint within/across K=15 coverage are:",
        "",
        "```text",
        best_common[["distance_bin", "proportion_both_ge_15"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        ),
        "```",
        "",
        "Regions with the highest mean across-bin subject-level across K=15 coverage are:",
        "",
        "```text",
        best_regions.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "```",
        "",
        "These are coverage diagnostics only. They do not establish that K=15 is statistically "
        "stable; that question requires the planned local-feature stability analysis.",
        "",
        f"Subjects: {len(manifest)}. Template anchor-bins: {4687 * 15}.",
    ]
    (args.output_dir / "METHODS_AND_RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    if len(manifest) != 313 or manifest["subject_id"].duplicated().any():
        raise RuntimeError("Formal manifest must contain 313 unique subjects")
    if not manifest["input_complete"].astype(bool).all():
        raise RuntimeError("Formal manifest contains incomplete inputs")

    metadata, matrices = build_geometry_cache(args)
    if len(metadata) != 4687 * 15:
        raise RuntimeError(f"Unexpected number of anchor-bin rows: {len(metadata)}")
    geometry = {
        pair_type: metadata[f"{pair_type}_n_geometry"].to_numpy(dtype=np.int64)
        for pair_type in PAIR_TYPES
    }
    template_summary = summarize_by_bin(metadata, geometry, "template", "anchor")
    template_common = common_coverage(metadata, geometry, "template", "anchor")
    template_regional = regional_across_summary(
        metadata, geometry["across"], "template", "anchor"
    )

    available_parts = {pair_type: [] for pair_type in PAIR_TYPES}
    anchor_valid_parts = []
    subject_index_parts = []
    for hemisphere in HEMISPHERES:
        subject_masks = load_subject_masks(manifest, hemisphere)
        hemi_select = metadata["hemisphere"].eq(hemisphere).to_numpy()
        hemi_meta = metadata.loc[hemi_select]
        anchors = hemi_meta["anchor_vertex"].to_numpy(dtype=np.int64)
        anchor_valid = subject_masks[:, anchors].T.astype(bool)
        anchor_valid_flat = anchor_valid.reshape(-1)
        anchor_valid_parts.append(anchor_valid_flat)
        subject_index_parts.append(np.tile(np.arange(len(manifest)), len(hemi_meta)))
        for pair_type in PAIR_TYPES:
            counts = matrices[pair_type][hemi_select].astype(np.int32).dot(subject_masks.T)
            counts = np.asarray(counts, dtype=np.int32)
            counts[~anchor_valid] = 0
            available_parts[pair_type].append(counts.reshape(-1))

    # Expand metadata in the same hemisphere-block, row-major order used above.
    expanded_parts = []
    for hemisphere in HEMISPHERES:
        hemi_meta = metadata.loc[metadata["hemisphere"].eq(hemisphere)]
        expanded_parts.append(hemi_meta.loc[hemi_meta.index.repeat(len(manifest))])
    expanded = pd.concat(expanded_parts, ignore_index=True)
    expanded["subject_index"] = np.concatenate(subject_index_parts)
    expanded["subject_id"] = manifest["subject_id"].to_numpy()[expanded["subject_index"]]
    available = {key: np.concatenate(parts) for key, parts in available_parts.items()}
    anchor_valid = np.concatenate(anchor_valid_parts)

    subject_all = summarize_by_bin(
        expanded, available, "subject", "all_subject_anchor", anchor_valid
    )
    subject_valid = summarize_by_bin(
        expanded, available, "subject", "valid_anchor", anchor_valid
    )
    subject_summary = pd.concat([subject_all, subject_valid], ignore_index=True)
    subject_common = pd.concat([
        common_coverage(expanded, available, "subject", "all_subject_anchor", anchor_valid),
        common_coverage(expanded, available, "subject", "valid_anchor", anchor_valid),
    ], ignore_index=True)
    subject_regional = pd.concat([
        regional_across_summary(
            expanded, available["across"], "subject", "all_subject_anchor", anchor_valid
        ),
        regional_across_summary(
            expanded, available["across"], "subject", "valid_anchor", anchor_valid
        ),
    ], ignore_index=True)

    template_summary.to_csv(args.output_dir / "A_template_geometry_coverage.csv", index=False)
    subject_summary.to_csv(args.output_dir / "B_subject_available_coverage.csv", index=False)
    pd.DataFrame([{
        "status": "not_computed",
        "reason": "No compatible fixed ico4 within/across pair table exists; formal table is ico5 within-only.",
    }]).to_csv(args.output_dir / "B_retained_coverage_status.csv", index=False)
    common = pd.concat([template_common, subject_common], ignore_index=True)
    common.to_csv(args.output_dir / "C_joint_coverage.csv", index=False)
    regional = pd.concat([template_regional, subject_regional], ignore_index=True)
    regional.to_csv(args.output_dir / "D_regional_across_coverage.csv", index=False)

    config = {
        "status": "complete",
        "scope": "coverage_only",
        "n_subjects": len(manifest),
        "n_ico4_anchors": int(metadata[["hemisphere", "anchor_vertex"]].drop_duplicates().shape[0]),
        "n_anchor_bins": len(metadata),
        "surface": "full-resolution fsaverage white mesh",
        "neighbors": "full-resolution same-hemisphere template-mask vertices",
        "distance_bins": [distance_label(i) for i in range(15)],
        "formal_notebook_anchor_difference": "formal=ico5; requested pilot=ico4",
        "n_retained": "not_computed_no_compatible_fixed_ico4_table",
        "manifest": str(args.manifest.resolve()),
        "ico4_node_table": str(args.ico4_node_table.resolve()),
        "template_root": str((args.subjects_dir / args.template).resolve()),
        "elapsed_seconds": time.time() - start,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(config, indent=2) + "\n")
    write_interpretation(
        args, template_summary, subject_summary, common, regional, manifest
    )

    compact = common[[
        "level", "denominator", "distance_bin",
        "proportion_both_ge_1", "proportion_both_ge_15",
    ]]
    print(compact.to_string(index=False))
    print(f"output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
