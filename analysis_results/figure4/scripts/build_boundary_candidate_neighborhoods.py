#!/usr/bin/env python3
"""Build reusable dense 0--10-mm candidates for formal ico5 anchors.

Distances, template mask, DK labels, and ico5 anchor mappings are inherited
from the frozen Figure 4A--C implementation. Subject profile validity is not
applied here; it is applied before matching for every subject.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse.csgraph import dijkstra

from boundary_matching import DISTANCE_EDGES_0_10, distance_bin_index
from build_geodesic_pair_tables import build_graph, valid_template_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--formal-pair-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", default="fsaverage")
    parser.add_argument("--batch-size", type=int, default=24)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template_root = args.subjects_dir / args.template
    formal = pd.read_parquet(
        args.formal_pair_file,
        columns=[
            "hemisphere", "ico5_vertex_id", "anchor_vertex",
            "anchor_dk_label", "anchor_region",
        ],
    ).drop_duplicates()
    if formal.duplicated(["hemisphere", "anchor_vertex"]).any():
        raise ValueError("Frozen formal pair table has inconsistent anchor metadata")

    reports = []
    count_parts = []
    output_files: dict[str, str] = {}
    for hemisphere in ("lh", "rh"):
        white, faces = fsio.read_geometry(str(template_root / "surf" / f"{hemisphere}.white"))
        mask, labels = valid_template_mask(template_root, hemisphere)
        anchors = formal.loc[formal["hemisphere"].eq(hemisphere)].sort_values(
            "ico5_vertex_id"
        )
        if not np.array_equal(
            labels[anchors["anchor_vertex"].to_numpy(np.int64)],
            anchors["anchor_dk_label"].to_numpy(np.int64),
        ):
            raise ValueError(f"{hemisphere}: frozen anchor labels differ from aparc")
        graph = build_graph(white, faces)
        output_path = args.output_dir / f"figure4D_dense_candidates_{hemisphere}.parquet"
        writer: pq.ParquetWriter | None = None
        hemisphere_rows = 0
        for start in range(0, len(anchors), args.batch_size):
            batch = anchors.iloc[start : start + args.batch_size]
            anchor_ids = batch["anchor_vertex"].to_numpy(np.int64)
            distances = dijkstra(
                graph, directed=False, indices=anchor_ids,
                limit=float(DISTANCE_EDGES_0_10[-1]),
            )
            parts = []
            for local_index, anchor_row in enumerate(batch.itertuples(index=False)):
                current = distances[local_index]
                candidate_ids = np.flatnonzero(
                    mask & np.isfinite(current) & (current > 0) &
                    (current <= DISTANCE_EDGES_0_10[-1])
                )
                candidate_ids = candidate_ids[candidate_ids != anchor_row.anchor_vertex]
                bin_index = distance_bin_index(current[candidate_ids])
                valid_bin = bin_index >= 0
                candidate_ids = candidate_ids[valid_bin]
                bin_index = bin_index[valid_bin]
                if not len(candidate_ids):
                    continue
                parts.append(
                    pd.DataFrame(
                        {
                            "hemisphere": hemisphere,
                            "anchor_ico5_id": np.int32(anchor_row.ico5_vertex_id),
                            "anchor_vertex": np.int32(anchor_row.anchor_vertex),
                            "anchor_dk_label": np.int16(anchor_row.anchor_dk_label),
                            "anchor_region": str(anchor_row.anchor_region),
                            "candidate_vertex": candidate_ids.astype(np.int32),
                            "candidate_dk_label": labels[candidate_ids].astype(np.int16),
                            "geodesic_distance_mm": current[candidate_ids].astype(np.float32),
                            "distance_bin_index": bin_index.astype(np.int8),
                            "distance_bin": [
                                f"{int(DISTANCE_EDGES_0_10[index])}-"
                                f"{int(DISTANCE_EDGES_0_10[index + 1])}"
                                for index in bin_index
                            ],
                        }
                    )
                )
            if parts:
                frame = pd.concat(parts, ignore_index=True)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
                hemisphere_rows += len(frame)
                counts = (
                    frame.assign(
                        candidate_type=np.where(
                            frame["candidate_dk_label"].eq(frame["anchor_dk_label"]),
                            "within", "across",
                        )
                    )
                    .groupby(
                        [
                            "hemisphere", "anchor_ico5_id", "anchor_vertex",
                            "anchor_dk_label", "anchor_region",
                            "distance_bin_index", "distance_bin", "candidate_type",
                        ],
                        as_index=False,
                        observed=True,
                    )
                    .agg(candidate_count=("candidate_vertex", "size"))
                )
                count_parts.append(counts)
            logging.info(
                "%s anchors %d/%d; candidate rows=%d",
                hemisphere, min(start + len(batch), len(anchors)),
                len(anchors), hemisphere_rows,
            )
        if writer is None:
            raise RuntimeError(f"{hemisphere}: no candidate rows generated")
        writer.close()
        output_files[hemisphere] = str(output_path)
        reports.append(
            {
                "hemisphere": hemisphere,
                "anchor_count": int(len(anchors)),
                "candidate_rows": int(hemisphere_rows),
                "output": str(output_path),
                "sha256": sha256(output_path),
            }
        )

    counts_long = pd.concat(count_parts, ignore_index=True)
    keys = [
        "hemisphere", "anchor_ico5_id", "anchor_vertex", "anchor_dk_label",
        "anchor_region", "distance_bin_index", "distance_bin",
    ]
    counts = counts_long.pivot(
        index=keys, columns="candidate_type", values="candidate_count"
    ).fillna(0).reset_index()
    counts.columns.name = None
    counts = counts.rename(columns={"within": "n_within", "across": "n_across"})
    for column in ("n_within", "n_across"):
        if column not in counts:
            counts[column] = 0
        counts[column] = counts[column].astype(np.int32)
    counts["both_candidate_types"] = counts["n_within"].gt(0) & counts["n_across"].gt(0)
    counts["geometric_k_effective"] = np.minimum(
        15, np.minimum(counts["n_within"], counts["n_across"])
    )
    counts.to_parquet(
        args.output_dir / "figure4D_dense_candidate_counts.parquet",
        index=False,
        compression="zstd",
    )
    summary = (
        counts.groupby(["distance_bin_index", "distance_bin"], as_index=False)
        .agg(
            anchor_bins=("anchor_vertex", "size"),
            anchor_bins_with_both=("both_candidate_types", "sum"),
            within_candidates=("n_within", "sum"),
            across_candidates=("n_across", "sum"),
            median_within_candidates=("n_within", "median"),
            median_across_candidates=("n_across", "median"),
            full_k15_geometric_coverage=(
                "geometric_k_effective", lambda values: values.eq(15).mean()
            ),
        )
    )
    summary.to_csv(args.output_dir / "figure4D_dense_candidate_summary.csv", index=False)
    report = {
        "status": "complete",
        "surface": "full-resolution fsaverage white mesh",
        "anchor_source": "frozen official ico5 mapping used by Figure 4A-C",
        "formal_pair_file": str(args.formal_pair_file),
        "distance_bin_edges_mm": DISTANCE_EDGES_0_10.tolist(),
        "bin_convention": "lower_open_upper_closed_matching_frozen_Figure4A",
        "candidate_mask": "fsaverage cortex label and valid DK annotation",
        "output_files": output_files,
        "hemisphere_reports": reports,
        "candidate_count_rows": int(len(counts)),
    }
    (args.output_dir / "figure4D_dense_candidates_status.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
