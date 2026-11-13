#!/usr/bin/env python3
"""Derive fixed 6- and 11-depth ico4 matrices from the frozen 21-depth matrix.

Description
-----------
Derive fixed 6- and 11-depth ico4 matrices from the frozen 21-depth matrix.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--feature-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure5/scripts/build_ico4_depth_sampling_features.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_DEPTH_COUNT = 21
TARGET_DEPTH_COUNTS = (6, 11)


def selected_depth_indices(count: int) -> np.ndarray:
    if count not in TARGET_DEPTH_COUNTS:
        raise ValueError(f"Unsupported depth count: {count}")
    indices = np.linspace(0, SOURCE_DEPTH_COUNT - 1, count, dtype=int)
    if len(np.unique(indices)) != count or indices[0] != 0 or indices[-1] != 20:
        raise RuntimeError(f"Invalid evenly spaced indices for {count} depths: {indices}")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-dir', type=Path, required=True)
    args = parser.parse_args()

    matrices = args.feature_dir / 'matrices'
    metadata_dir = args.feature_dir / 'feature_metadata'
    source_path = matrices / 'X_ico4_md.npy'
    metadata_path = metadata_dir / 'features_ico4_md.csv'
    source = np.load(source_path, mmap_mode='r')
    metadata = pd.read_csv(metadata_path)

    required = {'feature_index', 'depth_index', 'depth_fraction'}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Missing feature metadata columns: {sorted(required-set(metadata.columns))}")
    if not np.array_equal(metadata.feature_index.to_numpy(), np.arange(len(metadata))):
        raise ValueError('Source feature indices are not consecutive')
    if len(metadata) % SOURCE_DEPTH_COUNT:
        raise ValueError('ico4 feature count is not divisible by 21 depths')
    n_nodes = len(metadata) // SOURCE_DEPTH_COUNT
    expected_depth_pattern = np.tile(np.arange(SOURCE_DEPTH_COUNT), n_nodes)
    if not np.array_equal(metadata.depth_index.to_numpy(), expected_depth_pattern):
        raise ValueError('ico4 features are not ordered node-major then depth')
    if source.shape != (196, n_nodes * SOURCE_DEPTH_COUNT):
        raise ValueError(f'Unexpected source matrix shape: {source.shape}')

    report = {
        'status': 'complete',
        'source_matrix': str(source_path),
        'source_shape': list(source.shape),
        'source_depth_count': SOURCE_DEPTH_COUNT,
        'n_ico4_nodes': n_nodes,
        'targets': {},
    }
    for count in TARGET_DEPTH_COUNTS:
        indices = selected_depth_indices(count)
        name = f'ico4_md_depth{count}'
        target_path = matrices / f'X_{name}.npy'
        target = np.lib.format.open_memmap(
            target_path, mode='w+', dtype=np.float32,
            shape=(source.shape[0], n_nodes * count),
        )
        for start in range(0, source.shape[0], 16):
            stop = min(start + 16, source.shape[0])
            block = np.asarray(source[start:stop], dtype=np.float32)
            target[start:stop] = block.reshape(-1, n_nodes, SOURCE_DEPTH_COUNT)[
                :, :, indices
            ].reshape(stop-start, n_nodes*count)
        target.flush()
        del target

        selected = metadata.loc[metadata.depth_index.isin(indices)].copy().reset_index(drop=True)
        selected['source_feature_index'] = selected['feature_index'].astype(np.int64)
        selected['feature_index'] = np.arange(len(selected), dtype=np.int64)
        selected['depth_sampling_count'] = count
        target_metadata = metadata_dir / f'features_{name}.csv'
        selected.to_csv(target_metadata, index=False)
        report['targets'][name] = {
            'matrix': str(target_path),
            'shape': [int(source.shape[0]), int(n_nodes*count)],
            'metadata': str(target_metadata),
            'depth_indices': indices.tolist(),
            'depth_fractions': metadata.loc[
                metadata.depth_index.isin(indices), 'depth_fraction'
            ].drop_duplicates().tolist(),
        }

    report_path = args.feature_dir / 'ico4_depth_sampling_build_report.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
