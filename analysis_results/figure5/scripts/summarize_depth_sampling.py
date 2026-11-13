#!/usr/bin/env python3
"""Summarize the four ico4 depth-sampling models from frozen OOF outputs.

Description
-----------
Summarize the four ico4 depth-sampling models from frozen OOF outputs.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--model-root`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure5/scripts/summarize_depth_sampling.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPRESENTATIONS = (
    'ico4_column_mean_md',
    'ico4_md_depth6',
    'ico4_md_depth11',
    'ico4_md',
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    paired = None
    for representation in REPRESENTATIONS:
        model_dir = args.model_root / representation
        metrics = json.loads((model_dir/'metrics.json').read_text())
        if metrics['status'] != 'complete' or int(metrics['n_evaluated']) != 196:
            raise ValueError(f'{representation}: incomplete model output')
        rows.append({
            'representation': representation,
            'n': int(metrics['n_evaluated']),
            'mae': float(metrics['overall_oof_mae']),
            'pearson_r': float(metrics['overall_oof_pearson_r']),
            'solver': metrics['ridge_solver'],
            'n_features': int(metrics['matrix_shape'][1]),
        })
        prediction = pd.read_csv(model_dir/f'{representation}_oof_predictions.csv')
        component = prediction[['row_index', 'true_age', 'absolute_error']].rename(
            columns={'absolute_error': representation}
        )
        paired = component if paired is None else paired.merge(
            component.drop(columns='true_age'), on='row_index', how='inner',
            validate='one_to_one',
        )
    if paired is None or len(paired) != 196:
        raise RuntimeError('Depth-sampling OOF predictions are not fully paired')
    pd.DataFrame(rows).to_csv(args.output_dir/'model_summary.csv', index=False)
    paired.to_csv(args.output_dir/'paired_absolute_errors.csv', index=False)
    print(json.dumps({'status':'complete', 'n_models':len(rows), 'n_subjects':len(paired)}))


if __name__ == '__main__':
    main()
