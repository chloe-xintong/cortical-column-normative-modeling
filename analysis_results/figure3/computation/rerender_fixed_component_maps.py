#!/usr/bin/env python3
"""Re-render fixed-component surface maps from previously computed MGH files.

Description
-----------
Re-render fixed-component surface maps from previously computed MGH files.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input
- ``--fsaverage-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure3/computation/rerender_fixed_component_maps.py --help
"""

from pathlib import Path
import argparse
import sys

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qc.publication_style import set_publication_style
from compute_fixed_component_tracking import (
    map_scale,
    plot_loadings,
    plot_raw_adjusted_maps,
    plot_raw_component_maps,
)


def load_map(root, hemisphere, component, condition):
    path = root / "surface_maps" / (
        f"{hemisphere}.figure3_PC{component}_{condition}_fixed_mean.mgh"
    )
    values = np.asarray(nib.load(path).get_fdata(), dtype=float).reshape(-1)
    if not np.isfinite(values).any():
        raise ValueError(f"No finite values in {path}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fsaverage-dir", type=Path, required=True)
    parser.add_argument(
        "--component", type=int, choices=(1, 2),
        help="Render only one D component; omit to render B and both D components.",
    )
    parser.add_argument(
        "--b-only", action="store_true",
        help="Render the two B summary plots and B cortical maps, but not D.",
    )
    args = parser.parse_args()

    set_publication_style()
    raw_maps = {}
    adjusted_maps = {}
    for component_index, component in enumerate((1, 2)):
        for hemisphere in ("lh", "rh"):
            raw_maps[component_index, hemisphere] = load_map(
                args.output_dir, hemisphere, component, "raw"
            )
            adjusted_maps[component_index, hemisphere] = load_map(
                args.output_dir, hemisphere, component, "adjusted"
            )
    component_scales = {
        f'PC{component+1}': map_scale(
            [raw_maps[component, hemisphere] for hemisphere in ('lh', 'rh')] +
            [adjusted_maps[component, hemisphere] for hemisphere in ('lh', 'rh')]
        )
        for component in range(2)
    }
    if args.b_only and args.component is not None:
        parser.error('--b-only and --component cannot be combined')
    b_scales = None
    d_scales = None
    if args.component is None:
        loadings = pd.read_csv(
            args.output_dir / 'tables/figure3B_pca_loadings.csv')
        variance = pd.read_csv(
            args.output_dir / 'tables/figure3B_pca_explained_variance.csv')
        plot_loadings(args.output_dir, loadings, variance)
        b_scales = plot_raw_component_maps(
            args.output_dir, args.fsaverage_dir, raw_maps, component_scales)
    if not args.b_only:
        d_scales = plot_raw_adjusted_maps(
            args.output_dir,
            args.fsaverage_dir,
            raw_maps,
            adjusted_maps,
            components=(0, 1) if args.component is None else (args.component - 1,),
            display_scales=component_scales,
        )
    status_path = args.output_dir / 'qc/figure3_fixed_component_tracking.json'
    if status_path.is_file():
        import json
        status = json.loads(status_path.read_text())
        status['map_colormap'] = 'RdBu_r, matching final Figure 2 surface maps'
        status['map_normalization'] = 'symmetric around zero for signed PCA scores'
        if b_scales is not None:
            status['B_component_display_vmax'] = b_scales
            status.pop('B_common_display_vmax', None)
            status['B_scale_policy'] = (
                'Separate symmetric zero-centered range for each PC; same '
                'PC-specific range reused in D')
        status['D_component_display_vmax'] = component_scales
        status['D_scale_policy'] = (
            'Fixed within PC across B raw map and D raw/adjusted maps, '
            'hemispheres and views')
        if d_scales is not None:
            status['D_component_display_vmax'].update(d_scales)
        status_path.write_text(json.dumps(status, indent=2) + '\n')


if __name__ == "__main__":
    main()
