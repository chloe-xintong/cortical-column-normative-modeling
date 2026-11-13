#!/usr/bin/env python3
"""Re-render Figure 3C plots from the frozen saved tables.

Description
-----------
Re-render Figure 3C plots from the frozen saved tables.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input

Outputs
-------
- In-memory values returned to the calling code.

Usage
-----
python analysis_results/figure3/computation/rerender_figure3_c.py --help
"""

from pathlib import Path
import argparse

from compute_fixed_component_tracking import plot_thickness_associations
from qc.publication_style import set_publication_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    set_publication_style()
    plot_thickness_associations(args.output_dir)


if __name__ == '__main__':
    main()
