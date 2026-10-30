#!/usr/bin/env bash

set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v freeview >/dev/null 2>&1; then
    echo "freeview was not found. Load or source the FreeSurfer environment first." >&2
    exit 1
fi

open_case() {
    local data_dir="$1"
    local prefix="$2"
    local label="$3"

    echo "Opening ${label}."
    echo "Inspect axial, coronal, and sagittal views; toggle the DWI layer and vary its opacity."
    freeview \
        -v "${data_dir}/${prefix}_orig.mgz:colormap=grayscale" \
           "${data_dir}/${prefix}_dwi_masked.nii.gz:reg=${data_dir}/${prefix}_DWI2T1.dat:colormap=grayscale:opacity=0.5" \
        -f "${data_dir}/${prefix}_lh.white:edgecolor=yellow" \
           "${data_dir}/${prefix}_rh.white:edgecolor=yellow" \
           "${data_dir}/${prefix}_lh.pial:edgecolor=red" \
           "${data_dir}/${prefix}_rh.pial:edgecolor=red"
}

# Open a typical registration first to establish what agreement should look like.
# Close its Freeview window to advance to D0388.
open_case \
    "${package_dir}/median_controls" \
    "D0156_control" \
    "median-cost control D0156 (bbregister cost 0.123435)"

open_case \
    "${package_dir}/cost_outliers" \
    "D0388_cost0.354950" \
    "excluded case D0388 (bbregister cost 0.354950; analysis-mask coverage 2.39%)"
