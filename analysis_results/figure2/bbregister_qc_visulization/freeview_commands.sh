#!/usr/bin/env bash

set -euo pipefail

package_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while IFS=, read -r subject_id group cost; do
    if [[ "$subject_id" == "subject_id" ]]; then
        continue
    fi

    if [[ "$group" == "cost_outlier" ]]; then
        data_dir="$package_dir/cost_outliers"
        prefix="${subject_id}_cost${cost}"
    else
        data_dir="$package_dir/median_controls"
        prefix="${subject_id}_control"
    fi

    echo "Opening ${subject_id} (${group}, cost=${cost})"
    freeview \
        -v "${data_dir}/${prefix}_orig.mgz:visible=0" \
           "${data_dir}/${prefix}_dwi_masked.nii.gz:reg=${data_dir}/${prefix}_DWI2T1.dat" \
        -f "${data_dir}/${prefix}_lh.white" \
           "${data_dir}/${prefix}_rh.white"
done < "$package_dir/selection_manifest.csv"
