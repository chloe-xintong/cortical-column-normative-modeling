#!/bin/bash
#SBATCH --job-name=opt2-native-30
#SBATCH --partition=normal
#SBATCH --array=0-29
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/option2_native_30_%A_%a.out
#SBATCH --error=logs/option2_native_30_%A_%a.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis

manifest=manifests/template_comparison_30_subjects.csv
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$manifest")
subject=$(printf '%s\n' "$row" | cut -d, -f1)

mapping_root=/mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage6
option3_root=/mnt/newStor/paros/paros_WORK/xintong/results/o3_coordinate_profiles
output_root=/mnt/newStor/paros/paros_WORK/xintong/results/o2_column_native_profiles
completed=0
total=4

record_failure() {
    status=$?
    failure_file="$output_root/failure_summary.csv"
    mkdir -p "$output_root"
    if [ ! -e "$failure_file" ]; then
        printf 'subject_id,stage,exit_status\n' > "$failure_file"
    fi
    printf '%s,%s,%s\n' "$subject" "O2-Column" "$status" >> "$failure_file"
    exit "$status"
}

trap 'record_failure' ERR

echo "SUBJECT $subject"
echo "PROGRESS $completed/$total subject=$subject"

for template in fsaverage6 fsaverage; do
    for hemi in lh rh; do
        python -u surface_resampling/run_option2_native_profiles.py \
            --subject-id "$subject" \
            --hemisphere "$hemi" \
            --template "$template" \
            --mapping-root "$mapping_root" \
            --option3-root "$option3_root" \
            --output \
            "$output_root/$subject/$template/o2_column/${subject}_${hemi}_profiles.npz"
        completed=$((completed + 1))
        echo "PROGRESS $completed/$total subject=$subject template=$template hemisphere=$hemi"
    done
done

echo "OPTION2 NATIVE PROFILE SUBJECT COMPLETED subject=$subject"
