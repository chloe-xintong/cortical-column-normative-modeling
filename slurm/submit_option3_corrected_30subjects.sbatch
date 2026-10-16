#!/bin/bash
#SBATCH --job-name=opt3-ray-30
#SBATCH --partition=normal
#SBATCH --array=0-29
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/option3_ray_30_%A_%a.out
#SBATCH --error=logs/option3_ray_30_%A_%a.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis

export FREESURFER_HOME=/home/apps/freesurfer
export PATH="$FREESURFER_HOME/bin:$PATH"
export SUBJECTS_DIR="${TMPDIR:-/tmp}/option3_ray_subjects"

manifest=manifests/template_comparison_30_subjects.csv
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$manifest")
subject=$(printf '%s\n' "$row" | cut -d, -f1)
md_path=$(printf '%s\n' "$row" | cut -d, -f3)
reg_path=$(printf '%s\n' "$row" | cut -d, -f4)

mkdir -p "$SUBJECTS_DIR"
ln -sfn \
    "/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer/$subject" \
    "$SUBJECTS_DIR/$subject"
ln -sfn /home/apps/freesurfer/subjects/fsaverage "$SUBJECTS_DIR/fsaverage"
ln -sfn /home/apps/freesurfer/subjects/fsaverage6 "$SUBJECTS_DIR/fsaverage6"

output_root=/mnt/newStor/paros/paros_WORK/xintong/results/o3_coordinate_profiles
depths=0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
total=4
completed=0

record_failure() {
    status=$?
    failure_file="$output_root/failure_summary.csv"
    mkdir -p "$output_root"
    if [ ! -e "$failure_file" ]; then
        printf 'subject_id,stage,exit_status\n' > "$failure_file"
    fi
    printf '%s,%s,%s\n' "$subject" "${1:-unknown}" "$status" >> "$failure_file"
    exit "$status"
}

trap 'record_failure "option3_mapping"' ERR

echo "SUBJECT $subject"
echo "PROGRESS $completed/$total subject=$subject"

for template in fsaverage6 fsaverage; do
    for hemi in lh rh; do
        python -u surface_resampling/run_option3_target_driven.py \
            --subject-id "$subject" \
            --subjects-dir "$SUBJECTS_DIR" \
            --metric-image "$md_path" \
            --reg "$reg_path" \
            --template "$template" \
            --hemisphere "$hemi" \
            --depths "$depths" \
            --output \
            "$output_root/$subject/$template/o3_coordinate/${subject}_${hemi}_profiles.npz"
        completed=$((completed + 1))
        echo "PROGRESS $completed/$total subject=$subject template=$template hemisphere=$hemi"
    done
done

echo "OPTION3 CORRECTED TEMPLATES SUBJECT COMPLETED subject=$subject"
