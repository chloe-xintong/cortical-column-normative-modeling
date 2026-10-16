#!/bin/bash
#SBATCH --job-name=opt3-ray-5
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/option3_ray_5_%A_%a.out
#SBATCH --error=logs/option3_ray_5_%A_%a.err

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
ln -sfn /home/apps/freesurfer/subjects/fsaverage6 "$SUBJECTS_DIR/fsaverage6"

output_root=/mnt/newStor/paros/paros_WORK/xintong/results/o3_coordinate_fsaverage6_5subjects
depths=0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00

echo "SUBJECT $subject"
echo "PROGRESS 0/2 subject=$subject template=fsaverage6"

for hemi in lh rh; do
    python -u surface_resampling/run_option3_target_driven.py \
        --subject-id "$subject" \
        --subjects-dir "$SUBJECTS_DIR" \
        --metric-image "$md_path" \
        --reg "$reg_path" \
        --template fsaverage6 \
        --hemisphere "$hemi" \
        --depths "$depths" \
        --output \
        "$output_root/$subject/fsaverage6/o3_coordinate/${subject}_${hemi}_profiles.npz"
    echo "PROGRESS $([ "$hemi" = rh ] && echo 2 || echo 1)/2 subject=$subject hemisphere=$hemi"
done

echo "OPTION3 CORRECTED FSAVERAGE6 SUBJECT COMPLETED subject=$subject"
