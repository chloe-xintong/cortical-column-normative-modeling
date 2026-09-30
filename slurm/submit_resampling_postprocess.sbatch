#!/bin/bash
#SBATCH --job-name=resampling-post
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/resampling_post_%j.out
#SBATCH --error=logs/resampling_post_%j.err

set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
mkdir -p logs
export FREESURFER_HOME=/home/apps/freesurfer
export PATH="$FREESURFER_HOME/bin:$PATH"
export SUBJECTS_DIR=/home/apps/freesurfer/subjects

ROOT=/mnt/newStor/paros/paros_WORK/xintong/results/surface_resampling
MANIFEST=manifests/template_comparison_30_subjects.csv
DEPTHS=0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00
mapfile -t SUBJECTS < <(tail -n +2 "$MANIFEST" | cut -d, -f1)
TOTAL=${#SUBJECTS[@]}

for index in "${!SUBJECTS[@]}"; do
    subject="${SUBJECTS[$index]}"
    subject_root="$ROOT/$subject"
    python -u surface_resampling/compare_templates.py \
        --subject-id "$subject" --root "$subject_root" --depths "$DEPTHS" \
        --output "$ROOT/template_comparison_subject_${subject}.csv"
    python -u surface_resampling/adjacent_profile_similarity.py \
        --subject-id "$subject" --template fsaverage6 \
        --profiles "$subject_root/fsaverage6/lh/${subject}_lh_md_profiles.npz" "$subject_root/fsaverage6/rh/${subject}_rh_md_profiles.npz" \
        --surfaces "$SUBJECTS_DIR/fsaverage6/surf/lh.white" "$SUBJECTS_DIR/fsaverage6/surf/rh.white" \
        --output "$ROOT/adjacent_similarity_subject_${subject}.csv"
    echo "POSTPROCESS_PROGRESS $((index + 1))/$TOTAL subject=$subject"
done

echo "RESAMPLING POSTPROCESS SUBJECT-LEVEL METRICS COMPLETED"
