#!/bin/bash
#SBATCH --job-name=surf-resample-test
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/surf_resample_trial_%j.out
#SBATCH --error=logs/surf_resample_trial_%j.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
mkdir -p logs

export FREESURFER_HOME=/home/apps/freesurfer
export SUBJECTS_DIR="${TMPDIR:-/tmp}/surface_resampling_subjects"
export PATH="$FREESURFER_HOME/bin:$PATH"

mkdir -p "$SUBJECTS_DIR"
ln -sfn /mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer/D0097 "$SUBJECTS_DIR/D0097"
ln -sfn /home/apps/freesurfer/subjects/fsaverage6 "$SUBJECTS_DIR/fsaverage6"

echo "Host: $(hostname)"
echo "FreeSurfer directory: ${FREESURFER_HOME:-not_set}"
echo "mri_vol2surf: $(command -v mri_vol2surf || true)"

python -u surface_resampling/run_vol2surf.py \
    --subject-id D0097 \
    --subjects-dir "$SUBJECTS_DIR" \
    --metric-image /mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi/D0097/D0097_md.nii.gz \
    --reg /mnt/newStor/paros/paros_WORK/xintong/results/column_analysis/D0097/registration/D0097_DWI2T1.dat \
    --template fsaverage6 \
    --hemisphere lh \
    --depths 0.50 \
    --output-dir /mnt/newStor/paros/paros_WORK/xintong/results/surface_resampling_trial/D0097_lh
