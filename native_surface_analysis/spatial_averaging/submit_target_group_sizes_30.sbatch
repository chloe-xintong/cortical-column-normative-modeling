#!/bin/bash
#SBATCH --job-name=spTarget30
#SBATCH --partition=normal
#SBATCH --array=0-29%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/spTarget30_%A_%a.out
#SBATCH --error=logs/spTarget30_%A_%a.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
source /home/apps/ubuntu-22.04/anaconda3/etc/profile.d/conda.sh
conda activate cortical_trajectory_analysis
mapfile -t SUBJECTS < <(find /mnt/newStor/paros/paros_WORK/xintong/results/native_column_fsaverage6 -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | grep -Ev '^(D0007|D0029|D0049|D0050|D0051|D0097|D0163|D0231|D0330|D0388)$')
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
OUT=/mnt/newStor/paros/paros_WORK/xintong/results/native_surface_analysis/spatial_averaging_target_sizes
mkdir -p "$OUT" logs
echo "START subject=${SUBJECT} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
python -u -m native_surface_analysis.spatial_averaging.analyze_target_group_sizes \
  --subject "$SUBJECT" \
  --regions precentral superiorfrontal supramarginal superiortemporal \
  --target-sizes 50 100 200 400 \
  --output-dir "$OUT"
echo "DONE subject=${SUBJECT}"
