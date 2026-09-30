#!/bin/bash
#SBATCH --job-name=spAvgHiK
#SBATCH --partition=normal
#SBATCH --array=0-8%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/spAvgHiK_%A_%a.out
#SBATCH --error=logs/spAvgHiK_%A_%a.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
source /home/apps/ubuntu-22.04/anaconda3/etc/profile.d/conda.sh
conda activate cortical_trajectory_analysis
SUBJECTS=(D0097 D0163 D0231 D0330 D0007 D0029 D0049 D0050 D0051)
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
OUT=/mnt/newStor/paros/paros_WORK/xintong/results/native_surface_analysis/spatial_averaging_k64_k128/shards/${SUBJECT}
mkdir -p "$OUT" logs
echo "START subject=${SUBJECT} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
python -u -m native_surface_analysis.spatial_averaging.analyze_spatial_averaging \
  --subjects "$SUBJECT" \
  --regions precentral superiorfrontal supramarginal superiortemporal \
  --ks 64 128 \
  --random-repeats 0 \
  --output-dir "$OUT"
echo "DONE subject=${SUBJECT}"
