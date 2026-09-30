#!/bin/bash
#SBATCH --job-name=spAvgK40
#SBATCH --partition=normal
#SBATCH --array=0-30%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/spAvgK40_%A_%a.out
#SBATCH --error=logs/spAvgK40_%A_%a.err

set -euo pipefail

cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
source /home/apps/ubuntu-22.04/anaconda3/etc/profile.d/conda.sh
conda activate cortical_trajectory_analysis

SUBJECTS=(
    D0052 D0053 D0057 D0058 D0059 D0060 D0061 D0062 D0063 D0064
    D0065 D0066 D0069 D0070 D0071 D0073 D0074 D0075 D0076 D0078
    D0079 D0080 D0082 D0083 D0084 D0085 D0086 D0087 D0088 D0089
    D0090
)

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
OUT=/mnt/newStor/paros/paros_WORK/xintong/results/native_surface_analysis/spatial_averaging_fixed_k_40/shards/${SUBJECT}

mkdir -p "$OUT" logs
echo "START subject=${SUBJECT} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"

python -u -m native_surface_analysis.spatial_averaging.analyze_spatial_averaging \
    --subjects "$SUBJECT" \
    --regions precentral superiorfrontal supramarginal superiortemporal \
    --ks 1 2 4 8 16 32 \
    --random-repeats 20 \
    --output-dir "$OUT"

echo "DONE subject=${SUBJECT}"
