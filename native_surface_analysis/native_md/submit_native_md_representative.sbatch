#!/bin/bash
#SBATCH --job-name=nativeMDrep
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err
set -euo pipefail
cd /mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
subjects=(D0231 D0330 D0097 D0388 D0163)
subject=${subjects[$SLURM_ARRAY_TASK_ID]}
regions=(precentral superiorfrontal supramarginal superiortemporal middletemporal pericalcarine)
echo "[subject $((SLURM_ARRAY_TASK_ID+1))/5] $subject; regions=6; depths=3"
python -u -m native_surface_analysis.native_md.analyze_native_md +  --subject "$subject" --regions "${regions[@]}"
