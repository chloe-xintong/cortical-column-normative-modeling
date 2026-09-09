#!/bin/bash
#SBATCH --job-name=ADRC
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.out
#SBATCH --error=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/logs/%x_%A_%a.err

set -euo pipefail

PIPELINE_DIR="/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis"
MANIFEST="${PIPELINE_DIR}/adrc_subjects.csv"
MASTER_SCRIPT="${PIPELINE_DIR}/ADRC_MUSE_master_script.sh"
MONITOR_SCRIPT="${PIPELINE_DIR}/monitor_job_resources.sh"
LOG_DIR="${PIPELINE_DIR}/logs"

# These variables only exist inside a Slurm array job
: "${SLURM_ARRAY_TASK_ID:?Submit this script using sbatch --array}"
: "${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is not set}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
: "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is not set}"

cd "$PIPELINE_DIR"
mkdir -p "$LOG_DIR"

# Verify required files
if [[ ! -s "$MANIFEST" ]]; then
    echo "ERROR: Subject manifest is missing or empty: $MANIFEST"
    exit 1
fi

if [[ ! -f "$MASTER_SCRIPT" ]]; then
    echo "ERROR: Master pipeline is missing: $MASTER_SCRIPT"
    exit 1
fi

if [[ ! -x "$MONITOR_SCRIPT" ]]; then
    echo "ERROR: Monitor script is missing or not executable: $MONITOR_SCRIPT"
    exit 1
fi

# CSV row 1 is the header; array task 0 corresponds to CSV row 2
CSV_ROW=$((SLURM_ARRAY_TASK_ID + 2))

SUBJECT=$(
    awk -F',' -v row="$CSV_ROW" '
        NR == row {
            gsub(/\r/, "", $2)
            print $2
        }
    ' "$MANIFEST"
)

if [[ -z "$SUBJECT" ]]; then
    echo "ERROR: No subject found for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

if [[ ! "$SUBJECT" =~ ^D[0-9]+$ ]]; then
    echo "ERROR: Invalid subject ID read from manifest: $SUBJECT"
    exit 1
fi

echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Array job:    $SLURM_ARRAY_JOB_ID"
echo "Array task:   $SLURM_ARRAY_TASK_ID"
echo "CSV row:      $CSV_ROW"
echo "Subject:      $SUBJECT"
echo "Node:         $(hostname)"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Started:      $(date)"
echo "========================================"

# Restrict threaded programs to the CPUs allocated by Slurm
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$SLURM_CPUS_PER_TASK"
export MRTRIX_NTHREADS="$SLURM_CPUS_PER_TASK"

# Prevent hidden NumPy/BLAS thread pools
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RESOURCE_LOG="${LOG_DIR}/live_resources_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SUBJECT}.log"
TIME_LOG="${LOG_DIR}/final_resources_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SUBJECT}.txt"

# Run the pipeline in the background so the monitor can watch it
/usr/bin/time -v \
    -o "$TIME_LOG" \
    bash "$MASTER_SCRIPT" "$SUBJECT" &

PIPELINE_PID=$!

# Start resource monitoring every 60 seconds
"$MONITOR_SCRIPT" \
    "$PIPELINE_PID" \
    "$RESOURCE_LOG" \
    60 &

MONITOR_PID=$!

cleanup_monitor() {
    if kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi

    wait "$MONITOR_PID" 2>/dev/null || true
}

trap cleanup_monitor EXIT

# Preserve the pipeline's exit status
if wait "$PIPELINE_PID"; then
    PIPELINE_STATUS=0
else
    PIPELINE_STATUS=$?
fi

cleanup_monitor
trap - EXIT

echo "========================================"
echo "Subject:      $SUBJECT"
echo "Finished:     $(date)"
echo "Exit status:  $PIPELINE_STATUS"
echo "Resource log: $RESOURCE_LOG"
echo "Summary log:  $TIME_LOG"
echo "========================================"

exit "$PIPELINE_STATUS"