#!/usr/bin/env bash
# Log resource use from inside a Slurm job while a pipeline PID is alive.
# Usage: monitor_job_resources.sh PIPELINE_PID LOG_FILE [INTERVAL_SECONDS]

set -u

if (( $# < 2 || $# > 3 )); then
    echo "Usage: $0 PIPELINE_PID LOG_FILE [INTERVAL_SECONDS]" >&2
    exit 2
fi

WATCH_PID="$1"
LOG_FILE="$2"
INTERVAL="${3:-60}"

if [[ ! "$WATCH_PID" =~ ^[0-9]+$ ]]; then
    echo "PIPELINE_PID must be numeric: $WATCH_PID" >&2
    exit 2
fi

if [[ ! "$INTERVAL" =~ ^[0-9]+$ ]] || (( INTERVAL < 5 )); then
    echo "INTERVAL_SECONDS must be an integer of at least 5" >&2
    exit 2
fi

mkdir -p "$(dirname "$LOG_FILE")"

human_bytes() {
    local value="$1"
    if [[ "$value" == "max" ]]; then
        printf 'max'
    elif command -v numfmt >/dev/null 2>&1; then
        numfmt --to=iec --suffix=B "$value" 2>/dev/null || printf '%s bytes' "$value"
    else
        printf '%s bytes' "$value"
    fi
}

# On cgroup v2, this is the Slurm step cgroup containing this monitor and the
# pipeline when both are launched directly by the same batch script.
CGROUP_DIR=""
if [[ -r /proc/self/cgroup ]]; then
    CGROUP_REL="$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)"
    if [[ -n "$CGROUP_REL" && -d "/sys/fs/cgroup${CGROUP_REL}" ]]; then
        CGROUP_DIR="/sys/fs/cgroup${CGROUP_REL}"
    fi
fi

{
    echo "Resource monitor started: $(date --iso-8601=seconds)"
    echo "Host: $(hostname)"
    echo "Watch PID: $WATCH_PID"
    echo "Interval: ${INTERVAL}s"
    echo "Slurm job: ${SLURM_JOB_ID:-not-set}"
    echo "Slurm array task: ${SLURM_ARRAY_TASK_ID:-not-set}"
    echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-not-set}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
    echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-not-set}"
    echo "Cgroup: ${CGROUP_DIR:-not-detected}"
} > "$LOG_FILE"

while kill -0 "$WATCH_PID" 2>/dev/null; do
    {
        echo
        echo "===== $(date --iso-8601=seconds) ====="

        if [[ -n "$CGROUP_DIR" ]]; then
            echo "--- Slurm cgroup ---"
            for name in memory.current memory.peak memory.swap.current memory.max; do
                path="$CGROUP_DIR/$name"
                if [[ -r "$path" ]]; then
                    value="$(<"$path")"
                    printf '%-22s %s\n' "$name" "$(human_bytes "$value")"
                fi
            done
            if [[ -r "$CGROUP_DIR/cpu.stat" ]]; then
                echo "cpu.stat:"
                sed 's/^/  /' "$CGROUP_DIR/cpu.stat"
            fi
        fi

        echo "--- Node memory ---"
        free -h 2>&1

        echo "--- vmstat (r b swpd free buff cache si so bi bo in cs us sy id wa st) ---"
        vmstat 1 2 2>&1 | tail -1

        echo "--- Largest processes owned by $USER on this node ---"
        ps -u "$USER" \
            -o pid,ppid,%cpu,%mem,rss,vsz,stat,etime,comm \
            --sort=-rss 2>&1 | head -20

        if command -v pidstat >/dev/null 2>&1; then
            echo "--- Watched pipeline process ---"
            pidstat -h -u -r -p "$WATCH_PID" 1 1 2>&1 || true
        fi

        if [[ -n "${SLURM_JOB_GPUS:-}" || \
              ( -n "${CUDA_VISIBLE_DEVICES:-}" && \
                "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ) ]]; then
            echo "--- Allocated GPU ---"
            nvidia-smi \
                --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
                --format=csv,noheader 2>&1 || true
        else
            echo "--- GPU ---"
            echo "No GPU allocation detected for this job."
        fi
    } >> "$LOG_FILE" 2>&1

    sleep "$INTERVAL"
done

{
    echo
    echo "Resource monitor stopped: $(date --iso-8601=seconds)"
    echo "Watched PID $WATCH_PID is no longer running."
} >> "$LOG_FILE"
