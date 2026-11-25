#!/usr/bin/env bash
set -euo pipefail
repo=/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis
cd "$repo"
code=analysis_results/figure6_new
python_bin="$repo/.venv/figure6-pcntoolkit-1.3.0/bin/python"
root="$(mktemp -d "$repo/$code/outputs/warped_delta_XXXXXXXX")"
mkdir -p "$root"/{logs,shards,smoke/shards}
worker="$repo/$code/warped_delta.sbatch"
script="$repo/$code/warped_delta.py"
test -x "$python_bin"
submit() {
    local result
    result=$(sbatch --parsable --output="$root/logs/%x_%A_%a.log" "$@")
    printf '%s\n' "${result%%;*}"
}
smoke=$(submit --job-name=delta_smoke --time=00:30:00 "$worker" "$python_bin" "$script" smoke "$root")
array=$(submit --job-name=delta_warp --time=01:00:00 --array=0-937%100 --dependency="afterok:$smoke" "$worker" "$python_bin" "$script" shard "$root")
merge=$(submit --job-name=delta_merge --time=01:00:00 --dependency="afterok:$array" "$worker" "$python_bin" "$script" merge "$root")
second=$(submit --job-name=delta_age --time=04:00:00 --mem=8G --dependency="afterok:$merge" "$worker" "$python_bin" "$script" second "$root")
printf 'Run: %s\nSmoke: %s\nArray: %s\nMerge: %s\nStage 2: %s\n' "$root" "$smoke" "$array" "$merge" "$second"
