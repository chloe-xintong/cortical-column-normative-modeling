#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

root = Path("/mnt/newStor/paros/paros_WORK/xintong/results/fsaverage6_mapping_qc_shards")
out = Path("/mnt/newStor/paros/paros_WORK/xintong/results/fsaverage6_mapping_qc")
out.mkdir(parents=True, exist_ok=True)
def merge(name):
    paths = sorted(root.glob(f"shard_*/{name}"))
    if not paths: return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
for name in ("fsaverage_subject_region_mapping_qc.csv", "fsaverage_mapping_failures.csv", "fsaverage_mapping_skipped_subjects.csv"):
    merge(name).to_csv(out / name, index=False)
summary = merge("fsaverage_subject_region_mapping_qc.csv")
print(f"merged summaries={len(summary)}")
