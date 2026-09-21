#!/usr/bin/env python3
"""Merge completed spatial-QC v2 shards without requiring every shard to succeed."""
import argparse
from pathlib import Path
import pandas as pd

NAMES = ["spatial_similarity_group_summary_v2.csv", "spatial_similarity_distance_bins_v2.csv", "spatial_similarity_group_status_v2.csv"]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shards-root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--expected-shards',type=int,required=True)
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    missing=[]
    for name in NAMES:
        frames=[]
        for index in range(args.expected_shards):
            path=args.shards_root/f'shard_{index}'/name
            if path.is_file() and path.stat().st_size:
                frames.append(pd.read_csv(path))
            else:
                missing.append({'shard':index,'file':name,'reason':'missing_or_empty'})
        if frames:
            pd.concat(frames,ignore_index=True).to_csv(args.output_dir/name,index=False)
    pd.DataFrame(missing,columns=['shard','file','reason']).to_csv(args.output_dir/'spatial_similarity_v2_missing_shards.csv',index=False)
    print(f'Merged {args.expected_shards} expected shards; missing files={len(missing)}')

if __name__=='__main__': main()
