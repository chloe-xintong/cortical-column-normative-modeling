#!/usr/bin/env python3
"""Create balanced subject manifests for sharded fsaverage mapping QC."""
from pathlib import Path
import argparse

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=4)
    args = ap.parse_args()
    subjects = [p.name for p in sorted(args.mapping_root.glob("D*")) if p.is_dir() and len(list(p.glob("[lr]h/regions/*_native_to_fsaverage.csv.gz"))) == 68]
    if not subjects:
        raise SystemExit("No complete mapping subjects found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for shard in range(args.shards):
        values = subjects[shard::args.shards]
        (args.output_dir / f"fsaverage_mapping_qc_shard_{shard}.txt").write_text("\n".join(values) + "\n")
        print(f"shard {shard}: {len(values)} subjects")

if __name__ == "__main__":
    main()
