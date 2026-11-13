#!/usr/bin/env python3
"""Freeze the 186-person regional Warped-M2 cohort without reassigning folds.

Description
-----------
Freeze the 186-person regional Warped-M2 cohort without reassigning folds.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--source-audit`` command-line input
- ``--source-folds`` command-line input
- ``--output-dir`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/prepare_regional_warped_m2_186.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EXCLUSIONS = {
    "D0271": "source_image_structural_abnormality",
    "D0285": "source_image_structural_abnormality",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-audit", type=Path, required=True)
    p.add_argument("--source-folds", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    for d in (out, out / "logs", out / "warped_m2" / "shards", out / "notebooks"):
        d.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(a.source_audit, dtype={"subject_id": str})
    if int(audit.included.sum()) != 188:
        raise RuntimeError("Source audit must contain exactly 188 included subjects")
    for sid, reason in EXCLUSIONS.items():
        hit = audit.subject_id.eq(sid)
        if hit.sum() != 1 or not bool(audit.loc[hit, "included"].iloc[0]):
            raise RuntimeError(f"{sid} is not a unique included source subject")
        audit.loc[hit, "included"] = False
        audit.loc[hit, "exclusion_reason"] = reason
    audit.to_csv(out / "cohort_audit.csv", index=False)
    included = audit.loc[audit.included].copy()
    if len(included) != 186:
        raise RuntimeError("Final regional cohort is not 186 subjects")

    source = json.loads(a.source_folds.read_text(encoding="utf-8"))
    retained = set(included.subject_id)
    folds, assignments = [], []
    source_assignment = {}
    for item in source["folds"]:
        fold = int(item["outer_fold"])
        for sid in item["test_subject_ids"]:
            source_assignment[str(sid)] = fold
        train = [str(s) for s in item["train_subject_ids"] if str(s) in retained]
        test = [str(s) for s in item["test_subject_ids"] if str(s) in retained]
        if set(train) & set(test) or set(train) | set(test) != retained:
            raise RuntimeError(f"Invalid train/test partition in fold {fold}")
        folds.append({"outer_fold": fold, "train_subject_ids": train, "test_subject_ids": test})
        assignments.extend({"subject_id": sid, "outer_fold": fold} for sid in test)
    assignment = pd.DataFrame(assignments)
    if len(assignment) != 186 or assignment.subject_id.nunique() != 186:
        raise RuntimeError("Filtered folds do not partition the final cohort exactly once")
    if any(source_assignment[sid] != fold for sid, fold in assignment.itertuples(index=False)):
        raise RuntimeError("A retained subject changed outer-fold assignment")
    write_json(out / "fold_assignments.json", {
        "seed": source.get("seed"), "n_splits": 5,
        "source": str(a.source_folds.resolve()), "source_sha256": sha256(a.source_folds),
        "assignment_policy": "Original 188-person folds retained; D0271 and D0285 removed without reassignment.",
        "excluded": EXCLUSIONS, "folds": folds,
    })
    assignment.sort_values(["outer_fold", "subject_id"]).to_csv(out / "outer_fold_assignments.csv", index=False)
    balance = assignment.merge(included[["subject_id", "age", "age_bin", "sex_code"]], on="subject_id", validate="one_to_one")
    balance.groupby(["outer_fold", "age_bin", "sex_code"], observed=False).size().rename("n").reset_index().to_csv(
        out / "fold_balance.csv", index=False
    )
    pd.DataFrame([
        {"cohort": "source_regional_pilot", "n": 188},
        {"cohort": "final_exclude_D0271_D0285", "n": 186},
        *({"cohort": "excluded", "n": 1, "subject_id": sid, "reason": reason}
          for sid, reason in EXCLUSIONS.items()),
    ]).to_csv(out / "cohort_summary.csv", index=False)
    write_json(out / "run_config.json", {
        "cohort_n": 186, "excluded": EXCLUSIONS, "n_regions": 68, "n_depths": 21,
        "anchor_depth_indices": [0, 4, 8, 12, 16, 20],
        "model": "Warped M2: MD ~ linear age + categorical sex_code + local FreeSurfer thickness",
        "pcntoolkit": "1.3.0", "warp": "WarpSinhArcsinh", "heteroskedastic": False,
        "model_selection": "frozen; no Gaussian or M0-M3 rerun",
    })
    print(json.dumps({"status": "prepared", "n_included": 186,
                      "fold_sizes": balance.groupby("outer_fold").size().to_dict()}))


if __name__ == "__main__":
    main()
