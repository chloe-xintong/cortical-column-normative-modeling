#!/usr/bin/env python3
"""Create the frozen 187-person Figure 6 cohort without reassigning folds.

Description
-----------
Create the frozen 187-person Figure 6 cohort without reassigning folds.

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
python analysis_results/figure6/scripts/prepare_final_cohort.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EXCLUDED_SUBJECT = "D0271"
EXCLUSION_REASON = "source_image_structural_abnormality"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--source-folds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir
    for directory in (out, out / "logs", out / "shards", out / "warped_m2" / "shards", out / "figures"):
        directory.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(args.source_audit, dtype={"subject_id": str})
    if len(audit.loc[audit.included]) != 188:
        raise RuntimeError("Source regional pilot must contain exactly 188 included subjects")
    match = audit.subject_id.eq(EXCLUDED_SUBJECT)
    if match.sum() != 1 or not bool(audit.loc[match, "included"].iloc[0]):
        raise RuntimeError(f"{EXCLUDED_SUBJECT} is not a unique included source-pilot subject")
    audit.loc[match, "included"] = False
    audit.loc[match, "exclusion_reason"] = EXCLUSION_REASON
    audit.to_csv(out / "cohort_audit.csv", index=False)
    included = audit.loc[audit.included].copy()
    if len(included) != 187 or EXCLUDED_SUBJECT in set(included.subject_id):
        raise RuntimeError("Final cohort is not exactly 187 subjects after exclusion")

    source = json.loads(args.source_folds.read_text(encoding="utf-8"))
    folds: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    source_assignment: dict[str, int] = {}
    for item in source["folds"]:
        fold = int(item["outer_fold"])
        old_test = [str(x) for x in item["test_subject_ids"]]
        for subject_id in old_test:
            source_assignment[subject_id] = fold
        train = [str(x) for x in item["train_subject_ids"] if str(x) != EXCLUDED_SUBJECT]
        test = [str(x) for x in old_test if str(x) != EXCLUDED_SUBJECT]
        folds.append({"outer_fold": fold, "train_subject_ids": train, "test_subject_ids": test})
        assignments.extend({"subject_id": subject_id, "outer_fold": fold} for subject_id in test)
    assignment = pd.DataFrame(assignments)
    retained = set(included.subject_id)
    if len(assignment) != 187 or set(assignment.subject_id) != retained or assignment.subject_id.duplicated().any():
        raise RuntimeError("Filtered folds do not partition all retained subjects exactly once")
    if any(source_assignment[sid] != fold for sid, fold in assignment[["subject_id", "outer_fold"]].itertuples(index=False)):
        raise RuntimeError("A retained subject changed outer-fold assignment")
    for item in folds:
        train, test = set(item["train_subject_ids"]), set(item["test_subject_ids"])
        if train & test or train | test != retained:
            raise RuntimeError(f"Invalid train/test partition in fold {item['outer_fold']}")

    payload = {
        "seed": source.get("seed"),
        "n_splits": 5,
        "source": str(args.source_folds.resolve()),
        "source_sha256": sha256(args.source_folds),
        "assignment_policy": "Original 188-person outer-fold assignments retained; D0271 removed only, with no reassignment.",
        "excluded_subject_id": EXCLUDED_SUBJECT,
        "exclusion_reason": EXCLUSION_REASON,
        "folds": folds,
    }
    write_json(out / "fold_assignments.json", payload)

    balance = assignment.merge(included[["subject_id", "age", "age_bin", "sex_code"]], on="subject_id", validate="one_to_one")
    balance.groupby(["outer_fold", "age_bin", "sex_code"], observed=False).size().rename("n").reset_index().to_csv(
        out / "fold_balance.csv", index=False
    )
    fold_summary = balance.groupby("outer_fold", observed=True).agg(
        n_test=("subject_id", "size"), age_min=("age", "min"), age_max=("age", "max"),
        age_mean=("age", "mean"), age_sd=("age", "std")
    ).reset_index()
    sex = balance.groupby(["outer_fold", "sex_code"], observed=True).size().unstack(fill_value=0)
    sex.columns = [f"n_sex_code_{int(x)}" for x in sex.columns]
    fold_summary.merge(sex.reset_index(), on="outer_fold", validate="one_to_one").to_csv(
        out / "fold_distribution_validation.csv", index=False
    )
    assignment.sort_values(["outer_fold", "subject_id"]).to_csv(out / "outer_fold_assignments.csv", index=False)
    pd.DataFrame([
        {"cohort": "source_regional_pilot", "n": 188},
        {"cohort": "final_exclude_D0271", "n": 187},
        {"cohort": "excluded", "n": 1, "subject_id": EXCLUDED_SUBJECT, "reason": EXCLUSION_REASON},
    ]).to_csv(out / "cohort_summary.csv", index=False)
    print(json.dumps({"status": "prepared", "n_included": 187,
                      "fold_test_sizes": balance.groupby("outer_fold").size().to_dict()}))


if __name__ == "__main__":
    main()
