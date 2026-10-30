#!/usr/bin/env python3
"""Create deterministic shared age-stratified nested 5x5 CV folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-order", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def age_bins(ages: np.ndarray, n_splits: int) -> tuple[np.ndarray, int]:
    for n_bins in range(5, 1, -1):
        labels = np.asarray(pd.qcut(ages, q=n_bins, labels=False, duplicates="drop"))
        counts = pd.Series(labels).value_counts()
        if len(counts) >= 2 and int(counts.min()) >= n_splits:
            return labels.astype(int), int(len(counts))
    raise RuntimeError("Could not construct age bins with enough observations for stratification")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = pd.read_csv(args.subject_order, dtype={"subject_id": str})
    ids = subjects["subject_id"].to_numpy()
    ages = pd.to_numeric(subjects["age_at_scan"], errors="raise").to_numpy(dtype=float)
    if len(ids) < args.n_splits * args.n_splits:
        raise RuntimeError("Too few subjects for stable nested 5x5 CV")
    outer_labels, outer_bin_count = age_bins(ages, args.n_splits)
    outer_cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    outer_payload: dict[str, object] = {
        "seed": args.seed,
        "n_splits": args.n_splits,
        "stratification": "age quantile bins; bins reduced only if required for minimum fold size",
        "n_age_bins": outer_bin_count,
        "folds": [],
    }
    inner_payload: dict[str, object] = {"seed": args.seed, "n_splits": args.n_splits, "outer_folds": {}}
    summary_rows: list[dict[str, object]] = []
    seen_test: list[str] = []
    for outer_fold, (train_index, test_index) in enumerate(outer_cv.split(ids, outer_labels)):
        train_ids = ids[train_index].tolist()
        test_ids = ids[test_index].tolist()
        seen_test.extend(test_ids)
        outer_payload["folds"].append(
            {"outer_fold": outer_fold, "train_subject_ids": train_ids, "test_subject_ids": test_ids}
        )
        for split_name, indices in (("train", train_index), ("test", test_index)):
            summary_rows.append(
                {
                    "outer_fold": outer_fold,
                    "split": split_name,
                    "n": len(indices),
                    "age_mean": float(np.mean(ages[indices])),
                    "age_sd": float(np.std(ages[indices], ddof=1)),
                    "age_min": float(np.min(ages[indices])),
                    "age_max": float(np.max(ages[indices])),
                }
            )
        inner_ages = ages[train_index]
        inner_labels, inner_bin_count = age_bins(inner_ages, args.n_splits)
        inner_cv = StratifiedKFold(
            n_splits=args.n_splits, shuffle=True, random_state=args.seed + outer_fold + 1
        )
        inner_folds: list[dict[str, object]] = []
        for inner_fold, (inner_train_local, inner_validation_local) in enumerate(
            inner_cv.split(train_index, inner_labels)
        ):
            inner_folds.append(
                {
                    "inner_fold": inner_fold,
                    "train_subject_ids": ids[train_index[inner_train_local]].tolist(),
                    "validation_subject_ids": ids[train_index[inner_validation_local]].tolist(),
                }
            )
        inner_payload["outer_folds"][str(outer_fold)] = {
            "n_age_bins": inner_bin_count,
            "folds": inner_folds,
        }
    if len(seen_test) != len(ids) or set(seen_test) != set(ids) or len(set(seen_test)) != len(ids):
        raise RuntimeError("Outer test folds do not partition the cohort exactly once")
    (args.output_dir / "outer_folds.json").write_text(
        json.dumps(outer_payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "inner_folds_by_outer_fold.json").write_text(
        json.dumps(inner_payload, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "fold_age_summary.csv", index=False)
    print(json.dumps({"status": "complete", "n_subjects": len(ids), "n_outer_folds": args.n_splits}))


if __name__ == "__main__":
    main()
