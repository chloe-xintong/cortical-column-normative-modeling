#!/usr/bin/env python3
"""Combine per-subject O3 QC caches and report cohort completion."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robust-z-threshold", type=float, default=3.5)
    return parser.parse_args()


def robust_z(values: pd.Series) -> pd.Series:
    median = values.median()
    mad = np.median(np.abs(values - median))
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=values.index)
    return 0.6745 * (values - median) / mad


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    statuses = []
    summaries = []
    depths = []
    registrations = []
    runtimes = []
    for row in manifest.itertuples(index=False):
        subject_id = row.subject_id
        subject_dir = args.results_root / "subjects" / subject_id
        completion = subject_dir / "completion.json"
        failure = subject_dir / "failure.json"
        if completion.is_file():
            state = read_json(completion)
            status = "complete"
            reason = ""
            runtimes.append(state)
            summaries.append(
                pd.read_csv(subject_dir / "option3_mapping_qc_subject_hemi.csv")
            )
            depths.append(pd.read_csv(subject_dir / "option3_invalid_md_by_depth.csv"))
            registrations.append(pd.read_csv(subject_dir / "registration_cost.csv"))
        elif failure.is_file():
            state = read_json(failure)
            status = "failed"
            reason = str(state.get("error", ""))
            runtimes.append(state)
        elif subject_dir.exists():
            status = "incomplete"
            reason = (
                "Subject output directory exists without completion or failure JSON"
            )
        else:
            status = "not_submitted"
            reason = "No subject output directory"
        statuses.append(
            {
                "subject_id": subject_id,
                "status": status,
                "reason": reason,
            }
        )

    status_table = pd.DataFrame(statuses)
    status_table.to_csv(args.output_dir / "full_cohort_status.csv", index=False)
    status_table.loc[status_table["status"].eq("failed")].to_csv(
        args.output_dir / "full_cohort_failures.csv",
        index=False,
    )
    status_table.loc[~status_table["status"].eq("complete")].to_csv(
        args.output_dir / "subjects_requiring_rerun.csv",
        index=False,
    )
    pd.DataFrame(runtimes).to_csv(
        args.output_dir / "runtime_and_memory_summary.csv",
        index=False,
    )

    if summaries:
        summary = pd.concat(summaries, ignore_index=True)
        summary.to_csv(
            args.output_dir / "option3_mapping_qc_subject_hemi.csv",
            index=False,
        )
    if depths:
        pd.concat(depths, ignore_index=True).to_csv(
            args.output_dir / "option3_invalid_md_by_depth.csv",
            index=False,
        )
    if registrations:
        registration = pd.concat(registrations, ignore_index=True)
        registration["robust_z"] = robust_z(registration["bbregister_mincost"])
        registration["automatic_cost_outlier"] = (
            registration["robust_z"].abs().gt(args.robust_z_threshold)
        )
        registration["visual_qc_status"] = "not_reviewed"
        registration["final_exclusion_status"] = "not_excluded"
        registration.to_csv(
            args.output_dir / "registration_cost_by_subject.csv",
            index=False,
        )
        review = registration.loc[registration["automatic_cost_outlier"]].copy()
        review.to_csv(
            args.output_dir / "registration_visual_review_queue.csv",
            index=False,
        )
        command_path = args.output_dir / "freeview_registration_review_commands.sh"
        commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
        for row in review.itertuples(index=False):
            freesurfer_dir = Path(row.freesurfer_subject_dir)
            command = [
                "freeview",
                "-v",
                f"{freesurfer_dir / 'mri' / 'orig.mgz'}:visible=0",
                f"{row.moving_image_path}:reg={row.transform_path}",
                "-f",
                str(freesurfer_dir / "surf" / "lh.white"),
                str(freesurfer_dir / "surf" / "rh.white"),
            ]
            commands.append(f"# {row.subject_id}: automatic bbregister-cost review")
            commands.append(" ".join(shlex.quote(item) for item in command))
            commands.append("")
        command_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
        command_path.chmod(0o755)

    counts = status_table["status"].value_counts()
    report = pd.DataFrame(
        [
            {
                "n_requested": len(manifest),
                "n_input_complete": int(manifest["input_complete"].sum()),
                "n_submitted_or_started": int(
                    status_table["status"].ne("not_submitted").sum()
                ),
                "n_successful": int(counts.get("complete", 0)),
                "n_failed": int(counts.get("failed", 0)),
                "n_incomplete": int(counts.get("incomplete", 0)),
                "n_not_submitted": int(counts.get("not_submitted", 0)),
                "n_excluded_by_qc": 0,
            }
        ]
    )
    report.to_csv(args.output_dir / "full_cohort_counts.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
