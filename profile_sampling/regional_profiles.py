#!/usr/bin/env python

# Purpose:
# Compute regional mean cortical profiles and 95% confidence intervals.
#
# Usage:
# python profile_analysis/regional_profiles.py \
#   --subject-id <ID> \
#   --profile-dir <per-trajectory profile directory> \
#   --metric-name <md/fa/ad/rd> \
#   --output-dir <output directory>
#
# Input:
# One CSV per hemisphere and region.
# Rows are trajectories and columns are cortical depth samples.
#
# Output:
# Regional mean profiles, confidence intervals, and summary CSV files.

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import norm


REGION_LIST = [
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "paracentral",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "frontalpole",
    "temporalpole",
    "transversetemporal",
    "insula",
]


def load_profile(profile_path):
    profile_path = Path(profile_path)

    if not profile_path.is_file():
        raise FileNotFoundError(
            f"Profile file not found: {profile_path}"
        )

    values = np.loadtxt(
        profile_path,
        delimiter=",",
    )

    if values.ndim == 1:
        values = values.reshape(
            1,
            -1,
        )

    return values


def compute_mean_and_ci(
    values,
    ci_level=0.95,
):
    if values.size == 0:
        raise ValueError(
            "Empty profile data."
        )

    n_trajectories = values.shape[0]

    mean = values.mean(
        axis=0
    )

    if n_trajectories == 1:
        ci_low = mean.copy()
        ci_high = mean.copy()

        return (
            mean,
            ci_low,
            ci_high,
            n_trajectories,
        )

    z = norm.ppf(
        0.5 + ci_level / 2.0
    )

    std = values.std(
        axis=0,
        ddof=1,
    )

    sem = std / np.sqrt(
        n_trajectories
    )

    ci_low = mean - z * sem
    ci_high = mean + z * sem

    return (
        mean,
        ci_low,
        ci_high,
        n_trajectories,
    )


def save_mean_profile(
    mean,
    output_path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savetxt(
        output_path,
        mean.reshape(-1, 1),
        delimiter=",",
    )


def process_region(
    subject_id,
    hemisphere,
    region,
    metric_name,
    profile_dir,
    output_dir,
):
    profile_path = (
        Path(profile_dir)
        / (
            f"{subject_id}_"
            f"{hemisphere}_"
            f"{region}_"
            f"{metric_name}.csv"
        )
    )

    if not profile_path.is_file():
        print(
            f"Missing profile: {profile_path}"
        )
        return None

    values = load_profile(
        profile_path
    )

    (
        mean,
        ci_low,
        ci_high,
        n_trajectories,
    ) = compute_mean_and_ci(
        values,
        ci_level=0.95,
    )

    mean_dir = (
        Path(output_dir)
        / "region_mean"
    )

    mean_path = (
        mean_dir
        / (
            f"{subject_id}_"
            f"{hemisphere}_"
            f"{region}_"
            f"{metric_name}_mean.csv"
        )
    )

    save_mean_profile(
        mean,
        mean_path,
    )

    print(
        f"{hemisphere}_{region}: "
        f"{n_trajectories} trajectories × "
        f"{values.shape[1]} depths"
    )

    print(
        f"Mean profile saved: {mean_path}"
    )

    rows = []

    for depth_index in range(
        values.shape[1]
    ):
        rows.append(
            [
                hemisphere,
                region,
                depth_index,
                float(mean[depth_index]),
                float(ci_low[depth_index]),
                float(ci_high[depth_index]),
                int(n_trajectories),
            ]
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute regional mean cortical profiles "
            "and 95% confidence intervals."
        )
    )

    parser.add_argument(
        "--subject-id",
        required=True,
    )

    parser.add_argument(
        "--profile-dir",
        required=True,
        help=(
            "Directory containing per-trajectory "
            "profile CSV files."
        ),
    )

    parser.add_argument(
        "--metric-name",
        required=True,
        help="Metric name such as md, fa, ad, or rd.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    for hemisphere in [
        "lh",
        "rh",
    ]:
        for region in REGION_LIST:
            rows = process_region(
                subject_id=args.subject_id,
                hemisphere=hemisphere,
                region=region,
                metric_name=args.metric_name,
                profile_dir=args.profile_dir,
                output_dir=output_dir,
            )

            if rows is not None:
                all_rows.extend(
                    rows
                )

    qa_path = (
        output_dir
        / (
            f"{args.subject_id}_"
            f"profiles_QA_"
            f"{args.metric_name}.csv"
        )
    )

    with open(
        qa_path,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "hemi",
                "region",
                "depth_index",
                "mean",
                "ci_lower",
                "ci_upper",
                "n_trajectories",
            ]
        )

        writer.writerows(
            all_rows
        )

    print()
    print(
        f"QA profile summary saved: {qa_path}"
    )

    print()
    print(
        "Regional profile aggregation complete."
    )


if __name__ == "__main__":
    main()