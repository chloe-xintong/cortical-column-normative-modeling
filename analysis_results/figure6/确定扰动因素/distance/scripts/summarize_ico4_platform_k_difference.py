#!/usr/bin/env python3
"""Summarize same-seed K=30 minus K=15 platform differences."""

from pathlib import Path
import sys

import pandas as pd


REPO = Path(__file__).resolve().parents[5]
PILOT = REPO / "analysis_results/figure6/确定扰动因素/distance/local_similarity_stability"
ROSTER = REPO / "analysis_results/figure6/figure6_new/outputs/full/young_reference_roster.csv"


def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return frame.groupby(groups, observed=True, sort=True).agg(
        n_comparisons=("p_diff_30_minus_15", "size"),
        p_diff_mean=("p_diff_30_minus_15", "mean"),
        p_diff_median=("p_diff_30_minus_15", "median"),
        p_diff_q1=("p_diff_30_minus_15", lambda x: x.quantile(.25)),
        p_diff_q3=("p_diff_30_minus_15", lambda x: x.quantile(.75)),
        p_abs_diff_mean=("p_abs_diff", "mean"),
        p_abs_diff_median=("p_abs_diff", "median"),
        p_abs_diff_q1=("p_abs_diff", lambda x: x.quantile(.25)),
        p_abs_diff_q3=("p_abs_diff", lambda x: x.quantile(.75)),
        both_complete_fraction=("both_platform_complete", "mean"),
    ).reset_index()


def main() -> None:
    roster = pd.read_csv(ROSTER, dtype={"subject_id": str})
    parts = []
    for subject in roster.subject_id:
        features = pd.read_parquet(
            PILOT / "features/subjects" / f"{subject}.parquet",
            columns=["subject_id", "node_order", "hemisphere", "anchor_dk_region",
                     "distance_bin_index", "seed", "k", "p_real", "platform_complete"],
        )
        platform = features.loc[features.distance_bin_index.eq(10)].drop(columns="distance_bin_index")
        left = platform.loc[platform.k.eq(15)].drop(columns="k")
        right = platform.loc[platform.k.eq(30)].drop(columns="k")
        keys = ["subject_id", "node_order", "hemisphere", "anchor_dk_region", "seed"]
        joined = left.merge(right, on=keys, suffixes=("_k15", "_k30"), validate="one_to_one")
        joined["both_platform_complete"] = joined.platform_complete_k15 & joined.platform_complete_k30
        joined["p_diff_30_minus_15"] = joined.p_real_k30 - joined.p_real_k15
        joined["p_abs_diff"] = joined.p_diff_30_minus_15.abs()
        joined.loc[~joined.both_platform_complete, ["p_diff_30_minus_15", "p_abs_diff"]] = pd.NA
        parts.append(joined)
    detail = pd.concat(parts, ignore_index=True)
    out = PILOT / "stability"
    detail.to_parquet(out / "platform_k15_vs_k30_detail.parquet", index=False, compression="zstd")
    summarize(detail, []).to_csv(out / "platform_k15_vs_k30_summary.csv", index=False)
    summarize(detail, ["hemisphere", "anchor_dk_region"]).to_csv(
        out / "platform_k15_vs_k30_by_anchor_DK.csv", index=False)
    print(summarize(detail, []).to_string(index=False))
    print(f"output_dir={out.resolve()}")


if __name__ == "__main__":
    # pandas groupby([]) is undefined; use a constant only for the global table.
    original = summarize
    def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
        if groups:
            return original(frame, groups)
        return original(frame.assign(summary="all"), ["summary"]).drop(columns="summary")
    main()
