#!/usr/bin/env python3
"""Summarize K=30/100/300 distant-platform sampling sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[5]
ROOT = REPO / "analysis_results/figure6/确定扰动因素/distance"
PILOT = ROOT / "platform_sampling_sensitivity"
SOURCE = ROOT / "local_similarity_stability"
K_VALUES = (30, 100, 300)


def q25(x: pd.Series) -> float: return float(x.astype(float).quantile(.25))
def q75(x: pd.Series) -> float: return float(x.astype(float).quantile(.75))


def summary(frame: pd.DataFrame, groups: list[str], columns: list[str]) -> pd.DataFrame:
    work = frame if groups else frame.assign(_all="all")
    group = groups or ["_all"]
    spec = {}
    for col in columns:
        spec.update({f"{col}_mean": (col, "mean"), f"{col}_median": (col, "median"),
                     f"{col}_q1": (col, q25), f"{col}_q3": (col, q75)})
    spec["n_rows"] = (columns[0], "size")
    result = work.groupby(group, observed=True, sort=True).agg(**spec).reset_index()
    return result.drop(columns="_all", errors="ignore")


def main() -> None:
    roster = pd.read_csv(SOURCE / "young_cn37_roster.csv", dtype={"subject_id": str})
    platform_parts, local_parts = [], []
    for subject in roster.subject_id:
        base = PILOT / "features/subjects"
        platform_parts.append(pd.read_parquet(base / f"{subject}_platform.parquet"))
        local_parts.append(pd.read_parquet(base / f"{subject}_local_L.parquet"))
    platform = pd.concat(platform_parts, ignore_index=True)
    local = pd.concat(local_parts, ignore_index=True)

    # Use the exact intersection complete for every seed and all three K values.
    common = platform.groupby(["subject_id", "node_order"], observed=True).agg(
        common_complete_all_k_seed=("platform_complete", "all"),
        any_exhausted_any_k_seed=("any_candidate_exhausted", "any"),
    ).reset_index()
    platform = platform.merge(common, on=["subject_id", "node_order"], validate="many_to_one")
    local = local.merge(common, on=["subject_id", "node_order"], validate="many_to_one")

    keys = ["subject_id", "node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region", "platform_k"]
    seed_stability = platform.groupby(keys, observed=True, sort=False).agg(
        p_seed_mean=("p_real", "mean"), p_seed_sd=("p_real", "std"),
        p_seed_range=("p_real", lambda x: float(x.max() - x.min())),
        n_complete_seeds=("platform_complete", "sum"),
        minimum_valid_pairs=("minimum_valid_pairs", "median"),
        median_valid_pairs=("median_valid_pairs", "median"),
        mean_exhausted_bins=("n_exhausted_bins", "mean"),
        any_candidate_exhausted=("any_candidate_exhausted", "any"),
        common_complete_all_k_seed=("common_complete_all_k_seed", "all"),
    ).reset_index()
    seed_stability["all_seeds_complete"] = seed_stability.n_complete_seeds.eq(5)

    # Same-seed position changes between successive and extreme K values.
    changes = []
    change_keys = ["subject_id", "node_order", "hemisphere", "anchor_dk_label",
                   "anchor_dk_region", "seed", "common_complete_all_k_seed"]
    for low, high in ((30, 100), (100, 300), (30, 300)):
        left = platform.loc[platform.platform_k.eq(low), change_keys + ["p_real"]]
        right = platform.loc[platform.platform_k.eq(high), change_keys + ["p_real"]]
        joined = left.merge(right, on=change_keys, suffixes=("_low", "_high"), validate="one_to_one")
        joined["comparison"] = f"K{high}-K{low}"
        joined["p_diff"] = joined.p_real_high - joined.p_real_low
        joined["p_abs_diff"] = joined.p_diff.abs()
        joined.loc[~joined.common_complete_all_k_seed, ["p_diff", "p_abs_diff"]] = np.nan
        changes.append(joined)
    changes = pd.concat(changes, ignore_index=True)

    # Platform-only propagation to fixed local K30/seed0 r_real.
    local_seed = local.groupby([
        "subject_id", "node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region",
        "distance_bin_index", "distance_bin", "platform_k",
    ], observed=True, sort=False).agg(
        L_platform_seed_mean=("L_platform_only", "mean"),
        L_platform_seed_sd=("L_platform_only", "std"),
        L_platform_seed_range=("L_platform_only", lambda x: float(x.max() - x.min())),
        n_valid_platform_seeds=("L_platform_only", "count"),
        common_complete_all_k_seed=("common_complete_all_k_seed", "all"),
    ).reset_index()

    # Contrast isolated platform variation with the existing K30 analysis where local and platform vary together.
    old_parts = []
    for subject in roster.subject_id:
        old = pd.read_parquet(SOURCE / "stability/subjects" / f"{subject}_seed_stability.parquet")
        old_parts.append(old.loc[old.k.eq(30) & old.distance_bin_index.lt(5), [
            "subject_id", "node_order", "distance_bin_index", "L_seed_sd", "r_seed_sd",
        ]])
    old = pd.concat(old_parts, ignore_index=True).rename(columns={
        "L_seed_sd": "L_both_local_and_platform_seed_sd",
        "r_seed_sd": "r_local_seed_sd",
    })
    comparison = local_seed.loc[local_seed.platform_k.eq(30)].merge(
        old, on=["subject_id", "node_order", "distance_bin_index"], validate="one_to_one")
    comparison["platform_fraction_of_both_sd"] = np.where(
        comparison.L_both_local_and_platform_seed_sd > 1e-8,
        comparison.L_platform_seed_sd / comparison.L_both_local_and_platform_seed_sd,
        np.nan,
    )
    comparison["variance_fraction_platform"] = np.where(
        comparison.L_both_local_and_platform_seed_sd > 1e-8,
        comparison.L_platform_seed_sd.pow(2) / comparison.L_both_local_and_platform_seed_sd.pow(2),
        np.nan,
    )

    common_only = seed_stability.loc[seed_stability.common_complete_all_k_seed]
    platform_overall = summary(common_only, ["platform_k"], [
        "p_seed_sd", "p_seed_range", "minimum_valid_pairs", "median_valid_pairs",
        "mean_exhausted_bins", "any_candidate_exhausted",
    ])
    platform_dk = summary(common_only, ["platform_k", "hemisphere", "anchor_dk_region"], [
        "p_seed_sd", "p_seed_range", "minimum_valid_pairs", "mean_exhausted_bins",
    ])
    changes_overall = summary(changes, ["comparison"], ["p_diff", "p_abs_diff"])
    changes_dk = summary(changes, ["comparison", "hemisphere", "anchor_dk_region"], [
        "p_diff", "p_abs_diff",
    ])
    local_common = local_seed.loc[local_seed.common_complete_all_k_seed]
    local_by_bin = summary(local_common, ["platform_k", "distance_bin_index", "distance_bin"], [
        "L_platform_seed_sd", "L_platform_seed_range",
    ])
    local_dk = summary(local_common, ["platform_k", "hemisphere", "anchor_dk_region",
                                      "distance_bin_index", "distance_bin"], [
        "L_platform_seed_sd", "L_platform_seed_range",
    ])
    comparison_summary = summary(comparison.loc[comparison.common_complete_all_k_seed],
        ["distance_bin_index", "distance_bin"], [
            "L_platform_seed_sd", "L_both_local_and_platform_seed_sd", "r_local_seed_sd",
            "platform_fraction_of_both_sd", "variance_fraction_platform",
        ])
    comparison_dk = summary(comparison.loc[comparison.common_complete_all_k_seed],
        ["hemisphere", "anchor_dk_region", "distance_bin_index", "distance_bin"], [
            "L_platform_seed_sd", "L_both_local_and_platform_seed_sd",
            "platform_fraction_of_both_sd",
        ])
    coverage = platform.groupby("platform_k", observed=True).agg(
        n_subject_anchor_seed=("platform_complete", "size"),
        platform_complete_fraction=("platform_complete", "mean"),
        candidate_exhausted_fraction=("any_candidate_exhausted", "mean"),
        median_minimum_valid_pairs=("minimum_valid_pairs", "median"),
        median_median_valid_pairs=("median_valid_pairs", "median"),
    ).reset_index()
    coverage["common_complete_all_k_seed_fraction"] = common.common_complete_all_k_seed.mean()

    out = PILOT / "stability"
    out.mkdir(parents=True, exist_ok=True)
    seed_stability.to_parquet(out / "platform_seed_stability_detail.parquet", index=False, compression="zstd")
    changes.to_parquet(out / "platform_k_position_changes.parquet", index=False, compression="zstd")
    local_seed.to_parquet(out / "platform_only_local_L_stability_detail.parquet", index=False, compression="zstd")
    comparison.to_parquet(out / "platform_vs_combined_sampling_detail.parquet", index=False, compression="zstd")
    platform_overall.to_csv(out / "platform_stability_overall.csv", index=False)
    platform_dk.to_csv(out / "platform_stability_by_anchor_DK.csv", index=False)
    changes_overall.to_csv(out / "platform_k_changes_overall.csv", index=False)
    changes_dk.to_csv(out / "platform_k_changes_by_anchor_DK.csv", index=False)
    local_by_bin.to_csv(out / "platform_only_L_stability_by_local_bin.csv", index=False)
    local_dk.to_csv(out / "platform_only_L_stability_by_anchor_DK.csv", index=False)
    comparison_summary.to_csv(out / "platform_vs_combined_sampling_by_local_bin.csv", index=False)
    comparison_dk.to_csv(out / "platform_vs_combined_sampling_by_anchor_DK.csv", index=False)
    coverage.to_csv(out / "coverage_summary.csv", index=False)

    figure_dir = PILOT / "figures"; figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    x_position = np.arange(len(platform_overall))
    axes[0].plot(x_position, platform_overall.p_seed_sd_median, marker="o")
    axes[0].set_xticks(x_position, platform_overall.platform_k.astype(str))
    axes[0].set(xlabel="Platform K", ylabel="Median seed SD of p")
    axes[1].bar(changes_overall.comparison, changes_overall.p_abs_diff_median)
    axes[1].set(xlabel="Platform K comparison", ylabel="Median position |Δp|")
    for k in K_VALUES:
        q = local_by_bin.loc[local_by_bin.platform_k.eq(k)]
        axes[2].plot(q.distance_bin_index * 2 + 1, q.L_platform_seed_sd_median, marker="o", label=f"K={k}")
    axes[2].set(xlabel="Local distance (mm)", ylabel="Median platform-only seed SD of L")
    axes[2].legend(frameon=False)
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "platform_sampling_sensitivity.png", dpi=240)
    plt.close(fig)

    metadata = {
        "status": "complete", "n_subjects": 37, "platform_k": list(K_VALUES),
        "platform_bins": ["(20,22]", "(22,24]", "(24,26]", "(26,28]", "(28,30]"],
        "fixed_local": "existing K=30 seed=0 r_real in 0-10mm bins",
        "common_valid_definition": "platform complete for all 5 seeds and all K=30/100/300",
        "no_null": True, "no_perturbation_model": True,
    }
    (PILOT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(platform_overall.to_string(index=False))
    print(changes_overall.to_string(index=False))
    print(comparison_summary.to_string(index=False))
    print(f"output_dir={PILOT.resolve()}")


if __name__ == "__main__":
    main()
