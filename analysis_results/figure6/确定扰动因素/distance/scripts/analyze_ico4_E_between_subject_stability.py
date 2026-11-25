#!/usr/bin/env python3
"""Diagnose preservation of between-subject E differences using cached pilot results."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[5]
PILOT = REPO / "analysis_results/figure6/确定扰动因素/distance/null_adjusted_E_stability"
OUT = PILOT / "between_subject_stability"
EPS = 1e-8
MIN_N = 4


def row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    valid = np.isfinite(left) & np.isfinite(right)
    count = valid.sum(axis=1)
    left_mean = np.divide(np.nansum(np.where(valid, left, np.nan), axis=1), count,
                          out=np.full(len(left), np.nan), where=count > 0)
    right_mean = np.divide(np.nansum(np.where(valid, right, np.nan), axis=1), count,
                           out=np.full(len(right), np.nan), where=count > 0)
    lc = np.where(valid, left - left_mean[:, None], 0.0)
    rc = np.where(valid, right - right_mean[:, None], 0.0)
    denominator = np.sqrt(np.sum(lc * lc, axis=1) * np.sum(rc * rc, axis=1))
    return np.divide(np.sum(lc * rc, axis=1), denominator,
                     out=np.full(len(left), np.nan), where=denominator > EPS)


def compare_long(frame: pd.DataFrame, keys: list[str], label: str,
                 condition_type: str) -> pd.DataFrame:
    columns = sorted(frame.subject_id.unique())
    left = frame.pivot(index=keys, columns="subject_id", values="value_a").reindex(columns=columns)
    right = frame.pivot(index=keys, columns="subject_id", values="value_b").reindex(
        index=left.index, columns=columns)
    a, b = left.to_numpy(float), right.to_numpy(float)
    valid = np.isfinite(a) & np.isfinite(b)
    aa, bb = np.where(valid, a, np.nan), np.where(valid, b, np.nan)
    diff = aa - bb; average = (aa + bb) / 2
    n = valid.sum(axis=1)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        offset = np.nanmean(diff, axis=1)
        residual_sd = np.nanstd(diff - offset[:, None], axis=1, ddof=1)
        between_sd = np.nanstd(average, axis=1, ddof=1)
        sd_a = np.nanstd(aa, axis=1, ddof=1); sd_b = np.nanstd(bb, axis=1, ddof=1)
    pearson = row_correlation(aa, bb)
    rank_a = pd.DataFrame(aa).rank(axis=1, method="average", na_option="keep").to_numpy()
    rank_b = pd.DataFrame(bb).rank(axis=1, method="average", na_option="keep").to_numpy()
    spearman = row_correlation(rank_a, rank_b)
    result = left.index.to_frame(index=False)
    result["comparison_type"] = condition_type; result["comparison"] = label
    result["n_common_subjects"] = n
    result["common_offset_mean_A_minus_B"] = offset
    result["centered_residual_sd"] = residual_sd
    result["between_subject_sd_pair_average"] = between_sd
    result["between_subject_sd_A"] = sd_a; result["between_subject_sd_B"] = sd_b
    result["residual_to_between_sd_ratio"] = np.where(between_sd > EPS, residual_sd / between_sd, np.nan)
    result["pearson"] = pearson; result["spearman"] = spearman
    result["insufficient_subjects"] = n < MIN_N
    result["between_subject_near_zero"] = ~(between_sd > EPS)
    result["condition_A_near_zero"] = ~(sd_a > EPS)
    result["condition_B_near_zero"] = ~(sd_b > EPS)
    invalid = (result.insufficient_subjects | result.between_subject_near_zero
               | result.condition_A_near_zero | result.condition_B_near_zero)
    result.loc[invalid, ["pearson", "spearman", "residual_to_between_sd_ratio"]] = np.nan
    return result


def common_anchor_region(frame: pd.DataFrame, value_a: str, value_b: str,
                         label: str, condition_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = frame[[
        "subject_id", "node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region",
        "distance_bin_index", "distance_bin", value_a, value_b,
    ]].rename(columns={value_a: "value_a", value_b: "value_b"})
    current = current.loc[np.isfinite(current.value_a) & np.isfinite(current.value_b)].copy()
    group = ["subject_id", "hemisphere", "anchor_dk_label", "anchor_dk_region",
             "distance_bin_index", "distance_bin"]
    regional = current.groupby(group, observed=True, sort=True).agg(
        value_a=("value_a", "median"), value_b=("value_b", "median"),
        n_common_anchors=("node_order", "nunique"),
    ).reset_index()
    keys = ["hemisphere", "anchor_dk_label", "anchor_dk_region", "distance_bin_index", "distance_bin"]
    metrics = compare_long(regional, keys, label, condition_type)
    coverage = regional.groupby(keys, observed=True).agg(
        minimum_common_anchors=("n_common_anchors", "min"),
        median_common_anchors=("n_common_anchors", "median"),
        maximum_common_anchors=("n_common_anchors", "max"),
        n_subjects_with_region=("subject_id", "nunique"),
    ).reset_index()
    metrics = metrics.merge(coverage, on=keys, validate="one_to_one")
    regional["comparison_type"] = condition_type; regional["comparison"] = label
    return metrics, regional


def distribution(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    metrics = ["common_offset_mean_A_minus_B", "centered_residual_sd",
               "between_subject_sd_pair_average", "residual_to_between_sd_ratio",
               "pearson", "spearman"]
    rows = []
    for keys, part in frame.groupby(groups, observed=True, sort=True):
        if not isinstance(keys, tuple): keys = (keys,)
        row = dict(zip(groups, keys)); row["n_locations"] = len(part)
        row["insufficient_subjects_fraction"] = part.insufficient_subjects.mean()
        row["between_subject_near_zero_fraction"] = part.between_subject_near_zero.mean()
        row["correlation_computable_fraction"] = part.pearson.notna().mean()
        for metric in metrics:
            values = part[metric].dropna()
            row[f"{metric}_median"] = values.median() if len(values) else np.nan
            row[f"{metric}_q1"] = values.quantile(.25) if len(values) else np.nan
            row[f"{metric}_q3"] = values.quantile(.75) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(PILOT / "stability/E_final_and_null_batch_detail.parquet")
    source = source.loc[source.distance_bin_index.lt(5)].copy()
    anchor_keys = ["node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region",
                   "distance_bin_index", "distance_bin"]
    anchor_metrics, region_metrics = [], []
    regional_values = []

    # Pair-seed comparisons use the same combined 20 null realizations.
    for first, second in ((0, 1), (0, 2), (1, 2)):
        a = source.loc[source.pair_seed.eq(first), ["subject_id"] + anchor_keys + ["E_20"]]
        b = source.loc[source.pair_seed.eq(second), ["subject_id"] + anchor_keys + ["E_20"]]
        merged = a.merge(b, on=["subject_id"] + anchor_keys, suffixes=("_a", "_b"),
                         validate="one_to_one").rename(columns={"E_20_a": "value_a", "E_20_b": "value_b"})
        label = f"pair_seed_{first}_minus_{second}"
        anchor_metrics.append(compare_long(merged, anchor_keys, label, "pair_seed"))
        rm, rv = common_anchor_region(merged, "value_a", "value_b", label, "pair_seed")
        region_metrics.append(rm); regional_values.append(rv)

    # Null-batch comparisons fix each pair seed.
    for seed in (0, 1, 2):
        current = source.loc[source.pair_seed.eq(seed)].rename(columns={
            "batch_1_0_9": "value_a", "batch_2_10_19": "value_b"
        })
        label = f"null_batch1_minus_batch2_pair_seed_{seed}"
        anchor_metrics.append(compare_long(current, anchor_keys, label, "null_batch"))
        rm, rv = common_anchor_region(current, "value_a", "value_b", label, "null_batch")
        region_metrics.append(rm); regional_values.append(rv)

    anchor = pd.concat(anchor_metrics, ignore_index=True)
    region = pd.concat(region_metrics, ignore_index=True)
    regional_values = pd.concat(regional_values, ignore_index=True)
    anchor.to_parquet(OUT / "anchor_between_subject_stability_detail.parquet", index=False, compression="zstd")
    region.to_parquet(OUT / "DK_between_subject_stability_detail.parquet", index=False, compression="zstd")
    regional_values.to_parquet(OUT / "subject_DK_common_anchor_medians.parquet", index=False, compression="zstd")

    anchor_bin = distribution(anchor, ["comparison_type", "comparison", "distance_bin_index", "distance_bin"])
    anchor_dk = distribution(anchor, ["comparison_type", "comparison", "hemisphere", "anchor_dk_region",
                                      "distance_bin_index", "distance_bin"])
    region_bin = distribution(region, ["comparison_type", "comparison", "distance_bin_index", "distance_bin"])
    region_table = region.copy()
    anchor_bin.to_csv(OUT / "anchor_stability_by_distance.csv", index=False)
    anchor_dk.to_csv(OUT / "anchor_stability_by_DK.csv", index=False)
    region_bin.to_csv(OUT / "DK_summary_stability_by_distance.csv", index=False)
    region_table.to_csv(OUT / "DK_summary_stability_results.csv", index=False)

    # Compact comparison-type summaries pool the three prespecified comparisons.
    anchor_type = distribution(anchor, ["comparison_type", "distance_bin_index", "distance_bin"])
    region_type = distribution(region, ["comparison_type", "distance_bin_index", "distance_bin"])
    anchor_type.to_csv(OUT / "anchor_stability_comparison_type_summary.csv", index=False)
    region_type.to_csv(OUT / "DK_stability_comparison_type_summary.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
    specs = [
        ("pearson_median", "Median Pearson"), ("spearman_median", "Median Spearman"),
        ("residual_to_between_sd_ratio_median", "Median residual / between SD"),
    ]
    for column, ylabel in specs:
        index = specs.index((column, ylabel))
        for kind, color in (("pair_seed", "#4B74B2"), ("null_batch", "#FC8C5A")):
            q = anchor_type.loc[anchor_type.comparison_type.eq(kind)]
            axes[0, index].plot(q.distance_bin_index * 2 + 1, q[column], marker="o", label=kind, color=color)
            q = region_type.loc[region_type.comparison_type.eq(kind)]
            axes[1, index].plot(q.distance_bin_index * 2 + 1, q[column], marker="o", label=kind, color=color)
        axes[0, index].set(ylabel=ylabel, title="Anchor-bin")
        axes[1, index].set(xlabel="Distance (mm)", ylabel=ylabel, title="Hemisphere × DK median")
    for ax in axes.ravel():
        ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "between_subject_stability.png", dpi=240)
    plt.close(fig)

    report = {
        "status": "complete", "source": str((PILOT / "stability/E_final_and_null_batch_detail.parquet").resolve()),
        "n_subjects_max": 10, "pair_seed_comparisons": ["0-1", "0-2", "1-2"],
        "null_batch_comparisons": ["batch0-9 vs batch10-19 at pair seed 0/1/2"],
        "between_subject_sd_definition": "SD across common subjects of (E_A+E_B)/2",
        "common_offset_definition": "mean across common subjects of E_A-E_B",
        "minimum_subjects_for_correlation": MIN_N,
        "near_zero_sd_epsilon": EPS, "new_pairs": False, "new_null_realizations": False,
        "formal_reference_model": False, "automatic_exclusion": False,
    }
    (OUT / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(anchor_type.to_string(index=False)); print(region_type.to_string(index=False))
    print(f"output_dir={OUT.resolve()}")


if __name__ == "__main__":
    main()
