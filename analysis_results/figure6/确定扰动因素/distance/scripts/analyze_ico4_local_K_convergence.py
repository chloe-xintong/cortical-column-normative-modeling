#!/usr/bin/env python3
"""Summarize between-subject convergence of null-adjusted E at local K=30/100/300."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[5]
DISTANCE = REPO / "analysis_results/figure6/确定扰动因素/distance"
PILOT = DISTANCE / "null_adjusted_E_stability/local_K_convergence"
OUT = PILOT / "stability"
sys.path.insert(0, str(DISTANCE / "scripts"))
from analyze_ico4_E_between_subject_stability import compare_long, common_anchor_region  # noqa: E402

ANCHOR_KEYS = ["node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region",
               "distance_bin_index", "distance_bin"]


def summarize_locations(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    metrics = ["common_offset_mean_A_minus_B", "absolute_common_offset",
               "centered_residual_sd", "between_subject_sd_pair_average",
               "residual_to_between_sd_ratio", "pearson", "spearman"]
    rows = []
    for keys, part in frame.groupby(groups, observed=True, sort=True):
        if not isinstance(keys, tuple): keys = (keys,)
        row = dict(zip(groups, keys)); row["n_locations"] = len(part)
        row["correlation_computable_fraction"] = part.pearson.notna().mean()
        row["insufficient_subjects_fraction"] = part.insufficient_subjects.mean()
        row["between_subject_near_zero_fraction"] = part.between_subject_near_zero.mean()
        for metric in metrics:
            values = part[metric].dropna()
            for name, quantile in (("median", .5), ("q1", .25), ("q3", .75)):
                row[f"{metric}_{name}"] = values.quantile(quantile) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_realizations() -> Path:
    target = PILOT / "E_final_local_K.parquet"
    if target.exists():
        return target
    writer = None
    try:
        for path in sorted((PILOT / "subjects").glob("*/local_K_realizations.parquet")):
            frame = pd.read_parquet(path)
            keys = ["subject_id", "pair_seed", "local_k"] + ANCHOR_KEYS
            final = frame.groupby(keys, observed=True, sort=True).agg(
                E=("E_realization", "median"), r_real=("r_real", "first"),
                p_real=("p_real", "first"), L_real=("L_real", "first"),
                n_valid_real_pairs=("n_valid_real_pairs", "first"),
                n_valid_null_pairs_median=("n_valid_null_pairs", "median"),
                n_template_selected=("n_template_selected", "first"),
                candidate_exhausted=("candidate_exhausted", "first"),
                n_valid_platform_bins_real=("n_valid_platform_bins_real", "first"),
                n_valid_platform_bins_null_min=("n_valid_platform_bins_null", "min"),
                platform_complete_real=("platform_complete_real", "first"),
                platform_complete_null_all=("platform_complete_null", "all"),
                platform_candidate_exhausted=("platform_candidate_exhausted", "first"),
            ).reset_index()
            table = pa.Table.from_pandas(final, preserve_index=False)
            if writer is None: writer = pq.ParquetWriter(target, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None: writer.close()
    return target


def comparison(source: pd.DataFrame, condition: str, label: str,
               selector_a: dict, selector_b: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def select(filters):
        keep = np.ones(len(source), bool)
        for column, value in filters.items(): keep &= source[column].eq(value).to_numpy()
        return source.loc[keep, ["subject_id"] + ANCHOR_KEYS + ["E"]]
    a, b = select(selector_a), select(selector_b)
    merged = a.merge(b, on=["subject_id"] + ANCHOR_KEYS, suffixes=("_a", "_b"),
                     validate="one_to_one").rename(columns={"E_a": "value_a", "E_b": "value_b"})
    metrics = compare_long(merged, ANCHOR_KEYS, label, condition)
    metrics["absolute_common_offset"] = metrics.common_offset_mean_A_minus_B.abs()
    for column, value in {**selector_a, **{f"B_{k}": v for k, v in selector_b.items()}}.items():
        metrics[column] = value
    region, values = common_anchor_region(merged, "value_a", "value_b", label, condition)
    region["absolute_common_offset"] = region.common_offset_mean_A_minus_B.abs()
    for column, value in {**selector_a, **{f"B_{k}": v for k, v in selector_b.items()}}.items():
        region[column] = value
    return metrics, region, values


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_path = aggregate_realizations()
    source = pd.read_parquet(source_path)
    anchors, regions, regional_values = [], [], []
    for k in (30, 100, 300):
        for first, second in ((0, 1), (0, 2), (1, 2)):
            result = comparison(source, "pair_seed", f"seed_{first}_minus_{second}_K{k}",
                                {"local_k": k, "pair_seed": first},
                                {"local_k": k, "pair_seed": second})
            anchors.append(result[0]); regions.append(result[1]); regional_values.append(result[2])
    for seed in (0, 1, 2):
        for first, second in ((30, 100), (100, 300), (30, 300)):
            result = comparison(source, "local_K", f"K{first}_minus_K{second}_seed_{seed}",
                                {"pair_seed": seed, "local_k": first},
                                {"pair_seed": seed, "local_k": second})
            anchors.append(result[0]); regions.append(result[1]); regional_values.append(result[2])
    anchor = pd.concat(anchors, ignore_index=True); region = pd.concat(regions, ignore_index=True)
    pd.concat(regional_values, ignore_index=True).to_parquet(
        OUT / "subject_DK_common_anchor_medians.parquet", index=False, compression="zstd")
    anchor.to_parquet(OUT / "anchor_stability_detail.parquet", index=False, compression="zstd")
    region.to_parquet(OUT / "DK_stability_detail.parquet", index=False, compression="zstd")

    anchor_bin = summarize_locations(anchor, ["comparison_type", "comparison", "distance_bin_index", "distance_bin"])
    region_bin = summarize_locations(region, ["comparison_type", "comparison", "distance_bin_index", "distance_bin"])
    anchor_dk = summarize_locations(anchor, ["comparison_type", "comparison", "local_k", "hemisphere", "anchor_dk_region",
                                                   "distance_bin_index", "distance_bin"])
    anchor_bin.to_csv(OUT / "anchor_stability_by_distance.csv", index=False)
    region_bin.to_csv(OUT / "DK_stability_by_distance.csv", index=False)
    anchor_dk.to_csv(OUT / "anchor_stability_by_DK.csv", index=False)
    region.to_csv(OUT / "DK_stability_results.csv", index=False)
    focus = anchor_dk.loc[anchor_dk.anchor_dk_region.str.contains(
        "entorhinal|temporalpole|inferiortemporal|cingulate", case=False, regex=True, na=False)]
    focus.to_csv(OUT / "focus_regions_stability.csv", index=False)

    coverage = source.groupby(["local_k", "distance_bin_index", "distance_bin"], observed=True).agg(
        n_subject_anchor_seed=("E", "size"), E_valid_fraction=("E", lambda x: x.notna().mean()),
        candidate_exhausted_fraction=("candidate_exhausted", "mean"),
        platform_complete_real_fraction=("platform_complete_real", "mean"),
        platform_complete_null_all_fraction=("platform_complete_null_all", "mean"),
        n_valid_real_pairs_median=("n_valid_real_pairs", "median"),
        n_valid_real_pairs_q1=("n_valid_real_pairs", lambda x: x.quantile(.25)),
        n_valid_real_pairs_q3=("n_valid_real_pairs", lambda x: x.quantile(.75)),
    ).reset_index()
    coverage.to_csv(OUT / "coverage_by_distance.csv", index=False)

    pair_summary = summarize_locations(anchor.loc[anchor.comparison_type.eq("pair_seed")],
                                       ["local_k", "distance_bin_index", "distance_bin"])
    k_summary = summarize_locations(anchor.loc[anchor.comparison_type.eq("local_K")],
                                    ["comparison", "distance_bin_index", "distance_bin"])
    pair_summary.to_csv(OUT / "pair_seed_convergence_summary.csv", index=False)
    k_summary.to_csv(OUT / "within_seed_K_change_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for k, color in zip((30, 100, 300), ("#999999", "#4B74B2", "#D55E00")):
        q = pair_summary.loc[pair_summary.local_k.eq(k)]
        x = q.distance_bin_index * 2 + 1
        axes[0].plot(x, q.pearson_median, marker="o", color=color, label=f"K={k}")
        axes[1].plot(x, q.spearman_median, marker="o", color=color)
        axes[2].plot(x, q.residual_to_between_sd_ratio_median, marker="o", color=color)
    axes[0].set_ylabel("Median Pearson"); axes[1].set_ylabel("Median Spearman")
    axes[2].set_ylabel("Median centered residual / between SD")
    for ax in axes:
        ax.set_xlabel("Distance (mm)"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.savefig(OUT / "local_K_between_subject_convergence.png", dpi=240); plt.close(fig)

    completions = [json.loads(p.read_text()) for p in sorted((PILOT / "subjects").glob("*/completion.json"))]
    report = {
        "status": "complete", "n_subjects": len(completions), "local_k": [30, 100, 300],
        "pair_seeds": [0, 1, 2], "null_realizations": 20,
        "maximum_k30_r_null_absolute_error": max(x["maximum_k30_r_null_absolute_error"] for x in completions),
        "maximum_k30_E_absolute_error": max(x["maximum_k30_E_absolute_error"] for x in completions),
        "platform_identical_across_local_K": True, "new_null_realizations": False,
        "common_offset_reported_as_signed_and_absolute": True,
        "automatic_exclusion": False, "prespecified_threshold": False,
    }
    (OUT / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(pair_summary.to_string(index=False)); print(coverage.to_string(index=False))
    print(f"output_dir={OUT.resolve()}")


if __name__ == "__main__": main()
