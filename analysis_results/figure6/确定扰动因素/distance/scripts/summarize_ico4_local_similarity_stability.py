#!/usr/bin/env python3
"""Summarize pair-sampling stability for the ico4 real-data pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[5]
EPS = 1e-8


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-dir", type=Path, default=REPO /
        "analysis_results/figure6/确定扰动因素/distance/local_similarity_stability")
    p.add_argument("--roster", type=Path, default=REPO /
        "analysis_results/figure6/figure6_new/outputs/full/young_reference_roster.csv")
    return p.parse_args()


def q25(x: pd.Series) -> float: return float(x.astype(float).quantile(.25))
def q75(x: pd.Series) -> float: return float(x.astype(float).quantile(.75))


def grouped_summary(frame: pd.DataFrame, group: list[str], columns: list[str]) -> pd.DataFrame:
    specs = {}
    for column in columns:
        specs.update({
            f"{column}_mean": (column, "mean"), f"{column}_median": (column, "median"),
            f"{column}_q1": (column, q25), f"{column}_q3": (column, q75),
        })
    specs["n_rows"] = (columns[0], "size")
    return frame.groupby(group, observed=True, sort=True).agg(**specs).reset_index()


def subject_stability(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["subject_id", "node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region",
            "distance_bin_index", "distance_bin", "k"]
    seed = features.groupby(keys, observed=True, sort=False).agg(
        r_seed_mean=("r_real", "mean"), r_seed_sd=("r_real", "std"),
        r_seed_range=("r_real", lambda x: float(x.max() - x.min())),
        L_seed_mean=("L_real", "mean"), L_seed_sd=("L_real", "std"),
        L_seed_range=("L_real", lambda x: float(x.max() - x.min())),
        mean_n_valid_pairs=("n_valid_pairs", "mean"), min_n_valid_pairs=("n_valid_pairs", "min"),
        n_valid_seeds=("r_real", "count"), all_platform_complete=("platform_complete", "all"),
    ).reset_index()
    seed.loc[~seed.all_platform_complete, ["L_seed_mean", "L_seed_sd", "L_seed_range"]] = np.nan

    join = ["subject_id", "node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region",
            "distance_bin_index", "distance_bin", "seed"]
    left = features.loc[features.k.eq(15)].copy()
    right = features.loc[features.k.eq(30)].copy()
    diff = left.merge(right, on=join, suffixes=("_k15", "_k30"), validate="one_to_one")
    for metric in ("r_real", "L_real"):
        diff[f"{metric}_diff_30_minus_15"] = diff[f"{metric}_k30"] - diff[f"{metric}_k15"]
        diff[f"{metric}_abs_diff"] = diff[f"{metric}_diff_30_minus_15"].abs()
    diff["both_platform_complete"] = diff.platform_complete_k15 & diff.platform_complete_k30
    diff.loc[~diff.both_platform_complete, [
        "L_real_diff_30_minus_15", "L_real_abs_diff",
    ]] = np.nan
    diff["valid_pair_gain"] = diff.n_valid_pairs_k30 - diff.n_valid_pairs_k15

    platform = features.loc[features.distance_bin_index.eq(10), [
        "subject_id", "node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region",
        "seed", "k", "p_real", "n_valid_platform_bins", "platform_complete",
    ]].copy()
    platform_seed = platform.groupby([
        "subject_id", "node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region", "k"
    ], observed=True, sort=False).agg(
        p_seed_mean=("p_real", "mean"), p_seed_sd=("p_real", "std"),
        p_seed_range=("p_real", lambda x: float(x.max() - x.min())),
        n_complete_seeds=("platform_complete", "sum"),
        all_platform_complete=("platform_complete", "all"),
    ).reset_index()
    platform_seed.loc[~platform_seed.all_platform_complete, ["p_seed_mean", "p_seed_sd", "p_seed_range"]] = np.nan
    return seed, diff, platform_seed


def map_correlations(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject, k, bin_index, bin_label), part in features.groupby(
        ["subject_id", "k", "distance_bin_index", "distance_bin"], observed=True, sort=True
    ):
        wide_r = part.pivot(index="node_order", columns="seed", values="r_real")
        wide_l = part.loc[part.platform_complete].pivot(index="node_order", columns="seed", values="L_real")
        for metric, wide in (("r_real", wide_r), ("L_real_complete_platform", wide_l)):
            for seed in range(1, 5):
                if 0 not in wide or seed not in wide:
                    continue
                valid = wide[0].notna() & wide[seed].notna()
                correlation = wide.loc[valid, [0, seed]].corr().iloc[0, 1] if valid.sum() >= 3 else np.nan
                rows.append({"subject_id": subject, "k": k, "distance_bin_index": bin_index,
                             "distance_bin": bin_label, "metric": metric, "seed_a": 0,
                             "seed_b": seed, "n_common_nodes": int(valid.sum()),
                             "map_correlation": correlation,
                             "coverage_fraction": float(valid.mean())})
    return pd.DataFrame(rows)


def make_figures(pilot: Path, bin_summary: pd.DataFrame, platform_summary: pd.DataFrame,
                 kdiff_summary: pd.DataFrame, coverage_summary: pd.DataFrame) -> None:
    figure_dir = pilot / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.style.use("default")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for k, color in ((15, "#4B74B2"), (30, "#FC8C5A")):
        q = bin_summary.loc[bin_summary.k.eq(k)]
        axes[0].plot(q.distance_bin_index * 2 + 1, q.r_seed_sd_median, marker="o", ms=3, label=f"K={k}", color=color)
        axes[1].plot(q.distance_bin_index * 2 + 1, q.L_seed_sd_median, marker="o", ms=3, label=f"K={k}", color=color)
    q = kdiff_summary
    axes[2].plot(q.distance_bin_index * 2 + 1, q.r_real_abs_diff_median, marker="o", ms=3, color="#6B6B6B", label="r")
    axes[2].plot(q.distance_bin_index * 2 + 1, q.L_real_abs_diff_median, marker="s", ms=3, color="#8E5EA2", label="L")
    axes[0].set_ylabel("Median across-seed SD of r")
    axes[1].set_ylabel("Median across-seed SD of L")
    axes[2].set_ylabel("Median |K30 − K15|")
    for ax in axes:
        ax.set_xlabel("Geodesic distance (mm)"); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "sampling_stability_by_distance.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for k, color in ((15, "#4B74B2"), (30, "#FC8C5A")):
        q = platform_summary.loc[platform_summary.k.eq(k)].iloc[0]
        axes[0].bar(str(k), q.p_seed_sd_median, color=color)
        axes[1].bar(str(k), q.all_platform_complete_mean, color=color)
    axes[0].set(xlabel="K", ylabel="Median across-seed SD of p")
    axes[1].set(xlabel="K", ylabel="Fraction with all 5 platform bins\ncomplete for all seeds", ylim=(0, 1.02))
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "platform_stability_and_completeness.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.7), constrained_layout=True)
    for k, color in ((15, "#4B74B2"), (30, "#FC8C5A")):
        q = coverage_summary.loc[coverage_summary.k.eq(k)]
        ax.plot(q.coverage_order, q.r_seed_sd_median, marker="o", label=f"K={k}", color=color)
    ax.set_xticks(sorted(coverage_summary.coverage_order.unique()))
    labels = coverage_summary.sort_values("coverage_order").drop_duplicates("coverage_order").coverage_stratum
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set(xlabel="Valid selected pairs", ylabel="Median across-seed SD of r")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "coverage_vs_sampling_instability.png", dpi=240)
    plt.close(fig)


def main() -> None:
    a = args()
    roster = pd.read_csv(a.roster, dtype={"subject_id": str})
    if len(roster) != 37:
        raise RuntimeError("Expected 37-person frozen roster")
    detail_dir = a.pilot_dir / "stability" / "subjects"
    detail_dir.mkdir(parents=True, exist_ok=True)
    all_seed, all_diff, all_platform = [], [], []
    curve_rows = []
    map_parts = []
    for subject in roster.subject_id:
        path = a.pilot_dir / "features" / "subjects" / f"{subject}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        features = pd.read_parquet(path)
        seed, diff, platform = subject_stability(features)
        seed.to_parquet(detail_dir / f"{subject}_seed_stability.parquet", index=False, compression="zstd")
        diff.to_parquet(detail_dir / f"{subject}_k_difference.parquet", index=False, compression="zstd")
        all_seed.append(seed); all_diff.append(diff); all_platform.append(platform)
        curves = features.groupby(["subject_id", "seed", "k", "distance_bin_index", "distance_bin"], observed=True).agg(
            whole_cortex_r=("r_real", "median"), whole_cortex_L=("L_real", "median"),
            node_coverage=("r_real", "count"), complete_platform_node_coverage=("platform_complete", "sum"),
        ).reset_index()
        curve_rows.append(curves)
        map_parts.append(map_correlations(features))

    seed = pd.concat(all_seed, ignore_index=True)
    diff = pd.concat(all_diff, ignore_index=True)
    platform = pd.concat(all_platform, ignore_index=True)
    curves = pd.concat(curve_rows, ignore_index=True)
    maps = pd.concat(map_parts, ignore_index=True)

    # Main bin and DK summaries. L summaries use complete five-bin platforms only.
    bin_summary = grouped_summary(seed, ["k", "distance_bin_index", "distance_bin"], [
        "r_seed_sd", "r_seed_range", "L_seed_sd", "L_seed_range", "mean_n_valid_pairs",
        "n_valid_seeds", "all_platform_complete",
    ])
    dk_summary = grouped_summary(seed, ["k", "hemisphere", "anchor_dk_region", "distance_bin_index", "distance_bin"], [
        "r_seed_sd", "L_seed_sd", "mean_n_valid_pairs", "all_platform_complete",
    ])
    kdiff_summary = grouped_summary(diff, ["distance_bin_index", "distance_bin"], [
        "r_real_diff_30_minus_15", "r_real_abs_diff", "L_real_diff_30_minus_15",
        "L_real_abs_diff", "valid_pair_gain", "both_platform_complete",
    ])
    kdiff_dk = grouped_summary(diff, ["hemisphere", "anchor_dk_region", "distance_bin_index", "distance_bin"], [
        "r_real_abs_diff", "L_real_abs_diff", "valid_pair_gain", "both_platform_complete",
    ])
    platform_summary = grouped_summary(platform, ["k"], [
        "p_seed_sd", "p_seed_range", "all_platform_complete", "n_complete_seeds",
    ])
    platform_dk = grouped_summary(platform, ["k", "hemisphere", "anchor_dk_region"], [
        "p_seed_sd", "p_seed_range", "all_platform_complete",
    ])

    labels = [-1, 0, 5, 14, 15, 29, np.inf]
    names = ["0", "1-5", "6-14", "15", "16-29", "30"]
    seed["coverage_stratum"] = pd.cut(seed.mean_n_valid_pairs, bins=labels, labels=names).astype(str)
    order = {name: i for i, name in enumerate(names)}
    seed["coverage_order"] = seed.coverage_stratum.map(order).astype(int)
    coverage_summary = grouped_summary(seed, ["k", "coverage_order", "coverage_stratum"], [
        "r_seed_sd", "L_seed_sd", "mean_n_valid_pairs", "all_platform_complete",
    ])

    # Sampling variation relative to between-young-subject variation at each anchor-bin.
    variability = seed.groupby([
        "node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region",
        "distance_bin_index", "distance_bin", "k",
    ], observed=True, sort=False).agg(
        between_subject_sd_r=("r_seed_mean", "std"),
        median_sampling_sd_r=("r_seed_sd", "median"),
        between_subject_sd_L=("L_seed_mean", "std"),
        median_sampling_sd_L=("L_seed_sd", "median"),
        n_subjects_r=("r_seed_mean", "count"), n_subjects_L=("L_seed_mean", "count"),
        median_valid_pairs=("mean_n_valid_pairs", "median"),
    ).reset_index()
    variability["between_subject_near_zero_r"] = variability.between_subject_sd_r.le(EPS)
    variability["between_subject_near_zero_L"] = variability.between_subject_sd_L.le(EPS)
    variability["sampling_to_between_ratio_r"] = np.where(
        variability.between_subject_sd_r > EPS,
        variability.median_sampling_sd_r / variability.between_subject_sd_r, np.nan)
    variability["sampling_to_between_ratio_L"] = np.where(
        variability.between_subject_sd_L > EPS,
        variability.median_sampling_sd_L / variability.between_subject_sd_L, np.nan)
    variability_summary = grouped_summary(variability, ["k", "distance_bin_index", "distance_bin"], [
        "between_subject_sd_r", "median_sampling_sd_r", "sampling_to_between_ratio_r",
        "between_subject_sd_L", "median_sampling_sd_L", "sampling_to_between_ratio_L",
        "between_subject_near_zero_r", "between_subject_near_zero_L", "median_valid_pairs",
    ])

    map_summary = grouped_summary(maps, ["k", "distance_bin_index", "distance_bin", "metric"], [
        "map_correlation", "coverage_fraction", "n_common_nodes",
    ])
    curve_seed = curves.groupby(["subject_id", "k", "distance_bin_index", "distance_bin"], observed=True).agg(
        curve_r_seed_sd=("whole_cortex_r", "std"), curve_L_seed_sd=("whole_cortex_L", "std")
    ).reset_index()
    curve_summary = grouped_summary(curve_seed, ["k", "distance_bin_index", "distance_bin"], [
        "curve_r_seed_sd", "curve_L_seed_sd",
    ])

    out = a.pilot_dir / "stability"
    bin_summary.to_csv(out / "stability_by_distance_bin.csv", index=False)
    dk_summary.to_csv(out / "stability_by_anchor_DK.csv", index=False)
    kdiff_summary.to_csv(out / "k15_vs_k30_by_distance_bin.csv", index=False)
    kdiff_dk.to_csv(out / "k15_vs_k30_by_anchor_DK.csv", index=False)
    platform_summary.to_csv(out / "platform_stability_summary.csv", index=False)
    platform_dk.to_csv(out / "platform_stability_by_anchor_DK.csv", index=False)
    coverage_summary.to_csv(out / "stability_by_effective_pair_count.csv", index=False)
    variability.to_parquet(out / "anchor_bin_sampling_vs_between_subject.parquet", index=False, compression="zstd")
    variability_summary.to_csv(out / "sampling_vs_between_subject_by_distance.csv", index=False)
    maps.to_parquet(out / "supplement_map_correlations_subject.parquet", index=False, compression="zstd")
    map_summary.to_csv(out / "supplement_map_correlations_summary.csv", index=False)
    curves.to_parquet(out / "supplement_whole_cortex_curves.parquet", index=False, compression="zstd")
    curve_summary.to_csv(out / "supplement_whole_cortex_curve_stability.csv", index=False)
    make_figures(a.pilot_dir, bin_summary, platform_summary, kdiff_summary, coverage_summary)

    report = {
        "status": "complete", "n_subjects": 37, "seeds": list(range(5)), "k_values": [15, 30],
        "near_zero_between_subject_sd_epsilon": EPS,
        "L_name": "L_real (real-data platform subtraction only; no null correction)",
        "primary_local_bins": ["(0,2]", "(2,4]", "(4,6]", "(6,8]", "(8,10]"],
        "platform_bins": ["(20,22]", "(22,24]", "(24,26]", "(26,28]", "(28,30]"],
    }
    (a.pilot_dir / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(bin_summary.loc[bin_summary.distance_bin_index.lt(5), [
        "k", "distance_bin", "r_seed_sd_median", "L_seed_sd_median", "mean_n_valid_pairs_median"
    ]].to_string(index=False))
    print(platform_summary[["k", "p_seed_sd_median", "all_platform_complete_mean"]].to_string(index=False))
    print(f"output_dir={a.pilot_dir.resolve()}")


if __name__ == "__main__":
    main()
