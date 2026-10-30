#!/usr/bin/env python3
"""Summarize and plot the 20-subject ico5 nested-K convergence pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from fit_correlation_lengths import icc_two_way_single

try:
    from qc.publication_style import (
        FIGURE_WIDTH_IN, PALETTE, save_publication_figure,
        set_publication_style, style_axes,
    )
except ImportError:
    from publication_style import (
        FIGURE_WIDTH_IN, PALETTE, save_publication_figure,
        set_publication_style, style_axes,
    )


K_VALUES = (1, 3, 5, 10, 15, 20, 25, 30)
REFERENCE_K = 30
METRICS = {
    "profile_dissimilarity": "Profile dissimilarity (1 − r)",
    "profile_mae": "Profile MAE",
    "symmetric_relative_mae_pct": "sRMAE (%)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--pair-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-subjects", type=int, default=20)
    parser.add_argument("--subject-list", type=Path)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(K_VALUES))
    parser.add_argument("--reference-k", type=int, default=REFERENCE_K)
    return parser.parse_args()


def read_subject_outputs(root: Path, filename: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"subjects/*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No {filename} under {root}")
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def safe_corr(function, first: pd.Series, second: pd.Series) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3:
        return np.nan
    return float(function(first[valid], second[valid]).statistic)


def long_metrics(table: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame:
    return table.melt(
        id_vars=id_columns,
        value_vars=list(METRICS),
        var_name="metric", value_name="value",
    )


def make_coverage(pair_output_dir: Path, output_dir: Path) -> pd.DataFrame:
    counts = pd.read_parquet(pair_output_dir / "ico5_within_candidate_counts.parquet")
    group_columns = ["anchor_region", "distance_bin_index", "distance_bin"]
    rows = []
    for keys, part in counts.groupby(group_columns, observed=True, sort=True):
        values = part["n_candidates"].to_numpy()
        row = dict(zip(group_columns, keys))
        row.update({
            "anchor_bin_total": len(values),
            "candidate_median": float(np.median(values)),
            "candidate_q25": float(np.quantile(values, .25)),
            "candidate_q75": float(np.quantile(values, .75)),
            "candidate_min": int(values.min()),
            "candidate_max": int(values.max()),
            "candidate_total": int(values.sum()),
        })
        for k in K_VALUES:
            row[f"fraction_at_least_{k}"] = float(np.mean(values >= k))
            row[f"pairs_used_k{k}"] = int(np.minimum(values, k).sum())
            row[f"anchors_retained_k{k}"] = int(np.sum(values > 0))
        rows.append(row)
    coverage = pd.DataFrame(rows)
    coverage.to_csv(output_dir / "k_convergence_coverage.csv", index=False)
    return coverage


def main() -> None:
    global K_VALUES, REFERENCE_K
    args = parse_args()
    K_VALUES = tuple(sorted(set(args.k_values)))
    REFERENCE_K = args.reference_k
    if REFERENCE_K not in K_VALUES:
        raise ValueError(f"reference K={REFERENCE_K} is not in K values {K_VALUES}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_publication_style()
    subject = read_subject_outputs(args.results_root, "k_convergence_subject_bins.csv")
    region = read_subject_outputs(args.results_root, "k_convergence_region_bins.csv")
    if args.subject_list is not None:
        requested = {
            value.strip() for value in args.subject_list.read_text().splitlines()
            if value.strip()
        }
        subject = subject.loc[subject["subject_id"].astype(str).isin(requested)].copy()
        region = region.loc[region["subject_id"].astype(str).isin(requested)].copy()
    n_subjects = subject["subject_id"].nunique()
    if n_subjects != args.expected_subjects:
        raise RuntimeError(f"Expected {args.expected_subjects} subjects, found {n_subjects}")
    subject.to_csv(args.output_dir / "k_convergence_subject_bins.csv", index=False)
    region.to_csv(args.output_dir / "k_convergence_region_bins.csv", index=False)
    coverage = make_coverage(args.pair_output_dir, args.output_dir)

    subject_long = long_metrics(subject, [
        "subject_id", "seed", "k", "distance_bin_index", "distance_bin",
        "median_geodesic_distance_mm", "anchor_count", "pair_count",
    ])
    group = subject_long.groupby(
        ["seed", "k", "metric", "distance_bin_index", "distance_bin"],
        as_index=False, observed=True,
    ).agg(
        group_median=("value", "median"),
        group_q25=("value", lambda x: x.quantile(.25)),
        group_q75=("value", lambda x: x.quantile(.75)),
        n_subjects=("subject_id", "nunique"),
        anchor_count_total=("anchor_count", "sum"),
        pair_count_total=("pair_count", "sum"),
    )
    group.to_csv(args.output_dir / "k_convergence_group_summary.csv", index=False)

    stability_rows = []
    subject_local = subject_long.loc[subject_long["distance_bin_index"].lt(5)].groupby(
        ["subject_id", "seed", "k", "metric"], as_index=False, observed=True
    )["value"].mean().rename(columns={"value": "mean_0_10"})
    region_long = long_metrics(region, [
        "subject_id", "seed", "k", "anchor_region", "distance_bin_index",
        "distance_bin", "median_geodesic_distance_mm", "anchor_count",
        "pair_count", "hemispheres_contributing",
    ])
    region_local = region_long.loc[region_long["distance_bin_index"].lt(5)].groupby(
        ["subject_id", "seed", "k", "metric", "anchor_region"],
        as_index=False, observed=True,
    )["value"].mean().rename(columns={"value": "mean_0_10"})

    for metric in METRICS:
        for seed in sorted(group["seed"].unique()):
            reference_curve = group.loc[
                group["seed"].eq(seed) & group["k"].eq(REFERENCE_K)
                & group["metric"].eq(metric)
            ].set_index("distance_bin_index")["group_median"]
            reference_subject = subject_local.loc[
                subject_local["seed"].eq(seed) & subject_local["k"].eq(REFERENCE_K)
                & subject_local["metric"].eq(metric)
            ].set_index("subject_id")["mean_0_10"]
            for k in K_VALUES:
                curve = group.loc[
                    group["seed"].eq(seed) & group["k"].eq(k)
                    & group["metric"].eq(metric)
                ].set_index("distance_bin_index")["group_median"]
                curve_pair = pd.concat([curve, reference_curve], axis=1).dropna()
                curve_pair.columns = ["current", "reference"]
                difference = curve_pair["current"] - curve_pair["reference"]
                current_subject = subject_local.loc[
                    subject_local["seed"].eq(seed) & subject_local["k"].eq(k)
                    & subject_local["metric"].eq(metric)
                ].set_index("subject_id")["mean_0_10"]
                subject_pair = pd.concat([current_subject, reference_subject], axis=1).dropna()
                subject_pair.columns = ["current", "reference"]
                region_correlations = []
                region_absolute = []
                for subject_id in sorted(region_local["subject_id"].unique()):
                    current_region = region_local.loc[
                        region_local["subject_id"].eq(subject_id)
                        & region_local["seed"].eq(seed) & region_local["k"].eq(k)
                        & region_local["metric"].eq(metric)
                    ].set_index("anchor_region")["mean_0_10"]
                    reference_region = region_local.loc[
                        region_local["subject_id"].eq(subject_id)
                        & region_local["seed"].eq(seed)
                        & region_local["k"].eq(REFERENCE_K)
                        & region_local["metric"].eq(metric)
                    ].set_index("anchor_region")["mean_0_10"]
                    paired = pd.concat([current_region, reference_region], axis=1).dropna()
                    paired.columns = ["current", "reference"]
                    if len(paired) >= 3:
                        region_correlations.append(safe_corr(
                            spearmanr, paired["current"], paired["reference"]
                        ))
                        region_absolute.extend(np.abs(
                            paired["current"] - paired["reference"]
                        ).tolist())
                finite_region_correlations = np.asarray(region_correlations, dtype=float)
                finite_region_correlations = finite_region_correlations[
                    np.isfinite(finite_region_correlations)
                ]
                finite_region_absolute = np.asarray(region_absolute, dtype=float)
                finite_region_absolute = finite_region_absolute[
                    np.isfinite(finite_region_absolute)
                ]
                stability_rows.append({
                    "metric": metric, "seed": seed, "k": k,
                    "curve_median_absolute_difference": float(np.median(np.abs(difference))),
                    "curve_q95_absolute_difference": float(np.quantile(np.abs(difference), .95)),
                    "curve_maximum_absolute_difference": float(np.max(np.abs(difference))),
                    "curve_fraction_absolute_difference_below_0.002": float(
                        np.mean(np.abs(difference) < .002)
                    ),
                    "curve_mean_difference_0_10": float(difference.loc[difference.index < 5].mean()),
                    "curve_auc_difference_0_30": float(np.trapezoid(difference, dx=2.0)),
                    "subject_pearson_r": safe_corr(pearsonr, subject_pair["current"], subject_pair["reference"]),
                    "subject_spearman_rho": safe_corr(spearmanr, subject_pair["current"], subject_pair["reference"]),
                    "subject_icc_two_way_single": icc_two_way_single(
                        subject_pair["current"].to_numpy(), subject_pair["reference"].to_numpy()
                    ),
                    "subject_median_absolute_difference": float(np.median(np.abs(
                        subject_pair["current"] - subject_pair["reference"]
                    ))),
                    "region_spearman_median_subject": (
                        float(np.median(finite_region_correlations))
                        if len(finite_region_correlations) else np.nan
                    ),
                    "region_spearman_min_subject": (
                        float(np.min(finite_region_correlations))
                        if len(finite_region_correlations) else np.nan
                    ),
                    "region_median_absolute_difference": (
                        float(np.median(finite_region_absolute))
                        if len(finite_region_absolute) else np.nan
                    ),
                    "region_q90_absolute_difference": (
                        float(np.quantile(finite_region_absolute, .9))
                        if len(finite_region_absolute) else np.nan
                    ),
                })
    stability = pd.DataFrame(stability_rows)

    seed_rows = []
    for (metric, k, bin_index, distance_bin), part in group.groupby(
        ["metric", "k", "distance_bin_index", "distance_bin"], observed=True
    ):
        seed_rows.append({
            "metric": metric, "k": k, "distance_bin_index": bin_index,
            "distance_bin": distance_bin,
            "seed_min": part["group_median"].min(),
            "seed_max": part["group_median"].max(),
            "seed_range": part["group_median"].max() - part["group_median"].min(),
            "seed_sd": part["group_median"].std(ddof=1),
            "seed_q25": part["group_median"].quantile(.25),
            "seed_q75": part["group_median"].quantile(.75),
            "seed_iqr": part["group_median"].quantile(.75) - part["group_median"].quantile(.25),
        })
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(args.output_dir / "k_convergence_seed_summary.csv", index=False)
    stability = stability.merge(
        seed_summary.groupby(["metric", "k"], as_index=False)["seed_range"].max()
        .rename(columns={"seed_range": "maximum_seed_to_seed_curve_difference"}),
        on=["metric", "k"], how="left",
    )
    stability.to_csv(args.output_dir / "k_convergence_stability.csv", index=False)

    full20_difference_summary = pd.DataFrame()
    full20_subject_path = next(
        iter(sorted(args.results_root.glob(
            "subjects/*/k_convergence_full20_subject_bins.csv"
        ))), None
    )
    if full20_subject_path is not None:
        full20_subject = read_subject_outputs(
            args.results_root, "k_convergence_full20_subject_bins.csv"
        )
        if args.subject_list is not None:
            full20_subject = full20_subject.loc[
                full20_subject["subject_id"].astype(str).isin(requested)
            ].copy()
        full20_subject.to_csv(
            args.output_dir / "k_convergence_full20_subject_bins.csv", index=False
        )
        full20_long = long_metrics(full20_subject, [
            "subject_id", "seed", "k", "distance_bin_index", "distance_bin",
            "median_geodesic_distance_mm", "anchor_count", "pair_count",
        ])
        full20_group = full20_long.groupby(
            ["seed", "k", "metric", "distance_bin_index", "distance_bin"],
            as_index=False, observed=True,
        ).agg(
            group_median=("value", "median"),
            group_q25=("value", lambda x: x.quantile(.25)),
            group_q75=("value", lambda x: x.quantile(.75)),
            n_subjects=("subject_id", "nunique"),
            anchor_count_total=("anchor_count", "sum"),
            pair_count_total=("pair_count", "sum"),
        )
        full20_group.to_csv(
            args.output_dir / "k_convergence_full20_group_summary.csv", index=False
        )
        diagnostic_rows = []
        for metric in METRICS:
            for seed in sorted(full20_group["seed"].unique()):
                diagnostic_reference_k = 20
                reference = full20_group.loc[
                    full20_group["metric"].eq(metric)
                    & full20_group["seed"].eq(seed)
                    & full20_group["k"].eq(diagnostic_reference_k)
                ].set_index("distance_bin_index")["group_median"]
                for k in (10, 15):
                    current = full20_group.loc[
                        full20_group["metric"].eq(metric)
                        & full20_group["seed"].eq(seed)
                        & full20_group["k"].eq(k)
                    ].set_index("distance_bin_index")["group_median"]
                    paired = pd.concat([current, reference], axis=1).dropna()
                    paired.columns = ["current", "reference"]
                    absolute = np.abs(paired["current"] - paired["reference"])
                    diagnostic_rows.append({
                        "metric": metric, "seed": seed, "k": k,
                        "reference_k": diagnostic_reference_k,
                        "eligible_anchor_rule": "n_candidates >= 20",
                        "median_absolute_difference": float(np.median(absolute)),
                        "q95_absolute_difference": float(np.quantile(absolute, .95)),
                        "maximum_absolute_difference": float(np.max(absolute)),
                        "fraction_absolute_difference_below_0.002": float(
                            np.mean(absolute < .002)
                        ),
                    })
        full20_difference_summary = pd.DataFrame(diagnostic_rows)
        full20_difference_summary.to_csv(
            args.output_dir / "k_convergence_full20_difference_summary.csv",
            index=False,
        )

    full30_difference_summary = pd.DataFrame()
    full30_subject_path = next(
        iter(sorted(args.results_root.glob(
            "subjects/*/k_convergence_full30_subject_bins.csv"
        ))), None
    )
    if full30_subject_path is not None and 30 in K_VALUES:
        full30_subject = read_subject_outputs(
            args.results_root, "k_convergence_full30_subject_bins.csv"
        )
        if args.subject_list is not None:
            full30_subject = full30_subject.loc[
                full30_subject["subject_id"].astype(str).isin(requested)
            ].copy()
        full30_subject.to_csv(
            args.output_dir / "k_convergence_full30_subject_bins.csv", index=False
        )
        full30_long = long_metrics(full30_subject, [
            "subject_id", "seed", "k", "distance_bin_index", "distance_bin",
            "median_geodesic_distance_mm", "anchor_count", "pair_count",
        ])
        full30_group = full30_long.groupby(
            ["seed", "k", "metric", "distance_bin_index", "distance_bin"],
            as_index=False, observed=True,
        ).agg(
            group_median=("value", "median"),
            group_q25=("value", lambda x: x.quantile(.25)),
            group_q75=("value", lambda x: x.quantile(.75)),
            n_subjects=("subject_id", "nunique"),
            anchor_count_total=("anchor_count", "sum"),
            pair_count_total=("pair_count", "sum"),
        )
        full30_group.to_csv(
            args.output_dir / "k_convergence_full30_group_summary.csv", index=False
        )
        diagnostic_rows = []
        for metric in METRICS:
            for seed in sorted(full30_group["seed"].unique()):
                reference = full30_group.loc[
                    full30_group["metric"].eq(metric)
                    & full30_group["seed"].eq(seed)
                    & full30_group["k"].eq(30)
                ].set_index("distance_bin_index")["group_median"]
                for k in (20, 25):
                    current = full30_group.loc[
                        full30_group["metric"].eq(metric)
                        & full30_group["seed"].eq(seed)
                        & full30_group["k"].eq(k)
                    ].set_index("distance_bin_index")["group_median"]
                    paired = pd.concat([current, reference], axis=1).dropna()
                    paired.columns = ["current", "reference"]
                    absolute = np.abs(paired["current"] - paired["reference"])
                    diagnostic_rows.append({
                        "metric": metric, "seed": seed, "k": k,
                        "reference_k": 30,
                        "eligible_anchor_rule": "n_candidates >= 30",
                        "median_absolute_difference": float(np.median(absolute)),
                        "q95_absolute_difference": float(np.quantile(absolute, .95)),
                        "maximum_absolute_difference": float(np.max(absolute)),
                        "fraction_absolute_difference_below_0.002": float(
                            np.mean(absolute < .002)
                        ),
                    })
        full30_difference_summary = pd.DataFrame(diagnostic_rows)
        full30_difference_summary.to_csv(
            args.output_dir / "k_convergence_full30_difference_summary.csv",
            index=False,
        )

    dissimilarity = stability.loc[stability["metric"].eq("profile_dissimilarity")]
    criterion_results = {}
    coverage_rows = pd.read_parquet(
        args.pair_output_dir / "ico5_within_candidate_counts.parquet",
        columns=["distance_bin_index", "n_candidates"],
    )
    overall_coverage = coverage_rows["n_candidates"]
    d_seed = seed_summary.loc[seed_summary["metric"].eq("profile_dissimilarity")]
    seed_q95_sd = d_seed.groupby("k")["seed_sd"].quantile(.95).to_dict()
    seed_q95_iqr = d_seed.groupby("k")["seed_iqr"].quantile(.95).to_dict()
    for k in (value for value in K_VALUES if value != 1):
        part = dissimilarity.loc[dissimilarity["k"].eq(k)]
        secondary = stability.loc[
            stability["k"].eq(k) & stability["metric"].isin(
                ["profile_mae", "symmetric_relative_mae_pct"]
            )
        ]
        result = {
            "group_curve_median_abs_difference": float(
                part["curve_median_absolute_difference"].median()
            ),
            "group_curve_worst_seed_q95_abs_difference": float(
                part["curve_q95_absolute_difference"].max()
            ),
            "group_curve_worst_seed_max_abs_difference": float(
                part["curve_maximum_absolute_difference"].max()
            ),
            "minimum_fraction_bins_below_0.002": float(
                part["curve_fraction_absolute_difference_below_0.002"].min()
            ),
            "subject_spearman_at_least_0.95": bool(part["subject_spearman_rho"].min() >= .95),
            "subject_icc_at_least_0.90": bool(part["subject_icc_two_way_single"].min() >= .90),
            "region_spearman_at_least_0.90": bool(
                part["region_spearman_median_subject"].min() >= .90
            ),
            "seed_q95_sd": float(seed_q95_sd.get(k, np.nan)),
            "seed_q95_iqr": float(seed_q95_iqr.get(k, np.nan)),
            "fraction_anchor_bins_with_at_least_k_candidates": float(
                np.mean(overall_coverage >= k)
            ),
            "fraction_0_2_mm_anchors_with_at_least_k_candidates": float(
                np.mean(
                    coverage_rows.loc[
                        coverage_rows["distance_bin_index"].eq(0), "n_candidates"
                    ] >= k
                )
            ),
            "minimum_0_10_mm_bin_fraction_with_at_least_k_candidates": float(
                coverage_rows.loc[coverage_rows["distance_bin_index"].lt(5)]
                .groupby("distance_bin_index")["n_candidates"]
                .apply(lambda values, threshold=k: np.mean(values >= threshold))
                .min()
            ),
            "secondary_subject_spearman_at_least_0.95": bool(
                secondary["subject_spearman_rho"].min() >= .95
            ),
            "secondary_region_spearman_at_least_0.90": bool(
                secondary["region_spearman_median_subject"].min() >= .90
            ),
        }
        criterion_results[str(k)] = result

    seed_sd_reduction_25_to_30 = None
    if "25" in criterion_results and "30" in criterion_results:
        k25 = criterion_results["25"]
        k30 = criterion_results["30"]
        seed_sd_reduction_25_to_30 = (
            (k25["seed_q95_sd"] - k30["seed_q95_sd"]) / k25["seed_q95_sd"]
            if k25["seed_q95_sd"] > 0 else 0.0
        )
    selected_k = None
    decision_reason = (
        f"K={REFERENCE_K} is retained as the current high-K sensitivity reference because nominal-K "
        "coverage decreases substantially at short distances. Final K selection "
        "requires joint review of effective-K curves, per-bin coverage, and the "
        "n_candidates>=20 and >=30 restricted diagnostics; no automatic selection "
        "is made from the unrestricted maximum or seed metric alone."
    )

    decision = {
        "k_values": list(K_VALUES), "n_seeds": int(subject["seed"].nunique()),
        "n_subjects": int(n_subjects), "reference_k": REFERENCE_K,
        "selected_k": selected_k,
        "criteria": {
            "k25_worst_seed_q95_absolute_difference": .0025,
            "k25_minimum_fraction_bins_below_0.002": .90,
            "minimum_subject_spearman": .95, "minimum_subject_icc": .90,
            "minimum_region_spearman": .90,
            "minimum_geometric_coverage": .95,
            "maximum_seed_sd_reduction_25_to_30": .20,
            "k30_maximum_q95_seed_sd": .002,
        },
        "criterion_results": criterion_results,
        "seed_sd_reduction_25_to_30": seed_sd_reduction_25_to_30,
        "decision_status": "joint_coverage_and_stability_review_required",
        "full20_candidate_diagnostic_available": bool(
            not full20_difference_summary.empty
        ),
        "full30_candidate_diagnostic_available": bool(
            not full30_difference_summary.empty
        ),
        "secondary_metrics_reported": ["profile_mae", "symmetric_relative_mae_pct"],
        "decision_reason": decision_reason,
    }
    (args.output_dir / "k_convergence_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )

    colors = {
        1: PALETTE["gray"], 3: PALETTE["blue"], 5: PALETTE["orange"],
        10: "#009E73", 15: "#CC79A7", 20: "#56B4E9",
        25: "#7A4EAB", 30: PALETTE["red"],
    }
    curve_average = group.loc[group["metric"].eq("profile_dissimilarity")].groupby(
        ["k", "distance_bin_index"], as_index=False
    ).agg(value=("group_median", "median"), q25=("group_q25", "median"), q75=("group_q75", "median"))
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.35), constrained_layout=True)
    for k in K_VALUES:
        part = curve_average.loc[curve_average["k"].eq(k)].sort_values("distance_bin_index")
        current_x = part["distance_bin_index"].to_numpy() * 2 + 1
        ax.plot(current_x, part["value"], color=colors[k], label=f"K={k}")
        if k == REFERENCE_K:
            ax.fill_between(current_x, part["q25"], part["q75"], color=colors[k], alpha=.18, linewidth=0)
    ax.set(xlabel="White-surface geodesic distance (mm)", ylabel="Profile dissimilarity (1 − r)")
    ax.legend(frameon=False, ncol=4)
    style_axes(ax)
    save_publication_figure(fig, args.output_dir / "FigureK1_curves")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.35), constrained_layout=True)
    reference = group.loc[
        group["metric"].eq("profile_dissimilarity") & group["k"].eq(REFERENCE_K)
    ][["seed", "distance_bin_index", "group_median"]].rename(columns={"group_median": "reference"})
    difference = group.loc[
        group["metric"].eq("profile_dissimilarity") & group["k"].ne(REFERENCE_K)
    ].merge(reference, on=["seed", "distance_bin_index"])
    difference["difference"] = difference["group_median"] - difference["reference"]
    for k in (value for value in K_VALUES if value != REFERENCE_K):
        part = difference.loc[difference["k"].eq(k)].groupby("distance_bin_index")["difference"].agg(
            median="median", q25=lambda z: z.quantile(.25), q75=lambda z: z.quantile(.75)
        )
        current_x = part.index.to_numpy() * 2 + 1
        ax.plot(
            current_x, part["median"], color=colors[k], label=f"K={k}",
            linewidth=1.7 if k in (20, 25) else 1.0,
            alpha=1.0 if k in (20, 25) else .60,
        )
        ax.fill_between(current_x, part["q25"], part["q75"], color=colors[k], alpha=.15, linewidth=0)
    ax.axhline(0, color=PALETTE["reference"], linewidth=.8)
    ax.axhline(.002, color=PALETTE["gray"], linestyle="--", linewidth=.8)
    ax.axhline(-.002, color=PALETTE["gray"], linestyle="--", linewidth=.8)
    ax.set(
        xlabel="White-surface geodesic distance (mm)",
        ylabel=f"D(K) − D(K={REFERENCE_K})",
    )
    ax.legend(frameon=False, ncol=4)
    style_axes(ax)
    save_publication_figure(
        fig, args.output_dir / f"FigureK2_difference_from_k{REFERENCE_K}"
    )
    plt.close(fig)

    matrix = coverage.pivot(index="anchor_region", columns="distance_bin_index", values="fraction_at_least_3")
    order = matrix.mean(axis=1).sort_values(ascending=False).index
    matrix = matrix.loc[order]
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 6.2), constrained_layout=True)
    sns.heatmap(matrix, ax=ax, cmap="viridis", vmin=0, vmax=1,
                cbar_kws={"label": "Anchor fraction with ≥3 candidates"})
    ax.set(xlabel="White-surface geodesic distance bin (mm)", ylabel="Bilateral DK region")
    ax.set_xticklabels([f"{2*i}–{2*i+2}" for i in matrix.columns], rotation=45, ha="right")
    save_publication_figure(fig, args.output_dir / "FigureK3_k3_coverage")
    plt.close(fig)


if __name__ == "__main__":
    main()
