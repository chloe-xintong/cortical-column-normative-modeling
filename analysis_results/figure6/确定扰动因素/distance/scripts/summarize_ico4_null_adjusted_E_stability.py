#!/usr/bin/env python3
"""Summarize pair-seed and null-batch sensitivity of null-adjusted E."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[5]
PILOT = REPO / "analysis_results/figure6/确定扰动因素/distance/null_adjusted_E_stability"


def q25(x: pd.Series) -> float: return float(x.astype(float).quantile(.25))
def q75(x: pd.Series) -> float: return float(x.astype(float).quantile(.75))


def summarize(frame: pd.DataFrame, groups: list[str], columns: list[str]) -> pd.DataFrame:
    work = frame if groups else frame.assign(_all="all")
    group = groups or ["_all"]
    spec = {}
    for col in columns:
        spec.update({f"{col}_mean": (col, "mean"), f"{col}_median": (col, "median"),
                     f"{col}_q1": (col, q25), f"{col}_q3": (col, q75)})
    spec["n_rows"] = (columns[0], "size")
    return work.groupby(group, observed=True, sort=True).agg(**spec).reset_index().drop(
        columns="_all", errors="ignore")


def main() -> None:
    roster = pd.read_csv(PILOT / "pilot_roster.csv", dtype={"subject_id": str})
    parts = []
    completion_rows = []
    for subject in roster.subject_id:
        root = PILOT / "subjects" / subject
        completion = json.loads((root / "completion.json").read_text())
        if completion["replicates"] != 20 or completion["pair_seeds"] != [0, 1, 2]:
            raise RuntimeError(f"Incomplete configuration for {subject}")
        completion_rows.append({
            "subject_id": subject, "replicates": completion["replicates"],
            "same_null_reused_across_pair_seeds": completion["same_null_reused_across_pair_seeds"],
            "runtime_seconds": completion["runtime_seconds"], "max_rss_mb": completion["max_rss_mb"],
        })
        parts.append(pd.read_parquet(root / "null_adjusted_local_realizations.parquet"))
    data = pd.concat(parts, ignore_index=True)
    data["both_complete"] = data.platform_complete_real & data.platform_complete_null
    keys = ["subject_id", "node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region",
            "distance_bin_index", "distance_bin", "pair_seed"]
    final = data.groupby(keys, observed=True, sort=False).agg(
        E_20=("E_realization", "median"),
        E_realization_sd=("E_realization", "std"),
        E_realization_range=("E_realization", lambda x: float(x.max() - x.min())),
        r_real=("r_real", "first"), p_real=("p_real", "first"), L_real=("L_real", "first"),
        median_r_null=("r_null", "median"), median_p_null=("p_null", "median"),
        median_L_null=("L_null", "median"), n_valid_real_pairs=("n_valid_real_pairs", "first"),
        median_valid_null_pairs=("n_valid_null_pairs", "median"),
        platform_complete_real=("platform_complete_real", "first"),
        all_null_platforms_complete=("platform_complete_null", "all"),
        local_candidate_exhausted=("local_candidate_exhausted", "first"),
        platform_any_candidate_exhausted=("platform_any_candidate_exhausted", "first"),
        n_realizations_valid=("E_realization", "count"),
    ).reset_index()
    batches = []
    for label, select in (("batch_1_0_9", data.replicate.lt(10)),
                          ("batch_2_10_19", data.replicate.ge(10))):
        current = data.loc[select].groupby(keys, observed=True, sort=False).agg(
            E_batch=("E_realization", "median"), n_batch_valid=("E_realization", "count")
        ).reset_index()
        current["batch"] = label; batches.append(current)
    batches = pd.concat(batches, ignore_index=True)
    wide = batches.pivot(index=keys, columns="batch", values="E_batch").reset_index()
    final = final.merge(wide, on=keys, validate="one_to_one")
    final["batch_signed_difference"] = final.batch_2_10_19 - final.batch_1_0_9
    final["batch_absolute_difference"] = final.batch_signed_difference.abs()
    final["batch1_vs_20_absolute_difference"] = (final.batch_1_0_9 - final.E_20).abs()
    final["batch2_vs_20_absolute_difference"] = (final.batch_2_10_19 - final.E_20).abs()
    final["analysis_complete"] = (
        final.platform_complete_real & final.all_null_platforms_complete
        & final.n_realizations_valid.eq(20)
    )

    # Pair-seed sensitivity on the exact common complete position intersection.
    common = final.groupby([
        "subject_id", "node_order", "distance_bin_index"
    ], observed=True).analysis_complete.all().rename("common_pair_seed_complete").reset_index()
    final = final.merge(common, on=["subject_id", "node_order", "distance_bin_index"], validate="many_to_one")
    pair_seed = final.groupby([
        "subject_id", "node_order", "hemisphere", "anchor_dk_label", "anchor_dk_region",
        "distance_bin_index", "distance_bin",
    ], observed=True, sort=False).agg(
        E_pair_seed_mean=("E_20", "mean"), E_pair_seed_sd=("E_20", "std"),
        E_pair_seed_range=("E_20", lambda x: float(x.max() - x.min())),
        common_pair_seed_complete=("common_pair_seed_complete", "all"),
        minimum_real_pairs=("n_valid_real_pairs", "min"),
        minimum_null_pairs=("median_valid_null_pairs", "min"),
        any_local_candidate_exhausted=("local_candidate_exhausted", "any"),
        any_platform_candidate_exhausted=("platform_any_candidate_exhausted", "any"),
    ).reset_index()
    for first, second in ((0, 1), (0, 2), (1, 2)):
        left = final.loc[final.pair_seed.eq(first), ["subject_id", "node_order", "distance_bin_index", "E_20"]]
        right = final.loc[final.pair_seed.eq(second), ["subject_id", "node_order", "distance_bin_index", "E_20"]]
        delta = left.merge(right, on=["subject_id", "node_order", "distance_bin_index"],
                           suffixes=(f"_{first}", f"_{second}"), validate="one_to_one")
        delta[f"E_abs_diff_seed{first}_{second}"] = (
            delta[f"E_20_{first}"] - delta[f"E_20_{second}"]).abs()
        pair_seed = pair_seed.merge(delta[["subject_id", "node_order", "distance_bin_index",
            f"E_abs_diff_seed{first}_{second}"]],
            on=["subject_id", "node_order", "distance_bin_index"], validate="one_to_one")
    pair_seed["median_pairwise_E_absolute_difference"] = pair_seed[[
        "E_abs_diff_seed0_1", "E_abs_diff_seed0_2", "E_abs_diff_seed1_2"
    ]].median(axis=1)

    valid_final = final.loc[final.common_pair_seed_complete]
    valid_pair = pair_seed.loc[pair_seed.common_pair_seed_complete]
    pair_bin = summarize(valid_pair, ["distance_bin_index", "distance_bin"], [
        "E_pair_seed_sd", "E_pair_seed_range", "median_pairwise_E_absolute_difference",
        "minimum_real_pairs", "minimum_null_pairs", "any_local_candidate_exhausted",
        "any_platform_candidate_exhausted",
    ])
    pair_dk = summarize(valid_pair, ["hemisphere", "anchor_dk_region", "distance_bin_index", "distance_bin"], [
        "E_pair_seed_sd", "median_pairwise_E_absolute_difference", "minimum_real_pairs",
    ])
    null_bin = summarize(valid_final, ["pair_seed", "distance_bin_index", "distance_bin"], [
        "batch_absolute_difference", "batch1_vs_20_absolute_difference",
        "batch2_vs_20_absolute_difference", "E_realization_sd", "E_realization_range",
        "E_20", "n_valid_real_pairs", "median_valid_null_pairs",
    ])
    null_dk = summarize(valid_final, ["pair_seed", "hemisphere", "anchor_dk_region",
                                      "distance_bin_index", "distance_bin"], [
        "batch_absolute_difference", "E_realization_sd", "E_20",
    ])
    coverage = final.groupby(["pair_seed", "distance_bin_index", "distance_bin"], observed=True).agg(
        n_positions=("analysis_complete", "size"), analysis_complete_fraction=("analysis_complete", "mean"),
        common_pair_seed_complete_fraction=("common_pair_seed_complete", "mean"),
        local_candidate_exhausted_fraction=("local_candidate_exhausted", "mean"),
        platform_candidate_exhausted_fraction=("platform_any_candidate_exhausted", "mean"),
        median_real_pairs=("n_valid_real_pairs", "median"),
        median_null_pairs=("median_valid_null_pairs", "median"),
    ).reset_index()

    # Audit the effect of switching the earlier pilot cache to the stricter formal 4B mask.
    distance_root = PILOT.parent
    audit_parts = []
    for subject in roster.subject_id:
        old_local = pd.read_parquet(
            distance_root / "local_similarity_stability/features/subjects" / f"{subject}.parquet"
        ).loc[lambda x: x.k.eq(30) & x.seed.lt(3) & x.distance_bin_index.lt(5), [
            "subject_id", "node_order", "distance_bin_index", "seed", "r_real"
        ]].rename(columns={"seed": "pair_seed", "r_real": "r_old_pilot_mask"})
        old_platform = pd.read_parquet(
            distance_root / "platform_sampling_sensitivity/features/subjects" / f"{subject}_platform.parquet"
        ).loc[lambda x: x.platform_k.eq(300) & x.seed.lt(3), [
            "subject_id", "node_order", "seed", "p_real"
        ]].rename(columns={"seed": "pair_seed", "p_real": "p_old_pilot_mask"})
        old = old_local.merge(old_platform, on=["subject_id", "node_order", "pair_seed"],
                              validate="many_to_one")
        new = final.loc[final.subject_id.eq(subject), [
            "subject_id", "node_order", "distance_bin_index", "distance_bin", "pair_seed",
            "r_real", "p_real",
        ]]
        audit_parts.append(new.merge(old, on=["subject_id", "node_order", "distance_bin_index", "pair_seed"],
                                     validate="one_to_one"))
    audit = pd.concat(audit_parts, ignore_index=True)
    audit["r_formal_minus_old"] = audit.r_real - audit.r_old_pilot_mask
    audit["p_formal_minus_old"] = audit.p_real - audit.p_old_pilot_mask
    audit["r_absolute_difference"] = audit.r_formal_minus_old.abs()
    audit["p_absolute_difference"] = audit.p_formal_minus_old.abs()
    audit_summary = summarize(audit, ["distance_bin_index", "distance_bin"], [
        "r_formal_minus_old", "r_absolute_difference", "p_formal_minus_old", "p_absolute_difference",
    ])

    out = PILOT / "stability"; out.mkdir(exist_ok=True)
    final.to_parquet(out / "E_final_and_null_batch_detail.parquet", index=False, compression="zstd")
    pair_seed.to_parquet(out / "E_pair_seed_stability_detail.parquet", index=False, compression="zstd")
    pair_bin.to_csv(out / "E_pair_seed_stability_by_distance.csv", index=False)
    pair_dk.to_csv(out / "E_pair_seed_stability_by_anchor_DK.csv", index=False)
    null_bin.to_csv(out / "E_null_batch_stability_by_distance.csv", index=False)
    null_dk.to_csv(out / "E_null_batch_stability_by_anchor_DK.csv", index=False)
    coverage.to_csv(out / "coverage_summary.csv", index=False)
    audit.to_parquet(out / "formal_mask_vs_old_pilot_detail.parquet", index=False, compression="zstd")
    audit_summary.to_csv(out / "formal_mask_vs_old_pilot_summary.csv", index=False)
    pd.DataFrame(completion_rows).to_csv(out / "subject_run_inventory.csv", index=False)

    fig_dir = PILOT / "figures"; fig_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    axes[0].plot(pair_bin.distance_bin_index * 2 + 1,
                 pair_bin.median_pairwise_E_absolute_difference_median, marker="o")
    axes[0].set(xlabel="Distance (mm)", ylabel="Median pair-seed |ΔE|")
    for seed in (0, 1, 2):
        q = null_bin.loc[null_bin.pair_seed.eq(seed)]
        axes[1].plot(q.distance_bin_index * 2 + 1, q.batch_absolute_difference_median,
                     marker="o", label=f"pair seed {seed}")
    axes[1].set(xlabel="Distance (mm)", ylabel="Median |E batch2 − batch1|")
    axes[1].legend(frameon=False)
    axes[2].plot(pair_bin.distance_bin_index * 2 + 1,
                 pair_bin.E_pair_seed_sd_median, marker="o", label="pair-seed SD")
    q = null_bin.groupby(["distance_bin_index", "distance_bin"], observed=True).batch_absolute_difference_median.median().reset_index()
    axes[2].plot(q.distance_bin_index * 2 + 1, q.batch_absolute_difference_median,
                 marker="s", label="10-vs-10 |ΔE|")
    axes[2].set(xlabel="Distance (mm)", ylabel="Location-level instability")
    axes[2].legend(frameon=False)
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(fig_dir / "null_adjusted_E_stability.png", dpi=240)
    plt.close(fig)

    report = {
        "status": "complete", "n_subjects": 10, "selection_seed": 20261121,
        "pair_seeds": [0, 1, 2], "null_replicates": 20,
        "null_batches": {"batch_1": [0, 9], "batch_2": [10, 19]},
        "E_definition": "median_replicate((r_real-p_real)-(r_null-p_null))",
        "negative_E_retained": True, "no_threshold": True,
        "no_young_reference_model": True, "no_group_comparison": True,
    }
    (PILOT / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(pair_bin.to_string(index=False))
    print(null_bin.to_string(index=False))
    print(f"output_dir={PILOT.resolve()}")


if __name__ == "__main__":
    main()
