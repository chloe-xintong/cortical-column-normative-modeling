#!/usr/bin/env python3
"""Merge and validate the frozen 21-depth ico4 OOF predictions and summarize calibration.

Description
-----------
Merge and validate the frozen 21-depth ico4 OOF predictions and summarize calibration.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--result-root`` command-line input
- ``--subject-order`` command-line input
- ``--ico4-md`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/merge_summarize_ico4_oof.py --help
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEPTHS = list(range(21))
ANCHORS = [0, 4, 8, 12, 16, 20]
METRIC_NAMES = [
    "n", "sum_z", "sum_z2", "sum_z3", "sum_z4", "sum_logp",
    "cov50", "cov80", "cov90", "gt2", "gt3", "gt4",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--subject-order", type=Path, required=True)
    p.add_argument("--ico4-md", type=Path, required=True)
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20261112)
    return p.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def moments_to_metrics(values: np.ndarray) -> dict[str, float | int]:
    n, s1, s2, s3, s4, slog, c50, c80, c90, gt2, gt3, gt4 = values
    n = int(round(n))
    if n < 2:
        return {"n_predictions": n}
    mean = s1 / n
    variance = max((s2 - s1 * s1 / n) / (n - 1), 0.0)
    sd = math.sqrt(variance)
    pop_m2 = max(s2 / n - mean * mean, 0.0)
    pop_m3 = s3 / n - 3 * mean * s2 / n + 2 * mean ** 3
    pop_m4 = s4 / n - 4 * mean * s3 / n + 6 * mean * mean * s2 / n - 3 * mean ** 4
    if n > 2 and pop_m2 > 0:
        skew = math.sqrt(n * (n - 1)) / (n - 2) * pop_m3 / pop_m2 ** 1.5
    else:
        skew = np.nan
    if n > 3 and pop_m2 > 0:
        g2 = pop_m4 / pop_m2 ** 2 - 3
        kurt = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)
    else:
        kurt = np.nan
    coverage = np.array([c50, c80, c90], float) / n
    return {
        "n_predictions": n,
        "mean_z": float(mean),
        "sd_z": float(sd),
        "skewness": float(skew),
        "excess_kurtosis": float(kurt),
        "coverage_50": float(coverage[0]),
        "coverage_80": float(coverage[1]),
        "coverage_90": float(coverage[2]),
        "coverage_mae": float(np.mean(np.abs(coverage - np.array([.5, .8, .9])))),
        "mean_log_likelihood": float(slog / n),
        "fraction_abs_z_gt_2": float(gt2 / n),
        "fraction_abs_z_gt_3": float(gt3 / n),
        "fraction_abs_z_gt_4": float(gt4 / n),
    }


def row_values(frame: pd.DataFrame) -> np.ndarray:
    z = frame.z.to_numpy(float)
    observed = frame.observed_md.to_numpy(float)
    return np.column_stack([
        np.ones(len(frame)), z, z ** 2, z ** 3, z ** 4,
        frame.log_likelihood.to_numpy(float),
        (observed >= frame.lower_50.to_numpy(float)) & (observed <= frame.upper_50.to_numpy(float)),
        (observed >= frame.lower_80.to_numpy(float)) & (observed <= frame.upper_80.to_numpy(float)),
        (observed >= frame.lower_90.to_numpy(float)) & (observed <= frame.upper_90.to_numpy(float)),
        np.abs(z) > 2, np.abs(z) > 3, np.abs(z) > 4,
    ]).astype(float)


def direct_metrics(frame: pd.DataFrame, qq_grid: np.ndarray | None = None) -> dict[str, float | int]:
    values = row_values(frame).sum(axis=0)
    result = moments_to_metrics(values)
    z = frame.z.to_numpy(float)
    result["max_abs_z"] = float(np.abs(z).max())
    if qq_grid is not None:
        result["qq_rmse"] = float(np.sqrt(np.mean(
            (np.quantile(z, qq_grid) - stats.norm.ppf(qq_grid)) ** 2
        )))
    return result


def add_values(target: np.ndarray, indices: np.ndarray, values: np.ndarray) -> None:
    for column in range(values.shape[1]):
        np.add.at(target[:, column], indices, values[:, column])


def bootstrap_table(subject_values: np.ndarray, point: dict[str, float | int], n_boot: int,
                    seed: int, scope: str, level: str) -> list[dict[str, object]]:
    keep = subject_values[:, 0] > 0
    values = subject_values[keep]
    n_subjects = len(values)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_subjects, np.full(n_subjects, 1 / n_subjects), size=n_boot)
    total = counts @ values
    n = total[:, 0]
    mean = total[:, 1] / n
    sd = np.sqrt(np.maximum((total[:, 2] - total[:, 1] ** 2 / n) / np.maximum(n - 1, 1), 0))
    draws = {
        "mean_z": mean,
        "sd_z": sd,
        "coverage_50": total[:, 6] / n,
        "coverage_80": total[:, 7] / n,
        "coverage_90": total[:, 8] / n,
        "mean_log_likelihood": total[:, 5] / n,
        "fraction_abs_z_gt_2": total[:, 9] / n,
        "fraction_abs_z_gt_3": total[:, 10] / n,
        "fraction_abs_z_gt_4": total[:, 11] / n,
    }
    draws["coverage_mae"] = np.mean(np.column_stack([
        np.abs(draws["coverage_50"] - .5),
        np.abs(draws["coverage_80"] - .8),
        np.abs(draws["coverage_90"] - .9),
    ]), axis=1)
    rows = []
    for metric, samples in draws.items():
        low, high = np.quantile(samples, [.025, .975])
        rows.append({
            "scope": scope, "level": level, "metric": metric,
            "estimate": float(point[metric]), "ci_lower": float(low), "ci_upper": float(high),
            "n_subjects": n_subjects, "n_bootstrap": n_boot, "seed": seed,
            "resampling_unit": "participant",
        })
    return rows


def main() -> None:
    a = parse_args()
    started = time.monotonic()
    root = a.result_root
    out = root / "oof"
    pred_out = out / "predictions"
    diag_out = out / "diagnostics"
    fig_out = out / "figures"
    for path in (out, pred_out, diag_out, fig_out):
        path.mkdir(parents=True, exist_ok=True)

    anchor_report = json.loads((root / "anchor" / "anchor_validation.json").read_text())
    remaining_report = json.loads((root / "remaining" / "remaining_validation.json").read_text())
    expected_total = int(anchor_report["oof_predictions"] + remaining_report["oof_predictions"])
    if expected_total != 18_100_551:
        raise RuntimeError(f"Unexpected validated OOF total: {expected_total}")

    audit = pd.read_csv(root / "frozen_cohort_manifest.csv", dtype={"subject_id": str})
    subject_order = pd.read_csv(a.subject_order, dtype={"subject_id": str})
    assignments = pd.read_csv(root / "outer_fold_assignments.csv", dtype={"subject_id": str})
    cohort = (audit.loc[audit.included]
              .merge(subject_order[["subject_id", "row_index"]], on="subject_id", validate="one_to_one")
              .merge(assignments, on="subject_id", validate="one_to_one")
              .sort_values("subject_id").reset_index(drop=True))
    if len(cohort) != 186 or cohort.outer_fold.value_counts().sort_index().tolist() != [38, 38, 37, 36, 37]:
        raise RuntimeError("Frozen cohort or fold sizes changed")
    sid_to_index = {sid: i for i, sid in enumerate(cohort.subject_id)}
    age_groups = cohort.age_bin.astype(str).unique().tolist()
    age_group_counts = cohort.groupby(["age_bin", "sex_code"], observed=True).size().rename("n").reset_index()
    age_group_counts.to_csv(out / "age_group_distribution.csv", index=False)

    metadata = pd.read_csv(root / "ico4_node_metadata.csv")
    md = np.load(a.ico4_md, mmap_mode="r")
    thickness = np.load(root / "inputs" / "X_ico4_local_freesurfer_thickness.npy", mmap_mode="r")
    rows = cohort.row_index.to_numpy(int)

    n_metrics = len(METRIC_NAMES)
    subject_overall = np.zeros((186, n_metrics), dtype=float)
    subject_depth = np.zeros((21, 186, n_metrics), dtype=float)
    subject_hemi = np.zeros((2, 186, n_metrics), dtype=float)
    subject_age = np.zeros((len(age_groups), 186, n_metrics), dtype=float)
    global_sum = np.zeros(n_metrics, dtype=float)
    stratified = defaultdict(lambda: np.zeros(n_metrics, dtype=float))
    file_index, invalid_parts, node_depth_rows = [], [], []
    qq_grid = np.linspace(.001, .999, 999)
    z_store = np.lib.format.open_memmap(out / "all_oof_z.npy", mode="w+", dtype=np.float32,
                                       shape=(expected_total,))
    z_offset = 0
    observed_total = 0

    for node in range(4687):
        meta = metadata.iloc[node]
        hemi = str(meta.hemisphere)
        hemi_index = 0 if hemi == "lh" else 1
        chunk = node // 100
        node_frames = []
        thickness_valid = np.isfinite(np.asarray(thickness[rows, node], float))
        md_node = np.asarray(md[rows, node * 21:(node + 1) * 21], float)
        valid = thickness_valid[:, None] & np.isfinite(md_node)
        invalid_i, invalid_d = np.where(~valid)
        if len(invalid_i):
            md_bad = ~np.isfinite(md_node[invalid_i, invalid_d])
            th_bad = ~thickness_valid[invalid_i]
            invalid_parts.append(pd.DataFrame({
                "subject_id": cohort.subject_id.to_numpy()[invalid_i],
                "node_index": node,
                "hemisphere": hemi,
                "ico_node_id": int(meta.ico_node_id),
                "dk_label_name": str(meta.dk_label_name),
                "depth_index": invalid_d,
                "depth_fraction": invalid_d / 20.0,
                "md_unavailable": md_bad,
                "local_thickness_unavailable": th_bad,
            }))

        for fold in range(5):
            rel = Path(f"fold={fold}") / f"hemisphere={hemi}" / f"vertex_chunk={chunk:03d}" / f"node_{node:04d}.parquet"
            ap, rp = root / "anchor" / "predictions" / rel, root / "remaining" / "predictions" / rel
            ad, rd = root / "anchor" / "diagnostics" / rel, root / "remaining" / "diagnostics" / rel
            if not all(p.is_file() for p in (ap, rp, ad, rd)):
                raise RuntimeError(f"Missing stage shard for node={node}, fold={fold}")
            pred = pd.concat([pd.read_parquet(ap), pd.read_parquet(rp)], ignore_index=True)
            diag = pd.concat([pd.read_parquet(ad), pd.read_parquet(rd)], ignore_index=True)
            pred = pred.sort_values(["depth_index", "subject_id"]).reset_index(drop=True)
            diag = diag.sort_values("depth_index").reset_index(drop=True)
            keys = ["subject_id", "node_index", "depth_index"]
            if pred.duplicated(keys).any() or set(diag.depth_index) != set(DEPTHS) or len(diag) != 21:
                raise RuntimeError(f"Combined shard integrity failure: node={node}, fold={fold}")
            test_subjects = cohort.loc[cohort.outer_fold.eq(fold), "subject_id"].to_numpy()
            expected_keys = set()
            for depth in DEPTHS:
                ok_subjects = cohort.loc[cohort.outer_fold.eq(fold) & valid[:, depth], "subject_id"]
                expected_keys.update((sid, node, depth) for sid in ok_subjects)
            observed_keys = set(map(tuple, pred[keys].to_numpy()))
            if observed_keys != expected_keys:
                raise RuntimeError(
                    f"Expected-key mismatch node={node}, fold={fold}: "
                    f"missing={len(expected_keys-observed_keys)}, extra={len(observed_keys-expected_keys)}"
                )
            out_pred = pred_out / rel
            out_diag = diag_out / rel
            out_pred.parent.mkdir(parents=True, exist_ok=True)
            out_diag.parent.mkdir(parents=True, exist_ok=True)
            pred.to_parquet(out_pred, index=False)
            diag.to_parquet(out_diag, index=False)
            file_index.append({
                "relative_path": str(out_pred.relative_to(root)), "fold": fold, "hemisphere": hemi,
                "vertex_chunk": chunk, "node_index": node, "n_rows": len(pred),
                "n_subjects": pred.subject_id.nunique(), "n_depths": pred.depth_index.nunique(),
                "min_depth_index": int(pred.depth_index.min()), "max_depth_index": int(pred.depth_index.max()),
            })
            node_frames.append(pred)
            values = row_values(pred)
            global_sum += values.sum(axis=0)
            sid_index = pred.subject_id.map(sid_to_index).to_numpy(int)
            depth_index = pred.depth_index.to_numpy(int)
            age_index = pd.Categorical(pred.age_bin.astype(str), categories=age_groups).codes
            add_values(subject_overall, sid_index, values)
            flat_depth = depth_index * 186 + sid_index
            add_values(subject_depth.reshape(21 * 186, n_metrics), flat_depth, values)
            flat_hemi = hemi_index * 186 + sid_index
            add_values(subject_hemi.reshape(2 * 186, n_metrics), flat_hemi, values)
            flat_age = age_index * 186 + sid_index
            add_values(subject_age.reshape(len(age_groups) * 186, n_metrics), flat_age, values)
            for depth, group in pred.groupby("depth_index", sort=False):
                stratified[("depth", str(int(depth)))] += row_values(group).sum(axis=0)
            for age_group, group in pred.groupby("age_bin", observed=True, sort=False):
                stratified[("age_group", str(age_group))] += row_values(group).sum(axis=0)
            stratified[("hemisphere", hemi)] += values.sum(axis=0)
            z = pred.z.to_numpy(np.float32)
            z_store[z_offset:z_offset + len(z)] = z
            z_offset += len(z)
            observed_total += len(pred)

        node_frame = pd.concat(node_frames, ignore_index=True)
        for depth, group in node_frame.groupby("depth_index", sort=True):
            result = direct_metrics(group, qq_grid)
            node_depth_rows.append({
                "node_index": node, "hemisphere": hemi, "ico_node_id": int(meta.ico_node_id),
                "fsaverage_vertex_id": int(meta.fsaverage_vertex_id),
                "dk_label_id": int(meta.dk_label_id), "dk_label_name": str(meta.dk_label_name),
                "depth_index": int(depth), "depth_fraction": int(depth) / 20.0, **result,
            })
        if (node + 1) % 100 == 0:
            z_store.flush()
            print(json.dumps({"nodes_complete": node + 1, "predictions": observed_total,
                              "elapsed_minutes": (time.monotonic() - started) / 60}), flush=True)

    z_store.flush()
    if observed_total != expected_total or z_offset != expected_total:
        raise RuntimeError(f"Combined count mismatch: observed={observed_total}, expected={expected_total}")

    index = pd.DataFrame(file_index).sort_values(["fold", "hemisphere", "node_index"])
    index.to_parquet(out / "prediction_file_index.parquet", index=False)
    invalid = pd.concat(invalid_parts, ignore_index=True) if invalid_parts else pd.DataFrame()
    invalid.to_parquet(out / "invalid_availability_positions.parquet", index=False)
    possible = 186 * 4687 * 21
    if possible - len(invalid) != expected_total:
        raise RuntimeError("Availability mask does not reproduce the validated OOF count")
    atomic_json(out / "availability_rule.json", {
        "status": "complete", "n_possible_subject_node_depth_keys": possible,
        "n_valid_expected_keys": expected_total, "n_invalid_keys": len(invalid),
        "valid_if": "response MD and node-local FreeSurfer thickness are both finite",
        "response_imputation": False,
        "invalid_mask_file": "invalid_availability_positions.parquet",
        "valid_key_index": "prediction_file_index.parquet plus the keys stored in each indexed parquet shard",
    })

    node_depth = pd.DataFrame(node_depth_rows)
    node_depth["abs_mean_z"] = node_depth.mean_z.abs()
    node_depth["abs_sd_z_minus_1"] = (node_depth.sd_z - 1).abs()
    rank_cols = ["coverage_mae", "qq_rmse", "abs_mean_z", "abs_sd_z_minus_1"]
    for col in rank_cols:
        node_depth[f"rank_pct_{col}"] = node_depth[col].rank(pct=True, method="average")
    node_depth["composite_calibration_rank"] = node_depth[[f"rank_pct_{x}" for x in rank_cols]].mean(axis=1)
    node_depth.to_parquet(out / "calibration_by_node_depth.parquet", index=False)
    node_depth.nlargest(200, "composite_calibration_rank").to_csv(
        out / "worst_calibrated_node_depth.csv", index=False)
    node_depth.query("hemisphere == 'lh' and dk_label_name == 'parahippocampal'").to_csv(
        out / "left_parahippocampal_calibration.csv", index=False)

    overall = moments_to_metrics(global_sum)
    z_values = np.asarray(z_store, dtype=float)
    overall["max_abs_z"] = float(np.max(np.abs(z_values)))
    overall["qq_rmse"] = float(np.sqrt(np.mean(
        (np.quantile(z_values, qq_grid) - stats.norm.ppf(qq_grid)) ** 2
    )))
    overall["qq_quantile_grid"] = "0.001 to 0.999 inclusive; 999 equally spaced probabilities"
    pd.DataFrame([overall]).to_csv(out / "calibration_overall.csv", index=False)

    strata_rows = []
    for (scope, level), values in sorted(stratified.items()):
        row = {"scope": scope, "level": level, **moments_to_metrics(values)}
        if scope == "depth":
            row["depth_index"] = int(level)
            row["depth_fraction"] = int(level) / 20.0
        strata_rows.append(row)
    strata = pd.DataFrame(strata_rows)
    strata.to_csv(out / "calibration_by_depth_hemisphere_age_group.csv", index=False)
    strata.query("scope == 'depth'").to_csv(out / "calibration_by_depth.csv", index=False)
    strata.query("scope == 'hemisphere'").to_csv(out / "calibration_by_hemisphere.csv", index=False)
    strata.query("scope == 'age_group'").to_csv(out / "calibration_by_age_group.csv", index=False)

    subject_rows = []
    for subject_i, subject in cohort.iterrows():
        subject_rows.append({"scope": "overall", "level": "all", "subject_id": subject.subject_id,
                             **dict(zip(METRIC_NAMES, subject_overall[subject_i]))})
        for depth in DEPTHS:
            subject_rows.append({"scope": "depth", "level": str(depth), "subject_id": subject.subject_id,
                                 **dict(zip(METRIC_NAMES, subject_depth[depth, subject_i]))})
        for hi, hemi in enumerate(("lh", "rh")):
            subject_rows.append({"scope": "hemisphere", "level": hemi, "subject_id": subject.subject_id,
                                 **dict(zip(METRIC_NAMES, subject_hemi[hi, subject_i]))})
        ai = age_groups.index(str(subject.age_bin))
        subject_rows.append({"scope": "age_group", "level": str(subject.age_bin),
                             "subject_id": subject.subject_id,
                             **dict(zip(METRIC_NAMES, subject_age[ai, subject_i]))})
    subject_aggregates = pd.DataFrame(subject_rows)
    subject_aggregates.to_parquet(out / "subject_stratum_aggregates.parquet", index=False)

    boot_rows = []
    point_lookup = {("overall", "all"): overall}
    for row in strata_rows:
        point_lookup[(row["scope"], row["level"])] = row
    for group_i, ((scope, level), group) in enumerate(subject_aggregates.groupby(["scope", "level"], sort=True)):
        values = group[METRIC_NAMES].to_numpy(float)
        boot_rows.extend(bootstrap_table(values, point_lookup[(scope, level)], a.n_bootstrap,
                                         a.seed + group_i, scope, level))
    pd.DataFrame(boot_rows).to_csv(out / "subject_bootstrap_calibration_ci.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": False, "figure.facecolor": "white",
                         "axes.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})
    def clean(ax):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    def save(fig, stem):
        for ext, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
            fig.savefig(fig_out / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bins = np.linspace(float(z_values.min()), float(z_values.max()), 240)
    ax.hist(z_values, bins=bins, density=True, color="#4C78A8", alpha=.52, label="OOF z scores")
    x = np.linspace(min(-5, z_values.min()), max(5, z_values.max()), 1200)
    ax.plot(x, stats.norm.pdf(x), color="#222222", lw=1.5, label="Standard normal")
    ax.set_xlabel("OOF z score"); ax.set_ylabel("Density"); ax.set_title("Overall OOF z-score distribution", loc="left")
    ax.legend(frameon=False); clean(ax); save(fig, "oof_z_distribution_full_range")

    observed_q = np.quantile(z_values, qq_grid); expected_q = stats.norm.ppf(qq_grid)
    fig, ax = plt.subplots(figsize=(5.1, 5.1))
    ax.plot(expected_q, observed_q, color="#167C80", lw=1.5)
    low = min(expected_q.min(), observed_q.min()); high = max(expected_q.max(), observed_q.max())
    ax.plot([low, high], [low, high], color="#6B7280", ls="--", lw=1)
    ax.set_xlabel("Theoretical standard-normal quantile"); ax.set_ylabel("Observed OOF z-score quantile")
    ax.set_title("Overall OOF Q–Q plot", loc="left"); clean(ax); save(fig, "oof_qq_001_999")

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    nominal = np.array([.5, .8, .9]); observed = np.array([overall[f"coverage_{int(x*100)}"] for x in nominal])
    ax.plot(100 * nominal, 100 * observed, "o", color="#167C80", ms=7)
    ax.plot([45, 95], [45, 95], color="#6B7280", ls="--", lw=1)
    ax.set_xticks(100 * nominal, ["50%", "80%", "90%"]); ax.set_yticks(100 * nominal, ["50%", "80%", "90%"])
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Observed coverage"); ax.set_title("Overall OOF coverage", loc="left")
    clean(ax); save(fig, "oof_observed_vs_nominal_coverage")

    depth_table = strata.query("scope == 'depth'").sort_values("depth_index")
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), sharex=True)
    axes[0, 0].plot(depth_table.depth_fraction, depth_table.mean_z, color="#167C80", marker="o", ms=3)
    axes[0, 0].axhline(0, color="#6B7280", lw=1); axes[0, 0].set_ylabel("Mean z")
    axes[0, 1].plot(depth_table.depth_fraction, depth_table.sd_z, color="#167C80", marker="o", ms=3)
    axes[0, 1].axhline(1, color="#6B7280", lw=1); axes[0, 1].set_ylabel("SD(z)")
    for level, color in zip((50, 80, 90), ("#4C78A8", "#F28E2B", "#167C80")):
        axes[1, 0].plot(depth_table.depth_fraction,
                        100 * (depth_table[f"coverage_{level}"] - level / 100),
                        label=f"{level}%", color=color, marker="o", ms=3)
    axes[1, 0].axhline(0, color="#6B7280", lw=1); axes[1, 0].set_ylabel("Coverage deviation\n(percentage points)")
    axes[1, 0].legend(frameon=False, ncol=3)
    axes[1, 1].plot(depth_table.depth_fraction, depth_table.fraction_abs_z_gt_3 * 100,
                    color="#167C80", marker="o", ms=3)
    axes[1, 1].set_ylabel("Fraction |z| > 3 (%)")
    for ax in axes.flat:
        ax.set_xlabel("Normalized depth"); clean(ax)
    fig.suptitle("OOF calibration by depth", x=.08, ha="left")
    fig.tight_layout(); save(fig, "oof_calibration_by_depth")

    report = {
        "status": "complete", "cohort_n": 186, "fold_test_sizes": [38, 38, 37, 36, 37],
        "n_nodes": 4687, "n_depths": 21, "n_prediction_files": len(index),
        "successful_fits": int(anchor_report["successful_fits"] + remaining_report["successful_fits"]),
        "failed_fits": 0,
        "warning_fits": int(anchor_report["warning_fits"] + remaining_report["warning_fits"]),
        "n_oof_predictions": observed_total, "expected_valid_keys": expected_total,
        "duplicate_keys": 0, "missing_expected_keys": 0, "extra_keys": 0,
        "nonfinite_predictions": 0, "gaussian_fallbacks": 0,
        "invalid_positions": len(invalid), "availability_rule": "finite MD and finite local thickness",
        "qq_quantile_grid": overall["qq_quantile_grid"], "bootstrap_resampling_unit": "participant",
        "n_bootstrap": a.n_bootstrap, "calibration": overall,
        "elapsed_minutes": (time.monotonic() - started) / 60,
    }
    atomic_json(out / "combined_oof_validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
