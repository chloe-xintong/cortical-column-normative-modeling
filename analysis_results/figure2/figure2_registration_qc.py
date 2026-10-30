#!/usr/bin/env python3

"""Summarize existing bbregister minimum-cost outputs for Figure 2C."""

import argparse
import shlex
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VISUAL_QC_COLUMNS = [
    "subject_id",
    "visual_qc_status",
    "visual_qc_reason",
    "reviewer",
    "review_date",
    "final_registration_status",
    "final_exclusion_reason",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Figure 2C T1-to-DWI registration QC tables and plots."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robust-z-threshold", type=float, default=3.5)
    parser.add_argument("--top-review-count", type=int, default=10)
    return parser.parse_args()


def parse_mincost(path):
    raw_text = path.read_text(encoding="utf-8").strip()
    tokens = raw_text.split()
    if len(tokens) != 4:
        return raw_text, None

    try:
        values = np.asarray([float(token) for token in tokens], dtype=float)
    except ValueError:
        return raw_text, None

    if not np.isfinite(values).all() or not 0 <= values[0] <= 1:
        return raw_text, None

    return raw_text, values


def inspect_subject(row):
    subject_id = row["subject_id"]
    registration_dir = Path(row["registration_dir"])
    transform_path = Path(row["transform_path"])
    moving_image_path = Path(row["moving_image_path"])
    freesurfer_subject_dir = Path(row["freesurfer_subject_dir"])
    expected_mincost = Path(f"{transform_path}.mincost")

    transform_exists = transform_path.is_file() and transform_path.stat().st_size > 0
    moving_image_exists = (
        moving_image_path.is_file() and moving_image_path.stat().st_size > 0
    )
    lh_white_exists = (freesurfer_subject_dir / "surf" / "lh.white").is_file()
    rh_white_exists = (freesurfer_subject_dir / "surf" / "rh.white").is_file()

    mincost_path = expected_mincost
    status = "valid"
    if not transform_exists:
        status = "missing_transform"
    elif not expected_mincost.is_file():
        candidates = sorted(registration_dir.glob("*.mincost"))
        if len(candidates) == 0:
            status = "missing_mincost"
        elif len(candidates) == 1:
            mincost_path = candidates[0]
        else:
            status = "ambiguous_mincost"

    raw_text = ""
    values = None
    if status == "valid":
        raw_text, values = parse_mincost(mincost_path)
        if values is None:
            status = "invalid_mincost"

    if status == "valid" and not moving_image_exists:
        status = "missing_moving_image"
    if status == "valid" and not (lh_white_exists and rh_white_exists):
        status = "missing_freesurfer_surface"

    result = row.to_dict()
    result.update(
        {
            "transform_path": str(transform_path),
            "mincost_path": str(mincost_path),
            "moving_image_path": str(moving_image_path),
            "freesurfer_subject_dir": str(freesurfer_subject_dir),
            "transform_exists": transform_exists,
            "mincost_exists": mincost_path.is_file(),
            "moving_image_exists": moving_image_exists,
            "lh_white_exists": lh_white_exists,
            "rh_white_exists": rh_white_exists,
            "mincost_valid": status == "valid",
            "registration_output_status": status,
            "mincost_raw_text": raw_text,
            "bbregister_mincost": np.nan,
            "mean_white_signal": np.nan,
            "mean_gray_signal": np.nan,
            "percent_contrast": np.nan,
        }
    )
    if values is not None:
        (
            result["bbregister_mincost"],
            result["mean_white_signal"],
            result["mean_gray_signal"],
            result["percent_contrast"],
        ) = values
    return result


def add_robust_outliers(table, threshold):
    table = table.copy()
    valid = table.loc[table["mincost_valid"]].copy()

    use_protocol = (
        "protocol" in valid.columns
        and valid["protocol"].fillna("").ne("").all()
        and valid["protocol"].nunique() > 1
        and valid.groupby("protocol").size().min() >= 10
    )
    group_column = "protocol" if use_protocol else "cohort"
    table["outlier_scope"] = group_column
    table["outlier_group"] = table[group_column].fillna("not_available")
    table["robust_z"] = np.nan

    for group_name, indices in table.loc[table["mincost_valid"]].groupby(
        "outlier_group"
    ).groups.items():
        costs = table.loc[indices, "bbregister_mincost"]
        median = costs.median()
        mad = np.median(np.abs(costs - median))
        if mad > 0:
            table.loc[indices, "robust_z"] = 0.6745 * (costs - median) / mad

    table["automatic_cost_outlier"] = table["robust_z"].gt(threshold)
    return table


def load_visual_qc(path, subjects):
    if path.is_file():
        return pd.read_csv(path, dtype={"subject_id": str})

    visual_qc = pd.DataFrame({"subject_id": subjects})
    visual_qc["visual_qc_status"] = "not_reviewed"
    visual_qc["visual_qc_reason"] = ""
    visual_qc["reviewer"] = ""
    visual_qc["review_date"] = ""
    visual_qc["final_registration_status"] = "not_reviewed"
    visual_qc["final_exclusion_reason"] = ""
    visual_qc.to_csv(path, index=False)
    return visual_qc


def add_manual_status(table, visual_qc):
    table = table.merge(
        visual_qc[VISUAL_QC_COLUMNS],
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    text_columns = VISUAL_QC_COLUMNS[1:]
    table[text_columns] = table[text_columns].fillna("")

    table["plot_status"] = "not reviewed"
    included = table["visual_qc_status"].eq("pass") | table[
        "final_registration_status"
    ].eq("included")
    review = table["visual_qc_status"].eq("review") | table[
        "final_registration_status"
    ].eq("review")
    excluded = table["visual_qc_status"].eq("fail") | table[
        "final_registration_status"
    ].eq("excluded")
    table.loc[included, "plot_status"] = "included/pass"
    table.loc[review, "plot_status"] = "review"
    table.loc[excluded, "plot_status"] = "excluded/fail"
    return table


def summarize_costs(table, threshold):
    valid = table.loc[table["mincost_valid"], "bbregister_mincost"]
    summary = {
        "n_expected_subjects": len(table),
        "n_with_final_transform": int(table["transform_exists"].sum()),
        "n_with_readable_mincost": int(table["mincost_valid"].sum()),
        "n_missing_transform": int(
            table["registration_output_status"].eq("missing_transform").sum()
        ),
        "n_missing_mincost": int(
            table["registration_output_status"].eq("missing_mincost").sum()
        ),
        "n_ambiguous_mincost": int(
            table["registration_output_status"].eq("ambiguous_mincost").sum()
        ),
        "n_invalid_mincost": int(
            table["registration_output_status"].eq("invalid_mincost").sum()
        ),
        "median_cost": valid.median(),
        "q25_cost": valid.quantile(0.25),
        "q75_cost": valid.quantile(0.75),
        "minimum_cost": valid.min(),
        "maximum_cost": valid.max(),
        "robust_z_threshold": threshold,
        "n_automatic_cost_outliers": int(table["automatic_cost_outlier"].sum()),
        "n_visually_reviewed": int(
            table["visual_qc_status"].isin(["pass", "review", "fail"]).sum()
        ),
        "n_excluded": int(table["plot_status"].eq("excluded/fail").sum()),
        "n_missing_or_invalid_cost": int((~table["mincost_valid"]).sum()),
        "n_outliers_excluded": int(
            (table["automatic_cost_outlier"] & table["plot_status"].eq("excluded/fail"))
            .sum()
        ),
    }
    return pd.DataFrame([summary])


def summarize_by_protocol(table):
    valid = table.loc[table["mincost_valid"]].copy()
    if "protocol" not in valid.columns or valid["protocol"].fillna("").eq("").all():
        valid["protocol"] = "not_available"

    summary = (
        valid.groupby(["cohort", "protocol"], dropna=False)[
            "bbregister_mincost"
        ]
        .agg(
            n_subjects="size",
            median_cost="median",
            q25_cost=lambda values: values.quantile(0.25),
            q75_cost=lambda values: values.quantile(0.75),
            minimum_cost="min",
            maximum_cost="max",
        )
        .reset_index()
    )
    return summary


def build_review_queue(table, top_review_count):
    highest_cost_subjects = set(
        table.loc[table["mincost_valid"]]
        .nlargest(top_review_count, "bbregister_mincost")["subject_id"]
    )
    rows = []
    for row in table.itertuples(index=False):
        reasons = []
        if row.automatic_cost_outlier:
            reasons.append("automatic_cost_outlier")
        if row.subject_id in highest_cost_subjects:
            reasons.append(f"top_{top_review_count}_cost")
        if row.registration_output_status != "valid":
            reasons.append(row.registration_output_status)
        if row.visual_qc_status in {"review", "fail"}:
            reasons.append(f"visual_qc_{row.visual_qc_status}")
        if row.final_registration_status == "excluded":
            reasons.append("final_registration_excluded")
        if not reasons:
            continue

        rows.append(
            {
                "subject_id": row.subject_id,
                "cost": row.bbregister_mincost,
                "robust_z": row.robust_z,
                "automatic_cost_outlier": row.automatic_cost_outlier,
                "visual_qc_status": row.visual_qc_status,
                "reason_for_review": ";".join(reasons),
                "transform_path": row.transform_path,
                "moving_image_path": row.moving_image_path,
                "freesurfer_subject_dir": row.freesurfer_subject_dir,
            }
        )
    return pd.DataFrame(rows)


def write_freeview_commands(review_queue, path):
    lines = ["#!/usr/bin/env bash", "", "set -euo pipefail", ""]
    for row in review_queue.itertuples(index=False):
        freesurfer_dir = Path(row.freesurfer_subject_dir)
        t1 = freesurfer_dir / "mri" / "orig.mgz"
        lh_white = freesurfer_dir / "surf" / "lh.white"
        rh_white = freesurfer_dir / "surf" / "rh.white"
        command = [
            "freeview",
            "-v",
            f"{t1}:visible=0",
            f"{row.moving_image_path}:reg={row.transform_path}",
            "-f",
            str(lh_white),
            str(rh_white),
        ]
        lines.append(f"# {row.subject_id}: {row.reason_for_review}")
        lines.append(" ".join(shlex.quote(item) for item in command))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_registration_cost(table, summary, output_dir, threshold):
    valid = table.loc[table["mincost_valid"]].sort_values(
        "bbregister_mincost"
    ).copy()
    valid["rank"] = np.arange(1, len(valid) + 1)
    stats = summary.iloc[0]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    styles = {
        "included/pass": ("#0072B2", "o"),
        "review": ("#E69F00", "o"),
        "not reviewed": ("#8C8C8C", "o"),
        "excluded/fail": ("#D55E00", "x"),
    }
    for status, (color, marker) in styles.items():
        part = valid.loc[valid["plot_status"].eq(status)]
        if part.empty:
            continue
        ax.scatter(
            part["rank"],
            part["bbregister_mincost"],
            color=color,
            marker=marker,
            s=24,
            linewidth=1.0,
            label=status,
            zorder=3,
        )

    outliers = valid.loc[valid["automatic_cost_outlier"]]
    ax.scatter(
        outliers["rank"],
        outliers["bbregister_mincost"],
        facecolors="none",
        edgecolors="black",
        s=70,
        linewidth=1.0,
        label="automatic cost outlier",
        zorder=4,
    )

    labels = valid.loc[
        valid["automatic_cost_outlier"]
        | valid["plot_status"].eq("excluded/fail")
    ].sort_values("bbregister_mincost")
    label_y = labels["bbregister_mincost"].to_numpy(copy=True)
    minimum_gap = 0.012
    for index in range(1, len(label_y)):
        label_y[index] = max(label_y[index], label_y[index - 1] + minimum_gap)

    label_x = len(valid) + 8
    for row, y_position in zip(labels.itertuples(index=False), label_y):
        ax.annotate(
            row.subject_id,
            (row.rank, row.bbregister_mincost),
            xytext=(label_x, y_position),
            textcoords="data",
            arrowprops={"arrowstyle": "-", "color": "#666666", "linewidth": 0.5},
            va="center",
            fontsize=7,
        )

    median = stats["median_cost"]
    q25 = stats["q25_cost"]
    q75 = stats["q75_cost"]
    mad = np.median(np.abs(valid["bbregister_mincost"] - median))
    robust_cost_threshold = median + threshold * mad / 0.6745

    ax.axhspan(q25, q75, color="#0072B2", alpha=0.10, label="cohort IQR")
    ax.axhline(median, color="#0072B2", linestyle="--", linewidth=1.2)
    ax.axhline(
        robust_cost_threshold,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"robust z = {threshold:g}",
    )
    ax.axhline(
        0.5,
        color="#CC79A7",
        linestyle=":",
        linewidth=1.0,
        label="0.5 tutorial reference only",
    )
    ax.set(
        title="T1–DWI Boundary-Based Registration Quality Across Subjects",
        xlabel="Subjects ranked by boundary-based registration cost",
        ylabel="bbregister minimum cost\n(Lower is better)",
        xlim=(0, len(valid) + 38),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.text(
        0.01,
        0.98,
        (
            f"N valid={int(stats['n_with_readable_mincost'])}; "
            f"outliers={int(stats['n_automatic_cost_outliers'])}; "
            f"reviewed={int(stats['n_visually_reviewed'])}; "
            f"excluded={int(stats['n_excluded'])}; "
            f"missing/invalid={int(stats['n_missing_or_invalid_cost'])}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "Figure2C_T1_DWI_registration_cost.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "Figure2C_T1_DWI_registration_cost.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = parse_args()
    tables_dir = args.output_dir / "tables"
    qc_dir = args.output_dir / "qc"
    figures_dir = args.output_dir / "figures"
    for directory in (tables_dir, qc_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Reading registration cohort: {args.manifest}", flush=True)
    manifest = pd.read_csv(args.manifest, dtype={"subject_id": str})
    print(f"Inspecting {len(manifest)} subjects", flush=True)
    records = []
    for index, (_, row) in enumerate(manifest.iterrows(), start=1):
        records.append(inspect_subject(row))
        if index == 1 or index % 25 == 0 or index == len(manifest):
            print(f"  inspected {index}/{len(manifest)}", flush=True)

    table = pd.DataFrame(records)
    table = add_robust_outliers(table, args.robust_z_threshold)
    visual_qc_path = qc_dir / "registration_visual_qc.csv"
    visual_qc = load_visual_qc(visual_qc_path, table["subject_id"])
    table = add_manual_status(table, visual_qc)

    summary = summarize_costs(table, args.robust_z_threshold)
    protocol_summary = summarize_by_protocol(table)
    review_queue = build_review_queue(table, args.top_review_count)

    table.to_csv(
        tables_dir / "figure2C_registration_cost_by_subject.csv",
        index=False,
    )
    summary.to_csv(
        tables_dir / "figure2C_registration_cost_summary.csv",
        index=False,
    )
    protocol_summary.to_csv(
        tables_dir / "figure2C_registration_cost_by_protocol.csv",
        index=False,
    )
    table.to_csv(qc_dir / "figure2C_registration_input_qc.csv", index=False)
    table.loc[~table["mincost_valid"]].to_csv(
        qc_dir / "figure2C_registration_missing_outputs.csv",
        index=False,
    )
    review_queue.to_csv(
        qc_dir / "registration_visual_review_queue.csv",
        index=False,
    )
    write_freeview_commands(
        review_queue,
        qc_dir / "freeview_registration_review_commands.sh",
    )
    plot_registration_cost(
        table,
        summary,
        figures_dir,
        args.robust_z_threshold,
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Outputs written to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
