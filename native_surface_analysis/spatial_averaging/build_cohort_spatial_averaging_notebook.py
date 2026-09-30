from pathlib import Path

import nbformat as nbf


OUTPUT = Path(__file__).with_name("cohort_spatial_averaging_qc.ipynb")


LOAD = r'''
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

RESULT_ROOT = Path(
    "/mnt/newStor/paros/paros_WORK/xintong/results/native_surface_analysis"
)


def read_metric_files(paths):
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


fixed_paths = [
    RESULT_ROOT / "spatial_averaging_pilot/tables/metrics_summary.csv",
]
fixed_paths.extend(
    sorted(
        (
            RESULT_ROOT / "spatial_averaging_five_subjects/shards"
        ).glob("D*/tables/metrics_summary.csv")
    )
)
fixed_paths.extend(
    sorted(
        (
            RESULT_ROOT / "spatial_averaging_k64_k128/shards"
        ).glob("D*/tables/metrics_summary.csv")
    )
)
fixed_paths.extend(
    sorted(
        (
            RESULT_ROOT / "spatial_averaging_fixed_k_40/shards"
        ).glob("D*/tables/metrics_summary.csv")
    )
)

fixed = read_metric_files(fixed_paths)
fixed = fixed.drop_duplicates(
    subset=["subject", "hemisphere", "region", "method", "k", "repeat"]
)
fixed = fixed[fixed.status.eq("valid")].copy()
fixed = fixed[fixed.method.isin(["region", "spatial", "random"])]
fixed = fixed[fixed.k.isin([1, 2, 4, 8, 16, 32, 64, 128])]

target_paths = sorted(
    (RESULT_ROOT / "spatial_averaging_target_sizes").glob(
        "D*/tables/target_group_size_metrics.csv"
    )
)
target_paths += sorted(
    (RESULT_ROOT / "spatial_averaging_target_sizes_existing").glob(
        "D*/tables/target_group_size_metrics.csv"
    )
)
target = read_metric_files(target_paths)
target = target.drop_duplicates(
    subset=["subject", "hemisphere", "region", "target_group_size", "k"]
)
target = target[target.status.eq("valid")].copy()

connectivity_paths = [
    RESULT_ROOT / "spatial_averaging_pilot/tables/spatial_group_connectivity.csv",
]
connectivity_paths.extend(
    sorted(
        (
            RESULT_ROOT / "spatial_averaging_five_subjects/shards"
        ).glob("D*/tables/spatial_group_connectivity.csv")
    )
)
connectivity_paths.extend(
    sorted(
        (
            RESULT_ROOT / "spatial_averaging_k64_k128/shards"
        ).glob("D*/tables/spatial_group_connectivity.csv")
    )
)
connectivity_paths.extend(
    sorted(
        (
            RESULT_ROOT / "spatial_averaging_fixed_k_40/shards"
        ).glob("D*/tables/spatial_group_connectivity.csv")
    )
)
connectivity = read_metric_files(connectivity_paths)
connectivity = connectivity[
    connectivity.method.eq("spatial")
].copy()
connectivity = connectivity[
    connectivity.k.isin([1, 2, 4, 8, 16, 32, 64, 128])
].copy()

print(f"Fixed-K subjects: {fixed.subject.nunique()}")
print(f"Target-size subjects: {target.subject.nunique()}")
print(f"Fixed-K rows: {len(fixed):,}")
print(f"Target-size rows: {len(target):,}")
display(target.groupby("target_group_size")["subject"].nunique())
'''


SUBJECT_METRICS = r'''
keys = ["subject", "hemisphere", "region", "method", "k"]
subject_metrics = fixed.groupby(keys, as_index=False).agg(
    mae=("mae", "median"),
    rmse=("rmse", "median"),
)
baseline = subject_metrics[
    subject_metrics.method.eq("region") & subject_metrics.k.eq(1)
].rename(columns={"mae": "baseline_mae", "rmse": "baseline_rmse"})[
    ["subject", "hemisphere", "region", "baseline_mae", "baseline_rmse"]
]
subject_metrics = subject_metrics.merge(
    baseline,
    on=["subject", "hemisphere", "region"],
    how="left",
)
subject_metrics["mae_improvement_pct"] = 100 * (
    subject_metrics.baseline_mae - subject_metrics.mae
) / subject_metrics.baseline_mae
subject_metrics["rmse_improvement_pct"] = 100 * (
    subject_metrics.baseline_rmse - subject_metrics.rmse
) / subject_metrics.baseline_rmse
display(subject_metrics.head())
'''


MAE_CURVES = r'''
plot_data = subject_metrics[subject_metrics.k.le(32)].copy()
plot_data = plot_data.groupby(
    ["subject", "region", "method", "k"], as_index=False
).agg(mae=("mae", "median"))
regions = sorted(plot_data.region.unique())
method_styles = {
    "region": ("#777777", "--", "Baseline"),
    "spatial": ("#0072B2", "-", "Spatial grouping"),
    "random": ("#D55E00", "-", "Random grouping"),
}

fig, axes = plt.subplots(
    1,
    len(regions),
    figsize=(4 * len(regions), 3.4),
    squeeze=False,
)
for ax, region in zip(axes.ravel(), regions):
    region_data = plot_data[plot_data.region.eq(region)]
    for method, (color, linestyle, label) in method_styles.items():
        method_data = region_data[region_data.method.eq(method)]
        median = method_data.groupby("k", as_index=False).mae.median()
        ax.plot(
            median.k,
            median.mae,
            color=color,
            linestyle=linestyle,
            marker="o",
            linewidth=1.6,
            label=label,
        )
    ax.set_title(region)
    ax.set_xlabel("K")
    ax.set_ylabel("Subject-level median MAE")
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xlim(0.5, 32.5)
    ax.spines[["top", "right"]].set_visible(False)
axes.ravel()[0].legend(frameon=False, fontsize=7)
fig.suptitle(
    "Thin lines: subject-level median; thick lines: across-subject median",
    fontsize=9,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
'''


IMPROVEMENT_CURVES = r'''
plot_data = subject_metrics[
    subject_metrics.method.isin(["spatial", "random"])
].copy()
plot_data = plot_data.groupby(
    ["subject", "region", "method", "k"], as_index=False
).agg(mae_improvement_pct=("mae_improvement_pct", "median"))
regions = sorted(plot_data.region.unique())
method_styles = {
    "spatial": ("#0072B2", "Spatial grouping"),
    "random": ("#D55E00", "Random grouping"),
}

fig, axes = plt.subplots(
    1,
    len(regions),
    figsize=(4 * len(regions), 3.4),
    squeeze=False,
)
for ax, region in zip(axes.ravel(), regions):
    region_data = plot_data[plot_data.region.eq(region)]
    for method, (color, label) in method_styles.items():
        method_data = region_data[
            region_data.method.eq(method) & region_data.k.le(32)
        ]
        for subject, group in method_data.groupby("subject"):
            group = group.sort_values("k")
            ax.plot(group.k, group.mae_improvement_pct, color=color, alpha=0.2)
        median = method_data.groupby("k", as_index=False).mae_improvement_pct.median()
        ax.plot(
            median.k,
            median.mae_improvement_pct,
            color=color,
            marker="o",
            linewidth=1.6,
            label=label,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(region)
    ax.set_xlabel("K")
    ax.set_ylabel("MAE improvement vs baseline (%)")
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.set_xlim(1.5, 32.5)
    ax.spines[["top", "right"]].set_visible(False)
axes.ravel()[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
plt.show()

'''


SPATIAL_IMPROVEMENT = r'''
summary = subject_metrics.groupby(
    ["region", "method", "k"], as_index=False
).agg(
    n_subjects=("subject", "nunique"),
    mae_median=("mae", "median"),
    mae_q25=("mae", lambda values: values.quantile(0.25)),
    mae_q75=("mae", lambda values: values.quantile(0.75)),
    improvement_median=("mae_improvement_pct", "median"),
    improvement_q25=("mae_improvement_pct", lambda values: values.quantile(0.25)),
    improvement_q75=("mae_improvement_pct", lambda values: values.quantile(0.75)),
)
spatial = summary[
    summary.method.eq("spatial") & summary.k.le(32)
].copy()
fig, ax = plt.subplots(figsize=(6.5, 3.8))
sns.lineplot(
    data=spatial,
    x="k",
    y="improvement_median",
    hue="region",
    marker="o",
    ax=ax,
)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("K")
ax.set_ylabel("MAE improvement vs baseline (%)")
ax.set_xticks([2, 4, 8, 16, 32])
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()

'''


TARGET_SIZE = r'''
target_subject = target.groupby(
    ["subject", "hemisphere", "region", "target_group_size"],
    as_index=False,
).agg(mae=("mae", "median"), rmse=("rmse", "median"))
target_baseline = subject_metrics[
    subject_metrics.method.eq("region") & subject_metrics.k.eq(1)
][
    ["subject", "hemisphere", "region", "baseline_mae"]
].drop_duplicates()
target_subject = target_subject.merge(
    target_baseline,
    on=["subject", "hemisphere", "region"],
    how="left",
)
target_subject["mae_improvement_pct"] = 100 * (
    target_subject.baseline_mae - target_subject.mae
) / target_subject.baseline_mae

target_summary = target_subject.groupby(
    ["region", "target_group_size"], as_index=False
).agg(
    n_subjects=("subject", "nunique"),
    n_subjects_improvement=("mae_improvement_pct", lambda values: values.notna().sum()),
    mae_median=("mae", "median"),
    mae_q25=("mae", lambda values: values.quantile(0.25)),
    mae_q75=("mae", lambda values: values.quantile(0.75)),
    improvement_median=("mae_improvement_pct", "median"),
    improvement_q25=("mae_improvement_pct", lambda values: values.quantile(0.25)),
    improvement_q75=("mae_improvement_pct", lambda values: values.quantile(0.75)),
)
display(target_summary.round(4))

regions = sorted(target_summary.region.unique())
fig, axes = plt.subplots(
    1,
    len(regions),
    figsize=(4 * len(regions), 3.4),
    squeeze=False,
)
for ax, region in zip(axes.ravel(), regions):
    subject_part = target_subject[target_subject.region.eq(region)]
    for subject, group in subject_part.groupby("subject"):
        group = group.sort_values("target_group_size")
        ax.plot(
            group.target_group_size,
            group.mae,
            color="#0072B2",
            alpha=0.2,
            linewidth=0.7,
        )
    part = target_summary[target_summary.region.eq(region)]
    ax.plot(
        part.target_group_size,
        part.mae_median,
        color="#0072B2",
        marker="o",
        linewidth=1.8,
        label="Across-subject median",
    )
    ax.set_title(region)
    ax.set_xlabel("Target columns per spatial group")
    ax.set_ylabel("Subject-level median MAE")
    ax.set_xticks(sorted(target_summary.target_group_size.unique()))
    ax.spines[["top", "right"]].set_visible(False)
axes.ravel()[0].legend(frameon=False, fontsize=7)
fig.suptitle(
    "Thin lines: subject-level results; thick line: across-subject median",
    fontsize=9,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

regions = sorted(target_summary.region.unique())
fig, axes = plt.subplots(
    1,
    len(regions),
    figsize=(4 * len(regions), 3.4),
    squeeze=False,
)
for ax, region in zip(axes.ravel(), regions):
    subject_part = target_subject[target_subject.region.eq(region)]
    for subject, group in subject_part.groupby("subject"):
        group = group.sort_values("target_group_size")
        ax.plot(
            group.target_group_size,
            group.mae_improvement_pct,
            color="#0072B2",
            alpha=0.2,
            linewidth=0.7,
        )
    part = target_summary[target_summary.region.eq(region)]
    ax.plot(
        part.target_group_size,
        part.improvement_median,
        color="#0072B2",
        marker="o",
        linewidth=1.8,
        label="Across-subject median",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(region)
    ax.set_xlabel("Target columns per spatial group")
    ax.set_ylabel("MAE improvement vs whole-region baseline (%)")
    ax.set_xticks(sorted(target_summary.target_group_size.unique()))
    ax.spines[["top", "right"]].set_visible(False)
axes.ravel()[0].legend(frameon=False, fontsize=7)
fig.suptitle(
    "Thin lines: subject-level results; thick line: across-subject median",
    fontsize=9,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
'''


GROUP_SIZE = r'''
group_size_summary = connectivity.groupby(
    ["region", "k"], as_index=False
).agg(
    n_subjects=("subject", "nunique"),
    n_groups=("n_columns", "size"),
    mean_columns_per_group=("n_columns", "mean"),
    median_columns_per_group=("n_columns", "median"),
    q25_columns_per_group=("n_columns", lambda values: values.quantile(0.25)),
    q75_columns_per_group=("n_columns", lambda values: values.quantile(0.75)),
    min_columns_per_group=("n_columns", "min"),
    max_columns_per_group=("n_columns", "max"),
)
display(group_size_summary.round(2))

group_size_long = group_size_summary.melt(
    id_vars=["region", "k"],
    value_vars=[
        "mean_columns_per_group",
        "median_columns_per_group",
        "q25_columns_per_group",
        "q75_columns_per_group",
    ],
    var_name="statistic",
    value_name="columns",
)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
sns.lineplot(
    data=group_size_long[group_size_long.statistic.eq("median_columns_per_group")],
    x="k",
    y="columns",
    hue="region",
    marker="o",
    ax=ax,
)
ax.set_xlabel("K")
ax.set_ylabel("Median columns per spatial group")
ax.set_xticks(sorted(connectivity.k.unique()))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), squeeze=False)
for ax, k_values, label in zip(
    axes.ravel(),
    ([1, 2, 4], [8, 16, 32], [32, 64, 128]),
    ("K = 1–4", "K = 8–32", "K = 32–128"),
):
    plot_data = group_size_long[
        group_size_long.statistic.eq("median_columns_per_group")
        & group_size_long.k.isin(k_values)
    ]
    sns.lineplot(
        data=plot_data,
        x="k",
        y="columns",
        hue="region",
        marker="o",
        ax=ax,
    )
    ax.set_title(label)
    ax.set_xlabel("K")
    ax.set_ylabel("Median columns per spatial group")
    ax.set_xticks(k_values)
    ax.spines[["top", "right"]].set_visible(False)
    if ax is not axes.ravel()[0]:
        ax.get_legend().remove()
axes.ravel()[0].legend(frameon=False, fontsize=7)
fig.tight_layout(pad=1.2)
plt.show()

'''


SCATTER = r'''
group_size_by_unit = connectivity.groupby(
    ["subject", "hemisphere", "region", "k"],
    as_index=False,
).agg(
    median_columns_per_group=("n_columns", "median"),
)
spatial_improvement = subject_metrics[
    subject_metrics.method.eq("spatial")
][
    ["subject", "hemisphere", "region", "k", "mae_improvement_pct"]
].drop_duplicates()
scatter_data = group_size_by_unit.merge(
    spatial_improvement,
    on=["subject", "hemisphere", "region", "k"],
    how="inner",
)
fixed_k_values = [1, 2, 4, 8, 16, 32, 64, 128]
scatter_data = scatter_data[scatter_data.k.isin(fixed_k_values)].copy()
scatter_data["k_label"] = scatter_data.k.astype(int).astype(str)
print("Scatter K values:", sorted(scatter_data.k.unique()))

plt.close("all")
fig, ax = plt.subplots(figsize=(6.5, 3.8))
sns.scatterplot(
    data=scatter_data,
    x="median_columns_per_group",
    y="mae_improvement_pct",
    hue="k_label",
    hue_order=[str(value) for value in fixed_k_values],
    palette=dict(
        zip(
            [str(value) for value in fixed_k_values],
            sns.color_palette("viridis", n_colors=len(fixed_k_values)),
        )
    ),
    s=22,
    alpha=0.65,
    edgecolor="none",
    ax=ax,
)
if len(scatter_data) >= 2:
    sns.regplot(
        data=scatter_data,
        x="median_columns_per_group",
        y="mae_improvement_pct",
        scatter=False,
        color="#333333",
        line_kws={"linewidth": 1.2, "linestyle": "--"},
        ax=ax,
    )
ax.set_xlabel("Actual median columns per spatial group")
ax.set_ylabel("MAE improvement vs whole-region baseline (%)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
'''


cells = [
    nbf.v4.new_markdown_cell(
        "# Cohort spatial averaging QC\n\n"
        "This notebook combines fixed-K spatial averaging and target-group-size "
        "results. Subject-level summaries are computed before cohort summaries."
    ),
    nbf.v4.new_code_cell(LOAD),
    nbf.v4.new_markdown_cell("## Subject-level fixed-K metrics"),
    nbf.v4.new_code_cell(SUBJECT_METRICS),
    nbf.v4.new_markdown_cell("## Subject-level MAE curves through K=32"),
    nbf.v4.new_code_cell(MAE_CURVES),
    nbf.v4.new_markdown_cell("## Subject-level improvement curves"),
    nbf.v4.new_code_cell(IMPROVEMENT_CURVES),
    nbf.v4.new_markdown_cell("## Spatial grouping improvement"),
    nbf.v4.new_code_cell(SPATIAL_IMPROVEMENT),
    nbf.v4.new_markdown_cell("## Fixed-column-count grouping"),
    nbf.v4.new_code_cell(TARGET_SIZE),
    nbf.v4.new_markdown_cell("## Number of columns per K-based spatial group"),
    nbf.v4.new_code_cell(GROUP_SIZE),
    nbf.v4.new_markdown_cell("## Actual group size versus improvement"),
    nbf.v4.new_code_cell(SCATTER),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
OUTPUT.write_text(nbf.writes(notebook), encoding="utf-8")
print(OUTPUT)
