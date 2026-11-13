#!/usr/bin/env python3
"""Build the Figure 6 model-selection and calibration plotting notebook.

Description
-----------
Build the Figure 6 model-selection and calibration plotting notebook.

Methods
-------
1. Provide the reusable computations defined in this module.
2. Return results to the calling analysis code.

Inputs
------
- Data supplied through the module's public function arguments.

Outputs
-------
- In-memory values returned to the calling code.

Usage
-----
Import this module and call ``markdown(...)``; it has no command-line entry point.
"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_PATH = NOTEBOOK_DIR / "figure6_DK_region_model_selection_186.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        """
# Figure 6: model selection and calibration

This notebook generates three presentation-ready figures for the frozen 186-participant CN
cohort. It reads completed regional summaries, existing participant-bootstrap comparisons, and
validated ico4 stage summaries. It does **not** refit a normative model or repeat a bootstrap.

All figures use English text. Plotting is summary-based, while the source summaries themselves
were computed from the complete formal OOF predictions.
"""
    ),
    markdown("## 1. Data sources"),
    code(
        """
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
elif ROOT.name != "figure6":
    candidate = ROOT / "analysis_results" / "figure6"
    if candidate.is_dir():
        ROOT = candidate
    else:
        raise RuntimeError("Run from the repository root, result root, or notebook directory.")

REGIONAL = ROOT / "regional_warped_m2_186"
ICO4 = ROOT / "ico4_warped_m2_186"
FIGURE_DIR = ROOT / "notebook_figures" / "model_selection_calibration_186"
TABLE_DIR = ROOT / "plotting_inputs" / "model_selection_calibration_186"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

source_paths = {
    "regional_model_bootstrap": REGIONAL / "model_comparison_subject_paired_bootstrap.csv",
    "regional_distribution_summary": REGIONAL / "gaussian_M2_vs_warped_M2.csv",
    "regional_distribution_bootstrap": REGIONAL / "gaussian_vs_warped_subject_paired_bootstrap.csv",
    "regional_validation": REGIONAL / "unified_model_validation.json",
    "ico4_anchor_validation": ICO4 / "anchor" / "anchor_validation.json",
    "ico4_remaining_validation": ICO4 / "remaining" / "remaining_validation.json",
    "ico4_preflight": ICO4 / "preflight_report.json",
    "ico4_cohort": ICO4 / "frozen_cohort_manifest.csv",
}
missing = [str(path) for path in source_paths.values() if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing required formal result files:\\n" + "\\n".join(missing))

regional_boot = pd.read_csv(source_paths["regional_model_bootstrap"])
distribution = pd.read_csv(source_paths["regional_distribution_summary"])
distribution_boot = pd.read_csv(source_paths["regional_distribution_bootstrap"])
regional_validation = json.loads(source_paths["regional_validation"].read_text())
anchor = json.loads(source_paths["ico4_anchor_validation"].read_text())
remaining = json.loads(source_paths["ico4_remaining_validation"].read_text())
preflight = json.loads(source_paths["ico4_preflight"].read_text())
cohort = pd.read_csv(source_paths["ico4_cohort"])

pd.DataFrame(
    {"source": source_paths.keys(), "path": [str(p) for p in source_paths.values()]}
)
"""
    ),
    markdown("## 2. Metric verification"),
    code(
        """
# Cohort and integrity checks
assert regional_validation["status"] == "complete"
assert regional_validation["cohort_n"] == 186
assert regional_validation["folds"] == 5
assert regional_validation["failed_fits"] == 0
assert anchor["status"] == remaining["status"] == "complete"
assert anchor["failed_fits"] == remaining["failed_fits"] == 0
assert anchor["gaussian_fallbacks"] == remaining["gaussian_fallbacks"] == 0
assert preflight["cohort_n"] == int(cohort["included"].sum()) == 186
assert preflight["n_nodes"] == 4687
assert preflight["n_depths"] == 21

# Figure 1 input: use existing participant-bootstrap estimates and CIs.
contrast_order = ["M1-M0", "M3-M2", "M2-M0"]
contrast_labels = {
    "M1-M0": "M1 − M0: Nonlinear age, without thickness",
    "M3-M2": "M3 − M2: Nonlinear age, with thickness",
    "M2-M0": "M2 − M0: Adding local thickness",
}
fig1_input = (
    regional_boot.query("metric == 'mean_log_loss' and contrast in @contrast_order")
    .set_index("contrast").loc[contrast_order].reset_index()
    .rename(columns={
        "difference_candidate_minus_reference": "delta_mll",
        "ci_low": "ci_lower",
        "ci_high": "ci_upper",
    })
)
fig1_input["comparison_label"] = fig1_input["contrast"].map(contrast_labels)
fig1_input = fig1_input[["contrast", "comparison_label", "delta_mll", "ci_lower", "ci_upper",
                         "n_bootstrap", "seed"]]

# Figure 2A input: observed minus nominal coverage in percentage points.
nominal_map = {"coverage_50": 0.50, "coverage_80": 0.80, "coverage_90": 0.90}
dist21 = distribution.query("scope == 'all_21_depths'").set_index("distribution_model")
coverage_rows = []
for model in ["Gaussian_M2", "warped_M2"]:
    for metric, nominal in nominal_map.items():
        observed = float(dist21.loc[model, metric])
        coverage_rows.append({
            "model": model,
            "display_model": "Gaussian M2" if model == "Gaussian_M2" else "Warped M2",
            "nominal_coverage": nominal,
            "observed_coverage": observed,
            "deviation_percentage_points": 100 * (observed - nominal),
        })
fig2_coverage_input = pd.DataFrame(coverage_rows)

# Figure 2B input: use the existing participant-bootstrap comparison.
fig2_forest_input = (
    distribution_boot.query("metric == 'mean_log_loss'")
    .rename(columns={
        "difference_warped_minus_gaussian": "delta_mll",
        "ci_low": "ci_lower",
        "ci_high": "ci_upper",
    })
    [["contrast", "delta_mll", "ci_lower", "ci_upper", "n_bootstrap", "seed"]]
    .copy()
)
fig2_forest_input["comparison_label"] = "Warped M2 − Gaussian M2"

# Figure 3 input: combine two disjoint formal ico4 stages at full precision.
def pooled_stage_summary(first, second):
    n1 = int(first["calibration"]["n_predictions"])
    n2 = int(second["calibration"]["n_predictions"])
    n = n1 + n2
    c1, c2 = first["calibration"], second["calibration"]
    mean = (n1 * c1["mean_z"] + n2 * c2["mean_z"]) / n
    variance = (
        (n1 - 1) * c1["sd_z"] ** 2
        + (n2 - 1) * c2["sd_z"] ** 2
        + n1 * (c1["mean_z"] - mean) ** 2
        + n2 * (c2["mean_z"] - mean) ** 2
    ) / (n - 1)
    result = {"n_predictions": n, "mean_z": mean, "sd_z": np.sqrt(variance)}
    for metric in ["coverage_50", "coverage_80", "coverage_90"]:
        result[metric] = (n1 * c1[metric] + n2 * c2[metric]) / n
    return result

ico4_summary = pooled_stage_summary(anchor, remaining)
ico4_rows = []
for metric, nominal in nominal_map.items():
    observed = ico4_summary[metric]
    ico4_rows.append({
        "nominal_coverage": nominal,
        "observed_coverage": observed,
        "deviation_percentage_points": 100 * (observed - nominal),
        "mean_z": ico4_summary["mean_z"],
        "sd_z": ico4_summary["sd_z"],
        "n_predictions": ico4_summary["n_predictions"],
        "n_participants": preflight["cohort_n"],
        "n_nodes": preflight["n_nodes"],
        "n_depths": preflight["n_depths"],
    })
fig3_input = pd.DataFrame(ico4_rows)

# Audit against the values supplied for plotting. Values are compared at their stated precision.
expected_fig1 = {
    "M1-M0": (0.000390, -0.008315, 0.009655),
    "M3-M2": (-0.001967, -0.010866, 0.007417),
    "M2-M0": (-0.015627, -0.021215, -0.010069),
}
expected_fig2 = {
    ("Gaussian M2", 0.50): 3.41, ("Warped M2", 0.50): 0.44,
    ("Gaussian M2", 0.80): 2.25, ("Warped M2", 0.80): 0.51,
    ("Gaussian M2", 0.90): 0.91, ("Warped M2", 0.90): 0.42,
}
expected_fig2_forest = (-0.022927, -0.034351, -0.012741)
expected_fig3 = {0.50: -0.23, 0.80: 0.30, 0.90: 0.38}

audit_rows = []
for row in fig1_input.itertuples():
    for field, observed, expected in zip(
        ["delta_mll", "ci_lower", "ci_upper"],
        [row.delta_mll, row.ci_lower, row.ci_upper], expected_fig1[row.contrast]
    ):
        audit_rows.append({"figure": "Figure 1", "item": f"{row.contrast} {field}",
                           "formal_value": observed, "provided_value": expected,
                           "display_decimals": 6,
                           "matches_at_display_precision": round(observed, 6) == round(expected, 6)})
for row in fig2_coverage_input.itertuples():
    expected = expected_fig2[(row.display_model, row.nominal_coverage)]
    audit_rows.append({"figure": "Figure 2A",
                       "item": f"{row.display_model}, {row.nominal_coverage:.0%}",
                       "formal_value": row.deviation_percentage_points, "provided_value": expected,
                       "display_decimals": 2,
                       "matches_at_display_precision": round(row.deviation_percentage_points, 2) == expected})
forest_row = fig2_forest_input.iloc[0]
for field, expected in zip(["delta_mll", "ci_lower", "ci_upper"], expected_fig2_forest):
    observed = float(forest_row[field])
    audit_rows.append({"figure": "Figure 2B", "item": field, "formal_value": observed,
                       "provided_value": expected, "display_decimals": 6,
                       "matches_at_display_precision": round(observed, 6) == round(expected, 6)})
for row in fig3_input.itertuples():
    expected = expected_fig3[row.nominal_coverage]
    audit_rows.append({"figure": "Figure 3", "item": f"{row.nominal_coverage:.0%}",
                       "formal_value": row.deviation_percentage_points, "provided_value": expected,
                       "display_decimals": 2,
                       "matches_at_display_precision": round(row.deviation_percentage_points, 2) == expected})
audit = pd.DataFrame(audit_rows)
display(audit)
if not audit["matches_at_display_precision"].all():
    raise RuntimeError("Formal result values differ from the supplied values at display precision.")

print("All supplied values match the formal files at the requested display precision.")
print("No model was refit and no bootstrap was repeated.")
"""
    ),
    code(
        """
# Retain the exact plotting inputs and the value audit.
fig1_input.to_csv(TABLE_DIR / "figure1_regional_model_selection.csv", index=False)
fig2_coverage_input.to_csv(TABLE_DIR / "figure2a_regional_coverage_deviation.csv", index=False)
fig2_forest_input.to_csv(TABLE_DIR / "figure2b_warped_vs_gaussian_mll.csv", index=False)
fig3_input.to_csv(TABLE_DIR / "figure3_ico4_calibration.csv", index=False)
audit.to_csv(TABLE_DIR / "provided_value_audit.csv", index=False)

display(fig1_input)
display(fig2_coverage_input)
display(fig2_forest_input)
display(fig3_input)
"""
    ),
    markdown("## 3. Plotting"),
    code(
        """
# Shared restrained visual style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "gray": "#9AA0A6",
    "dark_gray": "#5F6368",
    "teal": "#167C80",
    "blue": "#4C78A8",
    "warped": "#159D8C",
    "zero": "#6B7280",
}

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")

def export_figure(fig, stem):
    paths = []
    for suffix, kwargs in [("png", {"dpi": 300}), ("pdf", {}), ("svg", {})]:
        path = FIGURE_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    return paths

exported = []
"""
    ),
    markdown("### Figure 1. Regional model selection"),
    code(
        """
plot = fig1_input.copy()
y = np.arange(len(plot))[::-1]
colors = [COLORS["gray"], COLORS["gray"], COLORS["teal"]]

fig, ax = plt.subplots(figsize=(9.2, 4.25))
for yi, row, color in zip(y, plot.itertuples(), colors):
    ax.errorbar(
        row.delta_mll, yi,
        xerr=[[row.delta_mll - row.ci_lower], [row.ci_upper - row.delta_mll]],
        fmt="o", ms=6.5, color=color, ecolor=color, elinewidth=1.6, capsize=3.5,
        markeredgewidth=0, zorder=3,
    )
ax.axvline(0, color=COLORS["zero"], linestyle=(0, (3, 3)), linewidth=1)
ax.set_yticks(y, plot["comparison_label"])
ax.set_xlabel("Difference in mean log loss")
ax.set_title("Regional model selection", loc="left", fontweight="semibold")
ax.set_xlim(-0.025, 0.013)
ax.set_ylim(-0.6, len(plot) - 0.4)
clean_axis(ax)
fig.subplots_adjust(left=0.39, bottom=0.29, right=0.98, top=0.88)
fig.text(0.39, 0.17, "Negative values favor the first model", color=COLORS["dark_gray"], fontsize=9)
fig.text(
    0.01, 0.035,
    "Mean log loss = −logp. Confidence intervals were obtained by resampling participants; "
    "region × depth observations were not treated as independent units.",
    ha="left", va="bottom", fontsize=8.4, color=COLORS["dark_gray"],
)
exported.extend(export_figure(fig, "figure1_regional_delta_mll_forest"))
plt.show()
"""
    ),
    markdown(
        "**Caption.** Regional model-selection contrasts in mean log loss (−logp; lower is better). "
        "Points are paired differences and horizontal intervals are 95% confidence intervals from "
        "participant-level bootstrap resampling."
    ),
    markdown("### Figure 2. Regional Gaussian versus Warped M2"),
    code(
        """
fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(10.5, 4.65), gridspec_kw={"width_ratios": [1.8, 1.0], "wspace": 0.42}
)

offsets = {"Gaussian M2": -1.25, "Warped M2": 1.25}
model_colors = {"Gaussian M2": COLORS["blue"], "Warped M2": COLORS["warped"]}
for model in ["Gaussian M2", "Warped M2"]:
    d = fig2_coverage_input.query("display_model == @model")
    x = 100 * d["nominal_coverage"].to_numpy() + offsets[model]
    ax_a.plot(x, d["deviation_percentage_points"], "o", ms=7, color=model_colors[model],
              markeredgewidth=0, label=model)
ax_a.axhline(0, color=COLORS["zero"], linewidth=1)
ax_a.set_xticks([50, 80, 90], ["50%", "80%", "90%"])
ax_a.set_xlim(43, 97)
ax_a.set_ylim(-0.75, 4.0)
ax_a.set_xlabel("Nominal coverage")
ax_a.set_ylabel("Coverage deviation (percentage points)")
ax_a.legend(frameon=False, loc="upper right")
ax_a.set_title("A", loc="left", fontweight="bold")
clean_axis(ax_a)

row = fig2_forest_input.iloc[0]
ax_b.errorbar(
    row.delta_mll, 0,
    xerr=[[row.delta_mll - row.ci_lower], [row.ci_upper - row.delta_mll]],
    fmt="o", ms=7, color=COLORS["warped"], ecolor=COLORS["warped"],
    elinewidth=1.6, capsize=3.5, markeredgewidth=0,
)
ax_b.axvline(0, color=COLORS["zero"], linestyle=(0, (3, 3)), linewidth=1)
ax_b.set_yticks([0], ["Warped M2 −\\nGaussian M2"])
ax_b.set_ylim(-0.75, 0.75)
ax_b.set_xlim(-0.040, 0.006)
ax_b.set_xlabel("Difference in mean log loss")
ax_b.set_title("B", loc="left", fontweight="bold")
ax_b.text(0.02, 0.08, "Lower MLL is better", transform=ax_b.transAxes,
          fontsize=8.8, color=COLORS["dark_gray"])
clean_axis(ax_b)

fig.subplots_adjust(left=0.10, bottom=0.27, right=0.98, top=0.90)
fig.text(
    0.01, 0.035,
    "A: Coverage deviation is 100 × (observed − nominal); zero denotes ideal calibration. "
    "B: Warped M2 minus Gaussian M2 mean log loss with a participant-bootstrap 95% CI.",
    ha="left", va="bottom", fontsize=8.4, color=COLORS["dark_gray"],
)
exported.extend(export_figure(fig, "figure2_regional_gaussian_vs_warped"))
plt.show()
"""
    ),
    markdown(
        "**Caption.** Regional Gaussian and Warped M2 calibration. Panel A reports observed-minus-"
        "nominal coverage without uncertainty intervals because no corresponding participant-bootstrap "
        "intervals were available. Panel B reports the mean-log-loss difference and its participant-"
        "bootstrap 95% confidence interval."
    ),
    markdown("### Figure 3. Overall ico4 calibration"),
    code(
        """
fig, ax = plt.subplots(figsize=(6.9, 4.65))
x = 100 * fig3_input["nominal_coverage"].to_numpy()
y = fig3_input["deviation_percentage_points"].to_numpy()
ax.plot(x, y, "o", ms=7.5, color=COLORS["warped"], markeredgewidth=0)
ax.axhline(0, color=COLORS["zero"], linewidth=1)
ax.set_xticks([50, 80, 90], ["50%", "80%", "90%"])
ax.set_xlim(43, 97)
ax.set_ylim(-0.75, 4.0)
ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Coverage deviation (percentage points)")
ax.set_title("Overall ico4 calibration", loc="left", fontweight="semibold")
ax.text(
    0.98, 0.93,
    f"Mean z = {ico4_summary['mean_z']:.5f}\\nSD(z) = {ico4_summary['sd_z']:.5f}",
    transform=ax.transAxes, ha="right", va="top", fontsize=9.3, color=COLORS["dark_gray"],
)
clean_axis(ax)
fig.subplots_adjust(left=0.16, bottom=0.30, right=0.97, top=0.88)
fig.text(
    0.01, 0.035,
    "Coverage deviation is 100 × (observed − nominal); zero denotes ideal calibration. "
    "186 participants, 4,687 bilateral ico4 nodes, 21 depths, and 18,100,551 valid OOF predictions.",
    ha="left", va="bottom", fontsize=8.4, color=COLORS["dark_gray"],
)
exported.extend(export_figure(fig, "figure3_ico4_overall_calibration"))
plt.show()
"""
    ),
    markdown(
        "**Caption.** Overall ico4 Warped-M2 OOF calibration across 186 participants, 4,687 bilateral "
        "nodes and 21 depths. Points show observed-minus-nominal coverage. No uncertainty intervals "
        "are shown because a combined participant-bootstrap analysis has not yet been performed."
    ),
    markdown("## 4. Export"),
    code(
        """
captions = pd.DataFrame([
    {
        "figure": "Figure 1",
        "caption": "Regional model-selection contrasts in mean log loss (−logp; lower is better). "
                   "Points are paired differences and horizontal intervals are 95% confidence "
                   "intervals from participant-level bootstrap resampling."
    },
    {
        "figure": "Figure 2",
        "caption": "Regional Gaussian and Warped M2 calibration. Panel A reports coverage deviation; "
                   "Panel B reports the mean-log-loss contrast with its participant-bootstrap 95% CI."
    },
    {
        "figure": "Figure 3",
        "caption": "Overall ico4 Warped-M2 OOF calibration across 186 participants, 4,687 bilateral "
                   "nodes and 21 depths. No participant-bootstrap intervals are shown."
    },
])
captions.to_csv(TABLE_DIR / "figure_captions.csv", index=False)

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

all_outputs = sorted(FIGURE_DIR.glob("figure*.*")) + sorted(TABLE_DIR.glob("*.csv"))
manifest = pd.DataFrame([
    {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
    for path in all_outputs
])
manifest.to_csv(TABLE_DIR / "export_manifest.csv", index=False)
display(manifest)

print(f"Exported {len(list(FIGURE_DIR.glob('*.png')))} PNG files, "
      f"{len(list(FIGURE_DIR.glob('*.pdf')))} PDF files, and "
      f"{len(list(FIGURE_DIR.glob('*.svg')))} SVG files.")
print(f"Figure directory: {FIGURE_DIR}")
print(f"Plotting-input directory: {TABLE_DIR}")
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
