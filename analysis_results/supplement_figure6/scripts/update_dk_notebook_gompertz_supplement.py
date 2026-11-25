#!/usr/bin/env python3
"""Add the publication-style Gompertz pilot supplement to the Figure 6 DK notebook.

Description
-----------
This updater is intentionally idempotent.  It preserves every unrelated notebook
cell, replaces only cells carrying its metadata tag, and inserts the supplement
immediately before the export section.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- No required command-line inputs; optional inputs are defined by the script.

Outputs
-------
- In-memory values returned to the calling code.

Usage
-----
python analysis_results/figure6/scripts/update_dk_notebook_gompertz_supplement.py --help
"""

from __future__ import annotations

from pathlib import Path
import re

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "figure6_DK_region_model_selection_186.ipynb"
TAG = "figure6_gompertz_pilot_supplement"


def markdown(text: str):
    cell = nbf.v4.new_markdown_cell(text.strip())
    cell.metadata[TAG] = True
    return cell


def code(text: str):
    cell = nbf.v4.new_code_cell(text.strip())
    cell.metadata[TAG] = True
    return cell


SUPPLEMENT_CAPTION = (
    "Warped Gompertz pilot compared with the reused Warped linear OOF predictions for the "
    "same 186 participants, six DK regions, and five normalized depths (5,580 observations). "
    "The left panel shows the overall candidate-minus-linear mean-log-loss difference and its "
    "95% confidence interval from 10,000 participant-level bootstrap resamples; negative values "
    "favor Gompertz. The right panel shows three nonexclusive optimization and identifiability "
    "flags among 150 region-by-depth-by-outer-fold fits: 90 outer optimizations did not report "
    "normal convergence, 50 fits ended at a shape bound, and 70 had greater than 5-year "
    "disagreement among multistart inflection-age solutions. All predictions were finite and no "
    "paired observation was excluded. The overall interval included zero and the pilot was not "
    "promoted. Depth-specific comparisons are reported separately in the supplementary table."
)


NEW_CELLS = [
    markdown(
        """
### Supplementary figure. Gompertz pilot performance and fit stability

This supplement places the paired predictive comparison beside the diagnostics that prevented
promotion of Gompertz. The baseline Warped linear OOF predictions are reused; this section does
not refit either model, repeat the bootstrap, or run ico4. The three plotted diagnostic categories
are nonexclusive. A near-zero age coefficient is retained only in the accompanying audit table;
it is not classified as a fitting failure because it may reflect a genuine weak age effect.
"""
    ),
    code(
        """
import sys

AGE_FUNCTION = ROOT / "regional_age_function_comparison_186" / "comparison"
gompertz_sources = {
    "paired_bootstrap": AGE_FUNCTION / "paired_subject_bootstrap_delta_mll.csv",
    "fit_stability": AGE_FUNCTION / "fit_stability_summary.csv",
    "model_decision": AGE_FUNCTION / "model_decision.json",
}
missing = [str(path) for path in gompertz_sources.values() if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing formal Gompertz comparison files:\\n" + "\\n".join(missing))

gompertz_bootstrap = pd.read_csv(gompertz_sources["paired_bootstrap"])
gompertz_bootstrap = gompertz_bootstrap.query("candidate == 'warped_gompertz'").copy()
gompertz_stability = pd.read_csv(gompertz_sources["fit_stability"])
gompertz_stability = gompertz_stability.query("model == 'warped_gompertz'").iloc[0]
gompertz_decision = json.loads(gompertz_sources["model_decision"].read_text())["warped_gompertz"]

# The pilot is exactly paired on 186 participants, six prespecified regions, and five depths.
overall = gompertz_bootstrap.query("scope == 'overall'").iloc[0]
by_depth = gompertz_bootstrap.query("scope == 'by_depth'").copy()
by_depth["depth_index"] = by_depth["depth_index"].astype(int)
assert len(gompertz_bootstrap) == 6
assert by_depth["depth_index"].tolist() == [0, 4, 10, 16, 20]
assert int(overall["n_subjects"]) == 186
assert int(overall["n_observations"]) == 5_580
assert int(overall["observations_per_subject"]) == 30
assert int(overall["n_bootstrap"]) == 10_000
assert int(gompertz_stability["fits_successful"]) == 150
assert int(gompertz_stability["task_failures"]) == 0
np.testing.assert_allclose(
    [overall["delta_mll"], overall["ci_low"], overall["ci_high"]],
    [gompertz_decision["delta_mll_pilot"], *gompertz_decision["ci_95"]],
    rtol=0, atol=1e-12,
)

overall_input = pd.DataFrame([overall])
overall_input["display_label"] = "Overall"
overall_input = overall_input[[
    "scope", "depth_index", "display_label", "delta_mll", "ci_low", "ci_high",
    "probability_candidate_better", "n_subjects", "n_observations",
    "observations_per_subject", "n_bootstrap", "seed",
]]
depth_input = by_depth.copy()
depth_input["display_label"] = [
    f"Depth {index / 20:.2f}" for index in depth_input["depth_index"]
]
depth_input = depth_input[[
    "scope", "depth_index", "display_label", "delta_mll", "ci_low", "ci_high",
    "probability_candidate_better", "n_subjects", "n_observations",
    "observations_per_subject", "n_bootstrap", "seed",
]]

diagnostic_specs = [
    ("Optimizer not normally converged", "outer_optimizer_failures", True,
     "Optimization convergence diagnostic"),
    ("Shape parameter at bound", "shape_at_bound", True,
     "Boundary diagnostic"),
    ("Inflection-age disagreement", "large_multistart_inflection_disagreement", True,
     "Multistart solutions differ by more than 5 years"),
    ("Near-zero age effect", "near_zero_age_effect", False,
     "Retained for audit only; not interpreted as a fitting failure"),
]
fit_total = int(gompertz_stability["fits_successful"])
diagnostic_input = pd.DataFrame([
    {
        "diagnostic": label,
        "source_field": field,
        "n_flagged": int(gompertz_stability[field]),
        "n_fits": fit_total,
        "percent_flagged": 100 * float(gompertz_stability[field]) / fit_total,
        "included_in_figure": include_in_figure,
        "interpretation": interpretation,
    }
    for label, field, include_in_figure, interpretation in diagnostic_specs
])
for row in diagnostic_input.itertuples():
    assert row.n_flagged == int(gompertz_decision[row.source_field])
plotted_diagnostics = diagnostic_input.query("included_in_figure").copy()
assert len(plotted_diagnostics) == 3

overall_input.to_csv(TABLE_DIR / "figureS_gompertz_delta_mll.csv", index=False)
depth_input.to_csv(TABLE_DIR / "figureS_gompertz_depth_delta_mll_supplement.csv", index=False)
diagnostic_input.to_csv(TABLE_DIR / "figureS_gompertz_fit_stability.csv", index=False)

# Use the paper-wide semantic palette, typography, no-grid rule, and export settings.
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from qc.publication_style import (  # noqa: E402
    FIGURE_WIDTH_IN, PALETTE, save_publication_figure, set_publication_style,
)

set_publication_style()
fig, (ax_effect, ax_stability) = plt.subplots(
    1, 2,
    figsize=(FIGURE_WIDTH_IN, 2.75),
    gridspec_kw={"width_ratios": [0.78, 1.22], "wspace": 0.78},
)

# Overall participant-bootstrap contrast; depth-specific results remain in a separate table.
overall_row = overall_input.iloc[0]
ax_effect.errorbar(
    overall_row["delta_mll"], 0,
    xerr=[
        [overall_row["delta_mll"] - overall_row["ci_low"]],
        [overall_row["ci_high"] - overall_row["delta_mll"]],
    ],
    fmt="o", markersize=3.8, markerfacecolor=PALETTE["blue"],
    markeredgecolor=PALETTE["blue"], markeredgewidth=0.8,
    color=PALETTE["blue"], ecolor=PALETTE["blue"],
    elinewidth=1.1, capsize=2.2, capthick=0.8, zorder=3,
)
ax_effect.axvline(0, color=PALETTE["gray"], linestyle=(0, (3, 3)), linewidth=0.9)
ax_effect.set_yticks([0], ["Overall"])
ax_effect.set_xlim(-0.025, 0.018)
ax_effect.set_ylim(-0.65, 0.65)
ax_effect.set_xlabel("Gompertz − linear mean log loss")
ax_effect.set_title("Overall ΔMLL", loc="left")

# These flags can overlap, so percentages are shown against the same 150-fit denominator.
bar_y = np.arange(len(plotted_diagnostics))[::-1]
bars = ax_stability.barh(
    bar_y, plotted_diagnostics["percent_flagged"], height=0.52,
    color=PALETTE["light_blue"], edgecolor=PALETTE["blue"], linewidth=0.8,
)
ax_stability.set_yticks(bar_y, plotted_diagnostics["diagnostic"])
ax_stability.set_xlim(0, 100)
ax_stability.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax_stability.set_xlabel("Fits flagged")
ax_stability.set_title("Optimization diagnostics", loc="left")
for bar, row in zip(bars, plotted_diagnostics.itertuples()):
    ax_stability.text(
        min(row.percent_flagged + 2.5, 93), bar.get_y() + bar.get_height() / 2,
        f"{row.n_flagged}", ha="left", va="center", fontsize=7.5,
    )

figure_stem = FIGURE_DIR / "figureS_gompertz_pilot_performance_stability"
save_publication_figure(fig, figure_stem)
plt.show()

display(overall_input)
display(depth_input)
display(diagnostic_input)
print("Saved:", figure_stem)
print("No model was refit, no bootstrap was repeated, and ico4 was not run.")
"""
    ),
    markdown(f"**Caption.** {SUPPLEMENT_CAPTION}"),
]


def main() -> None:
    notebook = nbf.read(NOTEBOOK, as_version=4)
    notebook.cells = [cell for cell in notebook.cells if not cell.metadata.get(TAG)]

    intro = notebook.cells[0]
    intro.source = intro.source.replace(
        "generates three presentation-ready figures",
        "generates four main figures and one supplementary figure",
    )

    export_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and cell.source.strip() == "## 4. Export"
    )
    notebook.cells[export_index:export_index] = NEW_CELLS

    export_code = next(
        cell
        for cell in notebook.cells[export_index + len(NEW_CELLS) :]
        if cell.cell_type == "code" and "captions.to_csv" in cell.source
    )
    marker = 'captions.to_csv(TABLE_DIR / "figure_captions.csv", index=False)'
    addition = (
        "captions = pd.concat([captions, pd.DataFrame([{\n"
        '    "figure": "Supplementary Figure",\n'
        f'    "caption": {SUPPLEMENT_CAPTION!r},\n'
        "}])], ignore_index=True)\n\n"
    )
    # Refresh a previously injected caption instead of accumulating stale versions.
    export_code.source = re.sub(
        r'captions = pd\.concat\(\[captions, pd\.DataFrame\(\[\{\n'
        r'    "figure": "Supplementary Figure",\n'
        r'    "caption": .*?\n'
        r'\}\]\)\], ignore_index=True\)\n\n',
        "",
        export_code.source,
        flags=re.DOTALL,
    )
    export_code.source = export_code.source.replace(marker, addition + marker)

    nbf.write(notebook, NOTEBOOK)
    print(f"Updated {NOTEBOOK}")


if __name__ == "__main__":
    main()
