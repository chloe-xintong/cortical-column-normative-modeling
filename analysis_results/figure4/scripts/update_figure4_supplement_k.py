#!/usr/bin/env python3
"""Idempotently add the ico5 K-convergence pilot to the Figure 4 supplement."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("analysis_results/figure4/notebooks/figure4_supplement.ipynb")
SECTION_IDS = {"s8-ico5-k-convergence", "s8-ico5-k-convergence-output"}


def lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    notebook["cells"] = [
        cell for cell in notebook["cells"] if cell.get("id") not in SECTION_IDS
    ]
    setup = next(cell for cell in notebook["cells"] if cell.get("id") == "setup")
    setup_source = "".join(setup["source"])
    definition = (
        "ICO5_K_DIR = ROOT / 'figure4_ico5_k_convergence_pilot20/cohort'\n"
        "ICO5_K20_DIR = ROOT / 'figure4_ico5_k_convergence_pilot20/cohort_k20'\n"
        "ICO5_K25_DIR = ROOT / 'figure4_ico5_k_convergence_pilot20/cohort_k25'\n"
        "ICO5_K30_DIR = ROOT / 'figure4_ico5_k_convergence_pilot20/cohort_k30'\n"
        "ICO5_K_PILOT5_DIR = ROOT / 'figure4_ico5_k_convergence_pilot20/cohort_pilot5'\n"
        "ICO5_K_PAIR_DIR = ROOT / 'figure4_ico5_k_convergence_pilot20/pairs'\n"
    )
    if "ICO5_K_DIR" not in setup_source:
        marker = "DK_DIR = ROOT / 'figure4_dk_all_pairs_region_balanced/cohort'\n"
        setup_source = setup_source.replace(marker, marker + definition)
    elif "ICO5_K_PILOT5_DIR" not in setup_source:
        marker = "ICO5_K_DIR = ROOT / 'figure4_ico5_k_convergence_pilot20/cohort'\n"
        setup_source = setup_source.replace(
            marker,
            marker + "ICO5_K_PILOT5_DIR = ROOT / 'figure4_ico5_k_convergence_pilot20/cohort_pilot5'\n",
        )
    if "ICO5_K20_DIR" not in setup_source:
        marker = "ICO5_K_DIR = ROOT / 'figure4_ico5_k_convergence_pilot20/cohort'\n"
        setup_source = setup_source.replace(
            marker,
            marker + "ICO5_K20_DIR = ROOT / 'figure4_ico5_k_convergence_pilot20/cohort_k20'\n",
        )
    if "ICO5_K30_DIR" not in setup_source:
        marker = "ICO5_K25_DIR = ROOT / 'figure4_ico5_k_convergence_pilot20/cohort_k25'\n"
        setup_source = setup_source.replace(
            marker,
            marker + "ICO5_K30_DIR = ROOT / 'figure4_ico5_k_convergence_pilot20/cohort_k30'\n",
        )
    if "ICO5_K25_DIR" not in setup_source:
        marker = "ICO5_K20_DIR = ROOT / 'figure4_ico5_k_convergence_pilot20/cohort_k20'\n"
        setup_source = setup_source.replace(
            marker,
            marker + "ICO5_K25_DIR = ROOT / 'figure4_ico5_k_convergence_pilot20/cohort_k25'\n",
        )
    setup["source"] = lines(setup_source)

    notebook["cells"].extend([
        {
            "cell_type": "markdown",
            "id": "s8-ico5-k-convergence",
            "metadata": {},
            "source": lines(
                """
## S8. ico5 within-region neighbour-number convergence

This pilot determines how many full-resolution, same-DK neighbours are needed per official ico5 anchor and 2-mm geodesic-distance bin. The official 10,242-vertex-per-hemisphere fsaverage5 sphere is mapped one-to-one to full-resolution fsaverage; the existing cortical analysis mask is then applied. For each of 10 random seeds, sampling is nested so that $P_1\\subset P_3\\subset P_5\\subset P_{10}\\subset P_{15}\\subset P_{20}\\subset P_{25}\\subset P_{30}$. Existing ranks 1–10 are retained exactly and only ranks 11–30 are appended. The same 20 subjects used in the acquisition-null pilot are analysed using the hierarchy pair $\\rightarrow$ anchor × bin median $\\rightarrow$ bilateral subject or DK-region median $\\rightarrow$ group summary. The current priority analysis reports $K=1,3,5,10,15,20,25$ with $K=25$ as a provisional high-$K$ reference; $K=30$ remains a later sensitivity reference rather than ground truth. Selection uses real profiles only and therefore is independent of the real–null effect.

The diagnostics report candidate-neighbour coverage, $D=1-r$, MAE and symmetric relative MAE (sRMAE). Figure K1 compares the group curves, Figure K2 shows $D_K-D_{K_{ref}}$ with the ±0.002 tolerance, and Figure K3 shows the bilateral region × distance-bin fraction of anchors having at least three candidates. Median, 95th-percentile and maximum absolute differences, seed SD/IQR, subject and region rank stability, and the analysis restricted to anchor-bins with at least 20 candidates are saved alongside the figures.
"""
            ),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "s8-ico5-k-convergence-output",
            "metadata": {},
            "outputs": [],
            "source": lines(
                """
k30_decision_path = ICO5_K30_DIR / 'k_convergence_decision.json'
k25_decision_path = ICO5_K25_DIR / 'k_convergence_decision.json'
k20_decision_path = ICO5_K20_DIR / 'k_convergence_decision.json'
full_decision_path = ICO5_K_DIR / 'k_convergence_decision.json'
pilot5_decision_path = ICO5_K_PILOT5_DIR / 'k_convergence_decision.json'
mapping_path = ICO5_K_PAIR_DIR / 'ico5_anchor_mapping.csv'
build_path = ICO5_K_PAIR_DIR / 'ico5_pair_build.json'

if k30_decision_path.is_file():
    result_dir = ICO5_K30_DIR
    result_label = '20-subject extended K=1,3,5,10,15,20,25,30 convergence pilot'
elif k25_decision_path.is_file():
    result_dir = ICO5_K25_DIR
    result_label = '20-subject priority K=1,3,5,10,15,20,25 convergence pilot'
elif k20_decision_path.is_file():
    result_dir = ICO5_K20_DIR
    result_label = '20-subject extended K=1,3,5,10,15,20 convergence pilot'
elif full_decision_path.is_file():
    result_dir = ICO5_K_DIR
    result_label = '20-subject K-convergence pilot'
elif pilot5_decision_path.is_file():
    result_dir = ICO5_K_PILOT5_DIR
    result_label = 'preliminary first-5-subject checkpoint'
else:
    result_dir = None

if result_dir is None:
    status = 'neighbourhood construction in progress'
    if build_path.is_file():
        status = 'neighbourhood construction complete; 20-subject K pilot or summary pending'
    display(Markdown(
        f'**ico5 K-convergence status:** {status}.  '
        'This section will load the cached results automatically after the dependent jobs finish.'
    ))
else:
    decision_path = result_dir / 'k_convergence_decision.json'
    coverage_path = result_dir / 'k_convergence_coverage.csv'
    stability_path = result_dir / 'k_convergence_stability.csv'
    decision = json.loads(decision_path.read_text())
    mapping = pd.read_csv(mapping_path)
    coverage = pd.read_csv(coverage_path)
    stability = pd.read_csv(stability_path)

    anchor_inventory = (
        mapping.groupby('hemisphere', as_index=False)
        .agg(
            valid_ico5_anchors=('ico5_vertex_id', 'nunique'),
            dk_regions=('dk_region', 'nunique'),
            maximum_mapping_error_degrees=(
                'angular_mapping_error_degrees', 'max'
            ),
        )
    )
    display(Markdown(
        f"**Result shown: {result_label}. Reference K: {decision['reference_k']}; "
        f"selected K: {decision['selected_k']}.** "
        f"{decision['decision_reason']}"
    ))
    display(anchor_inventory)

    available_k = sorted(stability['k'].astype(int).unique())
    coverage_columns = [
        f'fraction_at_least_{k}' for k in available_k
        if f'fraction_at_least_{k}' in coverage.columns
    ]
    coverage_by_bin = (
        coverage.groupby(['distance_bin_index', 'distance_bin'], as_index=False)[coverage_columns]
        .median()
        .rename(columns={name: f'region_median_{name}' for name in coverage_columns})
        .sort_values('distance_bin_index')
    )
    display(coverage_by_bin)

    primary_stability = stability.loc[
        stability['metric'].eq('profile_dissimilarity')
        & stability['k'].isin(available_k)
    ]
    stability_across_seeds = (
        primary_stability.groupby('k', as_index=False)
        .agg(
            worst_group_curve_difference=(
                'curve_maximum_absolute_difference', 'max'
            ),
            minimum_subject_spearman=('subject_spearman_rho', 'min'),
            minimum_subject_icc=('subject_icc_two_way_single', 'min'),
            minimum_median_region_spearman=(
                'region_spearman_median_subject', 'min'
            ),
            maximum_seed_curve_range=(
                'maximum_seed_to_seed_curve_difference', 'max'
            ),
        )
    )
    display(stability_across_seeds)

    for figure_name in (
        'FigureK1_curves.png',
        (
            'FigureK2_difference_from_k25.png'
            if (result_dir / 'FigureK2_difference_from_k25.png').is_file()
            else (
                'FigureK2_difference_from_k30.png'
                if (result_dir / 'FigureK2_difference_from_k30.png').is_file()
                else (
                    'FigureK2_difference_from_k20.png'
                    if (result_dir / 'FigureK2_difference_from_k20.png').is_file()
                    else 'FigureK2_difference_from_k10.png'
                )
            )
        ),
        'FigureK3_k3_coverage.png',
    ):
        show_image(result_dir / figure_name)
"""
            ),
        },
    ])
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
