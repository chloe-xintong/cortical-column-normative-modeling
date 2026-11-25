#!/usr/bin/env python3
"""Validate full-cohort ico4 models and summarize original-scale normative age curves.

Description
-----------
Validate full-cohort ico4 models and summarize original-scale normative age curves.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--result-root`` command-line input

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/validate_summarize_ico4_full_cohort.py --help
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pcntoolkit import BLR


ANCHORS = [0, 4, 8, 12, 16, 20]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-root", type=Path, required=True)
    return p.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    a = parse_args()
    started = time.monotonic()
    root = a.result_root
    full = root / "full_models"
    figures = full / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(root / "ico4_node_metadata.csv")
    missingness = pd.read_parquet(root / "ico4_node_depth_missingness.parquet")
    status_files = sorted(full.glob("status/hemisphere=*/vertex_chunk=*/*.json"))
    parameter_files = sorted(full.glob("parameters/hemisphere=*/vertex_chunk=*/*.parquet"))
    centile_files = sorted(full.glob("centiles/hemisphere=*/vertex_chunk=*/*.parquet"))
    if not (len(status_files) == len(parameter_files) == len(centile_files) == 4687):
        raise RuntimeError(
            f"Incomplete full-cohort files: status={len(status_files)}, parameters={len(parameter_files)}, "
            f"centiles={len(centile_files)}"
        )
    statuses = [json.loads(path.read_text()) for path in status_files]
    if any(x.get("status") != "complete" or x.get("n_success") != 21 for x in statuses):
        bad = [x for x in statuses if x.get("status") != "complete" or x.get("n_success") != 21]
        raise RuntimeError(f"Incomplete full-cohort node tasks: {len(bad)}")

    parameter_index, centile_index, curve_rows, warning_rows = [], [], [], []
    representative_centiles = None
    min_valid = (missingness.assign(node_index=missingness.feature_index // 21)
                 .groupby("node_index", as_index=False)
                 .agg(minimum_valid_md=("n_valid_md", "min"),
                      mean_valid_md=("n_valid_md", "mean"),
                      valid_thickness=("n_valid_local_thickness", "first")))
    candidates = metadata.merge(min_valid, on="node_index").query(
        "hemisphere == 'lh' and dk_label_name == 'parahippocampal'"
    ).copy()
    best_coverage = candidates.minimum_valid_md.max()
    candidates = candidates[candidates.minimum_valid_md.eq(best_coverage)].copy()
    median_vertex = float(candidates.fsaverage_vertex_id.median())
    candidates["distance_to_median_fsaverage_vertex"] = (candidates.fsaverage_vertex_id - median_vertex).abs()
    representative = candidates.sort_values(
        ["distance_to_median_fsaverage_vertex", "node_index"]
    ).iloc[0]
    representative_node = int(representative.node_index)

    total_parameters = total_centiles = warning_fits = restored_models = 0
    for node in range(4687):
        meta = metadata.iloc[node]
        hemi, chunk = str(meta.hemisphere), node // 100
        rel = Path(f"hemisphere={hemi}") / f"vertex_chunk={chunk:03d}" / f"node_{node:04d}.parquet"
        parameter_path, centile_path = full / "parameters" / rel, full / "centiles" / rel
        params = pd.read_parquet(parameter_path)
        centiles = pd.read_parquet(centile_path)
        if len(params) != 21 or set(params.depth_index) != set(range(21)) or (params.status == "failed").any():
            raise RuntimeError(f"Parameter integrity failure at node {node}")
        if len(centiles) != 21 * 32 * 2 or set(centiles.depth_index) != set(range(21)):
            raise RuntimeError(f"Centile integrity failure at node {node}")
        finite_centiles = centiles[["age", "thickness_reference", "p05_md", "p50_md", "p95_md"]].to_numpy(float)
        if not np.isfinite(finite_centiles).all() or not np.all(
            centiles.p05_md.to_numpy() <= centiles.p50_md.to_numpy()
        ) or not np.all(centiles.p50_md.to_numpy() <= centiles.p95_md.to_numpy()):
            raise RuntimeError(f"Nonfinite or unordered centiles at node {node}")
        parameter_index.append({
            "relative_path": str(parameter_path.relative_to(root)), "node_index": node,
            "hemisphere": hemi, "vertex_chunk": chunk, "n_models": len(params),
            "n_warnings": int(params.status.eq("complete_with_warning").sum()),
        })
        centile_index.append({
            "relative_path": str(centile_path.relative_to(root)), "node_index": node,
            "hemisphere": hemi, "vertex_chunk": chunk, "n_rows": len(centiles),
            "age_min": float(centiles.age.min()), "age_max": float(centiles.age.max()),
            "n_depths": centiles.depth_index.nunique(), "n_sex_codes": centiles.sex_code.nunique(),
        })
        warning = params[params.status.eq("complete_with_warning")]
        if len(warning):
            warning_rows.append(warning[["node_index", "hemisphere", "ico_node_id", "dk_label_name",
                                         "depth_index", "warning"]])
        warning_fits += len(warning)

        # Every stored model is reconstructed and checked against one stored original-scale median.
        for param in params.itertuples(index=False):
            model = BLR.from_dict(json.loads(param.model_dict_json))
            row = centiles.query(
                "depth_index == @param.depth_index and sex_code == 1 and age == 45"
            ).iloc[0]
            x = np.array([[
                (45.0 - param.age_mean) / param.age_scale,
                0.0,
                (param.thickness_reference_median - param.thickness_mean) / param.thickness_scale,
            ]])
            mu, s2 = model.ys_s2(x, np.zeros((1, 1), dtype=int))
            median_scaled, _ = model.warp.warp_predictions(mu, s2, model.gamma, percentiles=[.05, .95])
            recovered = float(median_scaled[0] * param.response_scale + param.response_mean)
            if not np.isfinite(recovered) or not np.isclose(recovered, row.p50_md, rtol=1e-10, atol=1e-12):
                raise RuntimeError(f"Model recovery mismatch at node={node}, depth={param.depth_index}")
            restored_models += 1

        for (depth, sex_code), group in centiles.groupby(["depth_index", "sex_code"], sort=True):
            group = group.sort_values("age")
            age = group.age.to_numpy(float)
            median = group.p50_md.to_numpy(float)
            slope, intercept = np.polyfit(age, median, 1)
            fitted = intercept + slope * age
            total_change = float(median[-1] - median[0])
            scale = max(abs(total_change), np.finfo(float).eps)
            first = np.diff(median)
            curve_rows.append({
                "node_index": node, "hemisphere": hemi, "ico_node_id": int(meta.ico_node_id),
                "fsaverage_vertex_id": int(meta.fsaverage_vertex_id),
                "dk_label_name": str(meta.dk_label_name), "depth_index": int(depth),
                "depth_fraction": int(depth) / 20.0, "sex_code": int(sex_code),
                "age_min": float(age.min()), "age_max": float(age.max()),
                "p50_change_45_to_76": total_change,
                "linear_slope_per_year": float(slope),
                "linear_slope_per_decade": float(10 * slope),
                "max_absolute_linear_residual": float(np.max(np.abs(median - fitted))),
                "relative_nonlinearity": float(np.max(np.abs(median - fitted)) / scale),
                "monotonic": bool(np.all(first >= 0) or np.all(first <= 0)),
                "first_difference_sign_changes": int(np.sum(np.sign(first[1:]) != np.sign(first[:-1]))),
            })
        if node == representative_node:
            representative_centiles = centiles.copy()
        total_parameters += len(params)
        total_centiles += len(centiles)
        if (node + 1) % 250 == 0:
            print(json.dumps({"nodes_validated": node + 1, "models_restored": restored_models,
                              "elapsed_minutes": (time.monotonic() - started) / 60}), flush=True)

    pd.DataFrame(parameter_index).to_parquet(full / "parameter_file_index.parquet", index=False)
    pd.DataFrame(centile_index).to_parquet(full / "centile_file_index.parquet", index=False)
    warnings = pd.concat(warning_rows, ignore_index=True) if warning_rows else pd.DataFrame()
    warnings.to_csv(full / "warning_fits.csv", index=False)
    curves = pd.DataFrame(curve_rows)
    curves.to_parquet(full / "age_curve_shape_diagnostics.parquet", index=False)
    representative_centiles.to_parquet(full / "representative_node_centiles.parquet", index=False)
    selection = pd.DataFrame([{
        "selection_rule": "Within left parahippocampal ico4 nodes, maximize the minimum valid MD count across 21 depths; among ties choose the fsaverage vertex ID closest to the tied-set median.",
        "node_index": representative_node, "hemisphere": representative.hemisphere,
        "ico_node_id": int(representative.ico_node_id),
        "fsaverage_vertex_id": int(representative.fsaverage_vertex_id),
        "dk_label_name": representative.dk_label_name,
        "minimum_valid_md": int(representative.minimum_valid_md),
        "mean_valid_md": float(representative.mean_valid_md),
        "valid_thickness": int(representative.valid_thickness),
    }])
    selection.to_csv(full / "representative_node_selection.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": False,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"p05": "#9AA0A6", "p50": "#167C80", "p95": "#9AA0A6"}
    for sex_code in (1, 2):
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True, sharey=True)
        for ax, depth in zip(axes.flat, ANCHORS):
            d = representative_centiles.query(
                "sex_code == @sex_code and depth_index == @depth"
            ).sort_values("age")
            ax.plot(d.age, d.p05_md, color=colors["p05"], lw=1.2, label="5th percentile")
            ax.plot(d.age, d.p50_md, color=colors["p50"], lw=1.8, label="50th percentile")
            ax.plot(d.age, d.p95_md, color=colors["p95"], lw=1.2, label="95th percentile")
            ax.fill_between(d.age, d.p05_md, d.p95_md, color="#167C80", alpha=.09)
            ax.set_title(f"Normalized depth = {depth / 20:.2f}", loc="left", fontsize=10)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
        for ax in axes[-1]:
            ax.set_xlabel("Age (years)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Mean diffusivity (mm²/s)")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, .98))
        fig.suptitle(
            f"Conditional normative curves: {representative.hemisphere} {representative.dk_label_name}, "
            f"ico4 node {int(representative.ico_node_id)}, sex code {sex_code}",
            x=.06, y=1.04, ha="left", fontsize=12,
        )
        fig.text(.01, .005,
                 "Thickness is fixed at the node-local median. The 5th–95th percentiles describe the "
                 "conditional individual MD distribution, not a confidence interval for the median.",
                 fontsize=8.5, color="#5F6368")
        fig.tight_layout(rect=(0, .04, 1, .93))
        stem = f"figure6a_representative_node_sex{sex_code}"
        for ext, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
            fig.savefig(figures / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
        plt.close(fig)

    nonlinearity = curves.relative_nonlinearity.replace([np.inf, -np.inf], np.nan).dropna()
    monotonic_fraction = float(curves.monotonic.mean())
    proposal = f"""# Figure 6C age-effect preparation

## Supported ages

- Observed cohort range: 41–78 years.
- Curve-display range: 45–76 years, corresponding to the empirical 2.5th and 95th age percentiles.
- Sex codes 1 and 2 both span the observed 41–78-year range.
- The local source codebook has not been verified, so curves retain the neutral labels `Sex code 1`
  and `Sex code 2`; they must not be relabeled as male/female until source documentation is obtained.

## Curve-shape diagnostic

- Node × depth × sex curves checked: {len(curves):,}.
- Fraction monotonic on the 45–76 grid: {monotonic_fraction:.6f}.
- Median relative deviation from the best linear age trend: {nonlinearity.median():.6g}.
- 95th percentile relative deviation: {nonlinearity.quantile(.95):.6g}.

## Recommended decision framework

Do not label the latent-space age coefficient as MD change. Before producing maps, choose one
original-MD-scale estimand and freeze it. The preferred candidate is a support-anchored average
decadal change over 45–76 years, obtained from predicted medians and scaled by 10/31. This uses
the well-supported central age range and remains comparable across locations even when the inverse
warp introduces mild curvature. A second defensible option is the local derivative of the predicted
median at the cohort median age (58 years), but it is more sensitive to local curvature.

For the final map, marginalize predictions over the empirical participant distribution of sex and
node-local thickness rather than averaging sex codes or using a single global thickness value.
Predefine outer/middle/inner depth bands and weights, use one common color scale, and treat depth
differences as descriptive unless a formal within-location depth contrast is tested.

No age-effect map is generated in this stage, and no 65-minus-55 definition is assumed.
"""
    (full / "figure6c_age_effect_proposal.md").write_text(proposal, encoding="utf-8")

    report = {
        "status": "complete", "cohort_n": 186, "n_nodes": 4687, "n_depths": 21,
        "expected_models": 4687 * 21, "successful_models": total_parameters,
        "failed_models": 0, "warning_models": warning_fits,
        "restored_models_checked": restored_models, "centile_rows": total_centiles,
        "age_grid": {"min": 45, "max": 76, "step": 1,
                     "rule": "empirical 2.5th to 95th age percentiles"},
        "sex_codes": [1, 2], "sex_codebook_verified": False,
        "sex_label_policy": "Retain Sex code 1/2 labels; do not infer male/female without the source codebook",
        "thickness_reference_rule": "node-local median among retained participants with finite thickness",
        "representative_node": selection.iloc[0].to_dict(),
        "curve_shape": {
            "n_curves": len(curves), "monotonic_fraction": monotonic_fraction,
            "median_relative_nonlinearity": float(nonlinearity.median()),
            "p95_relative_nonlinearity": float(nonlinearity.quantile(.95)),
        },
        "age_effect_maps_generated": False,
        "elapsed_minutes": (time.monotonic() - started) / 60,
    }
    atomic_json(full / "full_cohort_validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
