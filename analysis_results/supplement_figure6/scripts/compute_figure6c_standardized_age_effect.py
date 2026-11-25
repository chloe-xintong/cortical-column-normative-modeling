#!/usr/bin/env python3
"""Compute a standardized age-range effect from the frozen Figure 6C models.

Description
-----------
The estimand is the equal-weight mean of subject-conditional, original-MD-scale
predicted medians at each target age.  Age is changed while the same subjects'
stored sex code and node-local thickness are held fixed.  Nodes without finite
local thickness for all 186 frozen-cohort participants are retained in the
output tables but marked ineligible; their estimates are not imputed.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--result-root`` command-line input
- ``--subject-order`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/compute_figure6c_standardized_age_effect.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pcntoolkit import BLR


AGES = (45.0, 76.0)
AGE_SPAN = AGES[1] - AGES[0]
DECADE_FACTOR = 10.0 / AGE_SPAN
BANDS = {
    "Outer": tuple(range(0, 7)),
    "Middle": tuple(range(7, 14)),
    "Inner": tuple(range(14, 21)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--subject-order", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--age-min", type=float, default=45.0)
    parser.add_argument("--age-max", type=float, default=76.0)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def parameter_path(root: Path, node: int, hemisphere: str) -> Path:
    return (
        root
        / "full_models"
        / "parameters"
        / f"hemisphere={hemisphere}"
        / f"vertex_chunk={node // 100:03d}"
        / f"node_{node:04d}.parquet"
    )


def support_table(
    metadata: pd.DataFrame,
    thickness: np.ndarray,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cohort_age = cohort["age"].to_numpy(float)
    nearby = {age: np.abs(cohort_age - age) <= 3.0 for age in AGES}
    for node, meta in enumerate(metadata.itertuples(index=False)):
        values = np.asarray(thickness[:, node], float)
        finite = np.isfinite(values)
        row: dict[str, object] = {
            "node_index": node,
            "hemisphere": str(meta.hemisphere),
            "ico_node_id": int(meta.ico_node_id),
            "fsaverage_vertex_id": int(meta.fsaverage_vertex_id),
            "dk_label_id": int(meta.dk_label_id),
            "dk_label_name": str(meta.dk_label_name),
            "n_cohort": len(cohort),
            "n_finite_local_thickness": int(finite.sum()),
            "n_missing_local_thickness": int((~finite).sum()),
            "eligible_exact_186_subject_estimand": bool(finite.all()),
            "all_subject_thickness_min_mm": float(np.min(values[finite])),
            "all_subject_thickness_max_mm": float(np.max(values[finite])),
        }
        for age in AGES:
            window = nearby[age]
            local = values[window & finite]
            label = int(age)
            lower = float(np.min(local)) if len(local) else np.nan
            upper = float(np.max(local)) if len(local) else np.nan
            outside = finite & ((values < lower) | (values > upper)) if len(local) else finite
            row.update({
                f"age{label}_nearby_window": f"{label - 3}-{label + 3}",
                f"age{label}_nearby_subjects": int(window.sum()),
                f"age{label}_nearby_finite_thickness": int(len(local)),
                f"age{label}_nearby_thickness_min_mm": lower,
                f"age{label}_nearby_thickness_max_mm": upper,
                f"age{label}_all_subject_values_outside_nearby_range_n": int(outside.sum()),
                f"age{label}_all_subject_values_outside_nearby_range_fraction": float(
                    outside.sum() / finite.sum()
                ),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def predict_node_depth(
    param: object,
    sex_code: np.ndarray,
    local_thickness: np.ndarray,
) -> tuple[float, float, int]:
    model = BLR.from_dict(json.loads(param.model_dict_json))
    if type(model.warp).__name__ != "WarpSinhArcsinh":
        raise RuntimeError("Restored model is not WarpSinhArcsinh")
    shared = np.column_stack([
        (sex_code == 2).astype(float),
        (local_thickness - float(param.thickness_mean)) / float(param.thickness_scale),
    ])
    predictions: list[float] = []
    first_shared: np.ndarray | None = None
    for age in AGES:
        x = np.column_stack([
            np.full(len(sex_code), (age - float(param.age_mean)) / float(param.age_scale)),
            shared,
        ])
        if first_shared is None:
            first_shared = x[:, 1:].copy()
        elif not np.array_equal(first_shared, x[:, 1:]):
            raise RuntimeError("Sex/thickness inputs changed between target ages")
        latent_mean, latent_variance = model.ys_s2(x, np.zeros((len(x), 1), dtype=int))
        median_scaled, _ = model.warp.warp_predictions(
            latent_mean, latent_variance, model.gamma, percentiles=[0.05, 0.95]
        )
        median_md = median_scaled * float(param.response_scale) + float(param.response_mean)
        if median_md.shape != (len(sex_code),) or not np.isfinite(median_md).all():
            raise RuntimeError("Nonfinite or malformed original-scale median predictions")
        predictions.append(float(np.mean(median_md)))
    return predictions[0], predictions[1], len(sex_code)


def main() -> None:
    global AGES, AGE_SPAN, DECADE_FACTOR
    args = parse_args()
    AGES = (float(args.age_min), float(args.age_max))
    AGE_SPAN = AGES[1] - AGES[0]
    if AGE_SPAN <= 0:
        raise ValueError("--age-max must be greater than --age-min")
    DECADE_FACTOR = 10.0 / AGE_SPAN
    age_min_label, age_max_label = map(int, AGES)
    age_min_column = f"standardized_mean_predicted_median_md_age{age_min_label}"
    age_max_column = f"standardized_mean_predicted_median_md_age{age_max_label}"
    effect_column = f"average_md_change_per_decade_ages{age_min_label}_{age_max_label}"
    started = time.monotonic()
    root = args.result_root
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(root / "ico4_node_metadata.csv")
    if len(metadata) != 4687 or not np.array_equal(metadata.node_index, np.arange(4687)):
        raise RuntimeError("Unexpected ico4 node metadata")
    manifest = pd.read_csv(root / "frozen_cohort_manifest.csv", dtype={"subject_id": str})
    manifest = manifest.loc[manifest.included].copy()
    subject_order = pd.read_csv(args.subject_order, dtype={"subject_id": str})
    cohort = manifest.merge(
        subject_order[["subject_id", "row_index"]], on="subject_id", validate="one_to_one"
    ).sort_values("row_index").reset_index(drop=True)
    if len(cohort) != 186 or set(cohort.sex_code.astype(int)) != {1, 2}:
        raise RuntimeError("Frozen cohort or sex-code levels differ from the completed model")
    if not (cohort.age.min() <= AGES[0] < AGES[1] <= cohort.age.max()):
        raise RuntimeError("Target ages lie outside the observed cohort range")

    all_thickness = np.load(
        root / "inputs" / "X_ico4_local_freesurfer_thickness.npy", mmap_mode="r"
    )
    thickness = np.asarray(all_thickness[cohort.row_index.to_numpy(int), :], float)
    if thickness.shape != (186, 4687):
        raise RuntimeError("Local-thickness matrix does not match the frozen cohort and nodes")
    support = support_table(metadata, thickness, cohort)
    support.to_parquet(out / "node_covariate_support.parquet", index=False)
    support.to_csv(out / "node_covariate_support.csv", index=False)
    eligible = support.eligible_exact_186_subject_estimand.to_numpy(bool)

    sex_code = cohort.sex_code.to_numpy(int)
    prediction_rows: list[dict[str, object]] = []
    successful_models = 0
    for node, meta in enumerate(metadata.itertuples(index=False)):
        path = parameter_path(root, node, str(meta.hemisphere))
        params = pd.read_parquet(path).sort_values("depth_index")
        if len(params) != 21 or not np.array_equal(params.depth_index, np.arange(21)):
            raise RuntimeError(f"Incomplete parameter shard for node {node}")
        local_thickness = thickness[:, node]
        for param in params.itertuples(index=False):
            base = {
                "node_index": node,
                "hemisphere": str(meta.hemisphere),
                "ico_node_id": int(meta.ico_node_id),
                "fsaverage_vertex_id": int(meta.fsaverage_vertex_id),
                "dk_label_id": int(meta.dk_label_id),
                "dk_label_name": str(meta.dk_label_name),
                "depth_index": int(param.depth_index),
                "normalized_depth": float(param.depth_fraction),
                "n_model_train": int(param.n_train),
                "n_covariate_subjects": int(np.isfinite(local_thickness).sum()),
                "eligible_exact_186_subject_estimand": bool(eligible[node]),
                "exclusion_reason": "" if eligible[node] else "node_local_thickness_missing_for_one_or_more_frozen_cohort_subjects",
            }
            if eligible[node]:
                age_min_prediction, age_max_prediction, n_prediction = predict_node_depth(
                    param, sex_code, local_thickness
                )
                effect = (age_max_prediction - age_min_prediction) * DECADE_FACTOR
                if not np.isfinite([age_min_prediction, age_max_prediction, effect]).all() or n_prediction != 186:
                    raise RuntimeError(f"Invalid effect at node={node}, depth={param.depth_index}")
                base.update({
                    "n_prediction_subjects": n_prediction,
                    age_min_column: age_min_prediction,
                    age_max_column: age_max_prediction,
                    effect_column: effect,
                })
                successful_models += 1
            else:
                base.update({
                    "n_prediction_subjects": 0,
                    age_min_column: np.nan,
                    age_max_column: np.nan,
                    effect_column: np.nan,
                })
            prediction_rows.append(base)
        if (node + 1) % 100 == 0 or node + 1 == len(metadata):
            print(json.dumps({
                "nodes_processed": node + 1,
                "eligible_nodes": int(eligible[: node + 1].sum()),
                "successful_models": successful_models,
                "elapsed_minutes": round((time.monotonic() - started) / 60, 3),
            }), flush=True)

    predictions = pd.DataFrame(prediction_rows)
    expected_rows = 4687 * 21
    if len(predictions) != expected_rows or predictions.duplicated(
        ["node_index", "depth_index"]
    ).any():
        raise RuntimeError("Node-by-depth output key failure")
    finite_rows = predictions.eligible_exact_186_subject_estimand
    value_columns = [
        age_min_column,
        age_max_column,
        effect_column,
    ]
    if not np.isfinite(predictions.loc[finite_rows, value_columns].to_numpy(float)).all():
        raise RuntimeError("A prediction marked eligible is nonfinite")
    if predictions.loc[~finite_rows, value_columns].notna().any().any():
        raise RuntimeError("An ineligible node unexpectedly has an estimand")
    predictions.to_parquet(out / "node_depth_standardized_age_effect.parquet", index=False)
    predictions.to_csv(out / "node_depth_standardized_age_effect.csv.gz", index=False)

    depth_to_band = {depth: band for band, depths in BANDS.items() for depth in depths}
    predictions["depth_band"] = predictions.depth_index.map(depth_to_band)
    band = (
        predictions.groupby(
            [
                "node_index", "hemisphere", "ico_node_id", "fsaverage_vertex_id",
                "dk_label_id", "dk_label_name", "eligible_exact_186_subject_estimand",
                "exclusion_reason", "depth_band",
            ],
            observed=True,
            sort=False,
            dropna=False,
        )
        .agg(**{
            "n_depths": ("depth_index", "size"),
            "depth_indices": ("depth_index", lambda x: ",".join(map(str, sorted(x)))),
            f"mean_standardized_md_age{age_min_label}": (age_min_column, "mean"),
            f"mean_standardized_md_age{age_max_label}": (age_max_column, "mean"),
            "depth_band_mean_average_md_change_per_decade": (effect_column, "mean"),
        })
        .reset_index()
    )
    band["depth_band"] = pd.Categorical(
        band.depth_band, categories=list(BANDS), ordered=True
    )
    band = band.sort_values(["node_index", "depth_band"]).reset_index(drop=True)
    if len(band) != 4687 * 3 or not band.n_depths.eq(7).all():
        raise RuntimeError("Depth-band aggregation did not yield three seven-depth bands per node")
    band.to_parquet(out / "node_depth_band_standardized_age_effect.parquet", index=False)
    band.to_csv(out / "node_depth_band_standardized_age_effect.csv", index=False)

    age_support_rows: list[dict[str, object]] = []
    for age in AGES:
        hit = np.abs(cohort.age.to_numpy(float) - age) <= 3
        subset = cohort.loc[hit]
        age_support_rows.append({
            "target_age": int(age),
            "window_rule": "absolute age difference <= 3 years",
            "window_min": int(age - 3),
            "window_max": int(age + 3),
            "n_subjects": len(subset),
            "n_sex_code_1": int(subset.sex_code.eq(1).sum()),
            "n_sex_code_2": int(subset.sex_code.eq(2).sum()),
            "observed_age_min": float(subset.age.min()),
            "observed_age_max": float(subset.age.max()),
        })
    age_support = pd.DataFrame(age_support_rows)
    age_support.to_csv(out / "age_support.csv", index=False)

    curves = pd.read_parquet(root / "full_models" / "age_curve_shape_diagnostics.parquet")
    relative = curves.relative_nonlinearity.replace([np.inf, -np.inf], np.nan)
    nonlinearity = {
        "formula": "max_age |p50_md(age) - OLS_linear_fit(age)| / max(|p50_md(76) - p50_md(45)|, machine_epsilon)",
        "age_grid": "integer ages 45 through 76 inclusive",
        "source_curves": "saved node x depth x sex curves at node-local median thickness",
        "role": "recorded diagnostic only; not a prerequisite for the standardized estimand",
        "n_curves": int(len(curves)),
        "n_finite": int(relative.notna().sum()),
        "median": float(relative.median()),
        "p95": float(relative.quantile(0.95)),
        "monotonic_fraction": float(curves.monotonic.mean()),
    }

    outside_summary: dict[str, object] = {}
    for age in map(int, AGES):
        column = f"age{age}_all_subject_values_outside_nearby_range_fraction"
        values = support.loc[eligible, column]
        outside_summary[str(age)] = {
            "nearby_window_years": 3,
            "n_nearby_subjects": int(age_support.loc[age_support.target_age.eq(age), "n_subjects"].iloc[0]),
            "definition": "fraction of the same 186 subject-specific thickness values outside the observed min-max among subjects within +/-3 years",
            "nodes_assessed": int(len(values)),
            "nodes_with_any_outside": int(values.gt(0).sum()),
            "nodes_with_fraction_over_0_10": int(values.gt(0.10).sum()),
            "median_fraction_outside": float(values.median()),
            "p95_fraction_outside": float(values.quantile(0.95)),
            "maximum_fraction_outside": float(values.max()),
        }

    report = {
        "status": "complete",
        "estimand": f"equal-weight mean across the same frozen subjects of original-scale conditional predicted medians; age {age_max_label} minus age {age_min_label}, multiplied by 10/{AGE_SPAN:g}",
        "interpretation": "standardized predicted-median mean difference per decade; descriptive cross-sectional conditional age association, not longitudinal change and not a mixture-distribution median",
        "cohort_n": 186,
        "observed_age_range": [float(cohort.age.min()), float(cohort.age.max())],
        "target_ages": list(map(int, AGES)),
        "age_span_years": AGE_SPAN,
        "per_decade_multiplier": DECADE_FACTOR,
        "age_support_file": "age_support.csv",
        "sex_codes_used_as_stored": sorted(map(int, cohort.sex_code.unique())),
        "sex_semantic_label_policy": "numeric stored codes retained; no male/female semantic mapping inferred",
        "same_subjects_weights_sex_thickness_at_both_ages": True,
        "weight_per_subject": 1 / 186,
        "local_thickness_rule": "each subject's actual mapped node-local FreeSurfer thickness, unchanged between ages and shared across all 21 depths",
        "missing_covariate_rule": "no imputation and no available-case reweighting; a node is ineligible unless all 186 local-thickness values are finite",
        "n_nodes_fixed_mask": 4687,
        "n_nodes_exact_186_estimand": int(eligible.sum()),
        "n_nodes_excluded_missing_local_thickness": int((~eligible).sum()),
        "n_node_depth_models_predicted": successful_models,
        "n_node_depth_rows_total": expected_rows,
        "all_eligible_predictions_and_effects_finite": True,
        "model": "saved full-cohort PCNToolkit 1.3.0 Warped M2",
        "preprocessing": "saved per-model age, thickness, and response centering/scaling restored without refitting",
        "inverse_warp": "WarpSinhArcsinh warp_predictions median on scaled response, then response_scale and response_mean restore original MD units before subject averaging",
        "raw_md_unit": "mm^2/s",
        "display_md_unit": "x10^-3 mm^2/s",
        "depth_bands": {
            band_name: {
                "depth_indices": list(depths),
                "normalized_depth_min": min(depths) / 20,
                "normalized_depth_max": max(depths) / 20,
                "weights": "equal, 1/7",
            }
            for band_name, depths in BANDS.items()
        },
        "aggregation_order": f"predict original-scale median for every subject at each age and depth; average 186 subjects; difference and multiply 10/{AGE_SPAN:g}; then equal-weight average seven depth effects",
        "thickness_support_against_nearby_ages": outside_summary,
        "relative_nonlinearity": nonlinearity,
        "inference_boundary": "effect-size maps only; no significance marks and no test of depth-band differences",
        "elapsed_minutes": (time.monotonic() - started) / 60,
    }
    atomic_json(out / "computation_qc.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
