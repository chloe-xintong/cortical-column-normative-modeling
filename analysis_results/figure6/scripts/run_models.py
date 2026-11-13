#!/usr/bin/env python3
"""Run 5-fold OOF PCNToolkit BLR models for all 68 regional profiles.

Description
-----------
Run 5-fold OOF PCNToolkit BLR models for all 68 regional profiles.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--subject-order`` command-line input
- ``--cohort-audit`` command-line input
- ``--cohort-manifest`` command-line input
- ``--regional-profile`` command-line input
- ``--regional-thickness`` command-line input
- ``--profile-metadata`` command-line input
- ``--thickness-metadata`` command-line input
- ``--folds`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/run_models.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import pcntoolkit
import scipy
import xarray as xr
from pcntoolkit import BLR

from pilot_utils import MODEL_SPECS, age_bin, depth_band, write_json

_CONFIG: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-order", type=Path, required=True)
    parser.add_argument("--cohort-audit", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--regional-profile", type=Path, required=True)
    parser.add_argument("--regional-thickness", type=Path, required=True)
    parser.add_argument("--profile-metadata", type=Path, required=True)
    parser.add_argument("--thickness-metadata", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--regions", type=str, default="all", help="all, comma-separated indices, or start:end")
    return parser.parse_args()


def parse_regions(value: str) -> list[int]:
    if value == "all":
        return list(range(68))
    if ":" in value:
        start, stop = (int(x) for x in value.split(":"))
        result = list(range(start, stop))
    else:
        result = [int(x) for x in value.split(",")]
    if not result or min(result) < 0 or max(result) >= 68:
        raise ValueError("Region indices must be in [0, 67]")
    return result


def initialize_worker(config: dict[str, object]) -> None:
    global _CONFIG
    _CONFIG = config


def standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0, ddof=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise RuntimeError("Nonpositive training-fold covariate scale")
    return (train - mean) / scale, (test - mean) / scale, mean, scale


def spline_matrices(train_age: np.ndarray, test_age: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object], patsy.DesignInfo]:
    train_frame = pd.DataFrame({"age": train_age})
    train_design = patsy.dmatrix(
        "cr(age, df=3, constraints='center') - 1", train_frame, return_type="dataframe"
    )
    test_design = patsy.build_design_matrices(
        [train_design.design_info], pd.DataFrame({"age": test_age}), return_type="dataframe"
    )[0]
    transform_state = None
    for factor in train_design.design_info.factor_infos.values():
        transforms = factor.state.get("transforms", {})
        if transforms:
            transform_state = next(iter(transforms.values()))
            break
    state = {
        "columns": list(train_design.design_info.column_names),
        "training_age_min": float(np.min(train_age)), "training_age_max": float(np.max(train_age)),
        "all_knots": getattr(transform_state, "_all_knots", np.array([])).tolist(),
        "constraints": np.asarray(getattr(transform_state, "_constraints", np.array([]))).tolist(),
    }
    return np.asarray(train_design), np.asarray(test_design), state, train_design.design_info


def build_x(model: str, train: pd.DataFrame, test: pd.DataFrame, region_index: int,
            thickness_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object], patsy.DesignInfo | None]:
    spec = MODEL_SPECS[model]
    if spec["age_form"] == "linear":
        age_train, age_test, age_mean, age_scale = standardize_train_test(
            train[["age"]].to_numpy(float), test[["age"]].to_numpy(float)
        )
        transform = {"age_mean": age_mean.tolist(), "age_scale": age_scale.tolist()}
        design_info = None
    else:
        raw_train, raw_test, spline_state, design_info = spline_matrices(
            train.age.to_numpy(float), test.age.to_numpy(float)
        )
        age_train, age_test, age_mean, age_scale = standardize_train_test(raw_train, raw_test)
        transform = {"spline": spline_state, "spline_column_mean": age_mean.tolist(),
                     "spline_column_scale": age_scale.tolist()}
    columns_train = [age_train, (train.sex_code.to_numpy(float) == 2).astype(float)[:, None]]
    columns_test = [age_test, (test.sex_code.to_numpy(float) == 2).astype(float)[:, None]]
    covariate_names = (["age_z"] if spec["age_form"] == "linear" else [f"age_cr_df3_{i + 1}_z" for i in range(3)]) + ["sex_code_2"]
    if spec["thickness"]:
        train_thickness = thickness_matrix[train.row_index.to_numpy(int), region_index][:, None].astype(float)
        test_thickness = thickness_matrix[test.row_index.to_numpy(int), region_index][:, None].astype(float)
        ttrain, ttest, tmean, tscale = standardize_train_test(train_thickness, test_thickness)
        columns_train.append(ttrain); columns_test.append(ttest); covariate_names.append("local_thickness_z")
        transform.update({"thickness_mean": tmean.tolist(), "thickness_scale": tscale.tolist()})
    transform["covariate_names"] = covariate_names
    return np.column_stack(columns_train), np.column_stack(columns_test), transform, design_info


def curve_x(model: str, ages: np.ndarray, sex_code: int, transform: dict[str, object],
            design_info: patsy.DesignInfo | None) -> np.ndarray:
    spec = MODEL_SPECS[model]
    if spec["age_form"] == "linear":
        age_part = (ages[:, None] - np.asarray(transform["age_mean"])) / np.asarray(transform["age_scale"])
    else:
        raw = np.asarray(patsy.build_design_matrices(
            [design_info], pd.DataFrame({"age": ages}), return_type="dataframe"
        )[0])
        age_part = (raw - np.asarray(transform["spline_column_mean"])) / np.asarray(transform["spline_column_scale"])
    pieces = [age_part, np.full((len(ages), 1), float(sex_code == 2))]
    if spec["thickness"]:
        pieces.append(np.zeros((len(ages), 1)))
    return np.column_stack(pieces)


def worker(task: tuple[int, str, int]) -> dict[str, object]:
    region_index, model_name, fold = task
    started = time.monotonic()
    try:
        subject_order = pd.read_csv(_CONFIG["subject_order"], dtype={"subject_id": str})
        audit = pd.read_csv(_CONFIG["cohort_audit"], dtype={"subject_id": str})
        cohort = pd.read_csv(_CONFIG["cohort_manifest"], dtype={"subject_id": str})
        profile_meta = pd.read_csv(_CONFIG["profile_metadata"])
        thickness_meta = pd.read_csv(_CONFIG["thickness_metadata"])
        profile_matrix = np.load(_CONFIG["regional_profile"], mmap_mode="r")
        thickness_matrix = np.load(_CONFIG["regional_thickness"], mmap_mode="r")
        with Path(_CONFIG["folds"]).open() as stream:
            folds = json.load(stream)
        included = audit.loc[audit.included].merge(
            subject_order[["subject_id", "row_index"]], on="subject_id", validate="one_to_one"
        ).merge(cohort[["subject_id", "SUBJECT_SEX"]], on="subject_id", validate="one_to_one")
        included["sex_code"] = included.SUBJECT_SEX.astype(int)
        included["age_bin"] = age_bin(included.age).astype(str)
        fold_info = next(item for item in folds["folds"] if item["outer_fold"] == fold)
        train = included[included.subject_id.isin(fold_info["train_subject_ids"])].copy().reset_index(drop=True)
        test = included[included.subject_id.isin(fold_info["test_subject_ids"])].copy().reset_index(drop=True)
        if len(train) + len(test) != len(included):
            raise RuntimeError("Fold train/test subject mismatch")
        x_train, x_test, transform, design_info = build_x(
            model_name, train, test, region_index, thickness_matrix
        )
        feature_start = region_index * 21
        y_train = np.asarray(profile_matrix[train.row_index.to_numpy(int), feature_start:feature_start + 21], float)
        y_test = np.asarray(profile_matrix[test.row_index.to_numpy(int), feature_start:feature_start + 21], float)
        if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
            raise RuntimeError("MD response contains missing values; unified cohort rule requires an audit stop")
        y_mean = y_train.mean(axis=0)
        y_scale = y_train.std(axis=0, ddof=0)
        if np.any(y_scale <= 0):
            raise RuntimeError("Response has nonpositive training-fold scale")
        be_train = xr.DataArray(np.zeros((len(train), 1), dtype=int), dims=("observations", "batch_effect_dims"))
        be_test = np.zeros((len(test), 1), dtype=int)
        be_maps = {"dummy_batch_effect": {"0": 0}}
        x_train_da = xr.DataArray(x_train, dims=("observations", "covariates"))

        region = thickness_meta.iloc[region_index]
        depths = profile_meta.iloc[feature_start:feature_start + 21].reset_index(drop=True)
        predictions: list[dict[str, object]] = []
        effects: list[dict[str, object]] = []
        outcome_diagnostics: list[dict[str, object]] = []
        age_pair = np.array([40.0, 78.0])
        curve_by_sex = {code: curve_x(model_name, age_pair, code, transform, design_info) for code in (1, 2)}
        for depth_index in range(21):
            response_train = (y_train[:, depth_index] - y_mean[depth_index]) / y_scale[depth_index]
            blr = BLR(name=f"{model_name}_r{region_index:02d}_d{depth_index:02d}_f{fold}",
                      heteroskedastic=False, ard=False, optimizer="l-bfgs-b", n_iter=100, tol=1e-3)
            blr.fit(x_train_da, be_train, be_maps, xr.DataArray(response_train, dims=("observations",)))
            mean_zscale, total_var_zscale = blr.ys_s2(x_test, be_test)
            _, beta, _ = blr.parse_hyps(blr.hyp, blr.Phi_Phi_var(x_test, be_test)[0], np.zeros((len(test), 1)))
            noise_zscale = np.full(len(test), 1.0 / float(np.ravel(beta)[0]))
            mean_md = mean_zscale * y_scale[depth_index] + y_mean[depth_index]
            total_var_md = total_var_zscale * y_scale[depth_index] ** 2
            noise_var_md = noise_zscale * y_scale[depth_index] ** 2
            parameter_var_md = total_var_md - noise_var_md
            observed = y_test[:, depth_index]
            zscore = (observed - mean_md) / np.sqrt(total_var_md)
            logp = -0.5 * (np.log(2 * np.pi * total_var_md) + np.square(observed - mean_md) / total_var_md)
            for subject_i, (_, subject) in enumerate(test.iterrows()):
                predictions.append({
                    "subject_id": subject.subject_id, "age": float(subject.age), "age_bin": subject.age_bin,
                    "sex_code": int(subject.sex_code), "outer_fold": fold, "model": model_name,
                    "region_index": region_index, "hemisphere": region.hemisphere,
                    "dk_label_id": int(region.dk_label_id), "dk_label_name": region.dk_label_name,
                    "depth_index": depth_index, "depth": float(depths.iloc[depth_index].depth_fraction),
                    "depth_band": depth_band(float(depths.iloc[depth_index].depth_fraction)),
                    "local_thickness": float(thickness_matrix[int(subject.row_index), region_index]),
                    "observed_md": float(observed[subject_i]), "pred_mean": float(mean_md[subject_i]),
                    "pred_var": float(total_var_md[subject_i]), "noise_var": float(noise_var_md[subject_i]),
                    "parameter_var": float(parameter_var_md[subject_i]), "z": float(zscore[subject_i]),
                    "logp": float(logp[subject_i]), "nlpd": float(-logp[subject_i]),
                })
            curve_predictions = []
            for code in (1, 2):
                curve_mean, _ = blr.ys_s2(curve_by_sex[code], np.zeros((2, 1), dtype=int))
                curve_predictions.append(curve_mean * y_scale[depth_index] + y_mean[depth_index])
            average_curve = np.mean(curve_predictions, axis=0)
            effects.append({"model": model_name, "outer_fold": fold, "region_index": region_index,
                            "hemisphere": region.hemisphere, "dk_label_id": int(region.dk_label_id),
                            "dk_label_name": region.dk_label_name, "depth_index": depth_index,
                            "depth": float(depths.iloc[depth_index].depth_fraction),
                            "predicted_change_age_40_to_78": float(average_curve[1] - average_curve[0])})
            thickness_position = len(transform["covariate_names"]) - 1 if MODEL_SPECS[model_name]["thickness"] else None
            raw_thickness_beta = ""
            if thickness_position is not None:
                raw_thickness_beta = float(
                    blr.m[thickness_position] * y_scale[depth_index] / np.asarray(transform["thickness_scale"])[0]
                )
            outcome_diagnostics.append({
                "model": model_name, "outer_fold": fold, "region_index": region_index,
                "depth_index": depth_index, "status": "complete", "negative_log_evidence": float(blr.nlZ),
                "noise_precision_scaled": float(np.ravel(beta)[0]), "raw_thickness_beta": raw_thickness_beta,
                "hyperparameters": json.dumps(blr.hyp.tolist()),
            })
        return {"status": "complete", "region_index": region_index, "model": model_name, "outer_fold": fold,
                "predictions": predictions, "effects": effects, "outcome_diagnostics": outcome_diagnostics,
                "transform": transform, "elapsed_seconds": time.monotonic() - started}
    except Exception as error:
        return {"status": "failed", "region_index": region_index, "model": model_name, "outer_fold": fold,
                "error": repr(error), "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regions = parse_regions(args.regions)
    config = {key: str(getattr(args, key)) for key in (
        "subject_order", "cohort_audit", "cohort_manifest", "regional_profile", "regional_thickness",
        "profile_metadata", "thickness_metadata", "folds"
    )}
    tasks = [(region, model, fold) for region in regions for model in MODEL_SPECS for fold in range(5)]
    predictions: list[dict[str, object]] = []
    effects: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    transforms: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.n_jobs, initializer=initialize_worker, initargs=(config,)) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        for completed_i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["status"] == "failed":
                failures.append({key: result[key] for key in ("region_index", "model", "outer_fold", "error", "traceback", "elapsed_seconds")})
            else:
                predictions.extend(result["predictions"])
                effects.extend(result["effects"])
                diagnostics.extend(result["outcome_diagnostics"])
                transforms.append({"region_index": result["region_index"], "model": result["model"],
                                   "outer_fold": result["outer_fold"], "transform": result["transform"]})
            if completed_i % 20 == 0 or completed_i == len(tasks):
                print(json.dumps({"completed_tasks": completed_i, "total_tasks": len(tasks), "failures": len(failures)}), flush=True)

    region_tag = "all" if len(regions) == 68 else f"{min(regions):02d}_{max(regions):02d}"
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_parquet(shard_dir / f"predictions_{region_tag}.parquet", index=False)
    pd.DataFrame(effects).to_csv(shard_dir / f"age_effects_{region_tag}.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(shard_dir / f"model_diagnostics_{region_tag}.csv", index=False)
    write_json(shard_dir / f"fold_transforms_{region_tag}.json", transforms)
    write_json(shard_dir / f"failures_{region_tag}.json", failures)
    run_report = {
        "status": "failed" if failures else "complete", "regions": regions, "n_tasks": len(tasks),
        "n_predictions": len(predictions), "n_failures": len(failures), "elapsed_seconds": time.monotonic() - started,
        "python_version": platform.python_version(), "pcntoolkit_version": pcntoolkit.__version__,
        "numpy_version": np.__version__, "pandas_version": pd.__version__, "scipy_version": scipy.__version__,
        "patsy_version": patsy.__version__, "n_jobs": args.n_jobs,
        "predictive_variance_source": "BLR.ys_s2 = observation variance (1/beta) + posterior parameter uncertainty",
    }
    write_json(shard_dir / f"run_report_{region_tag}.json", run_report)
    if failures:
        raise RuntimeError(f"{len(failures)} region/model/fold tasks failed; see failures_{region_tag}.json")
    print(json.dumps(run_report))


if __name__ == "__main__":
    main()
