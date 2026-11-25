#!/usr/bin/env python3
"""Run leakage-safe regional Warped spline or constrained Gompertz candidates.

Description
-----------
Run leakage-safe regional Warped spline or constrained Gompertz candidates.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--candidate`` command-line input
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/run_warped_age_candidates.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pcntoolkit
from scipy import optimize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from pilot_utils import age_bin, depth_band, write_json  # noqa: E402
from run_models import build_x, parse_regions, standardize_train_test  # noqa: E402
from warped_age_utils import (fit_warped_blr, predict_warped,
                              standardized_gompertz)  # noqa: E402


PILOT_REGIONS = [4, 19, 26, 38, 54, 57]
PILOT_DEPTHS = [0, 4, 10, 16, 20]
GOMPERTZ_RATE_BOUNDS = (0.01, 0.50)  # per year
_CONFIG: dict[str, str] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("spline", "gompertz"), required=True)
    for name in ("subject-order", "cohort-audit", "cohort-manifest", "regional-profile",
                 "regional-thickness", "profile-metadata", "thickness-metadata", "folds"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regions", default=None)
    parser.add_argument("--depths", default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def parse_indices(value: str | None, default: list[int], upper: int) -> list[int]:
    if value is None:
        return list(default)
    if value == "all":
        return list(range(upper))
    result = [int(v) for v in value.split(",")]
    if not result or min(result) < 0 or max(result) >= upper or len(result) != len(set(result)):
        raise ValueError(f"Indices must be unique and in [0, {upper - 1}]")
    return result


def initialize_worker(config: dict[str, str]) -> None:
    global _CONFIG
    _CONFIG = config


def load_task_data(region_index: int, fold: int) -> dict[str, object]:
    subject_order = pd.read_csv(_CONFIG["subject_order"], dtype={"subject_id": str})
    audit = pd.read_csv(_CONFIG["cohort_audit"], dtype={"subject_id": str})
    cohort = pd.read_csv(_CONFIG["cohort_manifest"], dtype={"subject_id": str})
    folds = json.loads(Path(_CONFIG["folds"]).read_text(encoding="utf-8"))
    included = audit.loc[audit.included].merge(
        subject_order[["subject_id", "row_index"]], on="subject_id", validate="one_to_one"
    ).merge(cohort[["subject_id", "SUBJECT_SEX"]], on="subject_id", validate="one_to_one")
    included["sex_code"] = included.SUBJECT_SEX.astype(int)
    included["age_bin"] = age_bin(included.age).astype(str)
    fold_info = next(x for x in folds["folds"] if int(x["outer_fold"]) == fold)
    train = included[included.subject_id.isin(fold_info["train_subject_ids"])].copy().reset_index(drop=True)
    test = included[included.subject_id.isin(fold_info["test_subject_ids"])].copy().reset_index(drop=True)
    if len(train) + len(test) != 186 or set(train.subject_id) & set(test.subject_id):
        raise RuntimeError("Invalid frozen 186-person train/test partition")
    profile = np.load(_CONFIG["regional_profile"], mmap_mode="r")
    thickness = np.load(_CONFIG["regional_thickness"], mmap_mode="r")
    start = region_index * 21
    y_train = np.asarray(profile[train.row_index.to_numpy(int), start:start + 21], float)
    y_test = np.asarray(profile[test.row_index.to_numpy(int), start:start + 21], float)
    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        raise RuntimeError("Nonfinite MD in frozen cohort")
    return {"train": train, "test": test, "thickness": thickness,
            "y_train": y_train, "y_test": y_test}


def metadata(region_index: int, depth_index: int) -> dict[str, object]:
    regions = pd.read_csv(_CONFIG["thickness_metadata"])
    profiles = pd.read_csv(_CONFIG["profile_metadata"])
    region = regions.iloc[region_index]
    depth = float(profiles.iloc[region_index * 21 + depth_index].depth_fraction)
    return {"region_index": region_index, "hemisphere": region.hemisphere,
            "dk_label_id": int(region.dk_label_id), "dk_label_name": region.dk_label_name,
            "depth_index": depth_index, "depth": depth, "depth_band": depth_band(depth)}


def prediction_rows(pred: pd.DataFrame, test: pd.DataFrame, observed: np.ndarray,
                    thickness: np.ndarray, fold: int, model_name: str,
                    meta: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for i, subject in test.iterrows():
        row = {"subject_id": subject.subject_id, "age": float(subject.age),
               "age_bin": subject.age_bin, "sex_code": int(subject.sex_code),
               "outer_fold": fold, "model": model_name,
               "local_thickness": float(thickness[int(subject.row_index), int(meta["region_index"])]),
               "observed_md": float(observed[i]), **meta}
        row.update({key: float(value) for key, value in pred.iloc[i].items()})
        rows.append(row)
    return rows


def spline_worker(task: tuple[int, int, tuple[int, ...]]) -> dict[str, object]:
    region_index, fold, depths = task
    started = time.monotonic()
    try:
        data = load_task_data(region_index, fold)
        train, test = data["train"], data["test"]
        x_train, x_test, transform, _ = build_x(
            "M3", train, test, region_index, data["thickness"]
        )
        rows, diagnostics = [], []
        for depth_index in depths:
            y_train = data["y_train"][:, depth_index]
            y_test = data["y_test"][:, depth_index]
            y_mean, y_scale = float(y_train.mean()), float(y_train.std(ddof=0))
            fitted = fit_warped_blr(
                x_train, (y_train - y_mean) / y_scale,
                f"warped_spline_r{region_index:02d}_d{depth_index:02d}_f{fold}",
            )
            pred = predict_warped(fitted.model, x_test, (y_test - y_mean) / y_scale, y_mean, y_scale)
            meta = metadata(region_index, depth_index)
            rows.extend(prediction_rows(pred, test, y_test, data["thickness"], fold,
                                        "warped_spline_df3", meta))
            diagnostics.append({"model": "warped_spline_df3", "outer_fold": fold, **meta,
                                "status": "complete", "negative_log_evidence": float(fitted.model.nlZ),
                                "warp_epsilon": float(fitted.model.gamma[0]),
                                "warp_log_b": float(fitted.model.gamma[1]),
                                "warp_b": float(np.exp(fitted.model.gamma[1])),
                                "warnings": json.dumps(fitted.warnings),
                                "hyperparameters": json.dumps(fitted.model.hyp.tolist())})
        return {"status": "complete", "region_index": region_index, "outer_fold": fold,
                "predictions": rows, "diagnostics": diagnostics, "transform": transform,
                "elapsed_seconds": time.monotonic() - started}
    except Exception as error:
        return {"status": "failed", "region_index": region_index, "outer_fold": fold,
                "error": repr(error), "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started}


def gompertz_design(train: pd.DataFrame, test: pd.DataFrame, thickness: np.ndarray,
                     region_index: int, rate: float, inflection_age: float
                     ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    g_train, g_test, g_mean, g_scale = standardized_gompertz(
        train.age.to_numpy(float), test.age.to_numpy(float), rate, inflection_age
    )
    t_train = thickness[train.row_index.to_numpy(int), region_index][:, None].astype(float)
    t_test = thickness[test.row_index.to_numpy(int), region_index][:, None].astype(float)
    tz_train, tz_test, t_mean, t_scale = standardize_train_test(t_train, t_test)
    x_train = np.column_stack([g_train, (train.sex_code.to_numpy(int) == 2).astype(float), tz_train])
    x_test = np.column_stack([g_test, (test.sex_code.to_numpy(int) == 2).astype(float), tz_test])
    state = {"gompertz_rate_per_year": rate, "gompertz_inflection_age": inflection_age,
             "gompertz_basis_mean": g_mean, "gompertz_basis_scale": g_scale,
             "thickness_mean": float(t_mean[0]), "thickness_scale": float(t_scale[0])}
    return x_train, x_test, state


def gompertz_worker(task: tuple[int, int, int]) -> dict[str, object]:
    region_index, depth_index, fold = task
    started = time.monotonic()
    evaluations: list[dict[str, object]] = []
    try:
        data = load_task_data(region_index, fold)
        train, test = data["train"], data["test"]
        y_train = data["y_train"][:, depth_index]
        y_test = data["y_test"][:, depth_index]
        y_mean, y_scale = float(y_train.mean()), float(y_train.std(ddof=0))
        response_train = (y_train - y_mean) / y_scale
        lower = np.array([np.log(GOMPERTZ_RATE_BOUNDS[0]), float(train.age.min())])
        upper = np.array([np.log(GOMPERTZ_RATE_BOUNDS[1]), float(train.age.max())])

        cache: dict[tuple[float, float], tuple[float, list[str]]] = {}
        def objective(theta: np.ndarray) -> float:
            key = (round(float(theta[0]), 8), round(float(theta[1]), 8))
            if key in cache:
                return cache[key][0]
            rate, inflection = float(np.exp(theta[0])), float(theta[1])
            try:
                x_train, _, _ = gompertz_design(train, test, data["thickness"], region_index,
                                                 rate, inflection)
                fitted = fit_warped_blr(
                    x_train, response_train,
                    f"gompertz_search_r{region_index:02d}_d{depth_index:02d}_f{fold}",
                )
                value = float(fitted.model.nlZ)
                messages = fitted.warnings
                if not np.isfinite(value):
                    value = 1e12
            except Exception as error:
                value, messages = 1e12, [repr(error)]
            cache[key] = (value, messages)
            evaluations.append({"log_rate": key[0], "rate_per_year": rate,
                                "inflection_age": inflection, "negative_log_evidence": value,
                                "warnings": messages})
            return value

        age_quantiles = np.quantile(train.age.to_numpy(float), [0.25, 0.50, 0.75])
        starts = [np.array([np.log(rate), age0]) for rate in (0.03, 0.08, 0.20)
                  for age0 in age_quantiles]
        initial = [(objective(start), start) for start in starts]
        local_results = []
        for _, start in sorted(initial, key=lambda item: item[0])[:3]:
            result = optimize.minimize(
                objective, start, method="L-BFGS-B", bounds=list(zip(lower, upper)),
                options={"maxiter": 30, "ftol": 1e-6, "gtol": 1e-4,
                         "maxls": 10, "eps": np.array([0.03, 0.20])},
            )
            local_results.append(result)
        best = min(local_results, key=lambda result: float(result.fun))
        log_rate, inflection = map(float, best.x)
        rate = float(np.exp(log_rate))
        x_train, x_test, state = gompertz_design(
            train, test, data["thickness"], region_index, rate, inflection
        )
        fitted = fit_warped_blr(
            x_train, response_train,
            f"warped_gompertz_r{region_index:02d}_d{depth_index:02d}_f{fold}",
        )
        pred = predict_warped(fitted.model, x_test, (y_test - y_mean) / y_scale, y_mean, y_scale)
        meta = metadata(region_index, depth_index)
        rows = prediction_rows(pred, test, y_test, data["thickness"], fold,
                               "warped_gompertz", meta)
        age_beta = float(fitted.model.m[1])  # intercept is column 0 in PCNToolkit design matrix
        start_solutions = np.array([r.x for r in local_results], dtype=float)
        at_bound = bool(np.any(np.isclose(best.x, lower, atol=[0.03, 0.2])) or
                        np.any(np.isclose(best.x, upper, atol=[0.03, 0.2])))
        diagnostics = {"model": "warped_gompertz", "outer_fold": fold, **meta,
                       "status": "complete", "negative_log_evidence": float(fitted.model.nlZ),
                       **state, "age_beta_standardized": age_beta,
                       "trend_direction": "increasing" if age_beta > 0 else "decreasing",
                       "outer_optimizer_success": bool(best.success),
                       "outer_optimizer_message": str(best.message),
                       "outer_optimizer_nit": int(best.nit), "shape_at_bound": at_bound,
                       "multistart_log_rate_range": float(np.ptp(start_solutions[:, 0])),
                       "multistart_inflection_age_range": float(np.ptp(start_solutions[:, 1])),
                       "objective_evaluations": len(cache),
                       "warp_epsilon": float(fitted.model.gamma[0]),
                       "warp_log_b": float(fitted.model.gamma[1]),
                       "warp_b": float(np.exp(fitted.model.gamma[1])),
                       "warnings": json.dumps(fitted.warnings),
                       "hyperparameters": json.dumps(fitted.model.hyp.tolist())}
        return {"status": "complete", "region_index": region_index, "depth_index": depth_index,
                "outer_fold": fold, "predictions": rows, "diagnostics": [diagnostics],
                "transform": state, "search_trace": evaluations,
                "elapsed_seconds": time.monotonic() - started}
    except Exception as error:
        return {"status": "failed", "region_index": region_index, "depth_index": depth_index,
                "outer_fold": fold, "error": repr(error), "traceback": traceback.format_exc(),
                "search_trace": evaluations, "elapsed_seconds": time.monotonic() - started}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if args.candidate == "spline":
        regions = parse_regions(args.regions or "all")
        depths = parse_indices(args.depths, list(range(21)), 21)
        tasks = [(region, fold, tuple(depths)) for region in regions for fold in range(5)]
        worker = spline_worker
    else:
        regions = parse_indices(args.regions, PILOT_REGIONS, 68)
        depths = parse_indices(args.depths, PILOT_DEPTHS, 21)
        tasks = [(region, depth, fold) for region in regions for depth in depths for fold in range(5)]
        worker = gompertz_worker
    config = {name: str(getattr(args, name)) for name in (
        "subject_order", "cohort_audit", "cohort_manifest", "regional_profile", "regional_thickness",
        "profile_metadata", "thickness_metadata", "folds")}
    predictions, diagnostics, transforms, traces, failures = [], [], [], [], []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.n_jobs, initializer=initialize_worker,
                             initargs=(config,)) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["status"] == "failed":
                failures.append(result)
            else:
                predictions.extend(result["predictions"])
                diagnostics.extend(result["diagnostics"])
                transforms.append({key: result[key] for key in result
                                   if key in ("region_index", "depth_index", "outer_fold", "transform")})
            if "search_trace" in result:
                traces.append({"region_index": result["region_index"],
                               "depth_index": result["depth_index"],
                               "outer_fold": result["outer_fold"],
                               "evaluations": result["search_trace"]})
            if done % 10 == 0 or done == len(tasks):
                print(json.dumps({"candidate": args.candidate, "completed_tasks": done,
                                  "total_tasks": len(tasks), "failures": len(failures)}), flush=True)
    pd.DataFrame(predictions).to_parquet(shard_dir / "predictions.parquet", index=False)
    pd.DataFrame(diagnostics).to_csv(shard_dir / "model_diagnostics.csv", index=False)
    write_json(shard_dir / "fold_transforms.json", transforms)
    write_json(shard_dir / "search_traces.json", traces)
    write_json(shard_dir / "failures.json", failures)
    report = {"status": "failed" if failures else "complete", "candidate": args.candidate,
              "regions": regions, "depth_indices": depths, "n_tasks": len(tasks),
              "n_predictions": len(predictions), "n_failures": len(failures),
              "elapsed_seconds": time.monotonic() - started,
              "python_version": platform.python_version(),
              "pcntoolkit_version": pcntoolkit.__version__, "n_jobs": args.n_jobs,
              "warp": "WarpSinhArcsinh fitted independently within every training fold/outcome",
              "distribution": "homoskedastic warped BLR predictive density",
              "gompertz_formula": "G(age)=exp(-exp(-k*(age-t0))); unconstrained BLR coefficient permits either trend direction"
              if args.candidate == "gompertz" else None,
              "gompertz_constraints": {"rate_per_year": list(GOMPERTZ_RATE_BOUNDS),
                                        "inflection_age": "training-fold observed age range"}
              if args.candidate == "gompertz" else None}
    write_json(shard_dir / "run_report.json", report)
    if failures:
        raise RuntimeError(f"{len(failures)} {args.candidate} tasks failed")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
