#!/usr/bin/env python3
"""Run leakage-safe 5-fold OOF sinh-arcsinh warped M2 with PCNToolkit 1.3.0.

Description
-----------
Run leakage-safe 5-fold OOF sinh-arcsinh warped M2 with PCNToolkit 1.3.0.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--output-dir`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/run_warped_m2.py --help
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pcntoolkit
import xarray as xr
from pcntoolkit import BLR

ORIGINAL_SCRIPTS = Path(__file__).resolve().parents[2] / "figure6" / "scripts"
sys.path.insert(0, str(ORIGINAL_SCRIPTS))
from run_models import build_x, parse_regions  # noqa: E402

_CONFIG: dict[str, str] = {}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("subject-order", "cohort-audit", "cohort-manifest", "regional-profile",
                 "regional-thickness", "profile-metadata", "thickness-metadata", "folds"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regions", default="all")
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def initialize_worker(config: dict[str, str]) -> None:
    global _CONFIG
    _CONFIG = config


def worker(task: tuple[int, int]) -> dict[str, object]:
    region_index, fold = task
    started = time.monotonic()
    try:
        subject_order = pd.read_csv(_CONFIG["subject_order"], dtype={"subject_id": str})
        audit = pd.read_csv(_CONFIG["cohort_audit"], dtype={"subject_id": str})
        cohort = pd.read_csv(_CONFIG["cohort_manifest"], dtype={"subject_id": str})
        profile_meta = pd.read_csv(_CONFIG["profile_metadata"])
        thickness_meta = pd.read_csv(_CONFIG["thickness_metadata"])
        profile = np.load(_CONFIG["regional_profile"], mmap_mode="r")
        thickness = np.load(_CONFIG["regional_thickness"], mmap_mode="r")
        folds = json.loads(Path(_CONFIG["folds"]).read_text(encoding="utf-8"))
        included = audit.loc[audit.included].merge(
            subject_order[["subject_id", "row_index"]], on="subject_id", validate="one_to_one"
        ).merge(cohort[["subject_id", "SUBJECT_SEX"]], on="subject_id", validate="one_to_one")
        included["sex_code"] = included.SUBJECT_SEX.astype(int)
        fold_info = next(x for x in folds["folds"] if int(x["outer_fold"]) == fold)
        train = included[included.subject_id.isin(fold_info["train_subject_ids"])].copy().reset_index(drop=True)
        test = included[included.subject_id.isin(fold_info["test_subject_ids"])].copy().reset_index(drop=True)
        if len(train) + len(test) != len(included) or set(train.subject_id) & set(test.subject_id):
            raise RuntimeError(f"Invalid {len(included)}-person fold partition")

        x_train, x_test, transform, _ = build_x("M2", train, test, region_index, thickness)
        start = region_index * 21
        y_train = np.asarray(profile[train.row_index.to_numpy(int), start:start + 21], float)
        y_test = np.asarray(profile[test.row_index.to_numpy(int), start:start + 21], float)
        y_mean = y_train.mean(axis=0)
        y_scale = y_train.std(axis=0, ddof=0)
        if not np.isfinite(y_train).all() or not np.isfinite(y_test).all() or np.any(y_scale <= 0):
            raise RuntimeError("Invalid response values or training-fold response scale")

        x_train_da = xr.DataArray(x_train, dims=("observations", "covariates"))
        x_test_da = xr.DataArray(x_test, dims=("observations", "covariates"))
        be_train = xr.DataArray(np.zeros((len(train), 1), dtype=int), dims=("observations", "batch_effect_dims"))
        be_test = xr.DataArray(np.zeros((len(test), 1), dtype=int), dims=("observations", "batch_effect_dims"))
        be_maps = {"dummy_batch_effect": {"0": 0}}
        region = thickness_meta.iloc[region_index]
        depths = profile_meta.iloc[start:start + 21].reset_index(drop=True)
        rows: list[dict[str, object]] = []
        diagnostics: list[dict[str, object]] = []
        for depth_index in range(21):
            response_train = (y_train[:, depth_index] - y_mean[depth_index]) / y_scale[depth_index]
            response_test = (y_test[:, depth_index] - y_mean[depth_index]) / y_scale[depth_index]
            model = BLR(
                name=f"warped_M2_r{region_index:02d}_d{depth_index:02d}_f{fold}",
                heteroskedastic=False, ard=False, optimizer="l-bfgs-b", n_iter=100,
                tol=1e-3, warp_name="WarpSinhArcsinh", warp_reparam=True,
            )
            model.fit(x_train_da, be_train, be_maps, xr.DataArray(response_train, dims=("observations",)))
            response_test_da = xr.DataArray(response_test, dims=("observations",))
            z = model.forward(x_test_da, be_test, response_test_da).values
            logp_scaled = model.elemwise_logp(x_test_da, be_test, response_test_da).values
            logp_md = logp_scaled - np.log(y_scale[depth_index])
            latent_mean, latent_var = model.ys_s2(x_test, be_test.values)
            median_scaled, intervals_scaled = model.warp.warp_predictions(
                latent_mean, latent_var, model.gamma, percentiles=[0.05, 0.10, 0.25, 0.75, 0.90, 0.95]
            )
            median_md = median_scaled * y_scale[depth_index] + y_mean[depth_index]
            intervals_md = intervals_scaled * y_scale[depth_index] + y_mean[depth_index]
            if not (np.isfinite(z).all() and np.isfinite(logp_md).all() and np.isfinite(median_md).all()
                    and np.isfinite(intervals_md).all() and np.all(model.warp.df(response_test, model.gamma) > 0)):
                raise RuntimeError("Warped prediction contains non-finite values or invalid Jacobian")
            for i, subject in test.iterrows():
                rows.append({
                    "subject_id": subject.subject_id, "age": float(subject.age), "age_bin": subject.age_bin,
                    "sex_code": int(subject.sex_code), "outer_fold": fold, "model": "warped_M2",
                    "region_index": region_index, "hemisphere": region.hemisphere,
                    "dk_label_id": int(region.dk_label_id), "dk_label_name": region.dk_label_name,
                    "depth_index": depth_index, "depth": float(depths.iloc[depth_index].depth_fraction),
                    "local_thickness": float(thickness[int(subject.row_index), region_index]),
                    "observed_md": float(y_test[i, depth_index]), "pred_median": float(median_md[i]),
                    "z": float(z[i]), "logp": float(logp_md[i]), "mean_log_loss": float(-logp_md[i]),
                    "lower_90": float(intervals_md[i, 0]), "lower_80": float(intervals_md[i, 1]),
                    "lower_50": float(intervals_md[i, 2]), "upper_50": float(intervals_md[i, 3]),
                    "upper_80": float(intervals_md[i, 4]), "upper_90": float(intervals_md[i, 5]),
                })
            diagnostics.append({
                "model": "warped_M2", "outer_fold": fold, "region_index": region_index,
                "depth_index": depth_index, "status": "complete", "negative_log_evidence": float(model.nlZ),
                "warp_epsilon": float(model.gamma[0]), "warp_log_b": float(model.gamma[1]),
                "warp_b": float(np.exp(model.gamma[1])), "hyperparameters": json.dumps(model.hyp.tolist()),
            })
        return {"status": "complete", "region_index": region_index, "outer_fold": fold,
                "predictions": rows, "diagnostics": diagnostics, "transform": transform,
                "elapsed_seconds": time.monotonic() - started}
    except Exception as error:
        return {"status": "failed", "region_index": region_index, "outer_fold": fold,
                "error": repr(error), "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    regions = parse_regions(args.regions)
    config = {name.replace("_", "-"): "" for name in []}
    config = {name: str(getattr(args, name)) for name in (
        "subject_order", "cohort_audit", "cohort_manifest", "regional_profile", "regional_thickness",
        "profile_metadata", "thickness_metadata", "folds")}
    tasks = [(region, fold) for region in regions for fold in range(5)]
    predictions: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    transforms: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.n_jobs, initializer=initialize_worker, initargs=(config,)) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["status"] == "failed":
                failures.append(result)
            else:
                predictions.extend(result["predictions"])
                diagnostics.extend(result["diagnostics"])
                transforms.append({"region_index": result["region_index"], "outer_fold": result["outer_fold"],
                                   "transform": result["transform"]})
            if done % 10 == 0 or done == len(tasks):
                print(json.dumps({"completed_tasks": done, "total_tasks": len(tasks),
                                  "failures": len(failures)}), flush=True)
    tag = "all" if len(regions) == 68 else f"{min(regions):02d}_{max(regions):02d}"
    pd.DataFrame(predictions).to_parquet(shard_dir / f"predictions_{tag}.parquet", index=False)
    pd.DataFrame(diagnostics).to_csv(shard_dir / f"model_diagnostics_{tag}.csv", index=False)
    write_json(shard_dir / f"fold_transforms_{tag}.json", transforms)
    write_json(shard_dir / f"failures_{tag}.json", failures)
    report = {"status": "failed" if failures else "complete", "distribution": "sinh-arcsinh warped",
              "mean_formula": "M2: MD ~ linear age + categorical sex_code + local thickness",
              "warp_name": "WarpSinhArcsinh", "warp_reparam": True, "heteroskedastic": False,
              "n_tasks": len(tasks), "n_predictions": len(predictions), "n_failures": len(failures),
              "elapsed_seconds": time.monotonic() - started, "python_version": platform.python_version(),
              "pcntoolkit_version": pcntoolkit.__version__,
              "density_metric": "MLL = mean negative elementwise predictive log probability in original MD units; lower is better"}
    write_json(shard_dir / f"run_report_{tag}.json", report)
    if failures:
        raise RuntimeError(f"{len(failures)} warped tasks failed")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
