#!/usr/bin/env python3
"""Fit all 21 full-cohort Warped-M2 models for one frozen ico4 node.

Description
-----------
Fit all 21 full-cohort Warped-M2 models for one frozen ico4 node.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--result-root`` command-line input
- ``--subject-order`` command-line input
- ``--ico4-md`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/run_ico4_full_cohort_node.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pcntoolkit
import xarray as xr
from pcntoolkit import BLR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--subject-order", type=Path, required=True)
    p.add_argument("--ico4-md", type=Path, required=True)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--age-min", type=int, default=45)
    p.add_argument("--age-max", type=int, default=76)
    p.add_argument("--min-train", type=int, default=10)
    return p.parse_args()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot JSON-encode {type(value)}")


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False, default=json_default) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    a = parse_args()
    node = a.task_id if a.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    if node < 0 or node >= 4687:
        raise RuntimeError(f"Node task ID outside 0..4686: {node}")
    if a.age_min >= a.age_max:
        raise RuntimeError("Invalid supported age range")

    started = time.monotonic()
    root = a.result_root
    meta = pd.read_csv(root / "ico4_node_metadata.csv").iloc[node]
    hemi = str(meta.hemisphere)
    chunk = node // 100
    rel = Path(f"hemisphere={hemi}") / f"vertex_chunk={chunk:03d}"
    parameter_dir = root / "full_models" / "parameters" / rel
    centile_dir = root / "full_models" / "centiles" / rel
    status_dir = root / "full_models" / "status" / rel
    for path in (parameter_dir, centile_dir, status_dir):
        path.mkdir(parents=True, exist_ok=True)
    stem = f"node_{node:04d}"
    parameter_path = parameter_dir / f"{stem}.parquet"
    centile_path = centile_dir / f"{stem}.parquet"
    status_path = status_dir / f"{stem}.json"
    if parameter_path.is_file() and centile_path.is_file() and status_path.is_file():
        old = json.loads(status_path.read_text())
        if old.get("status") == "complete" and old.get("n_success") == 21:
            print(json.dumps({"status": "already_complete", "node_index": node}))
            return

    failures, parameters, centiles = [], [], []
    try:
        subject_order = pd.read_csv(a.subject_order, dtype={"subject_id": str})
        audit = pd.read_csv(root / "frozen_cohort_manifest.csv", dtype={"subject_id": str})
        cohort = (audit.loc[audit.included]
                  .merge(subject_order[["subject_id", "row_index"]], on="subject_id", validate="one_to_one")
                  .reset_index(drop=True))
        if len(cohort) != 186:
            raise RuntimeError("Full-cohort fit does not contain 186 retained participants")
        md = np.load(a.ico4_md, mmap_mode="r")
        thickness = np.load(root / "inputs" / "X_ico4_local_freesurfer_thickness.npy", mmap_mode="r")
        local_thickness = np.asarray(thickness[cohort.row_index.to_numpy(int), node], float)
        thickness_reference = float(np.nanmedian(local_thickness))
        if not np.isfinite(thickness_reference):
            raise RuntimeError("Nonfinite node-local thickness reference")
        age_grid = np.arange(a.age_min, a.age_max + 1, dtype=float)
        be_maps = {"dummy_batch_effect": {"0": 0}}

        for depth in range(21):
            fit_started = time.monotonic()
            try:
                response = np.asarray(md[cohort.row_index.to_numpy(int), node * 21 + depth], float)
                valid = np.isfinite(response) & np.isfinite(local_thickness)
                train = cohort.loc[valid].reset_index(drop=True)
                y = response[valid]
                t = local_thickness[valid]
                if len(train) < a.min_train:
                    raise RuntimeError(f"Insufficient full-cohort data: n_train={len(train)}")
                age_mean, age_scale = float(train.age.mean()), float(train.age.std(ddof=0))
                thickness_mean, thickness_scale = float(t.mean()), float(t.std(ddof=0))
                response_mean, response_scale = float(y.mean()), float(y.std(ddof=0))
                scales = np.array([age_scale, thickness_scale, response_scale])
                if not np.isfinite(scales).all() or np.any(scales <= 0):
                    raise RuntimeError("Nonfinite or nonpositive full-cohort scale")
                x = np.column_stack([
                    (train.age.to_numpy(float) - age_mean) / age_scale,
                    (train.sex_code.to_numpy(int) == 2).astype(float),
                    (t - thickness_mean) / thickness_scale,
                ])
                y_scaled = (y - response_mean) / response_scale
                x_da = xr.DataArray(x, dims=("observations", "covariates"))
                be_da = xr.DataArray(np.zeros((len(train), 1), dtype=int),
                                     dims=("observations", "batch_effect_dims"))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = BLR(
                        name=f"ico4_full_wM2_n{node}_d{depth}", heteroskedastic=False,
                        ard=False, optimizer="l-bfgs-b", n_iter=100, tol=1e-3,
                        warp_name="WarpSinhArcsinh", warp_reparam=True,
                    )
                    model.fit(x_da, be_da, be_maps,
                              xr.DataArray(y_scaled, dims=("observations",)))
                if type(model.warp).__name__ != "WarpSinhArcsinh" or model.heteroskedastic:
                    raise RuntimeError("Silent Gaussian fallback or model substitution")
                if not (np.isfinite(model.hyp).all() and np.isfinite(model.gamma).all()
                        and np.isfinite(model.m).all() and np.isfinite(model.A).all()
                        and np.isfinite(model.nlZ)):
                    raise RuntimeError("Nonfinite final full-cohort parameter")
                jacobian = model.warp.df(y_scaled, model.gamma)
                if not np.isfinite(jacobian).all() or np.any(jacobian <= 0):
                    raise RuntimeError("Invalid final warp Jacobian")
                model_dict = model.to_dict()
                restored = BLR.from_dict(model_dict)
                if type(restored.warp).__name__ != "WarpSinhArcsinh":
                    raise RuntimeError("Serialized model does not restore WarpSinhArcsinh")
                warning_text = " | ".join(dict.fromkeys(str(w.message) for w in caught))
                covariance = np.linalg.inv(np.asarray(model.A, float))
                parameters.append({
                    "status": "complete_with_warning" if warning_text else "complete",
                    "node_index": node, "hemisphere": hemi, "ico_node_id": int(meta.ico_node_id),
                    "fsaverage_vertex_id": int(meta.fsaverage_vertex_id),
                    "dk_label_id": int(meta.dk_label_id), "dk_label_name": str(meta.dk_label_name),
                    "depth_index": depth, "depth_fraction": depth / 20.0,
                    "n_train": len(train), "age_mean": age_mean, "age_scale": age_scale,
                    "thickness_mean": thickness_mean, "thickness_scale": thickness_scale,
                    "thickness_reference_median": thickness_reference,
                    "response_mean": response_mean, "response_scale": response_scale,
                    "coefficient_labels": json.dumps(["intercept", "age_z", "sex_code_2", "thickness_z"]),
                    "posterior_weight_mean": json.dumps(np.asarray(model.m).tolist()),
                    "posterior_weight_precision": json.dumps(np.asarray(model.A).tolist()),
                    "posterior_weight_covariance": json.dumps(covariance.tolist()),
                    "hyperparameters": json.dumps(np.asarray(model.hyp).tolist()),
                    "warp_parameters": json.dumps(np.asarray(model.gamma).tolist()),
                    "latent_noise_precision": float(np.asarray(model.beta).ravel()[0]),
                    "latent_residual_sd": float(1 / np.sqrt(np.asarray(model.beta).ravel()[0])),
                    "negative_log_evidence": float(model.nlZ),
                    "model_dict_json": json.dumps(model_dict, allow_nan=False, default=json_default),
                    "warning": warning_text, "elapsed_seconds": time.monotonic() - fit_started,
                    "pcntoolkit_version": pcntoolkit.__version__,
                    "model_formula": "MD ~ linear age + categorical sex_code + local FreeSurfer thickness",
                    "warp": "WarpSinhArcsinh", "heteroskedastic": False,
                })

                prediction_rows = []
                for sex_code in (1, 2):
                    x_grid = np.column_stack([
                        (age_grid - age_mean) / age_scale,
                        np.full(len(age_grid), float(sex_code == 2)),
                        np.full(len(age_grid), (thickness_reference - thickness_mean) / thickness_scale),
                    ])
                    be_grid = np.zeros((len(age_grid), 1), dtype=int)
                    latent_mean, latent_variance = model.ys_s2(x_grid, be_grid)
                    median_scaled, interval_scaled = model.warp.warp_predictions(
                        latent_mean, latent_variance, model.gamma, percentiles=[0.05, 0.95]
                    )
                    p05 = interval_scaled[:, 0] * response_scale + response_mean
                    p50 = median_scaled * response_scale + response_mean
                    p95 = interval_scaled[:, 1] * response_scale + response_mean
                    if not np.isfinite(np.column_stack([p05, p50, p95])).all():
                        raise RuntimeError("Nonfinite original-scale centile prediction")
                    for i, age in enumerate(age_grid):
                        prediction_rows.append({
                            "node_index": node, "hemisphere": hemi,
                            "ico_node_id": int(meta.ico_node_id),
                            "fsaverage_vertex_id": int(meta.fsaverage_vertex_id),
                            "dk_label_id": int(meta.dk_label_id),
                            "dk_label_name": str(meta.dk_label_name),
                            "depth_index": depth, "depth_fraction": depth / 20.0,
                            "age": float(age), "sex_code": sex_code,
                            "thickness_reference": thickness_reference,
                            "thickness_reference_rule": "node-local median among 186 retained participants with finite thickness",
                            "p05_md": float(p05[i]), "p50_md": float(p50[i]), "p95_md": float(p95[i]),
                        })
                centiles.extend(prediction_rows)
            except Exception as exc:
                failures.append({"node_index": node, "depth_index": depth, "error": repr(exc),
                                 "traceback": traceback.format_exc()})
                parameters.append({
                    "status": "failed", "node_index": node, "hemisphere": hemi,
                    "ico_node_id": int(meta.ico_node_id), "fsaverage_vertex_id": int(meta.fsaverage_vertex_id),
                    "dk_label_id": int(meta.dk_label_id), "dk_label_name": str(meta.dk_label_name),
                    "depth_index": depth, "depth_fraction": depth / 20.0, "n_train": 0,
                    "warning": repr(exc), "elapsed_seconds": time.monotonic() - fit_started,
                    "pcntoolkit_version": pcntoolkit.__version__,
                })

        pd.DataFrame(parameters).to_parquet(parameter_path, index=False)
        pd.DataFrame(centiles).to_parquet(centile_path, index=False)
        payload = {
            "status": "failed" if failures else "complete", "node_index": node,
            "hemisphere": hemi, "n_fits": 21, "n_success": 21 - len(failures),
            "n_failures": len(failures), "n_centile_rows": len(centiles),
            "age_grid": [a.age_min, a.age_max, 1], "sex_codes": [1, 2],
            "thickness_reference": thickness_reference,
            "failures": failures, "elapsed_seconds": time.monotonic() - started,
            "pcntoolkit_version": pcntoolkit.__version__, "model": "Warped M2",
            "warp": "WarpSinhArcsinh", "heteroskedastic": False, "gaussian_fallback": False,
        }
        atomic_json(status_path, payload)
        print(json.dumps({k: payload[k] for k in ("status", "node_index", "n_success", "n_failures",
                                                  "n_centile_rows", "elapsed_seconds")}))
        if failures:
            raise SystemExit(2)
    except Exception as exc:
        atomic_json(status_path, {
            "status": "task_failed", "node_index": node, "error": repr(exc),
            "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started,
        })
        raise


if __name__ == "__main__":
    main()
