#!/usr/bin/env python3
"""Run one resumable ico4 node x outer-fold Warped-M2 task.

Description
-----------
Run one resumable ico4 node x outer-fold Warped-M2 task.

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
- ``--stage`` command-line input
- ``--depth-indices`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure6/scripts/run_ico4_warped_m2_task.py --help
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
from numpy.polynomial.hermite import hermgauss
from pcntoolkit import BLR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--subject-order", type=Path, required=True)
    p.add_argument("--ico4-md", type=Path, required=True)
    p.add_argument("--stage", choices=("smoke", "anchor", "remaining"), required=True)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--node-list", type=Path)
    p.add_argument("--depth-indices", required=True)
    p.add_argument("--min-train", type=int, default=10)
    return p.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def predictive_moments(model: BLR, mu: np.ndarray, s2: np.ndarray, n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    points, weights = hermgauss(n)
    latent = mu[:, None] + np.sqrt(2.0 * s2[:, None]) * points[None, :]
    response = model.warp.invf(latent, model.gamma)
    w = weights / np.sqrt(np.pi)
    mean = response @ w
    var = ((response - mean[:, None]) ** 2) @ w
    return mean, var


def main() -> None:
    a = parse_args()
    task_id = a.task_id if a.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    if task_id < 0:
        raise RuntimeError("Provide --task-id or SLURM_ARRAY_TASK_ID")
    depths = [int(x) for x in a.depth_indices.split(",") if x.strip()]
    if not depths or any(x < 0 or x > 20 for x in depths) or len(set(depths)) != len(depths):
        raise RuntimeError("Invalid depth index list")
    if a.node_list:
        table = pd.read_csv(a.node_list)
        nodes = table.node_index.astype(int).tolist()
    else:
        nodes = list(range(4687))
    n_tasks = len(nodes) * 5
    if task_id >= n_tasks:
        raise RuntimeError(f"task_id {task_id} outside 0..{n_tasks - 1}")
    node_index, fold = nodes[task_id // 5], task_id % 5

    root = a.result_root
    meta = pd.read_csv(root / "ico4_node_metadata.csv").iloc[node_index]
    hemi = str(meta.hemisphere)
    chunk = node_index // 100
    base = root / a.stage
    prediction_dir = base / "predictions" / f"fold={fold}" / f"hemisphere={hemi}" / f"vertex_chunk={chunk:03d}"
    diagnostic_dir = base / "diagnostics" / f"fold={fold}" / f"hemisphere={hemi}" / f"vertex_chunk={chunk:03d}"
    status_dir = base / "status" / f"fold={fold}" / f"hemisphere={hemi}" / f"vertex_chunk={chunk:03d}"
    for d in (prediction_dir, diagnostic_dir, status_dir):
        d.mkdir(parents=True, exist_ok=True)
    stem = f"node_{node_index:04d}"
    pred_path, diag_path, status_path = prediction_dir / f"{stem}.parquet", diagnostic_dir / f"{stem}.parquet", status_dir / f"{stem}.json"
    if status_path.exists() and pred_path.exists() and diag_path.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("status") == "complete" and old.get("depth_indices") == depths:
            print(json.dumps({"status": "already_complete", "task_id": task_id, "node_index": node_index, "fold": fold}))
            return

    started = time.monotonic()
    failures, predictions, diagnostics = [], [], []
    try:
        subject_order = pd.read_csv(a.subject_order, dtype={"subject_id": str})
        audit = pd.read_csv(root / "frozen_cohort_manifest.csv", dtype={"subject_id": str})
        folds = json.loads((root / "fold_assignments.json").read_text(encoding="utf-8"))
        included = audit.loc[audit.included].merge(subject_order[["subject_id", "row_index"]], on="subject_id", validate="one_to_one")
        fold_info = next(x for x in folds["folds"] if int(x["outer_fold"]) == fold)
        train = included[included.subject_id.isin(fold_info["train_subject_ids"])].copy().reset_index(drop=True)
        test = included[included.subject_id.isin(fold_info["test_subject_ids"])].copy().reset_index(drop=True)
        if len(train) + len(test) != 186 or set(train.subject_id) & set(test.subject_id):
            raise RuntimeError("Invalid 186-person fold partition")
        md = np.load(a.ico4_md, mmap_mode="r")
        thickness = np.load(root / "inputs" / "X_ico4_local_freesurfer_thickness.npy", mmap_mode="r")
        t_train_all = np.asarray(thickness[train.row_index.to_numpy(int), node_index], float)
        t_test_all = np.asarray(thickness[test.row_index.to_numpy(int), node_index], float)
        be_maps = {"dummy_batch_effect": {"0": 0}}
        gh_n = 20
        for depth_index in depths:
            fit_start = time.monotonic()
            try:
                feature_index = node_index * 21 + depth_index
                y_train_all = np.asarray(md[train.row_index.to_numpy(int), feature_index], float)
                y_test_all = np.asarray(md[test.row_index.to_numpy(int), feature_index], float)
                train_valid = np.isfinite(y_train_all) & np.isfinite(t_train_all)
                test_valid = np.isfinite(y_test_all) & np.isfinite(t_test_all)
                tr, te = train.loc[train_valid].reset_index(drop=True), test.loc[test_valid].reset_index(drop=True)
                y_train, y_test = y_train_all[train_valid], y_test_all[test_valid]
                t_train, t_test = t_train_all[train_valid], t_test_all[test_valid]
                if len(tr) < a.min_train or len(te) == 0:
                    raise RuntimeError(f"Insufficient valid data: n_train={len(tr)}, n_test={len(te)}")
                age_mean, age_scale = float(tr.age.mean()), float(tr.age.std(ddof=0))
                thickness_mean, thickness_scale = float(t_train.mean()), float(t_train.std(ddof=0))
                y_mean, y_scale = float(y_train.mean()), float(y_train.std(ddof=0))
                if min(age_scale, thickness_scale, y_scale) <= 0 or not np.isfinite([age_scale, thickness_scale, y_scale]).all():
                    raise RuntimeError("Nonfinite or nonpositive training-fold scale")
                x_train = np.column_stack(((tr.age.to_numpy(float) - age_mean) / age_scale,
                                           (tr.sex_code.to_numpy(int) == 2).astype(float),
                                           (t_train - thickness_mean) / thickness_scale))
                x_test = np.column_stack(((te.age.to_numpy(float) - age_mean) / age_scale,
                                          (te.sex_code.to_numpy(int) == 2).astype(float),
                                          (t_test - thickness_mean) / thickness_scale))
                response_train, response_test = (y_train - y_mean) / y_scale, (y_test - y_mean) / y_scale
                xtr = xr.DataArray(x_train, dims=("observations", "covariates"))
                xte = xr.DataArray(x_test, dims=("observations", "covariates"))
                betr = xr.DataArray(np.zeros((len(tr), 1), dtype=int), dims=("observations", "batch_effect_dims"))
                bete = xr.DataArray(np.zeros((len(te), 1), dtype=int), dims=("observations", "batch_effect_dims"))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = BLR(name=f"ico4_wM2_n{node_index}_d{depth_index}_f{fold}", heteroskedastic=False,
                                ard=False, optimizer="l-bfgs-b", n_iter=100, tol=1e-3,
                                warp_name="WarpSinhArcsinh", warp_reparam=True)
                    model.fit(xtr, betr, be_maps, xr.DataArray(response_train, dims=("observations",)))
                if type(model.warp).__name__ != "WarpSinhArcsinh" or model.heteroskedastic:
                    raise RuntimeError("Silent model substitution or Gaussian fallback detected")
                if not (np.isfinite(model.hyp).all() and np.isfinite(model.gamma).all() and np.isfinite(model.nlZ)):
                    raise RuntimeError("Nonfinite final hyperparameters")
                response_test_da = xr.DataArray(response_test, dims=("observations",))
                z = model.forward(xte, bete, response_test_da).values
                log_likelihood = model.elemwise_logp(xte, bete, response_test_da).values - np.log(y_scale)
                latent_mean, latent_var = model.ys_s2(x_test, bete.values)
                median_scaled, intervals_scaled = model.warp.warp_predictions(
                    latent_mean, latent_var, model.gamma, percentiles=[0.05, 0.10, 0.25, 0.75, 0.90, 0.95]
                )
                pred_mean_scaled, pred_var_scaled = predictive_moments(model, latent_mean, latent_var, gh_n)
                jacobian = model.warp.df(response_test, model.gamma)
                values = [z, log_likelihood, latent_mean, latent_var, median_scaled, intervals_scaled,
                          pred_mean_scaled, pred_var_scaled, jacobian]
                if not all(np.isfinite(x).all() for x in values) or np.any(latent_var <= 0) or np.any(pred_var_scaled <= 0) or np.any(jacobian <= 0):
                    raise RuntimeError("Nonfinite prediction, invalid variance, or invalid Jacobian")
                intervals_md = intervals_scaled * y_scale + y_mean
                warning_text = " | ".join(dict.fromkeys(str(w.message) for w in caught))
                for i, s in te.iterrows():
                    predictions.append({
                        "subject_id": s.subject_id, "age": float(s.age), "age_bin": s.age_bin,
                        "sex_code": int(s.sex_code), "outer_fold": fold, "node_index": node_index,
                        "hemisphere": hemi, "ico_node_id": int(meta.ico_node_id),
                        "fsaverage_vertex_id": int(meta.fsaverage_vertex_id), "dk_label_id": int(meta.dk_label_id),
                        "dk_label_name": str(meta.dk_label_name), "depth_index": depth_index,
                        "depth_fraction": depth_index / 20.0, "local_thickness": float(t_test[i]),
                        "observed_md": float(y_test[i]), "pred_mean": float(pred_mean_scaled[i] * y_scale + y_mean),
                        "pred_median": float(median_scaled[i] * y_scale + y_mean),
                        "predictive_variance": float(pred_var_scaled[i] * y_scale ** 2),
                        "z": float(z[i]), "log_likelihood": float(log_likelihood[i]),
                        "lower_90": float(intervals_md[i, 0]), "lower_80": float(intervals_md[i, 1]),
                        "lower_50": float(intervals_md[i, 2]), "upper_50": float(intervals_md[i, 3]),
                        "upper_80": float(intervals_md[i, 4]), "upper_90": float(intervals_md[i, 5]),
                    })
                diagnostics.append({
                    "status": "complete_with_warning" if warning_text else "complete", "outer_fold": fold,
                    "node_index": node_index, "hemisphere": hemi, "ico_node_id": int(meta.ico_node_id),
                    "dk_label_name": str(meta.dk_label_name), "depth_index": depth_index,
                    "n_train": len(tr), "n_test_expected": int(test_valid.sum()), "n_predicted": len(te),
                    "negative_log_evidence": float(model.nlZ), "warp_epsilon": float(model.gamma[0]),
                    "warp_log_b": float(model.gamma[1]), "warp_b": float(np.exp(model.gamma[1])),
                    "hyperparameters": json.dumps(np.asarray(model.hyp).tolist()),
                    "age_mean": age_mean, "age_scale": age_scale, "thickness_mean": thickness_mean,
                    "thickness_scale": thickness_scale, "response_mean": y_mean, "response_scale": y_scale,
                    "warning": warning_text, "elapsed_seconds": time.monotonic() - fit_start,
                })
            except Exception as e:
                failures.append({"node_index": node_index, "outer_fold": fold, "depth_index": depth_index,
                                 "error": repr(e), "traceback": traceback.format_exc()})
                diagnostics.append({"status": "failed", "outer_fold": fold, "node_index": node_index,
                                    "hemisphere": hemi, "ico_node_id": int(meta.ico_node_id),
                                    "dk_label_name": str(meta.dk_label_name), "depth_index": depth_index,
                                    "n_train": 0, "n_test_expected": 0, "n_predicted": 0,
                                    "warning": repr(e), "elapsed_seconds": time.monotonic() - fit_start})
        pd.DataFrame(predictions).to_parquet(pred_path, index=False)
        pd.DataFrame(diagnostics).to_parquet(diag_path, index=False)
        payload = {"status": "failed" if failures else "complete", "stage": a.stage, "task_id": task_id,
                   "node_index": node_index, "outer_fold": fold, "hemisphere": hemi, "depth_indices": depths,
                   "n_fits": len(depths), "n_success": len(depths) - len(failures), "n_failures": len(failures),
                   "n_predictions": len(predictions), "failures": failures,
                   "elapsed_seconds": time.monotonic() - started, "pcntoolkit_version": pcntoolkit.__version__,
                   "model": "Warped M2: linear age + categorical sex_code + local FreeSurfer thickness",
                   "warp": "WarpSinhArcsinh", "heteroskedastic": False, "gaussian_fallback": False,
                   "density_metric": "elementwise predictive log likelihood (higher is better)"}
        atomic_json(status_path, payload)
        print(json.dumps({k: payload[k] for k in ("status", "task_id", "node_index", "outer_fold", "n_success", "n_failures", "elapsed_seconds")}))
        if failures:
            raise SystemExit(2)
    except Exception as e:
        atomic_json(status_path, {"status": "task_failed", "stage": a.stage, "task_id": task_id,
                                  "node_index": node_index, "outer_fold": fold, "depth_indices": depths,
                                  "error": repr(e), "traceback": traceback.format_exc(),
                                  "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
