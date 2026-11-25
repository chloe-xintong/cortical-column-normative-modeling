#!/usr/bin/env python3
"""Shared definitions for the Figure 6 regional normative-model pilot.

Description
-----------
Shared definitions for the Figure 6 regional normative-model pilot.

Methods
-------
1. Provide the reusable computations defined in this module.
2. Return results to the calling analysis code.

Inputs
------
- Data supplied through the module's public function arguments.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
Import this module and call ``write_json(...)``; it has no command-line entry point.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DEPTHS = np.round(np.linspace(0.0, 1.0, 21), 2)
AGE_BIN_EDGES = [40, 50, 60, 70, 80]
AGE_BIN_LABELS = ["40–49", "50–59", "60–69", "70–79"]
FOLD_SEED = 20261106
BOOTSTRAP_SEED = 20261107
N_BOOTSTRAP = 10_000

MODEL_SPECS = {
    "M0": {"age_form": "linear", "thickness": False, "formula": "MD ~ age + sex_code"},
    "M1": {
        "age_form": "natural_cubic_spline_df3",
        "thickness": False,
        "formula": "MD ~ natural_cubic_spline(age, df=3) + sex_code",
    },
    "M2": {
        "age_form": "linear",
        "thickness": True,
        "formula": "MD ~ age + sex_code + local_thickness",
    },
    "M3": {
        "age_form": "natural_cubic_spline_df3",
        "thickness": True,
        "formula": "MD ~ natural_cubic_spline(age, df=3) + sex_code + local_thickness",
    },
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def age_bin(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(values, AGE_BIN_EDGES, right=False, labels=AGE_BIN_LABELS)


def depth_band(depth: float) -> str:
    if np.isclose(depth, 0.0):
        return "endpoint_pial_0.00"
    if np.isclose(depth, 1.0):
        return "endpoint_white_1.00"
    if 0.10 - 1e-9 <= depth <= 0.30 + 1e-9:
        return "outer_0.10–0.30"
    if 0.40 - 1e-9 <= depth <= 0.60 + 1e-9:
        return "middle_0.40–0.60"
    if 0.70 - 1e-9 <= depth <= 0.90 + 1e-9:
        return "inner_0.70–0.90"
    return "other_nonprimary"


def metric_point(frame: pd.DataFrame) -> dict[str, float | int]:
    z = frame["z"].to_numpy(float)
    error = frame["observed_md"].to_numpy(float) - frame["pred_mean"].to_numpy(float)
    probs = np.linspace(0.01, 0.99, 99)
    expected = stats.norm.ppf(probs)
    observed = np.quantile(z, probs)
    thresholds = {"coverage_50": stats.norm.ppf(0.75), "coverage_80": stats.norm.ppf(0.90),
                  "coverage_90": stats.norm.ppf(0.95)}
    result: dict[str, float | int] = {
        "n_observations": int(len(frame)),
        "n_subjects": int(frame["subject_id"].nunique()),
        "mean_z": float(np.mean(z)),
        "sd_z": float(np.std(z, ddof=1)),
        "median_z": float(np.median(z)),
        "skewness_z": float(stats.skew(z, bias=False)),
        "excess_kurtosis_z": float(stats.kurtosis(z, fisher=True, bias=False)),
        "qq_rmse": float(np.sqrt(np.mean(np.square(observed - expected)))),
        "mean_nlpd": float(frame["nlpd"].mean()),
        "median_nlpd": float(frame["nlpd"].median()),
        "mse_md": float(np.mean(np.square(error))),
        "mae_md": float(np.mean(np.abs(error))),
    }
    expected_coverage = {"coverage_50": 0.50, "coverage_80": 0.80, "coverage_90": 0.90}
    for name, threshold in thresholds.items():
        coverage = float(np.mean(np.abs(z) <= threshold))
        result[name] = coverage
        result[f"{name}_absolute_error"] = abs(coverage - expected_coverage[name])
    return result


def subject_sufficient_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[["subject_id", "z", "nlpd", "observed_md", "pred_mean"]].copy()
    work["z2"] = work["z"] ** 2
    work["sqerr"] = (work["observed_md"] - work["pred_mean"]) ** 2
    work["abserr"] = (work["observed_md"] - work["pred_mean"]).abs()
    for label, p in (("cov50", 0.75), ("cov80", 0.90), ("cov90", 0.95)):
        work[label] = (work["z"].abs() <= stats.norm.ppf(p)).astype(float)
    return work.groupby("subject_id", sort=True).agg(
        n=("z", "size"), sum_z=("z", "sum"), sum_z2=("z2", "sum"),
        sum_nlpd=("nlpd", "sum"), sum_sqerr=("sqerr", "sum"), sum_abserr=("abserr", "sum"),
        sum_cov50=("cov50", "sum"), sum_cov80=("cov80", "sum"), sum_cov90=("cov90", "sum"),
    ).reset_index()


def bootstrap_additive_metrics(frame: pd.DataFrame, sample_counts: np.ndarray) -> dict[str, np.ndarray]:
    sufficient = subject_sufficient_statistics(frame).set_index("subject_id")
    subject_order = sorted(frame["subject_id"].unique())
    sufficient = sufficient.loc[subject_order]
    matrix = sufficient.drop(columns=[], errors="ignore").to_numpy(float)
    total = sample_counts @ matrix
    n, sum_z, sum_z2, sum_nlpd, sum_sqerr, sum_abserr, cov50, cov80, cov90 = total.T
    variance = (sum_z2 - np.square(sum_z) / n) / np.maximum(n - 1, 1)
    return {
        "mean_z": sum_z / n,
        "sd_z": np.sqrt(np.maximum(variance, 0)),
        "mean_nlpd": sum_nlpd / n,
        "mse_md": sum_sqerr / n,
        "mae_md": sum_abserr / n,
        "coverage_50": cov50 / n,
        "coverage_80": cov80 / n,
        "coverage_90": cov90 / n,
    }


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)
