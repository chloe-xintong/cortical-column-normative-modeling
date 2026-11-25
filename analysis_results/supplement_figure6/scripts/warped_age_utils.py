#!/usr/bin/env python3
"""Utilities shared by the regional warped age-function comparison.

Description
-----------
Utilities shared by the regional warped age-function comparison.

Methods
-------
1. Provide the reusable computations defined in this module.
2. Return results to the calling analysis code.

Inputs
------
- Data supplied through the module's public function arguments.

Outputs
-------
- In-memory values returned to the calling code.

Usage
-----
Import this module and call ``shash_inverse(...)``; it has no command-line entry point.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.polynomial.hermite import hermgauss

if TYPE_CHECKING:
    from pcntoolkit import BLR


PERCENTILES = [0.05, 0.10, 0.25, 0.75, 0.90, 0.95]
GH_NODES, GH_WEIGHTS = hermgauss(32)


def shash_inverse(latent: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Inverse PCNToolkit 1.3.0 ``WarpSinhArcsinh`` transformation."""
    epsilon, log_b = np.asarray(gamma, dtype=float)
    b = np.exp(log_b)
    a = -epsilon * b
    return np.sinh((np.arcsinh(latent) + a) / b)


def shash_forward(response: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Forward PCNToolkit 1.3.0 ``WarpSinhArcsinh`` transformation."""
    epsilon, log_b = np.asarray(gamma, dtype=float)
    b = np.exp(log_b)
    a = -epsilon * b
    return np.sinh(b * np.arcsinh(response) - a)


def shash_predictive_mean(latent_mean: np.ndarray, latent_var: np.ndarray,
                          gamma: np.ndarray) -> np.ndarray:
    """Mean after inverse warping a Gaussian using 32-node Gauss-Hermite."""
    mean = np.asarray(latent_mean, dtype=float).reshape(-1, 1)
    sd = np.sqrt(np.asarray(latent_var, dtype=float)).reshape(-1, 1)
    latent = mean + np.sqrt(2.0) * sd * GH_NODES.reshape(1, -1)
    values = shash_inverse(latent, gamma)
    return values @ GH_WEIGHTS / np.sqrt(np.pi)


@dataclass
class FitResult:
    model: Any
    warnings: list[str]


def fit_warped_blr(x_train: np.ndarray, response_train: np.ndarray, name: str) -> FitResult:
    """Fit the frozen homoskedastic sinh-arcsinh BLR configuration."""
    from pcntoolkit import BLR
    be_train = xr.DataArray(
        np.zeros((len(response_train), 1), dtype=int),
        dims=("observations", "batch_effect_dims"),
    )
    model = BLR(
        name=name, heteroskedastic=False, ard=False, optimizer="l-bfgs-b",
        n_iter=100, tol=1e-3, warp_name="WarpSinhArcsinh", warp_reparam=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(
            xr.DataArray(x_train, dims=("observations", "covariates")),
            be_train,
            {"dummy_batch_effect": {"0": 0}},
            xr.DataArray(response_train, dims=("observations",)),
        )
    messages = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    return FitResult(model=model, warnings=messages)


def predict_warped(model: Any, x_test: np.ndarray, response_test: np.ndarray,
                   y_mean: float, y_scale: float) -> pd.DataFrame:
    """Return density, z, quantiles, median, and mean in original MD units."""
    be_test = xr.DataArray(
        np.zeros((len(response_test), 1), dtype=int),
        dims=("observations", "batch_effect_dims"),
    )
    x_da = xr.DataArray(x_test, dims=("observations", "covariates"))
    y_da = xr.DataArray(response_test, dims=("observations",))
    z = np.asarray(model.forward(x_da, be_test, y_da).values, dtype=float)
    logp = np.asarray(model.elemwise_logp(x_da, be_test, y_da).values, dtype=float) - np.log(y_scale)
    latent_mean, latent_var = model.ys_s2(x_test, be_test.values)
    median_scaled, intervals_scaled = model.warp.warp_predictions(
        latent_mean, latent_var, model.gamma, percentiles=PERCENTILES
    )
    mean_scaled = shash_predictive_mean(latent_mean, latent_var, model.gamma)
    result = pd.DataFrame({
        "pred_mean": mean_scaled * y_scale + y_mean,
        "pred_median": np.asarray(median_scaled) * y_scale + y_mean,
        "z": z,
        "logp": logp,
        "mean_log_loss": -logp,
    })
    names = ["lower_90", "lower_80", "lower_50", "upper_50", "upper_80", "upper_90"]
    intervals_md = np.asarray(intervals_scaled) * y_scale + y_mean
    for j, name in enumerate(names):
        result[name] = intervals_md[:, j]
    finite = result.to_numpy(float)
    if not np.isfinite(finite).all() or not np.all(model.warp.df(response_test, model.gamma) > 0):
        raise RuntimeError("Warped prediction contains a nonfinite value or invalid Jacobian")
    return result


def standardized_gompertz(age_train: np.ndarray, age_test: np.ndarray, rate: float,
                           inflection_age: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build a train-standardized Gompertz CDF basis."""
    raw_train = np.exp(-np.exp(-rate * (age_train - inflection_age)))
    raw_test = np.exp(-np.exp(-rate * (age_test - inflection_age)))
    mean = float(raw_train.mean())
    scale = float(raw_train.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise RuntimeError("Degenerate Gompertz basis under proposed shape parameters")
    return (raw_train - mean) / scale, (raw_test - mean) / scale, mean, scale

