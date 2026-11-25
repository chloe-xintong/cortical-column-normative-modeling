from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "compare_loo_models.py"
SPEC = importlib.util.spec_from_file_location("compare_loo_models", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_ols_prediction_and_z() -> None:
    sex = np.array([0, 0, 1, 1, 0, 1], dtype=float)
    thickness = np.array([2.1, 2.2, 2.3, 2.4, 2.5, 2.6])
    noise = np.array([-1, 1, -1, 1, 0.5, -0.5]) * 1e-5
    y = 0.001 + 2e-5 * sex - 8e-5 * thickness + noise
    prediction, residual_sd, z = MODULE.fit_ols_predict(
        y[:-1], sex[:-1], thickness[:-1], y[-1], sex[-1], thickness[-1], True
    )
    assert np.isfinite([prediction, residual_sd, z]).all()
    assert residual_sd > 0
    assert np.isclose(z, (y[-1] - prediction) / residual_sd)


def test_depth_bands_partition_all_depths() -> None:
    depths = [depth for values in MODULE.DEPTH_BANDS.values() for depth in values]
    assert depths == list(range(21))
