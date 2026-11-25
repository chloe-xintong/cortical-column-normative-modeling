from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "run_young_reference.py"
SPEC = importlib.util.spec_from_file_location("young_reference", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_analytic_loo_matches_explicit_refits() -> None:
    rng = np.random.default_rng(20261113)
    n = 37
    sex = np.r_[np.zeros(15), np.ones(22)]
    thickness = rng.normal(2.5, 0.2, n)
    y = 0.0007 + 0.00002 * sex - 0.00008 * thickness + rng.normal(0, 2e-5, n)
    fit = MODULE.fit_position(y, sex, thickness, minimum_training_n=10)
    assert fit is not None
    error = MODULE.explicit_loo_check(y, sex, thickness, fit["loo_z"])
    assert error < 1e-9


def test_missing_observation_is_not_scored() -> None:
    rng = np.random.default_rng(9)
    n = 37
    sex = np.r_[np.zeros(15), np.ones(22)]
    thickness = rng.normal(2.5, 0.2, n)
    y = 0.001 - 0.0001 * thickness + rng.normal(0, 2e-5, n)
    y[3] = np.nan
    fit = MODULE.fit_position(y, sex, thickness, minimum_training_n=10)
    assert fit is not None
    assert fit["n_train"] == 36
    assert np.isnan(fit["loo_z"][3])
    assert np.isfinite(np.delete(fit["loo_z"], 3)).all()


def test_signed_tails_partition_absolute_tail() -> None:
    z = np.array([-3.0, -2.0, -1.0, np.nan, 0.0, 2.0, 2.1])
    valid = np.isfinite(z)
    positive = valid & (z > 2.0)
    negative = valid & (z < -2.0)
    absolute = valid & (np.abs(z) > 2.0)
    assert np.array_equal(positive | negative, absolute)
    assert int(positive.sum() + negative.sum()) == int(absolute.sum())
