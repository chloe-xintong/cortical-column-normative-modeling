"""Unit tests for the regional warped age-function comparison.

Description
-----------
Unit tests for the regional warped age-function comparison.

Methods
-------
1. Set up the test inputs.
2. Execute the target behavior.
3. Assert the expected results.

Inputs
------
- Data supplied through the module's public function arguments.

Outputs
-------
- Test pass/fail status; no analysis result files are generated.

Usage
-----
pytest analysis_results/figure6/tests/test_warped_age_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_models import spline_matrices  # noqa: E402
from warped_age_utils import (shash_forward, shash_inverse, shash_predictive_mean,
                              standardized_gompertz)  # noqa: E402


def test_sinh_arcsinh_round_trip() -> None:
    values = np.linspace(-4, 4, 101)
    for gamma in (np.array([0.0, 0.0]), np.array([-0.3, 0.2]), np.array([0.4, -0.15])):
        assert np.allclose(shash_inverse(shash_forward(values, gamma), gamma), values)


def test_gauss_hermite_mean_is_exact_for_identity_warp() -> None:
    means = np.array([-2.0, 0.0, 3.0])
    variances = np.array([0.2, 1.0, 4.0])
    observed = shash_predictive_mean(means, variances, np.array([0.0, 0.0]))
    assert np.allclose(observed, means, atol=1e-12)


def test_df3_natural_cubic_spline_has_three_columns_and_frozen_test_transform() -> None:
    train = np.array([41, 45, 50, 55, 60, 66, 72, 78], dtype=float)
    test = np.array([42, 58, 77], dtype=float)
    train_matrix, test_matrix, state, design_info = spline_matrices(train, test)
    assert train_matrix.shape == (8, 3)
    assert test_matrix.shape == (3, 3)
    assert len(state["columns"]) == 3
    assert state["all_knots"][0] == 41
    assert state["all_knots"][-1] == 78
    # Reusing DesignInfo must be deterministic and must not learn knots from test ages.
    import pandas as pd
    import patsy
    repeated = np.asarray(patsy.build_design_matrices(
        [design_info], pd.DataFrame({"age": test}), return_type="dataframe"
    )[0])
    assert np.allclose(repeated, test_matrix)


def test_gompertz_basis_is_train_standardized_and_monotone() -> None:
    train = np.linspace(41, 78, 50)
    test = np.array([45.0, 60.0, 75.0])
    train_basis, test_basis, mean, scale = standardized_gompertz(train, test, 0.08, 60.0)
    assert abs(train_basis.mean()) < 1e-12
    assert abs(train_basis.std(ddof=0) - 1) < 1e-12
    assert np.all(np.diff(test_basis) > 0)
    assert np.isfinite([mean, scale]).all()
