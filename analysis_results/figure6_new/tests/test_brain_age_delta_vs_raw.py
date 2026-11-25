from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "run_brain_age_delta_vs_raw.py"
SPEC = importlib.util.spec_from_file_location("brain_age_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_training_feature_mask_uses_training_rows_only() -> None:
    x = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, 9.0]])
    assert MODULE.training_feature_mask(x, np.array([0, 1])).tolist() == [True, False]
    assert MODULE.training_feature_mask(x, np.array([0, 2])).tolist() == [True, True]


def test_alpha_grid_matches_figure5() -> None:
    assert len(MODULE.ALPHAS) == 21
    assert np.isclose(MODULE.ALPHAS[0], 1e-4)
    assert np.isclose(MODULE.ALPHAS[-1], 1e6)
    assert MODULE.FOLD_SEED == 20261027
