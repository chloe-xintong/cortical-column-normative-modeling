#!/usr/bin/env python3
"""Unit and artifact integrity checks for the Figure 6 regional pilot.

Description
-----------
Unit and artifact integrity checks for the Figure 6 regional pilot.

Methods
-------
1. Set up the test inputs.
2. Execute the target behavior.
3. Assert the expected results.

Inputs
------
- No required command-line inputs; optional inputs are defined by the script.

Outputs
-------
- Test pass/fail status; no analysis result files are generated.

Usage
-----
pytest analysis_results/figure6/tests/test_regional_pilot.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pilot_utils import age_bin, depth_band  # noqa: E402


class DefinitionTests(unittest.TestCase):
    def test_fixed_age_bins(self) -> None:
        observed = list(age_bin(np.array([40, 49, 50, 59, 60, 69, 70, 79])).astype(str))
        self.assertEqual(observed, ["40–49", "40–49", "50–59", "50–59",
                                    "60–69", "60–69", "70–79", "70–79"])

    def test_depth_bands_and_endpoints(self) -> None:
        self.assertEqual(depth_band(0.0), "endpoint_pial_0.00")
        self.assertEqual(depth_band(1.0), "endpoint_white_1.00")
        self.assertEqual(depth_band(0.2), "outer_0.10–0.30")
        self.assertEqual(depth_band(0.5), "middle_0.40–0.60")
        self.assertEqual(depth_band(0.8), "inner_0.70–0.90")
        self.assertEqual(depth_band(0.35), "other_nonprimary")


class ArtifactIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (ROOT / "oof_predictions_long.parquet").exists():
            raise unittest.SkipTest("Formal pilot artifacts are absent")
        cls.data = pd.read_parquet(ROOT / "oof_predictions_long.parquet")

    def test_exact_oof_partition(self) -> None:
        self.assertEqual(len(self.data), 188 * 68 * 21 * 4)
        keys = ["subject_id", "model", "region_index", "depth_index"]
        self.assertFalse(self.data.duplicated(keys).any())
        self.assertTrue(self.data.groupby(["subject_id", "model"]).size().eq(68 * 21).all())

    def test_predictive_variance_decomposition(self) -> None:
        np.testing.assert_allclose(
            self.data.pred_var.to_numpy(),
            self.data.noise_var.to_numpy() + self.data.parameter_var.to_numpy(),
            rtol=1e-10, atol=1e-18,
        )
        self.assertTrue((self.data.pred_var > 0).all())
        reconstructed = (self.data.observed_md - self.data.pred_mean) / np.sqrt(self.data.pred_var)
        np.testing.assert_allclose(reconstructed, self.data.z, rtol=1e-10, atol=1e-10)

    def test_no_model_failures_and_no_ico4_submission(self) -> None:
        metadata = json.loads((ROOT / "run_metadata.json").read_text())
        self.assertEqual(metadata["failed_outcomes"], [])
        self.assertEqual(metadata["missing_oof_predictions"], 0)
        self.assertFalse(metadata["full_ico4_jobs_submitted"])


if __name__ == "__main__":
    unittest.main()
