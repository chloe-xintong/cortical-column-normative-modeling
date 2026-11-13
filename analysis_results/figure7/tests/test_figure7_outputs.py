"""Integrity tests for completed Figure 7 cached outputs.

Description
-----------
Integrity tests for completed Figure 7 cached outputs.

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
pytest analysis_results/figure7/tests/test_figure7_outputs.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_cohort_and_fold() -> None:
    roster = pd.read_csv(ROOT / "inputs" / "final_analysis_roster.csv", dtype={"subject_id": str})
    assert len(roster) == roster.subject_id.nunique() == 111
    assert roster.diagnosis_at_scan.value_counts().to_dict() == {
        "Normal": 38, "Unknown": 26, "Other/Impaired non-demented": 22, "MCI": 19, "AD": 6
    }
    assert set(roster.loc[roster.diagnosis_at_scan.eq("Normal"), "model_role"]) == {"fold1_heldout_cn"}
    assert not roster.model_role.eq("fold1_training_cn").any()


def test_burden_arithmetic_and_model_recovery() -> None:
    burden = pd.read_csv(ROOT / "outputs" / "participant_deviation_burden.csv")
    assert len(burden) == 111
    assert np.array_equal(
        burden.n_total_abs_z_gt_2.to_numpy(),
        burden.n_positive_z_gt_2.to_numpy() + burden.n_negative_z_lt_minus_2.to_numpy(),
    )
    assert np.allclose(burden.total_deviation_burden,
                       burden.n_total_abs_z_gt_2 / burden.n_valid_positions)
    assert np.allclose(burden.total_deviation_burden,
                       burden.positive_deviation_burden + burden.negative_deviation_burden)
    recovery = json.loads((ROOT / "outputs" / "model_recovery_validation.json").read_text())
    assert recovery["model_refit"] is False and recovery["folds_mixed"] is False
    assert recovery["maximum_saved_z_recovery_error"] < 1e-10
    assert recovery["recovery_positions"] == 98_385
    assert recovery["target_unavailable_positions_insufficient_saved_design_rank"] == 42


def test_primary_result_counts_and_notebook_outputs() -> None:
    a = pd.read_csv(ROOT / "outputs" / "figure7a_primary_results.csv")
    b = pd.read_csv(ROOT / "outputs" / "figure7b_primary_and_diagnosis_sensitivity.csv")
    assert set(a.n_participants) == {63}
    assert set(a.predictor) == {"MCI-CN", "AD-CN"}
    assert set(b.n_participants) == {109}
    assert set(b.sensitivity) == {
        "raw_unadjusted", "primary", "primary_diagnosis_adjusted"
    }
    assert (ROOT / "notebooks" / "figure7_fixed_fold1_analysis.ipynb").is_file()
    for stem in ("Figure7A_diagnostic_group_comparison",
                 "Figure7A_adjusted_group_contrasts",
                 "Figure7B_raw_global_cognition_association",
                 "Figure7B_adjusted_global_cognition_association",
                 "Figure7B_diagnosis_adjusted_global_cognition_association"):
        assert (ROOT / "figures" / f"{stem}.png").is_file()
        assert (ROOT / "figures" / f"{stem}.pdf").is_file()


def test_all_normal_five_fold_oof_sensitivity() -> None:
    data = pd.read_csv(ROOT / "outputs" / "all_normal_oof_burden_cognition.csv")
    results = pd.read_csv(ROOT / "outputs" / "all_normal_oof_cognition_results.csv")
    audit = json.loads((ROOT / "outputs" / "all_normal_oof_validation.json").read_text())
    assert len(data) == data.subject_id.nunique() == 186
    assert data.outer_fold.value_counts().sort_index().tolist() == [38, 38, 37, 36, 37]
    assert data.global_cognition.notna().all()
    assert data.age_at_scan.between(41, 78).all()
    assert np.allclose(
        data.total_deviation_burden,
        data.n_total_abs_z_gt_2 / data.n_valid_positions,
    )
    assert set(results.sensitivity) == {
        "primary_all_normal_oof", "fold_fixed_effect_sensitivity"
    }
    assert set(results.n_participants) == {186}
    assert audit["model_refit"] is False
    assert "no cross-fold averaging" in audit["per_subject_rule"]
    for suffix in ("png", "pdf"):
        assert (ROOT / "figures" / f"Figure7B_sensitivity_all_normal_oof.{suffix}").is_file()
