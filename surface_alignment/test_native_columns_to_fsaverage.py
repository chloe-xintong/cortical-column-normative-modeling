#!/usr/bin/env python3
"""Small synthetic checks for sphere nearest-neighbour and row recovery."""
import importlib.util
from pathlib import Path

import numpy as np

MODULE = Path(__file__).with_name("native_columns_to_fsaverage.py")
spec = importlib.util.spec_from_file_location("alignment", MODULE)
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)


def test_nearest_template_vertices():
    template = np.eye(3)
    native = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.99, 0.01, 0.0]])
    ids, _, angle = alignment.nearest_template_vertices(native, template)
    assert ids.tolist() == [0, 1, 0]
    assert np.all(np.isfinite(angle))
    assert angle[0] == 0.0


def test_row_recovery_fails_for_wrong_cleaned_rows():
    native_ids = np.array([0, 1, 2])
    profile = np.ones((1, 21))  # two retained rows expected after removing row 1
    surface = alignment.SurfaceSet(np.zeros((3, 3)), np.ones((3, 3)), np.eye(3))
    ids, sphere, angle = alignment.nearest_template_vertices(np.eye(3), np.eye(3))
    try:
        alignment.build_region_rows("S", "lh", "test", native_ids, {"md": profile}, [1], surface, surface, np.eye(3), ids, sphere, angle)
    except ValueError as exc:
        assert "Row recovery failed" in str(exc)
    else:
        raise AssertionError("Expected strict row-recovery validation")


if __name__ == "__main__":
    test_nearest_template_vertices()
    test_row_recovery_fails_for_wrong_cleaned_rows()
    print("synthetic surface-alignment tests passed")
