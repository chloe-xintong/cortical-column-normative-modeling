"""Minimal unit tests for Figure 4D boundary matching."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from boundary_matching import (
    deterministic_distance_match,
    distance_bin_index,
    filter_candidate_frame,
    label_candidate_pools,
)


class BoundaryMatchingTests(unittest.TestCase):
    def test_frozen_bin_boundaries(self):
        values = np.array([0.0, 0.01, 2.0, 2.01, 4.0, 8.01, 10.0, 10.01])
        expected = np.array([-1, 0, 0, 1, 1, 4, 4, -1], dtype=np.int8)
        np.testing.assert_array_equal(distance_bin_index(values), expected)

    def test_label_classification(self):
        frame = pd.DataFrame(
            {"candidate_vertex": [2, 3, 4], "candidate_dk_label": [7, 8, 7]}
        )
        within, across = label_candidate_pools(frame, anchor_label=7)
        self.assertEqual(within["candidate_vertex"].tolist(), [2, 4])
        self.assertEqual(across["candidate_vertex"].tolist(), [3])

    def test_self_pair_and_invalid_profile_removal(self):
        frame = pd.DataFrame(
            {"anchor_vertex": [1, 1, 1], "candidate_vertex": [1, 2, 3]}
        )
        valid = np.array([True, True, True, False])
        kept = filter_candidate_frame(frame, valid)
        self.assertEqual(kept["candidate_vertex"].tolist(), [2])

    def test_unique_deterministic_caliper_matching(self):
        within = pd.DataFrame(
            {
                "candidate_vertex": [10, 11, 12],
                "candidate_dk_label": [3, 3, 3],
                "geodesic_distance_mm": [1.00, 1.20, 1.40],
            }
        )
        across = pd.DataFrame(
            {
                "candidate_vertex": [20, 21, 22],
                "candidate_dk_label": [4, 4, 5],
                "geodesic_distance_mm": [1.04, 1.18, 1.80],
            }
        )
        first = deterministic_distance_match(within, across, 15, 0.1)
        second = deterministic_distance_match(within, across, 15, 0.1)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 2)
        self.assertFalse(first["within_candidate_vertex"].duplicated().any())
        self.assertFalse(first["across_candidate_vertex"].duplicated().any())
        self.assertLessEqual(first["absolute_distance_difference_mm"].max(), 0.1)

    def test_first_match_equals_brute_force_minimum(self):
        rng = np.random.default_rng(42)
        within = pd.DataFrame(
            {
                "candidate_vertex": np.arange(100, 140),
                "candidate_dk_label": 3,
                "geodesic_distance_mm": rng.uniform(2, 4, 40),
            }
        )
        across = pd.DataFrame(
            {
                "candidate_vertex": np.arange(200, 255),
                "candidate_dk_label": 4,
                "geodesic_distance_mm": rng.uniform(2, 4, 55),
            }
        )
        matched = deterministic_distance_match(within, across, 1, 2.0).iloc[0]
        brute = min(
            (
                abs(w.geodesic_distance_mm - a.geodesic_distance_mm),
                int(w.candidate_vertex), int(a.candidate_vertex),
            )
            for w in within.itertuples(index=False)
            for a in across.itertuples(index=False)
        )
        self.assertEqual(int(matched.within_candidate_vertex), brute[1])
        self.assertEqual(int(matched.across_candidate_vertex), brute[2])
        self.assertAlmostEqual(matched.absolute_distance_difference_mm, brute[0])


if __name__ == "__main__":
    unittest.main()
