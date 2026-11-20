from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tools.build_activation_calibration import (
    CUTOFFS, calibrate, cutoff_value, fit_discovery, held_out_rates, select_cutoff, split_rows,
)


class SceneActivationRateTests(unittest.TestCase):
    def test_cutoff_boundaries(self):
        maximum = 7.7
        self.assertIsNone(select_cutoff(0, maximum))
        self.assertIsNone(select_cutoff(10, 0))
        self.assertEqual(select_cutoff(np.nextafter(0.0, 1.0), maximum), 0)
        for previous, cutoff in zip(CUTOFFS, CUTOFFS[1:]):
            boundary = cutoff_value(maximum, cutoff)
            self.assertEqual(select_cutoff(np.nextafter(boundary, -np.inf), maximum), previous)
            self.assertEqual(select_cutoff(boundary, maximum), previous)
            self.assertEqual(select_cutoff(np.nextafter(boundary, np.inf), maximum), cutoff)
        self.assertEqual(select_cutoff(2 * maximum, maximum), 60)

    def test_rates_use_strict_comparison_and_separate_denominators(self):
        samples = [0, 2, 4, 6, 8]
        target = np.array([True] * 10 + [False] * 30)
        rows = held_out_rates(np.array(samples * 2 + [0, 2, 2, 6, 8] * 6), target, 10)
        for row, target_count, non_target_count in zip(rows, (4, 3, 2, 1), (4, 2, 2, 1)):
            self.assertEqual(row["target_total"], 10)
            self.assertEqual(row["non_target_total"], 30)
            self.assertEqual(row["target_activated"], 2 * target_count)
            self.assertEqual(row["non_target_activated"], 6 * non_target_count)
            self.assertEqual(row["target_pct"], target_count * 20)
            self.assertEqual(row["non_target_pct"], non_target_count * 20)

    def test_split_is_stratified_and_independent_of_row_order(self):
        scenes = pd.Series(["a"] * 40 + ["b"] * 54)
        filenames = pd.Series([f"{scene}/{i}.jpg" for i, scene in enumerate(scenes)])
        original = split_rows(filenames, scenes)
        shuffled = filenames.sample(frac=1, random_state=42)
        reordered = split_rows(shuffled, scenes.loc[shuffled.index])
        pd.testing.assert_series_equal(original, reordered.sort_index())
        self.assertEqual(scenes[original].value_counts().to_dict(), {"b": 14, "a": 10})

    def test_held_out_values_change_rates_without_changing_mapping_or_maximum(self):
        scenes = pd.Series(["a"] * 40 + ["b"] * 120)
        filenames = pd.Series([f"{scene}/{i}.jpg" for i, scene in enumerate(scenes)])
        activations = pd.DataFrame({"0": [5.0] * 40 + [0.0] * 120})
        held_out = split_rows(filenames, scenes)
        discovery, _, _ = fit_discovery(activations[~held_out], filenames[~held_out], scenes[~held_out])
        frozen = [{**row} for row in discovery]
        original = calibrate(discovery, activations[held_out], scenes[held_out], {"a": "A", "b": "B"})
        changed = calibrate(frozen, activations[held_out] + 1_000, scenes[held_out], {"a": "A", "b": "B"})
        self.assertEqual(discovery, frozen)
        self.assertEqual(discovery[0]["primary_scene"], "a")
        self.assertEqual(discovery[0]["reference_max"], 5.0)
        self.assertEqual([row["non_target_pct"] for row in original], [0] * 4)
        self.assertEqual([row["non_target_pct"] for row in changed], [100] * 4)


if __name__ == "__main__":
    unittest.main()
