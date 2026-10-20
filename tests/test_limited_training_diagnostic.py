"""Boundary, duplicate and replay safety for offline training-log reductions."""

from io import StringIO
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from src.evaluation.limited_training_diagnostic import (
    _branch_summary, _window, describe, number, read_dashboard,
    validate_replay_overlap, weighted_mean,
)


class LimitedTrainingDiagnosticTests(unittest.TestCase):
    def test_csv_duplicate_cells_are_verified_before_selection(self):
        content = ("record_type,route_active_advantage_mean,route_active_advantage_mean\n"
                   "ppo_epoch,0.2,0.2\nroute_sample,,\n")
        with patch.object(Path, "open", return_value=StringIO(content)):
            rows, audit = read_dashboard(Path("frozen.csv"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(audit["duplicate_columns"], ["route_active_advantage_mean"])
        with patch.object(Path, "open", return_value=StringIO(content+"route_sample,0.1,0.2\n")):
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                read_dashboard(Path("frozen.csv"))

    def test_csv_malformed_width_fails(self):
        with patch.object(Path, "open", return_value=StringIO("record_type,x\nppo_epoch,1,2\n")):
            with self.assertRaisesRegex(ValueError, "width mismatch"):
                read_dashboard(Path("bad.csv"))

    def test_missing_numeric_stays_na_and_nonfinite_fails(self):
        self.assertIsNone(number("NA"))
        self.assertIsNone(describe([None, ""])["mean"])
        self.assertIsNone(weighted_mean([(None, 10), (2, 0)]))
        for value in ("NaN", math.inf, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                number(value)

    def test_sample_weighting_is_not_mean_of_rollout_means(self):
        self.assertAlmostEqual(weighted_mean([(1, 1), (3, 9), (None, 100)]), 2.8)

    def test_replayed_epochs_do_not_multiply_activity_count(self):
        rows = [{"ppo_epoch_index": str(epoch), "update_index": "1",
                 "route_branch_ppo_active_count": "5",
                 "route_branch_ppo_advantage_mean": "2",
                 "route_branch_ppo_approx_kl_mean": str(epoch/1000),
                 "route_branch_ppo_clip_fraction": str(epoch/100)} for epoch in range(4)]
        result = _branch_summary(rows, "route")
        self.assertEqual(result["unique_rollout_active_samples"], 5)
        self.assertAlmostEqual(result["kl_sample_weighted_mean"], .0015)
        self.assertAlmostEqual(result["update_peak_kl"]["max"], .003)
        missing = _branch_summary(rows, "missing")
        self.assertIsNone(missing["unique_rollout_active_samples"])
        self.assertIsNone(missing["update_peak_kl"]["max"])

    def test_end_counter_window_excludes_boundary_and_unoptimized_tail(self):
        rows = [{"environment_steps": v} for v in (28000, 28160, 32000, 32256)]
        rows.append({"environment_steps": 31000, "optimized": False})
        self.assertEqual([r["environment_steps"] for r in _window(rows, 28000, 32000)],
                         [28160, 32000])

    def test_replay_overlap_verifies_not_double_counted(self):
        old = [{"environment_steps": 28160, "update_index": 109, "ppo_epochs": [{"loss": 1}]}]
        later = [dict(old[0]), {"environment_steps": 28416, "update_index": 110, "ppo_epochs": []}]
        audit = validate_replay_overlap(old, later)
        self.assertEqual(audit["overlap_update_count"], 1)
        self.assertEqual(audit["overlap_steps"], [28160])
        later[0] = {**later[0], "ppo_epochs": [{"loss": 2}]}
        with self.assertRaisesRegex(ValueError, "Replay telemetry conflict"):
            validate_replay_overlap(old, later)

    def test_duplicate_update_within_source_fails(self):
        row = {"environment_steps": 28160, "update_index": 109, "ppo_epochs": []}
        with self.assertRaisesRegex(ValueError, "Duplicate update"):
            validate_replay_overlap([row, row], [])


if __name__ == "__main__":
    unittest.main()
