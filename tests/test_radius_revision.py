"""Regression tests for the Small Scenario candidate-radius revision."""

from __future__ import annotations

import unittest

import numpy as np

from src.cli import build_arg_parser, build_run_config_from_args
from src.config import load_run_config
from src.env.mobility import MobilityModel
from src.env.topology import DynamicTopology


class CandidateRadiusRevisionTests(unittest.TestCase):
    def test_resolved_radius_is_small_specific_and_preserves_other_scenarios(self) -> None:
        small = load_run_config()
        medium = load_run_config(cli_overrides={"scenario_id": "medium"})
        large = load_run_config(cli_overrides={"scenario_id": "large"})

        self.assertEqual(small.environment.candidate_neighbor_radius_m, 525.0)
        self.assertEqual(medium.environment.candidate_neighbor_radius_m, 500.0)
        self.assertEqual(large.environment.candidate_neighbor_radius_m, 500.0)
        self.assertEqual(
            small.snapshot_dict()["environment"]["candidate_neighbor_radius_m"],
            525.0,
        )

    def test_cli_radius_override_remains_higher_precedence_than_scenario_default(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--scenario-id",
                "small",
                "--set",
                "environment.candidate_neighbor_radius_m=510",
            ]
        )

        config = build_run_config_from_args(args)

        self.assertEqual(config.environment.candidate_neighbor_radius_m, 510.0)

    def test_radius_is_candidate_edge_cutoff_with_boundary_equality(self) -> None:
        positions = np.array(
            [
                [0.0, 0.0, 80.0],
                [525.0, 0.0, 80.0],
                [525.01, 0.0, 80.0],
            ]
        )

        mask = DynamicTopology(525.0).compute(0, positions).candidate_neighbors

        self.assertTrue(mask[0, 1])
        self.assertFalse(mask[0, 2])
        self.assertFalse(np.any(np.diag(mask)))

    def test_seed42_key_dynamic_edge_has_partial_availability_at_525_m(self) -> None:
        config = load_run_config()
        mobility = MobilityModel.from_run_config(config)
        mobility.reset()
        topology = DynamicTopology(config.environment.candidate_neighbor_radius_m)
        availability = np.mean(
            np.stack(
                [
                    topology.compute(slot, state.positions_m).candidate_neighbors
                    for slot, state in enumerate(mobility._trajectory)
                ]
            ),
            axis=0,
        )

        self.assertGreater(availability[1, 2], 0.0)
        self.assertLess(availability[1, 2], 1.0)
        self.assertAlmostEqual(availability[1, 2], 0.514, delta=0.01)
        np.testing.assert_allclose(
            availability,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.514, 0.0],
                    [0.0, 0.514, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
            atol=0.01,
        )


if __name__ == "__main__":
    unittest.main()
