"""Strict semantic-isolation checks for ``remote_stress_v2``."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.cli import build_arg_parser
from src.config import SCENARIO_DEFAULTS, load_run_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMALL_32K_CONFIG = PROJECT_ROOT / (
    "tests/fixtures/configs/"
    "cooperative-load-v2-hierarchical-optionaware-v1-entropy-b0-32k.json"
)
REMOTE_STRESS_V2_32K_CONFIG = PROJECT_ROOT / (
    "tests/fixtures/configs/"
    "cooperative-load-v2-hierarchical-optionaware-v1-remote-stress-v2-32k.json"
)
ARRIVAL_ONLY_DIFF = {"environment.arrival_probabilities"}


def _flatten(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten(value, path))
        else:
            flattened[path] = value
    return flattened


def _changed_paths(left: Mapping[str, Any], right: Mapping[str, Any]) -> set[str]:
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    return {
        path
        for path in left_flat.keys() | right_flat.keys()
        if left_flat.get(path) != right_flat.get(path)
    }


class RemoteStressV2ScenarioTests(unittest.TestCase):
    def test_registry_accepts_remote_stress_v2(self) -> None:
        self.assertIn("remote_stress_v2", SCENARIO_DEFAULTS)
        config = load_run_config(cli_overrides={"scenario_id": "remote_stress_v2"})
        self.assertEqual(config.scenario_id, "remote_stress_v2")
        self.assertEqual(
            config.environment.arrival_probabilities,
            (0.0467, 0.11, 0.02, 0.0633),
        )

    def test_canonical_environment_diff_is_arrival_only(self) -> None:
        small = load_run_config(cli_overrides={"scenario_id": "small"})
        stress = load_run_config(cli_overrides={"scenario_id": "remote_stress_v2"})

        changed = _changed_paths(
            {"environment": asdict(small.environment)},
            {"environment": asdict(stress.environment)},
        )
        self.assertEqual(changed, ARRIVAL_ONLY_DIFF)

    def test_matched_32k_resolved_semantic_diff_is_arrival_only(self) -> None:
        small = load_run_config(SMALL_32K_CONFIG)
        stress = load_run_config(REMOTE_STRESS_V2_32K_CONFIG)
        small_semantic = small.resolved_dict()
        stress_semantic = stress.resolved_dict()
        small_semantic.pop("scenario_id")
        stress_semantic.pop("scenario_id")

        self.assertEqual(
            _changed_paths(small_semantic, stress_semantic),
            ARRIVAL_ONLY_DIFF,
        )

    def test_algorithm_identity_and_training_budget(self) -> None:
        small = load_run_config(SMALL_32K_CONFIG)
        stress = load_run_config(REMOTE_STRESS_V2_32K_CONFIG)

        self.assertEqual(stress.action, small.action)
        self.assertEqual(stress.reward, small.reward)
        self.assertEqual(stress.training.mappo, small.training.mappo)
        self.assertEqual(stress.reproducibility, small.reproducibility)
        self.assertEqual(stress.evaluation, small.evaluation)
        self.assertEqual(stress.output, small.output)
        self.assertEqual(stress.training.mappo.route_decoder_mode, "hierarchical_option_aware_v1")
        self.assertEqual(stress.seed, 42)
        self.assertEqual(stress.training.mappo.max_training_environment_steps, 32_000)
        self.assertEqual(stress.training.mappo.max_training_episodes, 64)

    def test_deadline_and_slack_match_small_b0(self) -> None:
        small = load_run_config(SMALL_32K_CONFIG)
        stress = load_run_config(REMOTE_STRESS_V2_32K_CONFIG)

        for config in (small, stress):
            self.assertEqual(config.environment.task_deadline_factor_min, 1.5)
            self.assertEqual(config.environment.task_deadline_factor_max, 3.0)
            self.assertEqual(config.environment.minimum_task_slack_slots, 10)
            self.assertEqual(config.environment.maximum_task_slack_slots, 150)

    def test_cli_contract_preserves_all_scenarios(self) -> None:
        parser = build_arg_parser()
        for scenario_id in (
            "small",
            "medium",
            "large",
            "remote_stress",
            "remote_stress_v2",
        ):
            with self.subTest(scenario_id=scenario_id):
                args = parser.parse_args(["--scenario-id", scenario_id, "--show-config"])
                self.assertEqual(args.scenario_id, scenario_id)
                self.assertTrue(args.show_config)


if __name__ == "__main__":
    unittest.main()
