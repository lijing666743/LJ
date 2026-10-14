"""Focused validation for the independent ``remote_stress`` scenario."""

from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from pathlib import Path

from src.cli import build_arg_parser
from src.config import SCENARIO_DEFAULTS, load_run_config
from src.env.environment import U2UMECEnvironment
from src.env.validation import run_environment_sanity
from src.policies.random_policy import RandomPolicy
from src.rollout import RolloutRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMALL_32K_CONFIG = PROJECT_ROOT / (
    "tests/fixtures/configs/"
    "cooperative-load-v2-hierarchical-optionaware-v1-entropy-b0-32k.json"
)
REMOTE_STRESS_32K_CONFIG = PROJECT_ROOT / (
    "tests/fixtures/configs/"
    "cooperative-load-v2-hierarchical-optionaware-v1-remote-stress-32k.json"
)


class RemoteStressScenarioTests(unittest.TestCase):
    def test_registry_preserves_small_and_adds_remote_stress(self) -> None:
        small = load_run_config(cli_overrides={"scenario_id": "small"})
        stress = load_run_config(cli_overrides={"scenario_id": "remote_stress"})

        self.assertEqual(small.environment.arrival_probabilities, (0.04, 0.05, 0.06, 0.05))
        self.assertEqual(small.environment.minimum_task_slack_slots, 10)
        self.assertEqual(small.environment.maximum_task_slack_slots, 150)
        self.assertEqual(small.environment.task_deadline_factor_min, 1.5)
        self.assertEqual(small.environment.task_deadline_factor_max, 3.0)

        self.assertEqual(stress.environment.arrival_probabilities, (0.05, 0.12, 0.02, 0.07))
        self.assertEqual(stress.environment.minimum_task_slack_slots, 18)
        self.assertEqual(stress.environment.maximum_task_slack_slots, 80)
        self.assertEqual(stress.environment.task_deadline_factor_min, 1.75)
        self.assertEqual(stress.environment.task_deadline_factor_max, 2.5)
        self.assertEqual(stress.environment.profile_assignment, small.environment.profile_assignment)
        self.assertEqual(stress.environment.task_data_bits_min, small.environment.task_data_bits_min)
        self.assertEqual(stress.environment.task_data_bits_max, small.environment.task_data_bits_max)
        self.assertEqual(
            stress.environment.task_cycles_per_bit_min,
            small.environment.task_cycles_per_bit_min,
        )
        self.assertEqual(
            stress.environment.task_cycles_per_bit_max,
            small.environment.task_cycles_per_bit_max,
        )
        self.assertEqual(stress.environment.channel_ablation_mode, small.environment.channel_ablation_mode)
        self.assertIn("remote_stress", SCENARIO_DEFAULTS)

    def test_cli_accepts_remote_stress(self) -> None:
        args = build_arg_parser().parse_args(["--scenario-id", "remote_stress", "--show-config"])
        self.assertEqual(args.scenario_id, "remote_stress")
        self.assertTrue(args.show_config)

    def test_experiment_config_changes_environment_not_algorithm_or_reward(self) -> None:
        small = load_run_config(SMALL_32K_CONFIG)
        stress = load_run_config(REMOTE_STRESS_32K_CONFIG)

        self.assertEqual(stress.scenario_id, "remote_stress")
        self.assertEqual(stress.action, small.action)
        self.assertEqual(stress.reward, small.reward)
        self.assertEqual(stress.training, small.training)
        self.assertEqual(stress.reproducibility, small.reproducibility)
        self.assertEqual(stress.evaluation, small.evaluation)

        small_environment = asdict(small.environment)
        stress_environment = asdict(stress.environment)
        changed_fields = {
            key
            for key in small_environment
            if small_environment[key] != stress_environment[key]
        }
        self.assertEqual(
            changed_fields,
            {
                "arrival_probabilities",
                "minimum_task_slack_slots",
                "maximum_task_slack_slots",
                "task_deadline_factor_min",
                "task_deadline_factor_max",
            },
        )

    def test_both_scenarios_pass_sanity_with_empty_initial_queues(self) -> None:
        for scenario_id in ("small", "remote_stress"):
            with self.subTest(scenario_id=scenario_id):
                config = load_run_config(cli_overrides={"scenario_id": scenario_id})
                outcome = run_environment_sanity(config)
                self.assertEqual(outcome.status, "completed", outcome.message)

                environment = U2UMECEnvironment(config)
                environment.reset()
                self.assertEqual(environment.snapshot()["tasks"], [])

    def test_remote_stress_random_policy_rollout_is_finite_and_conserved(self) -> None:
        config = load_run_config(
            cli_overrides={
                "mode": "random",
                "method_id": "random",
                "scenario_id": "remote_stress",
                "environment.episode_horizon": 64,
            }
        )
        outcome = RolloutRunner(config, RandomPolicy(config.seed)).run(write_artifacts=False)

        self.assertEqual(outcome.artifacts, ())
        self.assertEqual(outcome.summary["episode"]["slots_executed"], 64)
        self.assertTrue(outcome.summary["episode"]["truncated"])
        self.assertTrue(outcome.summary["conservation"]["is_conserved"])
        json.dumps(outcome.summary, ensure_ascii=False, sort_keys=True, allow_nan=False)
        for record in outcome.raw_records:
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
