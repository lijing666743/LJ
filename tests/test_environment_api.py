"""Public API tests for the fully integrated U2U MEC environment."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

import numpy as np

from src.cli import main
from src.config import RunConfig
from src.env.environment import EnvironmentError, U2UMECEnvironment
from src.runner import Runner


def make_environment_config(
    *,
    horizon: int = 3,
    seed: int = 42,
    arrival_probabilities: tuple[float, ...] | None = None,
) -> RunConfig:
    """Return a small, deterministic, validation-clean environment config."""

    base = RunConfig()
    arrivals = (
        tuple(0.0 for _ in range(base.environment.uav_count))
        if arrival_probabilities is None
        else arrival_probabilities
    )
    environment = replace(
        base.environment,
        episode_horizon=horizon,
        arrival_probabilities=arrivals,
        building_layout=(),
        candidate_neighbor_radius_m=2_000.0,
        velocity_std_mps=(0.0, 0.0),
        shadowing_std_db=0.0,
        csi_error_std_db=0.0,
        fixed_csi_aoi_slots=1,
    )
    config = replace(base, seed=seed, environment=environment)
    config.validate()
    return config


class EnvironmentApiTests(unittest.TestCase):
    def test_reset_returns_aligned_readonly_json_safe_records(self) -> None:
        environment = U2UMECEnvironment(make_environment_config())
        result = environment.reset()
        count = environment.config.environment.uav_count

        self.assertEqual(len(result.observations), count)
        self.assertIs(result.observation, result.observations)
        self.assertIs(result.actor_observations, result.observations)
        self.assertIs(result.state, result.centralized_state)
        self.assertEqual(result.centralized_state.slot, 0)
        self.assertEqual(
            [observation.uav_id for observation in result.observations],
            list(range(count)),
        )
        self.assertEqual(
            [observation.paper_uav_id for observation in result.observations],
            list(range(1, count + 1)),
        )
        self.assertFalse(result.centralized_state.positions_m.flags.writeable)
        self.assertFalse(result.centralized_state.true_channel.flags.writeable)
        self.assertFalse(result.centralized_state.stale_csi.flags.writeable)
        self.assertFalse(result.centralized_state.interference_history_w.flags.writeable)
        self.assertFalse(result.observations[0].candidate_neighbor_mask.flags.writeable)
        self.assertFalse(result.observations[0].edge_history.stale_csi.flags.writeable)
        self.assertFalse(environment.public_messages.valid_mask.flags.writeable)
        with self.assertRaises(ValueError):
            result.centralized_state.positions_m[0, 0] = -1.0
        with self.assertRaises(ValueError):
            result.observations[0].candidate_neighbor_mask[1] = False

        payload = {
            "observations": [item.snapshot() for item in result.observations],
            "state": result.centralized_state.snapshot(),
            "info": result.info,
        }
        json.dumps(payload, allow_nan=False, sort_keys=True)

    def test_canonical_proposals_are_legal_and_step_info_is_auditable(self) -> None:
        environment = U2UMECEnvironment(make_environment_config())
        environment.reset()
        proposals = environment.canonical_proposals()
        assert environment.current_observations is not None

        self.assertTrue(
            all(
                observation.action_masks.is_legal(proposal)
                for observation, proposal in zip(environment.current_observations, proposals)
            )
        )
        result = environment.step(proposals)

        self.assertFalse(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.info["slot"], 0)
        self.assertEqual(result.info["next_slot"], 1)
        self.assertEqual(len(result.info["proposal"]), len(proposals))
        self.assertEqual(len(result.info["executed"]), len(proposals))
        self.assertIn("rejection", result.info)
        self.assertIn("downgrade", result.info)
        self.assertIn("canonicalization", result.info)
        self.assertIn("reward", result.info)
        self.assertIn("conservation", result.info)
        assert result.observations is not None
        self.assertTrue(result.observations[0].previous_action.valid)
        self.assertEqual(result.observations[0].previous_action.source_slot, 0)
        self.assertEqual(result.observations[0].previous_action.proposal, proposals[0])

    def test_invalid_joint_actions_are_rejected_before_state_mutation(self) -> None:
        environment = U2UMECEnvironment(make_environment_config())
        environment.reset()
        proposals = environment.canonical_proposals()
        before = environment.snapshot()

        with self.assertRaises(EnvironmentError):
            environment.step(proposals[:-1])
        self.assertEqual(environment.snapshot(), before)

        duplicate = (proposals[0], proposals[0], proposals[2], proposals[3])
        with self.assertRaises(EnvironmentError):
            environment.step(duplicate)
        self.assertEqual(environment.snapshot(), before)

        illegal = list(proposals)
        illegal[0] = replace(illegal[0], route="local")
        with self.assertRaises(EnvironmentError):
            environment.step(illegal)
        self.assertEqual(environment.snapshot(), before)

        for invalid_id in (False, 0.0):
            invalid = list(proposals)
            invalid[0] = replace(invalid[0], uav_id=invalid_id)
            self.assertFalse(
                environment.current_observations[0].action_masks.is_legal(invalid[0])
            )
            with self.assertRaises(EnvironmentError):
                environment.step(invalid)
            self.assertEqual(environment.snapshot(), before)

    def test_horizon_step_returns_no_next_records_and_requires_reset(self) -> None:
        environment = U2UMECEnvironment(make_environment_config(horizon=1))
        environment.reset()
        result = environment.step(environment.canonical_proposals())

        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)
        self.assertIsNone(result.observations)
        self.assertIsNone(result.centralized_state)
        self.assertIsNone(result.next_observations)
        self.assertIsNone(result.next_state)
        self.assertTrue(environment.done)
        self.assertEqual(environment.slot, 1)
        with self.assertRaises(EnvironmentError):
            environment.step(())

        reset = environment.reset()
        self.assertFalse(environment.done)
        self.assertEqual(reset.centralized_state.slot, 0)

    def test_reset_clears_episode_owned_tasks_metrics_and_histories(self) -> None:
        arrivals = (1.0, 0.0, 0.0, 0.0)
        environment = U2UMECEnvironment(
            make_environment_config(horizon=3, arrival_probabilities=arrivals)
        )
        environment.reset()
        environment.step(environment.canonical_proposals())
        assert environment.metrics is not None
        self.assertEqual(environment.metrics.generated_task_count, 1)
        self.assertNotEqual(environment.traffic.history_snapshot(), ())

        result = environment.reset()
        assert environment.metrics is not None
        assert environment.lifecycle is not None
        assert environment.previous_actions is not None
        assert environment.public_messages is not None
        self.assertEqual(environment.lifecycle.tasks, {})
        self.assertEqual(environment.metrics.generated_task_count, 0)
        self.assertEqual(environment.metrics.completed_task_count, 0)
        self.assertEqual(environment.metrics.expired_task_count, 0)
        self.assertEqual(environment.metrics.truncated_task_count, 0)
        self.assertIsNone(environment.metrics.outage_ratio)
        self.assertIsNone(environment.metrics.mean_completed_e2e_latency_s)
        self.assertEqual(environment.traffic.history_snapshot(), ())
        self.assertEqual(environment.previous_actions.source_slot, -1)
        self.assertFalse(np.any(environment.public_messages.valid_mask))
        self.assertEqual(result.info["metrics"]["total_active_energy_j"], 0.0)

    def test_runner_uses_the_real_environment_sanity_backend(self) -> None:
        config = replace(
            make_environment_config(),
            mode="environment_sanity",
            method_id="environment",
        )
        result = Runner().run(config)

        self.assertEqual(result.status, "completed")
        self.assertIn("environment sanity passed", result.message)
        self.assertEqual(result.artifacts, ())

    def test_direct_and_interactive_launchers_resolve_the_same_backend(self) -> None:
        direct_output: list[str] = []
        direct_status = main(
            [
                "--mode",
                "environment_sanity",
                "--method-id",
                "environment",
                "--scenario-id",
                "small",
                "--seed",
                "42",
            ],
            output_fn=direct_output.append,
        )
        answers = iter(["1", "small", "", "42"])
        interactive_output: list[str] = []
        interactive_status = main(
            [],
            input_fn=lambda _prompt: next(answers),
            output_fn=interactive_output.append,
        )

        self.assertEqual(direct_status, 0)
        self.assertEqual(interactive_status, 0)
        direct_resolved = next(
            line for line in direct_output if line.startswith("Resolved RunConfig")
        )
        interactive_resolved = next(
            line
            for line in interactive_output
            if line.startswith("Resolved RunConfig")
        )
        self.assertEqual(direct_resolved, interactive_resolved)
        self.assertTrue(any("status=completed" in line for line in direct_output))
        self.assertTrue(any("status=completed" in line for line in interactive_output))


if __name__ == "__main__":
    unittest.main()
