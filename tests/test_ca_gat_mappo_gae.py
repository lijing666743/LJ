"""GAE + Return Computation Gate for CA-GAT-MAPPO."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import torch

from src.config import RunConfig
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import (
    ActorObservationTensorizer,
    CAGATMAPPOActor,
    CentralizedStateTensorizer,
    MAPPOCentralizedCritic,
)
from src.models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionMaskBatch,
)
from src.models.ca_gat_mappo_gae import (
    GAEComputationError,
    compute_gae_and_returns,
    compute_rollout_gae,
)
from src.models.ca_gat_mappo_rollout import (
    CAGATMAPPORolloutBuffer,
    CAGATMAPPORolloutTransition,
)


def compute_case(
    reward: list[float],
    *,
    old_value: list[float] | None = None,
    bootstrap_value: list[float] | None = None,
    terminated: list[bool] | None = None,
    truncated: list[bool] | None = None,
    episode_boundary: list[bool] | None = None,
    bootstrap_allowed: list[bool] | None = None,
    sequence_mask: list[bool] | None = None,
    gamma: float = 0.9,
    gae_lambda: float = 0.8,
):
    length = len(reward)
    old_value = old_value if old_value is not None else [0.0] * length
    bootstrap_value = (
        bootstrap_value if bootstrap_value is not None else [0.0] * length
    )
    terminated = terminated if terminated is not None else [False] * length
    truncated = truncated if truncated is not None else [False] * length
    if episode_boundary is None:
        episode_boundary = [
            ended or cut for ended, cut in zip(terminated, truncated)
        ]
    sequence_mask = sequence_mask if sequence_mask is not None else [True] * length
    if bootstrap_allowed is None:
        bootstrap_allowed = [
            valid and not boundary
            for valid, boundary in zip(sequence_mask, episode_boundary)
        ]
    return compute_gae_and_returns(
        reward=torch.tensor(reward, dtype=torch.float32),
        old_value=torch.tensor(old_value, dtype=torch.float32),
        bootstrap_value=torch.tensor(bootstrap_value, dtype=torch.float32),
        terminated=torch.tensor(terminated, dtype=torch.bool),
        truncated=torch.tensor(truncated, dtype=torch.bool),
        episode_boundary=torch.tensor(episode_boundary, dtype=torch.bool),
        bootstrap_allowed=torch.tensor(bootstrap_allowed, dtype=torch.bool),
        gamma=gamma,
        gae_lambda=gae_lambda,
        sequence_mask=torch.tensor(sequence_mask, dtype=torch.bool),
    )


def make_fixed_horizon_chunk():
    base = RunConfig()
    environment = replace(
        base.environment,
        episode_horizon=2,
        uav_count=4,
        arrival_probabilities=(0.0,) * 4,
        profile_assignment=("Balanced",) * 4,
        profile_perturbations=((0.0, 0.0, 0.0, 0.0),) * 4,
        building_layout=(),
        candidate_neighbor_radius_m=2_000.0,
        velocity_std_mps=(0.0, 0.0),
        shadowing_std_db=0.0,
        csi_error_std_db=0.0,
        fixed_csi_aoi_slots=1,
    )
    config = replace(base, environment=environment)
    config.validate()
    environment_api = U2UMECEnvironment(config)
    reset = environment_api.reset()
    observations = reset.observations
    state = reset.centralized_state
    actor_tensorizer = ActorObservationTensorizer(config)
    critic_tensorizer = CentralizedStateTensorizer(config)
    actor = CAGATMAPPOActor(config).eval()
    critic = MAPPOCentralizedCritic(config).eval()
    distribution = CAGATMAPPOActionDistribution(actor, config)
    hidden = actor.initial_hidden(1)
    episode_start = True
    buffer = CAGATMAPPORolloutBuffer(config, capacity=2)

    for _ in range(2):
        actor_batch = actor_tensorizer.encode_step(
            observations,
            episode_start=episode_start,
        )
        mask_batch = SequentialActionMaskBatch.from_observations(observations)
        critic_batch = critic_tensorizer.encode_step(state)
        hidden_before = hidden.detach().clone()
        with torch.no_grad():
            action_output = distribution.sample_actions(
                actor_batch,
                mask_batch,
                hidden_before,
            )
            old_value = critic(critic_batch)[0, 0, 0]
        slot = observations[0].slot
        step = environment_api.step(action_output.proposals[0][0])
        bootstrap_value = None
        if bool(step.info["bootstrap_allowed"]):
            assert step.centralized_state is not None
            next_state = critic_tensorizer.encode_step(step.centralized_state)
            with torch.no_grad():
                bootstrap_value = critic(next_state)[0, 0, 0]
        transition = CAGATMAPPORolloutTransition.from_step(
            spec=actor.spec,
            slot=slot,
            actor_batch=actor_batch,
            action_mask_batch=mask_batch,
            action_output=action_output,
            hidden_in=hidden_before,
            centralized_state=critic_batch,
            old_value=old_value,
            reward=step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
            episode_boundary=step.terminated or step.truncated,
            bootstrap_allowed=bool(step.info["bootstrap_allowed"]),
            bootstrap_value=bootstrap_value,
            executed_action_summary=step.info["executed"],
            rejection_or_downgrade_summary={
                "rejection": step.info["rejection"],
                "downgrade": step.info["downgrade"],
                "canonicalization": step.info["canonicalization"],
            },
        )
        buffer.append(transition)
        if not transition.episode_boundary:
            assert step.observations is not None
            assert step.centralized_state is not None
            observations = step.observations
            state = step.centralized_state
            hidden = action_output.hidden_out.detach().clone()
            episode_start = False
    return config, buffer.finalize()


class CAGATMAPPOGAETests(unittest.TestCase):
    def test_single_non_boundary_td_residual_uses_saved_bootstrap(self) -> None:
        output = compute_case(
            [1.0],
            old_value=[0.5],
            bootstrap_value=[0.25],
            gamma=0.9,
        )
        self.assertEqual(output.bootstrap_mask.tolist(), [True])
        self.assertAlmostEqual(float(output.td_residual[0]), 0.725, places=6)
        self.assertAlmostEqual(float(output.advantage[0]), 0.725, places=6)

    def test_single_terminated_transition_has_no_bootstrap(self) -> None:
        output = compute_case(
            [2.0],
            old_value=[0.75],
            bootstrap_value=[999.0],
            terminated=[True],
        )
        self.assertEqual(output.bootstrap_mask.tolist(), [False])
        self.assertAlmostEqual(float(output.td_residual[0]), 1.25, places=6)

    def test_single_truncated_transition_has_no_bootstrap(self) -> None:
        output = compute_case(
            [2.0],
            old_value=[0.75],
            bootstrap_value=[999.0],
            truncated=[True],
        )
        self.assertEqual(output.bootstrap_mask.tolist(), [False])
        self.assertAlmostEqual(float(output.td_residual[0]), 1.25, places=6)

    def test_fixed_horizon_final_slot_adapter_has_no_bootstrap(self) -> None:
        config, chunk = make_fixed_horizon_chunk()
        output = compute_rollout_gae(chunk, config)
        self.assertEqual(chunk.truncated.tolist(), [False, True])
        self.assertEqual(output.bootstrap_mask.tolist(), [True, False])
        self.assertAlmostEqual(
            float(output.td_residual[-1]),
            float(chunk.reward[-1] - chunk.old_value[-1]),
            places=6,
        )

    def test_multistep_gae_matches_hand_calculation(self) -> None:
        output = compute_case(
            [1.0, 2.0, 3.0],
            old_value=[0.5, 0.6, 0.7],
            bootstrap_value=[0.6, 0.7, 0.8],
            gamma=0.9,
            gae_lambda=0.8,
        )
        expected_delta = torch.tensor([1.04, 2.03, 3.02])
        expected_advantage = torch.tensor([4.067168, 4.2044, 3.02])
        torch.testing.assert_close(output.td_residual, expected_delta)
        torch.testing.assert_close(output.advantage, expected_advantage)

    def test_lambda_zero_reduces_to_one_step_td_residual(self) -> None:
        output = compute_case(
            [1.0, 2.0, 3.0],
            bootstrap_value=[0.5, 0.5, 0.5],
            gae_lambda=0.0,
        )
        torch.testing.assert_close(output.advantage, output.td_residual)

    def test_lambda_one_matches_discounted_residual_recursion(self) -> None:
        output = compute_case(
            [1.0, 2.0, 3.0],
            gamma=0.9,
            gae_lambda=1.0,
        )
        expected = torch.tensor([5.23, 4.7, 3.0])
        torch.testing.assert_close(output.advantage, expected)

    def test_gamma_accepts_values_strictly_inside_unit_interval(self) -> None:
        for gamma in (0.0001, 0.5, 0.9999):
            with self.subTest(gamma=gamma):
                compute_case([1.0], gamma=gamma)

    def test_gamma_rejects_boundaries_nonfinite_and_boolean(self) -> None:
        for gamma in (0.0, 1.0, -0.1, 1.1, math.nan, math.inf, True):
            with self.subTest(gamma=gamma):
                with self.assertRaises(GAEComputationError):
                    compute_case([1.0], gamma=gamma)

    def test_gae_lambda_accepts_closed_unit_interval(self) -> None:
        for gae_lambda in (0.0, 0.5, 1.0):
            with self.subTest(gae_lambda=gae_lambda):
                compute_case([1.0], gae_lambda=gae_lambda)

    def test_gae_lambda_rejects_invalid_nonfinite_and_boolean(self) -> None:
        for gae_lambda in (-0.1, 1.1, math.nan, math.inf, True):
            with self.subTest(gae_lambda=gae_lambda):
                with self.assertRaises(GAEComputationError):
                    compute_case([1.0], gae_lambda=gae_lambda)

    def test_non_boundary_rollout_tail_uses_saved_bootstrap_value(self) -> None:
        zero_tail = compute_case(
            [1.0, 1.0],
            bootstrap_value=[0.0, 0.0],
            gamma=0.9,
        )
        valued_tail = compute_case(
            [1.0, 1.0],
            bootstrap_value=[0.0, 2.0],
            gamma=0.9,
        )
        self.assertAlmostEqual(
            float(valued_tail.td_residual[-1] - zero_tail.td_residual[-1]),
            1.8,
            places=6,
        )

    def test_boundary_tail_does_not_require_a_real_next_value(self) -> None:
        zero = compute_case([1.0], bootstrap_value=[0.0], truncated=[True])
        placeholder = compute_case(
            [1.0],
            bootstrap_value=[1_000_000.0],
            truncated=[True],
        )
        torch.testing.assert_close(zero.td_residual, placeholder.td_residual)
        torch.testing.assert_close(zero.advantage, placeholder.advantage)

    def test_middle_episode_boundary_stops_reverse_recursion(self) -> None:
        output = compute_case(
            [1.0, 2.0, 100.0],
            terminated=[False, True, False],
            gamma=0.9,
            gae_lambda=1.0,
        )
        expected = torch.tensor([2.8, 2.0, 100.0])
        torch.testing.assert_close(output.advantage, expected)

    def test_truncated_boundary_is_not_treated_as_ordinary_cut(self) -> None:
        output = compute_case(
            [1.0, 2.0, 100.0],
            bootstrap_value=[0.0, 10_000.0, 0.0],
            truncated=[False, True, False],
            gamma=0.9,
            gae_lambda=1.0,
        )
        self.assertAlmostEqual(float(output.td_residual[1]), 2.0, places=6)
        self.assertAlmostEqual(float(output.advantage[1]), 2.0, places=6)

    def test_rollout_cut_is_not_treated_as_terminal(self) -> None:
        output = compute_case(
            [0.0, 0.0],
            old_value=[0.0, 0.0],
            bootstrap_value=[0.0, 3.0],
            gamma=0.9,
        )
        self.assertTrue(bool(output.bootstrap_mask[-1]))
        self.assertAlmostEqual(float(output.td_residual[-1]), 2.7, places=6)

    def test_return_target_equals_advantage_plus_old_value(self) -> None:
        output = compute_case(
            [1.0, 2.0, 3.0],
            old_value=[0.25, 0.5, 0.75],
        )
        torch.testing.assert_close(
            output.return_target,
            output.advantage + torch.tensor([0.25, 0.5, 0.75]),
        )

    def test_output_is_shared_team_time_vector_in_float32(self) -> None:
        output = compute_gae_and_returns(
            reward=torch.tensor([1.0, 2.0], dtype=torch.float64),
            old_value=torch.tensor([0.0, 0.0], dtype=torch.float64),
            bootstrap_value=torch.tensor([0.0, 0.0], dtype=torch.float64),
            terminated=torch.tensor([False, False]),
            truncated=torch.tensor([False, False]),
            episode_boundary=torch.tensor([False, False]),
            bootstrap_allowed=torch.tensor([True, True]),
            gamma=0.9,
            gae_lambda=0.8,
        )
        self.assertEqual(output.advantage.shape, (2,))
        self.assertEqual(output.return_target.shape, (2,))
        self.assertEqual(output.advantage.dtype, torch.float32)
        self.assertEqual(output.advantage.device.type, "cpu")

    def test_nonfinite_reward_is_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(GAEComputationError):
                    compute_case([value])

    def test_nonfinite_old_value_is_rejected(self) -> None:
        with self.assertRaises(GAEComputationError):
            compute_case([1.0], old_value=[math.inf])

    def test_nonfinite_bootstrap_is_rejected_even_when_masked(self) -> None:
        with self.assertRaises(GAEComputationError):
            compute_case(
                [1.0],
                bootstrap_value=[math.nan],
                truncated=[True],
            )

    def test_empty_rollout_is_rejected(self) -> None:
        with self.assertRaises(GAEComputationError):
            compute_case([])

    def test_malformed_rank_dtype_and_shape_are_rejected(self) -> None:
        valid = {
            "reward": torch.tensor([1.0]),
            "old_value": torch.tensor([0.0]),
            "bootstrap_value": torch.tensor([0.0]),
            "terminated": torch.tensor([False]),
            "truncated": torch.tensor([False]),
            "episode_boundary": torch.tensor([False]),
            "bootstrap_allowed": torch.tensor([True]),
            "gamma": 0.9,
            "gae_lambda": 0.8,
        }
        malformed = (
            {"reward": torch.tensor([[1.0]])},
            {"old_value": torch.tensor([0], dtype=torch.long)},
            {"terminated": torch.tensor([0], dtype=torch.long)},
            {"truncated": torch.tensor([False, False])},
        )
        for change in malformed:
            with self.subTest(change=tuple(change)):
                with self.assertRaises(GAEComputationError):
                    compute_gae_and_returns(**(valid | change))

    def test_boundary_flags_must_be_consistent_and_distinct(self) -> None:
        with self.assertRaises(GAEComputationError):
            compute_case([1.0], terminated=[True], episode_boundary=[False])
        with self.assertRaises(GAEComputationError):
            compute_case([1.0], terminated=[True], truncated=[True])

    def test_bootstrap_allowed_must_match_non_boundary_positions(self) -> None:
        with self.assertRaises(GAEComputationError):
            compute_case([1.0], bootstrap_allowed=[False])
        with self.assertRaises(GAEComputationError):
            compute_case(
                [1.0],
                truncated=[True],
                bootstrap_allowed=[True],
            )

    def test_sequence_mask_supports_trailing_padding_without_recursion(self) -> None:
        output = compute_case(
            [1.0, 2.0, 999.0],
            sequence_mask=[True, True, False],
            bootstrap_allowed=[True, True, False],
            gamma=0.9,
            gae_lambda=1.0,
        )
        self.assertEqual(output.sequence_mask.tolist(), [True, True, False])
        self.assertAlmostEqual(float(output.advantage[1]), 2.0, places=6)
        self.assertAlmostEqual(float(output.advantage[0]), 2.8, places=6)
        self.assertEqual(float(output.advantage[2]), 0.0)
        self.assertEqual(float(output.return_target[2]), 0.0)

    def test_sequence_mask_rejects_internal_gaps(self) -> None:
        with self.assertRaises(GAEComputationError):
            compute_case(
                [1.0, 2.0, 3.0],
                sequence_mask=[True, False, True],
                bootstrap_allowed=[True, False, True],
            )

    def test_raw_advantage_is_not_normalized(self) -> None:
        output = compute_case([1.0, 2.0, 3.0], gae_lambda=0.0)
        torch.testing.assert_close(output.advantage, output.td_residual)
        self.assertNotAlmostEqual(float(output.advantage.mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
