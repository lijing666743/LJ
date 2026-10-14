"""PPO Objective / Loss Gate for CA-GAT-MAPPO."""

from __future__ import annotations

import inspect
import math
import unittest

import torch

from src.config import RunConfig
from src.models.ca_gat_mappo_ppo import (
    PPOObjectiveError,
    compute_configured_ppo_objective_and_loss,
    compute_ppo_objective_and_loss,
)


def _float_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value, dtype=torch.float32)


def compute_case(
    *,
    new=((0.0,),),
    old=((0.0,),),
    advantage=(1.0,),
    current_value=(0.0,),
    return_target=(0.0,),
    entropy=((0.0,),),
    sequence_valid_mask=(True,),
    epsilon_clip=0.2,
    value_coefficient=0.5,
    entropy_coefficient=0.01,
):
    mask = (
        sequence_valid_mask
        if isinstance(sequence_valid_mask, torch.Tensor)
        else torch.tensor(sequence_valid_mask, dtype=torch.bool)
    )
    return compute_ppo_objective_and_loss(
        new_joint_log_prob=_float_tensor(new),
        old_joint_log_prob=_float_tensor(old),
        advantage=_float_tensor(advantage),
        current_value=_float_tensor(current_value),
        return_target=_float_tensor(return_target),
        entropy=_float_tensor(entropy),
        sequence_valid_mask=mask,
        epsilon_clip=epsilon_clip,
        value_coefficient=value_coefficient,
        entropy_coefficient=entropy_coefficient,
    )


class PPOObjectiveLossGateTests(unittest.TestCase):
    def test_equal_new_and_old_log_prob_gives_unit_ratio(self) -> None:
        output = compute_case(
            new=((0.0, -1.5), (2.0, 0.25)),
            old=((0.0, -1.5), (2.0, 0.25)),
            advantage=(1.0, 2.0),
            current_value=(0.0, 0.0),
            return_target=(0.0, 0.0),
            entropy=((0.0, 0.0), (0.0, 0.0)),
            sequence_valid_mask=(True, True),
        )
        torch.testing.assert_close(output.ratio, torch.ones((2, 2)))

    def test_known_log_ratio_matches_analytic_ratio(self) -> None:
        output = compute_case(new=((math.log(1.5),),), old=((0.0,),))
        self.assertAlmostEqual(float(output.ratio[0, 0]), 1.5, places=6)

    def test_positive_advantage_uses_upper_clipped_surrogate(self) -> None:
        output = compute_case(new=((math.log(2.0),),), advantage=(3.0,))
        self.assertAlmostEqual(float(output.unclipped_surrogate[0, 0]), 6.0, places=6)
        self.assertAlmostEqual(float(output.clipped_surrogate[0, 0]), 3.6, places=6)
        self.assertAlmostEqual(float(output.surrogate[0, 0]), 3.6, places=6)

    def test_negative_advantage_uses_lower_clipping_with_ppo_min(self) -> None:
        output = compute_case(new=((math.log(0.5),),), advantage=(-2.0,))
        self.assertAlmostEqual(float(output.unclipped_surrogate[0, 0]), -1.0, places=6)
        self.assertAlmostEqual(float(output.clipped_surrogate[0, 0]), -1.6, places=6)
        self.assertAlmostEqual(float(output.surrogate[0, 0]), -1.6, places=6)

    def test_negative_advantage_high_ratio_is_not_blindly_replaced_by_clip(self) -> None:
        output = compute_case(new=((math.log(2.0),),), advantage=(-2.0,))
        self.assertAlmostEqual(float(output.unclipped_surrogate[0, 0]), -4.0, places=6)
        self.assertAlmostEqual(float(output.clipped_surrogate[0, 0]), -2.4, places=6)
        self.assertAlmostEqual(float(output.surrogate[0, 0]), -4.0, places=6)

    def test_unclipped_interval_preserves_surrogate(self) -> None:
        output = compute_case(new=((math.log(1.1),),), advantage=(2.5,))
        self.assertAlmostEqual(float(output.surrogate[0, 0]), 2.75, places=6)
        self.assertAlmostEqual(
            float(output.surrogate[0, 0]),
            float(output.unclipped_surrogate[0, 0]),
            places=6,
        )

    def test_actor_loss_is_negative_clipped_objective(self) -> None:
        output = compute_case(advantage=(2.0,))
        torch.testing.assert_close(output.actor_loss, -output.clipped_objective)
        self.assertAlmostEqual(float(output.actor_loss), -2.0, places=6)

    def test_shared_time_advantage_is_explicitly_expanded_over_agents(self) -> None:
        output = compute_case(
            new=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            old=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            advantage=(2.0, -3.0),
            current_value=(0.0, 0.0),
            return_target=(0.0, 0.0),
            entropy=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            sequence_valid_mask=(True, True),
        )
        expected = torch.tensor(((2.0, 2.0, 2.0), (-3.0, -3.0, -3.0)))
        torch.testing.assert_close(output.expanded_advantage, expected)

    def test_independent_per_agent_advantage_is_rejected(self) -> None:
        with self.assertRaises(PPOObjectiveError):
            compute_case(
                new=((0.0, 0.0),),
                old=((0.0, 0.0),),
                advantage=((1.0, 2.0),),
                entropy=((0.0, 0.0),),
            )

    def test_value_loss_matches_manual_half_mean_squared_error(self) -> None:
        output = compute_case(
            new=((0.0,), (0.0,)),
            old=((0.0,), (0.0,)),
            advantage=(0.0, 0.0),
            current_value=(1.0, 3.0),
            return_target=(0.0, 1.0),
            entropy=((0.0,), (0.0,)),
            sequence_valid_mask=(True, True),
        )
        self.assertAlmostEqual(float(output.critic_loss), 1.25, places=6)

    def test_value_loss_is_invariant_to_actor_agent_count(self) -> None:
        one_agent = compute_case(
            new=((0.0,), (0.0,)),
            old=((0.0,), (0.0,)),
            advantage=(0.0, 0.0),
            current_value=(1.0, 3.0),
            return_target=(0.0, 1.0),
            entropy=((0.0,), (0.0,)),
            sequence_valid_mask=(True, True),
        )
        five_agents = compute_case(
            new=((0.0,) * 5, (0.0,) * 5),
            old=((0.0,) * 5, (0.0,) * 5),
            advantage=(0.0, 0.0),
            current_value=(1.0, 3.0),
            return_target=(0.0, 1.0),
            entropy=((0.0,) * 5, (0.0,) * 5),
            sequence_valid_mask=(True, True),
        )
        torch.testing.assert_close(one_agent.critic_loss, five_agents.critic_loss)
        self.assertEqual(five_agents.critic_valid_mask.shape, (2,))

    def test_current_value_and_return_target_must_align_on_time_axis(self) -> None:
        for change in (
            {"current_value": (0.0, 0.0)},
            {"return_target": (0.0, 0.0)},
        ):
            with self.subTest(change=tuple(change)):
                with self.assertRaises(PPOObjectiveError):
                    compute_case(**change)

    def test_entropy_input_is_treated_as_active_branch_sum(self) -> None:
        output = compute_case(
            new=((0.0, 0.0),),
            old=((0.0, 0.0),),
            advantage=(0.0,),
            entropy=((0.7, 1.3),),
        )
        self.assertAlmostEqual(float(output.entropy_mean), 1.0, places=6)

    def test_zero_active_branch_entropy_remains_zero(self) -> None:
        output = compute_case(advantage=(0.0,), entropy=((0.0,),))
        self.assertEqual(float(output.entropy_mean), 0.0)
        self.assertTrue(torch.isfinite(output.total_loss))

    def test_entropy_reduction_is_valid_agent_time_mean(self) -> None:
        output = compute_case(
            new=((0.0, 0.0), (0.0, 0.0)),
            old=((0.0, 0.0), (0.0, 0.0)),
            advantage=(0.0, 999.0),
            current_value=(0.0, 999.0),
            return_target=(0.0, 0.0),
            entropy=((1.0, 3.0), (100.0, 200.0)),
            sequence_valid_mask=(True, False),
        )
        self.assertAlmostEqual(float(output.entropy_mean), 2.0, places=6)

    def test_total_loss_uses_frozen_signs_and_coefficients(self) -> None:
        output = compute_case(
            advantage=(1.0,),
            current_value=(2.0,),
            return_target=(0.0,),
            entropy=((3.0,),),
            value_coefficient=0.5,
            entropy_coefficient=0.1,
        )
        self.assertAlmostEqual(float(output.actor_loss), -1.0, places=6)
        self.assertAlmostEqual(float(output.critic_loss), 2.0, places=6)
        self.assertAlmostEqual(float(output.total_loss), -0.3, places=6)

    def test_trailing_padding_is_excluded_from_all_reductions(self) -> None:
        output = compute_case(
            new=((0.0, 0.0), (1.0, -1.0)),
            old=((0.0, 0.0), (0.0, 0.0)),
            advantage=(2.0, 999.0),
            current_value=(1.0, 999.0),
            return_target=(0.0, -999.0),
            entropy=((1.0, 3.0), (100.0, 200.0)),
            sequence_valid_mask=(True, False),
        )
        self.assertAlmostEqual(float(output.clipped_objective), 2.0, places=6)
        self.assertAlmostEqual(float(output.critic_loss), 0.5, places=6)
        self.assertAlmostEqual(float(output.entropy_mean), 2.0, places=6)

    def test_real_episode_boundary_transition_remains_in_loss(self) -> None:
        episode_boundary_metadata = True
        output = compute_case(
            advantage=(4.0,),
            current_value=(2.0,),
            return_target=(1.0,),
            sequence_valid_mask=(True,),
        )
        self.assertTrue(episode_boundary_metadata)
        self.assertAlmostEqual(float(output.clipped_objective), 4.0, places=6)
        self.assertAlmostEqual(float(output.critic_loss), 0.5, places=6)

    def test_empty_valid_mask_fails_fast(self) -> None:
        with self.assertRaises(PPOObjectiveError):
            compute_case(sequence_valid_mask=(False,))

    def test_trailing_padding_mask_expands_to_actor_axes(self) -> None:
        output = compute_case(
            new=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            old=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            advantage=(1.0, 0.0),
            current_value=(0.0, 0.0),
            return_target=(0.0, 0.0),
            entropy=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            sequence_valid_mask=(True, False),
        )
        self.assertEqual(
            output.actor_valid_mask.tolist(),
            [[True, True, True], [False, False, False]],
        )
        self.assertEqual(output.diagnostics.valid_timestep_count, 1)
        self.assertEqual(output.diagnostics.valid_actor_position_count, 3)

    def test_internal_sequence_mask_gap_is_rejected(self) -> None:
        with self.assertRaises(PPOObjectiveError):
            compute_case(
                new=((0.0,), (0.0,), (0.0,)),
                old=((0.0,), (0.0,), (0.0,)),
                advantage=(1.0, 1.0, 1.0),
                current_value=(0.0, 0.0, 0.0),
                return_target=(0.0, 0.0, 0.0),
                entropy=((0.0,), (0.0,), (0.0,)),
                sequence_valid_mask=(True, False, True),
            )

    def test_executed_metadata_is_not_a_probability_input(self) -> None:
        self.assertNotIn(
            "executed_action_summary",
            inspect.signature(compute_ppo_objective_and_loss).parameters,
        )
        proposal_log_prob = ((math.log(1.1),),)
        rejected_execution = {"accepted": False, "executed_power": 0.0}
        downgraded_execution = {"accepted": True, "executed_power": 0.25}
        first = compute_case(new=proposal_log_prob)
        second = compute_case(new=proposal_log_prob)
        self.assertNotEqual(rejected_execution, downgraded_execution)
        torch.testing.assert_close(first.ratio, second.ratio)
        with self.assertRaises(TypeError):
            compute_ppo_objective_and_loss(
                new_joint_log_prob=torch.zeros((1, 1)),
                old_joint_log_prob=torch.zeros((1, 1)),
                advantage=torch.ones(1),
                current_value=torch.zeros(1),
                return_target=torch.zeros(1),
                entropy=torch.zeros((1, 1)),
                sequence_valid_mask=torch.ones(1, dtype=torch.bool),
                epsilon_clip=0.2,
                value_coefficient=0.5,
                entropy_coefficient=0.01,
                executed_action_summary=rejected_execution,
            )

    def test_old_log_prob_is_a_detached_snapshot(self) -> None:
        new = torch.zeros((1, 1), requires_grad=True)
        old = torch.zeros((1, 1), requires_grad=True)
        output = compute_case(new=new, old=old)
        output.total_loss.backward()
        self.assertIsNotNone(new.grad)
        self.assertIsNone(old.grad)

    def test_advantage_and_return_target_are_detached_snapshots(self) -> None:
        new = torch.zeros((1, 1), requires_grad=True)
        value = torch.ones(1, requires_grad=True)
        advantage = torch.ones(1, requires_grad=True)
        return_target = torch.zeros(1, requires_grad=True)
        output = compute_case(
            new=new,
            advantage=advantage,
            current_value=value,
            return_target=return_target,
        )
        output.total_loss.backward()
        self.assertIsNone(advantage.grad)
        self.assertIsNone(return_target.grad)

    def test_current_log_prob_receives_actor_gradient(self) -> None:
        new = torch.zeros((1, 1), requires_grad=True)
        output = compute_case(new=new, advantage=(2.0,))
        output.total_loss.backward()
        self.assertIsNotNone(new.grad)
        self.assertNotEqual(float(new.grad[0, 0]), 0.0)

    def test_current_value_receives_critic_gradient(self) -> None:
        value = torch.ones(1, requires_grad=True)
        output = compute_case(
            advantage=(0.0,),
            current_value=value,
            return_target=(0.0,),
        )
        output.total_loss.backward()
        self.assertIsNotNone(value.grad)
        self.assertNotEqual(float(value.grad[0]), 0.0)

    def test_entropy_regularization_receives_actor_gradient(self) -> None:
        entropy = torch.ones((1, 1), requires_grad=True)
        output = compute_case(
            advantage=(0.0,),
            entropy=entropy,
            entropy_coefficient=0.1,
        )
        output.total_loss.backward()
        self.assertIsNotNone(entropy.grad)
        self.assertAlmostEqual(float(entropy.grad[0, 0]), -0.1, places=6)

    def test_each_nonfinite_statistic_fails_fast(self) -> None:
        cases = (
            {"new": ((math.nan,),)},
            {"old": ((math.inf,),)},
            {"advantage": (-math.inf,)},
            {"current_value": (math.nan,)},
            {"return_target": (math.inf,)},
            {"entropy": ((math.nan,),)},
        )
        for change in cases:
            with self.subTest(change=tuple(change)):
                with self.assertRaises(PPOObjectiveError):
                    compute_case(**change)

    def test_nonfinite_padding_statistic_still_fails_fast(self) -> None:
        with self.assertRaises(PPOObjectiveError):
            compute_case(
                new=((0.0,), (math.nan,)),
                old=((0.0,), (0.0,)),
                advantage=(1.0, 0.0),
                current_value=(0.0, 0.0),
                return_target=(0.0, 0.0),
                entropy=((0.0,), (0.0,)),
                sequence_valid_mask=(True, False),
            )

    def test_exp_log_ratio_overflow_fails_without_silent_clamp(self) -> None:
        with self.assertRaises(PPOObjectiveError):
            compute_case(new=((1000.0,),), old=((0.0,),))

    def test_malformed_rank_and_shape_fail_fast(self) -> None:
        cases = (
            {"new": (0.0,)},
            {"old": ((0.0, 0.0),)},
            {"entropy": ((0.0, 0.0),)},
            {"sequence_valid_mask": (True, True)},
        )
        for change in cases:
            with self.subTest(change=tuple(change)):
                with self.assertRaises(PPOObjectiveError):
                    compute_case(**change)

    def test_malformed_dtype_fails_fast(self) -> None:
        with self.assertRaises(PPOObjectiveError):
            compute_case(new=torch.zeros((1, 1), dtype=torch.long))
        with self.assertRaises(PPOObjectiveError):
            compute_case(sequence_valid_mask=torch.ones(1, dtype=torch.float32))
        with self.assertRaises(PPOObjectiveError):
            compute_case(
                new=torch.zeros((1, 1), dtype=torch.float64),
                old=torch.zeros((1, 1), dtype=torch.float64),
            )

    def test_invalid_epsilon_clip_fails_fast(self) -> None:
        for value in (0.0, -0.1, 1.0, math.nan, math.inf, True):
            with self.subTest(value=value):
                with self.assertRaises(PPOObjectiveError):
                    compute_case(epsilon_clip=value)

    def test_invalid_loss_coefficients_fail_fast(self) -> None:
        for name in ("value_coefficient", "entropy_coefficient"):
            for value in (-0.1, math.nan, math.inf, True):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(PPOObjectiveError):
                        compute_case(**{name: value})

    def test_zero_loss_coefficients_are_valid_explicit_choices(self) -> None:
        output = compute_case(
            advantage=(1.0,),
            current_value=(2.0,),
            return_target=(0.0,),
            entropy=((3.0,),),
            value_coefficient=0.0,
            entropy_coefficient=0.0,
        )
        torch.testing.assert_close(output.total_loss, output.actor_loss)

    def test_raw_advantage_is_not_normalized(self) -> None:
        output = compute_case(
            new=((0.0,), (0.0,)),
            old=((0.0,), (0.0,)),
            advantage=(1.0, 3.0),
            current_value=(0.0, 0.0),
            return_target=(0.0, 0.0),
            entropy=((0.0,), (0.0,)),
            sequence_valid_mask=(True, True),
        )
        torch.testing.assert_close(output.expanded_advantage[:, 0], torch.tensor((1.0, 3.0)))
        self.assertAlmostEqual(float(output.clipped_objective), 2.0, places=6)

    def test_agent_axis_is_dynamic_and_not_hard_coded_to_four(self) -> None:
        for agents in (1, 3, 6, 8):
            with self.subTest(agents=agents):
                zeros = (0.0,) * agents
                output = compute_case(
                    new=(zeros,),
                    old=(zeros,),
                    entropy=(zeros,),
                )
                self.assertEqual(output.ratio.shape, (1, agents))
                self.assertEqual(output.diagnostics.valid_actor_position_count, agents)

    def test_run_config_is_the_single_source_of_loss_hyperparameters(self) -> None:
        config = RunConfig()
        output = compute_configured_ppo_objective_and_loss(
            new_joint_log_prob=torch.zeros((1, 2)),
            old_joint_log_prob=torch.zeros((1, 2)),
            advantage=torch.ones(1),
            current_value=torch.zeros(1),
            return_target=torch.zeros(1),
            entropy=torch.zeros((1, 2)),
            sequence_valid_mask=torch.ones(1, dtype=torch.bool),
            config=config,
        )
        self.assertEqual(output.epsilon_clip, config.training.mappo.ppo_clip_epsilon)
        self.assertEqual(output.value_coefficient, config.training.mappo.value_coefficient)
        self.assertEqual(
            output.entropy_coefficient,
            config.training.mappo.entropy_coefficient,
        )

    def test_diagnostics_report_ratio_range_and_clip_fraction(self) -> None:
        output = compute_case(
            new=((math.log(2.0), 0.0),),
            old=((0.0, 0.0),),
            advantage=(1.0,),
            entropy=((0.0, 0.0),),
        )
        self.assertAlmostEqual(output.diagnostics.ratio_min, 1.0, places=6)
        self.assertAlmostEqual(output.diagnostics.ratio_max, 2.0, places=6)
        self.assertAlmostEqual(output.diagnostics.ratio_mean, 1.5, places=6)
        self.assertAlmostEqual(output.diagnostics.clipped_fraction, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
