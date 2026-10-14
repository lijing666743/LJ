"""Rollout Buffer Gate for recurrent CA-GAT-MAPPO storage semantics."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, replace

import torch

from src.config import AgentCreditMode, RunConfig
from src.env.environment import StepResult, U2UMECEnvironment
from src.models.ca_gat_mappo import (
    ACTION_BRANCH_ORDER,
    ActorObservationTensorizer,
    CAGATMAPPOActor,
    CentralizedStateTensorizer,
    MAPPOCentralizedCritic,
)
from src.models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionDistributionOutput,
    SequentialActionMaskBatch,
)
from src.models.ca_gat_mappo_rollout import (
    CAGATMAPPORolloutBuffer,
    CAGATMAPPORolloutTransition,
    RolloutStorageError,
)
from src.models.ca_gat_mappo_update import build_recurrent_ppo_minibatch


def make_rollout_config(
    *,
    uav_count: int = 4,
    horizon: int = 4,
    arrival_probability: float = 1.0,
    agent_credit_mode: AgentCreditMode = AgentCreditMode.TEAM,
) -> RunConfig:
    base = RunConfig()
    environment = replace(
        base.environment,
        episode_horizon=horizon,
        uav_count=uav_count,
        arrival_probabilities=(arrival_probability,) * uav_count,
        profile_assignment=("Balanced",) * uav_count,
        profile_perturbations=((0.0, 0.0, 0.0, 0.0),) * uav_count,
        building_layout=(),
        candidate_neighbor_radius_m=2_000.0,
        velocity_std_mps=(0.0, 0.0),
        shadowing_std_db=0.0,
        csi_error_std_db=0.0,
        fixed_csi_aoi_slots=1,
    )
    mappo = replace(base.training.mappo, agent_credit_mode=agent_credit_mode)
    training = replace(base.training, mappo=mappo)
    config = replace(base, environment=environment, training=training)
    config.validate()
    return config


def force_remote_and_active_communication(actor: CAGATMAPPOActor) -> None:
    """Make stochastic fixture proposals effectively deterministic and active."""

    preferred = {
        "route": 3,
        "tx_select": 1,
        "resource_group": 1,
        "resource_width": 0,
        "power_level": 3,
        "cpu_queue": 0,
        "cpu_frequency": 0,
    }
    with torch.no_grad():
        for branch, head in actor.action_heads.items():
            head.weight.zero_()
            head.bias.fill_(-100.0)
            head.bias[preferred[branch]] = 100.0


@dataclass
class CollectedStep:
    transition: CAGATMAPPORolloutTransition
    hidden_before: torch.Tensor
    output: SequentialActionDistributionOutput
    step: StepResult


class RolloutCollectorFixture:
    def __init__(self, config: RunConfig, *, force_active: bool = False) -> None:
        self.config = config
        self.environment = U2UMECEnvironment(config)
        reset = self.environment.reset()
        self.observations = reset.observations
        self.state = reset.centralized_state
        self.actor_tensorizer = ActorObservationTensorizer(config)
        self.critic_tensorizer = CentralizedStateTensorizer(config)
        self.actor = CAGATMAPPOActor(config).eval()
        if force_active:
            force_remote_and_active_communication(self.actor)
        self.critic = MAPPOCentralizedCritic(config).eval()
        self.distribution = CAGATMAPPOActionDistribution(self.actor, config)
        self.hidden = self.actor.initial_hidden(1)
        self.episode_start = True

    def collect(self) -> CollectedStep:
        actor_batch = self.actor_tensorizer.encode_step(
            self.observations,
            episode_start=self.episode_start,
        )
        mask_batch = SequentialActionMaskBatch.from_observations(self.observations)
        critic_batch = self.critic_tensorizer.encode_step(self.state)
        hidden_before = self.hidden.detach().clone()
        with torch.no_grad():
            output = self.distribution.sample_actions(
                actor_batch,
                mask_batch,
                hidden_before,
            )
            raw_old_value = self.critic(critic_batch)
            credit_mode = AgentCreditMode(
                self.config.training.mappo.agent_credit_mode
            )
            old_value = (
                raw_old_value[0, 0, 0]
                if credit_mode is AgentCreditMode.TEAM
                else raw_old_value[0, 0]
            )
        slot = self.observations[0].slot
        step = self.environment.step(output.proposals[0][0])
        bootstrap_value = None
        if bool(step.info["bootstrap_allowed"]):
            assert step.centralized_state is not None
            next_critic_batch = self.critic_tensorizer.encode_step(step.centralized_state)
            with torch.no_grad():
                raw_bootstrap = self.critic(next_critic_batch)
                bootstrap_value = (
                    raw_bootstrap[0, 0, 0]
                    if credit_mode is AgentCreditMode.TEAM
                    else raw_bootstrap[0, 0]
                )
        transition = CAGATMAPPORolloutTransition.from_step(
            spec=self.actor.spec,
            slot=slot,
            actor_batch=actor_batch,
            action_mask_batch=mask_batch,
            action_output=output,
            hidden_in=hidden_before,
            centralized_state=critic_batch,
            old_value=old_value,
            reward=(
                step.reward
                if credit_mode is AgentCreditMode.TEAM
                else torch.tensor(step.info["agent_reward"], dtype=torch.float32)
            ),
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
        if not transition.episode_boundary:
            assert step.observations is not None
            assert step.centralized_state is not None
            self.observations = step.observations
            self.state = step.centralized_state
            self.hidden = output.hidden_out.detach().clone()
            self.episode_start = False
        return CollectedStep(transition, hidden_before, output, step)


class CAGATMAPPORolloutBufferTests(unittest.TestCase):
    def test_append_finalize_and_time_agent_shapes_are_correct(self) -> None:
        config = make_rollout_config()
        collector = RolloutCollectorFixture(config)
        collected = [collector.collect(), collector.collect()]
        buffer = CAGATMAPPORolloutBuffer(config, capacity=4)
        for item in collected:
            buffer.append(item.transition)

        self.assertEqual(len(buffer), 2)
        self.assertFalse(buffer.full)
        chunk = buffer.finalize()
        agents = config.environment.uav_count
        self.assertTrue(buffer.finalized)
        self.assertEqual(chunk.length, 2)
        self.assertEqual(chunk.action_indices.shape, (2, agents, 7))
        self.assertEqual(chunk.active_branch_indicators.shape, (2, agents, 7))
        self.assertEqual(chunk.old_branch_log_probs.shape, (2, agents, 7))
        self.assertEqual(chunk.old_joint_log_prob.shape, (2, agents))
        self.assertEqual(
            chunk.hidden_in.shape,
            (2, agents, collector.actor.spec.gru_hidden_dimension),
        )
        self.assertEqual(chunk.old_value.shape, (2,))
        self.assertEqual(chunk.reward.shape, (2,))
        self.assertEqual(chunk.actor_batch.self_features.shape[:3], (1, 2, agents))
        self.assertEqual(chunk.centralized_batch.features.shape[:2], (1, 2))
        self.assertEqual(chunk.action_indices.dtype, torch.long)
        self.assertEqual(chunk.active_branch_indicators.dtype, torch.bool)
        self.assertEqual(chunk.hidden_in.dtype, torch.float32)
        self.assertEqual(chunk.hidden_in.device.type, "cpu")

    def test_seven_branch_proposal_indices_masks_and_old_log_probs_are_exact(self) -> None:
        config = make_rollout_config()
        collector = RolloutCollectorFixture(config)
        collector.collect()
        transition = collector.collect().transition

        self.assertTrue(torch.any(transition.active_branch_indicators[:, 0]))
        self.assertTrue(
            torch.allclose(
                transition.old_joint_log_prob,
                transition.old_branch_log_probs.sum(dim=-1),
            )
        )
        for agent, (contract, proposal) in enumerate(
            zip(transition.action_mask_contracts, transition.proposal_actions)
        ):
            self.assertTrue(contract.is_legal(proposal))
            self.assertEqual(len(proposal.branches), 7)
            for branch_index, branch in enumerate(ACTION_BRANCH_ORDER):
                index = int(transition.action_indices[agent, branch_index])
                self.assertEqual(
                    contract.domain_for(branch)[index],
                    getattr(proposal, branch),
                )
                expected_mask = torch.tensor(
                    contract.mask_for(branch, proposal).copy(), dtype=torch.bool
                )
                self.assertTrue(
                    torch.equal(transition.branch_masks[branch][agent], expected_mask)
                )

    def test_real_executor_rejection_does_not_overwrite_proposal_action(self) -> None:
        config = make_rollout_config(horizon=5)
        collector = RolloutCollectorFixture(config, force_active=True)
        collector.collect()
        collector.collect()
        collected = collector.collect()
        transition = collected.transition

        rejected_agent = next(
            agent
            for agent, summary in enumerate(transition.executed_action_summary)
            if transition.proposal_actions[agent].power_level == 1.0
            and not bool(summary["communication"]["accepted"])
        )
        proposal = transition.proposal_actions[rejected_agent]
        executed = transition.executed_action_summary[rejected_agent]
        self.assertEqual(proposal.power_level, 1.0)
        self.assertEqual(executed["communication"]["executed_power_w"], 0.0)
        self.assertTrue(
            torch.allclose(
                transition.old_joint_log_prob,
                collected.output.joint_log_prob[0, 0].cpu(),
            )
        )

        collected.step.info["executed"][rejected_agent]["communication"][
            "executed_power_w"
        ] = 123.0
        self.assertEqual(
            transition.executed_action_summary[rejected_agent]["communication"][
                "executed_power_w"
            ],
            0.0,
        )
        self.assertEqual(transition.proposal_actions[rejected_agent], proposal)

    def test_hidden_in_is_saved_exactly_and_chunk_uses_start_position(self) -> None:
        config = make_rollout_config(horizon=5)
        collector = RolloutCollectorFixture(config)
        collected = [collector.collect(), collector.collect(), collector.collect()]
        buffer = CAGATMAPPORolloutBuffer(config, capacity=3)
        for item in collected:
            buffer.append(item.transition)
            self.assertTrue(
                torch.equal(item.transition.hidden_in, item.hidden_before[0])
            )
        buffer.finalize()

        chunk = buffer.get_sequence(1, 2)
        self.assertTrue(
            torch.equal(chunk.initial_hidden[0], collected[1].transition.hidden_in)
        )
        self.assertFalse(bool(chunk.episode_starts[0]))

    def test_episode_start_requires_exact_zero_hidden(self) -> None:
        config = make_rollout_config()
        transition = RolloutCollectorFixture(config).collect().transition
        self.assertTrue(transition.episode_start)
        self.assertEqual(torch.count_nonzero(transition.hidden_in).item(), 0)
        with self.assertRaises(RolloutStorageError):
            replace(transition, hidden_in=torch.ones_like(transition.hidden_in))
        with self.assertRaises(RolloutStorageError):
            replace(transition, episode_start=False)

    def test_middle_episode_boundary_resets_next_hidden_and_recurrent_mask(self) -> None:
        config = make_rollout_config(horizon=2)
        first_episode = RolloutCollectorFixture(config)
        first = first_episode.collect().transition
        boundary = first_episode.collect().transition
        second_start = RolloutCollectorFixture(config).collect().transition
        buffer = CAGATMAPPORolloutBuffer(config, capacity=3)
        for transition in (first, boundary, second_start):
            buffer.append(transition)
        chunk = buffer.finalize()

        self.assertEqual(chunk.episode_boundary.tolist(), [False, True, False])
        self.assertEqual(chunk.episode_starts.tolist(), [True, False, True])
        self.assertEqual(chunk.truncated.tolist(), [False, True, False])
        self.assertEqual(chunk.terminated.tolist(), [False, False, False])
        self.assertEqual(torch.count_nonzero(chunk.hidden_in[2]).item(), 0)
        self.assertTrue(torch.all(chunk.actor_batch.episode_starts[0, 2]))
        restarted = buffer.get_sequence(2, 1)
        self.assertEqual(torch.count_nonzero(restarted.initial_hidden).item(), 0)

    def test_terminated_and_truncated_are_distinct_flags(self) -> None:
        config = make_rollout_config(horizon=1)
        truncated = RolloutCollectorFixture(config).collect().transition
        self.assertFalse(truncated.terminated)
        self.assertTrue(truncated.truncated)
        terminated = replace(truncated, terminated=True, truncated=False)
        self.assertTrue(terminated.terminated)
        self.assertFalse(terminated.truncated)
        with self.assertRaises(RolloutStorageError):
            replace(truncated, terminated=True, truncated=True)

    def test_horizon_truncation_is_no_bootstrap_and_has_no_fake_next_value(self) -> None:
        config = make_rollout_config(horizon=2)
        collector = RolloutCollectorFixture(config)
        non_boundary = collector.collect().transition
        boundary = collector.collect().transition

        self.assertTrue(non_boundary.bootstrap_allowed)
        self.assertIsNotNone(non_boundary.bootstrap_value)
        self.assertFalse(boundary.terminated)
        self.assertTrue(boundary.truncated)
        self.assertTrue(boundary.episode_boundary)
        self.assertFalse(boundary.bootstrap_allowed)
        self.assertIsNone(boundary.bootstrap_value)
        with self.assertRaises(RolloutStorageError):
            replace(boundary, bootstrap_value=torch.tensor(1.0))

    def test_role_terminal_in_second_256_chunk_preserves_agent_bootstrap_axis(self) -> None:
        config = make_rollout_config(
            horizon=500,
            agent_credit_mode=AgentCreditMode.ROLE_DECOMPOSED,
        )
        first_episode = RolloutCollectorFixture(config)
        first_transitions = [
            first_episode.collect().transition
            for _ in range(config.environment.episode_horizon)
        ]
        next_episode = RolloutCollectorFixture(config)
        transitions = first_transitions[256:] + [
            next_episode.collect().transition for _ in range(12)
        ]

        buffer = CAGATMAPPORolloutBuffer(config, capacity=256)
        for transition in transitions:
            buffer.append(transition)
        chunk = buffer.finalize()

        agents = config.environment.uav_count
        self.assertEqual(chunk.length, 256)
        self.assertTrue(bool(chunk.episode_boundary[243]))
        self.assertTrue(bool(chunk.episode_starts[244]))
        self.assertEqual(tuple(chunk.transitions[243].reward.shape), (agents,))
        self.assertEqual(tuple(chunk.transitions[243].old_value.shape), (agents,))
        self.assertIsNone(chunk.transitions[243].bootstrap_value)
        self.assertEqual(tuple(chunk.bootstrap_values.shape), (256, agents))
        self.assertTrue(torch.equal(chunk.bootstrap_values[243], torch.zeros(agents)))

        minibatch = build_recurrent_ppo_minibatch(buffer, config)
        self.assertEqual(tuple(minibatch.advantage.shape), (8, 32, agents))
        self.assertEqual(tuple(minibatch.return_target.shape), (8, 32, agents))

    def test_short_role_episode_boundary_keeps_terminal_vectors(self) -> None:
        config = make_rollout_config(
            horizon=5,
            agent_credit_mode=AgentCreditMode.ROLE_DECOMPOSED,
        )
        collector = RolloutCollectorFixture(config)
        transitions = [collector.collect().transition for _ in range(5)]
        buffer = CAGATMAPPORolloutBuffer(config, capacity=5)
        for transition in transitions:
            buffer.append(transition)
        chunk = buffer.finalize()

        agents = config.environment.uav_count
        self.assertEqual(tuple(chunk.reward.shape), (5, agents))
        self.assertEqual(tuple(chunk.old_value.shape), (5, agents))
        self.assertEqual(tuple(chunk.bootstrap_values.shape), (5, agents))
        self.assertTrue(torch.equal(chunk.bootstrap_values[-1], torch.zeros(agents)))

    def test_team_terminal_boundary_keeps_legacy_scalar_contract(self) -> None:
        config = make_rollout_config(horizon=5, agent_credit_mode=AgentCreditMode.TEAM)
        collector = RolloutCollectorFixture(config)
        transitions = [collector.collect().transition for _ in range(5)]
        buffer = CAGATMAPPORolloutBuffer(config, capacity=5)
        for transition in transitions:
            buffer.append(transition)
        chunk = buffer.finalize()

        self.assertEqual(tuple(chunk.reward.shape), (5,))
        self.assertEqual(tuple(chunk.old_value.shape), (5,))
        self.assertEqual(tuple(chunk.bootstrap_values.shape), (5,))
        self.assertEqual(chunk.bootstrap_values[-1].shape, torch.Size([]))

    def test_non_boundary_requires_finite_bootstrap_value(self) -> None:
        config = make_rollout_config(horizon=2)
        transition = RolloutCollectorFixture(config).collect().transition
        with self.assertRaises(RolloutStorageError):
            replace(transition, bootstrap_value=None)
        with self.assertRaises(RolloutStorageError):
            replace(transition, bootstrap_value=torch.tensor(float("nan")))

    def test_chunk_can_re_evaluate_stored_proposals_with_identical_statistics(self) -> None:
        config = make_rollout_config(horizon=4)
        collector = RolloutCollectorFixture(config)
        transitions = [collector.collect().transition, collector.collect().transition]
        buffer = CAGATMAPPORolloutBuffer(config, capacity=2)
        for transition in transitions:
            buffer.append(transition)
        chunk = buffer.finalize()

        with torch.no_grad():
            evaluated = collector.distribution.evaluate_actions(
                chunk.actor_batch,
                chunk.action_mask_batch,
                chunk.proposals,
                chunk.initial_hidden,
            )
        self.assertTrue(
            torch.allclose(evaluated.joint_log_prob[0], chunk.old_joint_log_prob)
        )
        for branch_index, branch in enumerate(ACTION_BRANCH_ORDER):
            self.assertTrue(
                torch.equal(evaluated.action_masks[branch][0], chunk.branch_masks[branch])
            )
            self.assertTrue(
                torch.equal(
                    evaluated.action_indices[branch][0],
                    chunk.action_indices[..., branch_index],
                )
            )
            self.assertTrue(
                torch.allclose(
                    evaluated.branch_log_probs[branch][0],
                    chunk.old_branch_log_probs[..., branch_index],
                )
            )

    def test_capacity_overflow_and_append_after_finalize_fail_fast(self) -> None:
        config = make_rollout_config()
        transition = RolloutCollectorFixture(config).collect().transition
        buffer = CAGATMAPPORolloutBuffer(config, capacity=1)
        buffer.append(transition)
        self.assertTrue(buffer.full)
        with self.assertRaises(RolloutStorageError):
            buffer.append(transition)
        buffer.finalize()
        with self.assertRaises(RolloutStorageError):
            buffer.append(transition)

    def test_unwritten_positions_and_unfinalized_sequences_cannot_be_read(self) -> None:
        config = make_rollout_config()
        transition = RolloutCollectorFixture(config).collect().transition
        buffer = CAGATMAPPORolloutBuffer(config, capacity=3)
        buffer.append(transition)
        self.assertEqual(buffer.transition_at(0).slot, 0)
        with self.assertRaises(RolloutStorageError):
            buffer.transition_at(1)
        with self.assertRaises(RolloutStorageError):
            buffer.get_sequence(0, 1)
        buffer.finalize()
        with self.assertRaises(RolloutStorageError):
            buffer.get_sequence(0, 2)

    def test_malformed_shapes_and_dtypes_fail_fast(self) -> None:
        config = make_rollout_config()
        transition = RolloutCollectorFixture(config).collect().transition
        malformed = (
            {"hidden_in": transition.hidden_in[:-1]},
            {"action_indices": transition.action_indices.to(torch.float32)},
            {
                "active_branch_indicators": transition.active_branch_indicators.to(
                    torch.int64
                )
            },
            {"neighbor_mask": transition.neighbor_mask.to(torch.float32)},
        )
        for change in malformed:
            with self.subTest(change=tuple(change)):
                with self.assertRaises(RolloutStorageError):
                    replace(transition, **change)

    def test_nan_and_inf_statistics_are_rejected(self) -> None:
        config = make_rollout_config()
        transition = RolloutCollectorFixture(config).collect().transition
        invalid = (
            {"reward": torch.tensor(float("nan"))},
            {"old_value": torch.tensor(float("inf"))},
            {
                "old_branch_log_probs": torch.full_like(
                    transition.old_branch_log_probs, float("nan")
                )
            },
            {
                "old_joint_log_prob": torch.full_like(
                    transition.old_joint_log_prob, float("inf")
                )
            },
        )
        for change in invalid:
            with self.subTest(change=tuple(change)):
                with self.assertRaises(RolloutStorageError):
                    replace(transition, **change)

    def test_uav_count_is_configured_instead_of_hard_coded(self) -> None:
        for uav_count in (3, 5):
            with self.subTest(uav_count=uav_count):
                config = make_rollout_config(
                    uav_count=uav_count,
                    arrival_probability=0.0,
                )
                transition = RolloutCollectorFixture(config).collect().transition
                buffer = CAGATMAPPORolloutBuffer(config, capacity=1)
                buffer.append(transition)
                chunk = buffer.finalize()
                self.assertEqual(chunk.action_indices.shape, (1, uav_count, 7))
                self.assertEqual(chunk.hidden_in.shape[1], uav_count)
                self.assertEqual(
                    chunk.actor_batch.neighbor_mask.shape,
                    (1, 1, uav_count, uav_count),
                )

    def test_clear_resets_length_full_and_finalize_state(self) -> None:
        config = make_rollout_config()
        transition = RolloutCollectorFixture(config).collect().transition
        buffer = CAGATMAPPORolloutBuffer(config, capacity=1)
        buffer.append(transition)
        buffer.finalize()
        buffer.clear()
        self.assertEqual(len(buffer), 0)
        self.assertFalse(buffer.full)
        self.assertFalse(buffer.finalized)
        with self.assertRaises(RolloutStorageError):
            buffer.finalize()
        buffer.append(transition)
        self.assertEqual(len(buffer), 1)

    def test_diagnostic_metadata_is_separate_from_actor_and_critic_inputs(self) -> None:
        config = make_rollout_config()
        collected = RolloutCollectorFixture(config).collect()
        transition = collected.transition
        buffer = CAGATMAPPORolloutBuffer(config, capacity=1)
        buffer.append(transition)
        returned = buffer.transition_at(0)
        original_power = returned.executed_action_summary[0]["communication"][
            "executed_power_w"
        ]
        returned.executed_action_summary[0]["communication"][
            "executed_power_w"
        ] = 999.0
        self.assertEqual(
            buffer.transition_at(0).executed_action_summary[0]["communication"][
                "executed_power_w"
            ],
            original_power,
        )
        chunk = buffer.finalize()
        actor_fields = set(chunk.actor_batch.__dataclass_fields__)
        critic_fields = set(chunk.centralized_batch.__dataclass_fields__)
        self.assertNotIn("executed_action_summary", actor_fields)
        self.assertNotIn("executed_action_summary", critic_fields)
        self.assertEqual(len(chunk.executed_action_summaries), 1)


if __name__ == "__main__":
    unittest.main()
