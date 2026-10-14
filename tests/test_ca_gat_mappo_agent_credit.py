"""Acceptance tests for conservation-preserving per-agent MAPPO credit."""

from __future__ import annotations

import math
import random
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.cli import RL_PROFILES
from src.config import (
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_SCHEMA_VERSION,
    ActorRatioMode,
    AgentCreditMode,
    ConfigError,
    RunConfig,
    load_run_config,
    validate_mappo_checkpoint_resume_compatibility,
)
from src.env.actions import ActionProposal
from src.env.environment import StepResult, U2UMECEnvironment
from src.env.reward import (
    AGENT_REWARD_CONSERVATION_TOLERANCE,
    RewardCalculator,
    RewardError,
    RewardReferences,
    RewardWeights,
)
from src.env.tasks import Task, TaskStatus
from src.models.ca_gat_mappo import (
    CAGATMAPPOActor,
    MAPPONetworkError,
    MAPPOCentralizedCritic,
)
from src.models.ca_gat_mappo_gae import (
    GAEComputationError,
    compute_per_agent_gae_and_returns,
)
from src.models.ca_gat_mappo_ppo import (
    PPOObjectiveError,
    compute_ppo_objective_and_loss,
)
from src.models.ca_gat_mappo_rollout import (
    CAGATMAPPORolloutBuffer,
    RolloutStorageError,
)
from src.models.ca_gat_mappo_trainer import (
    CAGATMAPPOTrainer,
    CAGATMAPPOTrainerError,
)
from src.models.ca_gat_mappo_update import (
    CAGATMAPPORecurrentPPOUpdater,
    build_recurrent_ppo_minibatch,
)
from tests.test_ca_gat_mappo_update import (
    align_old_policy_snapshots,
    make_seed_transition,
    make_synthetic_buffer,
    make_update_config,
)


def credit_config(
    mode: AgentCreditMode,
    *,
    actor_ratio_mode: ActorRatioMode = ActorRatioMode.BRANCH_SPECIFIC,
) -> RunConfig:
    base = make_update_config()
    mappo = replace(
        base.training.mappo,
        actor_ratio_mode=actor_ratio_mode,
        agent_credit_mode=mode,
        training_device="cpu",
    )
    config = replace(base, training=replace(base.training, mappo=mappo))
    config.validate()
    return config


def idle_proposals(count: int) -> tuple[ActionProposal, ...]:
    return tuple(
        ActionProposal(
            uav_id=uav_id,
            route="idle",
            tx_select="idle",
            resource_group="idle",
            resource_width=1,
            power_level=0.0,
            cpu_queue="idle",
            cpu_frequency=0.0,
        )
        for uav_id in range(count)
    )


def reward_fixture() -> tuple[
    RewardCalculator,
    tuple[Task, ...],
    tuple[Task, ...],
]:
    local_done = Task(
        task_id=0,
        source_uav=0,
        data_bits=100.0,
        cpu_cycles=200.0,
        arrival_slot=0,
        deadline_slot=5,
        remaining_bits=0.0,
        remaining_cycles=0.0,
        destination=0,
        status=TaskStatus.DONE,
        completion_slot=5,
    )
    remote_done = Task(
        task_id=1,
        source_uav=0,
        data_bits=200.0,
        cpu_cycles=200.0,
        arrival_slot=0,
        deadline_slot=5,
        remaining_bits=0.0,
        remaining_cycles=0.0,
        destination=1,
        status=TaskStatus.DONE,
        completion_slot=5,
    )
    remote_expired = Task(
        task_id=2,
        source_uav=2,
        data_bits=100.0,
        cpu_cycles=200.0,
        arrival_slot=0,
        deadline_slot=5,
        remaining_bits=50.0,
        remaining_cycles=100.0,
        destination=3,
        status=TaskStatus.EXPIRED,
    )
    active_remote = Task(
        task_id=3,
        source_uav=0,
        data_bits=600.0,
        cpu_cycles=800.0,
        arrival_slot=0,
        deadline_slot=10,
        remaining_bits=300.0,
        remaining_cycles=400.0,
        destination=1,
        status=TaskStatus.CPU,
    )
    active_unbound = Task(
        task_id=4,
        source_uav=2,
        data_bits=100.0,
        cpu_cycles=200.0,
        arrival_slot=0,
        deadline_slot=5,
        remaining_bits=100.0,
        remaining_cycles=200.0,
        status=TaskStatus.UNBOUND,
    )
    calculator = RewardCalculator(
        RewardReferences(
            reference_rate_bps=100.0,
            reference_cpu_frequency_hz=200.0,
            workload_reference_s=1.0,
            task_count_reference=1.0,
            active_energy_reference_j=10.0,
        ),
        RewardWeights(
            completion=1.0,
            expiration=1.0,
            workload=0.2,
            energy=0.05,
        ),
    )
    return (
        calculator,
        (
            local_done,
            remote_done,
            remote_expired,
            active_remote,
            active_unbound,
        ),
        (local_done, remote_done, remote_expired),
    )


def decompose_fixture():
    calculator, tasks, settled = reward_fixture()
    team = calculator.calculate(
        slot=5,
        all_tasks=tasks,
        settled_tasks=settled,
        actual_energy_j=3.75,
    )
    agent = calculator.decompose_agent_credit(
        slot=5,
        uav_count=4,
        all_tasks=tasks,
        settled_tasks=settled,
        team_terms=team,
        agent_transmit_energy_j={0: 1.0, 2: 0.5},
        agent_cpu_energy_j={1: 2.0, 3: 0.25},
    )
    return calculator, tasks, settled, team, agent


def role_rollout_fixture() -> tuple[RunConfig, CAGATMAPPORolloutBuffer]:
    team_config = credit_config(AgentCreditMode.TEAM)
    role_config = credit_config(AgentCreditMode.ROLE_DECOMPOSED)
    seed = make_seed_transition(team_config)
    scalar_buffer = make_synthetic_buffer(team_config, seed)
    agent_count = role_config.environment.uav_count
    reward_fractions = torch.tensor(
        [0.55, 0.25, 0.15, 0.05],
        dtype=torch.float32,
    )
    value_offsets = torch.linspace(
        -0.15,
        0.15,
        steps=agent_count,
        dtype=torch.float32,
    )
    role_buffer = CAGATMAPPORolloutBuffer(
        role_config,
        capacity=len(scalar_buffer),
    )
    for index in range(len(scalar_buffer)):
        transition = scalar_buffer.transition_at(index)
        bootstrap = (
            None
            if transition.bootstrap_value is None
            else transition.bootstrap_value.repeat(agent_count) + value_offsets
        )
        role_buffer.append(
            replace(
                transition,
                reward=transition.reward * reward_fractions,
                old_value=transition.old_value.repeat(agent_count) + value_offsets,
                bootstrap_value=bootstrap,
            )
        )
    role_buffer.finalize()
    return role_config, align_old_policy_snapshots(role_config, role_buffer)


class AgentCreditConfigurationTests(unittest.TestCase):
    def test_default_team_mode_is_hidden_from_canonical_config(self) -> None:
        team = RunConfig()
        role = credit_config(AgentCreditMode.ROLE_DECOMPOSED)
        self.assertEqual(team.training.mappo.agent_credit_mode, AgentCreditMode.TEAM)
        self.assertNotIn("agent_credit_mode", team.resolved_dict()["training"]["mappo"])
        self.assertEqual(
            role.resolved_dict()["training"]["mappo"]["agent_credit_mode"],
            AgentCreditMode.ROLE_DECOMPOSED.value,
        )
        self.assertNotEqual(team.config_hash, role.config_hash)

    def test_invalid_mode_fails_and_long_smoke_remains_legacy_team(self) -> None:
        base = RunConfig()
        invalid = replace(
            base,
            training=replace(
                base.training,
                mappo=replace(base.training.mappo, agent_credit_mode="invalid"),
            ),
        )
        with self.assertRaisesRegex(ConfigError, "agent_credit_mode"):
            invalid.validate()
        long_smoke = RL_PROFILES["rl-long-smoke"]
        self.assertEqual(
            long_smoke["training.mappo.actor_ratio_mode"],
            ActorRatioMode.BRANCH_SPECIFIC.value,
        )
        self.assertEqual(
            long_smoke["training.mappo.agent_credit_mode"],
            AgentCreditMode.TEAM.value,
        )
        self.assertEqual(
            long_smoke["environment.workload_timing_mode"],
            "legacy_post_route",
        )

    def test_checkpoint_resume_rejects_cross_mode_and_accepts_legacy_team(self) -> None:
        common = {
            "mode": "rl",
            "method_id": "ca_gat_mappo",
            "training.mappo.training_device": "cpu",
        }
        team = load_run_config(cli_overrides=common)
        role = load_run_config(
            cli_overrides={
                **common,
                "training.mappo.agent_credit_mode": "role_decomposed",
            }
        )

        def validate(config: RunConfig, source_mode: str | None) -> None:
            validate_mappo_checkpoint_resume_compatibility(
                config,
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                method_id=config.method_id,
                git_commit=config.git_commit,
                config_hash=config.config_hash,
                training_device="cpu",
                cuda_available=False,
                checkpoint_actor_ratio_mode=None,
                checkpoint_agent_credit_mode=source_mode,
            )

        validate(team, None)
        validate(role, AgentCreditMode.ROLE_DECOMPOSED.value)
        with self.assertRaisesRegex(ConfigError, "agent_credit_mode mismatch"):
            validate(team, AgentCreditMode.ROLE_DECOMPOSED.value)
        with self.assertRaisesRegex(ConfigError, "agent_credit_mode mismatch"):
            validate(role, None)


class AgentRewardConservationTests(unittest.TestCase):
    def test_zero_active_task_slot_is_an_exact_zero_vector(self) -> None:
        calculator, _tasks, _settled = reward_fixture()
        team = calculator.calculate(
            slot=5,
            all_tasks=(),
            settled_tasks=(),
            actual_energy_j=0.0,
        )
        agent = calculator.decompose_agent_credit(
            slot=5,
            uav_count=4,
            all_tasks=(),
            settled_tasks=(),
            team_terms=team,
            agent_transmit_energy_j={},
            agent_cpu_energy_j={},
        )
        self.assertEqual(team.reward, 0.0)
        self.assertEqual(agent.agent_reward, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(agent.urgent_workload_s, (0.0, 0.0, 0.0, 0.0))

    def test_all_local_slot_attributes_completion_energy_to_same_agent(self) -> None:
        calculator, _tasks, _settled = reward_fixture()
        task = Task(
            task_id=9,
            source_uav=3,
            data_bits=100.0,
            cpu_cycles=200.0,
            arrival_slot=0,
            deadline_slot=5,
            remaining_bits=0.0,
            remaining_cycles=0.0,
            destination=3,
            status=TaskStatus.DONE,
            completion_slot=5,
        )
        team = calculator.calculate(
            slot=5,
            all_tasks=(task,),
            settled_tasks=(task,),
            actual_energy_j=0.5,
        )
        agent = calculator.decompose_agent_credit(
            slot=5,
            uav_count=4,
            all_tasks=(task,),
            settled_tasks=(task,),
            team_terms=team,
            agent_transmit_energy_j={},
            agent_cpu_energy_j={3: 0.5},
        )
        self.assertEqual(agent.completion_credit, (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(agent.actual_energy_j, (0.0, 0.0, 0.0, 0.5))
        self.assertEqual(agent.agent_reward[:3], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(agent.agent_reward[3], team.reward)

    def test_local_and_remote_completion_credit_use_intrinsic_work(self) -> None:
        _calculator, _tasks, _settled, team, agent = decompose_fixture()
        expected_eta = (200.0 / 100.0) / (
            200.0 / 100.0 + 200.0 / 200.0
        )
        self.assertEqual(team.completed_task_count, 2)
        self.assertAlmostEqual(agent.completion_credit[0], 1.0 + expected_eta)
        self.assertAlmostEqual(agent.completion_credit[1], 1.0 - expected_eta)
        self.assertEqual(agent.completion_credit[2:], (0.0, 0.0))
        self.assertEqual(len(agent.remote_completions), 1)
        remote = agent.remote_completions[0]
        self.assertAlmostEqual(remote.source_share, expected_eta)
        self.assertAlmostEqual(remote.destination_share, 1.0 - expected_eta)

    def test_expiration_and_workload_follow_source_and_cpu_owner(self) -> None:
        _calculator, _tasks, _settled, team, agent = decompose_fixture()
        self.assertEqual(agent.expiration_count, (0.0, 0.0, 1.0, 0.0))
        self.assertEqual(agent.communication_workload_s, (0.5, 0.0, 1.0, 0.0))
        self.assertAlmostEqual(agent.computation_workload_s[0], 0.0)
        self.assertAlmostEqual(agent.computation_workload_s[1], 1.0 / 3.0)
        self.assertAlmostEqual(agent.computation_workload_s[2], 1.0)
        self.assertAlmostEqual(agent.computation_workload_s[3], 0.0)
        # Task 4 is destination=None, so both its bit and cycle burden remain
        # with source UAV 2.
        self.assertAlmostEqual(agent.communication_workload_s[2], 1.0)
        self.assertAlmostEqual(agent.computation_workload_s[2], 1.0)
        self.assertAlmostEqual(math.fsum(agent.urgent_workload_s), team.urgent_workload_s)

    def test_actual_energy_is_charged_to_sender_and_executor(self) -> None:
        _calculator, _tasks, _settled, team, agent = decompose_fixture()
        self.assertEqual(agent.transmit_energy_j, (1.0, 0.0, 0.5, 0.0))
        self.assertEqual(agent.cpu_energy_j, (0.0, 2.0, 0.0, 0.25))
        self.assertEqual(agent.actual_energy_j, (1.0, 2.0, 0.5, 0.25))
        self.assertAlmostEqual(math.fsum(agent.actual_energy_j), team.actual_energy_j)

    def test_every_component_and_final_reward_conserve_exactly(self) -> None:
        _calculator, _tasks, _settled, team, agent = decompose_fixture()
        self.assertAlmostEqual(
            math.fsum(agent.completion_component),
            team.completion_component,
        )
        self.assertAlmostEqual(
            math.fsum(agent.expiration_penalty),
            team.expiration_penalty,
        )
        self.assertAlmostEqual(
            math.fsum(agent.workload_penalty),
            team.workload_penalty,
        )
        self.assertAlmostEqual(
            math.fsum(agent.energy_penalty),
            team.energy_penalty,
        )
        self.assertAlmostEqual(math.fsum(agent.agent_reward), team.reward)
        self.assertLessEqual(
            abs(agent.conservation_residual),
            AGENT_REWARD_CONSERVATION_TOLERANCE,
        )
        telemetry = agent.to_dict()
        self.assertEqual(len(telemetry["destination"]), 4)
        self.assertEqual(
            telemetry["destination"][1]["cpu_energy_j"],
            2.0,
        )

    def test_nonconservative_inputs_fail_fast(self) -> None:
        calculator, tasks, settled, team, _agent = decompose_fixture()
        fabricated = replace(team, actual_energy_j=team.actual_energy_j + 1.0)
        with self.assertRaisesRegex(RewardError, "energy attribution"):
            calculator.decompose_agent_credit(
                slot=5,
                uav_count=4,
                all_tasks=tasks,
                settled_tasks=settled,
                team_terms=fabricated,
                agent_transmit_energy_j={0: 1.0, 2: 0.5},
                agent_cpu_energy_j={1: 2.0, 3: 0.25},
            )

    def test_decomposition_does_not_advance_global_rng_state(self) -> None:
        calculator, tasks, settled, team, _agent = decompose_fixture()
        random.seed(917)
        np.random.seed(917)
        torch.manual_seed(917)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        cuda_before = (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else None
        )
        calculator.decompose_agent_credit(
            slot=5,
            uav_count=4,
            all_tasks=tasks,
            settled_tasks=settled,
            team_terms=team,
            agent_transmit_energy_j={0: 1.0, 2: 0.5},
            agent_cpu_energy_j={1: 2.0, 3: 0.25},
        )
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        if cuda_before is not None:
            self.assertEqual(len(torch.cuda.get_rng_state_all()), len(cuda_before))
            for before, after in zip(cuda_before, torch.cuda.get_rng_state_all()):
                self.assertTrue(torch.equal(before, after))


class AgentCreditEnvironmentAPITests(unittest.TestCase):
    def test_step_reward_stays_scalar_and_role_telemetry_is_additive(self) -> None:
        team_config = credit_config(AgentCreditMode.TEAM)
        role_config = credit_config(AgentCreditMode.ROLE_DECOMPOSED)
        zero_arrivals = (0.0,) * team_config.environment.uav_count
        team_config = replace(
            team_config,
            environment=replace(
                team_config.environment,
                arrival_probabilities=zero_arrivals,
            ),
        )
        role_config = replace(
            role_config,
            environment=replace(
                role_config.environment,
                arrival_probabilities=zero_arrivals,
            ),
        )
        team_config.validate()
        role_config.validate()
        team_environment = U2UMECEnvironment(team_config)
        role_environment = U2UMECEnvironment(role_config)
        team_environment.reset()
        role_environment.reset()
        proposals = idle_proposals(team_config.environment.uav_count)
        for _ in range(4):
            team_step = team_environment.step(proposals)
            role_step = role_environment.step(proposals)
            self.assertIsInstance(team_step.reward, float)
            self.assertIsInstance(role_step.reward, float)
            self.assertEqual(team_step.reward, role_step.reward)
            self.assertNotIn("agent_reward", team_step.info)
            self.assertNotIn("agent_credit", team_step.info)
            self.assertEqual(len(role_step.info["agent_reward"]), 4)
            self.assertAlmostEqual(
                math.fsum(role_step.info["agent_reward"]),
                role_step.reward,
            )
            self.assertLessEqual(
                abs(role_step.info["agent_credit"]["conservation_residual"]),
                AGENT_REWARD_CONSERVATION_TOLERANCE,
            )
            self.assertEqual(team_step.info["arrival"], role_step.info["arrival"])
            self.assertEqual(team_step.info["service"], role_step.info["service"])
            self.assertEqual(team_step.info["energy"], role_step.info["energy"])


class AgentCreditLearningPathTests(unittest.TestCase):
    def test_vector_critic_shape_and_head_specific_gradients(self) -> None:
        config = credit_config(AgentCreditMode.ROLE_DECOMPOSED)
        critic = MAPPOCentralizedCritic(config)
        feature_count = critic.spec.centralized_state_dim
        features = torch.linspace(
            -1.0,
            1.0,
            steps=2 * 3 * feature_count,
            dtype=torch.float32,
        ).reshape(2, 3, feature_count)
        values = critic(features)
        self.assertEqual(tuple(values.shape), (2, 3, 4))
        targets = torch.arange(1, 5, dtype=torch.float32).reshape(1, 1, 4)
        (values - targets).square().mean().backward()
        head_gradient = critic.value_network[-1].weight.grad
        self.assertIsNotNone(head_gradient)
        assert head_gradient is not None
        self.assertTrue(torch.all(head_gradient.norm(dim=1) > 0.0))
        self.assertFalse(torch.allclose(head_gradient[0], head_gradient[1]))

    def test_critic_head_count_mismatch_fails_fast(self) -> None:
        config = credit_config(AgentCreditMode.ROLE_DECOMPOSED)
        critic = MAPPOCentralizedCritic(config)
        hidden = config.training.mappo.encoder_hidden_dimension
        critic.value_network[-1] = torch.nn.Linear(hidden, 3)
        features = torch.zeros((1, 1, critic.spec.centralized_state_dim))
        with self.assertRaisesRegex(MAPPONetworkError, "head count"):
            critic(features)

    def test_per_agent_gae_matches_manual_recurrence(self) -> None:
        reward = torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        )
        output = compute_per_agent_gae_and_returns(
            reward=reward,
            old_value=torch.zeros_like(reward),
            bootstrap_value=torch.zeros_like(reward),
            terminated=torch.tensor([False, False, True]),
            truncated=torch.tensor([False, False, False]),
            episode_boundary=torch.tensor([False, False, True]),
            bootstrap_allowed=torch.tensor([True, True, False]),
            gamma=0.999,
            gae_lambda=1.0,
        )
        expected = torch.stack(
            (
                reward[0] + 0.999 * reward[1] + 0.999**2 * reward[2],
                reward[1] + 0.999 * reward[2],
                reward[2],
            )
        )
        self.assertEqual(tuple(output.advantage.shape), (3, 2))
        self.assertTrue(torch.allclose(output.td_residual, reward))
        self.assertTrue(torch.allclose(output.advantage, expected))
        self.assertTrue(torch.allclose(output.return_target, expected))

    def test_per_agent_gae_rejects_scalar_mismatch_and_nonfinite(self) -> None:
        common = {
            "terminated": torch.tensor([False, True]),
            "truncated": torch.tensor([False, False]),
            "episode_boundary": torch.tensor([False, True]),
            "bootstrap_allowed": torch.tensor([True, False]),
            "gamma": 0.99,
            "gae_lambda": 0.95,
        }
        with self.assertRaisesRegex(GAEComputationError, r"\[T,A\]"):
            compute_per_agent_gae_and_returns(
                reward=torch.ones(2),
                old_value=torch.ones((2, 2)),
                bootstrap_value=torch.ones((2, 2)),
                **common,
            )
        bad = torch.ones((2, 2))
        bad[0, 0] = float("nan")
        with self.assertRaisesRegex(GAEComputationError, "NaN or Inf"):
            compute_per_agent_gae_and_returns(
                reward=bad,
                old_value=torch.ones((2, 2)),
                bootstrap_value=torch.ones((2, 2)),
                **common,
            )

    def test_role_actor_credit_and_branch_axis_are_agent_specific(self) -> None:
        shape = (2, 2)
        advantage = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
        current_value = torch.zeros(shape, requires_grad=True)
        return_target = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        active = torch.ones((*shape, 7), dtype=torch.bool)

        def evaluate(agent_one_advantage: Tensor):
            branch = torch.full((*shape, 7), 0.05, requires_grad=True)
            output = compute_ppo_objective_and_loss(
                new_joint_log_prob=torch.zeros(shape, requires_grad=True),
                old_joint_log_prob=torch.zeros(shape),
                advantage=agent_one_advantage,
                current_value=current_value,
                return_target=return_target,
                entropy=torch.zeros(shape, requires_grad=True),
                sequence_valid_mask=torch.ones(2, dtype=torch.bool),
                epsilon_clip=0.2,
                value_coefficient=0.5,
                entropy_coefficient=0.01,
                actor_ratio_mode=ActorRatioMode.BRANCH_SPECIFIC.value,
                agent_credit_mode=AgentCreditMode.ROLE_DECOMPOSED.value,
                new_branch_log_probs=branch,
                old_branch_log_probs=torch.zeros_like(branch),
                active_branch_indicators=active,
            )
            output.total_loss.backward()
            assert branch.grad is not None
            return output, branch.grad.detach().clone()

        first, first_gradient = evaluate(advantage)
        changed = advantage.clone()
        changed[:, 1] = torch.tensor([-100.0, -200.0])
        second, second_gradient = evaluate(changed)
        self.assertTrue(torch.equal(first.expanded_advantage, advantage))
        self.assertEqual(tuple(first.branch_surrogate.shape), (2, 2, 7))
        self.assertTrue(
            torch.allclose(
                first.branch_surrogate[:, 0],
                second.branch_surrogate[:, 0],
            )
        )
        self.assertTrue(
            torch.allclose(first_gradient[:, 0], second_gradient[:, 0])
        )
        self.assertEqual(len(first.diagnostics.per_head_critic_loss), 2)
        self.assertEqual(
            first.diagnostics.same_timestep_advantage_equality_rate,
            0.0,
        )

    def test_role_ppo_and_rollout_reject_scalar_credit(self) -> None:
        with self.assertRaisesRegex(PPOObjectiveError, r"\[T,A\]"):
            compute_ppo_objective_and_loss(
                new_joint_log_prob=torch.zeros((2, 2)),
                old_joint_log_prob=torch.zeros((2, 2)),
                advantage=torch.ones(2),
                current_value=torch.ones((2, 2)),
                return_target=torch.ones((2, 2)),
                entropy=torch.zeros((2, 2)),
                sequence_valid_mask=torch.ones(2, dtype=torch.bool),
                epsilon_clip=0.2,
                value_coefficient=0.5,
                entropy_coefficient=0.01,
                agent_credit_mode=AgentCreditMode.ROLE_DECOMPOSED.value,
            )
        team_config = credit_config(AgentCreditMode.TEAM)
        role_config = credit_config(AgentCreditMode.ROLE_DECOMPOSED)
        scalar_transition = make_seed_transition(team_config)
        with self.assertRaisesRegex(RolloutStorageError, "reward shape"):
            CAGATMAPPORolloutBuffer(role_config, capacity=1).append(
                scalar_transition
            )
        vector_transition = replace(
            scalar_transition,
            old_value=scalar_transition.old_value.repeat(4),
            reward=scalar_transition.reward.repeat(4),
            bootstrap_value=scalar_transition.bootstrap_value.repeat(4),
        )
        with self.assertRaisesRegex(RolloutStorageError, "reward shape"):
            CAGATMAPPORolloutBuffer(team_config, capacity=1).append(
                vector_transition
            )

    def test_trainer_rejects_missing_wrong_or_nonconservative_agent_reward(self) -> None:
        trainer = object.__new__(CAGATMAPPOTrainer)
        trainer.config = credit_config(AgentCreditMode.ROLE_DECOMPOSED)
        trainer.actor = SimpleNamespace(spec=SimpleNamespace(uav_count=4))
        trainer.dtype = torch.float32
        trainer.device = torch.device("cpu")

        def result(info: dict[str, object]) -> StepResult:
            return StepResult(
                observations=None,
                centralized_state=None,
                reward=1.0,
                terminated=False,
                truncated=True,
                info=info,
            )

        with self.assertRaisesRegex(
            CAGATMAPPOTrainerError,
            r"requires agent_reward\[A\]",
        ):
            trainer._agent_reward_from_step(result({}))
        with self.assertRaisesRegex(CAGATMAPPOTrainerError, "shape differs"):
            trainer._agent_reward_from_step(
                result(
                    {
                        "agent_reward": [0.25, 0.25],
                        "agent_credit": {"conservation_residual": 0.0},
                    }
                )
            )
        with self.assertRaisesRegex(CAGATMAPPOTrainerError, "conservation"):
            trainer._agent_reward_from_step(
                result(
                    {
                        "agent_reward": [1.0, 1.0, 1.0, 1.0],
                        "agent_credit": {"conservation_residual": 0.0},
                    }
                )
            )
        with self.assertRaisesRegex(CAGATMAPPOTrainerError, "must be finite"):
            trainer._agent_reward_from_step(
                result(
                    {
                        "agent_reward": [1.0, 0.0, 0.0, float("nan")],
                        "agent_credit": {"conservation_residual": 0.0},
                    }
                )
            )

    def test_role_rollout_update_and_diagnostics_are_vectorized(self) -> None:
        config, buffer = role_rollout_fixture()
        first_transition = buffer.transition_at(0)
        self.assertEqual(tuple(first_transition.reward.shape), (4,))
        self.assertEqual(tuple(first_transition.old_value.shape), (4,))
        self.assertEqual(tuple(first_transition.bootstrap_value.shape), (4,))
        minibatch = build_recurrent_ppo_minibatch(buffer, config)
        self.assertEqual(tuple(minibatch.old_value.shape), (8, 32, 4))
        self.assertEqual(tuple(minibatch.td_residual.shape), (8, 32, 4))
        self.assertEqual(tuple(minibatch.advantage.shape), (8, 32, 4))
        self.assertEqual(tuple(minibatch.return_target.shape), (8, 32, 4))
        self.assertFalse(
            torch.allclose(
                minibatch.return_target[..., 0],
                minibatch.return_target[..., 1],
            )
        )
        critic = MAPPOCentralizedCritic(config)
        actor = CAGATMAPPOActor(config)
        updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, config)
        head_before = critic.value_network[-1].weight.detach().clone()
        backward_count = 0
        actor_forward_count = 0
        critic_forward_count = 0
        real_backward = torch.Tensor.backward
        real_actor_forward = actor.forward
        real_critic_forward = critic.forward

        def counted_backward(tensor, *args, **kwargs):
            nonlocal backward_count
            backward_count += 1
            return real_backward(tensor, *args, **kwargs)

        def counted_actor_forward(*args, **kwargs):
            nonlocal actor_forward_count
            actor_forward_count += 1
            return real_actor_forward(*args, **kwargs)

        def counted_critic_forward(*args, **kwargs):
            nonlocal critic_forward_count
            critic_forward_count += 1
            return real_critic_forward(*args, **kwargs)

        self.assertEqual(updater.action_distribution._generators, {})
        with (
            patch.object(torch.Tensor, "backward", new=counted_backward),
            patch.object(actor, "forward", side_effect=counted_actor_forward),
            patch.object(critic, "forward", side_effect=counted_critic_forward),
            patch.object(
                updater.action_distribution,
                "sample_actions",
                side_effect=AssertionError("update must not sample policy actions"),
            ),
            patch.object(
                U2UMECEnvironment,
                "step",
                side_effect=AssertionError("update must not step an environment"),
            ),
        ):
            output = updater.update(buffer)
        self.assertEqual(backward_count, 4)
        self.assertEqual(actor_forward_count, 4)
        self.assertEqual(critic_forward_count, 4)
        self.assertEqual(updater.action_distribution._generators, {})
        self.assertEqual(len(output.epoch_diagnostics), 4)
        self.assertFalse(
            torch.equal(head_before, critic.value_network[-1].weight.detach())
        )
        for epoch in output.epoch_diagnostics:
            telemetry = epoch.agent_credit_telemetry
            self.assertIsNotNone(telemetry)
            assert telemetry is not None
            self.assertEqual(len(telemetry.per_agent_value_mean), 4)
            self.assertEqual(len(telemetry.per_agent_td_residual_mean), 4)
            self.assertEqual(len(telemetry.per_agent_advantage_mean), 4)
            self.assertEqual(len(telemetry.per_agent_return_mean), 4)
            self.assertEqual(len(telemetry.per_head_critic_loss), 4)
            self.assertEqual(
                set(telemetry.route_category_agent_advantage_mean),
                {"local", "remote", "defer"},
            )
            self.assertLess(
                telemetry.same_timestep_advantage_equality_rate,
                1.0,
            )
            self.assertTrue(
                all(
                    math.isfinite(value)
                    for values in (
                        telemetry.per_agent_value_mean,
                        telemetry.per_agent_td_residual_mean,
                        telemetry.per_agent_advantage_mean,
                        telemetry.per_agent_return_mean,
                        telemetry.per_head_critic_loss,
                    )
                    for value in values
                )
            )


if __name__ == "__main__":
    unittest.main()
