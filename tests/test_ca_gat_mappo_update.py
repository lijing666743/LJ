"""Recurrent PPO Minibatch / Update Gate for CA-GAT-MAPPO."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from src.config import RunConfig
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import (
    ACTION_BRANCH_ORDER,
    ActorObservationTensorizer,
    ActorTensorBatch,
    CAGATMAPPOActor,
    CentralizedStateTensorizer,
    MAPPOCentralizedCritic,
)
from src.models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionMaskBatch,
)
from src.models.ca_gat_mappo_rollout import (
    CAGATMAPPORolloutBuffer,
    CAGATMAPPORolloutTransition,
)
from src.models.ca_gat_mappo_update import (
    CAGATMAPPORecurrentPPOUpdater,
    RecurrentPPOUpdateError,
    build_ca_gat_mappo_optimizers,
    build_recurrent_ppo_minibatch,
    compute_configured_batched_ppo_objective_and_loss,
)


def make_update_config() -> RunConfig:
    base = RunConfig()
    count = base.environment.uav_count
    environment = replace(
        base.environment,
        episode_horizon=500,
        arrival_probabilities=(1.0,) * count,
        profile_assignment=("Balanced",) * count,
        profile_perturbations=((0.0, 0.0, 0.0, 0.0),) * count,
        building_layout=(),
        candidate_neighbor_radius_m=2_000.0,
        velocity_std_mps=(0.0, 0.0),
        shadowing_std_db=0.0,
        csi_error_std_db=0.0,
        fixed_csi_aoi_slots=1,
    )
    config = replace(base, environment=environment)
    config.validate()
    return config


def make_seed_transition(config: RunConfig) -> CAGATMAPPORolloutTransition:
    environment = U2UMECEnvironment(config)
    reset = environment.reset()
    actor_tensorizer = ActorObservationTensorizer(config)
    critic_tensorizer = CentralizedStateTensorizer(config)
    actor = CAGATMAPPOActor(config).eval()
    critic = MAPPOCentralizedCritic(config).eval()
    distribution = CAGATMAPPOActionDistribution(actor, config)
    actor_batch = actor_tensorizer.encode_step(
        reset.observations,
        episode_start=True,
    )
    action_masks = SequentialActionMaskBatch.from_observations(reset.observations)
    critic_batch = critic_tensorizer.encode_step(reset.centralized_state)
    hidden = actor.initial_hidden(1)
    with torch.no_grad():
        action_output = distribution.sample_actions(actor_batch, action_masks, hidden)
    step = environment.step(action_output.proposals[0][0])
    assert step.observations is not None
    assert step.centralized_state is not None
    hidden = action_output.hidden_out.detach().clone()
    actor_batch = actor_tensorizer.encode_step(
        step.observations,
        episode_start=False,
    )
    action_masks = SequentialActionMaskBatch.from_observations(step.observations)
    critic_batch = critic_tensorizer.encode_step(step.centralized_state)
    with torch.no_grad():
        action_output = distribution.sample_actions(actor_batch, action_masks, hidden)
        old_value = critic(critic_batch)[0, 0, 0]
    step = environment.step(action_output.proposals[0][0])
    assert step.centralized_state is not None
    next_critic_batch = critic_tensorizer.encode_step(step.centralized_state)
    with torch.no_grad():
        bootstrap_value = critic(next_critic_batch)[0, 0, 0]
    return CAGATMAPPORolloutTransition.from_step(
        spec=actor.spec,
        slot=1,
        actor_batch=actor_batch,
        action_mask_batch=action_masks,
        action_output=action_output,
        hidden_in=hidden,
        centralized_state=critic_batch,
        old_value=old_value,
        reward=1.0,
        terminated=False,
        truncated=False,
        episode_boundary=False,
        bootstrap_allowed=True,
        bootstrap_value=bootstrap_value,
        executed_action_summary=step.info["executed"],
        rejection_or_downgrade_summary={
            "rejection": step.info["rejection"],
            "downgrade": step.info["downgrade"],
            "canonicalization": step.info["canonicalization"],
        },
    )


def make_synthetic_buffer(
    config: RunConfig,
    seed_transition: CAGATMAPPORolloutTransition,
    *,
    length: int = 256,
    finalized: bool = True,
    boundary_period: int | None = None,
) -> CAGATMAPPORolloutBuffer:
    buffer = CAGATMAPPORolloutBuffer(config, capacity=length)
    for index in range(length):
        slot = index if boundary_period is None else index % boundary_period
        episode_start = slot == 0
        boundary = boundary_period is not None and slot == boundary_period - 1
        hidden = (
            torch.zeros_like(seed_transition.hidden_in)
            if episode_start
            else torch.full_like(seed_transition.hidden_in, 0.001 * (index + 1))
        )
        executed = tuple(
            {**dict(item), "route": "executor-different-from-proposal"}
            for item in seed_transition.executed_action_summary
        )
        transition = replace(
            seed_transition,
            slot=slot,
            episode_start=episode_start,
            hidden_in=hidden,
            reward=torch.tensor(1.0 + 0.01 * (index % 7), dtype=torch.float32),
            terminated=False,
            truncated=boundary,
            episode_boundary=boundary,
            bootstrap_allowed=not boundary,
            bootstrap_value=(
                None
                if boundary
                else seed_transition.bootstrap_value.detach().clone()
            ),
            executed_action_summary=executed,
        )
        buffer.append(transition)
    if finalized:
        buffer.finalize()
    return buffer


def align_old_policy_snapshots(
    config: RunConfig,
    buffer: CAGATMAPPORolloutBuffer,
) -> CAGATMAPPORolloutBuffer:
    """Make synthetic old log-probs match its stored recurrent inputs exactly."""

    actor = CAGATMAPPOActor(config).eval()
    distribution = CAGATMAPPOActionDistribution(actor, config)
    minibatch = build_recurrent_ppo_minibatch(buffer, config)
    with torch.no_grad():
        evaluation = distribution.evaluate_actions(
            minibatch.actor_batch,
            minibatch.action_mask_batch,
            minibatch.proposals,
            minibatch.initial_hidden,
        )
    branch_log_probs = torch.stack(
        [evaluation.branch_log_probs[branch] for branch in ACTION_BRANCH_ORDER],
        dim=-1,
    )
    aligned = CAGATMAPPORolloutBuffer(config, capacity=len(buffer))
    for index in range(len(buffer)):
        batch_index, time_index = divmod(index, 32)
        transition = buffer.transition_at(index)
        aligned.append(
            replace(
                transition,
                old_branch_log_probs=branch_log_probs[
                    batch_index, time_index
                ].detach().cpu().clone(),
                old_joint_log_prob=evaluation.joint_log_prob[
                    batch_index, time_index
                ].detach().cpu().clone(),
            )
        )
    aligned.finalize()
    return aligned


def slice_actor_step(batch: ActorTensorBatch, batch_index: int, step: int) -> ActorTensorBatch:
    batch_slice = slice(batch_index, batch_index + 1)
    time_slice = slice(step, step + 1)
    return ActorTensorBatch(
        self_features=batch.self_features[batch_slice, time_slice],
        neighbor_public_features=batch.neighbor_public_features[batch_slice, time_slice],
        edge_features=batch.edge_features[batch_slice, time_slice],
        neighbor_mask=batch.neighbor_mask[batch_slice, time_slice],
        action_masks={
            branch: batch.action_masks[branch][batch_slice, time_slice]
            for branch in ACTION_BRANCH_ORDER
        },
        action_indices={
            branch: batch.action_indices[branch][batch_slice, time_slice]
            for branch in ACTION_BRANCH_ORDER
        },
        episode_starts=batch.episode_starts[batch_slice, time_slice],
    )


class RecurrentPPOUpdateGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = make_update_config()
        cls.seed_transition = make_seed_transition(cls.config)
        raw_buffer = make_synthetic_buffer(cls.config, cls.seed_transition)
        cls.buffer = align_old_policy_snapshots(cls.config, raw_buffer)
        cls.minibatch = build_recurrent_ppo_minibatch(cls.buffer, cls.config)
        cls.boundary_buffer = make_synthetic_buffer(
            cls.config,
            cls.seed_transition,
            boundary_period=10,
        )
        cls.boundary_minibatch = build_recurrent_ppo_minibatch(
            cls.boundary_buffer,
            cls.config,
        )

    def test_01_geometry_order_no_shuffle_and_no_padding(self) -> None:
        minibatch = self.minibatch
        self.assertEqual(tuple(minibatch.rollout_indices.shape), (8, 32))
        self.assertTrue(
            torch.equal(
                minibatch.rollout_indices,
                torch.arange(256, dtype=torch.long).reshape(8, 32),
            )
        )
        self.assertTrue(torch.all(minibatch.sequence_valid_mask))
        self.assertEqual(tuple(minibatch.actor_batch.self_features.shape[:2]), (8, 32))
        repeated = build_recurrent_ppo_minibatch(self.buffer, self.config)
        self.assertTrue(torch.equal(repeated.rollout_indices, minibatch.rollout_indices))
        self.assertEqual(repeated.proposals, minibatch.proposals)

    def test_02_unfinalized_rollout_fails_fast(self) -> None:
        buffer = make_synthetic_buffer(
            self.config,
            self.seed_transition,
            finalized=False,
        )
        with self.assertRaisesRegex(RecurrentPPOUpdateError, "finalized"):
            build_recurrent_ppo_minibatch(buffer, self.config)

    def test_03_illegal_rollout_lengths_fail_fast(self) -> None:
        for length in (255, 257):
            with self.subTest(length=length):
                buffer = make_synthetic_buffer(
                    self.config,
                    self.seed_transition,
                    length=length,
                )
                with self.assertRaisesRegex(RecurrentPPOUpdateError, "exactly 256"):
                    build_recurrent_ppo_minibatch(buffer, self.config)

    def test_04_nonfrozen_geometry_fails_fast(self) -> None:
        changed_mappo = replace(
            self.config.training.mappo,
            rollout_length_slots=128,
            recurrent_chunk_length_slots=16,
            sequence_minibatch_size=8,
        )
        changed = replace(
            self.config,
            training=replace(self.config.training, mappo=changed_mappo),
        )
        changed.validate()
        with self.assertRaisesRegex(RecurrentPPOUpdateError, "geometry"):
            build_recurrent_ppo_minibatch(self.buffer, changed)

    def test_05_chunk_starts_use_exact_stored_hidden_without_burnin(self) -> None:
        expected = torch.stack(
            [self.buffer.transition_at(index * 32).hidden_in for index in range(8)]
        )
        self.assertTrue(torch.equal(self.minibatch.initial_hidden, expected))
        actor = CAGATMAPPOActor(self.config)
        critic = MAPPOCentralizedCritic(self.config)
        updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, self.config)
        original_forward = actor.forward
        observed: list[tuple[tuple[int, ...], torch.Tensor]] = []

        def actor_forward(batch, hidden_in=None):
            observed.append((tuple(batch.self_features.shape[:2]), hidden_in.detach().clone()))
            return original_forward(batch, hidden_in)

        actor.forward = actor_forward
        with torch.no_grad():
            updater.evaluate_minibatch(self.minibatch)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0], (8, 32))
        self.assertTrue(torch.equal(observed[0][1], expected))

    def test_06_episode_boundary_resets_hidden_without_dropping_transition(self) -> None:
        actor = CAGATMAPPOActor(self.config)
        critic = MAPPOCentralizedCritic(self.config)
        updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, self.config)
        with torch.no_grad():
            evaluation = updater.evaluate_minibatch(self.boundary_minibatch)
            one_step = slice_actor_step(self.boundary_minibatch.actor_batch, 0, 10)
            isolated = actor(
                one_step,
                self.boundary_minibatch.stored_hidden_in[0, 10].unsqueeze(0),
            )
        self.assertTrue(self.boundary_buffer.transition_at(9).episode_boundary)
        self.assertTrue(self.boundary_buffer.transition_at(10).episode_start)
        self.assertEqual(
            torch.count_nonzero(self.boundary_minibatch.stored_hidden_in[0, 10]).item(),
            0,
        )
        self.assertTrue(
            torch.allclose(
                evaluation.policy.recurrent_features[0, 10],
                isolated.recurrent_features[0, 0],
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        )
        loss = compute_configured_batched_ppo_objective_and_loss(
            new_joint_log_prob=evaluation.policy.joint_log_prob,
            old_joint_log_prob=self.boundary_minibatch.old_joint_log_prob,
            advantage=self.boundary_minibatch.advantage,
            current_value=evaluation.current_value,
            return_target=self.boundary_minibatch.return_target,
            entropy=evaluation.policy.entropy,
            sequence_valid_mask=self.boundary_minibatch.sequence_valid_mask,
            config=self.config,
        )
        self.assertTrue(torch.all(loss.actor_valid_mask[0, 9]))
        self.assertTrue(bool(loss.critic_valid_mask[0, 9]))

    def test_07_proposals_masks_and_active_branches_are_re_evaluated_not_sampled(self) -> None:
        actor = CAGATMAPPOActor(self.config)
        critic = MAPPOCentralizedCritic(self.config)
        distribution = CAGATMAPPOActionDistribution(actor, self.config)
        updater = CAGATMAPPORecurrentPPOUpdater(
            actor,
            critic,
            self.config,
            action_distribution=distribution,
        )
        self.assertEqual(distribution._generators, {})
        with torch.no_grad():
            evaluation = updater.evaluate_minibatch(self.minibatch)
        self.assertEqual(distribution._generators, {})
        self.assertEqual(evaluation.policy.proposals, self.minibatch.proposals)
        for branch in ACTION_BRANCH_ORDER:
            self.assertTrue(
                torch.equal(
                    evaluation.policy.action_masks[branch],
                    self.minibatch.actor_batch.action_masks[branch],
                )
            )
        active = torch.stack(
            [evaluation.policy.active_branches[branch] for branch in ACTION_BRANCH_ORDER],
            dim=-1,
        )
        self.assertTrue(torch.equal(active, self.minibatch.active_branch_indicators))
        active_entropy = sum(evaluation.policy.branch_entropies.values())
        self.assertTrue(torch.allclose(active_entropy, evaluation.policy.entropy))

    def test_08_ratio_uses_stored_proposal_not_executor_metadata(self) -> None:
        actor = CAGATMAPPOActor(self.config)
        critic = MAPPOCentralizedCritic(self.config)
        updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, self.config)
        with torch.no_grad():
            evaluation = updater.evaluate_minibatch(self.minibatch)
        first = self.buffer.transition_at(0)
        self.assertNotEqual(
            first.proposal_actions[0].route,
            first.executed_action_summary[0]["route"],
        )
        self.assertFalse(hasattr(self.minibatch, "executed_action_summary"))
        loss = compute_configured_batched_ppo_objective_and_loss(
            new_joint_log_prob=evaluation.policy.joint_log_prob,
            old_joint_log_prob=self.minibatch.old_joint_log_prob,
            advantage=self.minibatch.advantage,
            current_value=evaluation.current_value,
            return_target=self.minibatch.return_target,
            entropy=evaluation.policy.entropy,
            sequence_valid_mask=self.minibatch.sequence_valid_mask,
            config=self.config,
        )
        expected = torch.exp(
            evaluation.policy.joint_log_prob - self.minibatch.old_joint_log_prob
        )
        self.assertTrue(torch.allclose(loss.ratio, expected))

    def test_09_batched_shapes_and_shared_raw_advantage(self) -> None:
        agents = self.config.environment.uav_count
        new = torch.zeros((8, 32, agents), requires_grad=True)
        old = torch.zeros_like(new)
        advantage = torch.arange(256, dtype=torch.float32).reshape(8, 32) / 100.0
        value = torch.zeros((8, 32), requires_grad=True)
        returns = torch.ones((8, 32))
        entropy = torch.full((8, 32, agents), 0.25, requires_grad=True)
        valid = torch.ones((8, 32), dtype=torch.bool)
        loss = compute_configured_batched_ppo_objective_and_loss(
            new_joint_log_prob=new,
            old_joint_log_prob=old,
            advantage=advantage,
            current_value=value,
            return_target=returns,
            entropy=entropy,
            sequence_valid_mask=valid,
            config=self.config,
        )
        self.assertEqual(tuple(loss.ratio.shape), (8, 32, agents))
        self.assertEqual(tuple(loss.expanded_advantage.shape), (8, 32, agents))
        self.assertTrue(
            torch.equal(loss.expanded_advantage, advantage.unsqueeze(-1).expand_as(new))
        )
        self.assertFalse(loss.expanded_advantage.requires_grad)
        self.assertTrue(torch.equal(advantage, torch.arange(256).reshape(8, 32) / 100.0))

    def test_10_uniform_actor_critic_and_entropy_reductions(self) -> None:
        agents = self.config.environment.uav_count
        advantage = torch.arange(256, dtype=torch.float32).reshape(8, 32) / 32.0
        current_value = torch.arange(256, dtype=torch.float32).reshape(8, 32) / 100.0
        return_target = torch.flip(current_value, dims=(1,))
        entropy = torch.arange(8 * 32 * agents, dtype=torch.float32).reshape(
            8, 32, agents
        ) / 1_000.0
        loss = compute_configured_batched_ppo_objective_and_loss(
            new_joint_log_prob=torch.zeros((8, 32, agents), requires_grad=True),
            old_joint_log_prob=torch.zeros((8, 32, agents)),
            advantage=advantage,
            current_value=current_value.requires_grad_(),
            return_target=return_target,
            entropy=entropy.requires_grad_(),
            sequence_valid_mask=torch.ones((8, 32), dtype=torch.bool),
            config=self.config,
        )
        self.assertTrue(torch.allclose(loss.actor_loss, -advantage.mean()))
        self.assertTrue(
            torch.allclose(
                loss.critic_loss,
                0.5 * (current_value - return_target).square().mean(),
            )
        )
        self.assertTrue(torch.allclose(loss.entropy_mean, entropy.mean()))

    def test_11_optimizer_contract_and_disjoint_parameter_ids(self) -> None:
        actor = CAGATMAPPOActor(self.config)
        critic = MAPPOCentralizedCritic(self.config)
        bundle = build_ca_gat_mappo_optimizers(actor, critic, self.config)
        self.assertIsInstance(bundle.actor_optimizer, torch.optim.Adam)
        self.assertIsInstance(bundle.critic_optimizer, torch.optim.Adam)
        self.assertFalse(
            set(map(id, bundle.actor_parameters)) & set(map(id, bundle.critic_parameters))
        )
        mappo = self.config.training.mappo
        for optimizer, learning_rate in (
            (bundle.actor_optimizer, mappo.actor_learning_rate),
            (bundle.critic_optimizer, mappo.critic_learning_rate),
        ):
            group = optimizer.param_groups[0]
            self.assertEqual(group["lr"], learning_rate)
            self.assertEqual(group["betas"], (mappo.adam_beta1, mappo.adam_beta2))
            self.assertEqual(group["eps"], mappo.adam_eps)
            self.assertEqual(group["weight_decay"], mappo.weight_decay)
        bundle.actor_optimizer.param_groups[0]["lr"] = 1.0e-2
        with self.assertRaisesRegex(RecurrentPPOUpdateError, "hyperparameters"):
            CAGATMAPPORecurrentPPOUpdater(
                actor, critic, self.config, optimizers=bundle
            )

    def test_12_shared_trainable_parameter_fails_fast(self) -> None:
        actor = CAGATMAPPOActor(self.config)
        critic = MAPPOCentralizedCritic(self.config)
        critic.register_parameter("accidental_shared", next(actor.parameters()))
        with self.assertRaisesRegex(RecurrentPPOUpdateError, "overlap"):
            build_ca_gat_mappo_optimizers(actor, critic, self.config)

    def test_13_cpu_update_order_clipping_snapshots_rng_and_parameter_steps(self) -> None:
        actor = CAGATMAPPOActor(self.config)
        critic = MAPPOCentralizedCritic(self.config)
        bundle = build_ca_gat_mappo_optimizers(actor, critic, self.config)
        updater = CAGATMAPPORecurrentPPOUpdater(
            actor,
            critic,
            self.config,
            optimizers=bundle,
        )
        with torch.no_grad():
            before_evaluation = updater.evaluate_minibatch(self.minibatch)
            before_log_prob = before_evaluation.policy.joint_log_prob.detach().clone()
        actor_before = {name: value.detach().clone() for name, value in actor.named_parameters()}
        critic_before = {
            name: value.detach().clone() for name, value in critic.named_parameters()
        }
        old_before = self.minibatch.old_joint_log_prob.clone()
        value_before = self.minibatch.old_value.clone()
        advantage_before = self.minibatch.advantage.clone()
        return_before = self.minibatch.return_target.clone()
        transition_before = self.buffer.transition_at(0)
        rng_before = torch.get_rng_state().clone()
        events: list[str] = []
        old_epoch_snapshots: list[torch.Tensor] = []
        advantage_epoch_snapshots: list[torch.Tensor] = []
        return_epoch_snapshots: list[torch.Tensor] = []
        real_actor_zero = bundle.actor_optimizer.zero_grad
        real_critic_zero = bundle.critic_optimizer.zero_grad
        real_actor_step = bundle.actor_optimizer.step
        real_critic_step = bundle.critic_optimizer.step
        real_actor_forward = actor.forward
        real_critic_forward = critic.forward
        real_clip = torch.nn.utils.clip_grad_norm_
        real_loss = compute_configured_batched_ppo_objective_and_loss
        actor_ids = set(map(id, bundle.actor_parameters))

        def actor_zero(*args, **kwargs):
            events.append("actor_zero")
            self.assertTrue(kwargs.get("set_to_none"))
            return real_actor_zero(*args, **kwargs)

        def critic_zero(*args, **kwargs):
            events.append("critic_zero")
            self.assertTrue(kwargs.get("set_to_none"))
            return real_critic_zero(*args, **kwargs)

        def actor_forward(*args, **kwargs):
            events.append("actor_forward")
            return real_actor_forward(*args, **kwargs)

        def critic_forward(*args, **kwargs):
            events.append("critic_forward")
            return real_critic_forward(*args, **kwargs)

        def loss_forward(**kwargs):
            events.append("loss")
            old_epoch_snapshots.append(kwargs["old_joint_log_prob"].detach().cpu().clone())
            advantage_epoch_snapshots.append(kwargs["advantage"].detach().cpu().clone())
            return_epoch_snapshots.append(kwargs["return_target"].detach().cpu().clone())
            return real_loss(**kwargs)

        def clip(parameters, *args, **kwargs):
            copied = tuple(parameters)
            label = "actor_clip" if id(copied[0]) in actor_ids else "critic_clip"
            events.append(label)
            self.assertEqual(kwargs["max_norm"], self.config.training.mappo.gradient_clip_norm)
            self.assertTrue(any(parameter.grad is not None for parameter in copied))
            return real_clip(copied, *args, **kwargs)

        def actor_step(*args, **kwargs):
            events.append("actor_step")
            return real_actor_step(*args, **kwargs)

        def critic_step(*args, **kwargs):
            events.append("critic_step")
            return real_critic_step(*args, **kwargs)

        with (
            patch.object(bundle.actor_optimizer, "zero_grad", side_effect=actor_zero),
            patch.object(bundle.critic_optimizer, "zero_grad", side_effect=critic_zero),
            patch.object(bundle.actor_optimizer, "step", side_effect=actor_step),
            patch.object(bundle.critic_optimizer, "step", side_effect=critic_step),
            patch.object(actor, "forward", side_effect=actor_forward),
            patch.object(critic, "forward", side_effect=critic_forward),
            patch(
                "src.models.ca_gat_mappo_update.compute_configured_batched_ppo_objective_and_loss",
                side_effect=loss_forward,
            ),
            patch("torch.nn.utils.clip_grad_norm_", side_effect=clip),
        ):
            output = updater.update(self.buffer)

        per_epoch = [
            "actor_zero",
            "critic_zero",
            "actor_forward",
            "critic_forward",
            "loss",
            "actor_clip",
            "critic_clip",
            "actor_step",
            "critic_step",
        ]
        self.assertEqual(events, per_epoch * 4)
        self.assertEqual([item.epoch_index for item in output.epoch_diagnostics], list(range(4)))
        self.assertTrue(all(item.clip_max_norm == 0.5 for item in output.epoch_diagnostics))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertTrue(
            any(
                not torch.equal(actor_before[name], parameter.detach())
                for name, parameter in actor.named_parameters()
            )
        )
        self.assertTrue(
            any(
                not torch.equal(critic_before[name], parameter.detach())
                for name, parameter in critic.named_parameters()
            )
        )
        self.assertEqual(len(old_epoch_snapshots), 4)
        self.assertTrue(all(torch.equal(item, old_before) for item in old_epoch_snapshots))
        self.assertTrue(
            all(torch.equal(item, advantage_before) for item in advantage_epoch_snapshots)
        )
        self.assertTrue(all(torch.equal(item, return_before) for item in return_epoch_snapshots))
        self.assertTrue(torch.equal(self.minibatch.old_joint_log_prob, old_before))
        self.assertTrue(torch.equal(self.minibatch.old_value, value_before))
        self.assertTrue(torch.equal(self.minibatch.advantage, advantage_before))
        self.assertTrue(torch.equal(self.minibatch.return_target, return_before))
        transition_after = self.buffer.transition_at(0)
        self.assertTrue(
            torch.equal(transition_after.self_features, transition_before.self_features)
        )
        self.assertTrue(
            torch.equal(transition_after.old_joint_log_prob, transition_before.old_joint_log_prob)
        )
        with torch.no_grad():
            after_log_prob = updater.evaluate_minibatch(
                self.minibatch
            ).policy.joint_log_prob
        self.assertFalse(torch.allclose(before_log_prob, after_log_prob))

    def test_14_nonfinite_gradient_norm_fails_fast(self) -> None:
        actor = CAGATMAPPOActor(self.config)
        critic = MAPPOCentralizedCritic(self.config)
        updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, self.config)
        parameter = next(actor.parameters())
        hook = parameter.register_hook(
            lambda gradient: torch.full_like(gradient, float("inf"))
        )
        try:
            with self.assertRaisesRegex(RecurrentPPOUpdateError, "NaN or Inf"):
                updater.update(self.buffer)
        finally:
            hook.remove()

    def test_15_schedule_enabled_update_propagates_batched_loss_telemetry(self) -> None:
        scheduled_mappo = replace(
            self.config.training.mappo,
            entropy_coefficient_schedule_enabled=True,
        )
        scheduled_config = replace(
            self.config,
            training=replace(self.config.training, mappo=scheduled_mappo),
        )
        scheduled_config.validate()
        actor = CAGATMAPPOActor(scheduled_config)
        critic = MAPPOCentralizedCritic(scheduled_config)
        updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, scheduled_config)

        output = updater.update(
            self.buffer,
            collected_environment_steps=3072,
        )

        self.assertEqual(len(output.epoch_diagnostics), 4)
        for epoch in output.epoch_diagnostics:
            self.assertEqual(epoch.route_entropy_coefficient, 0.03)
            self.assertEqual(epoch.route_entropy_schedule_progress, 0.0)
            self.assertIsNotNone(epoch.route_entropy_loss_contribution)
            self.assertIsNotNone(epoch.other_branch_entropy_loss_contribution)
            self.assertIsNotNone(epoch.global_entropy_loss_contribution)
            self.assertEqual(epoch.collected_environment_steps, 3072)
            self.assertAlmostEqual(
                epoch.global_entropy_loss_contribution,
                epoch.route_entropy_loss_contribution
                + epoch.other_branch_entropy_loss_contribution,
            )

    def test_15_actor_critic_device_mismatch_fails_without_fallback(self) -> None:
        actor = CAGATMAPPOActor(self.config)
        critic = MAPPOCentralizedCritic(self.config).to("meta")
        with self.assertRaisesRegex(RecurrentPPOUpdateError, "same device"):
            build_ca_gat_mappo_optimizers(actor, critic, self.config)
        self.assertEqual(next(actor.parameters()).device.type, "cpu")

    def test_16_cpu_device_and_detached_autograd_contract(self) -> None:
        actor = CAGATMAPPOActor(self.config)
        critic = MAPPOCentralizedCritic(self.config)
        updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, self.config)
        with torch.no_grad():
            evaluation = updater.evaluate_minibatch(self.minibatch)
        self.assertEqual(evaluation.policy.joint_log_prob.device.type, "cpu")
        self.assertEqual(evaluation.current_value.device.type, "cpu")
        for tensor in (
            self.minibatch.actor_batch.self_features,
            self.minibatch.initial_hidden,
            self.minibatch.old_joint_log_prob,
            self.minibatch.old_value,
            self.minibatch.advantage,
            self.minibatch.return_target,
        ):
            self.assertFalse(tensor.requires_grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_17_cuda_update_passes_on_model_device(self) -> None:
        actor = CAGATMAPPOActor(self.config).cuda()
        critic = MAPPOCentralizedCritic(self.config).cuda()
        updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, self.config)
        output = updater.update(self.buffer)
        self.assertEqual(len(output.epoch_diagnostics), 4)
        with torch.no_grad():
            evaluation = updater.evaluate_minibatch(self.minibatch)
        self.assertEqual(evaluation.policy.joint_log_prob.device.type, "cuda")
        self.assertEqual(evaluation.current_value.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
