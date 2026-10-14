"""Fix Package 4 gates for branch-specific PPO ratios."""

from __future__ import annotations

import csv
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from src.config import (
    ActorRatioMode,
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_SCHEMA_VERSION,
    ConfigError,
    RunConfig,
    validate_mappo_checkpoint_resume_compatibility,
)
from src.models.ca_gat_mappo import (
    ACTION_BRANCH_ORDER,
    CAGATMAPPOActor,
    MAPPOCentralizedCritic,
)
from src.models.ca_gat_mappo_checkpoint import (
    CheckpointError,
    validate_periodic_checkpoint_compatibility,
)
from src.models.ca_gat_mappo_ppo import compute_ppo_objective_and_loss
from src.models.ca_gat_mappo_route_telemetry import compute_branch_ppo_dynamics
from src.models.ca_gat_mappo_update import (
    CAGATMAPPORecurrentPPOUpdater,
    compute_configured_batched_ppo_objective_and_loss,
)
from src.training_artifacts import (
    _branch_activity_records,
    _csv_text,
    _json_lines,
)
from tests.test_ca_gat_mappo_checkpoint import make_checkpoint_config, make_payload
from tests.test_ca_gat_mappo_update import (
    align_old_policy_snapshots,
    make_seed_transition,
    make_synthetic_buffer,
    make_update_config,
)


def _rl_config(mode: ActorRatioMode = ActorRatioMode.JOINT) -> RunConfig:
    base = RunConfig()
    config = replace(
        base,
        mode="rl",
        method_id="ca_gat_mappo",
        training=replace(
            base.training,
            mappo=replace(
                base.training.mappo,
                actor_ratio_mode=mode,
                training_device="cpu",
            ),
            formal_rl_enabled=True,
        ),
    )
    object.__setattr__(config, "git_branch", "fix4-test")
    object.__setattr__(config, "git_commit", "a" * 40)
    object.__setattr__(config, "git_dirty", False)
    config.validate()
    return config


def _objective(
    new_branch_log_probs: torch.Tensor,
    old_branch_log_probs: torch.Tensor,
    active: torch.Tensor,
    advantage: torch.Tensor,
    mode: ActorRatioMode,
):
    time_steps, agent_count, _ = new_branch_log_probs.shape
    new_joint = new_branch_log_probs.masked_fill(~active, 0.0).sum(dim=-1)
    old_joint = old_branch_log_probs.masked_fill(~active, 0.0).sum(dim=-1)
    return compute_ppo_objective_and_loss(
        new_joint_log_prob=new_joint,
        old_joint_log_prob=old_joint,
        advantage=advantage,
        current_value=torch.linspace(
            0.1,
            0.1 * time_steps,
            time_steps,
            dtype=new_branch_log_probs.dtype,
            device=new_branch_log_probs.device,
        ),
        return_target=torch.zeros(
            time_steps,
            dtype=new_branch_log_probs.dtype,
            device=new_branch_log_probs.device,
        ),
        entropy=torch.full(
            (time_steps, agent_count),
            0.3,
            dtype=new_branch_log_probs.dtype,
            device=new_branch_log_probs.device,
        ),
        sequence_valid_mask=torch.ones(
            time_steps,
            dtype=torch.bool,
            device=new_branch_log_probs.device,
        ),
        epsilon_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        actor_ratio_mode=mode,
        new_branch_log_probs=new_branch_log_probs,
        old_branch_log_probs=old_branch_log_probs,
        active_branch_indicators=active,
    )


class Fix4ConfigAndCheckpointTests(unittest.TestCase):
    def test_default_joint_preserves_legacy_identity(self) -> None:
        joint = _rl_config()
        branch_specific = _rl_config(ActorRatioMode.BRANCH_SPECIFIC)
        self.assertEqual(joint.training.mappo.actor_ratio_mode, ActorRatioMode.JOINT)
        self.assertNotIn(
            "actor_ratio_mode",
            joint.resolved_dict()["training"]["mappo"],
        )
        self.assertEqual(
            branch_specific.resolved_dict()["training"]["mappo"][
                "actor_ratio_mode"
            ],
            "branch_specific",
        )
        self.assertNotEqual(joint.config_hash, branch_specific.config_hash)
        self.assertEqual(
            joint.environment.workload_timing_mode.value,
            "legacy_post_route",
        )

    def test_invalid_mode_is_rejected(self) -> None:
        config = _rl_config()
        invalid = replace(
            config,
            training=replace(
                config.training,
                mappo=replace(
                    config.training.mappo,
                    actor_ratio_mode="cross_agent",
                ),
            ),
        )
        with self.assertRaisesRegex(ConfigError, "actor_ratio_mode"):
            invalid.validate()

    def test_old_checkpoint_defaults_to_joint_and_mode_switch_needs_new_run(self) -> None:
        joint = _rl_config()
        self.assertIsNone(
            validate_mappo_checkpoint_resume_compatibility(
                joint,
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                method_id=joint.method_id,
                git_commit=joint.git_commit,
                config_hash=joint.config_hash,
                training_device="cpu",
                cuda_available=False,
                checkpoint_actor_ratio_mode=None,
            )
        )
        branch_specific = _rl_config(ActorRatioMode.BRANCH_SPECIFIC)
        with self.assertRaisesRegex(ConfigError, "requires a new run"):
            validate_mappo_checkpoint_resume_compatibility(
                branch_specific,
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                method_id=branch_specific.method_id,
                git_commit=branch_specific.git_commit,
                config_hash=joint.config_hash,
                training_device="cpu",
                cuda_available=False,
                checkpoint_actor_ratio_mode=None,
            )

    def test_checkpoint_v1_payload_guard_uses_snapshot_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = make_checkpoint_config(Path(directory))
            payload, _actor, _critic, _updater = make_payload(
                source,
                CHECKPOINT_KIND_PERIODIC_RESUME,
            )
            target = replace(
                source,
                training=replace(
                    source.training,
                    mappo=replace(
                        source.training.mappo,
                        actor_ratio_mode=ActorRatioMode.BRANCH_SPECIFIC,
                    ),
                ),
            )
            with self.assertRaisesRegex(CheckpointError, "requires a new run"):
                validate_periodic_checkpoint_compatibility(
                    target,
                    payload,
                    dtype=torch.float32,
                    cuda_available=False,
                )


class Fix4BranchRatioTests(unittest.TestCase):
    def test_other_branch_independence_and_route_ratio_correctness(self) -> None:
        old = torch.zeros((1, 1, 7), dtype=torch.float64)
        active = torch.zeros((1, 1, 7), dtype=torch.bool)
        active[..., 0] = True
        active[..., 6] = True
        first = old.clone()
        first[..., 0] = math.log(1.05)
        first[..., 6] = math.log(0.95)
        second = first.clone()
        second[..., 6] = math.log(1.0)
        advantage = torch.tensor([1.0], dtype=torch.float64)

        branch_first = _objective(
            first,
            old,
            active,
            advantage,
            ActorRatioMode.BRANCH_SPECIFIC,
        )
        branch_second = _objective(
            second,
            old,
            active,
            advantage,
            ActorRatioMode.BRANCH_SPECIFIC,
        )
        self.assertTrue(
            torch.equal(
                branch_first.branch_surrogate[..., 0],
                branch_second.branch_surrogate[..., 0],
            )
        )
        self.assertAlmostEqual(
            float(branch_first.branch_ratio[0, 0, 0]),
            math.exp(float(first[0, 0, 0] - old[0, 0, 0])),
            places=14,
        )

        joint_first = _objective(
            first,
            old,
            active,
            advantage,
            ActorRatioMode.JOINT,
        )
        joint_second = _objective(
            second,
            old,
            active,
            advantage,
            ActorRatioMode.JOINT,
        )
        self.assertFalse(torch.equal(joint_first.surrogate, joint_second.surrogate))

    def test_inactive_branch_has_zero_count_contribution_and_gradient(self) -> None:
        new = torch.nn.Parameter(torch.full((1, 1, 7), 0.1))
        old = torch.zeros_like(new.detach())
        active = torch.zeros((1, 1, 7), dtype=torch.bool)
        active[..., 0] = True
        output = _objective(
            new,
            old,
            active,
            torch.tensor([1.0]),
            ActorRatioMode.BRANCH_SPECIFIC,
        )
        output.actor_loss.backward()
        self.assertEqual(int(output.active_branch_count[0, 0]), 1)
        self.assertTrue(torch.equal(output.branch_surrogate[..., 1:], torch.zeros((1, 1, 6))))
        self.assertTrue(torch.equal(new.grad[..., 1:], torch.zeros((1, 1, 6))))
        self.assertNotEqual(float(new.grad[..., 0]), 0.0)

    def test_active_count_normalization_and_all_unit_ratios(self) -> None:
        new = torch.zeros((1, 1, 7))
        old = torch.zeros_like(new)
        one = torch.zeros((1, 1, 7), dtype=torch.bool)
        one[..., 0] = True
        all_branches = torch.ones((1, 1, 7), dtype=torch.bool)
        advantage = torch.tensor([2.0])
        one_output = _objective(
            new,
            old,
            one,
            advantage,
            ActorRatioMode.BRANCH_SPECIFIC,
        )
        all_output = _objective(
            new,
            old,
            all_branches,
            advantage,
            ActorRatioMode.BRANCH_SPECIFIC,
        )
        self.assertEqual(float(one_output.surrogate[0, 0]), 2.0)
        self.assertEqual(float(all_output.surrogate[0, 0]), 2.0)
        self.assertTrue(
            torch.equal(
                all_output.branch_ratio.masked_select(all_branches),
                torch.ones(7),
            )
        )
        self.assertTrue(
            torch.equal(
                all_output.branch_unclipped_surrogate.masked_select(all_branches),
                torch.full((7,), 2.0),
            )
        )

    def test_standard_ppo_clip_sign_semantics(self) -> None:
        old = torch.zeros((1, 1, 7), dtype=torch.float64)
        new = old.clone()
        new[..., 0] = math.log(2.0)
        active = torch.zeros((1, 1, 7), dtype=torch.bool)
        active[..., 0] = True
        positive = _objective(
            new,
            old,
            active,
            torch.tensor([1.0], dtype=torch.float64),
            ActorRatioMode.BRANCH_SPECIFIC,
        )
        negative = _objective(
            new,
            old,
            active,
            torch.tensor([-1.0], dtype=torch.float64),
            ActorRatioMode.BRANCH_SPECIFIC,
        )
        self.assertAlmostEqual(float(positive.branch_surrogate[0, 0, 0]), 1.2)
        self.assertAlmostEqual(float(negative.branch_surrogate[0, 0, 0]), -2.0)

    def test_real_recurrent_axes_are_preserved(self) -> None:
        config = _rl_config(ActorRatioMode.BRANCH_SPECIFIC)
        agents = 2
        old_branch = torch.zeros((8, 32, agents, 7))
        new_branch = old_branch.clone()
        active = torch.zeros((8, 32, agents, 7), dtype=torch.bool)
        active[..., 0] = True
        output = compute_configured_batched_ppo_objective_and_loss(
            new_joint_log_prob=torch.zeros((8, 32, agents)),
            old_joint_log_prob=torch.zeros((8, 32, agents)),
            advantage=torch.ones((8, 32)),
            current_value=torch.zeros((8, 32)),
            return_target=torch.zeros((8, 32)),
            entropy=torch.zeros((8, 32, agents)),
            sequence_valid_mask=torch.ones((8, 32), dtype=torch.bool),
            config=config,
            new_branch_log_probs=new_branch,
            old_branch_log_probs=old_branch,
            active_branch_indicators=active,
        )
        self.assertEqual(tuple(output.branch_ratio.shape), (8, 32, agents, 7))
        self.assertEqual(tuple(output.active_branch_count.shape), (8, 32, agents))
        self.assertEqual(tuple(output.surrogate.shape), (8, 32, agents))


class Fix4LegacyAndIsolationTests(unittest.TestCase):
    def test_joint_matches_legacy_loss_gradient_and_adam_update_exactly(self) -> None:
        actor_new = torch.nn.Parameter(
            torch.tensor([[0.05, -0.10], [0.20, 0.00]], dtype=torch.float64)
        )
        actor_reference = torch.nn.Parameter(actor_new.detach().clone())
        critic_new = torch.nn.Parameter(torch.tensor([0.2, -0.1], dtype=torch.float64))
        critic_reference = torch.nn.Parameter(critic_new.detach().clone())
        old = torch.tensor([[0.0, -0.05], [0.1, 0.0]], dtype=torch.float64)
        advantage = torch.tensor([1.0, -0.5], dtype=torch.float64)
        target = torch.tensor([0.0, 0.3], dtype=torch.float64)
        entropy = torch.tensor([[0.4, 0.3], [0.2, 0.1]], dtype=torch.float64)
        mask = torch.ones(2, dtype=torch.bool)
        output = compute_ppo_objective_and_loss(
            new_joint_log_prob=actor_new,
            old_joint_log_prob=old,
            advantage=advantage,
            current_value=critic_new,
            return_target=target,
            entropy=entropy,
            sequence_valid_mask=mask,
            epsilon_clip=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
            actor_ratio_mode=ActorRatioMode.JOINT,
        )

        log_ratio = actor_reference - old
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        expanded_advantage = advantage.reshape(2, 1).expand(2, 2)
        surrogate = torch.minimum(
            ratio * expanded_advantage,
            clipped_ratio * expanded_advantage,
        )
        reference_actor_loss = -surrogate.mean()
        reference_critic_loss = 0.5 * (critic_reference - target).square().mean()
        reference_entropy = entropy.mean()
        reference_total = (
            reference_actor_loss
            + 0.5 * reference_critic_loss
            - 0.01 * reference_entropy
        )
        self.assertTrue(torch.equal(output.ratio, ratio.detach()))
        self.assertTrue(torch.equal(output.actor_loss, reference_actor_loss.detach()))
        self.assertTrue(torch.equal(output.critic_loss, reference_critic_loss.detach()))
        self.assertTrue(torch.equal(output.entropy_mean, reference_entropy))
        self.assertEqual(output.agent_credit_mode, "team")
        self.assertEqual(
            output.diagnostics.clipped_fraction,
            float((ratio != clipped_ratio).double().mean()),
        )

        output.total_loss.backward()
        reference_total.backward()
        self.assertTrue(torch.equal(actor_new.grad, actor_reference.grad))
        self.assertTrue(torch.equal(critic_new.grad, critic_reference.grad))
        optimizer_new = torch.optim.Adam(
            [actor_new], lr=3.0e-4, betas=(0.9, 0.999), eps=1.0e-8
        )
        optimizer_reference = torch.optim.Adam(
            [actor_reference], lr=3.0e-4, betas=(0.9, 0.999), eps=1.0e-8
        )
        critic_optimizer_new = torch.optim.Adam(
            [critic_new], lr=3.0e-4, betas=(0.9, 0.999), eps=1.0e-8
        )
        critic_optimizer_reference = torch.optim.Adam(
            [critic_reference], lr=3.0e-4, betas=(0.9, 0.999), eps=1.0e-8
        )
        optimizer_new.step()
        optimizer_reference.step()
        critic_optimizer_new.step()
        critic_optimizer_reference.step()
        self.assertTrue(torch.equal(actor_new, actor_reference))
        self.assertTrue(torch.equal(critic_new, critic_reference))

    def test_branch_specific_diff_is_actor_ratio_only_and_rng_neutral(self) -> None:
        old = torch.zeros((2, 1, 7))
        new = old.clone()
        new[0, 0, 0] = math.log(1.1)
        new[0, 0, 6] = math.log(0.9)
        new[1, 0, 0] = math.log(1.05)
        new[1, 0, 6] = math.log(1.05)
        active = torch.zeros((2, 1, 7), dtype=torch.bool)
        active[..., 0] = True
        active[..., 6] = True
        advantage = torch.tensor([1.0, -1.0])
        before = torch.get_rng_state().clone()
        joint = _objective(new, old, active, advantage, ActorRatioMode.JOINT)
        branch = _objective(
            new,
            old,
            active,
            advantage,
            ActorRatioMode.BRANCH_SPECIFIC,
        )
        after = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(joint.critic_loss, branch.critic_loss))
        self.assertTrue(torch.equal(joint.entropy_mean, branch.entropy_mean))
        self.assertTrue(torch.equal(joint.expanded_advantage, branch.expanded_advantage))
        self.assertFalse(torch.equal(joint.actor_loss, branch.actor_loss))


class Fix4ProductionUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base = make_update_config()
        cls.config = replace(
            base,
            training=replace(
                base.training,
                mappo=replace(
                    base.training.mappo,
                    actor_ratio_mode=ActorRatioMode.BRANCH_SPECIFIC,
                ),
            ),
        )
        seed_transition = make_seed_transition(cls.config)
        raw = make_synthetic_buffer(cls.config, seed_transition)
        cls.buffer = align_old_policy_snapshots(cls.config, raw)

    def test_branch_specific_updater_uses_existing_rollout_fields(self) -> None:
        torch.manual_seed(904)
        actor = CAGATMAPPOActor(self.config)
        critic = MAPPOCentralizedCritic(self.config)
        updater = CAGATMAPPORecurrentPPOUpdater(
            actor,
            critic,
            self.config,
            route_telemetry_enabled=True,
        )
        output = updater.update(self.buffer)
        self.assertEqual(len(output.epoch_diagnostics), 4)
        self.assertTrue(
            all(math.isfinite(item.actor_loss) for item in output.epoch_diagnostics)
        )
        self.assertTrue(
            all(
                item.route_telemetry.actor_ratio_mode == "branch_specific"
                for item in output.epoch_diagnostics
            )
        )


class Fix4TelemetryTests(unittest.TestCase):
    def test_per_branch_statistics_use_only_active_samples(self) -> None:
        old = torch.zeros((1, 1, 2, 7))
        new = old.clone()
        new[..., 0] = math.log(1.1)
        new[..., 1] = math.log(0.9)
        active = torch.zeros((1, 1, 2, 7), dtype=torch.bool)
        active[0, 0, 0, 0] = True
        active[0, 0, 1, 1] = True
        advantage = torch.tensor([[[[2.0] * 7, [-1.0] * 7]]])
        telemetry = compute_branch_ppo_dynamics(
            old,
            new,
            active,
            advantage,
            0.2,
        )
        self.assertEqual(telemetry.groups["route"].active_count, 1)
        self.assertAlmostEqual(telemetry.groups["route"].ratio_mean, 1.1, places=6)
        self.assertEqual(telemetry.groups["tx_select"].active_count, 1)
        self.assertEqual(telemetry.groups["cpu_frequency"].active_count, 0)
        self.assertIsNone(telemetry.groups["cpu_frequency"].ratio_mean)
        self.assertIsNone(
            telemetry.groups["cpu_frequency"].surrogate_contribution_mean
        )

    def test_active_matrix_persists_once_per_rollout_timestep(self) -> None:
        config = _rl_config(ActorRatioMode.BRANCH_SPECIFIC)
        matrix = (
            ((True, False, False, False, False, False, False),),
            ((True, True, False, False, False, False, False),),
        )
        telemetry = SimpleNamespace(active_branch_matrix=matrix, schema_version=3)
        epoch = SimpleNamespace(route_telemetry=telemetry)
        update = SimpleNamespace(
            update_index=0,
            rollout_policy_version=0,
            policy_version_after_update=1,
            output=SimpleNamespace(epoch_diagnostics=(epoch,)),
        )
        training = SimpleNamespace(
            updates=(update,),
            total_environment_transitions=256,
        )
        records = _branch_activity_records(config, training, "signal-watch")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["branch_activity_matrix"], [list(matrix[0][0])])
        raw = [json.loads(line) for line in _json_lines(records).splitlines()]
        self.assertIsInstance(raw[0]["branch_activity_matrix"], list)
        csv_rows = list(csv.DictReader(io.StringIO(_csv_text(records))))
        self.assertEqual(
            json.loads(csv_rows[1]["branch_activity_matrix"]),
            [list(matrix[1][0])],
        )


if __name__ == "__main__":
    unittest.main()
