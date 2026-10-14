"""E3 single-variable PPO aggregation and identity gates."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from src.config import (
    ActorRatioMode,
    ConfigError,
    PolicyLossAggregationMode,
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_SCHEMA_VERSION,
    load_run_config,
    validate_mappo_checkpoint_resume_compatibility,
)
from src.models.ca_gat_mappo_ppo import compute_ppo_objective_and_loss
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import ACTION_BRANCH_ORDER, ActorObservationTensorizer, CAGATMAPPOActor, MAPPOCentralizedCritic
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution, SequentialActionMaskBatch
from src.models.ca_gat_mappo_update import CAGATMAPPORecurrentPPOUpdater
from tests.test_ca_gat_mappo_action_distribution import make_distribution_config
from tests.test_ca_gat_mappo_update import (
    align_old_policy_snapshots,
    make_seed_transition,
    make_synthetic_buffer,
    make_update_config,
)


_CONFIG_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "configs"
_E2 = _CONFIG_DIR / "cooperative-load-v2-hierarchical-optionaware-v1-hierarchical-tx-v1-remote-stress-v2-32k.json"
_E3 = _CONFIG_DIR / "cooperative-load-v2-hierarchical-optionaware-v1-hierarchical-tx-v1-remote-stress-v2-branchwise-32k.json"


def _objective(
    new: torch.Tensor,
    active: torch.Tensor,
    advantage: torch.Tensor,
    mode: PolicyLossAggregationMode,
    *,
    entropy: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
):
    time_steps, agents, _ = new.shape
    old = torch.zeros_like(new)
    return compute_ppo_objective_and_loss(
        new_joint_log_prob=new.masked_fill(~active, 0.0).sum(-1),
        old_joint_log_prob=torch.zeros((time_steps, agents), dtype=new.dtype),
        advantage=advantage,
        current_value=torch.zeros((time_steps, agents), dtype=new.dtype),
        return_target=torch.zeros((time_steps, agents), dtype=new.dtype),
        entropy=(
            torch.zeros((time_steps, agents), dtype=new.dtype)
            if entropy is None
            else entropy
        ),
        sequence_valid_mask=(
            torch.ones(time_steps, dtype=torch.bool) if valid is None else valid
        ),
        epsilon_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        actor_ratio_mode=ActorRatioMode.BRANCH_SPECIFIC,
        agent_credit_mode="role_decomposed",
        policy_loss_aggregation_mode=mode,
        new_branch_log_probs=new,
        old_branch_log_probs=old,
        active_branch_indicators=active,
    )


class E3BranchwiseAggregationTests(unittest.TestCase):
    def test_equal_tx_upper_samples_have_equal_new_weight_despite_two_vs_five_branches(self) -> None:
        active = torch.zeros((2, 1, 7), dtype=torch.bool)
        active[0, 0, [1, 5]] = True
        active[1, 0, [1, 2, 3, 4, 5]] = True
        new = torch.zeros((2, 1, 7), dtype=torch.float64, requires_grad=True)
        advantage = torch.ones((2, 1), dtype=torch.float64)
        legacy = _objective(new, active, advantage, PolicyLossAggregationMode.ACTIVE_BRANCH_MEAN_V1)
        old_grad = torch.autograd.grad(legacy.actor_loss, new, retain_graph=True)[0]
        branchwise = _objective(new, active, advantage, PolicyLossAggregationMode.BRANCHWISE_ACTIVE_MEAN_V1)
        new_grad = torch.autograd.grad(branchwise.actor_loss, new)[0]
        self.assertAlmostEqual(old_grad[0, 0, 1].item() / old_grad[1, 0, 1].item(), 2.5)
        self.assertAlmostEqual(new_grad[0, 0, 1].item(), new_grad[1, 0, 1].item())
        self.assertEqual(new_grad[0, 0, 2].item(), 0.0)

    def test_branchwise_clip_and_inactive_gradient(self) -> None:
        active = torch.zeros((2, 1, 7), dtype=torch.bool)
        active[..., 1] = True
        new = torch.zeros((2, 1, 7), dtype=torch.float64)
        new[0, 0, 1] = math.log(1.5)
        new[1, 0, 1] = math.log(0.5)
        new.requires_grad_()
        advantage = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
        result = _objective(new, active, advantage, PolicyLossAggregationMode.BRANCHWISE_ACTIVE_MEAN_V1)
        self.assertAlmostEqual(result.actor_loss.item(), -0.2)
        gradient = torch.autograd.grad(result.actor_loss, new)[0]
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_entropy_unchanged_and_padding_excluded(self) -> None:
        active = torch.ones((3, 1, 7), dtype=torch.bool)
        active[0, 0, 2:] = False
        new = torch.zeros((3, 1, 7), dtype=torch.float64, requires_grad=True)
        entropy = torch.tensor([[1.0], [2.0], [100.0]], dtype=torch.float64)
        valid = torch.tensor([True, True, False])
        advantage = torch.tensor([[1.0], [1.0], [100.0]], dtype=torch.float64)
        legacy = _objective(new, active, advantage, PolicyLossAggregationMode.ACTIVE_BRANCH_MEAN_V1, entropy=entropy, valid=valid)
        branchwise = _objective(new, active, advantage, PolicyLossAggregationMode.BRANCHWISE_ACTIVE_MEAN_V1, entropy=entropy, valid=valid)
        self.assertAlmostEqual(legacy.entropy_mean.item(), 1.5)
        self.assertAlmostEqual(branchwise.entropy_mean.item(), 1.5)
        self.assertAlmostEqual(branchwise.actor_loss.item(), -1.0)
        self.assertTrue(torch.isfinite(branchwise.total_loss))
        gradient = torch.autograd.grad(branchwise.actor_loss, new)[0]
        self.assertTrue(torch.equal(gradient[2], torch.zeros_like(gradient[2])))

    def test_default_identity_and_e3_config_differ_by_one_field(self) -> None:
        e2 = load_run_config(_E2)
        e3 = load_run_config(_E3)
        self.assertEqual(e2.config_hash[:12], "3f22fbf9f81c")
        self.assertNotIn("policy_loss_aggregation_mode", e2.resolved_dict()["training"]["mappo"])
        old = e2.resolved_dict()
        new = e3.resolved_dict()
        self.assertEqual(new["training"]["mappo"].pop("policy_loss_aggregation_mode"), "branchwise_active_mean_v1")
        self.assertEqual(old, new)
        self.assertEqual(e2.training.mappo.tx_decoder_mode, e3.training.mappo.tx_decoder_mode)
        self.assertEqual(e2.training.mappo.route_decoder_mode, e3.training.mappo.route_decoder_mode)

    def test_joint_ratio_rejects_branchwise_aggregation(self) -> None:
        with self.assertRaisesRegex(ConfigError, "branch_specific"):
            load_run_config(_E3, cli_overrides={"training.mappo.actor_ratio_mode": "joint"})

    def test_e2_periodic_checkpoint_cannot_resume_as_e3(self) -> None:
        raw_e2 = load_run_config(_E2)
        raw_e3 = load_run_config(_E3)
        e2 = replace(
            raw_e2,
            training=replace(
                raw_e2.training,
                mappo=replace(raw_e2.training.mappo, trajectory_credit_telemetry_enabled=False),
            ),
        )
        e3 = replace(
            raw_e3,
            training=replace(
                raw_e3.training,
                mappo=replace(raw_e3.training.mappo, trajectory_credit_telemetry_enabled=False),
            ),
        )
        with self.assertRaisesRegex(ConfigError, "config_hash mismatch"):
            validate_mappo_checkpoint_resume_compatibility(
                e3,
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                method_id=e3.method_id,
                git_commit=e3.git_commit,
                config_hash=e2.config_hash,
                training_device="cuda",
                cuda_available=True,
                checkpoint_actor_ratio_mode="branch_specific",
                checkpoint_agent_credit_mode="role_decomposed",
                checkpoint_route_decoder_mode="hierarchical_option_aware_v1",
                checkpoint_tx_decoder_mode="hierarchical_candidate_aware_v1",
                checkpoint_route_credit_mode="shared_gae",
            )

    def test_deterministic_action_and_log_probability_recompute_are_mode_neutral(self) -> None:
        base = make_distribution_config()
        legacy = replace(
            base,
            training=replace(
                base.training,
                mappo=replace(base.training.mappo, actor_ratio_mode=ActorRatioMode.BRANCH_SPECIFIC),
            ),
        )
        branchwise = replace(
            legacy,
            training=replace(
                legacy.training,
                mappo=replace(
                    legacy.training.mappo,
                    policy_loss_aggregation_mode=PolicyLossAggregationMode.BRANCHWISE_ACTIVE_MEAN_V1,
                ),
            ),
        )
        legacy.validate()
        branchwise.validate()
        environment = U2UMECEnvironment(legacy)
        reset = environment.reset()
        batch = ActorObservationTensorizer(legacy).encode_step(
            reset.observations, environment.canonical_proposals()
        )
        masks = SequentialActionMaskBatch.from_observations(reset.observations)
        old_actor = CAGATMAPPOActor(legacy).eval()
        new_actor = CAGATMAPPOActor(branchwise).eval()
        self.assertEqual(
            {key: tuple(value.shape) for key, value in old_actor.state_dict().items()},
            {key: tuple(value.shape) for key, value in new_actor.state_dict().items()},
        )
        new_actor.load_state_dict(old_actor.state_dict(), strict=True)
        with torch.no_grad():
            old_dist = CAGATMAPPOActionDistribution(old_actor, legacy)
            new_dist = CAGATMAPPOActionDistribution(new_actor, branchwise)
            old_output = old_dist.deterministic_actions(batch, masks)
            new_output = new_dist.deterministic_actions(batch, masks)
            proposals = old_output.proposals
            recomputed = new_dist.evaluate_actions(batch, masks, proposals)
        self.assertEqual(old_output.proposals, new_output.proposals)
        torch.testing.assert_close(old_output.joint_log_prob, recomputed.joint_log_prob)
        for branch in ACTION_BRANCH_ORDER:
            torch.testing.assert_close(
                old_output.branch_log_probs[branch],
                recomputed.branch_log_probs[branch],
            )
            self.assertTrue(torch.equal(old_output.action_masks[branch], new_output.action_masks[branch]))

    def test_recurrent_updater_accepts_branchwise_mode_with_finite_gradients(self) -> None:
        base = make_update_config()
        config = replace(
            base,
            training=replace(
                base.training,
                mappo=replace(
                    base.training.mappo,
                    actor_ratio_mode=ActorRatioMode.BRANCH_SPECIFIC,
                    policy_loss_aggregation_mode=PolicyLossAggregationMode.BRANCHWISE_ACTIVE_MEAN_V1,
                ),
            ),
        )
        config.validate()
        seed = make_seed_transition(config)
        buffer = align_old_policy_snapshots(config, make_synthetic_buffer(config, seed))
        actor = CAGATMAPPOActor(config)
        critic = MAPPOCentralizedCritic(config)
        before = {key: value.detach().clone() for key, value in actor.state_dict().items()}
        outcome = CAGATMAPPORecurrentPPOUpdater(actor, critic, config).update(buffer)
        self.assertEqual(len(outcome.epoch_diagnostics), config.training.mappo.update_epochs)
        self.assertTrue(all(math.isfinite(epoch.actor_loss) for epoch in outcome.epoch_diagnostics))
        self.assertTrue(any(not torch.equal(value, before[key]) for key, value in actor.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
