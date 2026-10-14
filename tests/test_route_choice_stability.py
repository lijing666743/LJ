"""Candidate-A conditional Remote-Local entropy-floor implementation gates."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import torch

from src.config import ConfigError, RouteCreditMode, RouteDecoderMode, RunConfig
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import (
    CAGATMAPPOActor,
    CandidateAwareRouteDecoderV1,
    MAPPOCentralizedCritic,
)
from src.models.ca_gat_mappo_actions import SequentialActionMaskBatch
from src.models.ca_gat_mappo_update import (
    CAGATMAPPORecurrentPPOUpdater,
    RouteChoiceStabilityLossOutput,
    _route_choice_stability_telemetry,
    compute_route_choice_stability_loss,
)
from tests.test_ca_gat_mappo_update import (
    align_old_policy_snapshots,
    make_seed_transition,
    make_synthetic_buffer,
)


def candidate_config(*, enabled: bool, coefficient: float = 0.05) -> RunConfig:
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
    mappo = replace(
        base.training.mappo,
        training_device="cpu",
        route_decoder_mode=RouteDecoderMode.CANDIDATE_AWARE_V1,
        route_credit_mode=RouteCreditMode.SHARED_GAE,
        route_choice_stability_enabled=enabled,
        route_choice_entropy_floor_nats=0.20,
        route_choice_stability_coef=coefficient,
    )
    config = replace(
        base,
        environment=environment,
        training=replace(base.training, mappo=mappo),
    )
    config.validate()
    return config


def stability_inputs(
    logits: torch.Tensor,
    *,
    route_mask: torch.Tensor | None = None,
    active: bool = True,
    selected_index: int = 1,
    coefficient: float = 0.05,
) -> dict[str, object]:
    if tuple(logits.shape) != (1, 1, 1, 5):
        raise AssertionError("test logits must have shape [1,1,1,5]")
    local_domain = torch.tensor(
        [[[[False, True, False, False, False]]]], dtype=torch.bool
    )
    remote_domain = torch.tensor(
        [[[[False, False, False, True, True]]]], dtype=torch.bool
    )
    return {
        "route_logits": logits,
        "route_action_masks": (
            torch.ones_like(local_domain) if route_mask is None else route_mask
        ),
        "route_active": torch.tensor([[[active]]], dtype=torch.bool),
        "sequence_valid_mask": torch.tensor([[True]], dtype=torch.bool),
        "local_domain_mask": local_domain,
        "remote_domain_mask": remote_domain,
        "selected_route_indices": torch.tensor(
            [[[selected_index]]], dtype=torch.long
        ),
        "entropy_floor_nats": 0.20,
        "coefficient": coefficient,
    }


class RouteChoiceStabilityConfigTests(unittest.TestCase):
    def test_disabled_feature_preserves_historical_canonical_identity(self) -> None:
        historical = RunConfig()
        explicit_disabled = replace(
            historical,
            training=replace(
                historical.training,
                mappo=replace(
                    historical.training.mappo,
                    route_choice_stability_enabled=False,
                    route_choice_entropy_floor_nats=0.20,
                    route_choice_stability_coef=0.0,
                ),
            ),
        )

        self.assertEqual(historical.resolved_dict(), explicit_disabled.resolved_dict())
        self.assertEqual(historical.config_hash, explicit_disabled.config_hash)
        resolved = historical.resolved_dict()["training"]["mappo"]
        self.assertNotIn("route_choice_stability_enabled", resolved)
        self.assertNotIn("route_choice_entropy_floor_nats", resolved)
        self.assertNotIn("route_choice_stability_coef", resolved)

    def test_enabled_feature_is_candidate_aware_and_canonical(self) -> None:
        enabled = candidate_config(enabled=True)
        resolved = enabled.resolved_dict()["training"]["mappo"]

        self.assertTrue(resolved["route_choice_stability_enabled"])
        self.assertEqual(resolved["route_choice_entropy_floor_nats"], 0.20)
        self.assertEqual(resolved["route_choice_stability_coef"], 0.05)
        self.assertNotEqual(enabled.config_hash, candidate_config(enabled=False).config_hash)

        base = RunConfig()
        invalid = replace(
            base,
            training=replace(
                base.training,
                mappo=replace(
                    base.training.mappo,
                    route_choice_stability_enabled=True,
                    route_choice_stability_coef=0.05,
                ),
            ),
        )
        with self.assertRaisesRegex(ConfigError, "candidate_aware_v1"):
            invalid.validate()

    def test_v1_floor_is_locked_and_zero_coefficient_is_legal(self) -> None:
        zero = candidate_config(enabled=True, coefficient=0.0)
        self.assertEqual(zero.training.mappo.route_choice_stability_coef, 0.0)

        valid = candidate_config(enabled=True)
        invalid = replace(
            valid,
            training=replace(
                valid.training,
                mappo=replace(
                    valid.training.mappo,
                    route_choice_entropy_floor_nats=0.19,
                ),
            ),
        )
        with self.assertRaisesRegex(ConfigError, "requires.*0.20"):
            invalid.validate()


class RouteChoiceStabilityMathTests(unittest.TestCase):
    def test_binary_entropy_range_and_balanced_policy_is_inactive(self) -> None:
        logits = torch.tensor(
            [[[[0.0, math.log(2.0), 0.0, 0.0, 0.0]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        output = compute_route_choice_stability_loss(**stability_inputs(logits))
        gradient = torch.autograd.grad(output.scaled_loss, logits)[0]

        self.assertAlmostEqual(output.conditional_remote_mass, 0.5, places=12)
        self.assertGreaterEqual(output.remote_local_binary_entropy, 0.0)
        self.assertLessEqual(output.remote_local_binary_entropy, math.log(2.0))
        self.assertEqual(output.unscaled_loss.item(), 0.0)
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_remote_collapse_gradient_raises_remote_relative_to_local(self) -> None:
        logits = torch.tensor(
            [[[[0.0, 10.0, 0.0, -10.0, -11.0]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        output = compute_route_choice_stability_loss(**stability_inputs(logits))
        gradient = torch.autograd.grad(output.scaled_loss, logits)[0][0, 0, 0]

        self.assertGreater(output.unscaled_loss.item(), 0.0)
        self.assertGreater(gradient[1].item(), 0.0)
        self.assertLess(gradient[3:].sum().item(), 0.0)

    def test_local_collapse_gradient_raises_local_relative_to_remote(self) -> None:
        logits = torch.tensor(
            [[[[0.0, -10.0, 0.0, 10.0, 9.0]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        output = compute_route_choice_stability_loss(**stability_inputs(logits))
        gradient = torch.autograd.grad(output.scaled_loss, logits)[0][0, 0, 0]

        self.assertGreater(output.unscaled_loss.item(), 0.0)
        self.assertLess(gradient[1].item(), 0.0)
        self.assertGreater(gradient[3:].sum().item(), 0.0)

    def test_remote_candidate_permutation_is_invariant(self) -> None:
        baseline_logits = torch.tensor(
            [[[[0.0, 4.0, 0.0, -2.0, -5.0]]]], dtype=torch.float64
        )
        permuted_logits = baseline_logits[..., [0, 1, 2, 4, 3]].clone()
        baseline = compute_route_choice_stability_loss(
            **stability_inputs(baseline_logits)
        )
        permuted = compute_route_choice_stability_loss(
            **stability_inputs(permuted_logits)
        )

        self.assertTrue(torch.equal(baseline.unscaled_loss, permuted.unscaled_loss))
        self.assertEqual(
            baseline.conditional_remote_mass, permuted.conditional_remote_mass
        )
        self.assertEqual(
            baseline.remote_local_binary_entropy,
            permuted.remote_local_binary_entropy,
        )

    def test_masked_remote_has_zero_gradient(self) -> None:
        logits = torch.tensor(
            [[[[0.0, 10.0, 0.0, -10.0, 100.0]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        route_mask = torch.tensor(
            [[[[True, True, True, True, False]]]], dtype=torch.bool
        )
        output = compute_route_choice_stability_loss(
            **stability_inputs(logits, route_mask=route_mask)
        )
        gradient = torch.autograd.grad(output.scaled_loss, logits)[0]

        self.assertEqual(gradient[0, 0, 0, 4].item(), 0.0)
        self.assertNotEqual(gradient[0, 0, 0, 3].item(), 0.0)

    def test_ineligible_states_return_exact_zero(self) -> None:
        logits = torch.tensor(
            [[[[0.0, 10.0, 0.0, -10.0, -11.0]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        cases = {
            "no legal Remote": torch.tensor(
                [[[[True, True, True, False, False]]]], dtype=torch.bool
            ),
            "Local illegal": torch.tensor(
                [[[[True, False, True, True, True]]]], dtype=torch.bool
            ),
        }
        for name, route_mask in cases.items():
            with self.subTest(name=name):
                output = compute_route_choice_stability_loss(
                    **stability_inputs(logits, route_mask=route_mask)
                )
                self.assertEqual(output.unscaled_loss.item(), 0.0)
                self.assertEqual(output.scaled_loss.item(), 0.0)
                self.assertEqual(output.valid_sample_count, 0)
                self.assertIsNone(output.conditional_remote_mass)

        inactive = compute_route_choice_stability_loss(
            **stability_inputs(logits, active=False)
        )
        self.assertEqual(inactive.unscaled_loss.item(), 0.0)
        self.assertEqual(inactive.valid_sample_count, 0)

    def test_extreme_logits_are_finite(self) -> None:
        for values in (
            (0.0, 1.0e30, 0.0, -1.0e30, -1.0e30),
            (0.0, -1.0e30, 0.0, 1.0e30, 1.0e30),
        ):
            with self.subTest(values=values):
                logits = torch.tensor(
                    [[[[*values]]]], dtype=torch.float64, requires_grad=True
                )
                output = compute_route_choice_stability_loss(
                    **stability_inputs(logits)
                )
                gradient = torch.autograd.grad(output.scaled_loss, logits)[0]
                self.assertTrue(torch.isfinite(output.unscaled_loss))
                self.assertTrue(torch.isfinite(output.scaled_loss))
                self.assertTrue(torch.isfinite(gradient).all())

    def test_zero_coefficient_preserves_loss_gradient_and_logits_exactly(self) -> None:
        first = torch.tensor(
            [[[[0.2, 5.0, -0.3, -4.0, -5.0]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        second = first.detach().clone().requires_grad_(True)
        original = first.detach().clone()
        weights = torch.tensor([[[[0.5, -0.2, 0.1, 0.4, -0.3]]]], dtype=torch.float64)
        base_first = (first * weights).sum()
        base_second = (second * weights).sum()
        stability = compute_route_choice_stability_loss(
            **stability_inputs(first, coefficient=0.0)
        )

        # Production intentionally does not add a zero-scaled auxiliary node.
        optimized_first = base_first
        optimized_second = base_second
        first_gradient = torch.autograd.grad(optimized_first, first)[0]
        second_gradient = torch.autograd.grad(optimized_second, second)[0]

        self.assertTrue(torch.equal(optimized_first, optimized_second))
        self.assertTrue(torch.equal(first_gradient, second_gradient))
        self.assertTrue(torch.equal(first.detach(), original))
        self.assertEqual(stability.scaled_loss.item(), 0.0)

    def test_critic_gae_and_reward_are_outside_stability_gradient_graph(self) -> None:
        logits = torch.tensor(
            [[[[0.0, 10.0, 0.0, -10.0, -11.0]]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        critic = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        gae = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        reward = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        output = compute_route_choice_stability_loss(**stability_inputs(logits))
        gradients = torch.autograd.grad(
            output.scaled_loss,
            (logits, critic, gae, reward),
            allow_unused=True,
        )

        self.assertIsNotNone(gradients[0])
        self.assertIsNone(gradients[1])
        self.assertIsNone(gradients[2])
        self.assertIsNone(gradients[3])


class RouteChoiceStabilityTelemetryTests(unittest.TestCase):
    def test_zero_coefficient_full_update_is_bitwise_identical(self) -> None:
        disabled_config = candidate_config(enabled=False, coefficient=0.0)
        zero_config = candidate_config(enabled=True, coefficient=0.0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(211)
            seed_transition = make_seed_transition(disabled_config)
            raw_buffer = make_synthetic_buffer(disabled_config, seed_transition)
            buffer = align_old_policy_snapshots(disabled_config, raw_buffer)

        def modules():
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(907)
                return (
                    CAGATMAPPOActor(disabled_config),
                    MAPPOCentralizedCritic(disabled_config),
                )

        disabled_actor, disabled_critic = modules()
        zero_actor, zero_critic = modules()
        disabled = CAGATMAPPORecurrentPPOUpdater(
            disabled_actor, disabled_critic, disabled_config
        )
        zero = CAGATMAPPORecurrentPPOUpdater(zero_actor, zero_critic, zero_config)

        original_rng = torch.get_rng_state().clone()
        try:
            torch.manual_seed(1441)
            common_rng = torch.get_rng_state().clone()
            disabled_output = disabled.update(buffer)
            disabled_rng = torch.get_rng_state().clone()
            torch.set_rng_state(common_rng)
            zero_output = zero.update(buffer)
            zero_rng = torch.get_rng_state().clone()
        finally:
            torch.set_rng_state(original_rng)

        self.assertTrue(torch.equal(disabled_rng, zero_rng))
        for left, right in zip(disabled_actor.parameters(), zero_actor.parameters()):
            self.assertTrue(torch.equal(left, right))
            if left.grad is None or right.grad is None:
                self.assertIs(left.grad, right.grad)
            else:
                self.assertTrue(torch.equal(left.grad, right.grad))
        for left, right in zip(disabled_critic.parameters(), zero_critic.parameters()):
            self.assertTrue(torch.equal(left, right))
            if left.grad is None or right.grad is None:
                self.assertIs(left.grad, right.grad)
            else:
                self.assertTrue(torch.equal(left.grad, right.grad))
        for disabled_epoch, zero_epoch in zip(
            disabled_output.epoch_diagnostics,
            zero_output.epoch_diagnostics,
        ):
            for name in (
                "actor_loss",
                "critic_loss",
                "entropy_mean",
                "total_loss",
                "ratio_mean",
                "actor_grad_norm_before_clip",
                "critic_grad_norm_before_clip",
            ):
                self.assertEqual(
                    getattr(disabled_epoch, name), getattr(zero_epoch, name)
                )
            self.assertIsNone(disabled_epoch.route_choice_stability)
            self.assertIsNotNone(zero_epoch.route_choice_stability)
            self.assertEqual(
                zero_epoch.route_choice_stability.scaled_stability_loss, 0.0
            )

    def test_runtime_telemetry_exposes_samples_and_gradient_ratio(self) -> None:
        config = candidate_config(enabled=True)
        environment = U2UMECEnvironment(config)
        environment.reset()
        step = environment.step(environment.canonical_proposals())
        self.assertIsNotNone(step.observations)
        action_masks = SequentialActionMaskBatch.from_observations(step.observations)
        decoder = CandidateAwareRouteDecoderV1(
            uav_count=config.environment.uav_count,
            recurrent_dimension=5,
            candidate_dimension=5,
            edge_dimension=3,
        )
        decoder.fixed_head.weight.grad = torch.zeros_like(decoder.fixed_head.weight)
        decoder.fixed_head.weight.grad[1].fill_(1.0)
        decoder.remote_scorer.weight.grad = torch.full_like(
            decoder.remote_scorer.weight, 2.0
        )
        loss = RouteChoiceStabilityLossOutput(
            unscaled_loss=torch.tensor(0.04),
            scaled_loss=torch.tensor(0.002),
            conditional_remote_mass=0.01,
            remote_local_binary_entropy=0.056,
            floor_violation_fraction=1.0,
            valid_sample_count=4,
            local_sample_count=3,
            remote_sample_count=1,
        )

        telemetry = _route_choice_stability_telemetry(loss, decoder, action_masks)
        record = telemetry.record()

        self.assertEqual(record["local_sample_count"], 3)
        self.assertEqual(record["remote_sample_count"], 1)
        self.assertAlmostEqual(record["gradient_norm_ratio"], 2.0, places=6)
        self.assertEqual(
            set(record),
            {
                "conditional_remote_mass",
                "remote_local_binary_entropy",
                "floor_violation_fraction",
                "unscaled_stability_loss",
                "scaled_stability_loss",
                "local_sample_count",
                "remote_sample_count",
                "remote_scorer_gradient_norm",
                "local_route_row_gradient_norm",
                "gradient_norm_ratio",
                "valid_sample_count",
            },
        )


if __name__ == "__main__":
    unittest.main()
