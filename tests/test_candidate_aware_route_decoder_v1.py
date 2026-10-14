"""Structural gates for Candidate-aware Route Decoder V1."""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any, Mapping

import torch

from src.config import (
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_SCHEMA_VERSION,
    ConfigError,
    RouteCreditMode,
    RouteDecoderMode,
    RunConfig,
    load_run_config,
    validate_mappo_checkpoint_resume_compatibility,
)
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import (
    ACTION_BRANCH_ORDER,
    ActorObservationTensorizer,
    CAGATMAPPOActor,
    CandidateAwareRouteDecoderV1,
    MAPPOCentralizedCritic,
)
from src.models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionMaskBatch,
)
from src.models.ca_gat_mappo_route_telemetry import (
    measure_route_head_gradients,
)
from src.models.ca_gat_mappo_update import build_ca_gat_mappo_optimizers


def make_decoder_config(mode: RouteDecoderMode) -> RunConfig:
    """Return one same-seed CPU config whose only A/B field is decoder mode."""

    base = RunConfig()
    count = base.environment.uav_count
    environment = replace(
        base.environment,
        episode_horizon=4,
        arrival_probabilities=(1.0,) * count,
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
        route_decoder_mode=mode,
        route_credit_mode=RouteCreditMode.SHARED_GAE,
    )
    config = replace(
        base,
        environment=environment,
        training=replace(base.training, mappo=mappo),
    )
    config.validate()
    return config


def route_active_fixture(mode: RouteDecoderMode):
    config = make_decoder_config(mode)
    environment = U2UMECEnvironment(config)
    environment.reset()
    first = environment.step(environment.canonical_proposals())
    assert first.observations is not None
    observations = tuple(first.observations)
    assert all(item.action_masks.route_branch_active for item in observations)
    tensorizer = ActorObservationTensorizer(config)
    batch = tensorizer.encode_step(observations, episode_start=False)
    masks = SequentialActionMaskBatch.from_observations(observations)
    return config, environment, observations, batch, masks


def differing_paths(
    left: Any,
    right: Any,
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: set[tuple[str, ...]] = set()
        for key in set(left) | set(right):
            if key not in left or key not in right:
                differences.add((*prefix, str(key)))
            else:
                differences.update(
                    differing_paths(left[key], right[key], (*prefix, str(key)))
                )
        return differences
    return set() if left == right else {prefix}


class CandidateAwareRouteDecoderConfigGates(unittest.TestCase):
    def test_same_commit_ab_configs_differ_only_by_decoder_mode(self) -> None:
        legacy = make_decoder_config(RouteDecoderMode.LEGACY)
        candidate = make_decoder_config(RouteDecoderMode.CANDIDATE_AWARE_V1)

        self.assertEqual(legacy.seed, candidate.seed, 42)
        self.assertEqual(legacy.git_commit, candidate.git_commit)
        self.assertEqual(
            legacy.training.mappo.route_credit_mode,
            RouteCreditMode.SHARED_GAE,
        )
        self.assertEqual(
            candidate.training.mappo.route_credit_mode,
            RouteCreditMode.SHARED_GAE,
        )
        self.assertEqual(
            differing_paths(legacy.resolved_dict(), candidate.resolved_dict()),
            {("training", "mappo", "route_decoder_mode")},
        )
        self.assertNotEqual(legacy.config_hash, candidate.config_hash)

    def test_cli_accepts_both_decoder_modes_without_changing_credit(self) -> None:
        for mode in RouteDecoderMode:
            with self.subTest(mode=mode.value):
                config = load_run_config(
                    cli_overrides={
                        "training.mappo.route_decoder_mode": mode.value,
                        "training.mappo.route_credit_mode": "shared_gae",
                    }
                )
                self.assertEqual(config.training.mappo.route_decoder_mode, mode)
                self.assertEqual(
                    config.training.mappo.route_credit_mode,
                    RouteCreditMode.SHARED_GAE,
                )

    def test_candidate_decoder_rejects_non_shared_credit(self) -> None:
        valid = make_decoder_config(RouteDecoderMode.CANDIDATE_AWARE_V1)
        invalid_mappo = replace(
            valid.training.mappo,
            route_credit_mode=RouteCreditMode.ROUTE_SPECIFIC_GAE,
        )
        with self.assertRaisesRegex(ConfigError, "requires.*shared_gae"):
            replace(
                valid,
                training=replace(valid.training, mappo=invalid_mappo),
            ).validate()


class CandidateAwareRouteDecoderStructuralGates(unittest.TestCase):
    def setUp(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            self.decoder = CandidateAwareRouteDecoderV1(
                uav_count=4,
                recurrent_dimension=7,
                candidate_dimension=5,
                edge_dimension=3,
            )
        self.recurrent = torch.randn(1, 2, 4, 7)
        self.public = torch.randn(1, 2, 4, 4, 5)
        self.edge = torch.randn(1, 2, 4, 4, 3)

    def test_candidate_permutation_swaps_only_corresponding_remote_logits(self) -> None:
        baseline = self.decoder(self.recurrent, self.public, self.edge)
        for ego in range(4):
            remote_ids = self.decoder.remote_candidate_ids[ego].tolist()
            first, second = remote_ids[:2]
            public = self.public.clone()
            edge = self.edge.clone()
            public[:, :, ego, [first, second], :] = public[
                :, :, ego, [second, first], :
            ]
            edge[:, :, ego, [first, second], :] = edge[
                :, :, ego, [second, first], :
            ]
            permuted = self.decoder(self.recurrent, public, edge)
            self.assertTrue(
                torch.equal(permuted[:, :, ego, :3], baseline[:, :, ego, :3])
            )
            first_slot = 3 + remote_ids.index(first)
            second_slot = 3 + remote_ids.index(second)
            self.assertTrue(
                torch.allclose(
                    permuted[:, :, ego, first_slot],
                    baseline[:, :, ego, second_slot],
                )
            )
            self.assertTrue(
                torch.allclose(
                    permuted[:, :, ego, second_slot],
                    baseline[:, :, ego, first_slot],
                )
            )
            for destination in set(remote_ids) - {first, second}:
                slot = 3 + remote_ids.index(destination)
                self.assertTrue(
                    torch.equal(
                        permuted[:, :, ego, slot], baseline[:, :, ego, slot]
                    )
                )

    def test_one_remote_logit_depends_only_on_its_wired_candidate_row(self) -> None:
        public = self.public.clone().requires_grad_(True)
        edge = self.edge.clone().requires_grad_(True)
        ego = 0
        destination = int(self.decoder.remote_candidate_ids[ego, 1])
        slot = 3 + self.decoder.remote_candidate_ids[ego].tolist().index(destination)
        logits = self.decoder(self.recurrent, public, edge)
        logits[0, 0, ego, slot].backward()

        self.assertGreater(float(public.grad[0, 0, ego, destination].norm()), 0.0)
        self.assertGreater(float(edge.grad[0, 0, ego, destination].norm()), 0.0)
        public_other = public.grad.clone()
        edge_other = edge.grad.clone()
        public_other[0, 0, ego, destination] = 0.0
        edge_other[0, 0, ego, destination] = 0.0
        self.assertEqual(float(public_other.abs().max()), 0.0)
        self.assertEqual(float(edge_other.abs().max()), 0.0)

    def test_fixed_logits_do_not_depend_on_candidate_inputs(self) -> None:
        public = self.public.clone().requires_grad_(True)
        edge = self.edge.clone().requires_grad_(True)
        logits = self.decoder(self.recurrent, public, edge)
        logits[..., :3].sum().backward()
        self.assertEqual(float(public.grad.abs().max()), 0.0)
        self.assertEqual(float(edge.grad.abs().max()), 0.0)


class CandidateAwareRouteDecoderIntegrationGates(unittest.TestCase):
    def test_legacy_forward_and_state_layout_are_unchanged(self) -> None:
        config, _, _, batch, _ = route_active_fixture(RouteDecoderMode.LEGACY)
        actor = CAGATMAPPOActor(config).eval()
        with torch.no_grad():
            output = actor(batch)
            expected = actor.action_heads["route"](output.recurrent_features)
        self.assertIsInstance(actor.action_heads["route"], torch.nn.Linear)
        self.assertEqual(
            set(name for name in actor.state_dict() if name.startswith("action_heads.route")),
            {"action_heads.route.weight", "action_heads.route.bias"},
        )
        self.assertTrue(torch.equal(output.raw_logits["route"], expected))

    def test_same_seed_common_modules_and_critic_initialize_identically(self) -> None:
        legacy_config = make_decoder_config(RouteDecoderMode.LEGACY)
        candidate_config = make_decoder_config(RouteDecoderMode.CANDIDATE_AWARE_V1)
        legacy = CAGATMAPPOActor(legacy_config)
        candidate = CAGATMAPPOActor(candidate_config)
        legacy_common = {
            name: tensor
            for name, tensor in legacy.state_dict().items()
            if not name.startswith("action_heads.route.")
        }
        candidate_common = {
            name: tensor
            for name, tensor in candidate.state_dict().items()
            if not name.startswith("action_heads.route.")
        }
        self.assertEqual(tuple(legacy_common), tuple(candidate_common))
        for name in legacy_common:
            self.assertTrue(torch.equal(legacy_common[name], candidate_common[name]), name)

        legacy_critic = MAPPOCentralizedCritic(legacy_config)
        candidate_critic = MAPPOCentralizedCritic(candidate_config)
        for name, tensor in legacy_critic.state_dict().items():
            self.assertTrue(
                torch.equal(tensor, candidate_critic.state_dict()[name]), name
            )

    def test_full_actor_route_branch_is_permutation_equivariant(self) -> None:
        config, _, _, batch, _ = route_active_fixture(
            RouteDecoderMode.CANDIDATE_AWARE_V1
        )
        actor = CAGATMAPPOActor(config).eval()
        with torch.no_grad():
            baseline = actor(batch).raw_logits["route"]
        decoder = actor.route_decoder
        assert isinstance(decoder, CandidateAwareRouteDecoderV1)

        for ego in range(config.environment.uav_count):
            remote_ids = decoder.remote_candidate_ids[ego].tolist()
            first, second = remote_ids[:2]
            public = batch.neighbor_public_features.clone()
            edge = batch.edge_features.clone()
            neighbor_mask = batch.neighbor_mask.clone()
            public[:, :, ego, [first, second], :] = public[
                :, :, ego, [second, first], :
            ]
            edge[:, :, ego, [first, second], :] = edge[
                :, :, ego, [second, first], :
            ]
            neighbor_mask[:, :, ego, [first, second]] = neighbor_mask[
                :, :, ego, [second, first]
            ]
            permuted_batch = replace(
                batch,
                neighbor_public_features=public,
                edge_features=edge,
                neighbor_mask=neighbor_mask,
            )
            with torch.no_grad():
                permuted = actor(permuted_batch).raw_logits["route"]
            self.assertTrue(
                torch.allclose(
                    permuted[:, :, ego, :3], baseline[:, :, ego, :3], atol=1.0e-6
                )
            )
            first_slot = 3 + remote_ids.index(first)
            second_slot = 3 + remote_ids.index(second)
            self.assertTrue(
                torch.allclose(
                    permuted[:, :, ego, first_slot],
                    baseline[:, :, ego, second_slot],
                    atol=1.0e-6,
                )
            )
            self.assertTrue(
                torch.allclose(
                    permuted[:, :, ego, second_slot],
                    baseline[:, :, ego, first_slot],
                    atol=1.0e-6,
                )
            )

    def test_mask_action_semantics_and_ppo_reevaluation_are_exact(self) -> None:
        config, _, observations, batch, masks = route_active_fixture(
            RouteDecoderMode.CANDIDATE_AWARE_V1
        )
        actor = CAGATMAPPOActor(config).eval()
        distribution = CAGATMAPPOActionDistribution(actor, config)
        with torch.no_grad():
            sampled = distribution.sample_actions(batch, masks)
            evaluated = distribution.evaluate_actions(
                batch, masks, sampled.proposals
            )

        decoder = actor.route_decoder
        self.assertIsInstance(decoder, CandidateAwareRouteDecoderV1)
        for agent, observation in enumerate(observations):
            self.assertEqual(
                tuple(observation.action_masks.route_domain[3:]),
                tuple(decoder.remote_candidate_ids[agent].tolist()),
            )
            self.assertTrue(
                observation.action_masks.is_legal(sampled.proposal_at(0, 0, agent))
            )
        for branch in ACTION_BRANCH_ORDER:
            self.assertTrue(
                torch.equal(sampled.action_masks[branch], evaluated.action_masks[branch])
            )
            self.assertTrue(
                torch.equal(sampled.action_indices[branch], evaluated.action_indices[branch])
            )
            self.assertTrue(
                torch.allclose(
                    sampled.raw_logits[branch], evaluated.raw_logits[branch]
                )
            )
            self.assertTrue(
                torch.allclose(
                    sampled.branch_log_probs[branch],
                    evaluated.branch_log_probs[branch],
                )
            )
            self.assertTrue(
                torch.all(sampled.probabilities[branch][~sampled.action_masks[branch]] == 0.0)
            )
        self.assertTrue(torch.allclose(sampled.joint_log_prob, evaluated.joint_log_prob))

    def test_candidate_decoder_gradients_and_optimizer_wiring_are_complete(self) -> None:
        config, _, observations, batch, _ = route_active_fixture(
            RouteDecoderMode.CANDIDATE_AWARE_V1
        )
        actor = CAGATMAPPOActor(config)
        critic = MAPPOCentralizedCritic(config)
        output = actor(batch)
        weights = torch.arange(
            1,
            output.raw_logits["route"].shape[-1] + 1,
            dtype=output.raw_logits["route"].dtype,
        )
        (output.raw_logits["route"] * weights).mean().backward()
        decoder = actor.route_decoder
        assert isinstance(decoder, CandidateAwareRouteDecoderV1)
        for name, parameter in decoder.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            assert parameter.grad is not None
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.norm()), 0.0, name)

        telemetry = measure_route_head_gradients(
            decoder,
            [observation.action_masks.route_domain for observation in observations],
        )
        self.assertEqual(telemetry.total.status, "finite_nonzero")
        self.assertEqual(
            tuple(row.semantic_role for row in telemetry.rows),
            ("idle", "local", "defer", "remote"),
        )
        self.assertEqual(telemetry.rows[-1].destination_uavs, tuple(range(4)))

        optimizers = build_ca_gat_mappo_optimizers(actor, critic, config)
        optimized_ids = {id(parameter) for parameter in optimizers.actor_parameters}
        self.assertTrue(
            {id(parameter) for parameter in decoder.parameters()} <= optimized_ids
        )

    def test_checkpoint_mode_policy_and_strict_state_lifecycle(self) -> None:
        candidate = replace(
            make_decoder_config(RouteDecoderMode.CANDIDATE_AWARE_V1),
            mode="rl",
            method_id="ca_gat_mappo",
        )
        candidate.validate()
        validate_mappo_checkpoint_resume_compatibility(
            candidate,
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
            method_id=candidate.method_id,
            git_commit=candidate.git_commit,
            config_hash=candidate.config_hash,
            training_device="cpu",
            cuda_available=False,
            checkpoint_route_decoder_mode=RouteDecoderMode.CANDIDATE_AWARE_V1.value,
            checkpoint_route_credit_mode=RouteCreditMode.SHARED_GAE.value,
        )
        with self.assertRaisesRegex(ConfigError, "route_decoder_mode mismatch"):
            validate_mappo_checkpoint_resume_compatibility(
                candidate,
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                method_id=candidate.method_id,
                git_commit=candidate.git_commit,
                config_hash=candidate.config_hash,
                training_device="cpu",
                cuda_available=False,
                checkpoint_route_decoder_mode=RouteDecoderMode.LEGACY.value,
                checkpoint_route_credit_mode=RouteCreditMode.SHARED_GAE.value,
            )

        source = CAGATMAPPOActor(candidate)
        restored = CAGATMAPPOActor(candidate)
        restored.load_state_dict(source.state_dict(), strict=True)
        for name, tensor in source.state_dict().items():
            self.assertTrue(torch.equal(tensor, restored.state_dict()[name]), name)
        legacy = CAGATMAPPOActor(make_decoder_config(RouteDecoderMode.LEGACY))
        with self.assertRaises(RuntimeError):
            legacy.load_state_dict(source.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
