"""Structural gates for Option-Aware Route Representation V1."""

from __future__ import annotations

import copy
from dataclasses import replace
import unittest

import torch

from src.config import (
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_SCHEMA_VERSION,
    ConfigError,
    RouteCreditMode,
    RouteDecoderMode,
    validate_mappo_checkpoint_resume_compatibility,
)
from src.evaluation.route_option_alignment import analyze_route_option_alignment
from src.models.ca_gat_mappo import (
    CAGATMAPPOActor,
    CandidateAwareRouteDecoderV1,
    MAPPOCentralizedCritic,
    OptionAwareRouteDecoderV1,
)
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution
from src.models.ca_gat_mappo_route_telemetry import (
    RouteDiagnosticSampleCollector,
    measure_route_head_gradients,
    reconstruct_route_diagnostic_utility,
)
from tests.test_candidate_aware_route_decoder_v1 import (
    differing_paths,
    make_decoder_config,
    route_active_fixture,
)


class OptionAwareRouteDecoderConfigGates(unittest.TestCase):
    def test_config_identity_and_shared_credit_gate(self) -> None:
        legacy = make_decoder_config(RouteDecoderMode.LEGACY)
        option = make_decoder_config(RouteDecoderMode.OPTION_AWARE_V1)
        self.assertEqual(
            differing_paths(legacy.resolved_dict(), option.resolved_dict()),
            {("training", "mappo", "route_decoder_mode")},
        )
        self.assertNotEqual(legacy.config_hash, option.config_hash)
        invalid = replace(
            option.training.mappo,
            route_credit_mode=RouteCreditMode.ROUTE_SPECIFIC_GAE,
        )
        with self.assertRaisesRegex(ConfigError, "requires.*shared_gae"):
            replace(
                option,
                training=replace(option.training, mappo=invalid),
            ).validate()

    def test_checkpoint_mode_is_strict_identity(self) -> None:
        option = replace(
            make_decoder_config(RouteDecoderMode.OPTION_AWARE_V1),
            mode="rl",
            method_id="ca_gat_mappo",
        )
        option.validate()
        validate_mappo_checkpoint_resume_compatibility(
            option,
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
            method_id=option.method_id,
            git_commit=option.git_commit,
            config_hash=option.config_hash,
            training_device="cpu",
            cuda_available=False,
            checkpoint_route_decoder_mode=RouteDecoderMode.OPTION_AWARE_V1.value,
            checkpoint_route_credit_mode=RouteCreditMode.SHARED_GAE.value,
        )
        with self.assertRaisesRegex(ConfigError, "route_decoder_mode mismatch"):
            validate_mappo_checkpoint_resume_compatibility(
                option,
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                method_id=option.method_id,
                git_commit=option.git_commit,
                config_hash=option.config_hash,
                training_device="cpu",
                cuda_available=False,
                checkpoint_route_decoder_mode=(
                    RouteDecoderMode.CANDIDATE_AWARE_V1.value
                ),
                checkpoint_route_credit_mode=RouteCreditMode.SHARED_GAE.value,
            )


class OptionAwareRouteDecoderStructuralGates(unittest.TestCase):
    def setUp(self) -> None:
        config = make_decoder_config(RouteDecoderMode.OPTION_AWARE_V1)
        actor = CAGATMAPPOActor(config)
        decoder = actor.route_decoder
        assert isinstance(decoder, OptionAwareRouteDecoderV1)
        self.decoder = decoder
        self.spec = actor.spec
        shape = (1, 2, self.spec.uav_count)
        self.recurrent = torch.randn(*shape, self.spec.gru_hidden_dimension)
        self.cooperative = torch.randn(*shape, self.spec.encoder_hidden_dimension)
        self.local_endpoint = torch.randn(
            *shape, self.spec.encoder_hidden_dimension
        )
        self.candidate_endpoint = torch.randn(
            *shape,
            self.spec.uav_count,
            self.spec.encoder_hidden_dimension,
        )
        self.self_features = torch.rand(*shape, self.spec.self_feature_dim)
        self.public = torch.rand(
            *shape,
            self.spec.uav_count,
            self.spec.neighbor_public_feature_dim,
        )
        self.edge = torch.rand(
            *shape, self.spec.uav_count, self.spec.edge_feature_dim
        )

    def _forward(self, **changes):
        values = {
            "recurrent_features": self.recurrent,
            "cooperative_features": self.cooperative,
            "local_endpoint_features": self.local_endpoint,
            "candidate_endpoint_features": self.candidate_endpoint,
            "self_features": self.self_features,
            "candidate_public_features": self.public,
            "candidate_edge_features": self.edge,
        }
        values.update(changes)
        return self.decoder(**values)

    def test_shared_local_remote_scorer_and_exact_logit_wiring(self) -> None:
        logits = self._forward()
        self.assertEqual(tuple(logits.shape), (1, 2, 4, 6))
        self.assertIsInstance(self.decoder.shared_option_scorer[-1], torch.nn.Linear)
        self.assertFalse(hasattr(self.decoder, "local_head"))
        self.assertEqual(self.decoder.control_head.out_features, 2)
        self.assertEqual(
            self.decoder.remote_candidate_ids.tolist(),
            [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
        )

    def test_destination_axis_bypasses_gru_and_is_candidate_aligned(self) -> None:
        endpoints = self.candidate_endpoint.clone().requires_grad_(True)
        public = self.public.clone().requires_grad_(True)
        edge = self.edge.clone().requires_grad_(True)
        ego = 0
        remote_slot = 4
        destination = int(self.decoder.remote_candidate_ids[ego, remote_slot - 3])
        self._forward(
            candidate_endpoint_features=endpoints,
            candidate_public_features=public,
            candidate_edge_features=edge,
        )[0, 0, ego, remote_slot].backward()
        self.assertGreater(float(endpoints.grad[0, 0, ego, destination].norm()), 0.0)
        self.assertGreater(float(public.grad[0, 0, ego, destination].norm()), 0.0)
        endpoint_other = endpoints.grad.clone()
        public_other = public.grad.clone()
        endpoint_other[0, 0, ego, destination] = 0.0
        public_other[0, 0, ego, destination] = 0.0
        self.assertEqual(float(endpoint_other.abs().max()), 0.0)
        self.assertEqual(float(public_other.abs().max()), 0.0)
        self.assertGreater(float(edge.grad[0, 0, ego, destination].norm()), 0.0)

    def test_candidate_permutation_swaps_only_corresponding_remote_logits(self) -> None:
        baseline = self._forward()
        ego = 1
        remote_ids = self.decoder.remote_candidate_ids[ego].tolist()
        first, second = remote_ids[:2]
        endpoints = self.candidate_endpoint.clone()
        public = self.public.clone()
        edge = self.edge.clone()
        self_features = self.self_features.clone()
        endpoints[:, :, ego, [first, second], :] = endpoints[
            :, :, ego, [second, first], :
        ]
        public[:, :, ego, [first, second], :] = public[
            :, :, ego, [second, first], :
        ]
        edge[:, :, ego, [first, second], :] = edge[
            :, :, ego, [second, first], :
        ]
        tx_start = self.decoder.feature_extractor.tx_start
        queue_width = 8
        tx = self_features[
            ..., tx_start : tx_start + self.spec.uav_count * queue_width
        ].reshape(1, 2, 4, 4, queue_width)
        tx[:, :, ego, [first, second], :] = tx[
            :, :, ego, [second, first], :
        ].clone()
        permuted = self._forward(
            candidate_endpoint_features=endpoints,
            candidate_public_features=public,
            candidate_edge_features=edge,
            self_features=self_features,
        )
        first_slot = 3 + remote_ids.index(first)
        second_slot = 3 + remote_ids.index(second)
        self.assertTrue(torch.equal(permuted[:, :, ego, :3], baseline[:, :, ego, :3]))
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
                torch.equal(permuted[:, :, ego, slot], baseline[:, :, ego, slot])
            )

    def test_task_and_source_changes_affect_option_comparison(self) -> None:
        baseline = self._forward()
        extractor = self.decoder.feature_extractor
        task_changed = self.self_features.clone()
        task_indices = [
            extractor.unbound_start + 4,
            extractor.unbound_start + 5,
            extractor.unbound_start + 6,
        ]
        task_changed[..., task_indices] += 0.125
        task_logits = self._forward(self_features=task_changed)
        self.assertTrue(torch.equal(task_logits[..., [0, 2]], baseline[..., [0, 2]]))
        for route_index in (1, 3, 4, 5):
            self.assertGreater(
                float(
                    (task_logits[..., route_index] - baseline[..., route_index])
                    .abs()
                    .max()
                ),
                0.0,
            )

        source_changed = self.self_features.clone()
        source_indices = [
            2,
            6,
            8,
            extractor.local_start,
            extractor.local_start + 2,
        ]
        source_changed[..., source_indices] += 0.125
        source_logits = self._forward(self_features=source_changed)
        baseline_margin = baseline[..., 1] - baseline[..., 3]
        changed_margin = source_logits[..., 1] - source_logits[..., 3]
        self.assertGreater(float((changed_margin - baseline_margin).abs().max()), 0.0)

    def test_local_and_remote_scores_receive_task_source_and_burden_gradients(self) -> None:
        local_endpoint = self.local_endpoint.clone().requires_grad_(True)
        self_features = self.self_features.clone().requires_grad_(True)
        public = self.public.clone().requires_grad_(True)
        logits = self._forward(
            local_endpoint_features=local_endpoint,
            self_features=self_features,
            candidate_public_features=public,
        )
        (logits[..., 1].sum() + logits[..., 3:].sum()).backward()
        extractor = self.decoder.feature_extractor
        semantic_indices = [
            2,
            extractor.unbound_start + 4,
            extractor.unbound_start + 5,
            extractor.unbound_start + 6,
            extractor.local_start + 2,
        ]
        self.assertGreater(float(self_features.grad[..., semantic_indices].norm()), 0.0)
        self.assertGreater(float(local_endpoint.grad.norm()), 0.0)
        self.assertGreater(float(public.grad[..., 12:14].norm()), 0.0)


class OptionAwareRouteDecoderIntegrationGates(unittest.TestCase):
    def test_legacy_candidate_and_common_initialization_remain_compatible(self) -> None:
        legacy = CAGATMAPPOActor(make_decoder_config(RouteDecoderMode.LEGACY))
        candidate = CAGATMAPPOActor(
            make_decoder_config(RouteDecoderMode.CANDIDATE_AWARE_V1)
        )
        option = CAGATMAPPOActor(
            make_decoder_config(RouteDecoderMode.OPTION_AWARE_V1)
        )
        self.assertIsInstance(legacy.route_decoder, torch.nn.Linear)
        self.assertIsInstance(candidate.route_decoder, CandidateAwareRouteDecoderV1)
        self.assertIsInstance(option.route_decoder, OptionAwareRouteDecoderV1)
        legacy_common = {
            name: value
            for name, value in legacy.state_dict().items()
            if not name.startswith("action_heads.route.")
        }
        for actor in (candidate, option):
            common = {
                name: value
                for name, value in actor.state_dict().items()
                if not name.startswith("action_heads.route.")
            }
            self.assertEqual(tuple(legacy_common), tuple(common))
            for name, value in legacy_common.items():
                self.assertTrue(torch.equal(value, common[name]), name)
        legacy_critic = MAPPOCentralizedCritic(
            make_decoder_config(RouteDecoderMode.LEGACY)
        )
        option_critic = MAPPOCentralizedCritic(
            make_decoder_config(RouteDecoderMode.OPTION_AWARE_V1)
        )
        for name, value in legacy_critic.state_dict().items():
            self.assertTrue(torch.equal(value, option_critic.state_dict()[name]), name)

        restored = CAGATMAPPOActor(
            make_decoder_config(RouteDecoderMode.OPTION_AWARE_V1)
        )
        restored.load_state_dict(option.state_dict(), strict=True)
        for name, value in option.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)
        with self.assertRaises(RuntimeError):
            legacy.load_state_dict(option.state_dict(), strict=True)

    def test_actor_mask_semantics_and_option_gradient_telemetry(self) -> None:
        config, _, observations, batch, masks = route_active_fixture(
            RouteDecoderMode.OPTION_AWARE_V1
        )
        actor = CAGATMAPPOActor(config)
        distribution = CAGATMAPPOActionDistribution(actor, config)
        output = distribution.deterministic_actions(batch, masks)
        decoder = actor.route_decoder
        assert isinstance(decoder, OptionAwareRouteDecoderV1)
        for agent, observation in enumerate(observations):
            self.assertEqual(
                tuple(observation.action_masks.route_domain[3:]),
                tuple(decoder.remote_candidate_ids[agent].tolist()),
            )
            self.assertTrue(
                torch.all(
                    output.probabilities["route"][0, 0, agent][
                        ~output.action_masks["route"][0, 0, agent]
                    ]
                    == 0.0
                )
            )
        weights = torch.arange(1, 7, dtype=output.raw_logits["route"].dtype)
        (output.raw_logits["route"] * weights).mean().backward()
        telemetry = measure_route_head_gradients(
            decoder,
            [observation.action_masks.route_domain for observation in observations],
        )
        self.assertEqual(telemetry.total.status, "finite_nonzero")
        self.assertEqual(
            tuple(row.semantic_role for row in telemetry.rows),
            ("idle", "local_remote_shared", "defer"),
        )
        self.assertEqual(telemetry.rows[1].destination_uavs, tuple(range(4)))

    def test_offline_alignment_has_fixed_windows_dra_and_logit_scales(self) -> None:
        config, _, observations, batch, masks = route_active_fixture(
            RouteDecoderMode.OPTION_AWARE_V1
        )
        actor = CAGATMAPPOActor(config).eval()
        distribution = CAGATMAPPOActionDistribution(actor, config)
        with torch.no_grad():
            output = distribution.deterministic_actions(batch, masks)
        proposals = tuple(output.proposals[0][0])
        collector = RouteDiagnosticSampleCollector(config)
        records = list(
            collector.collect(
                episode_id=0,
                global_environment_step=7,
                policy_version=0,
                observations=observations,
                proposals=proposals,
                action_output=output,
            )
        )
        utility = reconstruct_route_diagnostic_utility(
            records[0], collector.schema_header()
        )
        valid = [
            item
            for item in utility["remote_by_candidate"]
            if item["status"] == "valid"
        ]
        if len(valid) >= 2:
            ordered = sorted(valid, key=lambda item: item["T_remote_total_est_s"])
            probabilities = {
                item["candidate_uav"]: float(len(ordered) - index)
                for index, item in enumerate(ordered)
            }
            total = sum(probabilities.values())
            for item in records[0]["policy"]["remote_probability_by_candidate"]:
                if item["candidate_uav"] in probabilities:
                    item["probability"] = probabilities[item["candidate_uav"]] / total
        late = copy.deepcopy(records[0])
        late["global_environment_step"] = 31_999
        analysis = analyze_route_option_alignment(
            collector.schema_header(), [records[0], late]
        )
        self.assertEqual(
            tuple(analysis["windows"]),
            (
                "overall",
                "early_0_10k",
                "middle_10k_20k",
                "late_20k_32k",
                "final_20_percent_25_6k_32k",
            ),
        )
        self.assertEqual(analysis["windows"]["overall"]["sample_count"], 2)
        if len(valid) >= 2:
            self.assertEqual(
                analysis["windows"]["overall"][
                    "destination_ranking_accuracy"
                ]["accuracy"],
                1.0,
            )
        self.assertEqual(
            analysis["windows"]["final_20_percent_25_6k_32k"]["sample_count"],
            1,
        )
        scales = analysis["windows"]["overall"]["raw_logit_scale"]
        self.assertGreater(scales["local"]["count"], 0)
        self.assertGreater(scales["defer"]["count"], 0)
        self.assertGreater(scales["legal_remote"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
