"""Focused actor-visible TX decoder and compatibility gates."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import torch

from src.config import (
    CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_SCHEMA_VERSION,
    ConfigError,
    RouteDecoderMode,
    RunConfig,
    TxDecoderMode,
    canonical_config_identity_hash,
    load_run_config,
    validate_mappo_checkpoint_resume_compatibility,
)
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import (
    ActorObservationTensorizer,
    CAGATMAPPOActor,
    HierarchicalCandidateAwareTxDecoderV1,
    stack_actor_time,
)
from src.models.ca_gat_mappo_actions import (
    CAGATMAPPOActionDistribution,
    SequentialActionMaskBatch,
)


def tx_config() -> RunConfig:
    base = RunConfig()
    environment = replace(
        base.environment,
        episode_horizon=6,
        arrival_probabilities=(1.0, 1.0, 1.0, 1.0),
        building_layout=(),
        candidate_neighbor_radius_m=2_000.0,
        velocity_std_mps=(0.0, 0.0),
        shadowing_std_db=0.0,
        csi_error_std_db=0.0,
        fixed_csi_aoi_slots=1,
    )
    mappo = replace(
        base.training.mappo,
        tx_decoder_mode=TxDecoderMode.HIERARCHICAL_CANDIDATE_AWARE_V1,
    )
    config = replace(base, environment=environment, training=replace(base.training, mappo=mappo))
    config.validate()
    return config


class HierarchicalTxDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tx_config()
        self.environment = U2UMECEnvironment(self.config)
        self.reset = self.environment.reset()
        self.tensorizer = ActorObservationTensorizer(self.config)
        self.actor = CAGATMAPPOActor(self.config)
        self.distribution = CAGATMAPPOActionDistribution(self.actor, self.config)

    def _active_tx(self):
        first = self.environment.step(self.environment.canonical_proposals())
        assert first.observations is not None
        proposals = list(self.environment.canonical_proposals())
        for source, observation in enumerate(first.observations):
            route = next(
                value
                for value, legal in zip(
                    observation.action_masks.route_domain,
                    observation.action_masks.mask_for("route", {}),
                )
                if legal and isinstance(value, int) and not isinstance(value, bool)
            )
            proposals[source] = replace(proposals[source], route=route)
        second = self.environment.step(proposals)
        assert second.observations is not None
        self.assertTrue(any(item.action_masks.tx_branch_active for item in second.observations))
        return (
            second.observations,
            self.tensorizer.encode_step(second.observations),
            SequentialActionMaskBatch.from_observations(second.observations),
        )

    def test_config_identity_and_invalid_mode(self) -> None:
        base = RunConfig()
        explicit_legacy = replace(
            base,
            training=replace(
                base.training,
                mappo=replace(base.training.mappo, tx_decoder_mode=TxDecoderMode.LEGACY),
            ),
        )
        self.assertEqual(
            canonical_config_identity_hash(base.resolved_dict()),
            canonical_config_identity_hash(explicit_legacy.resolved_dict()),
        )
        self.assertNotEqual(
            canonical_config_identity_hash(base.resolved_dict()),
            canonical_config_identity_hash(self.config.resolved_dict()),
        )
        invalid = replace(
            base,
            training=replace(
                base.training,
                mappo=replace(base.training.mappo, tx_decoder_mode="unknown"),
            ),
        )
        with self.assertRaisesRegex(ConfigError, "tx_decoder_mode"):
            invalid.validate()

    def test_legacy_tx_head_keeps_linear_path(self) -> None:
        legacy = replace(
            self.config,
            training=replace(
                self.config.training,
                mappo=replace(self.config.training.mappo, tx_decoder_mode=TxDecoderMode.LEGACY),
            ),
        )
        actor = CAGATMAPPOActor(legacy)
        self.assertIsInstance(actor.action_heads["tx_select"], torch.nn.Linear)
        batch = ActorObservationTensorizer(legacy).encode_step(self.reset.observations)
        output = actor(batch)
        torch.testing.assert_close(
            output.raw_logits["tx_select"],
            actor.action_heads["tx_select"](output.recurrent_features),
        )

    def test_short_real_rollout_has_legal_finite_actions(self) -> None:
        observations, _, _ = self._active_tx()
        mappo = replace(
            self.config.training.mappo,
            route_decoder_mode=RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1,
        )
        config = replace(self.config, training=replace(self.config.training, mappo=mappo))
        actor = CAGATMAPPOActor(config)
        tensorizer = ActorObservationTensorizer(config)
        distribution = CAGATMAPPOActionDistribution(actor, config)
        hidden = actor.initial_hidden(1)
        for slot in range(3):
            batch = tensorizer.encode_step(observations)
            masks = SequentialActionMaskBatch.from_observations(observations)
            with torch.no_grad():
                action = (
                    distribution.deterministic_actions(batch, masks, hidden)
                    if slot % 2 == 0
                    else distribution.sample_actions(batch, masks, hidden)
                )
            self.assertTrue(torch.isfinite(action.joint_log_prob).all())
            self.assertTrue(torch.isfinite(action.entropy).all())
            for source, observation in enumerate(observations):
                self.assertTrue(observation.action_masks.is_legal(action.proposal_at(0, 0, source)))
            result = self.environment.step(action.proposals[0][0])
            assert result.observations is not None
            observations = result.observations
            hidden = action.hidden_out.detach()

    def test_real_masks_and_two_stage_deterministic_choice(self) -> None:
        observations, batch, masks = self._active_tx()
        head = self.actor.action_heads["tx_select"]
        self.assertIsInstance(head, HierarchicalCandidateAwareTxDecoderV1)
        with torch.no_grad():
            head.top_level_head[-1].weight.zero_()
            head.top_level_head[-1].bias.copy_(torch.tensor((0.0, 0.5)))
            head.destination_head.weight.zero_()
            head.destination_head.bias.zero_()
            output = self.distribution.deterministic_actions(batch, masks)
        for source, observation in enumerate(observations):
            proposal = output.proposal_at(0, 0, source)
            self.assertTrue(observation.action_masks.is_legal(proposal))
            if observation.action_masks.tx_branch_active:
                self.assertIsInstance(proposal.tx_select, int)
                probability = output.probabilities["tx_select"][0, 0, source]
                self.assertGreater(float(probability[1:].sum()), float(probability[0]))
                power_mask = observation.action_masks.mask_for(
                    "power_level",
                    {
                        "tx_select": proposal.tx_select,
                        "resource_group": proposal.resource_group,
                    },
                )
                if any(power_mask[1:]):
                    self.assertGreater(proposal.power_level, 0.0)
            else:
                self.assertEqual(proposal.tx_select, "idle")
        reevaluated = self.distribution.evaluate_actions(batch, masks, output.proposals)
        torch.testing.assert_close(
            output.branch_log_probs["tx_select"],
            reevaluated.branch_log_probs["tx_select"],
        )
        torch.testing.assert_close(
            output.branch_entropies["tx_select"],
            reevaluated.branch_entropies["tx_select"],
        )

    def test_top_level_choice_can_differ_from_flat_leaf_argmax(self) -> None:
        synthetic = tuple(
            replace(
                observation,
                action_masks=replace(
                    observation.action_masks,
                    tx_select_mask=np.ones(4, dtype=np.bool_),
                    tx_branch_active=True,
                ),
            )
            for observation in self.reset.observations
        )
        batch = self.tensorizer.encode_step(synthetic)
        masks = SequentialActionMaskBatch.from_observations(synthetic)
        head = self.actor.action_heads["tx_select"]
        assert isinstance(head, HierarchicalCandidateAwareTxDecoderV1)
        with torch.no_grad():
            head.top_level_head[-1].weight.zero_()
            head.top_level_head[-1].bias.copy_(torch.tensor((0.0, 0.5)))
            head.destination_head.weight.zero_()
            head.destination_head.bias.zero_()
            result = self.distribution.deterministic_actions(batch, masks)
        for source in range(4):
            probability = result.probabilities["tx_select"][0, 0, source]
            self.assertGreater(float(probability[0]), float(probability[1:].max()))
            self.assertGreater(float(probability[1:].sum()), float(probability[0]))
            self.assertNotEqual(result.proposal_at(0, 0, source).tx_select, "idle")

    def test_candidate_features_and_gradients(self) -> None:
        batch = self.tensorizer.encode_step(self.reset.observations)
        mask = torch.ones_like(batch.action_masks["tx_select"])
        modified_masks = dict(batch.action_masks)
        modified_masks["tx_select"] = mask
        batch = replace(batch, action_masks=modified_masks)
        head = self.actor.action_heads["tx_select"]
        assert isinstance(head, HierarchicalCandidateAwareTxDecoderV1)
        recurrent = torch.randn(1, 1, 4, self.actor.spec.gru_hidden_dimension)
        logits = head(recurrent, batch)
        self.assertEqual(tuple(logits.shape), (1, 1, 4, 4))
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(logits.exp().sum(dim=-1), torch.ones(1, 1, 4))
        altered_self = batch.self_features.clone()
        altered_self[..., 0, 25 + 8 * 1] += 4.0
        altered = replace(batch, self_features=altered_self)
        self.assertFalse(torch.allclose(logits[0, 0, 0], head(recurrent, altered)[0, 0, 0]))
        loss = -logits[..., 1].sum()
        loss.backward()
        for parameter in (
            head.candidate_encoder[0].weight,
            head.top_level_head[-1].weight,
            head.destination_head.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_stochastic_and_idle_only_masks(self) -> None:
        observations, batch, masks = self._active_tx()
        sample = self.distribution.sample_actions(batch, masks)
        for source, observation in enumerate(observations):
            self.assertTrue(observation.action_masks.is_legal(sample.proposal_at(0, 0, source)))
        replay = self.distribution.evaluate_actions(batch, masks, sample.proposals)
        torch.testing.assert_close(sample.branch_log_probs["tx_select"], replay.branch_log_probs["tx_select"])
        self.actor.zero_grad(set_to_none=True)
        (-replay.branch_log_probs["tx_select"].sum()).backward()
        head = self.actor.action_heads["tx_select"]
        assert isinstance(head, HierarchicalCandidateAwareTxDecoderV1)
        self.assertIsNotNone(head.top_level_head[-1].weight.grad)
        self.assertGreater(float(head.top_level_head[-1].weight.grad.abs().sum()), 0.0)
        idle_batch = self.tensorizer.encode_step(self.reset.observations)
        idle_masks = SequentialActionMaskBatch.from_observations(self.reset.observations)
        idle = self.distribution.sample_actions(idle_batch, idle_masks)
        self.assertTrue(torch.isfinite(idle.probabilities["tx_select"]).all())
        self.assertTrue(all(idle.proposal_at(0, 0, source).tx_select == "idle" for source in range(4)))

    def test_e1_route_and_e2_tx_actor_forward(self) -> None:
        mappo = replace(
            self.config.training.mappo,
            route_decoder_mode=RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1,
        )
        config = replace(self.config, training=replace(self.config.training, mappo=mappo))
        config.validate()
        actor = CAGATMAPPOActor(config)
        batch = ActorObservationTensorizer(config).encode_step(self.reset.observations)
        sequence = stack_actor_time((batch, batch)).repeat_batch(2)
        output = actor(sequence)
        self.assertEqual(tuple(output.raw_logits["route"].shape), (2, 2, 4, 6))
        self.assertEqual(tuple(output.raw_logits["tx_select"].shape), (2, 2, 4, 4))
        self.assertEqual(tuple(output.hidden_out.shape), (2, 4, 128))

    def test_same_seed_keeps_all_non_tx_initial_weights(self) -> None:
        mappo = replace(
            self.config.training.mappo,
            route_decoder_mode=RouteDecoderMode.HIERARCHICAL_OPTION_AWARE_V1,
        )
        treatment = replace(self.config, training=replace(self.config.training, mappo=mappo))
        control = replace(
            treatment,
            training=replace(
                treatment.training,
                mappo=replace(mappo, tx_decoder_mode=TxDecoderMode.LEGACY),
            ),
        )
        e1_state = CAGATMAPPOActor(control).state_dict()
        e2_state = CAGATMAPPOActor(treatment).state_dict()
        shared_keys = (key for key in e1_state if not key.startswith("action_heads.tx_select."))
        for key in shared_keys:
            with self.subTest(parameter=key):
                self.assertIn(key, e2_state)
                torch.testing.assert_close(e1_state[key], e2_state[key])

    def test_resume_rejects_tx_architecture_change(self) -> None:
        config = load_run_config(
            cli_overrides={
                "mode": "rl",
                "method_id": "ca_gat_mappo",
                "training.mappo.training_device": "cpu",
                "training.mappo.tx_decoder_mode": TxDecoderMode.HIERARCHICAL_CANDIDATE_AWARE_V1.value,
            }
        )
        arguments = dict(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
            method_id=config.method_id,
            git_commit=config.git_commit,
            config_hash=config.config_hash,
            training_device="cpu",
            cuda_available=False,
        )
        with self.assertRaisesRegex(ConfigError, "tx_decoder_mode mismatch"):
            validate_mappo_checkpoint_resume_compatibility(config, **arguments)
        validate_mappo_checkpoint_resume_compatibility(
            config,
            checkpoint_tx_decoder_mode=TxDecoderMode.HIERARCHICAL_CANDIDATE_AWARE_V1.value,
            **arguments,
        )


if __name__ == "__main__":
    unittest.main()
