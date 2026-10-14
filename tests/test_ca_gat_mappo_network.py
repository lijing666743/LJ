"""Stage 07B-1 tests for CA-GAT-MAPPO networks and tensor interfaces."""

from __future__ import annotations

import inspect
import unittest
from dataclasses import replace

import numpy as np
import torch

from src.config import RunConfig
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import (
    ACTION_BRANCH_ORDER,
    ActorObservationTensorizer,
    CAGATMAPPOActor,
    CentralizedStateTensorizer,
    MAPPOCentralizedCritic,
    MAPPONetworkError,
    MAPPOTensorSpec,
    stack_actor_time,
    stack_centralized_time,
)
from src.policies.heuristic_policy import HeuristicPolicy
from src.policies.random_policy import RandomPolicy


def make_network_config() -> RunConfig:
    """Return a small real-environment config for network contract tests."""

    base = RunConfig()
    environment = replace(
        base.environment,
        episode_horizon=3,
        arrival_probabilities=(1.0, 0.0, 0.0, 0.0),
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


class CAGATMAPPOContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_network_config()
        self.environment = U2UMECEnvironment(self.config)
        self.reset = self.environment.reset()
        self.proposals = self.environment.canonical_proposals()
        self.actor_tensorizer = ActorObservationTensorizer(self.config)
        self.critic_tensorizer = CentralizedStateTensorizer(self.config)
        self.actor_batch = self.actor_tensorizer.encode_step(
            self.reset.observations,
            self.proposals,
        )
        self.critic_batch = self.critic_tensorizer.encode_step(
            self.reset.centralized_state
        )

    def test_frozen_dimensions_and_real_mask_contract(self) -> None:
        spec = MAPPOTensorSpec.from_config(self.config)
        count = self.config.environment.uav_count

        self.assertEqual(spec.encoder_hidden_dimension, 128)
        self.assertEqual(spec.gat_layer_count, 1)
        self.assertEqual(spec.attention_head_count, 4)
        self.assertEqual(spec.gru_hidden_dimension, 128)
        self.assertEqual(
            dict(spec.action_dimensions),
            {
                "route": count + 2,
                "tx_select": count,
                "resource_group": 6,
                "resource_width": 2,
                "power_level": 4,
                "cpu_queue": count + 1,
                "cpu_frequency": 4,
            },
        )
        self.actor_batch.validate(spec)
        np.testing.assert_array_equal(
            self.actor_batch.neighbor_mask[0, 0].cpu().numpy(),
            np.stack(
                [
                    observation.candidate_neighbor_mask
                    for observation in self.reset.observations
                ]
            ),
        )
        for agent, (observation, proposal) in enumerate(
            zip(self.reset.observations, self.proposals)
        ):
            for branch in ACTION_BRANCH_ORDER:
                np.testing.assert_array_equal(
                    self.actor_batch.action_masks[branch][0, 0, agent]
                    .cpu()
                    .numpy(),
                    observation.action_masks.mask_for(branch, proposal),
                )

    def test_actor_forward_has_exactly_seven_finite_raw_logit_heads(self) -> None:
        actor = CAGATMAPPOActor(self.config).eval()
        hidden = actor.initial_hidden(1)
        with torch.no_grad():
            output = actor(self.actor_batch, hidden)

        self.assertEqual(tuple(output.raw_logits), ACTION_BRANCH_ORDER)
        self.assertEqual(len(output.raw_logits), 7)
        for branch in ACTION_BRANCH_ORDER:
            self.assertEqual(
                output.raw_logits[branch].shape,
                self.actor_batch.action_masks[branch].shape,
            )
            self.assertTrue(torch.isfinite(output.raw_logits[branch]).all())
        self.assertEqual(
            output.recurrent_features.shape,
            (1, 1, self.config.environment.uav_count, 128),
        )
        self.assertEqual(
            output.hidden_out.shape,
            (1, self.config.environment.uav_count, 128),
        )
        masked = output.masked_logits()
        for branch in ACTION_BRANCH_ORDER:
            self.assertTrue(
                torch.isfinite(masked[branch][output.action_masks[branch]]).all()
            )
            self.assertTrue(
                torch.isneginf(masked[branch][~output.action_masks[branch]]).all()
            )

    def test_recurrent_sequence_matches_stepwise_forward_and_is_deterministic(self) -> None:
        step_result = self.environment.step(self.proposals)
        assert step_result.observations is not None
        assert step_result.centralized_state is not None
        second_proposals = self.environment.canonical_proposals()
        second_actor = self.actor_tensorizer.encode_step(
            step_result.observations,
            second_proposals,
            episode_start=False,
        )
        sequence = stack_actor_time((self.actor_batch, second_actor)).repeat_batch(2)
        actor = CAGATMAPPOActor(self.config).eval()
        hidden = actor.initial_hidden(2)

        with torch.no_grad():
            combined = actor(sequence, hidden)
            repeated = actor(sequence, hidden)
            first = actor(self.actor_batch.repeat_batch(2), hidden)
            second = actor(second_actor.repeat_batch(2), first.hidden_out)

        for branch in ACTION_BRANCH_ORDER:
            self.assertTrue(
                torch.equal(combined.raw_logits[branch], repeated.raw_logits[branch])
            )
            stepwise = torch.cat(
                [first.raw_logits[branch], second.raw_logits[branch]], dim=1
            )
            self.assertTrue(
                torch.allclose(
                    combined.raw_logits[branch], stepwise, rtol=1.0e-5, atol=1.0e-6
                )
            )
        self.assertTrue(
            torch.allclose(combined.hidden_out, second.hidden_out, rtol=1.0e-5, atol=1.0e-6)
        )

    def test_seeded_initialization_and_eval_forward_are_reproducible(self) -> None:
        first = CAGATMAPPOActor(self.config).eval()
        second = CAGATMAPPOActor(self.config).eval()
        for left, right in zip(first.state_dict().values(), second.state_dict().values()):
            self.assertTrue(torch.equal(left, right))
        hidden = first.initial_hidden(1)
        with torch.no_grad():
            left = first(self.actor_batch, hidden)
            right = second(self.actor_batch, hidden)
        for branch in ACTION_BRANCH_ORDER:
            self.assertTrue(torch.equal(left.raw_logits[branch], right.raw_logits[branch]))

    def test_centralized_critic_value_shape_batch_time_and_finiteness(self) -> None:
        step_result = self.environment.step(self.proposals)
        assert step_result.centralized_state is not None
        second = self.critic_tensorizer.encode_step(step_result.centralized_state)
        sequence = stack_centralized_time((self.critic_batch, second)).repeat_batch(3)
        critic = MAPPOCentralizedCritic(self.config).eval()

        with torch.no_grad():
            values = critic(sequence)

        self.assertEqual(values.shape, (3, 2, 1))
        self.assertTrue(torch.isfinite(values).all())

    def test_bad_mask_shape_is_rejected(self) -> None:
        bad_masks = dict(self.actor_batch.action_masks)
        bad_masks["route"] = bad_masks["route"][..., :-1]
        bad_batch = replace(self.actor_batch, action_masks=bad_masks)
        with self.assertRaises(MAPPONetworkError):
            CAGATMAPPOActor(self.config).eval()(bad_batch)

    def test_actor_interface_excludes_privileged_state_and_old_policies_regress(self) -> None:
        signature = inspect.signature(CAGATMAPPOActor.forward)
        self.assertEqual(tuple(signature.parameters), ("self", "batch", "hidden_in"))
        tensor_fields = set(self.actor_batch.__dataclass_fields__)
        for prohibited in (
            "centralized_state",
            "true_channel",
            "current_sinr",
            "current_interference",
            "executed_action",
        ):
            self.assertNotIn(prohibited, tensor_fields)

        random_policy = RandomPolicy(self.config.seed)
        heuristic_policy = HeuristicPolicy(self.config)
        for observation in self.reset.observations:
            self.assertTrue(
                observation.action_masks.is_legal(random_policy.act(observation))
            )
            self.assertTrue(
                observation.action_masks.is_legal(heuristic_policy.act(observation))
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_actor_and_critic_forward_are_finite(self) -> None:
        device = torch.device("cuda")
        actor = CAGATMAPPOActor(self.config).to(device).eval()
        critic = MAPPOCentralizedCritic(self.config).to(device).eval()
        actor_batch = self.actor_batch.to(device)
        critic_batch = self.critic_batch.to(device)
        hidden = actor.initial_hidden(1, device=device)

        with torch.no_grad():
            actor_output = actor(actor_batch, hidden)
            values = critic(critic_batch)

        self.assertEqual(actor_output.hidden_out.device.type, "cuda")
        self.assertEqual(values.device.type, "cuda")
        self.assertTrue(torch.isfinite(actor_output.hidden_out).all())
        self.assertTrue(torch.isfinite(values).all())
        for branch in ACTION_BRANCH_ORDER:
            self.assertTrue(torch.isfinite(actor_output.raw_logits[branch]).all())


if __name__ == "__main__":
    unittest.main()
