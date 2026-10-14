"""Action Distribution Gate for sequential masked CA-GAT-MAPPO proposals."""

from __future__ import annotations

import copy
import unittest
from dataclasses import replace

import numpy as np
import torch

from src.config import RunConfig
from src.env.actions import ActionProposal
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import ACTION_BRANCH_ORDER, ActorObservationTensorizer, CAGATMAPPOActor
from src.models.ca_gat_mappo_actions import (
    ActionDistributionError,
    CAGATMAPPOActionDistribution,
    SequentialActionMaskBatch,
)


def make_distribution_config() -> RunConfig:
    """Return a short real-environment config with deterministic full arrivals."""

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
    config = replace(base, environment=environment)
    config.validate()
    return config


class CAGATMAPPOActionDistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = make_distribution_config()
        self.environment = U2UMECEnvironment(self.config)
        self.reset = self.environment.reset()
        self.tensorizer = ActorObservationTensorizer(self.config)
        self.actor = CAGATMAPPOActor(self.config).eval()
        self.distribution = CAGATMAPPOActionDistribution(self.actor, self.config)
        self.reset_batch = self.tensorizer.encode_step(
            self.reset.observations,
            self.environment.canonical_proposals(),
        )
        self.reset_masks = SequentialActionMaskBatch.from_observations(
            self.reset.observations
        )

    def _advance_to_tx_queues(self):
        first = self.environment.step(self.environment.canonical_proposals())
        assert first.observations is not None
        route_proposals = list(self.environment.canonical_proposals())
        for agent, observation in enumerate(first.observations):
            route_domain = observation.action_masks.route_domain
            route_mask = observation.action_masks.mask_for("route", {})
            remote = next(
                value
                for value, allowed in zip(route_domain, route_mask)
                if bool(allowed) and isinstance(value, int) and not isinstance(value, bool)
            )
            route_proposals[agent] = replace(route_proposals[agent], route=remote)
        second = self.environment.step(route_proposals)
        assert second.observations is not None
        self.assertTrue(
            all(observation.action_masks.tx_branch_active for observation in second.observations)
        )
        batch = self.tensorizer.encode_step(second.observations)
        masks = SequentialActionMaskBatch.from_observations(second.observations)
        return second.observations, batch, masks

    @staticmethod
    def _active_proposal(
        observation,
        *,
        resource_group: int = 4,
        resource_width: int = 2,
    ) -> ActionProposal:
        masks = observation.action_masks
        tx_mask = masks.mask_for("tx_select", {})
        tx_select = next(
            value
            for value, allowed in zip(masks.tx_select_domain, tx_mask)
            if bool(allowed) and isinstance(value, int) and not isinstance(value, bool)
        )
        proposal = ActionProposal(
            uav_id=observation.uav_id,
            route="defer",
            tx_select=tx_select,
            resource_group=resource_group,
            resource_width=resource_width,
            power_level=0.25,
            cpu_queue="idle",
            cpu_frequency=0.0,
        )
        if not masks.is_legal(proposal):
            raise AssertionError("test fixture must remain legal under the real masks")
        return proposal

    @staticmethod
    def _grid(proposals):
        return ((tuple(proposals),),)

    def test_sequential_sampling_uses_real_order_masks_and_legal_proposals(self) -> None:
        observations, batch, masks = self._advance_to_tx_queues()
        with torch.no_grad():
            output = self.distribution.sample_actions(batch, masks)

        self.assertEqual(tuple(output.action_indices), ACTION_BRANCH_ORDER)
        self.assertEqual(tuple(output.action_masks), ACTION_BRANCH_ORDER)
        self.assertEqual(tuple(output.branch_log_probs), ACTION_BRANCH_ORDER)
        for agent, observation in enumerate(observations):
            proposal = output.proposal_at(0, 0, agent)
            self.assertTrue(observation.action_masks.is_legal(proposal))
            context = {}
            for branch in ACTION_BRANCH_ORDER:
                expected = observation.action_masks.mask_for(branch, context)
                np.testing.assert_array_equal(
                    output.action_masks[branch][0, 0, agent].cpu().numpy(),
                    expected,
                )
                selected_index = int(output.action_indices[branch][0, 0, agent])
                self.assertTrue(bool(expected[selected_index]))
                context[branch] = getattr(proposal, branch)

    def test_conditional_masks_change_and_illegal_probabilities_are_zero(self) -> None:
        observations, batch, masks = self._advance_to_tx_queues()
        idle = self.environment.canonical_proposals()
        group_five = tuple(
            self._active_proposal(item, resource_group=5, resource_width=1)
            for item in observations
        )
        group_four = tuple(
            self._active_proposal(item, resource_group=4, resource_width=2)
            for item in observations
        )
        with torch.no_grad():
            idle_output = self.distribution.evaluate_actions(batch, masks, self._grid(idle))
            five_output = self.distribution.evaluate_actions(
                batch, masks, self._grid(group_five)
            )
            four_output = self.distribution.evaluate_actions(
                batch, masks, self._grid(group_four)
            )

        self.assertFalse(
            torch.equal(
                idle_output.action_masks["resource_group"],
                five_output.action_masks["resource_group"],
            )
        )
        self.assertEqual(
            five_output.action_masks["resource_width"][0, 0, 0].tolist(),
            [True, False],
        )
        self.assertEqual(
            four_output.action_masks["resource_width"][0, 0, 0].tolist(),
            [True, True],
        )
        for output in (idle_output, five_output, four_output):
            for branch in ACTION_BRANCH_ORDER:
                probabilities = output.probabilities[branch]
                legal = output.action_masks[branch]
                self.assertTrue(torch.all(probabilities[~legal] == 0.0))
                self.assertTrue(
                    torch.allclose(
                        probabilities.sum(dim=-1),
                        torch.ones_like(probabilities.sum(dim=-1)),
                    )
                )

    def test_deterministic_mode_selects_legal_masked_argmax(self) -> None:
        observations, batch, masks = self._advance_to_tx_queues()
        with torch.no_grad():
            output = self.distribution.deterministic_actions(batch, masks)

        for branch in ACTION_BRANCH_ORDER:
            expected = torch.argmax(output.probabilities[branch], dim=-1)
            self.assertTrue(torch.equal(output.action_indices[branch], expected))
            selected_legal = torch.gather(
                output.action_masks[branch],
                -1,
                output.action_indices[branch].unsqueeze(-1),
            ).squeeze(-1)
            self.assertTrue(torch.all(selected_legal))
        for agent, observation in enumerate(observations):
            self.assertTrue(
                observation.action_masks.is_legal(output.proposal_at(0, 0, agent))
            )

    def test_stochastic_sampling_sequence_is_reproducible(self) -> None:
        _, batch, masks = self._advance_to_tx_queues()
        first = CAGATMAPPOActionDistribution(self.actor, self.config)
        second = CAGATMAPPOActionDistribution(self.actor, self.config)
        with torch.no_grad():
            first_sequence = [first.sample_actions(batch, masks) for _ in range(3)]
            second_sequence = [second.sample_actions(batch, masks) for _ in range(3)]

        for left, right in zip(first_sequence, second_sequence):
            self.assertEqual(left.proposals, right.proposals)
            for branch in ACTION_BRANCH_ORDER:
                self.assertTrue(
                    torch.equal(left.action_indices[branch], right.action_indices[branch])
                )

    def test_active_log_probs_joint_log_prob_and_entropy_are_finite_sums(self) -> None:
        observations, batch, masks = self._advance_to_tx_queues()
        proposals = tuple(self._active_proposal(item) for item in observations)
        with torch.no_grad():
            output = self.distribution.evaluate_actions(
                batch, masks, self._grid(proposals)
            )

        self.assertTrue(torch.isfinite(output.joint_log_prob).all())
        self.assertTrue(torch.isfinite(output.entropy).all())
        self.assertTrue(
            torch.allclose(output.joint_log_prob, sum(output.branch_log_probs.values()))
        )
        self.assertTrue(
            torch.allclose(output.entropy, sum(output.branch_entropies.values()))
        )
        for branch in ACTION_BRANCH_ORDER:
            self.assertTrue(torch.isfinite(output.branch_log_probs[branch]).all())
            self.assertTrue(torch.isfinite(output.branch_entropies[branch]).all())
        self.assertTrue(torch.all(output.active_branches["tx_select"]))
        self.assertTrue(torch.all(output.active_branches["resource_group"]))
        self.assertTrue(torch.all(output.active_branches["resource_width"]))
        self.assertTrue(torch.all(output.active_branches["power_level"]))

    def test_inactive_branches_use_canonical_actions_and_zero_contribution(self) -> None:
        canonical = self.config.action.canonical_inactive_values
        with torch.no_grad():
            output = self.distribution.sample_actions(self.reset_batch, self.reset_masks)

        for agent in range(self.config.environment.uav_count):
            proposal = output.proposal_at(0, 0, agent)
            self.assertEqual(proposal.route, canonical["route"])
            self.assertEqual(proposal.tx_select, canonical["tx_select"])
            self.assertEqual(proposal.resource_group, canonical["resource_group"])
            self.assertEqual(proposal.resource_width, canonical["resource_width"])
            self.assertEqual(proposal.power_level, canonical["power_level"])
            self.assertEqual(proposal.cpu_queue, canonical["cpu_queue"])
            self.assertEqual(proposal.cpu_frequency, canonical["cpu_frequency"])
        for branch in ACTION_BRANCH_ORDER:
            self.assertFalse(torch.any(output.active_branches[branch]))
            self.assertTrue(torch.all(output.branch_log_probs[branch] == 0.0))
            self.assertTrue(torch.all(output.branch_entropies[branch] == 0.0))
        self.assertTrue(torch.all(output.joint_log_prob == 0.0))
        self.assertTrue(torch.all(output.entropy == 0.0))

    def test_sample_then_re_evaluate_has_identical_probability_statistics(self) -> None:
        _, batch, masks = self._advance_to_tx_queues()
        with torch.no_grad():
            sampled = self.distribution.sample_actions(batch, masks)
            evaluated = self.distribution.evaluate_actions(
                batch, masks, sampled.proposals
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
                    sampled.branch_log_probs[branch],
                    evaluated.branch_log_probs[branch],
                )
            )
            self.assertTrue(
                torch.allclose(
                    sampled.branch_entropies[branch],
                    evaluated.branch_entropies[branch],
                )
            )
        self.assertTrue(torch.allclose(sampled.joint_log_prob, evaluated.joint_log_prob))
        self.assertTrue(torch.allclose(sampled.entropy, evaluated.entropy))

    def test_externally_supplied_illegal_action_is_rejected(self) -> None:
        observations, batch, masks = self._advance_to_tx_queues()
        proposals = [
            self._active_proposal(item, resource_group=5, resource_width=1)
            for item in observations
        ]
        proposals[0] = replace(proposals[0], resource_width=2)
        with self.assertRaises(ActionDistributionError):
            self.distribution.evaluate_actions(batch, masks, self._grid(proposals))

    def test_malformed_and_all_invalid_masks_fail_fast(self) -> None:
        for malformed in ("all_invalid", "wrong_shape"):
            with self.subTest(malformed=malformed):
                contracts = list(self.reset_masks.contracts[0][0])
                broken = copy.copy(contracts[0])
                replacement = (
                    np.zeros(len(broken.route_domain), dtype=np.bool_)
                    if malformed == "all_invalid"
                    else np.ones(len(broken.route_domain) - 1, dtype=np.bool_)
                )
                replacement.setflags(write=False)
                object.__setattr__(broken, "route_mask", replacement)
                contracts[0] = broken
                mask_batch = SequentialActionMaskBatch(((tuple(contracts),),))
                with self.assertRaises(ActionDistributionError):
                    self.distribution.sample_actions(self.reset_batch, mask_batch)

    def test_batch_and_multi_agent_axes_are_preserved(self) -> None:
        observations, batch, masks = self._advance_to_tx_queues()
        repeated_batch = batch.repeat_batch(2)
        repeated_masks = masks.repeat_batch(2)
        with torch.no_grad():
            output = self.distribution.sample_actions(repeated_batch, repeated_masks)

        expected = (2, 1, self.config.environment.uav_count)
        self.assertEqual(output.joint_log_prob.shape, expected)
        self.assertEqual(output.entropy.shape, expected)
        self.assertEqual(len(output.proposals), 2)
        for branch in ACTION_BRANCH_ORDER:
            self.assertEqual(output.action_indices[branch].shape, expected)
            self.assertEqual(output.branch_log_probs[branch].shape, expected)
        for batch_index in range(2):
            for agent, observation in enumerate(observations):
                self.assertTrue(
                    observation.action_masks.is_legal(
                        output.proposal_at(batch_index, 0, agent)
                    )
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_sampling_and_re_evaluation_are_finite_and_consistent(self) -> None:
        _, batch, masks = self._advance_to_tx_queues()
        device = torch.device("cuda")
        actor = CAGATMAPPOActor(self.config).to(device).eval()
        distribution = CAGATMAPPOActionDistribution(actor, self.config)
        with torch.no_grad():
            sampled = distribution.sample_actions(batch.to(device), masks)
            evaluated = distribution.evaluate_actions(
                batch.to(device), masks, sampled.proposals
            )

        self.assertEqual(sampled.joint_log_prob.device.type, "cuda")
        self.assertTrue(torch.isfinite(sampled.joint_log_prob).all())
        self.assertTrue(torch.isfinite(sampled.entropy).all())
        self.assertTrue(torch.allclose(sampled.joint_log_prob, evaluated.joint_log_prob))
        self.assertTrue(torch.allclose(sampled.entropy, evaluated.entropy))


if __name__ == "__main__":
    unittest.main()
