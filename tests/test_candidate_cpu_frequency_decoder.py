"""E4-v1 model-only tests; no experimental training or checkpoint updates."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.config import (
    CHECKPOINT_KIND_FINAL_COMPLETED, CHECKPOINT_KIND_PERIODIC_RESUME,
    CHECKPOINT_SCHEMA_VERSION, ConfigError, CpuFrequencyDecoderMode, RunConfig,
    load_run_config,
    validate_mappo_checkpoint_resume_compatibility,
)
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import (
    ACTION_BRANCH_ORDER, ActorObservationTensorizer, CAGATMAPPOActor,
    MAPPOCentralizedCritic, stack_actor_time,
)
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution, SequentialActionMaskBatch
from src.models.ca_gat_mappo_ppo import compute_ppo_objective_and_loss
from src.evaluation.actor_loader import EvaluationCheckpointError, load_final_actor_for_evaluation
from src.evaluation.runner import FormalEvaluationRunner
from src.models.ca_gat_mappo_checkpoint import atomic_save_checkpoint
from tests.test_candidate_aware_route_decoder_v1 import differing_paths
from tests.test_environment_gate0 import make_environment_config
from tests.test_ca_gat_mappo_checkpoint import make_checkpoint_config, make_payload

ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / 'tests/fixtures/configs/cooperative-load-v2-hierarchical-optionaware-v1-hierarchical-tx-v1-remote-stress-v2-branchwise-32k.json'
E4 = ROOT / 'tests/fixtures/configs/e4-v1-candidate-cpu-frequency-seed43-32k.json'
E3_HISTORICAL_SNAPSHOT = ROOT / 'tests/fixtures/historical/e3-seed43-cfg735fb97120ce-config_snapshot.yaml'
GATE_M = ROOT / 'logs/gate_m/ca_gat_mappo__remote_stress_v2__seed-43__cfg-735fb97120ce__git-97b387271007'


def with_mode(config, mode='candidate_aware_v1'):
    result = replace(config, training=replace(config.training, mappo=replace(
        config.training.mappo, cpu_frequency_decoder_mode=CpuFrequencyDecoderMode(mode))))
    for name in ('git_commit', 'git_branch', 'git_dirty'):
        object.__setattr__(result, name, getattr(config, name))
    return result


def cpu_fixture():
    config = with_mode(make_environment_config(horizon=8))
    env = U2UMECEnvironment(config)
    env.reset()
    for _ in range(3):
        env.step(env.canonical_proposals())
    lifecycle = env.lifecycle
    # Unit fixture: two remote source queues and one local queue at UAV 1.
    for source, deadline, cycles in ((0, 6, 6e7), (2, 5, 8e7), (1, 7, 5e7)):
        task = lifecycle.create_task(source_uav=source, data_bits=3e5,
                                     cpu_cycles=cycles, arrival_slot=0, deadline_slot=deadline)
        env.metrics.record_generated((task,))
        lifecycle.route_task(task, slot=1, destination='local' if source == 1 else 1)
        if source == 1:
            env.metrics.record_local_bindings((task,))
        if source != 1:
            lifecycle.service_transmission(task, slot=2, bits=task.remaining_bits)
    env.current_observations, env.current_state = env._build_decision_records(env.traffic.estimate(env.slot))
    observations = env.current_observations
    batch = ActorObservationTensorizer(config).encode_step(observations)
    masks = SequentialActionMaskBatch.from_observations(observations)
    actor = CAGATMAPPOActor(config)
    return config, env, observations, batch, masks, actor


def force_cpu_queue(actor, index=1):
    with torch.no_grad():
        head = actor.action_heads['cpu_queue']
        head.weight.zero_()
        head.bias.fill_(-20)
        head.bias[index] = 20


class CandidateCpuFrequencyTests(unittest.TestCase):
    def test_legacy_identity_and_invalid_mode(self):
        config = load_run_config(E3, cli_overrides={'seed': 43})
        self.assertEqual(config.config_hash, 'cca983e51ebe1aac1f4391cbacbcf218d6bf91e81749fd039c3c2f9930653ee8')
        self.assertEqual(config.config_hash, with_mode(config, 'legacy').config_hash)
        self.assertNotIn('cpu_frequency_decoder_mode', config.resolved_dict()['training']['mappo'])
        with self.assertRaisesRegex(ConfigError, 'cpu_frequency_decoder_mode'):
            replace(config, training=replace(config.training, mappo=replace(
                config.training.mappo, cpu_frequency_decoder_mode='unknown'))).validate()

    def test_legacy_pre_edit_numerical_baseline(self):
        config = load_run_config(E3, cli_overrides={'seed': 43})
        actor = CAGATMAPPOActor(config).eval()
        digest = hashlib.sha256()
        for name, value in actor.state_dict().items():
            digest.update(name.encode()); digest.update(value.cpu().numpy().tobytes())
        self.assertEqual(digest.hexdigest(), 'b59ab3ba0b9a65c5d7ef01c381970b5b32bd76bfa6a78d1317a5853f94f246f8')
        env = U2UMECEnvironment(config)
        batch = ActorObservationTensorizer(config).encode_step(env.reset().observations)
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(4)
            with torch.no_grad():
                output = actor(batch)
            self.assertEqual(hashlib.sha256(output.raw_logits['cpu_frequency'].numpy().tobytes()).hexdigest(),
                             '9f1c8cc2951912ef1fa35e7e72a845fc25c4819445191a7fb197047087e1b92f')
            self.assertEqual(hashlib.sha256(output.hidden_out.numpy().tobytes()).hexdigest(),
                             '9b66b2e40c2e9c9e18e78dba060e4011e93586d9ae459437cbc94cef0a2e9ff0')
        finally:
            torch.set_num_threads(threads)

    def test_legacy_conditional_linear_path(self):
        config, _, _, batch, _, _ = cpu_fixture()
        actor = CAGATMAPPOActor(with_mode(config, 'legacy'))
        recurrent = actor(batch).recurrent_features
        indices = torch.full((1, 1, 4), 1, dtype=torch.long)
        actual = actor.branch_logits('cpu_frequency', recurrent, {'cpu_queue': indices}, batch)
        expected = actor.action_heads['cpu_frequency'](torch.cat(
            (recurrent, actor.branch_embeddings['cpu_queue'](indices)), dim=-1))
        self.assertTrue(torch.equal(actual, expected))

    def test_common_initial_weights_and_critic_unchanged(self):
        config = load_run_config(E3, cli_overrides={'seed': 43})
        legacy, candidate = CAGATMAPPOActor(config), CAGATMAPPOActor(with_mode(config))
        for name, value in legacy.state_dict().items():
            if not name.startswith('action_heads.cpu_frequency.'):
                self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)
        for left, right in zip(MAPPOCentralizedCritic(config).parameters(), MAPPOCentralizedCritic(with_mode(config)).parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_candidate_permutation_consistency(self):
        _, _, _, batch, _, actor = cpu_fixture()
        head = actor.action_heads['cpu_frequency']
        recurrent = actor(batch).recurrent_features
        indices = torch.ones((1, 1, 4), dtype=torch.long)
        features = head.candidate_features(batch, indices)
        permutation = torch.tensor([3, 1, 0, 2])
        logits = head.score_candidates(recurrent, features)
        permuted = head.score_candidates(recurrent, features[..., permutation, :])
        torch.testing.assert_close(permuted, logits[..., permutation], rtol=0, atol=0)
        mask = torch.tensor([True, True, False, True]).expand_as(logits)
        p = logits.masked_fill(~mask, -torch.inf).softmax(-1)
        pp = permuted.masked_fill(~mask[..., permutation], -torch.inf).softmax(-1)
        torch.testing.assert_close(pp, p[..., permutation])

    def test_feature_sources_remote_and_local(self):
        config, _, observations, batch, _, actor = cpu_fixture()
        head = actor.action_heads['cpu_frequency']
        for source in (0, 1, 2):
            indices = torch.full((1, 1, 4), source + 1, dtype=torch.long)
            feature = head.candidate_features(batch, indices)[0, 0, 1]
            q = observations[1].private_queues
            cycles = q.local_cpu.head_remaining_cycles if source == 1 else q.cpu_by_source.head_remaining_cycles[source]
            slack = q.local_cpu.head_slack_slots if source == 1 else q.cpu_by_source.head_slack_slots[source]
            workload = q.local_cpu.remaining_cycles if source == 1 else q.cpu_by_source.remaining_cycles[source]
            scale = config.environment.task_data_bits_max * config.environment.task_cycles_per_bit_max
            torch.testing.assert_close(feature[:, 0], torch.tensor(config.action.cpu_frequency_levels))
            torch.testing.assert_close(feature[:, 1], feature[:, 0] * observations[1].self_resources.max_cpu_frequency_hz * config.environment.slot_duration_s / scale)
            self.assertAlmostEqual(float(feature[0, 2]), cycles / scale, places=6)
            self.assertAlmostEqual(float(feature[0, 3]), slack / config.environment.maximum_task_slack_slots, places=6)
            self.assertAlmostEqual(float(feature[0, 4]), workload / (4 * 8 * scale), places=6)
            self.assertAlmostEqual(float(feature[0, 5]), observations[1].self_resources.residual_energy_ratio, places=6)

    def test_slack_causality_and_future_outcome_is_not_read(self):
        _, env, _, batch, _, actor = cpu_fixture()
        head = actor.action_heads['cpu_frequency']
        indices = torch.ones((1, 1, 4), dtype=torch.long)
        before = head.candidate_features(batch, indices)
        altered = replace(batch, self_features=batch.self_features.clone())
        slot = head.extractor.cpu_start + 6
        altered.self_features[..., 1, slot] -= 0.1
        after = head.candidate_features(altered, indices)
        self.assertFalse(torch.equal(before[..., 1, :, 3], after[..., 1, :, 3]))
        for feature in (0, 1, 2, 4, 5):
            self.assertTrue(torch.equal(before[..., feature], after[..., feature]))
        # Neither extractor nor scorer holds an environment/task/critic reference.
        env.lifecycle.tasks[0].completion_slot = 999
        env.lifecycle.tasks[0].terminal_reason = 'future-label-sentinel'
        self.assertTrue(torch.equal(before, head.candidate_features(batch, indices)))

    def test_future_time_suffix_does_not_change_prefix(self):
        _, _, _, batch, _, actor = cpu_fixture()
        sequence = stack_actor_time((batch, batch))
        altered = replace(sequence, self_features=sequence.self_features.clone())
        altered.self_features[:, 1] += 0.05
        torch.testing.assert_close(actor(sequence).raw_logits['cpu_frequency'][:, 0],
                                   actor(altered).raw_logits['cpu_frequency'][:, 0], rtol=0, atol=0)

    def test_unselected_queue_does_not_change_explicit_features(self):
        _, _, _, batch, _, actor = cpu_fixture()
        head = actor.action_heads['cpu_frequency']
        indices = torch.ones((1, 1, 4), dtype=torch.long)
        altered = replace(batch, self_features=batch.self_features.clone())
        start = head.extractor.cpu_start + 2 * 8
        altered.self_features[..., start:start + 8] = 0.75
        self.assertTrue(torch.equal(head.candidate_features(batch, indices), head.candidate_features(altered, indices)))

    def _decode(self, energy_mask=None):
        config, env, observations, batch, masks, actor = cpu_fixture()
        force_cpu_queue(actor)
        if energy_mask is not None:
            contracts = list(masks.contracts[0][0])
            frequencies = np.array(contracts[1].cpu_frequency_energy_mask, copy=True)
            frequencies[0] = energy_mask
            contracts[1] = replace(contracts[1], cpu_frequency_energy_mask=frequencies)
            masks = SequentialActionMaskBatch(((tuple(contracts),),))
        distribution = CAGATMAPPOActionDistribution(actor, config)
        output = distribution.deterministic_actions(batch, masks)
        return config, env, observations, batch, masks, actor, distribution, output

    def test_multiple_legal_frequencies_and_original_masks(self):
        _, _, observations, _, _, _, _, output = self._decode()
        self.assertEqual(int(output.action_masks['cpu_frequency'][0, 0, 1].sum()), 4)
        for agent, observation in enumerate(observations):
            self.assertTrue(observation.action_masks.is_legal(output.proposal_at(0, 0, agent)))
        torch.testing.assert_close(output.probabilities['cpu_frequency'].sum(-1), torch.ones((1, 1, 4)))

    def test_one_legal_frequency(self):
        *_, output = self._decode([True, False, False, False])
        self.assertEqual(output.proposal_at(0, 0, 1).cpu_frequency, 0)
        self.assertEqual(float(output.probabilities['cpu_frequency'][0, 0, 1, 0]), 1)
        self.assertEqual(float(output.branch_log_probs['cpu_frequency'][0, 0, 1]), 0)

    def test_energy_limited_mask(self):
        *_, output = self._decode([True, True, False, False])
        self.assertTrue(torch.equal(output.probabilities['cpu_frequency'][0, 0, 1, 2:], torch.zeros(2)))
        self.assertLessEqual(output.proposal_at(0, 0, 1).cpu_frequency, 0.25)

    def test_learned_scorer_can_choose_low_or_high_frequency(self):
        _, _, _, batch, masks, actor, distribution, _ = self._decode([True, True, True, True])
        head = actor.action_heads['cpu_frequency']
        for sign, expected in ((-1, 0.0), (1, 1.0)):
            with torch.no_grad():
                for parameter in head.parameters():
                    parameter.zero_()
                head.scorer[0].weight[0, actor.spec.gru_hidden_dimension] = sign
                head.scorer[2].weight[0, 0] = 1
            output = distribution.deterministic_actions(batch, masks)
            self.assertEqual(output.proposal_at(0, 0, 1).cpu_frequency, expected)

    def test_real_energy_mask(self):
        config, env, _, _, _, actor = cpu_fixture()
        env.resource_states[1].residual_energy_j = 0
        estimate = env.traffic.estimate(env.slot)
        observations, _ = env._build_decision_records(estimate)
        batch = ActorObservationTensorizer(config).encode_step(observations)
        masks = SequentialActionMaskBatch.from_observations(observations)
        force_cpu_queue(actor)
        output = CAGATMAPPOActionDistribution(actor, config).deterministic_actions(batch, masks)
        self.assertEqual(output.proposal_at(0, 0, 1).cpu_frequency, 0)
        self.assertTrue(observations[1].action_masks.is_legal(output.proposal_at(0, 0, 1)))

    def test_inactive_cpu_branch(self):
        config = with_mode(make_environment_config(horizon=4))
        env = U2UMECEnvironment(config)
        observations = env.reset().observations
        actor = CAGATMAPPOActor(config)
        batch = ActorObservationTensorizer(config).encode_step(observations)
        output = CAGATMAPPOActionDistribution(actor, config).deterministic_actions(
            batch, SequentialActionMaskBatch.from_observations(observations))
        self.assertFalse(bool(output.active_branches['cpu_frequency'].any()))
        self.assertTrue(torch.equal(output.branch_log_probs['cpu_frequency'], torch.zeros((1, 1, 4))))
        self.assertTrue(torch.equal(output.branch_entropies['cpu_frequency'], torch.zeros((1, 1, 4))))
        head = actor.action_heads['cpu_frequency']
        features = head.candidate_features(batch, torch.zeros((1, 1, 4), dtype=torch.long))
        self.assertTrue(torch.equal(features[..., 2:5], torch.zeros_like(features[..., 2:5])))

    def test_deterministic_decode_and_log_prob_recomputation(self):
        _, _, _, batch, masks, _, distribution, output = self._decode()
        repeated = distribution.deterministic_actions(batch, masks)
        self.assertEqual(output.proposals, repeated.proposals)
        evaluated = distribution.evaluate_actions(batch, masks, output.proposals)
        for branch in ACTION_BRANCH_ORDER:
            torch.testing.assert_close(output.branch_log_probs[branch], evaluated.branch_log_probs[branch], rtol=0, atol=0)

    def test_stochastic_log_prob_and_existing_ppo_ratio(self):
        _, _, _, batch, masks, _, distribution, _ = self._decode()
        hidden = distribution.actor.initial_hidden(1)
        output = distribution.sample_actions(batch, masks, hidden,
            generator=torch.Generator().manual_seed(17))
        evaluated = distribution.evaluate_actions(batch, masks, output.proposals, hidden)
        new = torch.stack([evaluated.branch_log_probs[b][0] for b in ACTION_BRANCH_ORDER], dim=-1)
        old = torch.stack([output.branch_log_probs[b][0].detach() for b in ACTION_BRANCH_ORDER], dim=-1)
        active = torch.stack([output.active_branches[b][0] for b in ACTION_BRANCH_ORDER], dim=-1)
        loss = compute_ppo_objective_and_loss(
            new_joint_log_prob=evaluated.joint_log_prob[0], old_joint_log_prob=output.joint_log_prob[0].detach(),
            advantage=torch.ones((1, 4)), current_value=torch.zeros((1, 4)), return_target=torch.zeros((1, 4)),
            entropy=evaluated.entropy[0], sequence_valid_mask=torch.ones(1, dtype=torch.bool),
            epsilon_clip=0.2, value_coefficient=0.5, entropy_coefficient=0.01,
            actor_ratio_mode='branch_specific', policy_loss_aggregation_mode='branchwise_active_mean_v1',
            agent_credit_mode='role_decomposed', new_branch_log_probs=new,
            old_branch_log_probs=old, active_branch_indicators=active)
        torch.testing.assert_close(loss.branch_ratio[active], torch.ones_like(loss.branch_ratio[active]))

    def test_model_backward_cpu_and_cuda(self):
        config, _, _, batch, masks, _, _, _ = self._decode()
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                actor = CAGATMAPPOActor(config).to(device)
                force_cpu_queue(actor)
                moved = batch.to(device=device)
                distribution = CAGATMAPPOActionDistribution(actor, config)
                output = distribution.deterministic_actions(moved, masks)
                evaluated = distribution.evaluate_actions(moved, masks, output.proposals)
                loss = -evaluated.branch_log_probs['cpu_frequency'].sum()
                loss.backward()  # One model test; no optimizer or parameter step.
                gradients = [p.grad for p in actor.action_heads['cpu_frequency'].parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)

    def test_e4_matched_configuration(self):
        source = load_run_config(E3_HISTORICAL_SNAPSHOT)
        target = load_run_config(E4)
        self.assertEqual(differing_paths(source.resolved_dict(), target.resolved_dict()),
                         {('training', 'mappo', 'cpu_frequency_decoder_mode')})
        self.assertEqual(target.seed, 43)
        self.assertEqual(target.training.mappo.max_training_environment_steps, 32000)
        self.assertEqual(target.evaluation.evaluation_seeds, (1042, 1043, 1044, 1045, 1046))


    def test_candidate_checkpoint_roundtrip_and_mode_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = with_mode(make_checkpoint_config(root, horizon=4, interval=4, budget=8))
            object.__setattr__(source, 'git_commit', 'a' * 40)
            payload, actor, _, _ = make_payload(source, CHECKPOINT_KIND_FINAL_COMPLETED)
            path = root / 'final.pt'
            atomic_save_checkpoint(payload, path)
            evaluation = replace(source, mode='evaluation', launch_profile=None)
            object.__setattr__(evaluation, 'git_commit', 'a' * 40)
            loaded = load_final_actor_for_evaluation(evaluation, path, evaluation_device='cpu')
            for name, value in actor.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), loaded.actor.state_dict()[name]), name)
            with self.assertRaisesRegex(EvaluationCheckpointError, 'cpu_frequency_decoder_mode'):
                load_final_actor_for_evaluation(with_mode(evaluation, 'legacy'), path, evaluation_device='cpu')


    def test_resume_metadata_rejects_decoder_change(self):
        config = with_mode(make_checkpoint_config(ROOT / 'logs/e4-v1-test-fixture', horizon=4, interval=4, budget=8))
        object.__setattr__(config, 'git_commit', 'a' * 40)
        with self.assertRaisesRegex(ConfigError, 'cpu_frequency_decoder_mode'):
            validate_mappo_checkpoint_resume_compatibility(
                config, schema_version=CHECKPOINT_SCHEMA_VERSION, checkpoint_kind=CHECKPOINT_KIND_PERIODIC_RESUME,
                method_id='ca_gat_mappo', git_commit=config.git_commit, config_hash=config.config_hash,
                training_device='cpu', cuda_available=False)


if __name__ == '__main__':
    unittest.main()
