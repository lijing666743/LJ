"""E12 bounded synthetic tests; no formal training or evaluation."""
import copy
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.config import ConfigError, load_run_config
from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution
from src.models.ca_gat_mappo_checkpoint import (
    CheckpointError, atomic_save_checkpoint, load_checkpoint_payload,
    restore_model_optimizer_and_rng, serialize_active_rollout,
    restore_active_rollout, validate_checkpoint_payload,
)
from src.models.ca_gat_mappo_update import (
    CAGATMAPPORecurrentPPOUpdater, build_recurrent_ppo_minibatch,
)
from src.models.ca_gat_mappo_ppo import PPOObjectiveError
from src.models.tx_upper_factual_credit import factual_team_credit
from tests.test_integrated_scheduler import fixture, ROOT, CONFIGS
from tests.test_route_private_backbone import gradient_sum
from tests.test_tx_upper_stability import preserve_git


class TXFactualCreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config = load_run_config(CONFIGS / 'e12-v1-tx-factual-team-gae-seed43-32k.json')
        cls.e11 = load_run_config(CONFIGS / 'e11-seed43-config-snapshot.yaml')
        cls.output = Path(tempfile.gettempdir()) / 'e12-v1-validation'
        cls.output.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self, config=None, device='cpu'):
        config = self.config if config is None else config
        return CAGATMAPPOActor(config).to(device), MAPPOCentralizedCritic(config).to(device)

    def helper(self, config=None):
        from tests.test_tx_upper_counterfactual import TXCounterfactualTests
        helper = TXCounterfactualTests()
        helper.config = self.config if config is None else config
        _, helper.batch, helper.masks = fixture(helper.config)
        return helper

    def credit(self, target, value=None, active=None, valid=None, mode='role_decomposed'):
        if value is None:
            value = torch.zeros_like(target)
        if active is None:
            active = torch.ones_like(target, dtype=torch.bool)
        if valid is None:
            valid = torch.ones(active.shape[:-1], dtype=torch.bool, device=target.device)
        return factual_team_credit(target, value, mode, active, valid)

    def test_raw_role_sum_then_shared_cohort_normalization(self):
        target = torch.tensor([[[1., 2.], [4., 8.], [16., 32.]]], requires_grad=True)
        value = torch.ones_like(target, requires_grad=True)
        active = torch.tensor([[[True, False], [True, True], [False, True]]])
        c = self.credit(target, value, active)
        expected = torch.tensor([[[1., 0.], [10., 10.], [0., 46.]]])
        torch.testing.assert_close(c.raw_advantage, expected, rtol=0, atol=0)
        cohort = expected[active]
        torch.testing.assert_close(c.normalized_advantage[active],
                                   (cohort-cohort.mean())/cohort.std(unbiased=False))
        for x in (c.team_return, c.team_old_value, c.raw_advantage, c.normalized_advantage):
            self.assertFalse(x.requires_grad)
        # Neither action labels nor policy probabilities enter this function.
        a = torch.tensor([[[True, False], [False, True], [False, True]]])
        self.assertEqual(c.telemetry(a)['normalization_mean'],
                         c.telemetry(~a)['normalization_mean'])
        self.assertEqual(c.telemetry(a)['sampled_transmit_count'], 3)
        self.assertEqual(c.telemetry(a)['sampled_idle_count'], 1)

    def test_team_scalar_not_multiplied_by_agents(self):
        target = torch.tensor([[3., 7.]])
        value = torch.tensor([[1., 2.]])
        c = self.credit(target, value, torch.ones(1, 2, 4, dtype=torch.bool), mode='team')
        torch.testing.assert_close(c.raw_advantage, torch.tensor([[[2.]*4, [5.]*4]]))

    def test_sequence_valid_and_inactive_nan_isolation(self):
        target = torch.tensor([[[1., 2.], [float('nan'), 2.], [float('nan'), 1.]]])
        active = torch.tensor([[[True, False], [True, True], [False, False]]])
        valid = torch.tensor([[True, False, True]])
        c = self.credit(target, active=active, valid=valid)
        self.assertEqual(c.active.sum(), 1)
        self.assertTrue((c.normalized_advantage == 0).all())
        self.assertTrue(torch.isfinite(c.raw_advantage).all())
        json.dumps(c.telemetry(active), allow_nan=False)

    def test_empty_singleton_constant_cohorts(self):
        for mask in (torch.zeros(1, 2, 2, dtype=torch.bool),
                     torch.tensor([[[True, False], [False, False]]]),
                     torch.ones(1, 2, 2, dtype=torch.bool)):
            c = self.credit(torch.ones(1, 2, 2), active=mask)
            self.assertTrue((c.normalized_advantage == 0).all())
            self.assertEqual(c.telemetry(mask)['credit_active_samples'], int(mask.sum()))
            json.dumps(c.telemetry(mask), allow_nan=False)

    def test_bad_shapes_modes_and_nonfinite_active_fail_closed(self):
        target = torch.ones(1, 2, 2)
        for change in ({'mode': 'bad'}, {'mode': 'team'},
                       {'active': torch.ones_like(target)},
                       {'valid': torch.ones(1, 3, dtype=torch.bool)}):
            with self.assertRaises(ValueError):
                self.credit(target, **change)
        target[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.credit(target)

    def test_team_gae_identity_terminal_truncated_and_tail(self):
        from src.models.ca_gat_mappo_gae import compute_per_agent_gae_and_returns, compute_gae_and_returns
        reward = torch.arange(8.).reshape(4, 2)
        value = reward * .1
        bootstrap = torch.full_like(value, 20.)
        terminated = torch.tensor([False, True, False, False])
        truncated = torch.tensor([False, False, True, False])
        boundary = terminated | truncated
        kw = dict(terminated=terminated, truncated=truncated, episode_boundary=boundary,
                  bootstrap_allowed=~boundary, gamma=.99, gae_lambda=.95)
        role = compute_per_agent_gae_and_returns(reward=reward, old_value=value,
                                                bootstrap_value=bootstrap, **kw)
        team = compute_gae_and_returns(reward=reward.sum(-1), old_value=value.sum(-1),
                                      bootstrap_value=bootstrap.sum(-1), **kw)
        c = self.credit(role.return_target, value)
        torch.testing.assert_close(c.raw_advantage[:, 0], role.advantage.sum(-1))
        torch.testing.assert_close(c.raw_advantage[:, 0], team.advantage)
        torch.testing.assert_close(c.team_return[1:3, 0], reward.sum(-1)[1:3])
        self.assertGreater(c.team_return[-1, 0], reward[-1].sum())

    def test_delayed_reward_across_chunk_boundary_and_expiration_not_terminal(self):
        from src.models.ca_gat_mappo_gae import compute_gae_and_returns
        reward = torch.zeros(256)
        reward[32] = 3.
        reward[60] = -2.  # Task expiration is a reward, not an episode boundary.
        mask = torch.zeros(256, dtype=torch.bool)
        result = compute_gae_and_returns(
            reward=reward, old_value=torch.zeros(256), bootstrap_value=torch.zeros(256),
            terminated=mask, truncated=mask, episode_boundary=mask,
            bootstrap_allowed=~mask, gamma=.99, gae_lambda=.95)
        self.assertAlmostEqual(float(result.advantage[31]),
                               3*.9405 - 2*.9405**29, places=5)
        self.assertAlmostEqual(float(result.advantage[59]), -2*.9405, places=5)

    def test_config_exact_two_effective_changes_and_fixed_budget(self):
        from src.evaluation.e4_v1_matched_audit import differences
        a, b = self.e11.resolved_dict(), self.config.resolved_dict()
        for d in (a, b):
            d['training']['mappo'].setdefault('tx_q_target_normalization', 'none')
        self.assertEqual([x['field'] for x in differences(a, b)],
                         ['training.mappo.tx_q_target_normalization', 'training.mappo.tx_upper_credit_mode'])
        m = self.config.training.mappo
        self.assertEqual((m.gamma, m.gae_lambda, m.rollout_length_slots,
                          m.recurrent_chunk_length_slots, m.sequence_minibatch_size,
                          m.update_epochs, m.max_training_environment_steps),
                         (.99, .95, 256, 32, 8, 4, 32000))
        self.assertEqual(m.policy_loss_aggregation_mode, 'branchwise_active_mean_v1')

    def test_invalid_popart_stability_and_missing_credit_rejected(self):
        for kw in ({'tx_q_target_normalization': 'popart_v1'},
                   {'tx_upper_optimization_mode': 'deterministic_stability_v1'},
                   {'route_private_backbone': False}):
            c = replace(self.config, training=replace(self.config.training,
                        mappo=replace(self.config.training.mappo, **kw)))
            with self.assertRaises(ConfigError):
                c.validate()
        h = self.helper()
        actor, _ = self.model()
        policy = CAGATMAPPOActionDistribution(actor, self.config).sample_actions(h.batch, h.masks)
        with self.assertRaises(PPOObjectiveError):
            h.loss(policy)

    def test_actor_original_value_and_optimizer_have_no_q(self):
        actor, critic = self.model()
        old_actor, old_critic = self.model(self.e11)
        for k, v in actor.state_dict().items():
            torch.testing.assert_close(v, old_actor.state_dict()[k], rtol=0, atol=0)
        for k, v in critic.value_network.state_dict().items():
            torch.testing.assert_close(v, old_critic.value_network.state_dict()[k], rtol=0, atol=0)
        self.assertFalse(hasattr(critic, 'tx_upper_q'))
        updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, self.config)
        self.assertEqual({id(x) for x in updater.optimizers.critic_parameters},
                         {id(x) for x in critic.value_network.parameters()})
        self.assertEqual(sum(p.numel() for p in old_critic.parameters()) -
                         sum(p.numel() for p in critic.parameters()), 317192)

    def test_non_tx_surrogates_and_value_loss_unchanged(self):
        h = self.helper()
        actor, _ = self.model()
        policy = CAGATMAPPOActionDistribution(actor, self.config).sample_actions(h.batch, h.masks)
        old = h.loss(policy, torch.tensor([[4., 3., 2., 1.]]), self.e11)
        new = h.loss(policy, torch.tensor([[1., 2., 3., 4.]]))
        for index in (0, 2, 3, 4, 5, 6):
            torch.testing.assert_close(new.branch_surrogate[..., index], old.branch_surrogate[..., index], rtol=0, atol=0)
        torch.testing.assert_close(new.critic_loss, old.critic_loss, rtol=0, atol=0)
        torch.testing.assert_close(new.global_entropy_loss_contribution, old.global_entropy_loss_contribution, rtol=0, atol=0)

    def gradient_check(self, device):
        h = self.helper()
        actor, critic = self.model(device=device)
        # Distinct time returns produce factual advantage of both signs.
        policy = CAGATMAPPOActionDistribution(actor, self.config).sample_actions(h.batch.to(device), h.masks)
        target = torch.tensor([[[1., 2., 3., 4.]], [[-1., -2., -3., -4.]]],
                              device=device, requires_grad=True)
        c = self.credit(target)
        loss = h.loss(policy, c.normalized_advantage[0])
        (-loss.branch_surrogate[..., 1].mean()).backward()
        self.assertGreater(gradient_sum(actor.tx_upper_private), 0)
        self.assertEqual(gradient_sum(critic), 0)
        self.assertEqual(gradient_sum(actor.cpu_joint_scheduler), 0)
        self.assertEqual(gradient_sum(actor.action_heads['tx_select']), 0)
        self.assertEqual(gradient_sum(actor.route_gru), 0)
        self.assertIsNone(target.grad)

    def test_cpu_actual_backward(self):
        self.gradient_check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_actual_backward(self):
        self.gradient_check('cuda')

    def recurrent_check(self, device):
        h = self.helper()
        actor, critic = self.model()
        buffer = h.buffer(actor, critic)
        prepared = build_recurrent_ppo_minibatch(buffer, self.config)
        self.assertEqual(prepared.return_target.shape, (8, 32, 4))
        torch.testing.assert_close(prepared.rollout_indices.flatten(), torch.arange(256))
        expected = prepared.return_target.sum(-1) - prepared.old_value.sum(-1)
        # This is a 256-slot GAE, not eight independent 32-slot estimates.
        self.assertAlmostEqual(float(expected.flatten()[31]),
                               4 * sum(.9405**k for k in range(225)), places=3)
        actor.to(device)
        critic.to(device)
        updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, self.config)
        from src.models.tx_upper_counterfactual import TXUpperCounterfactualCriticV1, TXQTargetNormalizerV1
        with patch('src.models.tx_upper_factual_credit.factual_team_credit', wraps=factual_team_credit) as credit_call, \
             patch.object(TXUpperCounterfactualCriticV1, 'forward', side_effect=AssertionError('Q forbidden')), \
             patch.object(TXQTargetNormalizerV1, 'update_and_compensate', side_effect=AssertionError('PopArt forbidden')):
            result = updater.update(buffer)
        self.assertEqual(credit_call.call_count, 1)
        self.assertTrue(result.old_policy_snapshot_preserved)
        epochs = updater.last_integrated_summary['ppo_epochs']
        self.assertEqual(len(epochs), 4)
        self.assertAlmostEqual(epochs[0]['tx_upper']['raw_a_tx_mean'], float(expected.mean()), places=4)
        for epoch in epochs:
            tx = epoch['tx_upper']
            for key in ('raw_a_tx_mean', 'raw_a_tx_std', 'normalized_a_tx_mean',
                        'normalized_a_tx_std', 'normalization_mean', 'normalization_std'):
                self.assertEqual(tx[key], epochs[0]['tx_upper'][key])
            self.assertIsNone(tx['q_critic_loss'])
            self.assertIsNone(tx['running_count'])
            self.assertEqual(tx['q_status'], 'inactive')
            self.assertGreaterEqual(tx['tx_grad_norm_before_clip'], 0)
            self.assertGreaterEqual(tx['tx_update_norm'], 0)
            for key in ('approx_kl', 'clip_fraction', 'entropy', 'p_transmit_mean',
                        'deterministic_transmit_fraction'):
                self.assertIsNotNone(tx[key])
        json.dumps(epochs, allow_nan=False)
        self.assertFalse(any('q' in k for k in critic.state_dict()))
        self.assertEqual(len(updater.optimizers.critic_optimizer.state),
                         len(tuple(critic.value_network.parameters())))

    def test_cpu_four_epoch_frozen_credit_and_no_q_popart(self):
        self.recurrent_check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_four_epoch_recurrent_update(self):
        self.recurrent_check('cuda')

    def test_original_value_identical_update_to_e11(self):
        actor, critic = self.model()
        old_actor, old = self.model(self.e11)
        buffer = self.helper().buffer(actor, critic)
        CAGATMAPPORecurrentPPOUpdater(actor, critic, self.config).update(buffer)
        CAGATMAPPORecurrentPPOUpdater(old_actor, old, self.e11).update(buffer)
        for key, value in old.value_network.state_dict().items():
            torch.testing.assert_close(value, critic.value_network.state_dict()[key], rtol=0, atol=0)

    def test_zero_active_recurrent_update_has_na_credit_and_finite_losses(self):
        from tests.test_tx_q_popart import TXQPopArtTests
        h = TXQPopArtTests()
        h.config = self.config
        actor, critic = self.model()
        buffer = h.buffer(actor, critic, tx_active=False)
        updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, self.config)
        updater.update(buffer)
        for epoch in updater.last_integrated_summary['ppo_epochs']:
            tx = epoch['tx_upper']
            self.assertEqual(tx['credit_active_samples'], 0)
            self.assertIsNone(tx['raw_a_tx_mean'])
            self.assertIsNone(tx['approx_kl'])
            self.assertIsNone(tx['q_critic_loss'])
        json.dumps(updater.last_integrated_summary, allow_nan=False)

    def test_partial_rollout_roundtrip(self):
        actor, critic = self.model()
        buffer = self.helper().buffer(actor, critic, 8)
        state = serialize_active_rollout(buffer, 0, 'PERIODIC_RESUME')
        restored, _ = restore_active_rollout(self.config, state, checkpoint_kind='PERIODIC_RESUME', policy_version=0)
        from tests.test_ca_gat_mappo_checkpoint import assert_nested_equal
        assert_nested_equal(self, state, serialize_active_rollout(restored, 0, 'PERIODIC_RESUME'))

    def test_full_checkpoint_adam_roundtrip_and_next_synthetic_update(self):
        from tests.test_ca_gat_mappo_checkpoint import make_payload, training_state, assert_nested_equal
        config = preserve_git(replace(self.config,
            environment=replace(self.config.environment, episode_horizon=512),
            training=replace(self.config.training, mappo=replace(self.config.training.mappo,
                training_device='cpu', checkpoint_interval_steps=4096, max_training_environment_steps=32768))), self.config)
        actor, critic = self.model(config)
        updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, config)
        buffer = self.helper(config).buffer(actor, critic)
        updater.update(buffer)
        payload, *_ = make_payload(config, 'FINAL_COMPLETED', actor=actor, critic=critic, updater=updater)
        payload['checkpoint_kind'] = 'PERIODIC_RESUME'
        payload['training_state'] = training_state(config, 'PERIODIC_RESUME', 0)
        with tempfile.TemporaryDirectory(dir=self.output) as directory:
            path = Path(directory)/'unit.pt'
            atomic_save_checkpoint(payload, path)
            saved = load_checkpoint_payload(path)
            a, c = self.model(config)
            u = CAGATMAPPORecurrentPPOUpdater(a, c, config)
            restore_model_optimizer_and_rng(config=config, payload=saved, actor=a, critic=c,
                                            optimizers=u.optimizers, dtype=torch.float32)
            assert_nested_equal(self, actor.state_dict(), a.state_dict())
            assert_nested_equal(self, critic.state_dict(), c.state_dict())
            assert_nested_equal(self, updater.optimizers.actor_optimizer.state_dict(), u.optimizers.actor_optimizer.state_dict())
            assert_nested_equal(self, updater.optimizers.critic_optimizer.state_dict(), u.optimizers.critic_optimizer.state_dict())
            updater.update(buffer)
            u.update(buffer)
            assert_nested_equal(self, actor.state_dict(), a.state_dict())
            assert_nested_equal(self, critic.state_dict(), c.state_dict())
            assert_nested_equal(self, updater.optimizers.actor_optimizer.state_dict(), u.optimizers.actor_optimizer.state_dict())
            final, *_ = make_payload(config, 'FINAL_COMPLETED', actor=actor, critic=critic, updater=updater)
            final_path = Path(directory)/'unit-final.pt'
            atomic_save_checkpoint(final, final_path)
            from src.evaluation.actor_loader import load_final_actor_for_evaluation
            ev = preserve_git(replace(config, mode='evaluation', launch_profile=None), config)
            loaded = load_final_actor_for_evaluation(ev, final_path, evaluation_device='cpu')
            assert_nested_equal(self, actor.state_dict(), loaded.actor.state_dict())
        contaminated = copy.deepcopy(payload)
        contaminated['model_state']['critic']['tx_upper_q.network.0.weight'] = torch.zeros(1)
        with self.assertRaisesRegex(CheckpointError, 'must not contain Q'):
            validate_checkpoint_payload(contaminated)

    def test_e10_e11_cross_mode_resume_rejected_and_real_legacy_loaders(self):
        from tests.test_ca_gat_mappo_checkpoint import make_payload, training_state, assert_nested_equal
        from src.evaluation.actor_loader import load_final_actor_for_evaluation
        for mode, normalization in (('counterfactual_q_v1', 'none'),
                                    ('counterfactual_q_v1', 'popart_v1')):
            target = preserve_git(replace(self.config,
                environment=replace(self.config.environment, episode_horizon=512),
                training=replace(self.config.training, mappo=replace(self.config.training.mappo,
                    training_device='cpu', checkpoint_interval_steps=4096,
                    max_training_environment_steps=32768))), self.config)
            source = preserve_git(replace(target, training=replace(target.training,
                mappo=replace(target.training.mappo, tx_upper_credit_mode=mode,
                              tx_q_target_normalization=normalization))), target)
            payload, *_ = make_payload(source, 'FINAL_COMPLETED')
            payload['checkpoint_kind'] = 'PERIODIC_RESUME'
            payload['training_state'] = training_state(source, 'PERIODIC_RESUME', 0)
            validate_checkpoint_payload(payload)
            actor, critic = self.model(target)
            updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, target)
            before = copy.deepcopy(actor.state_dict())
            with self.assertRaises(CheckpointError):
                restore_model_optimizer_and_rng(config=target, payload=payload, actor=actor, critic=critic,
                                                optimizers=updater.optimizers, dtype=torch.float32)
            assert_nested_equal(self, before, actor.state_dict())
            self.assertFalse(updater.optimizers.actor_optimizer.state)

    def test_trainer_writes_only_compact_and_telemetry_does_not_change_update(self):
        from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
        from src.models.tx_upper_factual_credit import TXFactualCredit
        with tempfile.TemporaryDirectory(dir=self.output) as directory:
            config = preserve_git(replace(self.config,
                training=replace(self.config.training, mappo=replace(self.config.training.mappo, training_device='cpu')),
                output=replace(self.config.output, logs_dir=directory)), self.config)
            trainer = CAGATMAPPOTrainer(config, progress_logger=lambda _: None,
                environment_factory=lambda _: self.fail('no environment collection allowed'))
            buffer = self.helper(config).buffer(trainer.actor, trainer.critic)
            reference_buffer = copy.deepcopy(buffer)
            trainer.rollout_buffer = buffer
            trainer._rollout_policy_version = trainer.policy_version
            trainer._transitions = 256
            rng = torch.get_rng_state().clone()
            trainer._perform_update(trainer._updates)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            path = Path(config.artifact_paths()['raw_metrics']).parent/'integrated_scheduler_updates_v1.jsonl'
            self.assertEqual(list(path.parent.glob('*.jsonl')), [path])
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['ppo_epochs'][0]['tx_upper']['credit_mode'], 'factual_team_gae_v1')
            actor, critic = self.model(config)
            updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, config)
            with patch.object(TXFactualCredit, 'telemetry', return_value={}):
                updater.update(reference_buffer)
            for k, v in actor.state_dict().items():
                torch.testing.assert_close(v, trainer.actor.state_dict()[k], rtol=0, atol=0)
            for k, v in critic.state_dict().items():
                torch.testing.assert_close(v, trainer.critic.state_dict()[k], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
