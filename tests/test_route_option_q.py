"""Bounded E14d unit/optimizer/serialization tests; no training run."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.config import ConfigError, load_run_config
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic, HierarchicalOptionAwareRouteDecoderV1
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution, SequentialActionMaskBatch
from src.models.ca_gat_mappo_runtime import CAGATMAPPOFactualActorRuntime
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer, CAGATMAPPORolloutChunk
from src.models.ca_gat_mappo_update import CAGATMAPPORecurrentPPOUpdater
from src.models.ca_gat_mappo_checkpoint import (atomic_save_checkpoint, load_checkpoint_payload,
    serialize_active_rollout, restore_active_rollout)
from src.evaluation.actor_loader import load_actor_for_development_diagnostic
from src.models.route_option_q import (MODE, RouteOptionQ, counterfactual_values, prepare_route_q,
    route_q_loss, epoch_q_records, calibration, write_route_q_telemetry)
from src.models.route_task_credit import TaskCreditBatch, KEY
from tests.test_route_exploration import E13, E14A, E14B, CONFIGS, changed, route_active_step
from tests.test_route_task_credit import E14C, synthetic_buffer
from tests.test_ca_gat_mappo_checkpoint import make_payload

E14D = CONFIGS / 'e14d-v1-route-counterfactual-q-seed42-8k-dev.json'


def tiny_batch(beta=1., q_values=None):
    """Three factual samples in one current-state batch, one per option type."""
    p = torch.tensor([[[[0., .2, .3, .5, 0., 0.]] * 4]])
    masks = p > 0
    features = torch.zeros(1, 1, 4, 6, 3)
    # Independent MLP test fixture can distinguish options; no outcome input.
    features[..., 0] = torch.arange(6)
    choices = torch.tensor([[[1, 2, 3, 1]]])
    policy = SimpleNamespace(route_candidate_features=features, probabilities={'route': p},
        action_masks={'route': masks}, action_indices={'route': choices})
    samples = []
    for u, category in enumerate(('local', 'defer', 'remote')):
        samples.append(dict(episode_id=0, source_uav=u, task_id=10+u, rollout_offset=0,
            global_rollout_index=0, route_slot=0, category=category, shared_advantage=.4,
            route_specific_advantage=10., combined_advantage=10.4, coefficient=1.,
            value=.1, **{'return': 1. if u != 1 else -.5}, route_action_index=int(choices[0, 0, u]),
            actor_leaf_probability=p[0, 0, u].tolist(), route_legal_mask=masks[0, 0, u].tolist(),
            actor_remote_conditional_probability=[1., 0., 0.],
            terminal_status='completed' if u != 1 else 'expired', terminal_slot=5,
            route_terminal_delay=5, censored=False))
    task = TaskCreditBatch(torch.full((1, 4), 10.4), torch.full((1, 4), 10.), samples,
                           torch.tensor([], dtype=torch.long), torch.empty(0, 18), torch.empty(0))
    q = RouteOptionQ(2, 3, 8, 42)
    if q_values is not None:
        def forward(state, candidates, legal):
            return candidates.new_tensor(q_values).expand(legal.shape).masked_fill(~legal, 0)
        q.forward = forward
    critic = SimpleNamespace(route_option_q=q)
    batch = prepare_route_q(critic, policy, torch.zeros(1, 1, 2), task, beta)
    return critic, policy, task, batch


def q_buffer(config, actor, critic):
    """Add collector-equivalent leaf snapshots to the inherited bounded fixture."""
    old = synthetic_buffer(config, actor, critic)
    result = CAGATMAPPORolloutBuffer(config)
    distribution = CAGATMAPPOActionDistribution(actor, config)
    for i in range(len(old)):
        tr = old.transition_at(i)
        chunk = CAGATMAPPORolloutChunk((tr,))
        policy = distribution.evaluate_actions(chunk.actor_batch, chunk.action_mask_batch, ((tr.proposal_actions,),),
                                                tr.hidden_in.unsqueeze(0))
        md = copy.deepcopy(tr.rejection_or_downgrade_summary)
        for d in md[KEY]['decisions']:
            u = d['source_uav']
            d.update(actor_leaf_probability=policy.probabilities['route'][0, 0, u].detach().tolist(),
                     route_legal_mask=policy.action_masks['route'][0, 0, u].tolist(),
                     route_action_index=int(policy.action_indices['route'][0, 0, u]))
            rm = policy.action_masks['route'][0, 0, u, 3:]
            rl = policy.raw_logits['route'][0, 0, u, 3:].detach()
            d['actor_remote_conditional_probability'] = (
                rl.masked_fill(~rm, -torch.inf).softmax(-1) if rm.any() else torch.zeros_like(rl)).tolist()
        result.append(replace(tr, rejection_or_downgrade_summary=md))
    result.finalize()
    return result


class RouteOptionQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.c = load_run_config(E14C, cli_overrides={'training.mappo.training_device': 'cpu'})
        cls.d = load_run_config(E14D, cli_overrides={'training.mappo.training_device': 'cpu'})

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_local_defer_remote_have_explicit_legal_q(self):
        _, _, _, b = tiny_batch(q_values=[0., 2., 1., 4., 999., 999.])
        for r in b.records:
            self.assertEqual(set(r['frozen_q']), {'Local', 'Defer', next(
                x for x in r['legal_options'] if x.startswith('Remote-'))})
            self.assertEqual((r['frozen_q']['Local'], r['frozen_q']['Defer']), (2., 1.))

    def test_illegal_options_never_enter_head_or_baseline(self):
        q = RouteOptionQ(2, 3, 8, 42)
        features = torch.randn(2, 5, 3); mask = torch.tensor([[1, 0, 1, 0, 1]]*2, dtype=torch.bool)
        features[~mask] = float('nan')
        calls = []
        handle = q.network.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[0]))
        out = q(torch.zeros(2, 2), features, mask); handle.remove()
        self.assertEqual(calls, [6]); self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((out[~mask] == 0).all())

    def test_hierarchical_remote_aggregation_and_baseline(self):
        q = torch.tensor([[0., 2., 1., 4., 8., 999.]])
        p = torch.tensor([[0., .2, .3, .2, .3, 0.]])
        b, cf, remote = counterfactual_values(q, p, p > 0, torch.tensor([3]))
        self.assertAlmostEqual(float(remote), 6.4, places=6)
        self.assertAlmostEqual(float(b), .2*2+.3*1+.5*6.4, places=6)
        self.assertAlmostEqual(float(cf), 4-float(b), places=6)

    def test_remote_unavailable_no_nan_no_hidden_contribution(self):
        q = torch.tensor([[0., 2., -1., float('nan')]])
        p = torch.tensor([[0., .4, .6, 0.]])
        b, cf, remote = counterfactual_values(q, p, p > 0, torch.tensor([1]))
        self.assertAlmostEqual(float(b), .2, places=6); self.assertEqual(float(remote), 0.)
        self.assertAlmostEqual(float(cf), 1.8, places=6)

    def test_legal_remote_underflow_keeps_conditional_q(self):
        q = torch.tensor([[0., 2., 1., 4., 8.]])
        p = torch.tensor([[0., .4, .6, 0., 0.]])
        legal = torch.tensor([[False, True, True, True, True]])
        b, _, remote = counterfactual_values(q, p, legal, torch.tensor([1]), torch.tensor([[.25, .75]]))
        self.assertAlmostEqual(float(b), 1.4, places=6); self.assertEqual(float(remote), 7.)

    def test_candidate_permutation_equivariance(self):
        q = RouteOptionQ(2, 3, 8, 42)
        with torch.no_grad(): q.network[-1].weight.fill_(.2)
        features = torch.randn(3, 7, 3); state = torch.randn(3, 2)
        mask = torch.rand(3, 7) > .2; permutation = torch.tensor([6, 1, 0, 4, 2, 5, 3])
        a = q(state, features, mask)[:, permutation]
        b = q(state, features[:, permutation], mask[:, permutation])
        torch.testing.assert_close(a, b)

    def test_variable_candidate_count_no_fixed_output_width(self):
        q = RouteOptionQ(2, 3, 8, 42)
        for count in (2, 3, 5, 9):
            self.assertEqual(q(torch.zeros(2, 2), torch.zeros(2, count, 3),
                               torch.ones(2, count, dtype=torch.bool)).shape, (2, count))

    def test_chosen_targets_exactly_reuse_event_return(self):
        _, _, task, b = tiny_batch()
        torch.testing.assert_close(b.targets, torch.tensor([s['return'] for s in task.samples]))
        self.assertEqual([r['supervised_option'] for r in b.records], ['Local', 'Defer', 'Remote-UAV0'])

    def test_unchosen_options_have_no_supervised_target(self):
        critic, _, _, b = tiny_batch()
        loss, q = route_q_loss(critic, b); q.retain_grad(); loss.backward()
        selected = torch.zeros_like(b.legal); selected.scatter_(-1, b.chosen.unsqueeze(-1), True)
        self.assertTrue((q.grad[~selected] == 0).all())
        self.assertTrue((q.grad[selected] != 0).all())
        self.assertTrue(all('unchosen_target' not in r for r in b.records))

    def test_local_remote_and_defer_supervision_symmetric(self):
        critic, _, _, b = tiny_batch()
        loss, _ = route_q_loss(critic, b)
        self.assertAlmostEqual(float(loss.detach()), .5*(1+1+.25)/3, places=6)

    def test_final_advantage_replaces_not_adds_task_advantage(self):
        _, _, task, b = tiny_batch(beta=.7, q_values=[0., 2., 1., 4., 0., 0.])
        for n, r in enumerate(b.records):
            self.assertAlmostEqual(r['final_route_advantage'], .4+.7*float(b.counterfactual_advantage[n]), places=6)
            self.assertEqual(r['task_advantage'], 10.); self.assertEqual(r['coefficient'], 0.)
        self.assertEqual(float(b.advantage[0, 3]), float(task.advantage[0, 3]))

    def test_zero_beta_shared_only_not_e14c_task_credit(self):
        _, _, _, b = tiny_batch(beta=0., q_values=[0., 2., 1., 4., 0., 0.])
        torch.testing.assert_close(b.advantage[0, :3], torch.full((3,), .4))

    def test_q_targets_baseline_and_advantage_detached_and_frozen(self):
        critic, _, _, b = tiny_batch()
        snapshots = [x.clone() for x in (b.targets, b.baseline, b.frozen_q, b.counterfactual_advantage, b.advantage)]
        optimizer = torch.optim.Adam(critic.route_option_q.parameters(), lr=.01)
        for _ in range(4):
            optimizer.zero_grad(); route_q_loss(critic, b)[0].backward(); optimizer.step()
        for snapshot, actual in zip(snapshots, (b.targets, b.baseline, b.frozen_q, b.counterfactual_advantage, b.advantage)):
            self.assertFalse(actual.requires_grad); self.assertTrue(torch.equal(snapshot, actual))

    def test_q_gradient_never_enters_actor_features_or_state(self):
        q = RouteOptionQ(2, 3, 8, 42)
        x = torch.randn(2, 2, requires_grad=True); c = torch.randn(2, 4, 3, requires_grad=True)
        q(x, c, torch.ones(2, 4, dtype=torch.bool)).sum().backward()
        self.assertIsNone(x.grad); self.assertIsNone(c.grad)

    def test_critic_head_preserves_actor_base_critic_task_value_rng(self):
        rng = torch.get_rng_state().clone()
        old, new = MAPPOCentralizedCritic(self.c), MAPPOCentralizedCritic(self.d)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, v in old.state_dict().items(): self.assertTrue(torch.equal(v, new.state_dict()[k]), k)
        a, b = CAGATMAPPOActor(self.c), CAGATMAPPOActor(self.d)
        for k, v in a.state_dict().items(): self.assertTrue(torch.equal(v, b.state_dict()[k]), k)
        self.assertFalse(hasattr(old, 'route_option_q'))

    def test_configuration_only_new_route_q_and_output(self):
        c, d = self.c.resolved_dict(), self.d.resolved_dict()
        for k in ('route_value_mode', 'route_counterfactual_credit_coefficient', 'route_q_value_loss_coefficient'):
            d['training']['mappo'].pop(k)
        d['output'] = c['output']; self.assertEqual(c, d)
        self.assertEqual(self.d.training.mappo.cpu_anti_starvation_mode, 'deadline_feasibility_v1')
        self.assertEqual(self.d.training.mappo.tx_upper_credit_mode, 'paired_shadow_v1')

    def test_invalid_configuration_rejected(self):
        for kwargs in ({'route_value_mode': 'unknown'}, {'route_q_value_loss_coefficient': 0.},
            {'route_q_value_loss_coefficient': float('nan')}, {'route_counterfactual_credit_coefficient': -1.},
            {'route_counterfactual_credit_coefficient': True}, {'route_credit_mode': 'shared_gae'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigError): changed(self.d, **kwargs).validate()

    def test_off_preserves_historical_config_identity(self):
        off = changed(self.c, route_counterfactual_credit_coefficient=8., route_q_value_loss_coefficient=2.)
        self.assertEqual(self.c.config_hash, off.config_hash)
        self.assertEqual(load_run_config(E14C).config_hash[:12], '998dfd95b691')
        self.assertEqual(load_run_config(E14B).config_hash[:12], '3688fe21edd9')
        self.assertEqual(load_run_config(E14A).config_hash[:12], '4f38ccf24177')

    def test_representation_interface_preserves_all_logits_hidden_masks_decode(self):
        env = U2UMECEnvironment(self.d); current = route_active_step(env)
        actor = CAGATMAPPOActor(self.d)
        runtime = CAGATMAPPOFactualActorRuntime(actor, self.d, device='cpu')
        factual = runtime.sample_step(current.observations, actor.initial_hidden(1), episode_start=False)
        p = factual.action_output
        replay = runtime.action_distribution.evaluate_actions(factual.actor_batch, factual.action_masks,
            p.proposals, actor.initial_hidden(1), return_route_candidates=True)
        self.assertEqual(p.proposals, replay.proposals); self.assertTrue(torch.equal(p.hidden_out, replay.hidden_out))
        self.assertIsNone(p.route_candidate_features)
        self.assertFalse(replay.route_candidate_features.requires_grad)
        self.assertEqual(replay.route_candidate_features.shape[-1], HierarchicalOptionAwareRouteDecoderV1.CANDIDATE_VALUE_DIMENSION)
        for branch in p.raw_logits:
            self.assertTrue(torch.equal(p.raw_logits[branch], replay.raw_logits[branch]), branch)
            self.assertTrue(torch.equal(p.action_masks[branch], replay.action_masks[branch]), branch)
            self.assertTrue(torch.equal(p.branch_log_probs[branch], replay.branch_log_probs[branch]), branch)

    def test_evaluation_never_calls_q_and_off_stochastic_identity(self):
        env = U2UMECEnvironment(self.d); current = route_active_step(env)
        outputs = []
        for config in (self.c, self.d, changed(self.c, route_q_value_loss_coefficient=3.)):
            actor = CAGATMAPPOActor(config); runtime = CAGATMAPPOFactualActorRuntime(actor, config, device='cpu')
            stochastic = runtime.sample_step(current.observations, actor.initial_hidden(1), episode_start=False)
            with patch.object(RouteOptionQ, 'forward', side_effect=AssertionError('Q in evaluation')):
                deterministic = runtime.action_distribution.deterministic_actions(stochastic.actor_batch, stochastic.action_masks)
            outputs.append((stochastic, deterministic))
        for stochastic, deterministic in outputs[1:]:
            self.assertEqual(stochastic.action_output.proposals, outputs[0][0].action_output.proposals)
            self.assertEqual(deterministic.proposals, outputs[0][1].proposals)
            self.assertTrue(torch.equal(stochastic.action_output.hidden_out, outputs[0][0].action_output.hidden_out))

    def test_future_terminal_labels_only_change_targets_not_q_inputs(self):
        critic, policy, task, original = tiny_batch()
        task.samples[0].update(terminal_status='expired', terminal_slot=99, **{'return': -7.})
        altered = prepare_route_q(critic, policy, torch.zeros(1, 1, 2), task, 1.)
        for name in ('state', 'candidates', 'frozen_q', 'baseline', 'counterfactual_advantage'):
            self.assertTrue(torch.equal(getattr(original, name), getattr(altered, name)), name)
        self.assertNotEqual(float(original.targets[0]), float(altered.targets[0]))

    def test_missing_leaf_snapshot_fails_closed(self):
        critic, policy, task, _ = tiny_batch(); task.samples[0].pop('actor_leaf_probability')
        with self.assertRaisesRegex(ValueError, 'factual Route leaf'): prepare_route_q(
            critic, policy, torch.zeros(1, 1, 2), task, 1.)

    def test_illegal_probability_and_chosen_action_rejected(self):
        q = torch.zeros(1, 5); p = torch.tensor([[0., .3, .2, .5, 0.]])
        with self.assertRaisesRegex(ValueError, 'chosen'): counterfactual_values(q, p, p > 0, torch.tensor([4]))
        with self.assertRaisesRegex(ValueError, 'illegal'): counterfactual_values(q, p, torch.tensor([[0, 1, 1, 0, 0]], dtype=torch.bool), torch.tensor([1]))

    def test_no_route_samples_zero_loss(self):
        critic, policy, task, _ = tiny_batch(); task.samples = []
        b = prepare_route_q(critic, policy, torch.zeros(1, 1, 2), task, 1.)
        loss, _ = route_q_loss(critic, b); self.assertEqual(float(loss.detach()), 0.)
        loss.backward(); self.assertEqual(b.records, [])

    def test_telemetry_ratio_surrogate_and_calibration_conserve(self):
        _, _, _, b = tiny_batch(q_values=[0., 2., 1., 4., 0., 0.])
        actor_rows = [{**s, 'epoch_index': 0, 'old_log_prob': math.log(.2), 'new_log_prob': math.log(.3),
                       'ppo_ratio': 1.5, 'unclipped_surrogate': 1.5*s['combined_advantage'],
                       'clipped_surrogate': 1.2*s['combined_advantage']} for s in b.credit_samples]
        rows = epoch_q_records(b, b.frozen_q, actor_rows)
        for r in rows:
            self.assertAlmostEqual(r['ppo_ratio'], math.exp(r['new_log_prob']-r['old_log_prob']))
            self.assertAlmostEqual(sum(r['frozen_leaf_probability'].values()), 1., places=6)
            self.assertAlmostEqual(sum(r['destination_conditional_probability'].values()), 1., places=6)
            self.assertEqual(r['clipped_surrogate'], 1.2*r['final_route_advantage'])
        self.assertEqual(calibration(rows)['local']['mae'], 1.)
        self.assertEqual(sum(g['unique_factual_count'] for g in calibration(rows).values()), 3)
        json.dumps(rows, allow_nan=False)

    def test_new_and_all_historical_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            for file in (E13, E14A, E14B, E14C, E14D):
                c = load_run_config(file, cli_overrides={'training.mappo.training_device': 'cpu'})
                payload, _, critic, *_ = make_payload(c, 'FINAL_COMPLETED')
                p = Path(directory)/f'{file.stem}.pt'; atomic_save_checkpoint(payload, p)
                loaded = load_checkpoint_payload(p)
                v = MAPPOCentralizedCritic(c); v.load_state_dict(loaded['model_state']['critic'])
                for k, value in critic.state_dict().items(): self.assertTrue(torch.equal(value, v.state_dict()[k]), k)
                evaluation = replace(c, mode='evaluation', launch_profile=None)
                for name in ('git_commit', 'git_branch', 'git_dirty'): object.__setattr__(evaluation, name, getattr(c, name))
                actor = load_actor_for_development_diagnostic(evaluation, p, evaluation_device='cpu').actor
                self.assertFalse(hasattr(actor, 'route_option_q'))

    def test_production_four_epochs_frozen_credit_ratio_and_isolated_aux_update(self):
        c = replace(self.d, environment=replace(self.d.environment, arrival_probabilities=(1.,)*4))
        actor, critic = CAGATMAPPOActor(c), MAPPOCentralizedCritic(c)
        # Nonzero old Q exercises CF immediately, rather than only cold zero.
        with torch.no_grad(): critic.route_option_q.network[-1].weight.fill_(.02)
        buffer = q_buffer(c, actor, critic)
        hidden = [buffer.transition_at(i).hidden_in.clone() for i in range(len(buffer))]
        q_state = copy.deepcopy(critic.route_option_q.state_dict())
        updater = CAGATMAPPORecurrentPPOUpdater(actor, critic, c)
        rng = torch.get_rng_state().clone()
        result = updater.update(buffer, collected_environment_steps=256)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        rows = updater.last_route_option_q_records
        n = sum(int(buffer.transition_at(i).active_branch_indicators[:, 0].sum()) for i in range(len(buffer)))
        self.assertEqual(len(result.epoch_diagnostics), 4); self.assertEqual(len(rows), 4*n)
        self.assertEqual(len(updater.last_route_option_q_losses), 4)
        groups = {}
        for r in rows:
            groups.setdefault((r['episode_id'], r['rollout_offset'], r['source_uav']), []).append(r)
            self.assertAlmostEqual(r['ppo_ratio'], math.exp(r['new_log_prob']-r['old_log_prob']), places=5)
            self.assertAlmostEqual(r['final_route_advantage'], r['shared_advantage']+r['counterfactual_advantage'], places=5)
        for rr in groups.values():
            for name in ('frozen_q', 'counterfactual_baseline', 'counterfactual_advantage', 'final_route_advantage', 'G_event'):
                self.assertEqual([r[name] for r in rr], [rr[0][name]]*4)
        self.assertTrue(any(not torch.equal(v, critic.route_option_q.state_dict()[k]) for k, v in q_state.items()))
        self.assertTrue(all(torch.equal(h, buffer.transition_at(i).hidden_in) for i, h in enumerate(hidden)))
        self.assertIsNotNone(updater.last_integrated_summary['paired_samples'])
        self.assertEqual({r['coefficient'] for r in updater.last_route_exploration_summary}, {.05*(1-256/6000)})
        legacy_config = changed(c, route_value_mode='off')
        legacy_actor, legacy_critic = CAGATMAPPOActor(legacy_config), MAPPOCentralizedCritic(legacy_config)
        legacy_updater = CAGATMAPPORecurrentPPOUpdater(legacy_actor, legacy_critic, legacy_config)
        legacy_updater.update(buffer, collected_environment_steps=256)
        for prefix in ('value_network', 'route_task_value'):
            for k, value in getattr(legacy_critic, prefix).state_dict().items():
                self.assertTrue(torch.equal(value, getattr(critic, prefix).state_dict()[k]), (prefix, k))
        # Generic model/Adam checkpoint includes the trained independent Q.
        payload, *_ = make_payload(c, 'FINAL_COMPLETED', actor=actor, critic=critic, updater=updater)
        self.assertEqual(payload['config_snapshot']['training']['mappo']['route_value_mode'], MODE)
        fresh = MAPPOCentralizedCritic(c); fresh.load_state_dict(payload['model_state']['critic'])
        fresh_updater = CAGATMAPPORecurrentPPOUpdater(CAGATMAPPOActor(c), fresh, c)
        fresh_updater.optimizers.critic_optimizer.load_state_dict(payload['optimizer_state']['critic_adam'])
        for p, q in zip(critic.route_option_q.parameters(), fresh.route_option_q.parameters()):
            self.assertTrue(torch.equal(p, q))
            for key, value in updater.optimizers.critic_optimizer.state[p].items():
                other = fresh_updater.optimizers.critic_optimizer.state[q][key]
                self.assertTrue(torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other)
        partial = CAGATMAPPORolloutBuffer(c); partial.append(buffer.transition_at(0))
        serialized = serialize_active_rollout(partial, 0, 'PERIODIC_RESUME')
        restored, _ = restore_active_rollout(c, serialized, checkpoint_kind='PERIODIC_RESUME', policy_version=0)
        self.assertEqual(restored.transition_at(0).rejection_or_downgrade_summary[KEY],
                         partial.transition_at(0).rejection_or_downgrade_summary[KEY])
        with tempfile.TemporaryDirectory() as directory:
            local = replace(c, output=replace(c.output, logs_dir=directory))
            row = write_route_q_telemetry(local, rows, 256, updater.last_route_option_q_losses, buffer=buffer)
            self.assertEqual(sum(g['unique_factual_count'] for g in row['calibration'].values()), n)
            self.assertEqual(len(row['route_decisions']), n)
            tail = write_route_q_telemetry(local, [], 256, [], buffer=buffer, optimized=False)
            self.assertEqual(tail['samples'], []); self.assertEqual(len(tail['task_terminals']), n)

    def test_off_mode_bitwise_e14c_optimizer(self):
        c = replace(self.c, environment=replace(self.c.environment, arrival_probabilities=(1.,)*4))
        off = changed(c, route_counterfactual_credit_coefficient=3., route_q_value_loss_coefficient=4.)
        actors = []; critics = []; updates = []
        for config in (c, off):
            a, v = CAGATMAPPOActor(config), MAPPOCentralizedCritic(config)
            u = CAGATMAPPORecurrentPPOUpdater(a, v, config)
            u.update(synthetic_buffer(config, a, v), collected_environment_steps=256)
            actors.append(a); critics.append(v); updates.append(u)
        for k, value in actors[0].state_dict().items(): self.assertTrue(torch.equal(value, actors[1].state_dict()[k]), k)
        for k, value in critics[0].state_dict().items(): self.assertTrue(torch.equal(value, critics[1].state_dict()[k]), k)
        self.assertEqual(updates[0].last_integrated_summary, updates[1].last_integrated_summary)
        self.assertEqual(updates[1].last_route_option_q_records, [])

    def test_expected_counterfactual_advantage_is_zero_and_shift_invariant(self):
        p = torch.tensor([[0., .2, .3, .1, .4]])
        q = torch.tensor([[0., 2., -1., 4., 3.]])
        b, _, _ = counterfactual_values(q, p, p > 0, torch.tensor([1]))
        self.assertAlmostEqual(float(((q-b.unsqueeze(-1))*p).sum()), 0., places=6)
        _, a, _ = counterfactual_values(q, p, p > 0, torch.tensor([1]))
        _, shifted, _ = counterfactual_values(q+5, p, p > 0, torch.tensor([1]))
        torch.testing.assert_close(a, shifted)

    def test_only_route_advantage_changes_tx_cpu_surrogates_unchanged(self):
        from tests.test_ca_gat_mappo_route_specific_gae import ppo_inputs, low_level_ppo
        inputs = ppo_inputs()
        a = low_level_ppo(inputs, route_advantage=None)
        b = low_level_ppo(inputs, route_advantage=inputs['advantage']+2.)
        for name in ('branch_advantage', 'branch_surrogate', 'branch_unclipped_surrogate', 'branch_clipped_surrogate'):
            self.assertTrue(torch.equal(getattr(a, name)[..., 1:], getattr(b, name)[..., 1:]), name)
        self.assertTrue(torch.equal(a.branch_ratio, b.branch_ratio))
        self.assertTrue(torch.equal(a.critic_loss, b.critic_loss))

    def test_collector_leaf_snapshot_observes_real_behavior_without_rng_or_state_change(self):
        from src.models.route_task_credit import capture_task_values, make_step_metadata
        from src.models.ca_gat_mappo import CentralizedStateTensorizer
        env = U2UMECEnvironment(self.d); current = route_active_step(env)
        actor, critic = CAGATMAPPOActor(self.d), MAPPOCentralizedCritic(self.d)
        runtime = CAGATMAPPOFactualActorRuntime(actor, self.d, device='cpu')
        tensorizer = CentralizedStateTensorizer(self.d)
        before = capture_task_values(env, tensorizer.encode_step(current.centralized_state).features, critic, self.d)
        policy = runtime.sample_step(current.observations, actor.initial_hidden(1), episode_start=False).action_output
        result = env.step(policy.proposals[0][0])
        after = capture_task_values(env, tensorizer.encode_step(result.centralized_state).features, critic, self.d)
        snapshot = copy.deepcopy(env.snapshot()); rng = torch.get_rng_state().clone()
        sampling_rng = runtime.policy_generator.get_state().clone()
        md = make_step_metadata(before=before, after=after, info=result.info, policy=policy,
            episode_id=0, environment_step=0, policy_version=0, capture_route_leaves=True)
        self.assertEqual(snapshot, env.snapshot()); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(sampling_rng, runtime.policy_generator.get_state()))
        for d in md['decisions']:
            u, index = d['source_uav'], d['route_action_index']
            self.assertAlmostEqual(math.log(d['actor_leaf_probability'][index]), d['stored_behavior_log_prob'], places=5)
            self.assertEqual(d['route_legal_mask'], policy.action_masks['route'][0, 0, u].tolist())

    def test_task_target_construction_unmodified_by_q_mode(self):
        from src.models.route_task_credit import prepare_task_credit
        actor, critic = CAGATMAPPOActor(self.c), MAPPOCentralizedCritic(self.c)
        buffer = synthetic_buffer(self.c, actor, critic)
        shared = torch.zeros(8, 32, 4)
        c = prepare_task_credit(buffer, shared, self.c)
        d = prepare_task_credit(buffer, shared, self.d)
        self.assertEqual(c.samples, d.samples)
        for name in ('advantage', 'task_advantage', 'task_features', 'value_targets'):
            self.assertTrue(torch.equal(getattr(c, name), getattr(d, name)), name)

    def test_trainer_writes_q_before_clearing_rollout_without_sampling(self):
        from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
        from tests.test_ca_gat_mappo_checkpoint import fixed_update_output
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.d, output=replace(self.d.output, logs_dir=directory),
                             environment=replace(self.d.environment, arrival_probabilities=(1.,)*4))
            updater = SimpleNamespace(update=lambda *a, **k: fixed_update_output(),
                last_route_task_records=[], last_route_task_value_losses=[],
                last_route_option_q_records=[], last_route_option_q_losses=[])
            trainer = CAGATMAPPOTrainer(config, updater_factory=lambda *_: updater)
            trainer.rollout_buffer = synthetic_buffer(config, trainer.actor, trainer.critic)
            trainer._rollout_policy_version = 0; trainer._transitions = 256
            rng = trainer.policy_generator.get_state().clone()
            with patch('src.models.route_exploration.write_route_rollout_telemetry'):
                trainer._perform_update(trainer._updates)
            self.assertTrue(torch.equal(rng, trainer.policy_generator.get_state()))
            self.assertEqual(len(trainer.rollout_buffer), 0)
            path = next(Path(directory).rglob('route_option_q_updates_v1.jsonl'))
            rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 1); self.assertEqual(rows[0]['environment_step_end'], 256)
            self.assertTrue(rows[0]['optimized']); self.assertGreater(len(rows[0]['route_decisions']), 0)


if __name__ == '__main__': unittest.main()
