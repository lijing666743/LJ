"""E14b bounded unit/regression checks; never a development training run."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from src.config import ConfigError, load_run_config
from src.env.environment import U2UMECEnvironment
from src.evaluation.actor_loader import load_actor_for_development_diagnostic
from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic, ACTION_BRANCH_ORDER
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution, SequentialActionMaskBatch
from src.models.ca_gat_mappo_checkpoint import (
    atomic_save_checkpoint, load_checkpoint_payload, serialize_active_rollout, restore_active_rollout)
from src.models.ca_gat_mappo_runtime import CAGATMAPPOFactualActorRuntime
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer
from src.models.ca_gat_mappo_update import CAGATMAPPORecurrentPPOUpdater, RecurrentPPOUpdateError
from src.models.route_exploration import (
    METADATA_KEY, route_top_statistics, route_exploration_coefficient,
    route_exploration_loss, collect_route_training_records,
    summarize_route_records, write_route_rollout_telemetry)
from tests.test_ca_gat_mappo_checkpoint import make_payload, fixed_update_output
from tests import test_tx_paired_shadow_credit as paired_fixture


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'tests/fixtures/configs'
E13 = CONFIGS / 'e13-v1-tx-paired-shadow-seed42-32k.json'
E14A = CONFIGS / 'e14a-v1-cpu-feasibility-seed42-8k-dev.json'
E14B = CONFIGS / 'e14b-v1-route-log-barrier-seed42-8k-dev.json'


def changed(config, **kwargs):
    return replace(config, training=replace(config.training, mappo=replace(
        config.training.mappo, **kwargs)))


def synthetic_policy(logits, mask=None, active=None):
    if mask is None:
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[..., 0] = False
    if active is None:
        active = torch.ones(logits.shape[:-1], dtype=torch.bool)
    return SimpleNamespace(raw_logits={'route': logits}, action_masks={'route': mask},
                           active_branches={'route': active})


def factual_telemetry_buffer(config):
    config = replace(config, environment=replace(
        config.environment, arrival_probabilities=(1.0,) * config.environment.uav_count))
    helper = paired_fixture.PairedTXCreditTests()
    helper.config = config
    actor, critic = CAGATMAPPOActor(config), MAPPOCentralizedCritic(config)
    seed = helper.buffer(actor, critic, length=1, config=config).transition_at(0)
    fixture = helper.helper(config)
    with torch.no_grad():
        policy = CAGATMAPPOActionDistribution(actor, config).evaluate_actions(
            fixture.batch, fixture.masks, ((seed.proposal_actions,),),
            seed.hidden_in.unsqueeze(0))
    records = collect_route_training_records(policy, environment_step=0, episode_id=0, slot=0)
    metadata = dict(seed.rejection_or_downgrade_summary)
    metadata[METADATA_KEY] = records
    buffer = CAGATMAPPORolloutBuffer(config)
    buffer.append(replace(seed, rejection_or_downgrade_summary=metadata))
    return buffer, records


def route_active_step(environment):
    environment.reset()
    for _ in range(100):
        step = environment.step(environment.canonical_proposals())
        masks = SequentialActionMaskBatch.from_observations(step.observations)
        if any(c.route_branch_active for c in masks.flattened()):
            return step
    raise AssertionError('fixed-seed fixture did not reach a Route-active state')


class RouteExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.a = load_run_config(E14A, cli_overrides={'training.mappo.training_device': 'cpu'})
        cls.b = load_run_config(E14B, cli_overrides={'training.mappo.training_device': 'cpu'})

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_historical_config_hashes_are_unchanged(self):
        self.assertEqual(load_run_config(E13).config_hash,
                         '16ed37e1f161f7bdadf44fef120e3cfe26f4142c001e0e3b0dee7e4dd99d5d27')
        self.assertEqual(load_run_config(E14A).config_hash,
                         '4f38ccf24177ad59384424e9e82f7b8aa44be9072655e68b98ca9ebb353c8f6f')
        off = changed(self.a, route_exploration_mode='off',
                      route_exploration_initial_coefficient=.9, route_exploration_anneal_steps=1)
        self.assertEqual(off.config_hash, self.a.config_hash)
        self.assertNotIn('route_exploration_mode', off.resolved_dict()['training']['mappo'])

    def test_development_config_changes_only_one_mechanism_telemetry_and_output(self):
        a, b = self.a.resolved_dict(), self.b.resolved_dict()
        for key in ('route_exploration_mode', 'route_exploration_initial_coefficient',
                    'route_exploration_anneal_steps', 'route_training_telemetry_enabled'):
            b['training']['mappo'].pop(key)
        b['output'] = a['output']
        self.assertEqual(a, b)
        self.assertEqual(self.b.training.mappo.cpu_anti_starvation_mode, 'deadline_feasibility_v1')
        self.assertEqual(self.b.training.mappo.tx_upper_credit_mode, 'paired_shadow_v1')
        self.assertEqual(self.b.training.mappo.max_training_environment_steps, 8000)
        self.assertEqual(self.b.training.mappo.checkpoint_interval_steps, 4000)

    def test_invalid_config_fails_closed(self):
        cases = ({'route_exploration_mode': 'bad'},
                 {'route_exploration_initial_coefficient': float('nan')},
                 {'route_exploration_initial_coefficient': -1},
                 {'route_exploration_initial_coefficient': True},
                 {'route_exploration_initial_coefficient': 0},
                 {'route_exploration_anneal_steps': 0},
                 {'route_exploration_anneal_steps': True},
                 {'route_training_telemetry_enabled': 1},
                 {'route_decoder_mode': 'legacy'})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigError):
                changed(self.b, **kwargs).validate()

    def test_absolute_schedule_releases_policy_before_final_checkpoint(self):
        m = self.b.training.mappo
        for step, expected in ((0,.05),(256,.05*(1-256/6000)),(3000,.025),
                               (6000,0.),(8000,0.),(16000,0.)):
            self.assertAlmostEqual(route_exploration_coefficient(m, step), expected)
        self.assertEqual(route_exploration_coefficient(self.a.training.mappo, 0), 0.)
        for step in (-1, True, 1.5):
            with self.assertRaises(ValueError):route_exploration_coefficient(m, step)

    def test_masked_top_probabilities_sum_remote_leaves_without_candidate_bias(self):
        for remote in ([.6], [.1,.2,.3]):
            values = [0., .1, .3, *remote]
            logits = torch.tensor(values, dtype=torch.float64).clamp_min(1e-200).log()[None,None,None]
            policy = synthetic_policy(logits)
            logs, support = route_top_statistics(logits, policy.action_masks['route'])
            torch.testing.assert_close(logs.exp(), torch.tensor([[[[.1,.6,.3]]]],dtype=torch.float64))
            self.assertTrue(support.all())
            output = route_exploration_loss(policy, self.b.training.mappo, 0, torch.ones(1,1,dtype=torch.bool))
            expected = sum(math.log((1/3)/p) for p in (.1,.6,.3))/3
            self.assertAlmostEqual(float(output.kl_mean), expected)

    def test_unavailable_remote_never_receives_prior_probability_or_gradient(self):
        logits = torch.tensor([[[[0.,-2.,1.,9.,10.]]]], requires_grad=True)
        mask = torch.tensor([[[[False,True,True,False,False]]]])
        p = synthetic_policy(logits, mask)
        logs, support = route_top_statistics(logits, mask)
        self.assertFalse(bool(support[...,1].any()))
        loss = route_exploration_loss(p, self.b.training.mappo, 0, torch.ones(1,1,dtype=torch.bool))
        loss.scaled_loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(logits.grad[...,3:], torch.zeros_like(logits.grad[...,3:])))
        self.assertLess(float(logits.grad[...,1]), 0.)

    def test_recovery_gradient_survives_probability_underflow_and_no_local_samples(self):
        z = torch.tensor([-800.,0.,0.], requires_grad=True)
        # Local, Defer, Remote; no selected action is needed by the regularizer.
        logits = torch.cat((z.new_zeros(1), z))[None,None,None]
        p = synthetic_policy(logits)
        output = route_exploration_loss(p, self.b.training.mappo, 0, torch.ones(1,1,dtype=torch.bool))
        self.assertTrue(torch.isfinite(output.scaled_loss))
        output.scaled_loss.backward()
        self.assertAlmostEqual(float(z.grad[0]), -.05/3, places=7)
        self.assertEqual(float(z.softmax(-1)[0]), 0.)

    def test_no_active_and_invalid_sequence_have_zero_finite_gradient(self):
        for inactive in (True,False):
            logits = torch.tensor([[[[0.,-800.,0.,0.]]]], requires_grad=True)
            mask = torch.tensor([[[[True,False,False,False]]]]) if inactive else None
            p = synthetic_policy(logits, mask, torch.full((1,1,1),not inactive))
            validity = torch.tensor([[inactive]])
            result = route_exploration_loss(p, self.b.training.mappo, 0, validity)
            self.assertEqual(result.active_count, 0)
            self.assertEqual(float(result.scaled_loss), 0.)
            result.scaled_loss.backward()
            self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_prior_is_over_options_and_leaves_destination_conditional_unchanged(self):
        top = torch.tensor([.1,.3,.6],dtype=torch.float64).log().requires_grad_()
        dest = torch.tensor([-.1,.4,.2],dtype=torch.float64,requires_grad=True)
        leaves = torch.cat((top.new_zeros(1), top.log_softmax(-1)[:2],
                            top.log_softmax(-1)[2:] + dest.log_softmax(-1)))
        output = route_exploration_loss(synthetic_policy(leaves[None,None,None]),
                                      self.b.training.mappo, 0, torch.ones(1,1,dtype=torch.bool))
        output.scaled_loss.backward()
        torch.testing.assert_close(dest.grad, torch.zeros_like(dest),atol=1e-14,rtol=0)
        self.assertGreater(float(top.grad.abs().sum()),0.)

    def test_factual_sampler_and_hidden_states_identical_with_mechanism_on(self):
        env = U2UMECEnvironment(self.b)
        step = route_active_step(env)
        ra = CAGATMAPPOFactualActorRuntime(CAGATMAPPOActor(self.a),self.a,device='cpu')
        rb = CAGATMAPPOFactualActorRuntime(CAGATMAPPOActor(self.b),self.b,device='cpu')
        ha, hb = ra.actor.initial_hidden(1), rb.actor.initial_hidden(1)
        for _ in range(30):
            oa = ra.sample_step(step.observations,ha,episode_start=False).action_output
            ob = rb.sample_step(step.observations,hb,episode_start=False).action_output
            self.assertEqual(oa.proposals,ob.proposals)
            for branch in ACTION_BRANCH_ORDER:
                self.assertTrue(torch.equal(oa.action_masks[branch],ob.action_masks[branch]))
                self.assertTrue(torch.equal(oa.probabilities[branch],ob.probabilities[branch]))
            self.assertTrue(torch.equal(oa.hidden_out,ob.hidden_out))
            self.assertTrue(torch.equal(ra.policy_generator.get_state(),rb.policy_generator.get_state()))
            ha,hb=oa.hidden_out.detach(),ob.hidden_out.detach()

    def test_stored_logprob_actual_sampler_and_replay_ratio_are_consistent(self):
        env=U2UMECEnvironment(self.b);step=route_active_step(env)
        runtime=CAGATMAPPOFactualActorRuntime(CAGATMAPPOActor(self.b),self.b,device='cpu')
        hidden=runtime.actor.initial_hidden(1)
        factual=runtime.sample_step(step.observations,hidden,episode_start=False)
        out=factual.action_output
        replay=runtime.action_distribution.evaluate_actions(factual.actor_batch,factual.action_masks,
                                                            out.proposals,hidden)
        chosen=out.probabilities['route'].gather(-1,out.action_indices['route'].unsqueeze(-1)).squeeze(-1)
        active=out.active_branches['route']
        torch.testing.assert_close(chosen[active].log(),out.branch_log_probs['route'][active],atol=2e-6,rtol=1e-6)
        self.assertTrue(torch.equal(out.branch_log_probs['route'],replay.branch_log_probs['route']))
        ratio=(replay.branch_log_probs['route']-out.branch_log_probs['route'].detach()).exp()
        self.assertTrue(torch.equal(ratio,torch.ones_like(ratio)))
        records=collect_route_training_records(out,environment_step=1,episode_id=0,slot=1)
        for r in records:
            self.assertEqual(r['stored_behavior_log_prob'],float(out.branch_log_probs['route'][0,0,r['uav_id']]))

    def test_regularizer_gradient_reaches_route_only(self):
        env=U2UMECEnvironment(self.b);step=route_active_step(env)
        runtime=CAGATMAPPOFactualActorRuntime(CAGATMAPPOActor(self.b),self.b,device='cpu')
        factual=runtime.sample_step(step.observations,runtime.actor.initial_hidden(1),episode_start=False)
        replay=runtime.action_distribution.evaluate_actions(factual.actor_batch,factual.action_masks,
                                                            factual.action_output.proposals)
        loss=route_exploration_loss(replay,self.b.training.mappo,0,torch.ones(1,1,dtype=torch.bool))
        loss.scaled_loss.backward()
        seen=False
        for name,p in runtime.actor.named_parameters():
            if name.startswith(('route_','action_heads.route.')):
                seen |= p.grad is not None and bool(p.grad.abs().sum()>0)
            else:
                self.assertTrue(p.grad is None or bool((p.grad==0).all()),name)
        self.assertTrue(seen)

    def test_training_telemetry_observes_without_rng_and_conserves_counts(self):
        env=U2UMECEnvironment(self.b);step=route_active_step(env)
        runtime=CAGATMAPPOFactualActorRuntime(CAGATMAPPOActor(self.b),self.b,device='cpu')
        out=runtime.sample_step(step.observations,runtime.actor.initial_hidden(1),episode_start=False).action_output
        before=runtime.policy_generator.get_state().clone();global_rng=torch.get_rng_state().clone()
        records=collect_route_training_records(out,environment_step=1,episode_id=0,slot=1)
        s=summarize_route_records(records)
        n=s['all_route_active']['route_active_count']
        self.assertEqual(sum(s['all_route_active']['sampled_count'].values()),n)
        self.assertEqual(s['remote_legal']['route_active_count']+s['remote_unavailable']['route_active_count'],n)
        self.assertAlmostEqual(sum(s['all_route_active']['mean_actor_probability'].values()),1.,places=6)
        self.assertTrue(torch.equal(before,runtime.policy_generator.get_state()))
        self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
        json.dumps(s,allow_nan=False)

    def test_illegal_local_remote_actions_never_sampled_or_reintroduced(self):
        env=U2UMECEnvironment(self.b);step=route_active_step(env)
        runtime=CAGATMAPPOFactualActorRuntime(CAGATMAPPOActor(self.b),self.b,device='cpu')
        batch=runtime.actor_tensorizer.encode_step(step.observations)
        originals=SequentialActionMaskBatch.from_observations(step.observations)
        for only in ('local','defer'):
            contracts=[]
            for c in originals.flattened():
                mask=np.zeros_like(c.route_mask);mask[c.route_domain.index(only)]=True
                contracts.append(replace(c,route_mask=mask,route_branch_active=True))
            masks=SequentialActionMaskBatch(((tuple(contracts),),))
            for _ in range(20):
                out=runtime.action_distribution.sample_actions(batch,masks)
                self.assertTrue(all(p.route==only for p in out.proposals[0][0]))
                r=route_exploration_loss(out,self.b.training.mappo,0,torch.ones(1,1,dtype=torch.bool))
                self.assertAlmostEqual(float(r.kl_mean),0.)

    def test_evaluation_is_exact_original_hierarchical_masked_argmax(self):
        env=U2UMECEnvironment(self.b);step=route_active_step(env)
        actor=CAGATMAPPOActor(self.b)
        runtime=CAGATMAPPOFactualActorRuntime(actor,self.b,device='cpu')
        batch=runtime.actor_tensorizer.encode_step(step.observations)
        masks=SequentialActionMaskBatch.from_observations(step.observations)
        for mode in ('train','eval'):
            getattr(actor,mode)()
            a=CAGATMAPPOActionDistribution(actor,self.a).deterministic_actions(batch,masks)
            b=CAGATMAPPOActionDistribution(actor,self.b).deterministic_actions(batch,masks)
            self.assertEqual(a.proposals,b.proposals)
            self.assertTrue(torch.equal(a.joint_log_prob,b.joint_log_prob))

    def test_new_and_historical_checkpoint_configuration_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            for label,config in [('e13',load_run_config(E13,cli_overrides={'training.mappo.training_device':'cpu'})),
                                 ('e14a',self.a),('e14b',self.b)]:
                payload,*_=make_payload(config,'FINAL_COMPLETED')
                path=Path(directory)/f'{label}.pt';atomic_save_checkpoint(payload,path)
                restored=load_checkpoint_payload(path)
                snapshot=restored['config_snapshot']['training']['mappo']
                self.assertEqual(snapshot.get('route_exploration_mode','off'),config.training.mappo.route_exploration_mode)
                evaluation=replace(config,mode='evaluation',launch_profile=None)
                for name in ('git_commit','git_branch','git_dirty'):
                    object.__setattr__(evaluation,name,getattr(config,name))
                loaded=load_actor_for_development_diagnostic(evaluation,
                                                            path,evaluation_device='cpu')
                self.assertEqual(loaded.actor.spec,CAGATMAPPOActor(config).spec)
                self.assertTrue(all(not p.requires_grad for p in loaded.actor.parameters()))

    def test_partial_rollout_telemetry_survives_checkpoint_serialization(self):
        config=replace(self.b,environment=replace(
            self.b.environment,arrival_probabilities=(1.0,)*self.b.environment.uav_count))
        buffer,records=factual_telemetry_buffer(config)
        state=serialize_active_rollout(buffer,0,'PERIODIC_RESUME')
        restored,version=restore_active_rollout(config,state,checkpoint_kind='PERIODIC_RESUME',policy_version=0)
        self.assertEqual(version,0)
        self.assertEqual(restored.transition_at(0).rejection_or_downgrade_summary[METADATA_KEY],records)

    def test_no_replay_double_count_and_final_tail_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            config=replace(self.b,output=replace(self.b.output,logs_dir=directory))
            buffer,records=factual_telemetry_buffer(config)
            row=write_route_rollout_telemetry(config,buffer,environment_steps=1,optimized=False)
            self.assertEqual(row['transition_count'],1)
            self.assertEqual(row['groups']['all_route_active']['route_active_count'],len(records))
            self.assertGreater(len(records),0)
            self.assertFalse(row['optimized'])
            self.assertEqual(len(list(Path(directory).rglob('route_training_rollouts_v1.jsonl'))),1)

    def test_production_updater_applies_four_fixed_schedule_epochs_and_releases_exactly(self):
        # Existing bounded paired-TX synthetic fixture; no shadow simulation or
        # real training run. Keeps the actual E14b optimizer/CPU/TX configuration.
        helper=paired_fixture.PairedTXCreditTests()
        config=replace(self.b,environment=replace(
            self.b.environment,arrival_probabilities=(1.0,)*self.b.environment.uav_count))
        helper.config=config
        actor=CAGATMAPPOActor(config);critic=MAPPOCentralizedCritic(config)
        buffer=helper.buffer(actor,critic,config=config)
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,config)
        with self.assertRaises(RecurrentPPOUpdateError):updater.update(buffer)
        output=updater.update(buffer,collected_environment_steps=256)
        self.assertEqual(len(output.epoch_diagnostics),4)
        rows=updater.last_route_exploration_summary
        self.assertEqual(len(rows),4)
        self.assertEqual({r['coefficient'] for r in rows},{.05*(1-256/6000)})
        self.assertTrue(all(r['loss_contribution']>0 for r in rows))
        self.assertIsNotNone(updater.last_integrated_summary['paired_samples'])
        # At and after release the production updates are bitwise the E14a loss.
        off=changed(config,route_exploration_mode='off')
        actors=[CAGATMAPPOActor(c) for c in (off,config)]
        critics=[MAPPOCentralizedCritic(c) for c in (off,config)]
        for c,a,v in zip((off,config),actors,critics):
            h=paired_fixture.PairedTXCreditTests();h.config=c
            u=CAGATMAPPORecurrentPPOUpdater(a,v,c)
            u.update(h.buffer(a,v,config=c),collected_environment_steps=6000)
        for k,v in actors[0].state_dict().items():self.assertTrue(torch.equal(v,actors[1].state_dict()[k]),k)
        for k,v in critics[0].state_dict().items():self.assertTrue(torch.equal(v,critics[1].state_dict()[k]),k)

    def test_trainer_flushes_factual_rollout_once_and_passes_absolute_counter(self):
        from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
        # Exercise wiring with a non-learning updater and synthetic transitions;
        # never call train(), launch a run, or create development artifacts.
        with tempfile.TemporaryDirectory() as directory:
            config=replace(self.b,output=replace(self.b.output,logs_dir=directory))
            source,records=factual_telemetry_buffer(config)
            seed=source.transition_at(0)
            buffer=CAGATMAPPORolloutBuffer(config)
            for index in range(256):
                metadata=copy.deepcopy(seed.rejection_or_downgrade_summary)
                metadata[METADATA_KEY]=[
                    {**record,'environment_step':index,'slot':index} for record in records]
                metadata['tx_paired_shadow']['rollout_offset']=index
                if index:
                    metadata['tx_paired_shadow']['samples']=[]
                    metadata['tx_paired_shadow']['paired_mask']=[False]*4
                buffer.append(replace(seed,slot=index,episode_start=index==0,
                                      rejection_or_downgrade_summary=metadata))
            counters=[]
            def update(actual_buffer,*,collected_environment_steps):
                self.assertTrue(actual_buffer.finalized)
                counters.append(collected_environment_steps)
                return fixed_update_output()
            updater=SimpleNamespace(update=update,last_route_exploration_summary=[
                {'epoch_index':i,'coefficient':route_exploration_coefficient(config.training.mappo,256)}
                for i in range(4)])
            trainer=CAGATMAPPOTrainer(config,updater_factory=lambda *_:updater)
            trainer.rollout_buffer=buffer
            trainer._rollout_policy_version=0
            trainer._transitions=256
            rng=trainer.policy_generator.get_state().clone()
            trainer._perform_update(trainer._updates)
            self.assertEqual(counters,[256])
            self.assertEqual(len(trainer.rollout_buffer),0)
            self.assertTrue(torch.equal(rng,trainer.policy_generator.get_state()))
            path=next(Path(directory).rglob('route_training_rollouts_v1.jsonl'))
            lines=path.read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(lines),1)
            row=json.loads(lines[0])
            self.assertEqual(row['transition_count'],256)
            self.assertEqual(len(row['ppo_epochs']),4)
            self.assertEqual(len(row['samples']),256*len(records))
            self.assertEqual(sum(row['groups']['all_route_active']['sampled_count'].values()),
                             256*len(records))

    def test_off_mode_full_episode_telemetry_keeps_actions_rewards_outcomes_and_rng(self):
        configs=(self.a,changed(self.a,route_training_telemetry_enabled=True))
        envs=[U2UMECEnvironment(c) for c in configs]
        results=[e.reset() for e in envs]
        runtimes=[CAGATMAPPOFactualActorRuntime(CAGATMAPPOActor(c),c,device='cpu') for c in configs]
        hidden=[r.actor.initial_hidden(1) for r in runtimes]
        for slot in range(500):
            outputs=[r.sample_step(s.observations,h,episode_start=slot==0).action_output
                     for r,s,h in zip(runtimes,results,hidden)]
            collect_route_training_records(outputs[1],environment_step=slot,episode_id=0,slot=slot)
            self.assertEqual(outputs[0].proposals,outputs[1].proposals)
            self.assertTrue(torch.equal(outputs[0].hidden_out,outputs[1].hidden_out))
            results=[e.step(o.proposals[0][0]) for e,o in zip(envs,outputs)]
            self.assertEqual(results[0].reward,results[1].reward)
            self.assertEqual(results[0].terminated,results[1].terminated)
            self.assertEqual(results[0].truncated,results[1].truncated)
            hidden=[o.hidden_out.detach() for o in outputs]
        self.assertEqual([t.snapshot() for t in envs[0].lifecycle.tasks.values()],
                         [t.snapshot() for t in envs[1].lifecycle.tasks.values()])
        self.assertTrue(torch.equal(runtimes[0].policy_generator.get_state(),runtimes[1].policy_generator.get_state()))


if __name__ == '__main__':
    unittest.main()
