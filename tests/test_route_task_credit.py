"""E14c bounded credit/optimizer fixtures; never launch a training/eval run."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from src.config import ConfigError, load_run_config
from src.env.environment import U2UMECEnvironment
from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic, CentralizedStateTensorizer
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution
from src.models.ca_gat_mappo_runtime import CAGATMAPPOFactualActorRuntime
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer
from src.models.ca_gat_mappo_update import CAGATMAPPORecurrentPPOUpdater
from src.models.ca_gat_mappo_checkpoint import (atomic_save_checkpoint, load_checkpoint_payload,
    serialize_active_rollout, restore_active_rollout, restore_model_optimizer_and_rng)
from src.evaluation.actor_loader import load_actor_for_development_diagnostic
from src.models.route_task_credit import (KEY, MODE, RouteTaskValue, construct_task_credit,
    capture_task_values, make_step_metadata, task_features, epoch_task_records,
    summarize_credit_records, write_task_credit_telemetry)
from src.models.route_exploration import collect_route_training_records, route_exploration_coefficient
from tests.test_route_exploration import CONFIGS, E13, E14A, E14B, changed, route_active_step
from tests.test_ca_gat_mappo_checkpoint import make_payload
from tests import test_tx_paired_shadow_credit as paired_fixture

E14C = CONFIGS / 'e14c-v1-route-task-credit-seed42-8k-dev.json'


def state(task=11, source=0, value=0., status='unbound', dim=18):
    return dict(task_id=task, source_uav=source, old_value=value, status=status,
                destination=None if status == 'unbound' else source, features=[0.] * dim)


def decision(action='local', task=11, source=0, slot=0, episode=0):
    return dict(category=action, task_id=task, source_uav=source, episode_id=episode,
                global_rollout_index=slot, actor_top_probability={'Local': .3, 'Remote': .4, 'Defer': .3})


def step(slot=0, episode=0, before=(), after=(), decisions=(), terminals=(), version=0):
    return dict(slot=slot, episode_id=episode, policy_version=version, before=list(before),
                after=list(after), decisions=list(decisions), terminals=list(terminals))


def terminal(kind='completed', task=11, source=0, slot=2, payoff=1.):
    return dict(task_id=task, source_uav=source, kind=kind, slot=slot, payoff=payoff)


def credit(rows, gamma=.9, coefficient=1., shared=None):
    active = torch.zeros(len(rows), 4, dtype=torch.bool)
    for i,m in enumerate(rows):
        for d in m['decisions']: active[i,d['source_uav']] = True
    shared = torch.zeros_like(active,dtype=torch.float32) if shared is None else shared
    return construct_task_credit(rows,shared,active,gamma,coefficient)


def synthetic_buffer(config, actor, critic):
    """Attach known per-task outcomes to the existing bounded PPO fixture."""
    helper=paired_fixture.PairedTXCreditTests();helper.config=config
    original=helper.buffer(actor,critic,config=config)
    buffer=CAGATMAPPORolloutBuffer(config)
    for i in range(256):
        tr=original.transition_at(i)
        metadata=copy.deepcopy(tr.rejection_or_downgrade_summary)
        before=[]; decisions=[]; terminals=[]
        for u,p in enumerate(tr.proposal_actions):
            if not tr.active_branch_indicators[u,0]: continue
            task=i*4+u
            category=p.route if isinstance(p.route,str) else 'remote'
            before.append(state(task,u))
            decisions.append(decision(category,task,u,i))
            terminals.append(terminal('expired' if category=='defer' else 'completed',task,u,i,
                                      -1. if category=='defer' else 1.))
        metadata[KEY]=step(i,before=before,decisions=decisions,terminals=terminals)
        buffer.append(replace(tr,rejection_or_downgrade_summary=metadata))
    buffer.finalize()
    return buffer


class RouteTaskCreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.b=load_run_config(E14B,cli_overrides={'training.mappo.training_device':'cpu'})
        cls.c=load_run_config(E14C,cli_overrides={'training.mappo.training_device':'cpu'})

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_local_and_remote_done_and_expired_discount_and_baseline(self):
        for action in ('local','remote'):
            for kind,payoff in (('completed',1.),('expired',-1.)):
                with self.subTest(action=action,kind=kind):
                    rows=[step(before=[state(value=.2)],after=[state(status='local')],
                               decisions=[decision(action)]),
                          step(1,before=[state(status='local')],after=[state(status='local')]),
                          step(2,before=[state(status='local')],terminals=[terminal(kind,payoff=payoff)])]
                    out=credit(rows,shared=torch.full((3,4),.3))
                    expected=.9**2*payoff-.2
                    self.assertAlmostEqual(float(out.task_advantage[0,0]),expected,places=6)
                    self.assertAlmostEqual(float(out.advantage[0,0]),.3+expected,places=6)
                    self.assertEqual(out.samples[0]['route_terminal_delay'],2)
                    self.assertEqual(out.samples[0]['terminal_status'],kind)
                    self.assertFalse(out.advantage.requires_grad)

    def test_unbound_expiry_reaches_all_prior_defer(self):
        rows=[step(i,before=[state()],after=[state()] if i<2 else [],
                   decisions=[decision('defer',slot=i)],
                   terminals=[terminal('expired',slot=2,payoff=-1.)] if i==2 else []) for i in range(3)]
        out=credit(rows)
        torch.testing.assert_close(out.task_advantage[:,0],torch.tensor([-.81,-.9,-1.]))
        self.assertEqual({r['terminal_status'] for r in out.samples},{'expired'})

    def test_successful_bind_does_not_copy_done_to_prior_defer(self):
        rows=[step(before=[state()],after=[state()],decisions=[decision('defer')]),
              step(1,before=[state()],after=[state(value=.4,status='local')],decisions=[decision(slot=1)]),
              step(2,before=[state(status='local')],terminals=[terminal()])]
        out=credit(rows)
        self.assertAlmostEqual(out.samples[0]['return'],.9**2*.4)
        self.assertEqual(out.samples[0]['credit_stop'],'binding_value_bootstrap')
        self.assertAlmostEqual(out.samples[1]['return'],.9)
        rows[2]['terminals'][0]['payoff']=10.
        other=credit(rows)
        self.assertEqual(out.samples[0]['return'],other.samples[0]['return'])
        self.assertNotEqual(out.samples[1]['return'],other.samples[1]['return'])

    def test_same_slot_bind_expiry_and_horizon_are_supported(self):
        for kind,payoff in (('expired',-1.),('truncated',0.)):
            rows=[step(before=[state()],after=[state()],decisions=[decision('defer')]),
                  step(1,before=[state()],decisions=[decision(slot=1)],
                       terminals=[terminal(kind,slot=1,payoff=payoff)])]
            out=credit(rows)
            self.assertAlmostEqual(out.samples[0]['return'],.9*payoff)
            self.assertEqual(out.samples[0]['credit_stop'],'binding_same_slot_terminal')
            self.assertEqual(out.samples[1]['return'],payoff)

    def test_truncation_censors_instead_of_fake_expiry(self):
        out=credit([step(before=[state()],decisions=[decision('defer')],
                         terminals=[terminal('truncated',slot=0,payoff=0.)])])
        self.assertEqual(out.samples[0]['return'],0.)
        self.assertTrue(out.samples[0]['censored'])

    def test_rollout_frontier_is_next_state_with_correct_exponent(self):
        out=credit([step(7,before=[state(value=.1)],after=[state(value=.7,status='local')],
                         decisions=[decision(slot=7)])])
        self.assertAlmostEqual(out.samples[0]['return'],.9*.7)
        self.assertEqual(out.samples[0]['credit_stop'],'rollout_bootstrap')
        self.assertIsNone(out.samples[0]['terminal_slot'])

    def test_no_task_episode_or_source_cross_contamination(self):
        rows=[step(before=[state()],decisions=[decision()],terminals=[terminal(slot=0)]),
              step(0,episode=1,before=[state()],decisions=[decision('defer',episode=1)],
                   terminals=[terminal('expired',slot=0,payoff=-1.)])]
        out=credit(rows)
        self.assertEqual(out.task_advantage[:,0].tolist(),[1.,-1.])
        rows=[step(before=[state(11,0),state(12,1)],decisions=[decision(task=11),decision(task=12,source=1)],
                   terminals=[terminal(task=11,slot=0),terminal('expired',task=12,source=1,slot=0,payoff=-1.)])]
        self.assertEqual(credit(rows).task_advantage[0,:2].tolist(),[1.,-1.])

    def test_mixed_policy_versions_and_stale_frontier_fail_closed(self):
        rows=[step(before=[state()],after=[state()],decisions=[decision('defer')]),step(1,version=1)]
        with self.assertRaisesRegex(ValueError,'policy-version'):credit(rows)
        rows[1]['policy_version']=0
        with self.assertRaisesRegex(ValueError,'stale'):credit(rows)

    def test_inactive_route_and_missing_metadata_fail_closed(self):
        m=[step(before=[state()],decisions=[decision()],terminals=[terminal(slot=0)])]
        with self.assertRaisesRegex(ValueError,'active Route'):construct_task_credit(
            m,torch.zeros(1,4),torch.zeros(1,4,dtype=torch.bool),.99,1.)
        with self.assertRaisesRegex(ValueError,'missing'):construct_task_credit(
            [step()],torch.zeros(1,4),torch.ones(1,4,dtype=torch.bool),.99,1.)

    def test_new_configuration_preserves_every_other_mechanism(self):
        b,c=self.b.resolved_dict(),self.c.resolved_dict()
        for k in ('route_credit_mode','route_task_credit_coefficient','route_task_value_coefficient'):
            c['training']['mappo'].pop(k)
        c['output']=b['output'];self.assertEqual(b,c)
        self.assertEqual(self.c.training.mappo.cpu_anti_starvation_mode,'deadline_feasibility_v1')
        self.assertEqual(self.c.training.mappo.tx_upper_credit_mode,'paired_shadow_v1')
        self.assertEqual(route_exploration_coefficient(self.c.training.mappo,6000),0.)

    def test_invalid_config_and_mode_off_identity(self):
        for kwargs in ({'route_task_credit_coefficient':-1.},{'route_task_value_coefficient':0.},
                       {'route_task_value_coefficient':float('nan')},{'route_task_credit_coefficient':True},
                       {'route_private_backbone':False},{'route_training_telemetry_enabled':False}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ConfigError):changed(self.c,**kwargs).validate()
        off=changed(self.b,route_task_credit_coefficient=17.,route_task_value_coefficient=12.)
        self.assertEqual(self.b.config_hash,off.config_hash)
        self.assertFalse(hasattr(MAPPOCentralizedCritic(off),'route_task_value'))
        self.assertEqual(load_run_config(E14B).config_hash,
                         '3688fe21edd901a5fae1bb6d71815a4b0b73bab2a1736305f66fdd647bcea0cc')

    def test_task_head_is_independent_and_preserves_actor_base_critic_and_rng(self):
        rng=torch.get_rng_state().clone()
        b,c=MAPPOCentralizedCritic(self.b),MAPPOCentralizedCritic(self.c)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in b.state_dict().items():self.assertTrue(torch.equal(v,c.state_dict()[k]),k)
        a1,a2=CAGATMAPPOActor(self.b),CAGATMAPPOActor(self.c)
        for k,v in a1.state_dict().items():self.assertTrue(torch.equal(v,a2.state_dict()[k]),k)
        x=torch.randn(3,c.spec.centralized_state_dim,requires_grad=True)
        c.route_task_value(x.detach(),torch.zeros(3,18)).square().sum().backward()
        self.assertIsNone(x.grad)
        self.assertTrue(all(p.grad is None for p in c.value_network.parameters()))
        if torch.cuda.is_available():
            cuda_rng=[s.clone() for s in torch.cuda.get_rng_state_all()]
            RouteTaskValue(c.spec.centralized_state_dim,128,4,self.c.seed)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())))

    def test_current_task_features_have_no_terminal_labels_or_ids(self):
        task=dict(task_id=11,status='unbound',source_uav=0,destination=None,data_bits=1e6,
                  remaining_bits=1e6,cpu_cycles=1e9,remaining_cycles=1e9,arrival_slot=0,deadline_slot=50)
        f=task_features(task,7,self.c)
        altered={**task,'task_id':999,'completion_slot':37,'terminal_reason':'future'}
        self.assertEqual(f,task_features(altered,7,self.c));self.assertEqual(len(f),18)

    def test_real_capture_and_terminal_attribution_are_observation_only(self):
        env=U2UMECEnvironment(self.c);current=route_active_step(env)
        actor=CAGATMAPPOActor(self.c);critic=MAPPOCentralizedCritic(self.c)
        runtime=CAGATMAPPOFactualActorRuntime(actor,self.c,device='cpu')
        tensorizer=CentralizedStateTensorizer(self.c)
        features=tensorizer.encode_step(current.centralized_state).features
        snapshot=copy.deepcopy(env.snapshot());rng=torch.get_rng_state().clone()
        before=capture_task_values(env,features,critic,self.c)
        self.assertEqual(env.snapshot(),snapshot);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        output=runtime.sample_step(current.observations,actor.initial_hidden(1),episode_start=False).action_output
        result=env.step(output.proposals[0][0])
        after=capture_task_values(env,tensorizer.encode_step(result.centralized_state).features,critic,self.c)
        md=make_step_metadata(before=before,after=after,info=result.info,policy=output,
                              episode_id=0,environment_step=1,policy_version=0)
        self.assertEqual(len(md['decisions']),int(output.active_branches['route'].sum()))
        json.dumps(md,allow_nan=False)

    def test_ppo_ratio_and_surrogate_are_the_actual_route_leaf_distribution(self):
        batch=credit([step(before=[state()],decisions=[decision()],terminals=[terminal(slot=0)])])
        policy=SimpleNamespace(branch_log_probs={'route':torch.tensor([[[math.log(.6),0.,0.,0.]]])},
            raw_logits={'route':torch.zeros(1,1,4,6)},action_masks={'route':torch.tensor(
                [[[[False,True,True,True,False,False]]*4]])})
        old=torch.tensor([[[math.log(.3),0.,0.,0.]]])
        row=epoch_task_records(batch,policy,old,0,.2)[0]
        self.assertAlmostEqual(row['ppo_ratio'],2.,places=6)
        self.assertAlmostEqual(row['unclipped_surrogate'],2.,places=6)
        self.assertAlmostEqual(row['clipped_surrogate'],1.2,places=6)
        self.assertAlmostEqual(sum(row['new_actor_top_probability'].values()),1.,places=6)

    def test_factual_sampling_logprob_replay_and_direct_route_gradient_isolation(self):
        env=U2UMECEnvironment(self.c);current=route_active_step(env)
        outputs=[];runtimes=[]
        for config in (self.b,self.c):
            actor=CAGATMAPPOActor(config)
            runtime=CAGATMAPPOFactualActorRuntime(actor,config,device='cpu')
            factual=runtime.sample_step(current.observations,actor.initial_hidden(1),episode_start=False)
            outputs.append(factual);runtimes.append(runtime)
        self.assertEqual(outputs[0].action_output.proposals,outputs[1].action_output.proposals)
        self.assertTrue(torch.equal(outputs[0].action_output.hidden_out,outputs[1].action_output.hidden_out))
        factual=outputs[1];runtime=runtimes[1]
        replay=runtime.action_distribution.evaluate_actions(
            factual.actor_batch,factual.action_masks,factual.action_output.proposals,
            runtime.actor.initial_hidden(1))
        torch.testing.assert_close(replay.joint_log_prob,factual.action_output.joint_log_prob)
        torch.testing.assert_close(replay.branch_log_probs['route'],factual.action_output.branch_log_probs['route'])
        (-replay.branch_log_probs['route'][replay.active_branches['route']].mean()).backward()
        nonzero=False
        for name,p in runtime.actor.named_parameters():
            if name.startswith(('route_','action_heads.route.')):
                nonzero |= p.grad is not None and bool(p.grad.abs().sum()>0)
            else:self.assertTrue(p.grad is None or bool((p.grad==0).all()),name)
        self.assertTrue(nonzero)

    def test_real_short_lifecycle_uses_existing_terminal_allocations_and_no_cross_episode_leak(self):
        c=replace(self.c,environment=replace(self.c.environment,episode_horizon=20,
                   minimum_task_slack_slots=1,maximum_task_slack_slots=5,arrival_probabilities=(1.,)*4))
        env=U2UMECEnvironment(c);current=env.reset()
        actor,critic=CAGATMAPPOActor(c),MAPPOCentralizedCritic(c)
        runtime=CAGATMAPPOFactualActorRuntime(actor,c,device='cpu')
        tensorizer=CentralizedStateTensorizer(c);hidden=actor.initial_hidden(1)
        rows=[];attributed=0
        for i in range(20):
            before=capture_task_values(env,tensorizer.encode_step(current.centralized_state).features,critic,c)
            factual=runtime.sample_step(current.observations,hidden,episode_start=i==0)
            hidden=factual.action_output.hidden_out.detach()
            result=env.step(factual.action_output.proposals[0][0])
            after=[] if result.truncated else capture_task_values(
                env,tensorizer.encode_step(result.centralized_state).features,critic,c)
            md=make_step_metadata(before=before,after=after,info=result.info,policy=factual.action_output,
                                 episode_id=0,environment_step=i,policy_version=0)
            allocations={a['task_id']:a for a in result.info['agent_credit']['terminal_attributions']}
            for t in md['terminals']:
                if t['kind']=='truncated':self.assertEqual(t['payoff'],0.);continue
                attributed+=1
                field,suffix=('completion_allocation','reward') if t['kind']=='completed' else ('expiration_allocation','penalty')
                expected=sum(allocations[t['task_id']][field][f'{role}_{suffix}'] for role in ('source','destination'))
                self.assertAlmostEqual(t['payoff'],expected if t['kind']=='completed' else -expected)
            rows.append(md);current=result
        self.assertGreater(attributed,0)
        out=credit(rows,gamma=c.training.mappo.gamma)
        self.assertGreater(len(out.samples),0)
        self.assertTrue(all(s['terminal_status'] is not None for s in out.samples))

    def test_production_optimizer_uses_credit_four_epochs_and_preserves_paired_tx(self):
        c=replace(self.c,environment=replace(self.c.environment,arrival_probabilities=(1.,)*4))
        actor,critic=CAGATMAPPOActor(c),MAPPOCentralizedCritic(c)
        buffer=synthetic_buffer(c,actor,critic)
        hidden=[buffer.transition_at(i).hidden_in.clone() for i in range(256)]
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,c)
        result=updater.update(buffer,collected_environment_steps=256)
        rows=updater.last_route_task_records
        self.assertEqual(len(result.epoch_diagnostics),4)
        self.assertEqual(len(rows),4*sum(int(buffer.transition_at(i).active_branch_indicators[:,0].sum())
                                       for i in range(256)))
        # The inherited synthetic fixture deliberately has artificial recurrent
        # hidden snapshots: verify the actual ratio, rather than claiming these
        # are freshly collected on-policy probabilities.
        self.assertTrue(all(math.isclose(r['ppo_ratio'],math.exp(r['new_log_prob']-r['old_log_prob']),
                                        rel_tol=2e-6) for r in rows))
        self.assertTrue(all(math.isclose(r['combined_advantage'],r['shared_advantage']+
                                        r['route_specific_advantage'],abs_tol=2e-5) for r in rows))
        self.assertEqual(len(updater.last_route_task_value_losses),4)
        self.assertGreater(sum(float(p.abs().sum()) for p in critic.route_task_value.parameters()),0.)
        self.assertGreater(float(critic.route_task_value.network[-1].weight.detach().abs().sum()),0.)
        self.assertTrue(all(torch.equal(h,buffer.transition_at(i).hidden_in) for i,h in enumerate(hidden)))
        self.assertIsNotNone(updater.last_integrated_summary['paired_samples'])
        self.assertEqual({r['coefficient'] for r in updater.last_route_exploration_summary},{.05*(1-256/6000)})
        payload,_,_,_=make_payload(c,'FINAL_COMPLETED',actor=actor,critic=critic,updater=updater)
        fresh_actor,fresh_critic=CAGATMAPPOActor(c),MAPPOCentralizedCritic(c)
        fresh=CAGATMAPPORecurrentPPOUpdater(fresh_actor,fresh_critic,c)
        fresh_critic.load_state_dict(payload['model_state']['critic'])
        fresh.optimizers.critic_optimizer.load_state_dict(payload['optimizer_state']['critic_adam'])
        self.assertEqual(len(fresh.optimizers.critic_optimizer.state),len(updater.optimizers.critic_optimizer.state))
        for p,q in zip(critic.route_task_value.parameters(),fresh_critic.route_task_value.parameters()):
            self.assertTrue(torch.equal(p,q))
            for key,value in updater.optimizers.critic_optimizer.state[p].items():
                other=fresh.optimizers.critic_optimizer.state[q][key]
                self.assertTrue(torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other)

    def test_zero_task_credit_is_bitwise_e14b_optimizer_even_with_aux_value_training(self):
        b=replace(self.b,environment=replace(self.b.environment,arrival_probabilities=(1.,)*4))
        c=changed(b,route_credit_mode=MODE,route_task_credit_coefficient=0.)
        actors=[];critics=[];updaters=[]
        for config in (b,c):
            a,v=CAGATMAPPOActor(config),MAPPOCentralizedCritic(config)
            u=CAGATMAPPORecurrentPPOUpdater(a,v,config)
            u.update(synthetic_buffer(config,a,v),collected_environment_steps=256)
            actors.append(a);critics.append(v);updaters.append(u)
        for k,v in actors[0].state_dict().items():self.assertTrue(torch.equal(v,actors[1].state_dict()[k]),k)
        for k,v in critics[0].state_dict().items():self.assertTrue(torch.equal(v,critics[1].state_dict()[k]),k)
        self.assertEqual(updaters[0].last_integrated_summary,updaters[1].last_integrated_summary)

    def test_new_and_historical_checkpoint_roundtrip_and_development_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            for file in (E13,E14A,E14B,E14C):
                config=load_run_config(file,cli_overrides={'training.mappo.training_device':'cpu'})
                payload,actor,critic,*_=make_payload(config,'FINAL_COMPLETED')
                path=Path(directory)/f'{file.stem}.pt';atomic_save_checkpoint(payload,path)
                restored=load_checkpoint_payload(path)
                model=MAPPOCentralizedCritic(config);model.load_state_dict(restored['model_state']['critic'])
                for k,v in critic.state_dict().items():self.assertTrue(torch.equal(v,model.state_dict()[k]),k)
                evaluation=replace(config,mode='evaluation',launch_profile=None)
                for name in ('git_commit','git_branch','git_dirty'):
                    object.__setattr__(evaluation,name,getattr(config,name))
                loaded=load_actor_for_development_diagnostic(evaluation,path,evaluation_device='cpu')
                self.assertFalse(hasattr(loaded.actor,'route_task_value'))

    def test_evaluation_masks_indices_hidden_and_actions_equal_e14b(self):
        env=U2UMECEnvironment(self.c);current=route_active_step(env)
        actor=CAGATMAPPOActor(self.c)
        runtime=CAGATMAPPOFactualActorRuntime(actor,self.c,device='cpu')
        batch=runtime.actor_tensorizer.encode_step(current.observations)
        from src.models.ca_gat_mappo_actions import SequentialActionMaskBatch
        masks=SequentialActionMaskBatch.from_observations(current.observations)
        outputs=[]
        for config in (self.b,self.c):
            out=CAGATMAPPOActionDistribution(actor,config).deterministic_actions(batch,masks)
            outputs.append(out)
        self.assertEqual(outputs[0].proposals,outputs[1].proposals)
        self.assertTrue(torch.equal(outputs[0].joint_log_prob,outputs[1].joint_log_prob))
        self.assertTrue(torch.equal(outputs[0].hidden_out,outputs[1].hidden_out))
        for branch in outputs[0].action_masks:
            self.assertTrue(torch.equal(outputs[0].action_masks[branch],outputs[1].action_masks[branch]))

    def test_telemetry_conservation_and_partial_checkpoint_metadata(self):
        rows=[step(before=[state()],decisions=[decision()],terminals=[terminal(slot=0)])]
        b=credit(rows)
        samples=[{**b.samples[0],'ppo_ratio':1.,'surrogate':1.,'epoch_index':i} for i in range(4)]
        g=summarize_credit_records(samples)['0-2000/local']
        self.assertEqual((g['count'],g['unique_factual_count']),(4,1))
        with tempfile.TemporaryDirectory() as directory:
            c=replace(self.c,output=replace(self.c.output,logs_dir=directory))
            record=write_task_credit_telemetry(c,samples,256,[.1]*4)
            self.assertEqual(record['groups']['0-2000/local'],g)
            self.assertEqual(len(list(Path(directory).rglob('route_task_credit_updates_v1.jsonl'))),1)
        a,v=CAGATMAPPOActor(self.c),MAPPOCentralizedCritic(self.c)
        buffer=synthetic_buffer(self.c,a,v)
        partial=CAGATMAPPORolloutBuffer(self.c);partial.append(buffer.transition_at(0))
        serialized=serialize_active_rollout(partial,0,'PERIODIC_RESUME')
        restored,_=restore_active_rollout(self.c,serialized,checkpoint_kind='PERIODIC_RESUME',policy_version=0)
        self.assertEqual(restored.transition_at(0).rejection_or_downgrade_summary[KEY],
                         partial.transition_at(0).rejection_or_downgrade_summary[KEY])


if __name__ == '__main__': unittest.main()
