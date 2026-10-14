"""E13 bounded unit/synthetic integration checks; never the 4K smoke run."""
import copy
import json
import math
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.config import ConfigError, load_run_config
from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution
from src.models.ca_gat_mappo_checkpoint import (CheckpointError, serialize_active_rollout, restore_active_rollout,
    atomic_save_checkpoint, load_checkpoint_payload, restore_model_optimizer_and_rng)
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutBuffer
from src.models.ca_gat_mappo_update import CAGATMAPPORecurrentPPOUpdater, build_recurrent_ppo_minibatch
from src.models.ca_gat_mappo_ppo import PPOObjectiveError
from src.models.tx_paired_shadow import (KEY, SCHEMA, paired_credit, credit_from_buffer,
    selection_state, selector_sources, FrozenShadowSampler, run_pair)
from tests.test_integrated_scheduler import ROOT, CONFIGS, fixture
from tests.test_tx_upper_stability import preserve_git


class PairedTXCreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config = load_run_config(CONFIGS / 'e13-v1-tx-paired-shadow-seed43-4k.json')
        cls.output = Path(tempfile.gettempdir()) / 'e13-v1-validation'
        cls.output.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def credit(self, d, p, y, mask=None, device='cpu'):
        d = torch.tensor([d], device=device, dtype=torch.float32)
        p = torch.tensor([p], device=device, dtype=torch.float32, requires_grad=True)
        y = torch.tensor([y], device=device, dtype=torch.bool)
        active = torch.ones_like(y)
        mask = active if mask is None else torch.tensor([mask], device=device, dtype=torch.bool)
        return paired_credit(d, torch.zeros_like(d), p, y, mask, active,
                             torch.ones(d.shape[:-1], device=device, dtype=torch.bool))

    def helper(self, config=None):
        from tests.test_tx_upper_counterfactual import TXCounterfactualTests
        h = TXCounterfactualTests()
        h.config = self.config if config is None else config
        _, h.batch, h.masks = fixture(h.config)
        return h

    def buffer(self, actor, critic, length=256, config=None, empty=False, version=0, slot_offset=0):
        config = self.config if config is None else config
        h = self.helper(config)
        original = h.buffer(actor, critic, length)
        buffer = CAGATMAPPORolloutBuffer(config)
        for index in range(length):
            transition = original.transition_at(index)
            transition = replace(transition, slot=slot_offset+index, episode_start=slot_offset+index==0)
            metadata = copy.deepcopy(transition.rejection_or_downgrade_summary)
            record = {'schema': SCHEMA, 'policy_version': version, 'rollout_offset': index,
                      'paired_mask': [False] * 4, 'samples': []}
            if index == 0 and not empty:
                names = metadata['policy_components_v1']['names']
                for source, d in zip(selector_sources(version,4,2),(2.,-3.)):
                    y = transition.proposal_actions[source].tx_select != 'idle'
                    selected_p = math.exp(metadata['policy_components_v1']['log_probs'][source][names.index('tx_upper')])
                    p = selected_p if y else 1 - selected_p
                    record['samples'].append({'source': source, 'slot': transition.slot, 'y': y, 'old_p': p,
                        'G_T': d, 'G_I': 0., 'd': d, 'A_TX_raw': (float(y)-p)*d,
                        'paired_mask': True, 'lifecycle_join_status': 'PASS',
                        'shadow_steps': 2*(config.environment.episode_horizon-transition.slot)})
                    record['paired_mask'][source] = True
            metadata[KEY] = record
            buffer.append(replace(transition, rejection_or_downgrade_summary=metadata))
        if length == 256:
            buffer.finalize()
        return buffer

    def test_hand_formula_action_sign_and_binary_score(self):
        c = self.credit([2.,2.,-2.,-2.], [.25]*4, [True,False,True,False])
        torch.testing.assert_close(c.raw_advantage, torch.tensor([[1.5,-.5,-1.5,.5]]))
        expected = c.d * (c.transmitted.float()-c.old_p).square()
        torch.testing.assert_close(c.raw_advantage*(c.transmitted.float()-c.old_p), expected)
        self.assertTrue(torch.equal(torch.sign(c.raw_advantage), torch.sign(c.normalized_advantage)))
        self.assertFalse(c.normalized_advantage.requires_grad)

    def test_mask_absence_is_nan_never_zero_credit(self):
        c = self.credit([2.,999.],[.2,.8],[True,False],[True,False])
        self.assertTrue(torch.isnan(c.raw_advantage[0,1]))
        self.assertTrue(torch.isnan(c.normalized_advantage[0,1]))
        self.assertEqual(c.telemetry()['paired_sample_count'], 1)

    def test_zero_singleton_constant_d_and_empty_cohorts(self):
        for d, p, y, mask in (([0.],[.5],[True],None), ([2.],[.5],[True],None),
                              ([3.,3.],[.5,.5],[True,True],None),([4.],[.5],[True],[False])):
            c = self.credit(d,p,y,mask)
            self.assertTrue(torch.isfinite(c.normalized_advantage[c.active]).all())
            if c.active.any() and d[0] != 0:
                torch.testing.assert_close(c.normalized_advantage[c.active], torch.ones(int(c.active.sum())))
            json.dumps(c.telemetry(),allow_nan=False)

    def test_invalid_active_mask_and_probability_fail_closed(self):
        for p in ([float('nan')], [-.1], [1.1]):
            with self.assertRaises(ValueError):self.credit([1.],p,[True])
        with self.assertRaises(ValueError):
            paired_credit(torch.ones(1,1),torch.zeros(1,1),torch.ones(1,1)*.5,
                          torch.ones(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool),
                          torch.zeros(1,1,dtype=torch.bool),torch.ones(1,dtype=torch.bool))

    def test_cyclic_selector_and_config_budgets_are_explicit(self):
        self.assertEqual(selector_sources(0,4,2),(0,1))
        self.assertEqual(selector_sources(3,4,2),(3,0))
        m = self.config.training.mappo
        self.assertEqual((m.tx_shadow_max_pairs_per_rollout,m.tx_shadow_step_budget_per_rollout),(2,2000))
        self.assertEqual((m.rollout_length_slots,m.recurrent_chunk_length_slots,m.sequence_minibatch_size,m.update_epochs),(256,32,8,4))
        self.assertEqual((m.gamma,m.gae_lambda,m.max_training_environment_steps),(.99,.95,4000))

    def test_config_rejects_invalid_protocol_popart_and_unbounded_subset(self):
        for changes in ({'tx_shadow_step_budget_per_rollout':0},{'tx_shadow_max_pairs_per_rollout':5},
                        {'tx_shadow_max_conditional_draws':0},{'tx_shadow_selector':'oracle'},
                        {'tx_q_target_normalization':'popart_v1'}, {'tx_upper_optimization_mode':'deterministic_stability_v1'}):
            c = replace(self.config,training=replace(self.config.training,mappo=replace(self.config.training.mappo,**changes)))
            with self.assertRaises(ConfigError):c.validate()

    def test_no_new_parameters_and_legacy_actor_initialization_identical(self):
        old=load_run_config(CONFIGS/'e12-v1-tx-factual-team-gae-seed43-32k.json')
        for cls in (CAGATMAPPOActor,MAPPOCentralizedCritic):
            a,b=cls(self.config),cls(old)
            self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
            for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
        self.assertFalse(hasattr(MAPPOCentralizedCritic(self.config),'tx_upper_q'))

    def loss(self, policy, advantages, mask):
        from src.models.ca_gat_mappo_ppo import compute_configured_ppo_objective_and_loss
        logs=torch.stack(list(policy.component_log_probs.values()),-1).reshape(1,4,-1)
        return compute_configured_ppo_objective_and_loss(new_joint_log_prob=policy.joint_log_prob.reshape(1,4),
            old_joint_log_prob=policy.joint_log_prob.detach().reshape(1,4),advantage=torch.tensor([[-2.,-1.,1.,2.]],device=logs.device),
            current_value=torch.zeros(1,4,device=logs.device),return_target=torch.ones(1,4,device=logs.device),
            entropy=policy.entropy.reshape(1,4),sequence_valid_mask=torch.ones(1,dtype=torch.bool,device=logs.device),config=self.config,
            new_branch_log_probs=logs,old_branch_log_probs=logs.detach(),
            active_branch_indicators=torch.stack(list(policy.component_active.values()),-1).reshape(1,4,-1),
            route_entropy=policy.branch_entropies['route'].reshape(1,4),tx_upper_advantage=advantages,tx_upper_paired_mask=mask)

    def gradient_check(self,device):
        _,batch,masks=fixture(self.config)
        actor=CAGATMAPPOActor(self.config).to(device)
        policy=CAGATMAPPOActionDistribution(actor,self.config).sample_actions(batch.to(device),masks)
        mask=torch.tensor([[True,False,False,False]],device=device)
        adv=torch.full((1,4),float('nan'),device=device);adv[0,0]=1.
        loss=self.loss(policy,adv,mask)
        self.assertEqual(int(loss.active_branch_indicators[...,1].sum()),1)
        self.assertTrue((loss.branch_surrogate[0,1:,1]==0).all())
        upper=-loss.branch_surrogate[...,1].sum();upper.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in actor.tx_upper_private.parameters() if p.grad is not None),0)

    def test_cpu_backward_only_paired_factual_sample(self):self.gradient_check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_backward_only_paired_factual_sample(self):self.gradient_check('cuda')

    def test_missing_mask_and_credit_do_not_fallback(self):
        _,batch,masks=fixture(self.config);actor=CAGATMAPPOActor(self.config)
        p=CAGATMAPPOActionDistribution(actor,self.config).sample_actions(batch,masks)
        with self.assertRaises(PPOObjectiveError):self.loss(p,torch.ones(1,4),None)
        with self.assertRaises(PPOObjectiveError):self.loss(p,None,torch.ones(1,4,dtype=torch.bool))

    def recurrent_check(self,device,empty=False):
        actor,critic=CAGATMAPPOActor(self.config),MAPPOCentralizedCritic(self.config)
        buffer=self.buffer(actor,critic,empty=empty)
        batch=build_recurrent_ppo_minibatch(buffer,self.config)
        self.assertEqual(tuple(batch.rollout_indices.shape),(8,32))
        actor.to(device);critic.to(device)
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,self.config)
        from src.models.tx_upper_counterfactual import TXUpperCounterfactualCriticV1
        with patch('src.models.tx_paired_shadow.credit_from_buffer',wraps=credit_from_buffer) as factory, \
             patch('src.models.tx_upper_factual_credit.factual_team_credit',side_effect=AssertionError('GAE fallback forbidden')), \
             patch.object(TXUpperCounterfactualCriticV1,'forward',side_effect=AssertionError('Q forbidden')):
            result=updater.update(buffer)
        self.assertEqual(factory.call_count,1)
        self.assertTrue(result.old_policy_snapshot_preserved)
        summaries=updater.last_integrated_summary
        self.assertEqual(len(summaries['paired_samples']),0 if empty else 2)
        first=summaries['ppo_epochs'][0]['tx_upper']
        for e in summaries['ppo_epochs']:
            tx=e['tx_upper']
            for key in ('d_mean','d_std','signed_score_mean','A_given_T_raw','A_given_I_raw','normalization_scale'):
                self.assertEqual(tx[key],first[key])
            self.assertGreaterEqual(tx['tx_grad_norm_before_clip'],0)
            self.assertGreaterEqual(tx['tx_update_norm'],0)
            self.assertEqual(tx['q_status'],'inactive')
        json.dumps(summaries,allow_nan=False)

    def test_recurrent_8x32_four_epoch_credit_frozen_no_q_no_gae(self):self.recurrent_check('cpu')

    def test_zero_paired_cohort_recurrent_update(self):self.recurrent_check('cpu',True)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_recurrent_four_epoch_update(self):self.recurrent_check('cuda')

    def test_checkpoint_partial_rollout_selector_roundtrip_and_tamper(self):
        actor,critic=CAGATMAPPOActor(self.config),MAPPOCentralizedCritic(self.config)
        buffer=self.buffer(actor,critic,8)
        state=serialize_active_rollout(buffer,0,'PERIODIC_RESUME')
        restored,_=restore_active_rollout(self.config,state,checkpoint_kind='PERIODIC_RESUME',policy_version=0)
        self.assertEqual(selection_state(buffer,self.config,0),selection_state(restored,self.config,0))
        from tests.test_ca_gat_mappo_checkpoint import assert_nested_equal
        assert_nested_equal(self,state,serialize_active_rollout(restored,0,'PERIODIC_RESUME'))
        bad=copy.deepcopy(state)
        bad['ordered_transitions'][0]['rejection_or_downgrade_summary'][KEY]['paired_mask'][0]=False
        with self.assertRaises(CheckpointError):restore_active_rollout(self.config,bad,checkpoint_kind='PERIODIC_RESUME',policy_version=0)

    def test_checkpoint_model_optimizer_resume_next_update_and_actor_loader(self):
        from tests.test_ca_gat_mappo_checkpoint import make_payload,training_state,assert_nested_equal
        from src.evaluation.actor_loader import load_final_actor_for_evaluation
        config=preserve_git(replace(self.config,environment=replace(self.config.environment,episode_horizon=512),
            training=replace(self.config.training,mappo=replace(self.config.training.mappo,training_device='cpu',
                checkpoint_interval_steps=4096,max_training_environment_steps=8192,tx_shadow_step_budget_per_rollout=2048))),self.config)
        actor,critic=CAGATMAPPOActor(config),MAPPOCentralizedCritic(config)
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,config)
        buffer=self.buffer(actor,critic,config=config);updater.update(buffer)
        payload,*_=make_payload(config,'FINAL_COMPLETED',actor=actor,critic=critic,updater=updater)
        payload['checkpoint_kind']='PERIODIC_RESUME';payload['training_state']=training_state(config,'PERIODIC_RESUME',0)
        with tempfile.TemporaryDirectory(dir=self.output) as directory:
            path=Path(directory)/'unit.pt';atomic_save_checkpoint(payload,path);saved=load_checkpoint_payload(path)
            a,c=CAGATMAPPOActor(config),MAPPOCentralizedCritic(config)
            u=CAGATMAPPORecurrentPPOUpdater(a,c,config)
            restore_model_optimizer_and_rng(config=config,payload=saved,actor=a,critic=c,optimizers=u.optimizers,dtype=torch.float32)
            updater.update(buffer);u.update(buffer)
            assert_nested_equal(self,actor.state_dict(),a.state_dict());assert_nested_equal(self,critic.state_dict(),c.state_dict())
            assert_nested_equal(self,updater.optimizers.actor_optimizer.state_dict(),u.optimizers.actor_optimizer.state_dict())
            final,*_=make_payload(config,'FINAL_COMPLETED',actor=actor,critic=critic,updater=updater)
            path=Path(directory)/'unit-final.pt'
            atomic_save_checkpoint(final,path)
            loaded=load_final_actor_for_evaluation(preserve_git(replace(config,mode='evaluation',launch_profile=None),config),path,evaluation_device='cpu')
            assert_nested_equal(self,actor.state_dict(),loaded.actor.state_dict())

    def short_environment(self):
        from src.env.environment import U2UMECEnvironment
        from src.evaluation.tx_paired_shadow_pilot import LifecycleLedger
        config=replace(self.config,environment=replace(self.config.environment,episode_horizon=8,
            arrival_probabilities=(1.,)*4,building_layout=(),candidate_neighbor_radius_m=2000.))
        env=U2UMECEnvironment(config);env.reset();prefix=LifecycleLedger()
        r=env.step(env.canonical_proposals());prefix.update(env,r)
        actions=list(env.canonical_proposals())
        for source,obs in enumerate(env.current_observations):
            dest=next(v for v,valid in zip(obs.action_masks.route_domain,obs.action_masks.route_mask) if valid and isinstance(v,int))
            actions[source]=replace(actions[source],route=dest)
        r=env.step(actions);prefix.update(env,r)
        return env,prefix,config

    def test_real_short_shadow_protocol_reward_join_state_rng_and_trainer_hook(self):
        from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
        from src.evaluation.tx_paired_shadow_pilot import digest
        env,prefix,config=self.short_environment()
        trainer=CAGATMAPPOTrainer(config,progress_logger=lambda _:None)
        trainer._tx_shadow_ledger=prefix
        h=trainer.actor.initial_hidden(1,device=trainer.device)
        step=trainer.factual_actor_runtime.sample_step(env.current_observations,h,episode_start=False)
        actions=tuple(step.action_output.proposals[0][0])
        before=env.state_fingerprint();rng=trainer.policy_generator.get_state().clone()
        record=trainer._collect_paired_tx(env,h,step.action_output,actions)
        self.assertEqual(before,env.state_fingerprint())
        self.assertTrue(torch.equal(rng,trainer.policy_generator.get_state()))
        self.assertEqual(len(record['samples']),2)
        actual=env.step(actions)
        for row in record['samples']:
            self.assertEqual(row['lifecycle_join_status'],'PASS')
            self.assertEqual(row['factual_first_step_digest'],digest(actual.info))
            self.assertEqual(row['shadow_steps'],12)
            self.assertAlmostEqual(row['A_TX_raw'],(float(row['y'])-row['old_p'])*row['d'])

    def test_frozen_shadow_sampler_owns_actor_and_rng_namespace(self):
        env,prefix,config=self.short_environment();actor=CAGATMAPPOActor(config)
        sampler=FrozenShadowSampler(actor,config,0)
        self.assertIsNot(actor,sampler.actor)
        self.assertTrue(all(not p.requires_grad for p in sampler.actor.parameters()))
        self.assertNotEqual(sampler.seed(env,'continuation'),sampler.seed(env,'conditional:0'))

    def test_selector_incremental_cache_rebuild_and_no_repeated_rollout_scan(self):
        from src.env.environment import U2UMECEnvironment
        from src.models.ca_gat_mappo_trainer import CAGATMAPPOTrainer
        _,_,config=self.short_environment()
        env=U2UMECEnvironment(config);env.reset()
        trainer=CAGATMAPPOTrainer(config,progress_logger=lambda _:None)
        template=self.buffer(CAGATMAPPOActor(config),MAPPOCentralizedCritic(config),1,
                             config=config,empty=True).transition_at(0)
        h=trainer.actor.initial_hidden(1,device=trainer.device)
        rng=trainer.policy_generator.get_state().clone()
        with patch('src.models.tx_paired_shadow.selection_state',wraps=selection_state) as recover:
            for i in range(4):
                record=trainer._collect_paired_tx(env,h,None,())
                meta=copy.deepcopy(template.rejection_or_downgrade_summary);meta[KEY]=record
                trainer.rollout_buffer.append(replace(template,slot=i,episode_start=i==0,rejection_or_downgrade_summary=meta))
            self.assertEqual(recover.call_count,1)
            cached=trainer._collect_paired_tx(env,h,None,())
            self.assertEqual(recover.call_count,1)
            # No append: the prospective cache length must be rejected.
            interrupted=trainer._collect_paired_tx(env,h,None,())
            self.assertEqual(recover.call_count,2)
            self.assertEqual(cached,interrupted)
            trainer._tx_shadow_selector_cache=None
            cold=trainer._collect_paired_tx(env,h,None,())
            self.assertEqual(recover.call_count,3)
        self.assertEqual(cached,cold)
        self.assertTrue(torch.equal(rng,trainer.policy_generator.get_state()))

    def test_telemetry_does_not_mutate_credit_or_rng(self):
        c=self.credit([2.,-3.],[.2,.7],[True,False])
        before=torch.get_rng_state().clone();raw=c.raw_advantage.clone()
        for _ in range(3):json.dumps(c.telemetry(),allow_nan=False)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(torch.equal(raw,c.raw_advantage))

    def test_other_components_entropy_and_value_unchanged_with_paired_mask(self):
        from src.models.ca_gat_mappo_ppo import compute_configured_ppo_objective_and_loss
        _,batch,masks=fixture(self.config);actor=CAGATMAPPOActor(self.config)
        p=CAGATMAPPOActionDistribution(actor,self.config).sample_actions(batch,masks)
        mask=torch.tensor([[True,False,False,False]])
        new=self.loss(p,torch.tensor([[1.,float('nan'),float('nan'),float('nan')]]),mask)
        h=self.helper();oldconfig=load_run_config(CONFIGS/'e12-v1-tx-factual-team-gae-seed43-32k.json')
        old=h.loss(p,torch.ones(1,4),oldconfig)
        for index in (0,2,3,4,5,6):
            torch.testing.assert_close(new.branch_surrogate[...,index],old.branch_surrogate[...,index],rtol=0,atol=0)
        torch.testing.assert_close(new.global_entropy_loss_contribution,old.global_entropy_loss_contribution,rtol=0,atol=0)
        torch.testing.assert_close(new.critic_loss,old.critic_loss,rtol=0,atol=0)

    def test_selector_budget_resume_and_rng_are_identical(self):
        from src.models.tx_paired_shadow import select_sources
        actor,critic=CAGATMAPPOActor(self.config),MAPPOCentralizedCritic(self.config)
        buffer=self.buffer(actor,critic,8)
        state=serialize_active_rollout(buffer,0,'PERIODIC_RESUME')
        restored,_=restore_active_rollout(self.config,state,checkpoint_kind='PERIODIC_RESUME',policy_version=0)
        env,_,_=self.short_environment()
        before=torch.get_rng_state().clone()
        self.assertEqual(select_sources(buffer,self.config,0,env),())
        self.assertEqual(select_sources(restored,self.config,0,env),())
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_missing_pair_metadata_and_corrupt_old_probability_fail_closed(self):
        actor,critic=CAGATMAPPOActor(self.config),MAPPOCentralizedCritic(self.config)
        old=self.helper().buffer(actor,critic,8)
        with self.assertRaises(ValueError):selection_state(old,self.config,0)
        buffer=self.buffer(actor,critic)
        bad=CAGATMAPPORolloutBuffer(self.config)
        for i in range(256):
            tr=buffer.transition_at(i);meta=copy.deepcopy(tr.rejection_or_downgrade_summary)
            if i==0:
                row=meta[KEY]['samples'][0];row['old_p']=.123456
                row['A_TX_raw']=(float(row['y'])-row['old_p'])*row['d']
            bad.append(replace(tr,rejection_or_downgrade_summary=meta))
        bad.finalize()
        active=torch.ones(8,32,4,dtype=torch.bool);valid=torch.ones(8,32,dtype=torch.bool)
        with self.assertRaises(ValueError):credit_from_buffer(bad,self.config,'cpu',active,valid)

    def test_small_double_return_contrast_not_erased_before_ppo(self):
        actor,critic=CAGATMAPPOActor(self.config),MAPPOCentralizedCritic(self.config)
        original=self.buffer(actor,critic)
        buffer=CAGATMAPPORolloutBuffer(self.config)
        for i in range(256):
            tr=original.transition_at(i);meta=copy.deepcopy(tr.rejection_or_downgrade_summary)
            if i==0:
                row=meta[KEY]['samples'][0]
                row.update(G_T=1e8+1.,G_I=1e8,d=1.,A_TX_raw=float(row['y'])-row['old_p'])
            buffer.append(replace(tr,rejection_or_downgrade_summary=meta))
        buffer.finalize()
        c,_=credit_from_buffer(buffer,self.config,'cpu',torch.ones(8,32,4,dtype=torch.bool),torch.ones(8,32,dtype=torch.bool))
        self.assertEqual(float(c.d[0,0,0]),1.)

    def test_full_periodic_partial_rollout_checkpoint_resume_matches_next_update(self):
        from tests.test_ca_gat_mappo_checkpoint import make_payload,training_state,assert_nested_equal
        config=preserve_git(replace(self.config,training=replace(self.config.training,
                mappo=replace(self.config.training.mappo,training_device='cpu'))),self.config)
        actor,critic=CAGATMAPPOActor(config),MAPPOCentralizedCritic(config)
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,config)
        partial=self.buffer(actor,critic,208,config=config,version=7,slot_offset=292)
        last=partial.transition_at(207)
        closed=CAGATMAPPORolloutBuffer(config)
        for i in range(207):closed.append(partial.transition_at(i))
        closed.append(replace(last,truncated=True,episode_boundary=True,bootstrap_allowed=False,bootstrap_value=None))
        payload,*_=make_payload(config,'FINAL_COMPLETED',actor=actor,critic=critic,updater=updater)
        payload['checkpoint_kind']='PERIODIC_RESUME'
        payload['training_state']=training_state(config,'PERIODIC_RESUME',208)
        payload['active_rollout_state']=serialize_active_rollout(closed,7,'PERIODIC_RESUME')
        with tempfile.TemporaryDirectory(dir=self.output) as directory:
            path=Path(directory)/'periodic-partial.pt';atomic_save_checkpoint(payload,path);saved=load_checkpoint_payload(path)
            a,c=CAGATMAPPOActor(config),MAPPOCentralizedCritic(config);u=CAGATMAPPORecurrentPPOUpdater(a,c,config)
            restore_model_optimizer_and_rng(config=config,payload=saved,actor=a,critic=c,optimizers=u.optimizers,dtype=torch.float32)
            restored,version=restore_active_rollout(config,saved['active_rollout_state'],checkpoint_kind='PERIODIC_RESUME',policy_version=7)
            self.assertEqual(version,7)
            self.assertEqual(selection_state(closed,config,7),selection_state(restored,config,7))
            continuation=self.buffer(actor,critic,48,config=config,empty=True,version=7)
            for i in range(48):
                tr=continuation.transition_at(i);meta=copy.deepcopy(tr.rejection_or_downgrade_summary)
                meta[KEY]['rollout_offset']=208+i
                tr=replace(tr,rejection_or_downgrade_summary=meta)
                closed.append(tr);restored.append(tr)
            closed.finalize();restored.finalize()
            updater.update(closed);u.update(restored)
            assert_nested_equal(self,actor.state_dict(),a.state_dict())
            assert_nested_equal(self,critic.state_dict(),c.state_dict())
            assert_nested_equal(self,updater.optimizers.actor_optimizer.state_dict(),u.optimizers.actor_optimizer.state_dict())


if __name__=='__main__':unittest.main()
