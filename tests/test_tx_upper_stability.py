"""E9 bounded unit fixtures; never dispatch formal training/evaluation."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from src.config import ConfigError, load_run_config
from src.models.ca_gat_mappo import CAGATMAPPOActor, MAPPOCentralizedCritic, CentralizedStateTensorBatch
from src.models.ca_gat_mappo_actions import CAGATMAPPOActionDistribution
from src.models.ca_gat_mappo_ppo import compute_configured_ppo_objective_and_loss, PPOObjectiveError
from src.models.ca_gat_mappo_rollout import CAGATMAPPORolloutTransition
from src.models.ca_gat_mappo_update import (CAGATMAPPORecurrentPPOUpdater,
    build_recurrent_ppo_minibatch, compute_configured_batched_ppo_objective_and_loss,
    build_ca_gat_mappo_optimizers)
from src.models.ca_gat_mappo_checkpoint import (serialize_active_rollout, restore_active_rollout,
    atomic_save_checkpoint, load_checkpoint_payload, restore_model_optimizer_and_rng)
from src.models.integrated_scheduler import rollout_policy_components
from src.models.tx_upper_stability import TXUpperDeterministicStabilityPPOV1 as Stability
from tests.test_integrated_scheduler import fixture, vector_buffer, CONFIGS, ROOT
from tests.test_route_private_backbone import gradient_sum


def preserve_git(config, reference):
    for name in ('git_commit','git_branch','git_dirty'):
        object.__setattr__(config,name,getattr(reference,name))
    return config


class TXUpperStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.e8=load_run_config(CONFIGS/'e8-v1-integrated-tx-deadline-cpu-seed43-32k.json')
        cls.e9=load_run_config(CONFIGS/'e9-v1-tx-deterministic-stability-seed43-32k.json')
        cls.seed,cls.batch,cls.masks=fixture(cls.e9)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_exact_single_canonical_diff_and_defaults(self):
        from src.evaluation.e4_v1_matched_audit import differences
        a,b=self.e8.resolved_dict(),self.e9.resolved_dict()
        a['training']['mappo'].setdefault('tx_upper_optimization_mode','ppo_standard')
        self.assertEqual(differences(a,b),[{'field':'training.mappo.tx_upper_optimization_mode',
            'E3':'ppo_standard','E4':'deterministic_stability_v1'}])
        self.assertEqual(self.e8.config_hash,'b87fff30280e3b7d5284b5a1ec68b8ce06f28d02bf3ecd98a30842b24dc00084')
        # The user has since completed E9; its saved snapshot may now exist.
        snapshot=Path(self.e9.artifact_paths()['config_snapshot'])
        if snapshot.exists():
            self.assertEqual(load_run_config(snapshot).config_hash,self.e9.config_hash)
        for value in ['bad',True,[],None]:
            c=replace(self.e9,training=replace(self.e9.training,mappo=replace(self.e9.training.mappo,tx_upper_optimization_mode=value)))
            with self.assertRaises(ConfigError):c.validate()
        bad=replace(self.e9,training=replace(self.e9.training,mappo=replace(self.e9.training.mappo,tx_upper_decoder_mode='legacy')))
        with self.assertRaises(ConfigError):bad.validate()

    def test_normalization_uses_only_active_cohort_and_detaches(self):
        a=torch.tensor([-4.,0.,4.,1.e6],requires_grad=True);active=torch.tensor([1,1,1,0],dtype=torch.bool)
        result,mean,std=Stability.normalize_advantage(a,active)
        torch.testing.assert_close(result[:3],torch.tensor([-1.2247449,0,1.2247449]))
        self.assertEqual(result[-1],0);self.assertEqual(mean,0);self.assertFalse(result.requires_grad)
        changed=a.detach().clone();changed[-1]=-1.e9
        torch.testing.assert_close(result,Stability.normalize_advantage(changed,active)[0],rtol=0,atol=0)
        self.assertAlmostEqual(float(std),float(torch.tensor([-4.,0.,4.]).std(unbiased=False)),places=6)

    def test_constant_singleton_empty_cohorts_are_zero(self):
        for a,active in [(torch.ones(3),torch.ones(3,dtype=torch.bool)),
                         (torch.tensor([5.,9.]),torch.tensor([1,0],dtype=torch.bool)),
                         (torch.ones(2),torch.zeros(2,dtype=torch.bool))]:
            self.assertTrue((Stability.normalize_advantage(a,active)[0]==0).all())

    def test_nonpositive_advantage_has_zero_margin_and_gradient(self):
        gap=torch.tensor([.1,-.1],requires_grad=True)
        loss=Stability.binary_margin(gap,torch.tensor([1,0],dtype=torch.bool),torch.tensor([0.,-2.]),torch.ones(2,dtype=torch.bool))
        self.assertEqual(float(loss),0);loss.backward();self.assertTrue((gap.grad==0).all())

    def test_positive_transmit_and_idle_margin_monotonic(self):
        active=torch.ones(1,dtype=torch.bool)
        def loss(gap,transmit):return Stability.binary_margin(torch.tensor([gap]),torch.tensor([transmit]),torch.ones(1),active)
        self.assertLess(loss(1.,True),loss(-1.,True))
        self.assertLess(loss(-1.,False),loss(1.,False))

    def test_action_symmetry_and_no_transmit_bias(self):
        gap=torch.tensor([.2,-.6,1.],requires_grad=True);a=torch.tensor([1.,2.,-1.]);actions=torch.tensor([1,0,1],dtype=torch.bool);active=torch.ones(3,dtype=torch.bool)
        x=Stability.binary_margin(gap,actions,a,active);y=Stability.binary_margin(-gap,~actions,a,active)
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        zero=torch.zeros(2,requires_grad=True);loss=Stability.binary_margin(zero,torch.tensor([1,0],dtype=torch.bool),torch.ones(2),torch.ones(2,dtype=torch.bool));loss.backward()
        self.assertAlmostEqual(float(zero.grad[0]),-float(zero.grad[1]),places=7)

    def test_fixed_margin_weight_math_and_bounded_positive_weights(self):
        self.assertEqual(Stability.MARGIN_LOGIT,.5);self.assertEqual(Stability.MARGIN_WEIGHT,.05)
        self.assertAlmostEqual(float(torch.sigmoid(torch.tensor(.5))),.6224593312,places=7)
        gap=torch.zeros(1);mask=torch.ones(1,dtype=torch.bool)
        got=Stability.binary_margin(gap,mask,torch.tensor([100.]),mask)
        expected=5*torch.nn.functional.softplus(torch.tensor(.5));torch.testing.assert_close(got,expected)

    def test_entropy_schedule_all_boundaries(self):
        for steps,beta in [(0,.01),(7999,.01),(8000,.01),(8001,.009999375),
                           (16000,.005),(23999,.000000625),(24000,0.),(24001,0.),(32000,0.),(40000,0.)]:
            self.assertAlmostEqual(Stability.entropy_coefficient(.01,steps,32000),beta,places=12)
        for steps,budget in [(-1,32000),(True,32000),(0,0),(0,True)]:
            with self.assertRaises(ValueError):Stability.entropy_coefficient(.01,steps,budget)

    def toy_loss(self,config,*,advantage=None,steps=32000,active=None):
        t=4;gap=torch.tensor([[-.3],[.3],[-.2],[.2]],requires_grad=True)
        logits=torch.stack((torch.zeros_like(gap),gap),-1);d=torch.distributions.Categorical(logits=logits)
        actions=torch.tensor([[0],[1],[0],[1]])
        upper=d.log_prob(actions);old=(upper.detach()-.3).requires_grad_(True)
        flags=torch.ones(t,1,7,dtype=torch.bool) if active is None else active
        new=torch.zeros(t,1,7);older=torch.zeros_like(new);new[...,1]=upper;older[...,1]=old
        new=torch.where(flags,new,torch.zeros_like(new));older=torch.where(flags,older,torch.zeros_like(older))
        a=torch.tensor([[-3.],[-1.],[1.],[3.]]) if advantage is None else advantage
        kwargs={}
        if config.training.mappo.tx_upper_optimization_mode!='ppo_standard':
            kwargs=dict(tx_upper_logit_gap=gap,tx_upper_transmitted=actions.bool(),
                        tx_upper_entropy=d.entropy(),entropy_without_tx_upper=torch.zeros(t,1))
        loss=compute_configured_ppo_objective_and_loss(new_joint_log_prob=new.sum(-1),old_joint_log_prob=older.sum(-1),
            advantage=a,current_value=torch.zeros(t,1),return_target=torch.zeros(t,1),entropy=d.entropy(),
            sequence_valid_mask=torch.ones(t,dtype=torch.bool),config=config,new_branch_log_probs=new,
            old_branch_log_probs=older,active_branch_indicators=flags,collected_environment_steps=steps,**kwargs)
        return loss,gap,old

    def test_upper_normalized_advantage_only_other_branches_and_critic_unchanged(self):
        e8,_,_=self.toy_loss(self.e8);e9,_,_=self.toy_loss(self.e9)
        idx=[0,2,3,4,5,6]
        torch.testing.assert_close(e8.branch_advantage[...,idx],e9.branch_advantage[...,idx],rtol=0,atol=0)
        torch.testing.assert_close(e8.branch_surrogate[...,idx],e9.branch_surrogate[...,idx],rtol=0,atol=0)
        torch.testing.assert_close(e8.critic_loss,e9.critic_loss,rtol=0,atol=0)
        self.assertAlmostEqual(float(e9.branch_advantage[...,1].std(unbiased=False)),1.,places=6)
        self.assertEqual(float(e9.branch_advantage[...,1].mean()),0.)

    def test_clipped_PPO_joint_targets_and_total_signs_preserved(self):
        loss,gap,old=self.toy_loss(self.e9,steps=16000)
        a=loss.tx_upper_stability.normalized_advantage
        expected=torch.where(a>0,1.2*a,torch.exp(torch.tensor(.3))*a)
        torch.testing.assert_close(loss.branch_surrogate[...,1],expected)
        x=loss.tx_upper_stability;c=loss.tx_upper_objective_divisor;self.assertEqual(c,7)
        torch.testing.assert_close(loss.total_loss,loss.actor_loss+.5*loss.critic_loss-.005*x.entropy_mean/c+.05*x.margin_loss/c)
        loss.total_loss.backward();self.assertIsNone(old.grad);self.assertGreater(float(gap.grad.abs().sum()),0)

    def test_inactive_actor_advantage_does_not_change_upper_normalization(self):
        flags=torch.ones(4,1,7,dtype=torch.bool);flags[0,:,1]=False
        a=torch.tensor([[1000.],[-1.],[1.],[3.]])
        b=a.clone();b[0]=-.01
        x,_,_=self.toy_loss(self.e9,advantage=a,active=flags)
        y,_,_=self.toy_loss(self.e9,advantage=b,active=flags)
        torch.testing.assert_close(x.branch_advantage[...,1],y.branch_advantage[...,1],rtol=0,atol=0)
        torch.testing.assert_close(x.branch_surrogate[...,1],y.branch_surrogate[...,1],rtol=0,atol=0)

    def test_missing_progress_and_statistics_fail_closed(self):
        # Production must never silently restart its entropy schedule at zero.
        actor=CAGATMAPPOActor(self.e9);critic=MAPPOCentralizedCritic(self.e9)
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,self.e9)
        with self.assertRaisesRegex(ValueError,'collected environment steps'):
            updater.update(None)

    def loss_from_policy(self,policy,config,advantage,steps):
        logs=torch.stack(list(policy.component_log_probs.values()),-1).reshape(1,4,-1)
        active=torch.stack(list(policy.component_active.values()),-1).reshape(1,4,-1)
        kwargs={}
        if config.training.mappo.tx_upper_optimization_mode!='ppo_standard':
            kwargs=dict(tx_upper_logit_gap=policy.scheduler_metrics['tx_upper_logit_gap'].reshape(1,4),
                tx_upper_transmitted=policy.scheduler_metrics['tx_sampled_transmit'].reshape(1,4),
                tx_upper_entropy=policy.component_entropies['tx_upper'].reshape(1,4),
                entropy_without_tx_upper=policy.scheduler_metrics['entropy_without_tx_upper'].reshape(1,4))
        return compute_configured_ppo_objective_and_loss(new_joint_log_prob=policy.joint_log_prob.reshape(1,4),
            old_joint_log_prob=policy.joint_log_prob.detach().reshape(1,4),advantage=advantage,
            current_value=torch.ones_like(advantage),return_target=torch.zeros_like(advantage),
            entropy=policy.entropy.reshape(1,4),sequence_valid_mask=torch.ones(1,dtype=torch.bool,device=advantage.device),
            config=config,new_branch_log_probs=logs,old_branch_log_probs=logs.detach(),active_branch_indicators=active,
            route_entropy=policy.branch_entropies['route'].reshape(1,4),collected_environment_steps=steps,**kwargs)

    def gradient_check(self,device):
        actor=CAGATMAPPOActor(self.e9).to(device);d=CAGATMAPPOActionDistribution(actor,self.e9)
        policy=d.sample_actions(self.batch.to(device),self.masks)
        loss=self.loss_from_policy(policy,self.e9,torch.tensor([[-3.,-1.,1.,3.]],device=device),32000)
        loss.tx_upper_stability.margin_loss.backward(retain_graph=True)
        self.assertGreater(gradient_sum(actor.tx_upper_private),0)
        self.assertEqual(gradient_sum(actor.action_heads['tx_select']),0)
        self.assertEqual(gradient_sum(actor.cpu_joint_scheduler),0)
        for name in ['route_self_encoder','route_public_encoder','route_graph_encoder','route_gru','route_decoder','graph_encoder','gru']:
            self.assertEqual(gradient_sum(getattr(actor,name)),0)
        actor.zero_grad(set_to_none=True)
        # Upper PPO plus margin only: other branches must remain isolated.
        upper=-loss.branch_surrogate[...,1].mean()+.05*loss.tx_upper_stability.margin_loss
        upper.backward();self.assertGreater(gradient_sum(actor.tx_upper_private),0)
        self.assertEqual(gradient_sum(actor.action_heads['tx_select']),0)
        self.assertEqual(gradient_sum(actor.cpu_joint_scheduler),0)
        for name in ['route_self_encoder','route_public_encoder','route_graph_encoder','route_gru','route_decoder']:
            self.assertEqual(gradient_sum(getattr(actor,name)),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters()))

    def test_gradient_isolation_cpu(self):self.gradient_check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_gradient_isolation_cuda(self):self.gradient_check('cuda')

    def test_other_entropy_gradients_unchanged_and_upper_zero_after_schedule(self):
        actors=[CAGATMAPPOActor(c) for c in (self.e8,self.e9)]
        for a,c in zip(actors,(self.e8,self.e9)):
            p=CAGATMAPPOActionDistribution(a,c).deterministic_actions(self.batch,self.masks)
            loss=self.loss_from_policy(p,c,torch.zeros(1,4),32000)
            (-loss.global_entropy_loss_contribution).backward()
        self.assertGreater(gradient_sum(actors[0].tx_upper_private),0)
        self.assertLess(gradient_sum(actors[1].tx_upper_private),1.e-7)
        for (name,a),(_,b) in zip(actors[0].named_parameters(),actors[1].named_parameters()):
            if name.startswith('tx_upper_private'):continue
            if a.grad is None:self.assertIsNone(b.grad)
            else:torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-8)

    def test_same_architecture_forward_sampling_and_optimizer_groups(self):
        a,b=CAGATMAPPOActor(self.e8),CAGATMAPPOActor(self.e9)
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        p,q=[CAGATMAPPOActionDistribution(x,c).sample_actions(self.batch,self.masks) for x,c in [(a,self.e8),(b,self.e9)]]
        self.assertEqual(p.proposals,q.proposals)
        for attr in ['joint_log_prob','entropy','hidden_out']:torch.testing.assert_close(getattr(p,attr),getattr(q,attr),rtol=0,atol=0)
        o=build_ca_gat_mappo_optimizers(b,MAPPOCentralizedCritic(self.e9),self.e9)
        lr={id(p):g['lr'] for g in o.actor_optimizer.param_groups for p in g['params']}
        for name,p in b.named_parameters():self.assertAlmostEqual(lr[id(p)],3e-5 if name.startswith(b.route_parameter_prefixes) else 3e-4)


    def test_partial_checkpoint_and_recurrent_PPO_replay_compact_telemetry(self):
        actor=CAGATMAPPOActor(self.e9);critic=MAPPOCentralizedCritic(self.e9)
        with torch.no_grad():p=CAGATMAPPOActionDistribution(actor,self.e9).sample_actions(self.batch,self.masks)
        seed=CAGATMAPPORolloutTransition.from_step(spec=actor.spec,slot=0,actor_batch=self.batch,action_mask_batch=self.masks,
            action_output=p,hidden_in=actor.initial_hidden(1),centralized_state=CentralizedStateTensorBatch(torch.zeros(1,1,critic.spec.centralized_state_dim)),
            old_value=torch.zeros(4),reward=torch.ones(4),terminated=False,truncated=False,episode_boundary=False,
            bootstrap_allowed=True,bootstrap_value=torch.zeros(4),executed_action_summary=tuple({'uav_id':i} for i in range(4)),
            rejection_or_downgrade_summary={'rejection':[],'downgrade':[],'canonicalization':[]})
        partial=vector_buffer(self.e9,seed,length=8,finalized=False)
        state=serialize_active_rollout(partial,rollout_policy_version=0,checkpoint_kind='PERIODIC_RESUME')
        restored,_=restore_active_rollout(self.e9,state,checkpoint_kind='PERIODIC_RESUME',policy_version=0)
        self.assertEqual(restored.transition_at(0).rejection_or_downgrade_summary,partial.transition_at(0).rejection_or_downgrade_summary)
        buffer=vector_buffer(self.e9,seed);old,active=rollout_policy_components(buffer,self.e9,'cpu');frozen=old.clone()
        updater=CAGATMAPPORecurrentPPOUpdater(actor,critic,self.e9)
        output=updater.update(buffer,collected_environment_steps=32000)
        self.assertTrue(output.old_policy_snapshot_preserved)
        torch.testing.assert_close(old,frozen,rtol=0,atol=0)
        summary=updater.last_integrated_summary;json.dumps(summary,allow_nan=False)
        for epoch in summary['ppo_epochs']:
            tx=epoch['tx_upper'];self.assertEqual(tx['beta_tx'],0.)
            for k in ['margin_loss','mean_abs_p_transmit_minus_half','mean_abs_logit_gap',
                      'positive_advantage_transmit_count','positive_advantage_idle_count','ppo_loss','entropy']:
                self.assertIn(k,tx)

    def test_model_optimizer_checkpoint_roundtrip_and_final_loader(self):
        from tests.test_ca_gat_mappo_checkpoint import make_payload,training_state,assert_nested_equal
        from src.evaluation.actor_loader import load_final_actor_for_evaluation
        config=preserve_git(replace(self.e9,environment=replace(self.e9.environment,episode_horizon=512),
            training=replace(self.e9.training,mappo=replace(self.e9.training.mappo,training_device='cpu',checkpoint_interval_steps=4096,max_training_environment_steps=32768))),self.e9)
        payload,actor,critic,updater=make_payload(config,'FINAL_COMPLETED')
        payload['checkpoint_kind']='PERIODIC_RESUME';payload['training_state']=training_state(config,'PERIODIC_RESUME',0)
        sum(p.square().sum() for p in actor.tx_upper_private.parameters()).backward();updater.optimizers.actor_optimizer.step()
        payload['model_state']['actor']={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()};payload['optimizer_state']['actor_adam']=copy.deepcopy(updater.optimizers.actor_optimizer.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'unit-periodic.pt';atomic_save_checkpoint(payload,path);read=load_checkpoint_payload(path)
            a,c=CAGATMAPPOActor(config),MAPPOCentralizedCritic(config);o=build_ca_gat_mappo_optimizers(a,c,config)
            restore_model_optimizer_and_rng(config=config,payload=read,actor=a,critic=c,optimizers=o,dtype=torch.float32)
            assert_nested_equal(self,actor.state_dict(),a.state_dict());assert_nested_equal(self,updater.optimizers.actor_optimizer.state_dict(),o.actor_optimizer.state_dict())
            final,*_=make_payload(config,'FINAL_COMPLETED',actor=actor,critic=critic,updater=updater)
            path=Path(directory)/'unit-final.pt';atomic_save_checkpoint(final,path)
            ev=preserve_git(replace(config,mode='evaluation',launch_profile=None),config)
            loaded=load_final_actor_for_evaluation(ev,path,evaluation_device='cpu');assert_nested_equal(self,actor.state_dict(),loaded.actor.state_dict())


if __name__=='__main__':unittest.main()
